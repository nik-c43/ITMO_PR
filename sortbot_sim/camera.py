"""Камера симулятора: выдаёт кадры в том же виде, что и драйвер настоящей камеры.

Сама камера MuJoCo идеальна: глубина берётся из буфера глубины рендера и точна в каждом
пикселе. Чтобы зрение отлаживалось на правдоподобных данных, поверх рендера накладывается
модель сенсора (SensorModel): угол обзора, рабочая дальность, шум и пропуски глубины.
"""
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from sortbot_vision.types import CameraIntrinsics, RgbdFrame

# Камера MuJoCo смотрит вдоль -Z, ось Y вверх. Оптический кадр ROS: Z вперёд, Y вниз.
MUJOCO_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class SensorModel:
    """Чем настоящая камера отличается от идеальной."""
    name: str
    width: int
    height: int
    fovy_deg: float | None = None  # вертикальный угол обзора; None: как у камеры в модели робота
    min_depth: float = 0.0  # ближе этого глубины нет, м
    max_depth: float = np.inf  # дальше этого глубины нет, м
    depth_noise_at_1m: float = 0.0  # СКО шума глубины на 1 м, м; растёт как квадрат расстояния
    noise_blur_px: float = 0.0  # шум соседних пикселей связан: радиус размытия шума, пикс
    edge_dropout_px: int = 0  # ширина полосы без глубины вдоль перепадов глубины, пикс
    edge_step: float = 0.015  # какой перепад глубины считается краем предмета, м
    rgb_noise: float = 0.0  # СКО шума цвета, единицы яркости 0..255


# Идеальная камера: то, что MuJoCo рендерит сам.
IDEAL = SensorModel("ideal", 640, 480)

# Intel RealSense D435 с глубиной, совмещённой с цветным кадром (align_depth в драйвере).
# Угол обзора и разрешение взяты у цветного сенсора: 69 x 42 градуса, 1280 x 720.
# Шум: оценка для стереопары с базой 50 мм по формуле z^2 * 0.08 / (f * b), f = 447 пикс
# (глубина 848 x 480). Это расчёт, а не измерение на настоящей камере.
D435 = SensorModel(
    "RealSense D435", 1280, 720, fovy_deg=42.5,
    min_depth=0.28, max_depth=3.0,
    depth_noise_at_1m=0.0036, noise_blur_px=3.0,
    edge_dropout_px=2, rgb_noise=2.0,
)


class SimCamera:
    def __init__(self, sim, name="robot/wrist", sensor=D435, seed=0):
        if mujoco.GLContext.__module__ == "mujoco.glfw":
            # Рендер через GLFW конфликтует с окном просмотрщика и роняет программу.
            raise RuntimeError("Камере симулятора нужен рендер без окна: импортируйте sortbot_sim "
                               "раньше, чем mujoco, или задайте переменную окружения MUJOCO_GL=egl")
        self.sim = sim
        self.sensor = sensor
        self.camera_id = sim.model.camera(name).id
        if sensor.fovy_deg is not None:
            sim.model.cam_fovy[self.camera_id] = sensor.fovy_deg
        self.intrinsics = CameraIntrinsics.from_fov(sensor.width, sensor.height, sim.model.cam_fovy[self.camera_id])
        self.renderer = mujoco.Renderer(sim.model, height=sensor.height, width=sensor.width)
        self.rng = np.random.default_rng(seed)

    def capture(self):
        """Снять кадр: цвет, глубина и поза камеры в мире на этот момент."""
        data = self.sim.data
        self.renderer.update_scene(data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        rgb, depth = self._apply_sensor(rgb, depth)
        return RgbdFrame(rgb, depth, self.intrinsics, self.pose(), stamp=data.time)

    def pose(self):
        """Поза оптического кадра камеры в мире, матрица 4x4. Рендер для этого не нужен."""
        data = self.sim.data
        world_from_camera = np.eye(4)
        world_from_camera[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3) @ MUJOCO_FROM_OPTICAL
        world_from_camera[:3, 3] = data.cam_xpos[self.camera_id]
        return world_from_camera

    def _apply_sensor(self, rgb, depth):
        """Испортить идеальный кадр так, как это делает настоящий сенсор."""
        s = self.sensor
        missing = (depth < s.min_depth) | (depth > s.max_depth)

        if s.edge_dropout_px > 0:
            # На границе предмета стереокамера не может сопоставить левый и правый кадры:
            # вдоль перепадов глубины остаётся полоса без данных.
            size = 2 * s.edge_dropout_px + 1
            kernel = np.ones((size, size), np.uint8)
            missing |= (cv2.dilate(depth, kernel) - cv2.erode(depth, kernel)) > s.edge_step

        if s.depth_noise_at_1m > 0:
            noise = self.rng.standard_normal(depth.shape, dtype=np.float32)
            if s.noise_blur_px > 0:
                # Глубина считается по окну пикселей, поэтому ошибки соседей похожи.
                noise = cv2.GaussianBlur(noise, (0, 0), s.noise_blur_px)
                noise /= noise.std()
            depth = depth + s.depth_noise_at_1m * depth**2 * noise

        depth = depth.astype(np.float32)
        depth[missing] = 0.0  # так RealSense отмечает пиксели без глубины

        if s.rgb_noise > 0:
            # Шум считает OpenCV: это в несколько раз быстрее, чем numpy на полном кадре.
            cv2.setRNGSeed(int(self.rng.integers(2**31 - 1)))
            noise = np.empty(rgb.shape, np.int16)
            cv2.randn(noise, 0.0, s.rgb_noise)
            rgb = cv2.add(rgb, noise, dtype=cv2.CV_8U)
        return rgb, depth

    def close(self):
        self.renderer.close()


# Палитра для показа глубины: 255 цветов от синего к красному, цвет 0 чёрный (нет данных).
_DEPTH_COLORS = cv2.cvtColor(
    cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
).reshape(256, 3)
_DEPTH_COLORS[0] = 0


def depth_to_rgb(depth, near, far):
    """Раскрасить карту глубины для показа: близкое красное, далёкое синее, нет данных чёрное."""
    level = np.clip((far - depth) * np.float32(254.0 / (far - near)), 0, 254) + 1
    level[depth <= 0] = 0
    return _DEPTH_COLORS[level.astype(np.uint8)]
