"""Окно просмотрщика, которое идёт в ногу с настоящим временем и показывает вид с камеры."""
import dataclasses
import sys
import threading
import time

import cv2
import mujoco
import mujoco.viewer

from sortbot_sim.camera import D435, SimCamera, depth_to_rgb

PAD = 10  # отступ картинок от края окна и друг от друга, пикс
MAX_LAG = 0.5  # если симуляция отстала от настоящего времени сильнее, не навёрстываем, с
# Окно обновляется 50 раз в секунду модельного времени, а не на каждом шаге физики (их 500):
# одно обновление занимает около 5 мс, и на каждом шаге симуляция шла бы вдвое медленнее часов.
SYNC_PERIOD = 0.02
MIN_VIEW_WIDTH = 60  # в окне уже этого картинки с камеры не показываются, пикс


class Window:
    """headless=True: окна нет, симуляция считается так быстро, как может.

    Окно нужно создавать раньше камер SimCamera, а закрывать позже них: в другом порядке
    программа падает при выходе. Камеры перечисляются в close_first.
    """

    def __init__(self, sim, headless=False, key_callback=None):
        self.sim = sim
        threads_before = set(threading.enumerate())
        self.viewer = None if headless else mujoco.viewer.launch_passive(
            sim.model, sim.data, key_callback=key_callback)
        # Окно рисуется в отдельном потоке. Запоминаем его, чтобы при закрытии дождаться конца.
        self.viewer_threads = [t for t in threading.enumerate() if t not in threads_before]
        self.wall_start = time.time() - sim.data.time
        self.close_first = []
        self.next_sync_time = 0.0
        self.view_camera = None
        self.view_period = self.view_min_period = 0.0
        self.view_depth_range = (0.0, 1.0)
        self.view_fraction = 0.0
        self.next_view_time = 0.0

    def show_camera(self, name="robot/wrist", sensor=D435, fraction=0.45, rate=10.0, max_depth=2.0,
                    render_width=900):
        """Показывать в правом нижнем углу окна цвет и глубину с камеры.

        fraction: ширина каждой из двух картинок как доля ширины области со сценой
        (окно без боковых панелей). Размер пересчитывается при каждом обновлении, поэтому
        картинки растут и сжимаются вместе с окном и одинаково выглядят на любом экране.
        render_width: с какой шириной кадр считается. Он растягивается или сжимается до нужного
        размера: считать кадр сразу под большое окно было бы слишком долго.
        Картинки обновляются до rate раз в секунду модельного времени, а на медленной машине
        реже: так, чтобы симуляция не отставала от настоящего времени.
        Для них заводится отдельная камера с той же моделью сенсора, но без шума цвета:
        на глаз он не виден, а считается долго.
        Глубина раскрашена от ближней границы сенсора до max_depth метров.
        """
        if self.viewer is None:
            return
        scale = render_width / sensor.width
        small = dataclasses.replace(
            sensor, width=render_width, height=round(sensor.height * scale), rgb_noise=0.0,
            noise_blur_px=sensor.noise_blur_px * scale,
            edge_dropout_px=max(1, round(sensor.edge_dropout_px * scale)) if sensor.edge_dropout_px else 0)
        self.view_camera = SimCamera(self.sim, name, sensor=small, seed=1)
        self.close_first.append(self.view_camera)
        self.view_fraction = fraction
        self.view_period = self.view_min_period = 1.0 / rate
        self.view_depth_range = (sensor.min_depth, max_depth)

    def on_step(self):
        """Вызывать после каждого шага физики."""
        if self.viewer is None:
            return
        now = self.sim.data.time
        # Ещё рано. Разность больше периода значит, что время симуляции сбросили: тогда обновляем.
        if 0 < self.next_sync_time - now <= SYNC_PERIOD:
            return
        self.next_sync_time = now + SYNC_PERIOD
        if not self.viewer.is_running():  # пользователь закрыл окно
            self.close()
            sys.exit(0)
        self._update_camera_view()
        self.viewer.sync()

        # Ждём, чтобы модельное время не обгоняло настоящее. Если время симуляции
        # сбросили или она сильно отстала, начинаем отсчёт заново.
        lead = self.wall_start + self.sim.data.time - time.time()
        if lead > SYNC_PERIOD + MAX_LAG or lead < -MAX_LAG:
            self.wall_start = time.time() - self.sim.data.time
            lead = 0.0
        time.sleep(max(0.0, lead))

    def _update_camera_view(self):
        now = self.sim.data.time
        if self.view_camera is None or 0 < self.next_view_time - now <= self.view_period:
            return
        started = time.time()

        frame = self.view_camera.capture()
        color = frame.rgb
        depth = depth_to_rgb(frame.depth, *self.view_depth_range)

        # Окно могли закрыть, пока снимался кадр: в закрытое окно картинки отправлять нельзя.
        area = self.viewer.viewport if self.viewer.is_running() else None
        if area is not None:
            # Ширина картинки: заданная доля области со сценой, но так, чтобы две картинки
            # с отступами поместились по ширине, а по высоте не вылезли за окно.
            aspect = color.shape[0] / color.shape[1]
            width = int(min(self.view_fraction * area.width, (area.width - 3 * PAD) / 2,
                            (area.height - 2 * PAD) / aspect))
            height = round(width * aspect)
            if width >= MIN_VIEW_WIDTH:
                if (height, width) != color.shape[:2]:
                    how = cv2.INTER_AREA if width < color.shape[1] else cv2.INTER_LINEAR
                    color = cv2.resize(color, (width, height), interpolation=how)
                    depth = cv2.resize(depth, (width, height), interpolation=how)
                near, far = self.view_depth_range
                _label(color, "RGB")
                _label(depth, f"depth {near:.2f}-{far:.0f} m")
                # Правый нижний угол области со сценой.
                left = area.left + area.width - 2 * width - 2 * PAD
                bottom = area.bottom + PAD
                self.viewer.set_images([
                    (mujoco.MjrRect(left, bottom, width, height), color),
                    (mujoco.MjrRect(left + width + PAD, bottom, width, height), depth),
                ])
            else:
                self.viewer.clear_images()  # окно слишком маленькое

        # Следующее обновление не раньше, чем через период, и так, чтобы картинки
        # занимали не больше половины времени: иначе симуляция отстанет от часов.
        self.view_period = max(self.view_min_period, 2.0 * (time.time() - started))
        self.next_view_time = now + self.view_period

    def hold_open(self):
        """Продолжать симуляцию, пока окно не закроют."""
        if self.viewer is None:
            return
        print("Закройте окно, чтобы выйти")
        while self.viewer.is_running():
            self.sim.step()
            self.on_step()

    def close(self):
        for item in self.close_first:
            item.close()
        self.close_first = []
        if self.viewer is not None:
            self.viewer.close()
            # Без ожидания программа иногда зависает на выходе: поток окна ещё рисует,
            # а графическая библиотека уже сворачивается.
            for thread in self.viewer_threads:
                thread.join(timeout=5.0)


def _label(image, text):
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
