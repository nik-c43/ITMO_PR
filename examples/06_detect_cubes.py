"""Шаг 6. Зрение: найти кубики по кадру с камеры на руке и сравнить с истиной.

Запуск: python examples/06_detect_cubes.py
Результат: таблица в терминале и картинка out/06_detections.png.

Робот подъезжает к столу, поднимает камеру над ним и делает один кадр.
Детектор получает только то, что будет и на настоящем роботе: цвет, глубину,
параметры камеры и её позу. Истинные координаты кубиков берутся из симулятора
лишь для проверки ошибки.

Камера по умолчанию изображает RealSense D435: шум и пропуски глубины, рабочая
дальность от 0.28 м. С ключом --ideal кадр берётся прямо из рендера MuJoCo, без искажений.
"""
import sys
from pathlib import Path

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim, solve_ik
from sortbot_sim.camera import D435, IDEAL, SimCamera
from sortbot_vision.cubes import CubeDetector, draw_detections

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parents[1] / "out"

OBSERVE_BASE_X = 0.35  # где встать: в 0.65 м от края стола с кубиками
# Поза осмотра: захват в 0.30 м перед базой на высоте 0.90 м, наклонён на 50 градусов от вертикали.
OBSERVE_PINCH = np.array([0.30, 0.0, 0.90])
OBSERVE_TILT = np.radians(50)
ARM_TOP_DOWN_SEED = np.array([0.0, 0.6, 3.14, -1.9, 0.0, -0.65, 1.57])


def go_to_observation_pose(sim):
    # Проехать вперёд до точки осмотра.
    sim.set_cmd_vel(0.3, 0.0, 0.0)
    while sim.base_pose[0] < OBSERVE_BASE_X:
        sim.step()
    sim.set_cmd_vel(0.0, 0.0, 0.0)

    # Навести камеру: ось захвата (и камеры) смотрит вперёд-вниз.
    t = OBSERVE_TILT
    orientation = np.column_stack([[0, 1, 0], [np.cos(t), 0, np.sin(t)], [np.sin(t), 0, -np.cos(t)]])
    target = OBSERVE_PINCH + [sim.base_pose[0], sim.base_pose[1], 0.0]
    q_goal, ok = solve_ik(sim, target, orientation, q_init=ARM_TOP_DOWN_SEED)
    if not ok:
        raise RuntimeError("Поза осмотра недостижима")
    q_start = sim.arm_target
    steps = int(2.0 / sim.dt)
    for i in range(1, steps + 1):
        sim.set_arm_target(q_start + i / steps * (q_goal - q_start))
        sim.step()
    for _ in range(int(1.0 / sim.dt)):  # дать руке успокоиться, иначе кадр смажет поза
        sim.step()


def true_cubes(sim):
    """Истинные положения и повороты кубиков из симулятора: {имя: (цвет, положение, угол)}."""
    cubes = {}
    for body in range(sim.model.nbody):
        name = sim.model.body(body).name
        if name.startswith("cube_"):
            yaw = (np.degrees(sim.body_yaw(name)) + 45) % 90 - 45
            cubes[name] = (name.split("_")[1], sim.body_pos(name), yaw)
    return cubes


def main():
    sim = SortingSim()
    sensor = IDEAL if "--ideal" in sys.argv else D435
    camera = SimCamera(sim, "robot/wrist", sensor=sensor)
    print(f"Камера: {sensor.name}, {sensor.width}x{sensor.height}")
    go_to_observation_pose(sim)

    # Высота столешницы известна заранее (сейчас из симулятора, на роботе из карты).
    top = sim.model.geom("table_source_top")
    surface_z = sim.data.geom(top.id).xpos[2] + top.size[2]

    frame = camera.capture()
    detections = CubeDetector().detect(frame, surface_z=surface_z)

    truth = true_cubes(sim)
    print(f"Найдено кубиков: {len(detections)} из {len(truth)}\n")
    print(f"{'цвет':7s} {'x':>7s} {'y':>7s} {'z':>7s} {'угол':>6s}   {'ближайший':13s} {'ошибка, мм':>10s} {'по углу':>8s}")
    for det in sorted(detections, key=lambda d: (d.color, d.position[1])):
        # Ближайший настоящий кубик того же цвета.
        name, (_, pos, yaw) = min(((n, c) for n, c in truth.items() if c[0] == det.color),
                                  key=lambda item: np.linalg.norm(item[1][1] - det.position))
        error_mm = 1000 * np.linalg.norm(pos - det.position)
        yaw_error = (np.degrees(det.yaw) - yaw + 45) % 90 - 45
        print(f"{det.color:7s} {det.position[0]:7.3f} {det.position[1]:7.3f} {det.position[2]:7.3f} "
              f"{np.degrees(det.yaw):6.1f}   {name:13s} {error_mm:10.1f} {yaw_error:7.1f}°")

    OUT_DIR.mkdir(exist_ok=True)
    Image.fromarray(draw_detections(frame, detections)).save(OUT_DIR / "06_detections.png")
    print("\nКартинка с находками:", OUT_DIR / "06_detections.png")
    camera.close()


if __name__ == "__main__":
    main()
