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

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from _setup import OUT_DIR
from sortbot_sim import SortingSim
from sortbot_sim.camera import D435, IDEAL, SimCamera
from sortbot_sim.world import cube_positions, tables_from_sim
from sortbot_motion import Skills
from sortbot_vision.cubes import CubeDetector, draw_detections

import numpy as np
from PIL import Image



def main():
    sim = SortingSim()
    sensor = IDEAL if "--ideal" in sys.argv else D435
    camera = SimCamera(sim, "robot/wrist", sensor=sensor)
    print(f"Камера: {sensor.name}, {sensor.width}x{sensor.height}")

    # Встать перед столом с кубиками и навести на него камеру.
    table = next(t for t in tables_from_sim(sim) if t.color is None)
    Skills(sim).look_over_table(table.near_edge, table.heading, table.surface_z)

    frame = camera.capture()
    # Высота столешницы известна заранее (сейчас из симулятора, на роботе из карты).
    detections = CubeDetector().detect(frame, surface_z=table.surface_z)

    truth = cube_positions(sim)
    print(f"Найдено кубиков: {len(detections)} из {len(truth)}\n")
    print(f"{'цвет':7s} {'x':>7s} {'y':>7s} {'z':>7s} {'угол':>6s}   {'ближайший':13s} {'ошибка, мм':>10s} {'по углу':>8s}")
    for det in sorted(detections, key=lambda d: (d.color, d.position[1])):
        # Ближайший настоящий кубик того же цвета (цвет стоит в имени: cube_<цвет>_<номер>).
        name = min((n for n in truth if n.split("_")[1] == det.color),
                   key=lambda n: np.linalg.norm(truth[n] - det.position))
        error_mm = 1000 * np.linalg.norm(truth[name] - det.position)
        # У куба симметрия 90 градусов, поэтому ошибку угла приводим к диапазону [-45, 45).
        yaw_error = (np.degrees(det.yaw - sim.body_yaw(name)) + 45) % 90 - 45
        print(f"{det.color:7s} {det.position[0]:7.3f} {det.position[1]:7.3f} {det.position[2]:7.3f} "
              f"{np.degrees(det.yaw):6.1f}   {name:13s} {error_mm:10.1f} {yaw_error:7.1f}°")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_detections(frame, detections)).save(OUT_DIR / "06_detections.png")
    print("\nКартинка с находками:", OUT_DIR / "06_detections.png")
    camera.close()


if __name__ == "__main__":
    main()
