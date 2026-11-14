"""Шаг 8. Полный цикл: развезти все кубики по столам их цвета.

Запуск с окном : python examples/08_sort_all.py
Без окна (тест): python examples/08_sort_all.py --headless

Робот осматривает стол, запоминает кубики, возит их по одному и в конце осматривает
стол ещё раз, чтобы убедиться, что он пуст. Положение кубиков он знает только из кадров
камеры, а взят ли кубик, понимает по тому, насколько сжался захват.
Из симулятора берутся поза базы и описание столов. Истинные положения кубиков нужны
только для итоговой проверки в конце.
"""
import sys

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.window import Window
from sortbot_sim.world import tables_from_sim
from sortbot_motion import Skills
from sortbot_task import Sorter
from sortbot_vision.cubes import CubeDetector


def check_with_ground_truth(sim, tables):
    """Истинное положение каждого кубика: на каком он столе и тот ли это стол."""
    correct = 0
    names = [sim.model.body(b).name for b in range(sim.model.nbody)]
    for name in sorted(n for n in names if n.startswith("cube_")):
        color, position = name.split("_")[1], sim.body_pos(name)
        on = next((t for t in tables if t.contains(position) and abs(position[2] - t.surface_z - 0.02) < 0.005), None)
        right = on is not None and on.color == color
        correct += right
        print(f"  {name:13s} {'на ' + on.name if on else 'не на столе':16s} {'верно' if right else 'НЕВЕРНО'}")
    return correct, sum(n.startswith("cube_") for n in names)


def main():
    headless = "--headless" in sys.argv
    sim = SortingSim()
    window = Window(sim, headless)  # окно создаётся раньше камеры
    window.show_camera()  # в углу окна: что сейчас видит камера на запястье
    camera = SimCamera(sim)  # модель RealSense D435
    window.close_first.append(camera)

    tables = tables_from_sim(sim)
    sorter = Sorter(Skills(sim, on_step=window.on_step), camera, CubeDetector(), tables)
    sim.set_gripper(0.0)
    result = sorter.run()

    print(f"\nРобот считает: отвезено {len(result.delivered)}, осталось {len(result.left)}, "
          f"осмотров стола {result.rounds}, время симуляции {sim.data.time:.0f} с")
    print("Проверка по симулятору:")
    correct, total = check_with_ground_truth(sim, tables)
    print(f"РЕЗУЛЬТАТ: на своих столах {correct} из {total}")

    window.hold_open()
    window.close()
    sys.exit(0 if correct == total else 1)


if __name__ == "__main__":
    main()
