"""Шаг 7. Взять один кубик по данным зрения и отвезти на стол его цвета.

Запуск с окном : python examples/07_pick_by_vision.py
Без окна (тест): python examples/07_pick_by_vision.py --headless
Выбрать цвет   : python examples/07_pick_by_vision.py --color blue

В шаге 4 положение кубика бралось прямо из симулятора. Здесь робот узнаёт его
только из кадров камеры на запястье (модель RealSense D435), в два взгляда:

1. Общий: с 0.8-1.0 м виден весь стол. Точность 3-6 мм, хватает, чтобы выбрать кубик и подъехать.
2. Уточняющий: камера зависает в 0.34 м над выбранным кубиком. Точность лучше 1 мм, по ней и берём.

Ближе 0.28 м D435 глубину не даёт, поэтому на последних сантиметрах рука идёт вслепую,
по результату второго взгляда. Из симулятора берутся только поза базы и положение столов,
а истинное положение кубика используется лишь для печати ошибки и проверки результата.

Оба взгляда и сама перевозка живут в sortbot_task.Sorter: здесь из него вызваны
отдельные шаги для одного кубика. Шаг 8 запускает тот же Sorter целиком.
"""
import argparse
import sys

from _setup import build
from sortbot_sim.world import cube_positions

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="без окна")
    parser.add_argument("--color", choices=["red", "green", "blue"], help="какой кубик брать (по умолчанию ближайший)")
    args = parser.parse_args()
    sim, window, tables, sorter = build(args.headless)

    # 1. Общий взгляд на стол с кубиками: всё увиденное попадает в память сортировщика.
    print("Еду к столу и осматриваю его")
    seen = sorter.observe_source()
    print("Вижу кубики:", ", ".join(sorted(cube.color for cube in seen)) or "ни одного")
    candidates = [cube for cube in sorter.deliverable() if args.color in (None, cube.color)]
    if not candidates:
        print("РЕЗУЛЬТАТ: подходящего кубика не видно")
        sys.exit(1)
    # Берём ближайший к роботу: до него проще дотянуться, и он не заслонён другими.
    cube = min(candidates, key=lambda c: np.linalg.norm(c.position[:2] - sim.base_pose[:2]))

    # Какой это кубик на самом деле: только для печати ошибки и итоговой проверки.
    truth = cube_positions(sim)
    name = min(truth, key=lambda n: np.linalg.norm(truth[n] - cube.position))
    print(f"Выбран {cube.color} кубик, ошибка общего взгляда {1000 * np.linalg.norm(truth[name] - cube.position):.1f} мм")

    # 2. Подъехать, посмотреть вблизи, взять, отвезти, поставить.
    delivered = sorter.deliver(cube)

    table = next(t for t in tables if t.color == cube.color)
    on_table = delivered and table.holds_cube(sim.body_pos(name))
    print(f"Время симуляции {sim.time:.1f} с")
    print("РЕЗУЛЬТАТ:", f"кубик {name} на столе {table.name}" if on_table else f"кубик {name} НЕ на своём столе")

    window.hold_open()
    window.close()  # сначала закрывает камеру, потом окно
    sys.exit(0 if on_table else 1)


if __name__ == "__main__":
    main()
