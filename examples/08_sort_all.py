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

from _setup import build
from sortbot_sim.world import cube_positions, placement_report


def main():
    sim, window, tables, sorter = build(headless="--headless" in sys.argv)
    result = sorter.run()

    print(f"\nРобот считает: отвезено {len(result.delivered)}, осталось {len(result.left)}, "
          f"осмотров стола {result.rounds}, время симуляции {sim.time:.0f} с")
    lines, correct, total = placement_report(tables, cube_positions(sim))
    print("Проверка по симулятору:")
    print("\n".join(lines))
    print(f"РЕЗУЛЬТАТ: на своих столах {correct} из {total}")

    window.hold_open()
    window.close()
    sys.exit(0 if correct == total else 1)


if __name__ == "__main__":
    main()
