"""Шаг 4. Программа управления: взять красный кубик и отвезти на красный стол.

Запуск с окном : python examples/04_pick_and_place.py
Без окна (тест): python examples/04_pick_and_place.py --headless

Это проверка, что сцена пригодна для задачи: до столов можно дотянуться,
кубик удерживается в захвате. Положение кубика здесь берётся прямо из
симулятора («идеальное зрение»). Начиная с шага 7 его даёт камера.

Программа собрана из готовых действий пакета sortbot_motion: подъехать, взять,
сложить руку, поставить. Как они устроены внутри, смотрите в sortbot_motion/skills.py.
"""
import sys

from sortbot_sim import SortingSim
from sortbot_sim.window import Window
from sortbot_sim.world import tables_from_sim
from sortbot_motion import Skills

CUBE = "cube_red_1"
CUBE_HALF = 0.02
RELEASE_HEIGHT = 0.005  # с какой высоты над столом кубик отпускается, м


def main():
    sim = SortingSim()
    window = Window(sim, headless="--headless" in sys.argv)
    skills = Skills(sim, on_step=window.on_step)  # on_step: обновлять окно после каждого шага физики
    tables = {table.name: table for table in tables_from_sim(sim)}
    source, target = tables["table_source"], tables["table_red"]
    arm_folded = sim.arm_target  # в этой позе рука сложена для езды

    # 1. Подъехать к кубику со стороны робота и взять его.
    position, yaw = sim.body_pos(CUBE), sim.body_yaw(CUBE)
    print(f"Кубик {CUBE} в точке {position.round(3)}, еду к нему")
    skills.approach(position, source.heading)
    print("Беру кубик")
    skills.pick(position, yaw)

    # 2. Сложить руку, отъехать от стола и переехать к столу назначения.
    skills.move_arm(arm_folded)
    skills.retreat()
    slot = target.place_slots()[0]
    place = [slot[0], slot[1], target.surface_z + CUBE_HALF + RELEASE_HEIGHT]
    print(f"Везу на {target.name}")
    skills.approach(place, target.heading)

    # 3. Поставить кубик и отъехать.
    print("Ставлю кубик")
    skills.place(place, target.heading)
    skills.move_arm(arm_folded)
    skills.retreat()
    skills.wait(1.0)

    position = sim.body_pos(CUBE)
    on_table = target.holds_cube(position)
    print(f"Кубик в точке {position.round(3)}, время симуляции {sim.time:.1f} с")
    print("РЕЗУЛЬТАТ:", "кубик на красном столе" if on_table else "кубик НЕ на столе")

    window.hold_open()
    window.close()
    sys.exit(0 if on_table else 1)


if __name__ == "__main__":
    main()
