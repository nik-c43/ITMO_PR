"""Шаг 7. Взять кубик по данным зрения и отвезти на стол его цвета.

Запуск с окном : python examples/07_pick_by_vision.py
Без окна (тест): python examples/07_pick_by_vision.py --headless
Выбрать цвет   : python examples/07_pick_by_vision.py --color blue

В примере 4 положение кубика бралось прямо из симулятора. Здесь робот узнаёт его
только из кадров камеры на запястье (модель RealSense D435), в два взгляда:

1. Общий: с 0.8-1.0 м виден весь стол. Точность 3-6 мм, хватает, чтобы выбрать кубик и подъехать.
2. Уточняющий: камера зависает в 0.34 м над выбранным кубиком. Точность лучше 1 мм, по ней и берём.

Ближе 0.28 м D435 глубину не даёт, поэтому на последних сантиметрах рука идёт вслепую,
по результату второго взгляда. Из симулятора берутся только поза базы и положение столов,
а истинное положение кубика используется лишь для печати ошибки и проверки результата.
"""
import argparse
import sys

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.window import Window
from sortbot_motion import Skills
from sortbot_vision.cubes import CubeDetector

import numpy as np

SOURCE_TABLE_TOP = "table_source_top"
TABLE_OF_COLOR = {"red": "table_red", "green": "table_green", "blue": "table_blue"}
CUBE_HALF = 0.02

OBSERVE_BASE = (0.35, 0.0, 0.0)  # где встать для общего взгляда: в 0.65 м от края стола
# Общий взгляд: захват в 0.30 м перед базой на высоте 0.90 м, наклонён на 50 градусов от вертикали.
OBSERVE_TOOL = np.array([0.30, 0.0, 0.90])
OBSERVE_TILT = np.radians(50)
MATCH_RADIUS = 0.03  # насколько второй взгляд может разойтись с первым, м


def fmt(point):
    return "(" + ", ".join(f"{v:.3f}" for v in point) + ")"


class Demo:
    def __init__(self, headless):
        self.sim = SortingSim()
        # Окно создаётся раньше камеры, а закрывается позже неё: в другом порядке
        # программа падает при выходе (два графических контекста мешают друг другу).
        self.window = Window(self.sim, headless)
        self.window.show_camera()  # в углу окна: что сейчас видит камера на запястье
        self.camera = SimCamera(self.sim)  # по умолчанию модель RealSense D435
        self.window.close_first.append(self.camera)
        self.detector = CubeDetector()
        self.skills = Skills(self.sim, on_step=self.window.on_step)
        self.arm_home = self.sim.arm_target

        # Высота столешницы известна заранее (сейчас из симулятора, на роботе из карты).
        top = self.sim.model.geom(SOURCE_TABLE_TOP)
        self.surface_z = self.sim.data.geom(top.id).xpos[2] + top.size[2]

    def see_cubes(self):
        frame = self.camera.capture()
        return self.detector.detect(frame, surface_z=self.surface_z)

    def true_cube_near(self, position):
        """Имя и истинное положение ближайшего настоящего кубика. Только для проверки."""
        names = [self.sim.model.body(b).name for b in range(self.sim.model.nbody)]
        name = min((n for n in names if n.startswith("cube_")),
                   key=lambda n: np.linalg.norm(self.sim.body_pos(n) - position))
        return name, self.sim.body_pos(name)

    def run(self, color):
        sim, skills = self.sim, self.skills
        sim.set_gripper(0.0)

        # 1. Общий взгляд на стол с кубиками.
        print("Еду к столу и осматриваю его")
        skills.drive_to(*OBSERVE_BASE)
        t = OBSERVE_TILT
        tilted = np.column_stack([[0, 1, 0], [np.cos(t), 0, np.sin(t)], [np.sin(t), 0, -np.cos(t)]])
        skills.move_tool(OBSERVE_TOOL + [sim.base_pose[0], sim.base_pose[1], 0.0], tilted,
                         q_init=skills.config.arm_seed)
        skills.wait(0.5)
        seen = self.see_cubes()
        print("Вижу кубики:", ", ".join(sorted(d.color for d in seen)) or "ни одного")
        candidates = [d for d in seen if color in (None, d.color)]
        if not candidates:
            print("РЕЗУЛЬТАТ: подходящего кубика не видно")
            return False
        # Берём ближайший к роботу: до него проще дотянуться, и он не заслонён другими.
        coarse = min(candidates, key=lambda d: np.linalg.norm(d.position[:2] - sim.base_pose[:2]))
        cube_name, truth = self.true_cube_near(coarse.position)
        print(f"Выбран {coarse.color} кубик в точке {fmt(coarse.position)}, "
              f"ошибка общего взгляда {1000 * np.linalg.norm(coarse.position - truth):.1f} мм")

        # 2. Подъехать и посмотреть на кубик сверху вблизи.
        skills.drive_to(*skills.grasp_base_pose(coarse.position, heading=0.0))
        skills.look_down_at(coarse.position)
        close = [d for d in self.see_cubes() if d.color == coarse.color
                 and np.linalg.norm(d.position - coarse.position) < MATCH_RADIUS]
        if close:
            target = min(close, key=lambda d: np.linalg.norm(d.position - coarse.position))
            print(f"Уточнённое положение {fmt(target.position)}, "
                  f"ошибка {1000 * np.linalg.norm(target.position - truth):.1f} мм")
        else:
            target = coarse
            print("Вблизи кубик не найден, беру по общему взгляду")

        # 3. Взять кубик.
        print("Беру кубик")
        skills.pick(target.position, target.yaw)
        skills.move_arm(self.arm_home)
        lifted = sim.body_pos(cube_name)[2] > self.surface_z + 0.10
        if not lifted:
            print("РЕЗУЛЬТАТ: кубик не удержался в захвате")
            return False

        # 4. Отвезти на стол своего цвета. Положение стола пока берётся из симулятора.
        table_name = TABLE_OF_COLOR[target.color]
        table = sim.body_pos(table_name)
        direction = np.arctan2(table[1], table[0])  # столы стоят вокруг начала координат
        toward = np.array([np.cos(direction), np.sin(direction)])
        spot = np.array([*(table[:2] - 0.15 * toward), self.surface_z + CUBE_HALF + 0.005])  # ближе к краю
        print(f"Везу на {table_name}")
        skills.drive_to(*skills.grasp_base_pose(spot, heading=direction))
        skills.place(spot, direction)
        skills.move_arm(self.arm_home)
        skills.wait(1.0)

        # 5. Проверка по истинному положению кубика.
        cube = sim.body_pos(cube_name)
        on_table = (np.all(np.abs(cube[:2] - table[:2]) < 0.3)
                    and abs(cube[2] - (self.surface_z + CUBE_HALF)) < 0.005)
        print(f"Кубик {cube_name} в точке {fmt(cube)}, время симуляции {sim.data.time:.1f} с")
        print("РЕЗУЛЬТАТ:", f"кубик на столе {table_name}" if on_table else "кубик НЕ на своём столе")
        return on_table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--headless", action="store_true", help="без окна")
    parser.add_argument("--color", choices=sorted(TABLE_OF_COLOR), help="какой кубик брать (по умолчанию ближайший)")
    args = parser.parse_args()

    demo = Demo(args.headless)
    ok = demo.run(args.color)
    demo.window.hold_open()
    demo.window.close()  # сначала закрывает камеру, потом окно
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
