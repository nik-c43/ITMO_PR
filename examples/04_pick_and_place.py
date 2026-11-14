"""Шаг 4. Программа управления: взять красный кубик и отвезти на красный стол.

Запуск с окном : python examples/04_pick_and_place.py
Без окна (тест): python examples/04_pick_and_place.py --headless

Это проверка, что сцена пригодна для задачи: до столов можно дотянуться,
кубик удерживается в захвате. Положение кубика здесь берётся прямо из
симулятора («идеальное зрение»). В настоящем проекте его будет давать камера.
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from sortbot_sim import SortingSim, solve_ik

CUBE = "cube_red_1"
TABLE = "table_red"

TABLE_TOP_Z = 0.40
CUBE_HALF = 0.02
STANDOFF = 0.55  # расстояние от центра базы до кубика при захвате, м
APPROACH_HEIGHT = 0.12  # высота над кубиком, с которой захват опускается, м
# Точка захвата (pinch_site) находится у основания подушечек пальцев: при сжатии они
# уходят на 4 см ниже неё. Поэтому целимся выше центра кубика, иначе пальцы упрутся в стол.
GRASP_OFFSET = np.array([0.0, 0.0, 0.035])

# Поза руки «захват смотрит вниз перед роботом»: начальное приближение для обратной кинематики.
ARM_TOP_DOWN_SEED = np.array([0.0, 0.6, 3.14, -1.9, 0.0, -0.65, 1.57])


class Demo:
    def __init__(self, headless):
        self.sim = SortingSim()
        self.viewer = None if headless else mujoco.viewer.launch_passive(self.sim.model, self.sim.data)
        self.arm_home = self.sim.arm_target

    # ---- базовые действия ----

    def tick(self):
        start = time.time()
        self.sim.step()
        if self.viewer is not None:
            if not self.viewer.is_running():
                sys.exit(0)
            self.viewer.sync()
            time.sleep(max(0.0, self.sim.dt - (time.time() - start)))

    def wait(self, seconds):
        for _ in range(int(seconds / self.sim.dt)):
            self.tick()

    def drive_to(self, x, y, th, timeout=20.0):
        """Доехать до позы (x, y, th) пропорциональным регулятором по скорости."""
        goal = np.array([x, y, th])
        for _ in range(int(timeout / self.sim.dt)):
            err = goal - self.sim.base_pose
            err[2] = (err[2] + np.pi) % (2 * np.pi) - np.pi
            if np.linalg.norm(err[:2]) < 0.005 and abs(err[2]) < 0.01:
                break
            # Ошибка известна в мировых осях, а cmd_vel задаётся в осях робота.
            c, s = np.cos(self.sim.base_pose[2]), np.sin(self.sim.base_pose[2])
            vx = np.clip(2.0 * (c * err[0] + s * err[1]), -0.5, 0.5)
            vy = np.clip(2.0 * (-s * err[0] + c * err[1]), -0.5, 0.5)
            wz = np.clip(2.0 * err[2], -1.0, 1.0)
            self.sim.set_cmd_vel(vx, vy, wz)
            self.tick()
        self.sim.set_cmd_vel(0, 0, 0)
        self.wait(0.3)

    def move_arm(self, q_goal, duration=2.0):
        """Плавно перевести цель приводов руки в q_goal за duration секунд."""
        q_start = self.sim.arm_target
        steps = int(duration / self.sim.dt)
        for i in range(1, steps + 1):
            s = i / steps
            s = 3 * s**2 - 2 * s**3  # плавный разгон и торможение
            self.sim.set_arm_target(q_start + s * (q_goal - q_start))
            self.tick()
        self.wait(0.3)

    def move_gripper_to(self, pos, yaw, duration=2.0, q_init=None):
        """Поставить захват в точку pos пальцами вниз, развернув его на угол yaw."""
        c, s = np.cos(yaw), np.sin(yaw)
        # Столбцы: оси захвата в мире. X: линия между пальцами, Z: направление подхода (вниз).
        mat = np.array([[c, s, 0.0],
                        [s, -c, 0.0],
                        [0.0, 0.0, -1.0]])
        q, ok = solve_ik(self.sim, pos, mat, q_init=q_init)
        if not ok:
            raise RuntimeError(f"Обратная кинематика не сошлась для точки {np.round(pos, 3)}")
        self.move_arm(q, duration)

    # ---- сценарий ----

    def run(self):
        sim = self.sim
        sim.set_gripper(0.0)
        self.wait(0.5)

        # 1. Подъехать к столу с кубиками, встав напротив кубика.
        cube = sim.body_pos(CUBE)
        print(f"Кубик {CUBE} в точке {np.round(cube, 3)}, еду к нему")
        self.drive_to(cube[0] - STANDOFF, cube[1], 0.0)

        # 2. Опустить захват на кубик и взять его.
        grasp = sim.body_pos(CUBE) + GRASP_OFFSET
        cube_yaw = sim.body_yaw(CUBE)
        above = grasp + [0, 0, APPROACH_HEIGHT]
        print("Беру кубик")
        self.move_gripper_to(above, cube_yaw, q_init=ARM_TOP_DOWN_SEED)
        self.move_gripper_to(grasp, cube_yaw, duration=1.5)
        sim.set_gripper(1.0)
        self.wait(1.0)
        self.move_gripper_to(above, cube_yaw, duration=1.5)

        # 3. Сложить руку и переехать к столу назначения, развернувшись к нему лицом.
        self.move_arm(self.arm_home)
        table = sim.body_pos(TABLE)
        direction = np.arctan2(table[1], table[0])  # столы стоят вокруг начала координат
        stop_distance = np.linalg.norm(table[:2]) - 0.3 - 0.40
        print(f"Везу на {TABLE}")
        self.drive_to(stop_distance * np.cos(direction), stop_distance * np.sin(direction), direction)

        # 4. Поставить кубик на стол и отпустить.
        place = np.array([table[0], table[1], TABLE_TOP_Z + CUBE_HALF + 0.005]) + GRASP_OFFSET
        place[:2] -= 0.15 * np.array([np.cos(direction), np.sin(direction)])  # ближе к краю стола
        above = place + [0, 0, APPROACH_HEIGHT]
        print("Ставлю кубик")
        self.move_gripper_to(above, direction, q_init=ARM_TOP_DOWN_SEED)
        self.move_gripper_to(place, direction, duration=1.5)
        sim.set_gripper(0.0)
        self.wait(1.0)
        self.move_gripper_to(above, direction, duration=1.5)
        self.move_arm(self.arm_home)
        self.wait(1.0)

        return self.check()

    def check(self):
        cube = self.sim.body_pos(CUBE)
        table = self.sim.body_pos(TABLE)
        on_table = (abs(cube[0] - table[0]) < 0.3 and abs(cube[1] - table[1]) < 0.3
                    and abs(cube[2] - (TABLE_TOP_Z + CUBE_HALF)) < 0.005)
        print(f"Кубик в точке {np.round(cube, 3)}, время симуляции {self.sim.data.time:.1f} с")
        print("РЕЗУЛЬТАТ:", "кубик на красном столе" if on_table else "кубик НЕ на столе")
        return on_table


def main():
    headless = "--headless" in sys.argv
    demo = Demo(headless)
    ok = demo.run()
    if demo.viewer is not None:
        print("Закройте окно, чтобы выйти")
        while demo.viewer.is_running():
            demo.tick()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
