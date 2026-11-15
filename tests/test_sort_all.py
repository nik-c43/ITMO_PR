"""Сквозные проверки полного цикла сортировки."""
# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.world import cube_positions, placement_report, tables_from_sim
from sortbot_motion import Skills
from sortbot_task import Sorter
from sortbot_vision.cubes import CubeDetector

import numpy as np


class Run:
    """Сортировка без окна с наблюдением за тем, чего робот касается."""

    def __init__(self):
        self.sim = SortingSim()
        self.camera = SimCamera(self.sim)
        self.tables = tables_from_sim(self.sim)
        self.touched_tables = set()
        self.steps = 0
        self.log = []
        self.on_log = None
        self.sorter = Sorter(Skills(self.sim, on_step=self.watch_contacts), self.camera, CubeDetector(),
                             self.tables, log=self.record)
        self.sim.set_gripper(0.0)

    def record(self, line):
        self.log.append(line)
        if self.on_log is not None:
            self.on_log(line)

    def watch_contacts(self):
        # Каждый пятый шаг (10 мс): касание стола длится дольше, а тест идёт втрое быстрее.
        self.steps += 1
        if self.steps % 5:
            return
        model, data = self.sim.model, self.sim.data
        for contact in data.contact[:data.ncon]:
            bodies = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
            robot = [b for b in bodies if b.startswith("robot/")]
            table = [b for b in bodies if b.startswith("table_")]
            if robot and table:
                self.touched_tables.add((robot[0], table[0]))

    def move_cube(self, name, position):
        """Переставить кубик рукой экспериментатора."""
        joint = self.sim.model.body(name).jntadr[0]
        qadr, dadr = self.sim.model.jnt_qposadr[joint], self.sim.model.jnt_dofadr[joint]
        self.sim.data.qpos[qadr:qadr + 3] = position
        self.sim.data.qvel[dadr:dadr + 6] = 0

    def finish(self):
        result = self.sorter.run()
        self.camera.close()
        return result

    def correctly_placed(self, exclude=()):
        """Сколько кубиков (кроме exclude) стоит на столах своего цвета и сколько их всего."""
        positions = {name: p for name, p in cube_positions(self.sim).items() if name not in exclude}
        _, correct, total = placement_report(self.tables, positions)
        return correct, total


def test_all_cubes_end_up_on_tables_of_their_color():
    run = Run()
    result = run.finish()
    assert not result.left and len(result.delivered) == 6 and result.rounds == 2
    assert run.correctly_placed() == (6, 6)
    assert not run.touched_tables, f"робот задел стол: {run.touched_tables}"
    # Кубики на одном столе стоят порознь, а не друг на друге.
    for color in ("red", "green", "blue"):
        a, b = (run.sim.body_pos(f"cube_{color}_{i}") for i in (1, 2))
        assert np.linalg.norm(a[:2] - b[:2]) > 0.08


def test_cubes_moved_and_removed_after_the_first_look():
    """После первого осмотра один кубик сдвинули на 7 см, другой унесли, третий переложили."""
    run = Run()
    done = []

    def meddle(line):
        if line.startswith("Кубик") and not done:
            done.append(True)
            run.move_cube("cube_red_2", [1.26, 0.22, 0.42])  # сдвинут
            run.move_cube("cube_blue_2", [3.0, 3.0, 0.02])  # унесён на пол вдали от столов
            run.move_cube("cube_green_2", [1.20, -0.40, 0.42])  # переложен в другое место стола

    run.on_log = meddle
    result = run.finish()
    assert not result.left and len(result.delivered) == 5
    assert run.correctly_placed(exclude=["cube_blue_2"]) == (5, 5), run.log
    assert run.correctly_placed() == (5, 6)  # унесённый кубик робот не трогал
    assert not run.touched_tables
