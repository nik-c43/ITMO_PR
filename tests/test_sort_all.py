"""Сквозные проверки полного цикла сортировки."""
# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.world import tables_from_sim
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

    def table_of(self, name):
        """На каком столе кубик на самом деле."""
        position = self.sim.body_pos(name)
        for table in self.tables:
            if table.contains(position) and abs(position[2] - table.surface_z - 0.02) < 0.005:
                return table.name
        return None


CUBES = ["cube_red_1", "cube_red_2", "cube_green_1", "cube_green_2", "cube_blue_1", "cube_blue_2"]


def test_all_cubes_end_up_on_tables_of_their_color():
    run = Run()
    result = run.finish()
    assert result.ok and len(result.delivered) == 6 and result.rounds == 2
    assert {name: run.table_of(name) for name in CUBES} == {name: "table_" + name.split("_")[1] for name in CUBES}
    assert not run.touched_tables, f"робот задел стол: {run.touched_tables}"
    # Кубики на одном столе стоят порознь, а не друг на друге.
    for color in ("red", "green", "blue"):
        a, b = (run.sim.body_pos(f"cube_{color}_{i}") for i in (1, 2))
        assert np.linalg.norm(a[:2] - b[:2]) > 0.08


def test_cubes_moved_and_removed_after_the_first_look():
    """После первого осмотра один кубик сдвинули на 7 см, другой унесли, третий подложили."""
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
    assert result.ok and len(result.delivered) == 5
    assert run.table_of("cube_blue_2") is None
    for name in set(CUBES) - {"cube_blue_2"}:
        assert run.table_of(name) == "table_" + name.split("_")[1], (name, run.log)
    assert not run.touched_tables
