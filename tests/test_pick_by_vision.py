"""Сквозная проверка: робот берёт кубик по данным камеры и отвозит на стол его цвета."""
import importlib.util
from pathlib import Path

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
import sortbot_sim  # noqa: F401

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "07_pick_by_vision.py"


def load_example():
    spec = importlib.util.spec_from_file_location("pick_by_vision", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("color", ["red", "green", "blue"])
def test_cube_is_delivered_to_table_of_its_color(color):
    demo = load_example().Demo(headless=True)
    sim, model = demo.sim, demo.sim.model
    cubes = [model.body(b).name for b in range(model.nbody) if model.body(b).name.startswith("cube_")]
    start = {name: sim.body_pos(name) for name in cubes}

    touched_tables = set()

    def watch_contacts():
        for contact in sim.data.contact[:sim.data.ncon]:
            bodies = [model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)]
            robot = [b for b in bodies if b.startswith("robot/")]
            table = [b for b in bodies if b.startswith("table_")]
            if robot and table:
                touched_tables.add((robot[0], table[0]))

    demo.skills.on_step = watch_contacts
    delivered = demo.run(color)
    demo.window.close()

    assert delivered
    assert not touched_tables, f"робот задел стол: {touched_tables}"
    moved = [name for name in cubes if np.linalg.norm(sim.body_pos(name) - start[name]) > 0.005]
    assert len(moved) == 1 and moved[0].startswith(f"cube_{color}_"), moved
