"""Проверки памяти о кубиках и описания столов. Симулятор здесь не нужен."""
import numpy as np

from sortbot_task.memory import CubeMemory
from sortbot_task.world import Table
from sortbot_vision.types import CubeDetection


def det(color, x, y, yaw=0.0):
    return CubeDetection(color, np.array([x, y, 0.42]), yaw, (0.04, 0.04))


def everything(_):
    return True


def test_same_cube_seen_twice_is_one_cube_with_latest_position():
    memory = CubeMemory()
    memory.observe([det("red", 1.000, 0.0, yaw=0.1)])
    memory.observe([det("red", 1.004, 0.0, yaw=0.2)])
    (cube,) = memory.waiting()
    assert cube.yaw == 0.2
    np.testing.assert_allclose(cube.position[:2], [1.004, 0.0])


def test_other_color_or_far_away_is_a_new_cube():
    memory = CubeMemory()
    memory.observe([det("red", 1.0, 0.0)])
    memory.observe([det("blue", 1.0, 0.0), det("red", 1.1, 0.0)])
    assert sorted(c.color for c in memory.waiting()) == ["blue", "red", "red"]


def test_two_close_detections_in_one_frame_stay_two_cubes():
    memory = CubeMemory(match_radius=0.03)
    memory.observe([det("red", 1.00, 0.0), det("red", 1.02, 0.0)])
    assert len(memory.waiting()) == 2


def test_cube_missing_from_view_is_forgotten_but_cube_outside_view_is_kept():
    memory = CubeMemory()
    memory.observe([det("red", 1.0, 0.0), det("green", 1.0, 0.5)])
    memory.observe([], in_view=lambda p: p[1] < 0.2)
    assert [c.color for c in memory.waiting()] == ["green"]
    memory.observe([])  # кадр без сведений о том, что в него попадает: ничего не забываем
    assert [c.color for c in memory.waiting()] == ["green"]


def test_delivered_cube_is_not_forgotten_and_not_matched_again():
    memory = CubeMemory()
    (cube,) = memory.observe([det("red", 1.0, 0.0)])
    memory.mark_carried(cube)
    memory.observe([], in_view=everything)
    assert memory.cubes == [cube]
    memory.mark_delivered(cube, [0.0, 1.2, 0.42], "table_red")
    memory.observe([det("red", 0.0, 1.2)], in_view=everything)  # тот же кубик, увиденный на столе назначения
    assert len(memory.delivered("table_red")) == 1 and len(memory.waiting()) == 1


def test_table_slots_are_on_the_table_near_the_robot_side():
    table = Table("table_red", np.array([0.0, 1.4]), np.array([0.3, 0.3]), 0.40, heading=np.pi / 2, color="red")
    slots = table.place_slots()
    assert len(slots) == 3 and all(table.contains(s, margin=0.05) for s in slots)
    np.testing.assert_allclose(slots[0], [0.0, 1.25], atol=1e-9)  # первым занимается место посередине
    assert all(s[1] < table.center[1] for s in slots)  # робот подъезжает со стороны меньших y
    assert not table.contains([0.0, 1.75])


def test_table_knows_whether_a_cube_stands_on_it():
    table = Table("table_red", np.array([0.0, 1.4]), np.array([0.3, 0.3]), 0.40, heading=np.pi / 2, color="red")
    assert table.holds_cube(np.array([0.1, 1.3, 0.42]))
    assert not table.holds_cube(np.array([0.1, 1.3, 0.46]))  # висит над столом, например в захвате
    assert not table.holds_cube(np.array([0.5, 1.3, 0.42]))  # в стороне от столешницы
