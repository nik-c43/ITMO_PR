"""Проверки контроллера руки: робот управляется редкими командами, как через ROS 2."""
from sortbot_sim import SortingSim
from sortbot_sim.controller import ArmController
from sortbot_motion import Skills

import numpy as np
import pytest


class LowRateRobot:
    """SortingSim, которым управляют 50 раз в секунду через контроллер руки.

    stall_at: модельное время, в которое программа управления «зависает» на stall секунд,
    то есть очередная команда приходит с опозданием.
    """

    def __init__(self, max_speed, interpolate=True, stall_at=None, stall=0.5):
        self.sim = SortingSim()
        self.controller = ArmController(self.sim, max_speed=max_speed)
        self.interpolate = interpolate
        self.dt = 0.02
        self.stall_at, self.stall = stall_at, stall
        self._arm_target = self.sim.arm_target
        self.max_step = 0.0  # самый большой скачок цели приводов за шаг физики, рад

    time = property(lambda self: self.sim.data.time)
    base_pose = property(lambda self: self.sim.base_pose)
    arm_target = property(lambda self: self._arm_target.copy())
    gripper_closure = property(lambda self: self.sim.gripper_closure)

    def set_cmd_vel(self, *v):
        self.sim.set_cmd_vel(*v)

    def set_gripper(self, closed):
        self.sim.set_gripper(closed)

    def solve_ik(self, pos, mat, q_init=None):
        return self.sim.solve_ik(pos, mat, q_init)

    def set_arm_target(self, q):
        self._arm_target = np.array(q, dtype=float)

    def step(self):
        duration = self.dt
        if self.stall_at is not None and self.sim.data.time >= self.stall_at:
            duration, self.stall_at = self.stall, None
        if self.interpolate:
            self.controller.command(self._arm_target)
        for _ in range(round(duration / self.sim.dt)):
            before = self.sim.arm_target
            if self.interpolate:
                self.controller.update()
            else:
                self.sim.set_arm_target(self._arm_target)  # команда сразу подаётся приводам
            self.max_step = max(self.max_step, np.abs(self.sim.arm_target - before).max())
            self.sim.step()


def pick_and_carry(robot):
    """Взять кубик, сложить руку и отвезти к красному столу. Возвращает, держит ли его захват в конце."""
    skills = Skills(robot)
    arm_folded = robot.arm_target
    cube = "cube_red_1"
    position, yaw = robot.sim.body_pos(cube), robot.sim.body_yaw(cube)
    skills.approach(position, 0.0)
    skills.pick(position, yaw)
    skills.move_arm(arm_folded)
    skills.retreat()
    skills.approach(np.array([0.0, 1.25, 0.42]), np.pi / 2)
    return skills.holding() and robot.sim.body_pos(cube)[2] > 0.6


@pytest.mark.parametrize("period", [0.02, 0.04, 0.08])
def test_cube_survives_low_rate_commands(period):
    robot = LowRateRobot(max_speed=2.0)
    robot.dt = period
    assert pick_and_carry(robot)
    assert robot.max_step <= 2.0 * robot.sim.dt + 1e-9


def test_without_the_controller_low_rate_commands_drop_the_cube():
    """Зачем контроллер нужен: те же команды 25 раз в секунду, поданные приводам напрямую."""
    robot = LowRateRobot(max_speed=2.0, interpolate=False)
    robot.dt = 0.04
    assert not pick_and_carry(robot)


@pytest.mark.parametrize("stall_at", [11.5, 12.0])
def test_cube_survives_a_stalled_control_loop(stall_at):
    """Программа управления замирает на секунду, когда рука с кубиком складывается.

    Без предела скорости рука догоняет цель рывком и в эти моменты роняет кубик.
    """
    assert not pick_and_carry(LowRateRobot(max_speed=np.inf, stall_at=stall_at, stall=1.0))
    robot = LowRateRobot(max_speed=2.0, stall_at=stall_at, stall=1.0)
    assert pick_and_carry(robot)
    assert robot.max_step <= 2.0 * robot.sim.dt + 1e-9
