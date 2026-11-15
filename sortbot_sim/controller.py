"""Контроллер руки симулятора: принимает редкие команды, как контроллер приводов настоящего робота.

Физика шагает 500 раз в секунду, а целевые углы от программы управления приходят десятки
раз в секунду и иногда с задержкой. Если подавать их приводам как есть, рука дёргается
на каждой команде и роняет кубик. Контроллер ведёт руку к последней команде плавно
и не быстрее, чем могут настоящие приводы.
"""
import numpy as np


class ArmController:
    def __init__(self, sim, max_speed=2.0, default_interval=0.02):
        """max_speed: предел скорости сочленения, рад/с."""
        self.sim = sim
        self.max_speed = max_speed
        self.default_interval = default_interval
        self.goal = sim.arm_target
        self.goal_time = sim.data.time
        self.last_command_time = None

    def command(self, q):
        """Пришла новая цель. Дойти до неё нужно к следующей команде, то есть через тот же интервал."""
        now = self.sim.data.time
        interval = self.default_interval if self.last_command_time is None else now - self.last_command_time
        self.last_command_time = now
        self.goal = np.array(q, dtype=float)
        self.goal_time = now + float(np.clip(interval, 2 * self.sim.dt, 0.1))

    def update(self):
        """Вызывать перед каждым шагом физики: сдвинуть цель приводов к последней команде."""
        sim = self.sim
        current = sim.arm_target
        remaining = self.goal_time - sim.data.time
        move = self.goal - current
        if remaining > sim.dt:
            move = move * sim.dt / remaining
        limit = self.max_speed * sim.dt
        sim.set_arm_target(current + np.clip(move, -limit, limit))
