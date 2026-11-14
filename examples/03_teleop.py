"""Шаг 3. Управление роботом с клавиатуры.

Запуск: python examples/03_teleop.py   (окно просмотрщика должно быть в фокусе)

  стрелки вверх/вниз   : быстрее вперёд / назад
  стрелки влево/вправо : быстрее влево / вправо (боком, база всенаправленная)
  7 / 9                : вращение против / по часовой стрелке
  8                    : полная остановка
  6                    : открыть / закрыть захват
  Backspace            : сбросить сцену

Просмотрщик сообщает только о нажатии клавиши, но не об отпускании, поэтому
каждое нажатие меняет скорость на одну ступень, а не «едет, пока держишь».

В отличие от шага 1, здесь цикл физики крутит наш код, а просмотрщик только
рисует (режим launch_passive). Так устроена любая программа управления.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from sortbot_sim import SortingSim

LIN_STEP, LIN_MAX = 0.1, 0.5  # м/с
ANG_STEP, ANG_MAX = 0.3, 1.2  # рад/с

# Коды клавиш GLFW
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_BACKSPACE = 262, 263, 264, 265, 259


class Teleop:
    def __init__(self, sim):
        self.sim = sim
        self.vel = np.zeros(3)
        self.gripper_closed = False
        self.reset_requested = False

    def on_key(self, key):
        # Вызывается из потока окна, поэтому здесь только запоминаем команду.
        if key == KEY_UP:
            self.vel[0] += LIN_STEP
        elif key == KEY_DOWN:
            self.vel[0] -= LIN_STEP
        elif key == KEY_LEFT:
            self.vel[1] += LIN_STEP
        elif key == KEY_RIGHT:
            self.vel[1] -= LIN_STEP
        elif key == ord("7"):
            self.vel[2] += ANG_STEP
        elif key == ord("9"):
            self.vel[2] -= ANG_STEP
        elif key == ord("8"):
            self.vel[:] = 0
        elif key == ord("6"):
            self.gripper_closed = not self.gripper_closed
        elif key == KEY_BACKSPACE:
            self.reset_requested = True
        else:
            return
        self.vel[:2] = np.clip(self.vel[:2], -LIN_MAX, LIN_MAX)
        self.vel[2] = np.clip(self.vel[2], -ANG_MAX, ANG_MAX)
        print(f"cmd_vel: vx={self.vel[0]:+.1f} м/с  vy={self.vel[1]:+.1f} м/с  wz={self.vel[2]:+.1f} рад/с  "
              f"захват: {'закрыт' if self.gripper_closed else 'открыт'}")

    def apply(self):
        if self.reset_requested:
            self.reset_requested = False
            self.vel[:] = 0
            self.gripper_closed = False
            self.sim.reset()
        self.sim.set_cmd_vel(*self.vel)
        self.sim.set_gripper(1.0 if self.gripper_closed else 0.0)


def main():
    sim = SortingSim()
    teleop = Teleop(sim)
    print(__doc__)

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=teleop.on_key) as viewer:
        while viewer.is_running():
            start = time.time()
            teleop.apply()
            sim.step()
            viewer.sync()
            # Ждём, чтобы симуляция шла в реальном времени.
            time.sleep(max(0.0, sim.dt - (time.time() - start)))


if __name__ == "__main__":
    main()
