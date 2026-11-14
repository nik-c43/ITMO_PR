"""Тонкая обёртка над MuJoCo для сцены сортировки.

MuJoCo оперирует номерами: «привод №3», «ячейка qpos №7». Этот класс один раз
находит нужные номера по именам и даёт интерфейс в терминах робота:
скорость базы, углы руки, захват.
"""
from pathlib import Path

import mujoco
import numpy as np

from sortbot_sim.ik import solve_ik

SCENE_PATH = Path(__file__).resolve().parents[1] / "scenes" / "sorting.xml"

BASE_JOINTS = ["robot/joint_x", "robot/joint_y", "robot/joint_th"]
ARM_JOINTS = [f"robot/joint_{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "robot/fingers_actuator"
PINCH_SITE = "robot/pinch_site"
GRIPPER_FULL_TRAVEL = 0.8  # ход привода захвата от открытого до сжатого до упора, рад

# Насколько цель привода базы может опережать саму базу (м, м, рад).
# Привод базы очень жёсткий: без этого предела робот, упёршись в стол,
# давил бы на него с силой в десятки килоньютонов.
BASE_MAX_LEAD = np.array([0.05, 0.05, 0.10])


class SortingSim:
    def __init__(self, scene_path=SCENE_PATH):
        # MjModel: неизменяемое описание мира (тела, массы, приводы).
        # MjData: изменяемое состояние (время, координаты, скорости, силы).
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        m = self.model
        # Адреса координат в data.qpos (положения) и data.qvel (скорости).
        self.base_qadr = np.array([m.joint(n).qposadr[0] for n in BASE_JOINTS])
        self.arm_qadr = np.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.arm_dofadr = np.array([m.joint(n).dofadr[0] for n in ARM_JOINTS])
        # Номера приводов в data.ctrl. Имена приводов совпадают с именами сочленений.
        self.base_act = np.array([m.actuator(n).id for n in BASE_JOINTS])
        self.arm_act = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self.gripper_act = m.actuator(GRIPPER_ACTUATOR).id
        self.pinch_site = m.site(PINCH_SITE).id

        self.cmd_vel = np.zeros(3)
        self.reset()

    def reset(self, keyframe="robot/home"):
        """Вернуть мир в исходное состояние, робот в позе из ключевого кадра."""
        key_id = self.model.key(keyframe).id
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        self.cmd_vel[:] = 0

    # ---- чтение состояния ----

    @property
    def dt(self):
        return self.model.opt.timestep

    @property
    def base_pose(self):
        """Положение базы в мире: (x, y, угол поворота)."""
        return self.data.qpos[self.base_qadr].copy()

    @property
    def arm_q(self):
        """Углы семи сочленений руки, рад."""
        return self.data.qpos[self.arm_qadr].copy()

    @property
    def arm_target(self):
        """Углы, которые сейчас заданы приводам руки, рад."""
        return self.data.ctrl[self.arm_act].copy()

    @property
    def gripper_closure(self):
        """Насколько сжат захват на самом деле: 0 открыт, 1 сжат до упора.

        Сжатый пустой захват даёт около 1, а с кубиком 4 см между пальцами около 0.57:
        так робот узнаёт, что кубик взят, не подглядывая в симулятор.
        """
        return float(np.clip(self.data.actuator_length[self.gripper_act] / GRIPPER_FULL_TRAVEL, 0, 1))

    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def body_yaw(self, name):
        """Угол поворота тела вокруг вертикали, рад."""
        mat = self.data.body(name).xmat.reshape(3, 3)
        return np.arctan2(mat[1, 0], mat[0, 0])

    def solve_ik(self, target_pos, target_mat, q_init=None):
        """Углы руки, при которых точка захвата окажется в заданной позе в мире: (углы, сошлось ли)."""
        return solve_ik(self, target_pos, target_mat, q_init=q_init)

    # ---- команды ----

    def set_cmd_vel(self, vx, vy, wz):
        """Скорость базы в системе координат робота, как ROS-сообщение cmd_vel.

        vx: вперёд (м/с), vy: влево (м/с), wz: поворот против часовой (рад/с).
        """
        self.cmd_vel[:] = vx, vy, wz

    def set_arm_target(self, q):
        self.data.ctrl[self.arm_act] = q

    def set_gripper(self, closed):
        """0: захват открыт, 1: закрыт."""
        self.data.ctrl[self.gripper_act] = 255 * float(np.clip(closed, 0, 1))

    def step(self):
        """Один шаг физики (model.opt.timestep секунд)."""
        self._integrate_cmd_vel()
        mujoco.mj_step(self.model, self.data)

    def _integrate_cmd_vel(self):
        # Приводы базы принимают целевое ПОЛОЖЕНИЕ (x, y, угол) в мире,
        # а мы хотим командовать СКОРОСТЬЮ в осях робота. Поэтому на каждом
        # шаге поворачиваем скорость в мировые оси и сдвигаем цель на v * dt.
        target = self.data.ctrl[self.base_act].copy()
        vx, vy, wz = self.cmd_vel
        th = target[2]
        target[0] += (vx * np.cos(th) - vy * np.sin(th)) * self.dt
        target[1] += (vx * np.sin(th) + vy * np.cos(th)) * self.dt
        target[2] += wz * self.dt

        pose = self.base_pose
        target = pose + np.clip(target - pose, -BASE_MAX_LEAD, BASE_MAX_LEAD)
        self.data.ctrl[self.base_act] = target
