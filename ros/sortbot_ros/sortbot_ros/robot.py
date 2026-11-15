"""Робот и зрение глазами ядра, но через топики ROS 2.

RosRobot даёт тот же интерфейс, что и SortingSim в симуляции (его ждёт sortbot_motion.Skills),
а RosVision тот же, что пара «камера + детектор» (его ждёт sortbot_task.Sorter).
Поэтому логика сортировки и движения работает через ROS без единой правки.
"""
import threading

# sortbot_sim импортируется раньше mujoco. Здесь MuJoCo нужен только как модель кинематики
# руки для обратной задачи: физика и рендер в этом узле не запускаются.
from sortbot_sim import SortingSim
from sortbot_sim.sim import GRIPPER_FULL_TRAVEL

import numpy as np
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformException
from vision_msgs.msg import Detection3DArray

from sortbot_ros import topics
from sortbot_ros.conversions import (from_stamp, msg_to_detections, msg_to_intrinsics, quaternion_to_yaw,
                                     transform_to_matrix)
from sortbot_vision.types import RgbdFrame


class RosRobot:
    """Интерфейс робота для sortbot_motion.Skills поверх топиков."""

    def __init__(self, node, tf_buffer, control_rate=50.0):
        self.node = node
        self.tf_buffer = tf_buffer
        self.dt = 1.0 / control_rate
        self.steps = 0
        self.first_step_time = None
        self.lock = threading.Lock()
        self.joints = {}  # имя сочленения -> положение по последнему /joint_states
        self.cmd_vel = np.zeros(3)
        self._arm_target = None
        self._gripper = 0.0
        self.next_tick = None
        self.kinematics = SortingSim()  # модель робота для обратной кинематики

        node.create_subscription(JointState, topics.JOINT_STATES, self._on_joint_states, 10)
        self.cmd_vel_pub = node.create_publisher(Twist, topics.CMD_VEL, 10)
        self.arm_pub = node.create_publisher(Float64MultiArray, topics.ARM_COMMAND, 10)
        self.gripper_pub = node.create_publisher(Float64MultiArray, topics.GRIPPER_COMMAND, 10)

    def _on_joint_states(self, msg):
        with self.lock:
            self.joints.update(zip(msg.name, msg.position))

    def wait_until_ready(self, timeout=30.0):
        """Дождаться первых данных от робота: углов сочленений и положения базы."""
        clock = self.node.get_clock()
        deadline = clock.now() + Duration(seconds=timeout)
        while True:
            with self.lock:
                have_joints = all(name in self.joints for name in topics.ARM_JOINT_NAMES)
            if have_joints and self.tf_buffer.can_transform(topics.MAP_FRAME, topics.BASE_FRAME, Time()):
                break
            if clock.now().nanoseconds > 0 and clock.now() > deadline:
                raise TimeoutError("Робот не отвечает: нет /joint_states или TF map -> base_link")
            threading.Event().wait(0.05)
        self._arm_target = self.arm_q  # начинаем с того положения, в котором рука сейчас

    # ---- состояние ----

    @property
    def time(self):
        """Часы робота, с. В симуляции это модельное время из топика /clock."""
        return self.node.get_clock().now().nanoseconds * 1e-9

    @property
    def actual_period(self):
        """Сколько в среднем длился период управления на самом деле, с."""
        if self.steps < 2:
            return self.dt
        return (self.time - self.first_step_time) / (self.steps - 1)

    @property
    def base_pose(self):
        transform = self.tf_buffer.lookup_transform(topics.MAP_FRAME, topics.BASE_FRAME, Time()).transform
        r = transform.rotation
        return np.array([transform.translation.x, transform.translation.y, quaternion_to_yaw(r.x, r.y, r.z, r.w)])

    @property
    def arm_q(self):
        with self.lock:
            return np.array([self.joints[name] for name in topics.ARM_JOINT_NAMES])

    @property
    def arm_target(self):
        return self._arm_target.copy()

    @property
    def gripper_closure(self):
        with self.lock:
            return float(np.clip(self.joints.get(topics.GRIPPER_JOINT_NAME, 0.0) / GRIPPER_FULL_TRAVEL, 0, 1))

    # ---- команды ----

    def set_cmd_vel(self, vx, vy, wz):
        self.cmd_vel[:] = vx, vy, wz

    def set_arm_target(self, q):
        self._arm_target = np.array(q, dtype=float)

    def set_gripper(self, closed):
        self._gripper = float(np.clip(closed, 0, 1))

    def solve_ik(self, target_pos, target_mat, q_init=None):
        model = self.kinematics
        model.data.qpos[model.base_qadr] = self.base_pose
        model.data.qpos[model.arm_qadr] = self.arm_q
        return model.solve_ik(target_pos, target_mat, q_init)

    def step(self):
        """Отправить текущие команды и дождаться следующего периода управления."""
        twist = Twist()
        twist.linear.x, twist.linear.y, twist.angular.z = map(float, self.cmd_vel)
        self.cmd_vel_pub.publish(twist)
        self.arm_pub.publish(Float64MultiArray(data=[float(q) for q in self._arm_target]))
        self.gripper_pub.publish(Float64MultiArray(data=[self._gripper]))

        clock = self.node.get_clock()
        now = clock.now()
        self.steps += 1
        if self.first_step_time is None:
            self.first_step_time = now.nanoseconds * 1e-9
        if self.next_tick is None or now - self.next_tick > Duration(seconds=5 * self.dt):
            self.next_tick = now  # первый вызов или сильно отстали: отсчёт заново
        self.next_tick = self.next_tick + Duration(seconds=self.dt)
        clock.sleep_until(self.next_tick)


class RosVision:
    """Заменяет ядру камеру и детектор: кадр здесь это свежее сообщение от узла зрения."""

    def __init__(self, node, tf_buffer, timeout=10.0):
        self.node = node
        self.tf_buffer = tf_buffer
        self.timeout = timeout
        self.lock = threading.Lock()
        self.latest = None
        self.intrinsics = None
        self.detections = {}  # время кадра -> находки
        node.create_subscription(Detection3DArray, topics.DETECTIONS, self._on_detections, 5)
        node.create_subscription(CameraInfo, topics.COLOR_INFO, self._on_camera_info, 5)

    def _on_detections(self, msg):
        with self.lock:
            self.latest = msg

    def _on_camera_info(self, msg):
        self.intrinsics = msg_to_intrinsics(msg)

    def capture(self):
        """Дождаться находок по кадру, снятому после этого вызова, и вернуть описание кадра."""
        clock = self.node.get_clock()
        asked = clock.now()
        deadline = asked + Duration(seconds=self.timeout)
        while True:
            with self.lock:
                msg = self.latest
            if msg is not None and self.intrinsics is not None and Time.from_msg(msg.header.stamp) > asked:
                break
            if clock.now() > deadline:
                raise RuntimeError("Узел зрения не прислал находки по свежему кадру")
            clock.sleep_for(Duration(seconds=0.02))

        try:
            transform = self.tf_buffer.lookup_transform(
                topics.MAP_FRAME, topics.CAMERA_FRAME, Time.from_msg(msg.header.stamp), timeout=Duration(seconds=1.0))
        except TransformException as error:
            raise RuntimeError(f"Нет позы камеры на момент кадра: {error}")
        stamp = from_stamp(msg.header.stamp)
        self.detections = {stamp: msg_to_detections(msg)}
        # Самих изображений здесь нет: их обрабатывает узел зрения. Ядру от кадра нужны
        # только параметры камеры и её поза, чтобы понять, что попадало в поле зрения.
        return RgbdFrame(rgb=None, depth=None, intrinsics=self.intrinsics,
                         world_from_camera=transform_to_matrix(transform.transform), stamp=stamp)

    def detect(self, frame, surface_z=None):
        """Находки для кадра из capture(). Высоту стола узел зрения берёт из своего параметра."""
        return self.detections[frame.stamp]
