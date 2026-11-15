"""Узел-мост: симулятор MuJoCo, который снаружи выглядит как настоящий робот.

Узел крутит физику и общается с остальной системой только через топики ROS 2:

    слушает      /cmd_vel, /arm_controller/commands, /gripper_controller/commands
    публикует    /clock, /joint_states, /odom, TF, кадры камеры

На настоящем роботе этого узла нет: те же топики обслуживают драйвер платы youBot,
драйвер realsense2_camera и robot_state_publisher. Остальные узлы разницы не видят.
"""
import json

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.controller import ArmController
from sortbot_sim.world import cube_positions

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rosgraph_msgs.msg import Clock as ClockMsg
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from sortbot_ros import topics
from sortbot_ros.conversions import (depth_to_msg, image_to_msg, intrinsics_to_msg, matrix_to_transform,
                                     to_stamp, yaw_to_quaternion)

TICK = 0.01  # как часто узел просыпается, секунды настоящего времени
MAX_ADVANCE = 0.1  # самое большее модельного времени за одно пробуждение, с


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")
        self.speed = self.declare_parameter("speed", 1.0).value  # во сколько раз быстрее настоящего времени
        self.state_period = 1.0 / self.declare_parameter("state_rate", 50.0).value
        self.camera_period = 1.0 / self.declare_parameter("camera_rate", 5.0).value
        self.cmd_vel_timeout = self.declare_parameter("cmd_vel_timeout", 0.5).value

        self.sim = SortingSim()
        self.camera = SimCamera(self.sim)  # модель RealSense D435
        self.sim.set_gripper(0.0)
        self.last_cmd_vel_time = -np.inf
        self.target_time = 0.0  # до какого модельного времени нужно досчитать
        # Целевые углы руки приходят редко (десятки раз в секунду), а физика шагает 500 раз.
        # Как и контроллер приводов настоящего робота, мост плавно ведёт руку от одной
        # цели к следующей и ограничивает скорость. Без этого рука дёргается и роняет кубик.
        self.arm = ArmController(self.sim, max_speed=self.declare_parameter("arm_max_speed", 2.0).value)
        self.next_state_time = 0.0
        self.next_camera_time = 0.0
        self.next_truth_time = 0.0
        self.base_dofadr = [self.sim.model.joint(n).dofadr[0] for n in
                            ("robot/joint_x", "robot/joint_y", "robot/joint_th")]

        self.create_subscription(Twist, topics.CMD_VEL, self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, topics.ARM_COMMAND, self.on_arm_command, 10)
        self.create_subscription(Float64MultiArray, topics.GRIPPER_COMMAND, self.on_gripper_command, 10)

        self.clock_pub = self.create_publisher(ClockMsg, "/clock", 10)
        self.joint_pub = self.create_publisher(JointState, topics.JOINT_STATES, 10)
        self.odom_pub = self.create_publisher(Odometry, topics.ODOM, 10)
        self.color_pub = self.create_publisher(Image, topics.COLOR_IMAGE, 5)
        self.depth_pub = self.create_publisher(Image, topics.DEPTH_IMAGE, 5)
        self.info_pub = self.create_publisher(CameraInfo, topics.COLOR_INFO, 5)
        self.truth_pub = self.create_publisher(String, topics.GROUND_TRUTH, 5)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Локализации пока нет: карта и одометрия совпадают. Позже map -> odom будет публиковать
        # узел локализации по лидару.
        self.static_broadcaster = StaticTransformBroadcaster(self)
        identity = TransformStamped()
        identity.header.frame_id = topics.MAP_FRAME
        identity.child_frame_id = topics.ODOM_FRAME
        identity.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(identity)

        # Таймер идёт по настоящим часам: модельное время создаёт сам этот узел.
        self.wall = Clock(clock_type=ClockType.STEADY_TIME)
        self.last_wall = self.wall.now()
        self.create_timer(TICK, self.on_tick, clock=self.wall)
        self.get_logger().info(f"Симуляция запущена, скорость x{self.speed:g}")

    # ---- команды ----

    def on_cmd_vel(self, msg):
        self.sim.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)
        self.last_cmd_vel_time = self.sim.data.time

    def on_arm_command(self, msg):
        if len(msg.data) == len(topics.ARM_JOINT_NAMES):
            self.arm.command(msg.data)

    def on_gripper_command(self, msg):
        if msg.data:
            self.sim.set_gripper(msg.data[0])

    # ---- физика и публикация ----

    def on_tick(self):
        now = self.wall.now()
        elapsed = (now - self.last_wall).nanoseconds * 1e-9
        self.last_wall = now
        # Цель копится отдельно от времени симуляции: физика шагает целыми шагами и каждый раз
        # чуть перелетает цель, и без этого симуляция шла бы на несколько процентов быстрее заданного.
        self.target_time = max(self.target_time, self.sim.data.time - self.sim.dt)
        self.target_time += min(self.speed * elapsed, MAX_ADVANCE)
        while self.sim.data.time < self.target_time:
            # Как на настоящем роботе: если команды скорости перестали приходить, база останавливается.
            if self.sim.data.time - self.last_cmd_vel_time > self.cmd_vel_timeout:
                self.sim.set_cmd_vel(0.0, 0.0, 0.0)
            self.arm.update()
            self.sim.step()
            if self.sim.data.time >= self.next_state_time:
                self.next_state_time += self.state_period
                # Часы публикуются вместе с состоянием, 50 раз в секунду модельного времени:
                # остальные узлы отсчитывают по ним свои периоды управления.
                self.clock_pub.publish(ClockMsg(clock=to_stamp(self.sim.data.time)))
                self.publish_state()

    def publish_state(self):
        sim, data = self.sim, self.sim.data
        stamp = to_stamp(data.time)

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = topics.ARM_JOINT_NAMES + [topics.GRIPPER_JOINT_NAME]
        joints.position = [float(q) for q in sim.arm_q] + [float(data.actuator_length[sim.gripper_act])]
        self.joint_pub.publish(joints)

        # Положение базы. В симуляции одометрия идеальна; на роботе её считает драйвер по колёсам.
        x, y, th = sim.base_pose
        base = TransformStamped()
        base.header.stamp = stamp
        base.header.frame_id = topics.ODOM_FRAME
        base.child_frame_id = topics.BASE_FRAME
        base.transform.translation.x, base.transform.translation.y = float(x), float(y)
        q = yaw_to_quaternion(th)
        base.transform.rotation.z, base.transform.rotation.w = float(q[2]), float(q[3])

        odom = Odometry()
        odom.header = base.header
        odom.child_frame_id = topics.BASE_FRAME
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
        odom.pose.pose.orientation = base.transform.rotation
        vx, vy, wz = data.qvel[self.base_dofadr]  # скорость базы в осях мира
        odom.twist.twist.linear.x = float(np.cos(th) * vx + np.sin(th) * vy)
        odom.twist.twist.linear.y = float(-np.sin(th) * vx + np.cos(th) * vy)
        odom.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(odom)

        # Где камера относительно базы. На роботе это даёт robot_state_publisher по углам руки.
        world_from_base = np.eye(4)
        world_from_base[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
        world_from_base[:2, 3] = x, y
        world_from_camera = self.camera.pose()
        camera = TransformStamped()
        camera.header.stamp = stamp
        camera.header.frame_id = topics.BASE_FRAME
        camera.child_frame_id = topics.CAMERA_FRAME
        camera.transform = matrix_to_transform(np.linalg.inv(world_from_base) @ world_from_camera)
        self.tf_broadcaster.sendTransform([base, camera])

        if data.time >= self.next_camera_time:
            self.next_camera_time = data.time + self.camera_period
            frame = self.camera.capture()
            self.color_pub.publish(image_to_msg(frame.rgb, "rgb8", stamp, topics.CAMERA_FRAME))
            self.depth_pub.publish(depth_to_msg(frame.depth, stamp, topics.CAMERA_FRAME))
            self.info_pub.publish(intrinsics_to_msg(frame.intrinsics, stamp, topics.CAMERA_FRAME))

        if data.time >= self.next_truth_time:
            self.next_truth_time = data.time + 0.5
            truth = {name: [round(float(v), 4) for v in position] for name, position in cube_positions(sim).items()}
            self.truth_pub.publish(String(data=json.dumps({"time": round(data.time, 2), "cubes": truth})))


def main():
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.camera.close()


if __name__ == "__main__":
    main()
