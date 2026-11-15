"""Узел зрения: кадры камеры -> найденные кубики.

    слушает      цвет, глубину, camera_info (топики как у realsense2_camera), TF
    публикует    /cubes/detections (vision_msgs/Detection3DArray, кадр map)

Вся обработка изображения живёт в sortbot_vision. Узел только собирает RgbdFrame
из сообщений ROS и переводит результат обратно. С настоящей камерой он работает без правок.
"""
import math

import rclpy
from message_filters import Subscriber, TimeSynchronizer
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from sortbot_ros import topics
from sortbot_ros.conversions import (detections_to_msg, image_to_msg, msg_to_depth, msg_to_image,
                                     msg_to_intrinsics, transform_to_matrix)
from sortbot_vision.cubes import CubeDetector, CubeDetectorConfig, draw_detections
from sortbot_vision.types import RgbdFrame


class CubeDetectorNode(Node):
    def __init__(self):
        super().__init__("cube_detector")
        # Высота столешницы, на которой лежат кубики. nan: неизвестна, искать по всему кадру.
        surface_z = self.declare_parameter("surface_z", math.nan).value
        self.surface_z = None if math.isnan(surface_z) else surface_z
        self.target_frame = self.declare_parameter("target_frame", topics.MAP_FRAME).value
        cube_size = self.declare_parameter("cube_size", 0.04).value
        self.detector = CubeDetector(CubeDetectorConfig(cube_size=cube_size))

        self.intrinsics = None
        self.create_subscription(CameraInfo, topics.COLOR_INFO, self.on_camera_info, 5)
        # Цвет и глубина приходят отдельными сообщениями с одинаковым временем: собираем их в пары.
        color = Subscriber(self, Image, topics.COLOR_IMAGE)
        depth = Subscriber(self, Image, topics.DEPTH_IMAGE)
        self.sync = TimeSynchronizer([color, depth], queue_size=3)
        self.sync.registerCallback(self.on_frame)

        # Кадр может прийти чуть раньше позы камеры на тот же момент, и обработчик кадра её ждёт.
        # Чтобы TF в это время продолжал приниматься, узел работает в несколько потоков (см. main).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.detections_pub = self.create_publisher(Detection3DArray, topics.DETECTIONS, 5)
        self.debug_pub = self.create_publisher(Image, topics.DEBUG_IMAGE, 2)

    def on_camera_info(self, msg):
        self.intrinsics = msg_to_intrinsics(msg)

    def on_frame(self, color_msg, depth_msg):
        if self.intrinsics is None:
            return
        stamp = color_msg.header.stamp
        try:
            # Где была камера в момент кадра.
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, color_msg.header.frame_id, Time.from_msg(stamp), timeout=Duration(seconds=0.5))
        except TransformException as error:
            self.get_logger().warn(f"Нет позы камеры на момент кадра: {error}", throttle_duration_sec=2.0)
            return

        frame = RgbdFrame(
            rgb=msg_to_image(color_msg),
            depth=msg_to_depth(depth_msg),
            intrinsics=self.intrinsics,
            world_from_camera=transform_to_matrix(transform.transform),
            stamp=stamp.sec + stamp.nanosec * 1e-9,
        )
        detections = self.detector.detect(frame, surface_z=self.surface_z)
        self.detections_pub.publish(
            detections_to_msg(detections, stamp, self.target_frame, self.detector.config.cube_size))

        if self.debug_pub.get_subscription_count() > 0:  # рисуем, только если кто-то смотрит
            image = draw_detections(frame, detections, self.detector.config.cube_size)
            self.debug_pub.publish(image_to_msg(image, "rgb8", stamp, color_msg.header.frame_id))


def main():
    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(CubeDetectorNode())
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
