"""Проверки перевода между структурами ядра и сообщениями ROS 2. Запускаются в контейнере:

    docker/run.sh python3 -m pytest -p no:cacheprovider ros/sortbot_ros/test
"""
import numpy as np

from sortbot_ros.conversions import (depth_to_msg, detections_to_msg, from_stamp, image_to_msg,
                                     intrinsics_to_msg, matrix_to_quaternion, matrix_to_transform,
                                     msg_to_depth, msg_to_detections, msg_to_image, msg_to_intrinsics,
                                     quaternion_to_matrix, to_stamp, transform_to_matrix)
from sortbot_vision.types import CameraIntrinsics, CubeDetection


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quaternion_to_matrix(*q)


def test_stamp_round_trip():
    for seconds in (0.0, 1.5, 123.456789, 703.02):
        assert abs(from_stamp(to_stamp(seconds)) - seconds) < 1e-8


def test_rotation_and_transform_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rotation = random_rotation(rng)
        np.testing.assert_allclose(quaternion_to_matrix(*matrix_to_quaternion(rotation)), rotation, atol=1e-9)
        a_from_b = np.eye(4)
        a_from_b[:3, :3] = rotation
        a_from_b[:3, 3] = rng.normal(size=3)
        np.testing.assert_allclose(transform_to_matrix(matrix_to_transform(a_from_b)), a_from_b, atol=1e-9)


def test_quaternion_of_half_turns():
    # Повороты на 180 градусов: случай, где простая формула делит на ноль.
    for diagonal in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1]):
        rotation = np.diag(diagonal).astype(float)
        np.testing.assert_allclose(quaternion_to_matrix(*matrix_to_quaternion(rotation)), rotation, atol=1e-9)


def test_color_image_round_trip():
    rgb = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    msg = image_to_msg(rgb, "rgb8", to_stamp(1.0), "camera")
    assert (msg.height, msg.width, msg.step, msg.encoding) == (48, 64, 64 * 3, "rgb8")
    np.testing.assert_array_equal(msg_to_image(msg), rgb)


def test_depth_goes_through_millimeters_and_keeps_missing_pixels():
    depth = np.array([[0.0, 0.3404, 1.2346], [2.9999, 0.0, 0.28]], np.float32)
    restored = msg_to_depth(depth_to_msg(depth, to_stamp(1.0), "camera"))
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, depth, atol=0.0005)  # точность 1 мм, как у RealSense
    assert restored[0, 0] == 0.0 and restored[1, 1] == 0.0


def test_intrinsics_round_trip():
    intrinsics = CameraIntrinsics.from_fov(1280, 720, 42.5)
    assert msg_to_intrinsics(intrinsics_to_msg(intrinsics, to_stamp(0.0), "camera")) == intrinsics


def test_detections_round_trip():
    detections = [
        CubeDetection("red", np.array([1.12, -0.3, 0.42]), np.radians(15), (0.041, 0.039)),
        CubeDetection("blue", np.array([1.26, 0.15, 0.42]), np.radians(-40), (0.040, 0.040)),
    ]
    msg = detections_to_msg(detections, to_stamp(5.0), "map")
    assert msg.header.frame_id == "map" and len(msg.detections) == 2
    for before, after in zip(detections, msg_to_detections(msg)):
        assert after.color == before.color
        np.testing.assert_allclose(after.position, before.position)
        assert abs(after.yaw - before.yaw) < 1e-9
        np.testing.assert_allclose(after.size, before.size)
