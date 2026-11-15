"""Перевод между структурами ядра (numpy, dataclass) и сообщениями ROS 2."""
import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Transform
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from sortbot_vision.types import CameraIntrinsics, CubeDetection

# ---- время ----


def to_stamp(seconds):
    whole = int(seconds)
    return Time(sec=whole, nanosec=int(round((seconds - whole) * 1e9)) % 1_000_000_000)


def from_stamp(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


# ---- повороты ----


def matrix_to_quaternion(rotation):
    """Матрица поворота 3x3 -> кватернион (x, y, z, w), как в сообщениях ROS."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    quat = np.zeros(4)
    quat[i] = 0.25 * s
    quat[j] = (m[j, i] + m[i, j]) / s
    quat[k] = (m[k, i] + m[i, k]) / s
    quat[3] = (m[k, j] - m[j, k]) / s
    return quat


def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_to_quaternion(yaw):
    return np.array([0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)])


def quaternion_to_yaw(x, y, z, w):
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def matrix_to_transform(a_from_b):
    """Матрица 4x4 -> geometry_msgs/Transform."""
    transform = Transform()
    transform.translation.x, transform.translation.y, transform.translation.z = map(float, a_from_b[:3, 3])
    q = matrix_to_quaternion(a_from_b[:3, :3])
    transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w = map(float, q)
    return transform


def transform_to_matrix(transform):
    a_from_b = np.eye(4)
    r = transform.rotation
    a_from_b[:3, :3] = quaternion_to_matrix(r.x, r.y, r.z, r.w)
    t = transform.translation
    a_from_b[:3, 3] = [t.x, t.y, t.z]
    return a_from_b


# ---- изображения ----

_IMAGE_TYPES = {"rgb8": (np.uint8, 3), "16UC1": (np.uint16, 1), "32FC1": (np.float32, 1)}


def image_to_msg(array, encoding, stamp, frame_id):
    dtype, channels = _IMAGE_TYPES[encoding]
    array = np.ascontiguousarray(array, dtype=dtype)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = array.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels * array.itemsize
    msg.data = array.tobytes()
    return msg


def msg_to_image(msg):
    dtype, channels = _IMAGE_TYPES[msg.encoding]
    array = np.frombuffer(msg.data, dtype=dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    return array.reshape(shape)


def depth_to_msg(depth_m, stamp, frame_id):
    """Глубина в метрах -> 16UC1 в миллиметрах, как отдаёт драйвер RealSense. Ноль: нет данных."""
    return image_to_msg(np.round(depth_m * 1000.0).astype(np.uint16), "16UC1", stamp, frame_id)


def msg_to_depth(msg):
    """Изображение глубины (16UC1 в миллиметрах или 32FC1 в метрах) -> метры, float32."""
    array = msg_to_image(msg)
    if msg.encoding == "16UC1":
        return array.astype(np.float32) / 1000.0
    return np.nan_to_num(array.astype(np.float32), nan=0.0)


def intrinsics_to_msg(intrinsics, stamp, frame_id):
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width, msg.height = intrinsics.width, intrinsics.height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def msg_to_intrinsics(msg):
    return CameraIntrinsics(msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])


# ---- находки ----


def detections_to_msg(detections, stamp, frame_id, cube_size=0.04):
    """Список CubeDetection -> vision_msgs/Detection3DArray.

    Цвет кладётся в class_id, центр и поворот кубика в bbox.center, стороны в bbox.size.
    """
    msg = Detection3DArray()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    for det in detections:
        item = Detection3D()
        item.header = msg.header
        pose = item.bbox.center
        pose.position.x, pose.position.y, pose.position.z = map(float, det.position)
        q = yaw_to_quaternion(det.yaw)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
        item.bbox.size.x, item.bbox.size.y, item.bbox.size.z = float(det.size[0]), float(det.size[1]), cube_size
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = det.color
        hypothesis.hypothesis.score = 1.0
        hypothesis.pose.pose = pose
        item.results.append(hypothesis)
        msg.detections.append(item)
    return msg


def msg_to_detections(msg):
    detections = []
    for item in msg.detections:
        pose, size = item.bbox.center, item.bbox.size
        o = pose.orientation
        detections.append(CubeDetection(
            color=item.results[0].hypothesis.class_id,
            position=np.array([pose.position.x, pose.position.y, pose.position.z]),
            yaw=quaternion_to_yaw(o.x, o.y, o.z, o.w),
            size=(size.x, size.y),
        ))
    return detections
