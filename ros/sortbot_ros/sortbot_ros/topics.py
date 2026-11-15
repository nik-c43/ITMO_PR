"""Имена топиков и кадров в одном месте.

Камера названа так же, как её называет драйвер realsense2_camera, чтобы узел зрения
без правок работал и с симулятором, и с настоящей RealSense D435.
"""
CMD_VEL = "/cmd_vel"  # geometry_msgs/Twist: скорость базы в осях робота
ARM_COMMAND = "/arm_controller/commands"  # std_msgs/Float64MultiArray: целевые углы руки, рад
GRIPPER_COMMAND = "/gripper_controller/commands"  # std_msgs/Float64MultiArray: [0 открыт .. 1 закрыт]
JOINT_STATES = "/joint_states"  # sensor_msgs/JointState
ODOM = "/odom"  # nav_msgs/Odometry

COLOR_IMAGE = "/camera/camera/color/image_raw"  # sensor_msgs/Image, rgb8
COLOR_INFO = "/camera/camera/color/camera_info"  # sensor_msgs/CameraInfo
DEPTH_IMAGE = "/camera/camera/aligned_depth_to_color/image_raw"  # sensor_msgs/Image, 16UC1, мм

DETECTIONS = "/cubes/detections"  # vision_msgs/Detection3DArray в кадре map
DEBUG_IMAGE = "/cubes/debug_image"  # sensor_msgs/Image: кадр с обведёнными находками

GROUND_TRUTH = "/sim/ground_truth"  # std_msgs/String (JSON): истинные положения кубиков, только для проверки

MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_color_optical_frame"

ARM_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAME = "right_driver_joint"
