"""Запуск всей системы в симуляции: мост с MuJoCo, зрение, сортировка.

    ros2 launch sortbot_ros sorting.launch.py              в реальном времени
    ros2 launch sortbot_ros sorting.launch.py speed:=3.0   симуляция втрое быстрее

Когда узел сортировки заканчивает, результат сверяется с симулятором, и всё останавливается.
На настоящем роботе вместо sim_bridge запускаются драйверы, а use_sim_time выключается.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(get_package_share_directory("sortbot_ros"), "config", "sorting.yaml")
    sim_time = {"use_sim_time": True}  # время берётся из топика /clock, который публикует мост

    bridge = Node(package="sortbot_ros", executable="sim_bridge", output="screen",
                  parameters=[config, {"speed": ParameterValue(LaunchConfiguration("speed"), value_type=float)}])
    detector = Node(package="sortbot_ros", executable="cube_detector", output="screen",
                    parameters=[config, sim_time])
    sorter = Node(package="sortbot_ros", executable="sorter", output="screen", parameters=[config, sim_time])
    check = Node(package="sortbot_ros", executable="check_sorted", output="screen", parameters=[config])

    return LaunchDescription([
        DeclareLaunchArgument("speed", default_value="1.0", description="во сколько раз симуляция быстрее часов"),
        bridge,
        detector,
        sorter,
        RegisterEventHandler(OnProcessExit(target_action=sorter, on_exit=[check])),
        RegisterEventHandler(OnProcessExit(target_action=check, on_exit=[EmitEvent(event=Shutdown())])),
    ])
