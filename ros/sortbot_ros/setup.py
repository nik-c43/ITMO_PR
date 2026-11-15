from glob import glob

from setuptools import setup

package_name = "sortbot_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Узлы ROS 2 для сортировки кубиков",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_bridge = sortbot_ros.sim_bridge:main",
            "cube_detector = sortbot_ros.cube_detector_node:main",
            "sorter = sortbot_ros.sorter_node:main",
            "check_sorted = sortbot_ros.check_sorted:main",
        ],
    },
)
