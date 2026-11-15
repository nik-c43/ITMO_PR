"""Узел сортировки: запускает логику из sortbot_task поверх топиков ROS 2.

    слушает      /cubes/detections, /joint_states, camera_info, TF
    публикует    /cmd_vel, /arm_controller/commands, /gripper_controller/commands

Описание столов приходит параметрами (config/sorting.yaml). Узел не знает,
симулятор на другом конце или настоящий робот.
"""
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from sortbot_motion import Skills
from sortbot_ros.robot import RosRobot, RosVision
from sortbot_ros.tables import read_tables
from sortbot_task import Sorter


def main():
    rclpy.init()
    node = Node("sorter", automatically_declare_parameters_from_overrides=True)
    log = node.get_logger()

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)  # noqa: F841
    params = node.get_parameters_by_prefix("")
    control_rate = params["control_rate"].value if "control_rate" in params else 50.0
    robot = RosRobot(node, tf_buffer, control_rate)
    vision = RosVision(node, tf_buffer)

    # Сообщения принимаются в фоновом потоке, а логика сортировки идёт в главном:
    # она написана как обычная последовательная программа с ожиданиями.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except Exception:  # при завершении узла фоновый поток прерывается: это не ошибка
            pass

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()

    try:
        tables = read_tables(node)
        log.info("Столы: " + ", ".join(f"{t.name} ({t.color or 'источник'})" for t in tables))
        robot.wait_until_ready()
        robot.set_gripper(0.0)
        sorter = Sorter(Skills(robot), vision, vision, tables, log=log.info)
        result = sorter.run()
        log.info(f"Готово: отвезено {len(result.delivered)}, осталось {len(result.left)}, "
                 f"осмотров стола {result.rounds}, время {robot.time:.0f} с")
        log.info(f"Период управления: задан {1000 * robot.dt:.0f} мс, "
                 f"в среднем вышел {1000 * robot.actual_period:.0f} мс")
    except KeyboardInterrupt:
        pass
    finally:
        # Сначала остановить приём сообщений и дождаться фонового потока, потом закрывать ROS:
        # в обратном порядке узел падает при выходе.
        executor.shutdown(timeout_sec=2.0)
        spinner.join(timeout=2.0)
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
