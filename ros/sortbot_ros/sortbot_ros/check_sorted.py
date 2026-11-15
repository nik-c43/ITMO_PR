"""Проверка результата по истинным положениям кубиков из симулятора.

Нужна только для отладки в симуляции: на настоящем роботе топика /sim/ground_truth нет.
Код возврата 0, если каждый кубик стоит на столе своего цвета.
"""
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from sortbot_ros import topics
from sortbot_ros.tables import read_tables
from sortbot_sim.world import placement_report


def main():
    rclpy.init()
    node = Node("check_sorted", automatically_declare_parameters_from_overrides=True)
    tables = read_tables(node)
    received = []
    node.create_subscription(String, topics.GROUND_TRUTH, lambda msg: received.append(json.loads(msg.data)), 5)
    while rclpy.ok() and not received:
        rclpy.spin_once(node, timeout_sec=0.5)

    truth = received[0]
    lines, correct, total = placement_report(tables, truth["cubes"])
    print(f"Проверка по симулятору (модельное время {truth['time']:.0f} с):")
    print("\n".join(lines))
    print(f"РЕЗУЛЬТАТ: на своих столах {correct} из {total}")
    rclpy.try_shutdown()
    sys.exit(0 if correct == total else 1)


if __name__ == "__main__":
    main()
