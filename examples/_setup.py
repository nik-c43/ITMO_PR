"""Общее для примеров: куда сохранять картинки и сборка системы для шагов 7 и 8."""
import os
from pathlib import Path

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import SimCamera
from sortbot_sim.window import Window
from sortbot_sim.world import tables_from_sim
from sortbot_motion import Skills
from sortbot_task import Sorter
from sortbot_vision.cubes import CubeDetector

# Куда примеры сохраняют картинки. В контейнере папка проекта доступна только для чтения,
# поэтому docker/run.sh направляет вывод в другое место переменной SORTBOT_OUT.
OUT_DIR = Path(os.environ.get("SORTBOT_OUT", Path(__file__).resolve().parents[1] / "out"))


def build(headless):
    """Собрать систему. Возвращает (симулятор, окно, столы, сортировщик)."""
    sim = SortingSim()
    window = Window(sim, headless)  # окно создаётся раньше камеры
    window.show_camera()  # в углу окна: что сейчас видит камера на запястье
    camera = SimCamera(sim)  # модель RealSense D435
    window.close_first.append(camera)

    tables = tables_from_sim(sim)
    sorter = Sorter(Skills(sim, on_step=window.on_step), camera, CubeDetector(), tables)
    sim.set_gripper(0.0)
    return sim, window, tables, sorter
