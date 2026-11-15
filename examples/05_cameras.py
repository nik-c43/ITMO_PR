"""Шаг 5. Снимки с камер без окна: так программа «видит» мир.

Запуск: python examples/05_cameras.py
Результат: файлы PNG в папке out/.

На роботе две камеры: robot/base (на базе, смотрит вперёд-вниз) и robot/wrist
(на запястье). Для каждой сохраняем цветной кадр и карту глубины: расстояние
до поверхности в каждом пикселе. Это то же, что даёт стереокамера.
"""
# sortbot_sim импортируется раньше mujoco: он включает рендер без окна (MUJOCO_GL=egl).
from _setup import OUT_DIR
from sortbot_sim import SortingSim

import mujoco
import numpy as np
from PIL import Image

MAX_DEPTH = 4.0  # м, всё, что дальше, на карте глубины будет белым

sim = SortingSim()
# Подъедем к столу с кубиками, чтобы камерам было на что смотреть.
sim.set_cmd_vel(0.3, 0.0, 0.0)
for _ in range(int(1.5 / sim.dt)):
    sim.step()

OUT_DIR.mkdir(parents=True, exist_ok=True)
with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
    for camera in ["overview", "robot/base", "robot/wrist"]:
        name = camera.replace("/", "_")

        renderer.update_scene(sim.data, camera=camera)
        Image.fromarray(renderer.render()).save(OUT_DIR / f"{name}_rgb.png")

        renderer.enable_depth_rendering()
        depth = renderer.render()  # массив float32, метры
        renderer.disable_depth_rendering()
        gray = (np.clip(depth / MAX_DEPTH, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(gray).save(OUT_DIR / f"{name}_depth.png")

        print(f"{camera:12s} глубина от {depth.min():.2f} до {min(depth.max(), 99):.2f} м")

print("Сохранено в", OUT_DIR)
