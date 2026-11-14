import os

# Камера симулятора рендерит без окна через EGL. MuJoCo читает эту переменную в момент
# своего импорта, поэтому она задаётся здесь, до первого "import mujoco".
os.environ.setdefault("MUJOCO_GL", "egl")

from sortbot_sim.ik import solve_ik
from sortbot_sim.sim import SCENE_PATH, SortingSim

__all__ = ["SCENE_PATH", "SortingSim", "solve_ik"]
