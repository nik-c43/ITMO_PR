import os

# Камера симулятора рендерит без окна через EGL. MuJoCo читает эту переменную в момент
# своего импорта, поэтому она задаётся здесь, до первого "import mujoco".
os.environ.setdefault("MUJOCO_GL", "egl")

from sortbot_sim.sim import SortingSim

__all__ = ["SortingSim"]
