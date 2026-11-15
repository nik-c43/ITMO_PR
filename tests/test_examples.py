"""Каждый пример без окна запускается и завершается успешно."""
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("command", [
    "02_inspect.py",
    "04_pick_and_place.py --headless",
    "05_cameras.py",
    "06_detect_cubes.py",
    "06_detect_cubes.py --ideal",
    "07_pick_by_vision.py --headless",
    "07_pick_by_vision.py --headless --color red",
    "07_pick_by_vision.py --headless --color blue",
    "08_sort_all.py --headless",
])
def test_example_runs(command):
    script, *args = command.split()
    done = subprocess.run([sys.executable, str(EXAMPLES / script), *args], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
