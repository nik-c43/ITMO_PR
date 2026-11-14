"""Проверка размера картинок с камеры в окне. Настоящее окно не открывается: вместо него подставка."""
# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.window import PAD, Window

import mujoco
import pytest


class FakeViewer:
    def __init__(self, width, height, left=540):
        self.viewport = mujoco.MjrRect(left, 0, width, height)
        self.images = None

    def is_running(self):
        return True

    def set_images(self, items):
        self.images = [(rect.left, rect.bottom, rect.width, rect.height, image.shape) for rect, image in items]

    def clear_images(self):
        self.images = []

    def sync(self):
        pass


@pytest.fixture(scope="module")
def window():
    win = Window(SortingSim(), headless=True)
    win.viewer = FakeViewer(1988, 1862)
    win.show_camera(fraction=0.45)
    yield win
    win.view_camera.close()


def show(window, width, height):
    window.viewer = FakeViewer(width, height)
    window.next_view_time = 0.0
    window._update_camera_view()
    return window.viewer.images


@pytest.mark.parametrize("area_width", [1988, 1200, 800, 3000])
def test_each_view_takes_the_given_fraction_of_the_scene_width(window, area_width):
    color, depth = show(window, area_width, 2000)
    left, bottom, width, height, shape = color
    assert width == int(0.45 * area_width) and height == round(width * 506 / 900)
    assert shape == (height, width, 3) and depth[2:] == (width, height, shape)
    # Обе картинки целиком внутри области сцены, прижаты к правому нижнему углу.
    assert depth[0] + width == 540 + area_width - PAD and bottom == PAD
    assert left >= 540 and depth[0] == left + width + PAD


def test_views_shrink_to_fit_a_low_window(window):
    color, _ = show(window, 3000, 300)
    assert color[3] <= 300 - 2 * PAD and color[2] < int(0.45 * 3000)


def test_views_are_hidden_in_a_tiny_window(window):
    assert show(window, 100, 100) == []
