"""Проверки зрения. Запуск: pytest"""
import dataclasses
from pathlib import Path

# sortbot_sim импортируется раньше mujoco: он включает рендер без окна.
from sortbot_sim import SortingSim
from sortbot_sim.camera import D435, IDEAL, SimCamera
from sortbot_sim.world import cube_positions, tables_from_sim
from sortbot_motion import Skills
from sortbot_vision.cubes import CubeDetector
from sortbot_vision.geometry import deproject, project, transform_points
from sortbot_vision.types import CameraIntrinsics

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SURFACE_Z = 0.40


def look_at_table(sensor, frames=1):
    """Робот у стола с кубиками, камера на руке смотрит на стол. Возвращает (кадры, истинные кубики)."""
    sim = SortingSim()
    table = next(t for t in tables_from_sim(sim) if t.color is None)
    Skills(sim).look_over_table(table.near_edge, table.heading, table.surface_z)

    camera = SimCamera(sim, sensor=sensor)
    captured = [camera.capture() for _ in range(frames)]
    camera.close()
    # Цвет кубика стоит в его имени: cube_<цвет>_<номер>.
    truth = [(name.split("_")[1], position, sim.body_yaw(name)) for name, position in cube_positions(sim).items()]
    return captured, truth


@pytest.fixture(scope="module")
def scene():
    """Кадр идеальной камеры: проверяет сам алгоритм, без помех сенсора."""
    frames, truth = look_at_table(IDEAL)
    return frames[0], truth


@pytest.fixture(scope="module")
def d435_scene():
    """Десять кадров камеры с моделью RealSense D435: шум, пропуски на краях, предел дальности."""
    return look_at_table(D435, frames=10)


def match_errors(detections, truth):
    """Для каждой находки: (ошибка положения в м, ошибка угла в градусах) до ближайшего кубика того же цвета."""
    errors = []
    for det in detections:
        _, pos, yaw = min((c for c in truth if c[0] == det.color), key=lambda c: np.linalg.norm(c[1] - det.position))
        yaw_error = (np.degrees(det.yaw - yaw) + 45) % 90 - 45
        errors.append((np.linalg.norm(pos - det.position), abs(yaw_error)))
    return np.array(errors)


def test_deproject_and_project_are_inverse():
    intrinsics = CameraIntrinsics.from_fov(640, 480, 42.0)
    depth = np.full((480, 640), 0.7, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[100, 200] = mask[400, 50] = True
    points = deproject(depth, intrinsics, mask)
    np.testing.assert_allclose(points[:, 2], 0.7, rtol=1e-6)
    np.testing.assert_allclose(project(points, intrinsics), [[200, 100], [50, 400]], atol=1e-6)


def test_transform_points_applies_rotation_and_shift():
    a_from_b = np.eye(4)
    a_from_b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # поворот на 90 градусов вокруг Z
    a_from_b[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(transform_points(a_from_b, np.array([[1.0, 0.0, 0.0]])), [[1, 3, 3]])


def test_finds_all_cubes(scene):
    frame, truth = scene
    detections = CubeDetector().detect(frame, surface_z=SURFACE_Z)
    assert sorted(d.color for d in detections) == sorted(c[0] for c in truth)
    errors = match_errors(detections, truth)
    assert errors[:, 0].max() < 0.003
    assert errors[:, 1].max() < 1.0


def test_works_without_known_surface_height(scene):
    frame, truth = scene
    detections = CubeDetector().detect(frame)
    assert sorted(d.color for d in detections) == sorted(c[0] for c in truth)


def test_survives_sensor_noise(scene):
    frame, truth = scene
    rng = np.random.default_rng(0)
    for _ in range(10):
        noisy = dataclasses.replace(
            frame,
            depth=(frame.depth + rng.normal(0, 0.005, frame.depth.shape)).astype(np.float32),
            rgb=np.clip(frame.rgb + rng.normal(0, 8, frame.rgb.shape), 0, 255).astype(np.uint8),
        )
        detections = CubeDetector().detect(noisy, surface_z=SURFACE_Z)
        assert len(detections) == len(truth)
        assert match_errors(detections, truth)[:, 0].max() < 0.008


def test_finds_all_cubes_with_d435(d435_scene):
    frames, truth = d435_scene
    for frame in frames:
        detections = CubeDetector().detect(frame, surface_z=SURFACE_Z)
        assert sorted(d.color for d in detections) == sorted(c[0] for c in truth)
        assert match_errors(detections, truth)[:, 0].max() < 0.008


def test_d435_has_no_depth_outside_working_range(d435_scene):
    frames, _ = d435_scene
    depth = frames[0].depth
    known = depth[depth > 0]
    assert known.min() >= D435.min_depth * 0.95 and known.max() <= D435.max_depth * 1.05
    assert 0.5 < (depth > 0).mean() < 1.0  # часть кадра без глубины: края предметов и дальний пол
    assert frames[0].rgb.shape == (D435.height, D435.width, 3)


def test_core_does_not_depend_on_simulator_or_ros():
    for package in ("sortbot_vision", "sortbot_motion", "sortbot_task"):
        for path in (REPO / package).glob("*.py"):
            source = path.read_text()
            assert "mujoco" not in source and "rclpy" not in source, path.name
