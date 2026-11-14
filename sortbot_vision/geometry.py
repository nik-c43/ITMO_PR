"""Геометрия камеры: из пикселей с глубиной в точки пространства."""
import numpy as np


def deproject(depth, intrinsics, mask):
    """Точки в оптическом кадре камеры для пикселей маски с известной глубиной.

    Модель камеры-обскуры: луч через пиксель (u, v) имеет направление
    ((u - cx) / fx, (v - cy) / fy, 1), а глубина говорит, как далеко по нему идти.

    Возвращает массив (N, 3).
    """
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - intrinsics.cx) / intrinsics.fx * z
    y = (v - intrinsics.cy) / intrinsics.fy * z
    return np.column_stack([x, y, z])


def transform_points(a_from_b, points_b):
    """Перевести точки (N, 3) из кадра b в кадр a матрицей 4x4."""
    return points_b @ a_from_b[:3, :3].T + a_from_b[:3, 3]


def project(points_camera, intrinsics):
    """Обратная операция: точки в кадре камеры -> пиксели (N, 2)."""
    points_camera = np.atleast_2d(points_camera)
    u = intrinsics.fx * points_camera[:, 0] / points_camera[:, 2] + intrinsics.cx
    v = intrinsics.fy * points_camera[:, 1] / points_camera[:, 2] + intrinsics.cy
    return np.column_stack([u, v])


def in_view(frame, point_world, margin=0):
    """Попадает ли точка мира в кадр, не ближе margin пикселей к его краю."""
    camera_from_world = np.linalg.inv(frame.world_from_camera)
    point = transform_points(camera_from_world, np.atleast_2d(point_world))[0]
    if point[2] <= 0:  # точка позади камеры
        return False
    u, v = project(point, frame.intrinsics)[0]
    return bool(margin <= u <= frame.intrinsics.width - 1 - margin
                and margin <= v <= frame.intrinsics.height - 1 - margin)
