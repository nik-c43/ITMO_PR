"""Структуры данных зрения. Поля подобраны так, чтобы напрямую соответствовать сообщениям ROS 2."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Параметры объектива (в ROS 2: sensor_msgs/CameraInfo, матрица K)."""
    width: int
    height: int
    fx: float  # фокусное расстояние в пикселях
    fy: float
    cx: float  # пиксель, в который смотрит оптическая ось
    cy: float

    @classmethod
    def from_fov(cls, width, height, fovy_deg):
        """Идеальная камера с вертикальным углом обзора fovy_deg и квадратными пикселями."""
        f = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
        return cls(width, height, f, f, 0.5 * (width - 1), 0.5 * (height - 1))


@dataclass
class RgbdFrame:
    """Один кадр: цвет, совмещённая с ним глубина и то, откуда снято."""
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, метры вдоль оптической оси; 0 или nan: нет данных
    intrinsics: CameraIntrinsics
    world_from_camera: np.ndarray  # (4, 4) поза оптического кадра камеры в мире
    stamp: float = 0.0  # время кадра, с


@dataclass
class CubeDetection:
    """Найденный кубик (в ROS 2: vision_msgs/Detection3D)."""
    color: str
    position: np.ndarray  # (3,) центр кубика в мире, м
    yaw: float  # поворот вокруг вертикали, рад, от -pi/4 до pi/4
    size: tuple  # (длинная, короткая) стороны верхней грани, м
    pixel: tuple  # (u, v) центр пятна в кадре
    num_points: int  # сколько точек глубины попало на верхнюю грань
