"""Что робот знает о столах заранее."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Table:
    """Стол со сторонами вдоль осей мира."""
    name: str
    center: np.ndarray  # (2,) центр столешницы в мире, м
    half_size: np.ndarray  # (2,) половины сторон столешницы вдоль X и Y мира, м
    surface_z: float  # высота столешницы, м
    heading: float  # куда смотрит робот, стоя у стола, рад
    color: str | None = None  # кубики какого цвета сюда везти; None: стол, с которого забираем

    @property
    def direction(self):
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def depth(self):
        """Половина размера стола вдоль направления подъезда, м."""
        return float(np.abs(self.direction) @ self.half_size)

    @property
    def near_edge(self):
        """Середина ближнего к роботу края столешницы, (x, y)."""
        return self.center - self.depth * self.direction

    def contains(self, point, margin=0.0):
        """Лежит ли точка над столешницей (с запасом margin от края)."""
        return bool(np.all(np.abs(np.asarray(point)[:2] - self.center) <= self.half_size - margin))

    def holds_cube(self, position, cube_size=0.04, tolerance=0.005):
        """Стоит ли на столешнице кубик с центром в position."""
        return self.contains(position) and abs(position[2] - self.surface_z - 0.5 * cube_size) < tolerance

    def place_slots(self, edge_offset=0.15, spacing=0.12, count=3):
        """Места для кубиков: ряд вдоль ближнего к роботу края, от середины в стороны."""
        side = np.array([-self.direction[1], self.direction[0]])
        row = self.near_edge + edge_offset * self.direction
        offsets = sorted(spacing * (np.arange(count) - (count - 1) / 2), key=abs)
        return [row + offset * side for offset in offsets]
