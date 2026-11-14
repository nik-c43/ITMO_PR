"""Память о кубиках: что робот видел, где оно лежит и что уже отвезено.

Камера видит кубики только в момент кадра, а возить их приходится по одному.
Память сводит находки с разных кадров в список кубиков и ведёт состояние каждого.
"""
from dataclasses import dataclass, field

import numpy as np

WAITING = "waiting"  # лежит и ждёт, пока его заберут
CARRIED = "carried"  # в захвате робота
DELIVERED = "delivered"  # отвезён на свой стол


@dataclass
class TrackedCube:
    id: int
    color: str
    position: np.ndarray  # (3,) центр кубика в мире по последнему кадру, м
    yaw: float
    stamp: float  # время последнего кадра, где он был виден, с
    sightings: int = 1  # сколько раз его видели
    state: str = WAITING
    table: str | None = None  # на какой стол отвезён


@dataclass
class CubeMemory:
    match_radius: float = 0.03  # находка ближе этого к известному кубику того же цвета считается им же, м
    cubes: list = field(default_factory=list)
    _next_id: int = 1

    def observe(self, detections, stamp=0.0, in_view=None):
        """Учесть находки одного кадра.

        Находка рядом с известным кубиком того же цвета обновляет его, остальные добавляются как новые.
        in_view(position) говорит, попадает ли точка в кадр. Если ожидающий кубик должен
        быть виден, но его нет среди находок, он забывается: его сдвинули, забрали или он
        был ошибкой зрения. Если кубик на месте, следующий кадр найдёт его заново.
        """
        seen = set()
        for det in detections:
            cube = self._match(det.color, det.position, exclude=seen)
            if cube is None:
                cube = TrackedCube(self._next_id, det.color, np.array(det.position), det.yaw, stamp, sightings=0)
                self._next_id += 1
                self.cubes.append(cube)
            cube.position, cube.yaw, cube.stamp = np.array(det.position), det.yaw, stamp
            cube.sightings += 1
            seen.add(cube.id)
        if in_view is not None:
            self.cubes = [c for c in self.cubes
                          if c.state != WAITING or c.id in seen or not in_view(c.position)]
        return [c for c in self.cubes if c.id in seen]

    def _match(self, color, position, exclude=()):
        candidates = [c for c in self.cubes if c.state == WAITING and c.color == color and c.id not in exclude
                      and np.linalg.norm(c.position - position) < self.match_radius]
        return min(candidates, key=lambda c: np.linalg.norm(c.position - position), default=None)

    def waiting(self):
        return [c for c in self.cubes if c.state == WAITING]

    def delivered(self, table=None):
        return [c for c in self.cubes if c.state == DELIVERED and table in (None, c.table)]

    def forget(self, cube):
        self.cubes = [c for c in self.cubes if c.id != cube.id]

    def mark_carried(self, cube):
        cube.state = CARRIED

    def mark_delivered(self, cube, position, table):
        cube.state, cube.position, cube.table = DELIVERED, np.array(position), table
