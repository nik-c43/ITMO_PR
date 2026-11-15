"""Поиск кубиков в кадре: цвет, положение в мире и поворот."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from sortbot_vision.color import DEFAULT_HUE_RANGES, segment_colors
from sortbot_vision.geometry import deproject, project, transform_points
from sortbot_vision.types import CubeDetection

DRAW_COLORS = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
TRIM_PERCENT = 3.0  # сколько процентов точек отбрасывается с каждого края при измерении грани


@dataclass
class CubeDetectorConfig:
    cube_size: float = 0.04  # ребро кубика, м
    size_tolerance: float = 0.012  # насколько измеренная сторона может отличаться от ребра, м
    top_face_thickness: float = 0.008  # точки не ниже этого от верха пятна считаются верхней гранью, м
    min_pixels: int = 30  # пятна меньше этого считаются шумом
    hue_ranges: dict = field(default_factory=lambda: dict(DEFAULT_HUE_RANGES))
    min_saturation: int = 120
    min_value: int = 40


class CubeDetector:
    def __init__(self, config=None):
        self.config = config or CubeDetectorConfig()

    def detect(self, frame, surface_z=None):
        """Найти кубики в кадре.

        surface_z: высота поверхности, на которой лежат кубики (столешницы), в мире.
        Если она известна, всё, что ниже или заметно выше кубика на этой поверхности,
        игнорируется: пол, сам стол, рука робота.
        """
        cfg = self.config
        masks = segment_colors(frame.rgb, cfg.hue_ranges, cfg.min_saturation, cfg.min_value)
        zone = self._working_zone(frame, surface_z)

        detections = []
        kernel = np.ones((3, 3), np.uint8)
        for color, mask in masks.items():
            if zone is not None:
                mask = mask & zone
            # Убираем одиночные пиксели шума и размытые границы пятен.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for label in range(1, count):  # метка 0 это фон
                if stats[label, cv2.CC_STAT_AREA] < cfg.min_pixels:
                    continue
                detection = self._measure(frame, labels == label, color)
                if detection is not None:
                    detections.append(detection)
        return detections

    def _working_zone(self, frame, surface_z):
        """Маска пикселей, чьи точки лежат в слое, где могут быть кубики."""
        if surface_z is None:
            return None
        cfg = self.config
        everything = np.ones(frame.depth.shape, bool)
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        points = transform_points(frame.world_from_camera, deproject(frame.depth, frame.intrinsics, everything))
        height = np.full(frame.depth.shape, -np.inf)
        height[valid] = points[:, 2] - surface_z
        zone = (height > 0.005) & (height < cfg.cube_size + cfg.size_tolerance)
        return zone.astype(np.uint8) * 255

    def _measure(self, frame, blob, color):
        """По пятну одного цвета измерить кубик. Вернуть None, если это не кубик."""
        cfg = self.config
        points = transform_points(frame.world_from_camera, deproject(frame.depth, frame.intrinsics, blob))
        if len(points) < cfg.min_pixels:
            return None

        # Верхняя грань: самые высокие точки пятна. 95-й процентиль вместо максимума,
        # чтобы одна шумная точка не сдвинула оценку.
        top_z = np.percentile(points[:, 2], 95)
        top = points[points[:, 2] > top_z - cfg.top_face_thickness]
        if len(top) < cfg.min_pixels:
            return None

        # Поворот: угол прямоугольника минимальной площади вокруг верхней грани в плоскости XY.
        # У куба симметрия 90 градусов, поэтому угол приводим к диапазону [-45, 45).
        xy = top[:, :2]
        _, _, angle_deg = cv2.minAreaRect(xy.astype(np.float32))
        yaw = np.radians((angle_deg + 45.0) % 90.0 - 45.0)

        # Размер и центр меряем в осях кубика по процентилям, а не по крайним точкам:
        # на шумной глубине несколько точек-выбросов иначе раздувают грань.
        c, s = np.cos(yaw), np.sin(yaw)
        cube_from_world = np.array([[c, s], [-s, c]])
        low, high = np.percentile(xy @ cube_from_world.T, [TRIM_PERCENT, 100 - TRIM_PERCENT], axis=0)
        sides = (high - low) / (1 - 2 * TRIM_PERCENT / 100)  # поправка на отброшенные края
        long_side, short_side = sides.max(), sides.min()
        if abs(long_side - cfg.cube_size) > cfg.size_tolerance or abs(short_side - cfg.cube_size) > cfg.size_tolerance:
            return None

        cx, cy = 0.5 * (low + high) @ cube_from_world
        center = np.array([cx, cy, np.median(top[:, 2]) - 0.5 * cfg.cube_size])
        return CubeDetection(color, center, float(yaw), (float(long_side), float(short_side)))


def draw_detections(frame, detections, cube_size=0.04):
    """Копия кадра с обведёнными верхними гранями найденных кубиков (для отладки)."""
    image = frame.rgb.copy()
    camera_from_world = np.linalg.inv(frame.world_from_camera)
    half = 0.5 * cube_size
    for det in detections:
        c, s = np.cos(det.yaw), np.sin(det.yaw)
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        corners = corners @ np.array([[c, s], [-s, c]]) + det.position[:2]
        corners = np.column_stack([corners, np.full(4, det.position[2] + half)])
        pixels = project(transform_points(camera_from_world, corners), frame.intrinsics)
        pixels = np.round(pixels).astype(np.int32)
        cv2.polylines(image, [pixels], True, (255, 255, 255), 2)
        cv2.polylines(image, [pixels], True, DRAW_COLORS.get(det.color, (255, 255, 0)), 1)
        label = f"{det.color} {np.degrees(det.yaw):+.0f}"
        origin = (int(pixels[:, 0].min()), int(pixels[:, 1].min()) - 6)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image
