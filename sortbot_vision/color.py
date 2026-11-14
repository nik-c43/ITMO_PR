"""Выделение цветов: для каждого цвета маска пикселей, похожих на него.

Работаем в пространстве HSV: тон (H) почти не зависит от освещения, поэтому
тень на боковой грани кубика остаётся «красной», хотя и темнее.
"""
import cv2
import numpy as np

# Диапазоны тона в единицах OpenCV (0..179, то есть градусы пополам).
# Красный лежит на стыке шкалы, поэтому у него два диапазона.
DEFAULT_HUE_RANGES = {
    "red": [(0, 10), (170, 179)],
    "green": [(45, 85)],
    "blue": [(100, 130)],
}


def segment_colors(rgb, hue_ranges=DEFAULT_HUE_RANGES, min_saturation=120, min_value=40):
    """Вернуть словарь {цвет: маска uint8 (0 или 255)}.

    min_saturation отсекает бледное и серое, min_value отсекает слишком тёмное,
    где тон определяется ненадёжно.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    masks = {}
    for color, ranges in hue_ranges.items():
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for hue_low, hue_high in ranges:
            lower = np.array([hue_low, min_saturation, min_value], np.uint8)
            upper = np.array([hue_high, 255, 255], np.uint8)
            mask |= cv2.inRange(hsv, lower, upper)
        masks[color] = mask
    return masks
