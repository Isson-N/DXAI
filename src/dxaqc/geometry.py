"""Геометрия поясничного отдела: из точек — в признаки и решения по критериям ТЗ.

Те же формулы, что в `training/eval_geometry.py`, но здесь они работают на инференсе:
точки приходят от модели ключевых точек, а не от человека. Координаты — в пикселях
изображения, масштаб задаётся отдельно (у выгрузки 0,600 × 0,606 мм, решение 17.09.2026).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PIXEL_MM = (0.600, 0.606)
VERTEBRAE = ("Th12", "L1", "L2", "L3", "L4", "L5")
AXIS_THRESHOLD_DEG = 5.0


@dataclass(frozen=True)
class SpineGeometry:
    """Признаки оси поясницы. Углы — к вертикали кадра (оси скана), градусы; длины — мм."""

    angle_chord: float
    angle_ls: float
    angle_robust: float
    max_dev_chord: float
    curvature: float
    span_mm: float
    n_points: int

    def as_dict(self) -> dict:
        return {
            "angle_chord": self.angle_chord, "angle_ls": self.angle_ls,
            "angle_robust": self.angle_robust, "max_dev_chord": self.max_dev_chord,
            "curvature": self.curvature, "span_mm": self.span_mm, "n_points": self.n_points,
        }


def to_millimetres(points, pixel_mm=PIXEL_MM):
    """Точки (x, y) в пикселях → массив в мм, упорядоченный сверху вниз."""
    valid = [(float(x), float(y)) for x, y in points
             if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    array = np.array(valid, dtype=float) * np.array(pixel_mm, dtype=float)
    return array[np.argsort(array[:, 1])] if len(array) else array.reshape(0, 2)


def _angle_to_vertical(dx: float, dy: float) -> float:
    """Угол направления к вертикали: 0° — строго вдоль оси скана."""
    return math.degrees(math.atan2(abs(dx), abs(dy))) if dy or dx else float("nan")


def _slope_angle(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Наклон прямой x = a·y + b (МНК, при weights — взвешенный) в градусах к вертикали."""
    w = np.ones(len(y)) if weights is None else weights
    if w.sum() <= 0 or len(y) < 2:
        return float("nan")
    my, mx = np.average(y, weights=w), np.average(x, weights=w)
    denominator = float((w * (y - my) ** 2).sum())
    if denominator < 1e-9:
        return float("nan")
    a = float((w * (y - my) * (x - mx)).sum() / denominator)
    return math.degrees(math.atan(abs(a)))


def spine_geometry(points, pixel_mm=PIXEL_MM) -> SpineGeometry:
    """Признаки по центрам тел позвонков. Меньше двух точек — всё NaN, но объект возвращается."""
    pts = to_millimetres(points, pixel_mm)
    nan = float("nan")
    if len(pts) < 2:
        return SpineGeometry(nan, nan, nan, nan, nan, nan, len(pts))
    x, y = pts[:, 0], pts[:, 1]
    first, last = pts[0], pts[-1]
    chord = last - first
    angle_chord = _angle_to_vertical(chord[0], chord[1])
    angle_ls = _slope_angle(x, y)

    # Робастный наклон: при сколиозе центры ложатся на дугу, и МНК даёт ложный наклон.
    weights = np.ones(len(y))
    angle_robust = angle_ls
    for _ in range(5):
        my, mx = np.average(y, weights=weights), np.average(x, weights=weights)
        denominator = float((weights * (y - my) ** 2).sum())
        if denominator < 1e-9:
            break
        a = float((weights * (y - my) * (x - mx)).sum() / denominator)
        residual = x - (a * (y - my) + mx)
        scale = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826 + 1e-6
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-8))
        angle_robust = math.degrees(math.atan(abs(a)))

    length = float(np.linalg.norm(chord))
    deviation = nan
    if length > 1e-8:
        deviation = float(max(abs(chord[0] * (p - first)[1] - chord[1] * (p - first)[0]) / length
                              for p in pts))
    curvature = nan
    if len(pts) >= 3:
        middle = len(pts) // 2
        upper, lower = pts[middle] - pts[0], pts[-1] - pts[middle]
        if np.linalg.norm(upper) > 1e-8 and np.linalg.norm(lower) > 1e-8:
            cosine = float(np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower)))
            curvature = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
    return SpineGeometry(angle_chord, angle_ls, angle_robust, deviation, curvature, length, len(pts))


def crest_rule(crest_left: str, crest_right: str) -> bool:
    """Укладка поясницы: хотя бы один гребень подвздошной кости не попал в кадр.

    На ручной разметке правило дало TP 5/6 при FP 0 — детерминированная ветвь сервиса.
    Состояние «частично» нарушением не считается: оно расходится между повторами разметки.
    """
    return "out_of_frame" in (str(crest_left), str(crest_right))
