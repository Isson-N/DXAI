"""Инференс модели ключевых точек поясницы в сервисе.

Веса готовит `training/train_keypoints.py --final-model`. Предобработка обязана совпадать
с обучением (перцентильная нормировка, ресайз по длинной стороне, центральный паддинг),
иначе точки поедут. Torch импортируется лениво: остальной пакет работает и без него.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import PIXEL_MM, SpineGeometry, crest_rule, spine_geometry

VERTEBRA_POINTS = ("Th12", "L1", "L2", "L3", "L4", "L5")


@dataclass(frozen=True)
class SpinePrediction:
    points: dict[str, tuple[float, float] | None]
    scores: dict[str, float]
    states: dict[str, str]
    geometry: SpineGeometry

    @property
    def crest_out_of_frame(self) -> bool:
        return crest_rule(self.states.get("crest_left", ""), self.states.get("crest_right", ""))


def fit_canvas(image: np.ndarray, size: int):
    """Вписывает изображение в квадрат size×size; возвращает холст и обратное преобразование."""
    import torch
    import torch.nn.functional as F

    height, width = image.shape
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.interpolate(torch.from_numpy(np.ascontiguousarray(image))[None, None],
                            size=(new_height, new_width), mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.zeros((size, size), dtype=torch.float32)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas, scale, float(left), float(top)


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [0.5, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


class SpineKeypointModel:
    """Обёртка над обученной U-Net: пиксели → точки, состояния гребней и геометрия оси."""

    def __init__(self, weights: str | Path, device: str = "cpu"):
        import torch

        payload = torch.load(str(weights), map_location=device, weights_only=False)
        self.size = int(payload["size"])
        self.names = list(payload["points"])
        self.states = payload["states"]
        self.threshold = float(payload.get("point_threshold", 0.2))
        self.device = device
        self.net = self._build(payload["state_dict"], device)

    def _build(self, state_dict, device):
        from .nets import KeypointNet

        net = KeypointNet(pretrained=False)
        net.load_state_dict(state_dict)
        return net.to(device).eval()

    def predict(self, image: np.ndarray, pixel_mm=PIXEL_MM) -> SpinePrediction:
        import torch

        canvas, scale, left, top = fit_canvas(normalize(image), self.size)
        with torch.inference_mode():
            heatmaps, logits = self.net(canvas[None, None].expand(1, 3, -1, -1).to(self.device))
        probability = torch.sigmoid(heatmaps)[0].cpu().numpy()
        points: dict[str, tuple[float, float] | None] = {}
        scores: dict[str, float] = {}
        for index, name in enumerate(self.names):
            plane = probability[index]
            score = float(plane.max())
            scores[name] = score
            if score < self.threshold:
                points[name] = None
                continue
            y, x = np.unravel_index(int(plane.argmax()), plane.shape)
            y0, y1 = max(0, y - 2), min(plane.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(plane.shape[1], x + 3)
            patch = plane[y0:y1, x0:x1]
            weight = patch.sum()
            gy, gx = np.mgrid[y0:y1, x0:x1]
            cy = (gy * patch).sum() / weight if weight else y
            cx = (gx * patch).sum() / weight if weight else x
            ratio = self.size / plane.shape[0]
            points[name] = ((cx * ratio - left) / scale, (cy * ratio - top) / scale)
        classes = logits[0].argmax(-1).cpu().numpy()
        names = ["crest_left", "crest_right", "th12_half_visible"]
        states = {key: self.states[key][int(classes[i])] for i, key in enumerate(names)}
        centers = [points.get(name) for name in VERTEBRA_POINTS]
        geometry = spine_geometry([p for p in centers if p is not None], pixel_mm)
        return SpinePrediction(points, scores, states, geometry)
