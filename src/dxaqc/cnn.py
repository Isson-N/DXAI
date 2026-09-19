"""Инференс многоголовой CNN в сервисе.

Предобработка обязана повторять обучение (`training/train_baselines.py`): перцентильная
нормировка 0,5–99,5, ресайз по длинной стороне с центральным паддингом, три «физических»
канала (снимок, лапласиан, маска валидной области) и приведение левого бедра к правому.

Сторона при инференсе неизвестна (тег пуст), поэтому модель применяется дважды: первый проход
по неотзеркаленному снимку даёт область, и если это левое бедро — второй проход по зеркалу.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HEADS = ("region", "spine_any", "spine_pos", "spine_axis", "spine_foreign",
         "hip_any", "hip_pos", "hip_roi")


def prepare(image: np.ndarray, size: int, channels: str = "physical"):
    """Пиксели → тензор 1×3×size×size ровно так, как при обучении."""
    import torch
    import torch.nn.functional as F

    array = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(array, [0.5, 99.5])
    array = np.clip((array - low) / (high - low), 0, 1) if high > low else np.zeros_like(array)
    height, width = array.shape
    scale = size / max(height, width)
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    resized = F.interpolate(torch.from_numpy(np.ascontiguousarray(array))[None, None],
                            size=(new_height, new_width), mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.zeros((size, size), dtype=torch.float32)
    mask = torch.zeros((size, size), dtype=torch.float32)
    top, left = (size - new_height) // 2, (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    mask[top:top + new_height, left:left + new_width] = 1.0
    x = canvas[None]
    if channels == "physical":
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
        high_pass = F.conv2d(x[None], kernel, padding=1)[0]
        return torch.cat([(x - 0.485) / 0.229, (high_pass * 5).clamp(-3, 3), mask[None] - 0.5])[None]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((x.expand(3, -1, -1) - mean) / std)[None]


class QualityCNN:
    """Веса из `train_baselines.py --final-model`; наружу — словарь вероятностей по головам."""

    def __init__(self, weights: str | Path, device: str = "cpu"):
        import timm
        import torch

        from .nets import MultiHeadCNN

        payload = torch.load(str(weights), map_location=device, weights_only=False)
        self.size = int(payload.get("size", 384))
        self.channels = payload.get("channels", "physical")
        self.heads = tuple(payload.get("heads", HEADS))
        self.thresholds = dict(payload.get("thresholds", {}))
        self.device = device
        encoder = timm.create_model("resnet18", pretrained=False, num_classes=0, global_pool="avg")
        net = MultiHeadCNN(encoder, n_violations=len(self.heads) - 1)
        net.load_state_dict(payload["state_dict"])
        self.net = net.to(device).eval()

    def _forward(self, image: np.ndarray) -> dict[str, float]:
        import torch

        with torch.inference_mode():
            region, quality = self.net(prepare(image, self.size, self.channels).to(self.device))
        probabilities = {"region": float(region.float().softmax(1)[0, 1])}
        values = quality.float().sigmoid()[0].tolist()
        probabilities.update(dict(zip(self.heads[1:], values)))
        return probabilities

    def probabilities(self, image) -> dict[str, float]:
        from .nets import hip_side

        pixels = np.asarray(getattr(image, "pixels", image))
        result = self._forward(pixels)
        # Обучение видело все бёдра приведёнными к правому: левое надо отзеркалить и повторить.
        if result["region"] >= 0.5 and hip_side(pixels) == "L":
            mirrored = self._forward(pixels[:, ::-1].copy())
            mirrored["region"] = result["region"]
            return mirrored
        return result
