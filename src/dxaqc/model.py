"""Интерфейс модели и временная заглушка (этап 1 плана)."""
from __future__ import annotations

from dataclasses import dataclass

from .io import DicomImage
from .labels import REGION_HIP, REGION_SPINE, VIOLATIONS


@dataclass
class Prediction:
    region: str
    quality_prob: float                  # вероятность «есть нарушение качества»
    violations: dict[str, float]         # вероятность по каждому нарушению области
    thresholds: dict[str, float]         # порог по каждому нарушению

    @property
    def violation_list(self) -> list[str]:
        order = VIOLATIONS[self.region]
        return [v for v in order if self.violations.get(v, 0.0) >= self.thresholds[v]]

    @property
    def quality_class(self) -> int:
        # План v2, этап 5: причину не придумываем — брак, если хотя бы одно нарушение прошло порог
        return int(bool(self.violation_list))


class StubModel:
    """Заглушка: область по размеру кадра, нарушений нет. Только для проверки конвейера."""

    version = "stub-0"

    def predict(self, image: DicomImage) -> Prediction:
        region = REGION_SPINE if image.columns == 300 else REGION_HIP
        names = VIOLATIONS[region]
        return Prediction(
            region=region,
            quality_prob=0.0,
            violations={v: 0.0 for v in names},
            thresholds={v: 0.5 for v in names},
        )
