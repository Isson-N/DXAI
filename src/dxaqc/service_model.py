"""Рабочая модель сервиса: геометрия поясницы + CNN по остальным нарушениям.

Устройство определяется тем, что показала валидация (`experiments/stage4_geometry_2026-09-20.md`):

* укладка поясницы — детерминированное правило по состояниям гребней подвздошных костей
  (F1 0,909 на внешних фолдах, столько же, сколько по ручной разметке, против 0,46 у CNN);
* ось позвоночника — угол хорды между крайними центрами тел: непрерывная оценка для ROC-AUC
  и порог 5° из ТЗ для решения (потолок F1 ≈ 0,48, метка не является функцией угла);
* посторонние предметы и обе метки бедра — головы CNN;
* область (поясница/бедро) — голова CNN, а не размер кадра: по размеру метка угадывается
  на обучающей выборке, но это свойство выгрузки, а не снимка.

Любая часть может отсутствовать: тогда её нарушения получают вероятность 0 и в отчёт не попадают,
а в `notes` пишется причина. Сервис не должен падать из-за отсутствия весов.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import AXIS_THRESHOLD_DEG, crest_rule
from .io import DicomImage
from .labels import (REGION_HIP, REGION_SPINE, V_AXIS, V_FOREIGN, V_POSITIONING, V_ROI,
                     VIOLATIONS)
from .model import Prediction

# Головы CNN в том же порядке, что в training/train_baselines.py
CNN_HEADS = ("region", "spine_any", "spine_pos", "spine_axis", "spine_foreign",
             "hip_any", "hip_pos", "hip_roi")
HEAD_BY_VIOLATION = {
    (REGION_SPINE, V_POSITIONING): "spine_pos",
    (REGION_SPINE, V_AXIS): "spine_axis",
    (REGION_SPINE, V_FOREIGN): "spine_foreign",
    (REGION_HIP, V_POSITIONING): "hip_pos",
    (REGION_HIP, V_ROI): "hip_roi",
}


def angle_to_probability(angle_deg: float, threshold: float = AXIS_THRESHOLD_DEG,
                         width: float = 1.5) -> float:
    """Гладкая оценка вероятности по углу: 0,5 ровно на пороге ТЗ.

    Нужна не как калибровка (её не на чем строить: 10 положительных), а чтобы у метки «ось»
    была непрерывная оценка для ROC-AUC организатора, согласованная с бинарным решением.
    """
    if angle_deg is None or not math.isfinite(angle_deg):
        return 0.0
    return float(1.0 / (1.0 + math.exp(-(abs(angle_deg) - threshold) / width)))


@dataclass
class ServiceModel:
    """Собирает прогноз по одному изображению из доступных частей."""

    keypoints: object | None = None          # dxaqc.keypoints.SpineKeypointModel
    cnn: object | None = None                # обёртка над B2 (интерфейс: probabilities(image) -> dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    version: str = "geometry+cnn-1"
    notes: list[str] = field(default_factory=list)

    def region_of(self, image: DicomImage, cnn_probs: dict[str, float] | None) -> str:
        if cnn_probs is not None and "region" in cnn_probs:
            return REGION_HIP if cnn_probs["region"] >= 0.5 else REGION_SPINE
        # Запасной путь: размер кадра. Отмечаем в примечаниях — это подгонка под выгрузку.
        self.notes.append("область определена по размеру кадра: веса CNN недоступны")
        return REGION_SPINE if image.columns == 300 else REGION_HIP

    def predict(self, image: DicomImage) -> Prediction:
        cnn_probs = self.cnn.probabilities(image) if self.cnn is not None else None
        region = self.region_of(image, cnn_probs)
        names = VIOLATIONS[region]
        probabilities = {name: 0.0 for name in names}
        thresholds = {name: float(self.thresholds.get(HEAD_BY_VIOLATION[(region, name)], 0.5))
                      for name in names}

        if region == REGION_SPINE and self.keypoints is not None:
            spine = self.keypoints.predict(np.asarray(image.pixels))
            # Укладка: решение детерминированное, поэтому вероятность выставляется в 0/1.
            probabilities[V_POSITIONING] = 1.0 if spine.crest_out_of_frame else 0.0
            thresholds[V_POSITIONING] = 0.5
            probabilities[V_AXIS] = angle_to_probability(spine.geometry.angle_chord)
            thresholds[V_AXIS] = 0.5
        elif region == REGION_SPINE:
            self.notes.append("укладка и ось поясницы не оценены: нет модели ключевых точек")

        if cnn_probs is not None:
            for name in names:
                head = HEAD_BY_VIOLATION[(region, name)]
                # Геометрия по пояснице уже дала решение — CNN её не перебивает.
                if region == REGION_SPINE and name in (V_POSITIONING, V_AXIS) and self.keypoints:
                    continue
                if head in cnn_probs:
                    probabilities[name] = float(cnn_probs[head])

        any_head = "spine_any" if region == REGION_SPINE else "hip_any"
        if cnn_probs is not None and any_head in cnn_probs:
            quality_prob = float(cnn_probs[any_head])
        else:
            quality_prob = 0.0
        # Вероятность «есть нарушение» не может быть меньше уверенности в конкретном нарушении.
        quality_prob = max([quality_prob] + list(probabilities.values()))

        return Prediction(region=region, quality_prob=quality_prob,
                          violations=probabilities, thresholds=thresholds)


def load(models_dir: str | Path, device: str = "cpu") -> ServiceModel:
    """Собирает модель из того, что лежит в каталоге весов; чего нет — то пропускается."""
    directory = Path(models_dir)
    model = ServiceModel()
    keypoints_path = directory / "spine_keypoints.pt"
    if keypoints_path.exists():
        from .keypoints import SpineKeypointModel
        model.keypoints = SpineKeypointModel(keypoints_path, device)
    else:
        model.notes.append(f"нет файла {keypoints_path.name}")
    cnn_path = directory / "quality_cnn.pt"
    if cnn_path.exists():
        from .cnn import QualityCNN
        model.cnn = QualityCNN(cnn_path, device)
        model.thresholds = dict(model.cnn.thresholds)
    else:
        model.notes.append(f"нет файла {cnn_path.name}")
    return model
