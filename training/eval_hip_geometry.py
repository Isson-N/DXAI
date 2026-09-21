#!/usr/bin/env python3
"""Сравнить признак ротации бедра с текущей CNN на одних и тех же внешних фолдах.

Критерий остановки задан до прогона (astra, 22.09.2026): если геометрия по точкам,
предсказанным ВНЕ обучения, не превосходит текущую CNN-голову `hip_pos`, направление
закрывается. Сравниваются три источника:

  * `cnn`       — вероятность головы `hip_pos` из experiments/results/b2/oof.csv;
  * `oracle`    — f_simple по ручным точкам (потолок при идеальной разметке,
                  в сервисе недоступен, считается только на размеченных снимках);
  * `predicted` — f_simple по точкам сети (то, что реально будет в сервисе).

Метрики: AUC с кластерным бутстрэпом по исследованиям и F1 с порогом, подобранным
на внутренних фолдах, — порог НИКОГДА не выбирается на том же фолде, где оценивается.

    python training/eval_hip_geometry.py \
        --points experiments/results/hip_keypoints/oof_points.csv \
        --out experiments/results/hip_keypoints/geometry.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

POINTS = ["H", "B1", "B2", "T", "D", "D2"]
PIXEL = np.array([0.600, 0.606])          # мм на пиксель по x и y


def f_simple(row: pd.Series, suffix: str) -> float:
    """Отстояние малого вертела в медиальную сторону от оси диафиза, в ширинах шейки.

    Медиальное направление берётся из анатомии: головка бедра всегда медиальнее
    диафиза. Ни сторона из метаданных, ни подбор знака по результату не участвуют.
    """
    def point(name):
        x, y = row.get(f"{name}_x{suffix}"), row.get(f"{name}_y{suffix}")
        if pd.isna(x) or pd.isna(y):
            return None
        return np.array([float(x), float(y)]) * PIXEL

    head, neck1, neck2 = point("H"), point("B1"), point("B2")
    troch, shaft = point("T"), point("D")
    if head is None or shaft is None or troch is None:
        return float("nan")
    if neck1 is None or neck2 is None:
        return float("nan")
    width = float(np.linalg.norm(neck1 - neck2))
    if width < 1e-6:
        return float("nan")
    medial = 1.0 if head[0] > shaft[0] else -1.0
    return float((troch[0] - shaft[0]) * medial / width)


def auc(positive, negative) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not len(positive) or not len(negative):
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Средние ранги для совпадающих значений, иначе AUC смещается при связках.
    values = np.concatenate([positive, negative])
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    total = ranks[:len(positive)].sum()
    return float((total - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def cluster_ci(labels, scores, groups, repeats=2000, seed=11):
    """Бутстрэп по исследованиям: два бедра одного пациента не независимы."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"y": labels, "s": scores, "g": groups}).dropna()
    studies = frame.g.unique()
    values = []
    for _ in range(repeats):
        chosen = rng.choice(studies, size=len(studies), replace=True)
        sample = pd.concat([frame[frame.g == study] for study in chosen])
        if sample.y.nunique() < 2:
            continue
        value = auc(sample[sample.y == 1].s, sample[sample.y == 0].s)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def f1_nested(frame: pd.DataFrame, column: str) -> float:
    """F1 с порогом из ОСТАЛЬНЫХ фолдов: на фолде оценки порог не подбирается."""
    predictions = np.zeros(len(frame), dtype=int)
    for fold in sorted(frame.fold.unique()):
        inner = frame[frame.fold != fold].dropna(subset=[column])
        outer = frame.fold == fold
        if inner.empty:
            continue
        best_threshold, best_score = None, -1.0
        for candidate in np.quantile(inner[column], np.linspace(0.02, 0.98, 60)):
            guess = (inner[column] >= candidate).astype(int)
            tp = int(((guess == 1) & (inner.y == 1)).sum())
            fp = int(((guess == 1) & (inner.y == 0)).sum())
            fn = int(((guess == 0) & (inner.y == 1)).sum())
            score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if score > best_score:
                best_threshold, best_score = candidate, score
        predictions[outer.values] = (frame.loc[outer, column] >= best_threshold).fillna(False).astype(int)
    tp = int(((predictions == 1) & (frame.y == 1)).sum())
    fp = int(((predictions == 1) & (frame.y == 0)).sum())
    fn = int(((predictions == 0) & (frame.y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", default="experiments/results/hip_keypoints/oof_points.csv")
    parser.add_argument("--cnn", default="experiments/results/b2/oof.csv")
    parser.add_argument("--index", default="data/index/images.csv")
    parser.add_argument("--out", default="experiments/results/hip_keypoints/geometry.json")
    args = parser.parse_args()

    points = pd.read_csv(args.points)
    cnn = pd.read_csv(args.cnn)
    index = pd.read_csv(args.index).set_index("sop_uid")

    points["oracle"] = points.apply(lambda r: f_simple(r, "_true"), axis=1)
    points["predicted"] = points.apply(lambda r: f_simple(r, "_pred"), axis=1)

    merged = points.merge(
        cnn[["sop_uid", "hip_pos_prob", "hip_pos_true", "fold", "study"]],
        on="sop_uid", how="inner", suffixes=("", "_cnn"))
    merged = merged.rename(columns={"hip_pos_prob": "cnn", "hip_pos_true": "y"})
    # В общем файле предсказаний есть и снимки поясницы: у них метки бедра нет.
    merged = merged[merged["y"].notna()].copy()
    merged["y"] = merged["y"].astype(int)

    report = {"n": int(len(merged)), "n_positive": int(merged.y.sum()), "sources": {}}
    print(f"снимков в сравнении: {len(merged)}, положительных: {int(merged.y.sum())}")
    print(f"{'источник':12s} {'AUC':>6s} {'95% ДИ':>18s} {'F1':>6s}  покрытие")
    for column, label in [("cnn", "CNN (сейчас)"),
                          ("oracle", "ручные точки"),
                          ("predicted", "точки сети")]:
        if column not in merged:
            continue
        value = auc(merged[merged.y == 1][column], merged[merged.y == 0][column])
        low, high = cluster_ci(merged.y, merged[column], merged.study)
        score = f1_nested(merged.dropna(subset=[column]), column)
        covered = int(merged[column].notna().sum())
        report["sources"][column] = {"auc": round(value, 4), "ci95": [round(low, 4), round(high, 4)],
                                     "f1": round(score, 4), "covered": covered}
        print(f"{label:12s} {value:6.3f}  [{low:5.3f}, {high:5.3f}] {score:6.3f}  {covered}/{len(merged)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nсохранено: {args.out}")


if __name__ == "__main__":
    main()
