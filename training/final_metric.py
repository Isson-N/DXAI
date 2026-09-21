#!/usr/bin/env python3
"""Итоговая метрика сервиса: macro-F1 по пяти нарушениям с маршрутизацией из протокола.

Метрики отдельных голов уже считались, но числа сервиса не было: «обрезка поясницы»
и «ось» берутся не из CNN, а из геометрии по предсказанным точкам (docs/final_protocol.md,
раздел 1). Здесь собирается ровно тот прогноз, который выдаёт `service_model`,
и считается macro-F1 с кластерным бутстрэпом по исследованиям.

Маршрутизация (зафиксирована 20.09.2026, при расчёте не меняется):

  обрезка поясницы  → правило гребней: хотя бы один `out_of_frame`
  ось позвоночника  → |угол хорды Th12→L5| > 5°, порог из ТЗ
  предметы, ротация, область интереса → соответствующие головы CNN

    python training/final_metric.py --cnn experiments/results_final/b2/oof.csv \
        --out experiments/results_final/service_metric.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AXIS_THRESHOLD_DEG = 5.0

# Нарушение → (колонка истины в oof.csv, источник прогноза)
VIOLATIONS = {
    "обрезка поясницы": ("spine_pos_true", "geometry_crest"),
    "ось позвоночника": ("spine_axis_true", "geometry_angle"),
    "предметы": ("spine_foreign_true", "spine_foreign_pred"),
    "ротация бедра": ("hip_pos_true", "hip_pos_pred"),
    "область интереса бедра": ("hip_roi_true", "hip_roi_pred"),
}


def f1_of(truth: np.ndarray, guess: np.ndarray) -> float:
    tp = int(((guess == 1) & (truth == 1)).sum())
    fp = int(((guess == 1) & (truth == 0)).sum())
    fn = int(((guess == 0) & (truth == 1)).sum())
    if 2 * tp + fp + fn == 0:
        return 0.0                      # протокол: неопределённый F1 считается нулём
    return 2 * tp / (2 * tp + fp + fn)


def build(cnn: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """Склейка предсказаний сервиса: геометрия для двух меток, CNN для остальных."""
    frame = cnn.merge(geometry, on="sop_uid", how="left", suffixes=("", "_geo"))
    crest = frame.get("crest_left_state_pred"), frame.get("crest_right_state_pred")
    frame["geometry_crest"] = (
        (crest[0].astype(str) == "out_of_frame") | (crest[1].astype(str) == "out_of_frame")
    ).astype(int) if crest[0] is not None else 0
    angle = frame.get("angle_chord_pred")
    frame["geometry_angle"] = (angle.abs() > AXIS_THRESHOLD_DEG).fillna(False).astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", default="experiments/results_final/b2/oof.csv")
    parser.add_argument("--geometry", default="experiments/results/keypoints/oof_features.csv")
    parser.add_argument("--out", default="experiments/results_final/service_metric.json")
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()

    cnn = pd.read_csv(args.cnn)
    geometry = pd.read_csv(args.geometry)
    frame = build(cnn, geometry)

    report, per_violation = {"violations": {}}, {}
    print(f"{'нарушение':24s} {'n':>4s} {'pos':>4s} {'F1':>6s}   источник")
    for name, (truth_column, source) in VIOLATIONS.items():
        if truth_column not in frame:
            continue
        subset = frame[frame[truth_column].notna()]
        truth = subset[truth_column].to_numpy(dtype=int)
        guess = subset[source].fillna(0).to_numpy(dtype=int)
        score = f1_of(truth, guess)
        per_violation[name] = (subset.study.to_numpy(), truth, guess)
        report["violations"][name] = {
            "n": int(len(subset)), "n_positive": int(truth.sum()),
            "f1": round(score, 4), "source": source,
            "tp": int(((guess == 1) & (truth == 1)).sum()),
            "fp": int(((guess == 1) & (truth == 0)).sum()),
            "fn": int(((guess == 0) & (truth == 1)).sum()),
        }
        print(f"{name:24s} {len(subset):4d} {int(truth.sum()):4d} {score:6.3f}   {source}")

    macro = float(np.mean([v["f1"] for v in report["violations"].values()]))
    report["macro_f1"] = round(macro, 4)

    # Бутстрэп по исследованиям: две проекции одного пациента не независимы.
    rng = np.random.default_rng(11)
    studies = np.unique(np.concatenate([g for g, _, _ in per_violation.values()]))
    draws = []
    for _ in range(args.repeats):
        chosen = rng.choice(studies, size=len(studies), replace=True)
        scores = []
        for groups, truth, guess in per_violation.values():
            index = np.concatenate([np.flatnonzero(groups == s) for s in chosen])
            scores.append(f1_of(truth[index], guess[index]))
        draws.append(float(np.mean(scores)))
    draws.sort()
    low, high = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
    report["macro_f1_ci95"] = [round(low, 4), round(high, 4)]

    print(f"\nmacro-F1 сервиса: {macro:.4f}  95% ДИ [{low:.3f}, {high:.3f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
