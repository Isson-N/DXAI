#!/usr/bin/env python3
"""Сравнить патч-классификатор предметов со старой CNN-головой на одних и тех же снимках.

Протокол: `docs/foreign_patch_protocol.md`. Три уровня утверждения различаются заранее:

  точечный результат        ΔF1 >= 0,10
  свидетельство улучшения   95 % интервал разности не включает ноль
  подтверждение пользы      нижняя граница интервала >= 0,10

Обе головы оцениваются на общих снимках; разность считается парным кластерным
бутстрэпом по исследованиям — одни и те же реплики для обеих моделей, иначе разница
утонет в шуме при 17 положительных.

    python training/compare_foreign.py \
        --new experiments/results/foreign_patch/oof.csv \
        --old experiments/results_final/b2/oof.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def f1_of(truth: np.ndarray, guess: np.ndarray) -> float:
    tp = int(((guess == 1) & (truth == 1)).sum())
    fp = int(((guess == 1) & (truth == 0)).sum())
    fn = int(((guess == 0) & (truth == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", default="experiments/results/foreign_patch/oof.csv")
    parser.add_argument("--old", default="experiments/results_final/b2/oof.csv")
    parser.add_argument("--out", default="experiments/results/foreign_patch/comparison.json")
    parser.add_argument("--repeats", type=int, default=3000)
    args = parser.parse_args()

    new = pd.read_csv(args.new)
    old = pd.read_csv(args.old)
    old = old[old.spine_foreign_true.notna()][
        ["sop_uid", "study", "spine_foreign_true", "spine_foreign_pred"]]
    merged = new.merge(old, on="sop_uid", how="inner", suffixes=("", "_old"))

    truth = merged.y_true.to_numpy(dtype=int)
    guess_new = merged.y_pred.to_numpy(dtype=int)
    guess_old = merged.spine_foreign_pred.to_numpy(dtype=int)
    groups = merged.study.to_numpy(dtype=str) if "study" in merged else merged.study_old.to_numpy(dtype=str)

    # Метки двух источников обязаны совпадать: иначе сравниваем разные задачи.
    mismatch = int((truth != merged.spine_foreign_true.to_numpy(dtype=int)).sum())
    if mismatch:
        raise SystemExit(f"метки расходятся на {mismatch} снимках — сравнение бессмысленно")

    score_new, score_old = f1_of(truth, guess_new), f1_of(truth, guess_old)
    print(f"снимков {len(merged)}, положительных {int(truth.sum())}")
    print(f"  старая голова (CNN):      F1 {score_old:.4f}")
    print(f"  патч-классификатор:       F1 {score_new:.4f}")
    print(f"  разность:                 {score_new - score_old:+.4f}")

    rng = np.random.default_rng(11)
    studies = np.unique(groups)
    differences = []
    for _ in range(args.repeats):
        chosen = rng.choice(studies, size=len(studies), replace=True)
        index = np.concatenate([np.flatnonzero(groups == s) for s in chosen])
        differences.append(f1_of(truth[index], guess_new[index])
                           - f1_of(truth[index], guess_old[index]))
    differences.sort()
    low = differences[int(0.025 * len(differences))]
    high = differences[int(0.975 * len(differences))]
    better = float(np.mean(np.asarray(differences) > 0))

    print(f"  95 % ДИ разности:         [{low:+.4f}, {high:+.4f}]")
    print(f"  доля реплик, где новая лучше: {better:.1%}")

    verdict = ("подтверждение заявленной пользы" if low >= 0.10 else
               "свидетельство улучшения" if low > 0 else
               "точечный результат без статистического подтверждения"
               if score_new - score_old >= 0.10 else "улучшения не показано")
    print(f"\nвывод по протоколу: {verdict}")

    report = {"n": int(len(merged)), "n_positive": int(truth.sum()),
              "f1_old": round(score_old, 4), "f1_new": round(score_new, 4),
              "delta": round(score_new - score_old, 4),
              "delta_ci95": [round(low, 4), round(high, 4)],
              "share_new_better": round(better, 4), "verdict": verdict}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сохранено: {args.out}")


if __name__ == "__main__":
    main()
