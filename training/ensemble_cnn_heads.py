#!/usr/bin/env python3
"""Усреднение вероятностей CNN-голов по нескольким seed и честное сравнение с одной моделью.

Порог для каждого внешнего фолда выбирается по объединённым OOF остальных фолдов
(максимум F1 при специфичности не ниже 0,80, как в docs/final_protocol.md) — одинаково
для одиночной модели и ансамбля, иначе выигрыш ансамбля окажется выигрышем порога.

    python training/ensemble_cnn_heads.py experiments/results_final/b2/oof.csv \
        experiments/results_seed1/b2/oof.csv experiments/results_seed2/b2/oof.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "training")
from eval_hip_geometry import auc  # noqa: E402

HEADS = ["hip_pos", "hip_roi", "spine_foreign", "spine_axis", "spine_pos"]
MIN_SPECIFICITY = 0.80


def f1_of(t, g):
    tp = ((g == 1) & (t == 1)).sum(); fp = ((g == 1) & (t == 0)).sum(); fn = ((g == 0) & (t == 1)).sum()
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def pooled_threshold_f1(y, p, folds):
    pred = np.zeros(len(y), int)
    for k in np.unique(folds):
        o = folds != k
        yo, po = y[o], p[o]
        best, best_f1 = 1.0, -1.0
        for c in np.unique(po):
            g = (po >= c).astype(int)
            spec = ((g == 0) & (yo == 0)).sum() / max((yo == 0).sum(), 1)
            if spec < MIN_SPECIFICITY:
                continue
            v = f1_of(yo, g)
            if v > best_f1:
                best, best_f1 = c, v
        pred[folds == k] = (p[folds == k] >= best).astype(int)
    return f1_of(y, pred)


def main():
    frames = [pd.read_csv(f) for f in sys.argv[1:]]
    base = frames[0]
    print(f"прогонов: {len(frames)}")
    print(f"{'голова':14s} {'F1 одна':>8s} {'F1 ансамбль':>12s} {'AUC одна':>9s} {'AUC анс.':>9s}")
    for h in HEADS:
        col_t, col_p = f"{h}_true", f"{h}_prob"
        if col_p not in base:
            continue
        mask = base[col_t].notna().to_numpy()
        y = base.loc[mask, col_t].astype(int).to_numpy()
        folds = base.loc[mask, "fold"].to_numpy()
        uids = base.loc[mask, "sop_uid"]
        probs = [f.set_index("sop_uid").loc[uids, col_p].to_numpy() for f in frames]
        single, ens = probs[0], np.mean(probs, axis=0)
        print(f"{h:14s} {pooled_threshold_f1(y, single, folds):8.3f} {pooled_threshold_f1(y, ens, folds):12.3f} "
              f"{auc(single[y == 1], single[y == 0]):9.3f} {auc(ens[y == 1], ens[y == 0]):9.3f}")


if __name__ == "__main__":
    main()
