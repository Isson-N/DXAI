#!/usr/bin/env python3
"""CNN b2 с энкодером, предобученным на синтетической ротации бедра, против обычной b2.

Попарно по seed (одинаковые фолды и seed, отличается только инициализация энкодера).
Критерий внедрения из docs/drr_hip_rotation_plan.md: AUC hip_pos выше на КАЖДОМ seed
и в среднем не меньше +0,02. F1 — с порогом по OOF остальных фолдов (как ensemble_cnn_heads).

    python training/compare_drr_init.py 42:results_final 1:results_seed1 2:results_seed2
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "training")
from ensemble_cnn_heads import f1_of  # noqa: E402
from eval_hip_geometry import auc  # noqa: E402

HEADS = ["hip_pos", "hip_roi", "spine_foreign", "spine_axis", "spine_pos"]


def scores(frame, head, uids=None):
    t, p = f"{head}_true", f"{head}_prob"
    sub = frame[frame[t].notna()].set_index("sop_uid")
    if uids is not None:
        sub = sub.loc[uids]
    y, prob = sub[t].astype(int).to_numpy(), sub[p].to_numpy()
    # F1 по решениям самого прогона: порог выбран по ВНУТРЕННИМ OOF (вложенно)
    pred = sub[f"{head}_pred"].astype(int).to_numpy()
    return sub.index, auc(prob[y == 1], prob[y == 0]), f1_of(y, pred)


def bootstrap(pairs, n=2000, seed=0):
    """Парный бутстрэп по исследованиям: dAUC hip_pos для вероятностей, усреднённых по seed."""
    base, drr = [], []
    for pair in pairs:
        s, b = pair.split(":")
        base.append(pd.read_csv(f"experiments/{b}/b2/oof.csv").set_index("sop_uid"))
        drr.append(pd.read_csv(f"experiments/results_drr_seed{s}/b2/oof.csv").set_index("sop_uid"))
    sub = base[0][base[0]["hip_pos_true"].notna()]
    uids, y, study = sub.index, sub["hip_pos_true"].astype(int).to_numpy(), sub["study"].to_numpy()
    pa = np.mean([f.loc[uids, "hip_pos_prob"].to_numpy() for f in base], axis=0)
    pb = np.mean([f.loc[uids, "hip_pos_prob"].to_numpy() for f in drr], axis=0)
    studies = np.unique(study); rng = np.random.default_rng(seed); diffs = []
    groups = {s: np.flatnonzero(study == s) for s in studies}
    for _ in range(n):
        idx = np.concatenate([groups[s] for s in rng.choice(studies, len(studies))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(auc(pb[idx][yy == 1], pb[idx][yy == 0]) - auc(pa[idx][yy == 1], pa[idx][yy == 0]))
    diffs = np.asarray(diffs)
    point = auc(pb[y == 1], pb[y == 0]) - auc(pa[y == 1], pa[y == 0])
    return (f"\nпарный бутстрэп по исследованиям (среднее по seed), dAUC hip_pos = {point:+.3f}, "
            f"95% ДИ [{np.quantile(diffs, .025):+.3f}, {np.quantile(diffs, .975):+.3f}], "
            f"P(d<=0) = {(diffs <= 0).mean():.3f}")


def main():
    rows = []
    for pair in sys.argv[1:]:
        seed, base = pair.split(":")
        a = pd.read_csv(f"experiments/{base}/b2/oof.csv")
        b = pd.read_csv(f"experiments/results_drr_seed{seed}/b2/oof.csv")
        for head in HEADS:
            uids, auc_a, f1_a = scores(a, head)
            _, auc_b, f1_b = scores(b, head, uids)
            rows.append({"seed": seed, "head": head, "auc_base": auc_a, "auc_drr": auc_b,
                         "f1_base": f1_a, "f1_drr": f1_b})
    frame = pd.DataFrame(rows)
    frame["d_auc"] = frame.auc_drr - frame.auc_base
    print(frame.round(3).to_string(index=False))
    print("\nсреднее по seed:")
    print(frame.groupby("head")[["auc_base", "auc_drr", "d_auc", "f1_base", "f1_drr"]].mean().round(3))
    hip = frame[frame["head"] == "hip_pos"]
    ok = bool((hip.d_auc > 0).all() and hip.d_auc.mean() >= 0.02)
    print(bootstrap(sys.argv[1:]))
    print(f"\nкритерий внедрения (hip_pos: AUC выше на каждом seed и среднее >= +0,02): {'ВЫПОЛНЕН' if ok else 'не выполнен'}")


if __name__ == "__main__":
    main()
