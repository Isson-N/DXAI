#!/usr/bin/env python3
"""Аудит OOF-предсказаний бейзлайнов: TP/FP/FN, разброс по фолдам, парное сравнение моделей.

Точечные F1 при 6–17 положительных сравнивать между моделями нельзя: разница попадает
в шум. Здесь — парный кластерный бутстрэп по исследованиям на ОБЩИХ снимках двух моделей
(одни и те же реплики для обеих), доля реплик, где модель A лучше B, и ДИ разницы.

    python training/audit_oof.py experiments/results/b0/oof.csv experiments/results/b1/oof.csv \
        experiments/results/b2/oof.csv --out experiments/results/audit.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_auc import prepare_auc, weighted_auc

HEADS = ["region", "spine_pos", "spine_axis", "spine_foreign", "spine_any",
         "hip_pos", "hip_roi", "hip_any"]


def f1_score_counts(y, pred, weights):
    tp = weights[(y == 1) & (pred == 1)].sum()
    fp = weights[(y == 0) & (pred == 1)].sum()
    fn = weights[(y == 1) & (pred == 0)].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0


def auc(y, p, weights, prepared=None):
    if weights[y == 1].sum() == 0 or weights[y == 0].sum() == 0:
        return np.nan
    return weighted_auc(y, weights, prepared if prepared is not None else prepare_auc(p))


def load(paths):
    tables = {}
    for path in paths:
        name = Path(path).parent.name
        tables[name] = pd.read_csv(path)
    return tables


def counts(df, head):
    y = df[f"{head}_true"].to_numpy()
    pred = df[f"{head}_pred"].to_numpy()
    keep = np.isfinite(y)
    y, pred = y[keep].astype(int), pred[keep].astype(int)
    return {"n": int(len(y)), "n_pos": int(y.sum()),
            "tp": int(((y == 1) & (pred == 1)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tn": int(((y == 0) & (pred == 0)).sum())}


def per_fold(df, head):
    out = {}
    for fold, part in df.groupby("fold"):
        y = part[f"{head}_true"].to_numpy()
        keep = np.isfinite(y)
        if keep.sum() == 0:
            continue
        y = y[keep].astype(int)
        pred = part[f"{head}_pred"].to_numpy()[keep].astype(int)
        p = part[f"{head}_prob"].to_numpy()[keep]
        w = np.ones(len(y))
        out[int(fold)] = {"n_pos": int(y.sum()), "f1": round(f1_score_counts(y, pred, w), 3),
                          "auc": None if np.isnan(auc(y, p, w)) else round(auc(y, p, w), 3)}
    return out


def paired_bootstrap(a, b, head, n_boot, seed):
    """Разница метрик A − B на общих снимках, одни и те же реплики исследований для обеих моделей."""
    merged = a.merge(b, on="sop_uid", suffixes=("_a", "_b"))
    y = merged[f"{head}_true_a"].to_numpy()
    keep = np.isfinite(y)
    merged, y = merged[keep], y[keep].astype(int)
    if len(y) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
        return None
    pa, pb = merged[f"{head}_prob_a"].to_numpy(), merged[f"{head}_prob_b"].to_numpy()
    da, db = merged[f"{head}_pred_a"].to_numpy().astype(int), merged[f"{head}_pred_b"].to_numpy().astype(int)
    groups, studies = pd.factorize(merged["study_a"], sort=True)
    rng = np.random.default_rng(seed)
    ones = np.ones(len(y))
    diff_f1, diff_auc = [], []
    prep_a, prep_b = prepare_auc(pa), prepare_auc(pb)
    base = {"f1_a": f1_score_counts(y, da, ones), "f1_b": f1_score_counts(y, db, ones),
            "auc_a": auc(y, pa, ones, prep_a), "auc_b": auc(y, pb, ones, prep_b)}
    for _ in range(n_boot):
        pick = rng.integers(0, len(studies), len(studies))
        w = np.bincount(pick, minlength=len(studies))[groups].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        diff_f1.append(f1_score_counts(y, da, w) - f1_score_counts(y, db, w))
        diff_auc.append(auc(y, pa, w, prep_a) - auc(y, pb, w, prep_b))
    if not diff_f1:
        return None
    diff_f1, diff_auc = np.array(diff_f1), np.array(diff_auc)
    return {
        **{k: (None if v is None or np.isnan(v) else round(float(v), 3)) for k, v in base.items()},
        "n": int(len(y)), "n_pos": int(y.sum()),
        "delta_f1": round(float(base["f1_a"] - base["f1_b"]), 3),
        "delta_f1_ci95": [round(float(x), 3) for x in np.percentile(diff_f1, [2.5, 97.5])],
        "delta_auc": round(float(base["auc_a"] - base["auc_b"]), 3),
        "delta_auc_ci95": [round(float(x), 3) for x in np.percentile(diff_auc, [2.5, 97.5])],
        "p_auc_a_better": round(float((diff_auc > 0).mean()), 3),
        "replicates": int(len(diff_f1)),
    }


def error_correlation(a, b, head):
    """Корреляция ошибок двух моделей: если она высокая, ансамбль бесполезен."""
    merged = a.merge(b, on="sop_uid", suffixes=("_a", "_b"))
    y = merged[f"{head}_true_a"].to_numpy()
    keep = np.isfinite(y)
    if keep.sum() < 5:
        return None
    y = y[keep].astype(int)
    ea = np.abs(y - merged[f"{head}_prob_a"].to_numpy()[keep])
    eb = np.abs(y - merged[f"{head}_prob_b"].to_numpy()[keep])
    if ea.std() == 0 or eb.std() == 0:
        return None
    return round(float(np.corrcoef(ea, eb)[0, 1]), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("oof", nargs="+")
    ap.add_argument("--out", default="experiments/results/audit.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    tables = load(args.oof)
    report = {"models": list(tables), "heads": {}}
    for head in HEADS:
        entry = {"counts": {}, "per_fold": {}, "paired": {}, "error_correlation": {}}
        for name, df in tables.items():
            if f"{head}_true" not in df.columns:
                continue
            entry["counts"][name] = counts(df, head)
            entry["per_fold"][name] = per_fold(df, head)
        present = [n for n in tables if f"{head}_true" in tables[n].columns]
        for x, y in itertools.combinations(present, 2):
            res = paired_bootstrap(tables[x], tables[y], head, args.n_boot, args.seed)
            if res:
                entry["paired"][f"{x}_vs_{y}"] = res
                entry["error_correlation"][f"{x}_vs_{y}"] = error_correlation(tables[x], tables[y], head)
        report["heads"][head] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'голова':15s} {'n+':>3s}  TP/FP/FN по моделям")
    for head, entry in report["heads"].items():
        cells = " | ".join(f"{n}: {c['tp']}/{c['fp']}/{c['fn']}" for n, c in entry["counts"].items())
        npos = next(iter(entry["counts"].values()))["n_pos"] if entry["counts"] else 0
        print(f"{head:15s} {npos:3d}  {cells}")
    print("\nПарное сравнение (Δ = первая модель − вторая):")
    for head, entry in report["heads"].items():
        for pair, res in entry["paired"].items():
            corr = entry["error_correlation"].get(pair)
            print(f"  {head:15s} {pair:9s} ΔAUC {res['delta_auc']:+.3f} "
                  f"[{res['delta_auc_ci95'][0]:+.2f}; {res['delta_auc_ci95'][1]:+.2f}] "
                  f"P(A лучше)={res['p_auc_a_better']:.2f}  ΔF1 {res['delta_f1']:+.3f} "
                  f"[{res['delta_f1_ci95'][0]:+.2f}; {res['delta_f1_ci95'][1]:+.2f}]"
                  + (f"  corr ошибок {corr:+.2f}" if corr is not None else ""))
    print(f"\nОтчёт: {args.out}")


if __name__ == "__main__":
    main()
