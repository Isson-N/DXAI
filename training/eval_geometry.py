#!/usr/bin/env python3
"""Этап 4: геометрия поясницы → метки «ось не выровнена» и «некорректная укладка».

Сравнивает два источника признаков на одних и тех же снимках и фолдах:
  * `_true` — ручная разметка (oracle: потолок, в сервисе недоступен);
  * `_pred` — предсказания модели ключевых точек (то, что реально будет в сервисе).

Считает: AUC отдельных признаков, детерминированные правила ТЗ (угол > 5°, гребень вне кадра)
и логистическую регрессию на всех признаках во внешнем CV по тем же фолдам.

    python training/eval_geometry.py experiments/results/keypoints/oof_features.csv \
        --out experiments/results/keypoints/geometry.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ANGLES = ["angle_chord", "angle_ls", "angle_robust"]
NUMERIC = ANGLES + ["max_dev_chord", "curvature", "n_points", "span_mm"]
STATES = ["crest_left_state", "crest_right_state", "th12_half_visible"]


def cluster_ci(y, score, groups, n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        w = np.bincount(pick, minlength=len(keys))[np.searchsorted(keys, groups)].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        keep = w > 0
        values.append(roc_auc_score(y[keep], score[keep], sample_weight=w[keep]))
    if not values:
        return [None, None]
    return [round(float(x), 3) for x in np.percentile(values, [2.5, 97.5])]


def auc_of(y, score, groups):
    good = np.isfinite(score) & np.isfinite(y)
    if good.sum() < 5 or len(np.unique(y[good])) < 2:
        return None
    y, score, groups = y[good].astype(int), score[good], groups[good]
    # Признак может быть как прямым, так и обратным: берём как есть, знак в отчёте.
    return {"auc": round(float(roc_auc_score(y, score)), 3),
            "ci95": cluster_ci(y, score, groups), "n": int(len(y)), "n_pos": int(y.sum())}


def rule_counts(y, decision):
    good = np.isfinite(y)
    y, decision = y[good].astype(int), decision[good].astype(int)
    return {"tp": int(((y == 1) & (decision == 1)).sum()), "fp": int(((y == 0) & (decision == 1)).sum()),
            "fn": int(((y == 1) & (decision == 0)).sum()), "tn": int(((y == 0) & (decision == 0)).sum()),
            "f1": round(float(2 * ((y == 1) & (decision == 1)).sum() /
                              max(2 * ((y == 1) & (decision == 1)).sum() +
                                  ((y == 0) & (decision == 1)).sum() +
                                  ((y == 1) & (decision == 0)).sum(), 1)), 3)}


def design_matrix(df, suffix):
    """Числовые признаки + состояния гребней и Th12 как индикаторы."""
    cols = []
    names = []
    for name in NUMERIC:
        col = df[f"{name}_{suffix}"].to_numpy(dtype=float)
        median = np.nanmedian(col) if np.isfinite(col).any() else 0.0
        cols.append(np.where(np.isfinite(col), col, median))
        names.append(name)
        cols.append((~np.isfinite(df[f"{name}_{suffix}"].to_numpy(dtype=float))).astype(float))
        names.append(f"{name}_missing")
    for name in STATES:
        series = df[f"{name}_{suffix}"].astype(str)
        for value in sorted(set(series) - {"nan", ""}):
            cols.append((series == value).to_numpy(dtype=float))
            names.append(f"{name}={value}")
    x = np.column_stack(cols)
    # Углы берём по модулю: метка «наклон» симметрична по знаку.
    for j, name in enumerate(names):
        if name in ANGLES:
            x[:, j] = np.abs(x[:, j])
    return x, names


def cv_logistic(x, y, folds, seed=42):
    """Внешний CV по фиксированным фолдам; C подбирается во внутреннем CV по AUC."""
    probability = np.full(len(y), np.nan)
    for outer in sorted(set(folds)):
        test = folds == outer
        train = ~test & np.isfinite(y)
        if train.sum() < 5 or len(np.unique(y[train])) < 2:
            continue
        best = None
        for c in (0.01, 0.1, 1.0, 10.0):
            inner = np.full(len(y), np.nan)
            for val in sorted(set(folds[train])):
                tr = train & (folds != val)
                va = train & (folds == val)
                if len(np.unique(y[tr])) < 2 or va.sum() == 0:
                    continue
                model = make_pipeline(StandardScaler(),
                                      LogisticRegression(C=c, max_iter=3000, random_state=seed))
                model.fit(x[tr], y[tr])
                inner[va] = model.predict_proba(x[va])[:, 1]
            good = np.isfinite(inner) & np.isfinite(y)
            if good.sum() < 5 or len(np.unique(y[good])) < 2:
                continue
            score = roc_auc_score(y[good], inner[good])
            if best is None or score > best[0]:
                best = (score, c)
        c = best[1] if best else 1.0
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, random_state=seed))
        model.fit(x[train], y[train])
        probability[test] = model.predict_proba(x[test])[:, 1]
    return probability


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features")
    ap.add_argument("--out", default="experiments/results/keypoints/geometry.json")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    groups = pd.factorize(df["study"], sort=True)[0]
    folds = df["fold"].to_numpy()
    report = {"n": int(len(df)), "targets": {}}

    for target in ("y_axis", "y_pos"):
        y = df[target].to_numpy(dtype=float)
        entry = {"n_pos": int(np.nansum(y)), "features": {}, "rules": {}, "model": {}}
        for suffix in ("true", "pred"):
            entry["features"][suffix] = {
                name: auc_of(y, np.abs(df[f"{name}_{suffix}"].to_numpy(dtype=float)) if name in ANGLES
                             else df[f"{name}_{suffix}"].to_numpy(dtype=float), groups)
                for name in NUMERIC
            }
            if target == "y_axis":
                for name in ANGLES:
                    decision = np.abs(df[f"{name}_{suffix}"].to_numpy(dtype=float)) > 5
                    entry["rules"][f"{suffix}:{name}>5°"] = rule_counts(y, decision)
            else:
                left = df[f"crest_left_state_{suffix}"].astype(str)
                right = df[f"crest_right_state_{suffix}"].astype(str)
                th12 = df[f"th12_half_visible_{suffix}"].astype(str)
                out_of_frame = (left == "out_of_frame") | (right == "out_of_frame")
                entry["rules"][f"{suffix}:гребень вне кадра"] = rule_counts(y, out_of_frame.to_numpy())
                entry["rules"][f"{suffix}:гребень вне кадра или Th12<половины"] = rule_counts(
                    y, (out_of_frame | (th12 == "no")).to_numpy())
            x, names = design_matrix(df, suffix)
            p = cv_logistic(x, np.nan_to_num(y, nan=0.0), folds)
            good = np.isfinite(p) & np.isfinite(y)
            entry["model"][suffix] = {
                "auc": round(float(roc_auc_score(y[good], p[good])), 3) if len(np.unique(y[good])) > 1 else None,
                "ci95": cluster_ci(y[good].astype(int), p[good], groups[good]),
                "n_features": len(names),
            }
        report["targets"][target] = entry

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for target, entry in report["targets"].items():
        print(f"\n=== {target} (положительных {entry['n_pos']})")
        for suffix in ("true", "pred"):
            best = [(v["auc"], k) for k, v in entry["features"][suffix].items() if v]
            best.sort(reverse=True)
            source = "ручная разметка" if suffix == "true" else "модель точек"
            print(f"  {source}: лучший признак {best[0][1]} AUC {best[0][0]}" if best else f"  {source}: нет")
            print(f"    логрег на всех признаках: AUC {entry['model'][suffix]['auc']} "
                  f"{entry['model'][suffix]['ci95']}")
        for name, counts in entry["rules"].items():
            print(f"  правило {name}: TP={counts['tp']} FP={counts['fp']} FN={counts['fn']} F1={counts['f1']}")
    print(f"\nОтчёт: {args.out}")


if __name__ == "__main__":
    main()
