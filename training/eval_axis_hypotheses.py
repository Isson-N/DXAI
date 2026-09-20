"""Проверка гипотез о том, что на самом деле размечали меткой «ось».

Четыре геометрические гипотезы уже провалились (угол линии гребней, отклонение
от хорды, нормированное отклонение, кривизна). Здесь считаются признаки,
предложенные astra и fable 20.09.2026:

    A1 (astra)  смещение позвоночника в кадре, а не наклон:
                c_x = |mean(x) - W/2| / W,  a_x = |m_L - m_R| / (m_L + m_R)
    F1 (fable)  косость таза как разновысотность гребней, а не угол их линии:
                p = |Δy(гребни)| / d(L5, середина гребней)
    F3 (fable)  непараллельность краю кадра, а не ось тела:
                e = (d(L5, край) - d(Th12, край)) / W

Для каждого признака — AUC с кластерным бутстрэпом по исследованиям и
корреляция с углом хорды (для F3 она решающая: если признак повторяет угол,
гипотеза ничего не объясняет).

Запуск:
    python training/eval_axis_hypotheses.py \
        --points experiments/results/keypoints/oof_points.csv \
        --features experiments/results/keypoints/oof_features.csv \
        --index data/index/images.csv \
        --out experiments/results/keypoints/axis_hypotheses.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_auc import cluster_bootstrap_auc, prepare_auc, weighted_auc

VERTEBRAE = ("Th12", "L1", "L2", "L3", "L4", "L5")


def columns(df: pd.DataFrame, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    """Матрицы x и y центров позвонков (строки — снимки, столбцы — позвонки)."""
    xs = np.column_stack([df[f"{v}_x_{suffix}"].to_numpy(dtype=float) for v in VERTEBRAE])
    ys = np.column_stack([df[f"{v}_y_{suffix}"].to_numpy(dtype=float) for v in VERTEBRAE])
    return xs, ys


def centre_offset(xs: np.ndarray, width: np.ndarray) -> np.ndarray:
    """A1: насколько середина позвоночника смещена от середины кадра."""
    return np.abs(np.nanmean(xs, axis=1) - width / 2.0) / width


def margin_asymmetry(xs: np.ndarray, width: np.ndarray) -> np.ndarray:
    """A1: асимметрия полей слева и справа от позвоночника."""
    left = np.nanmin(xs, axis=1)
    right = width - np.nanmax(xs, axis=1)
    total = left + right
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(left - right) / np.where(total > 0, total, np.nan)


def frame_trapezoid(xs: np.ndarray, width: np.ndarray) -> np.ndarray:
    """F3: расхождение расстояний до левого края у L5 и Th12."""
    return (xs[:, -1] - xs[:, 0]) / width


def crest_height_difference(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """F1: разновысотность гребней, нормированная на расстояние до L5."""
    lx = df[f"crest_left_x_{suffix}"].to_numpy(dtype=float)
    ly = df[f"crest_left_y_{suffix}"].to_numpy(dtype=float)
    rx = df[f"crest_right_x_{suffix}"].to_numpy(dtype=float)
    ry = df[f"crest_right_y_{suffix}"].to_numpy(dtype=float)
    l5x = df["L5_x_" + suffix].to_numpy(dtype=float)
    l5y = df["L5_y_" + suffix].to_numpy(dtype=float)
    mid_x, mid_y = (lx + rx) / 2.0, (ly + ry) / 2.0
    distance = np.hypot(l5x - mid_x, l5y - mid_y)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(ly - ry) / np.where(distance > 0, distance, np.nan)


def auc_with_ci(y: np.ndarray, score: np.ndarray, groups: np.ndarray, n_boot: int = 2000):
    """AUC и 95% ДИ кластерным бутстрэпом; None, если данных не хватает."""
    good = np.isfinite(score) & np.isfinite(y)
    y_g, s_g, g_g = y[good].astype(int), score[good], groups[good]
    if len(np.unique(y_g)) < 2 or len(y_g) < 10:
        return None
    prep = prepare_auc(s_g)
    point = weighted_auc(y_g, np.ones(len(y_g)), prep)
    (lo, hi), _ = cluster_bootstrap_auc(y_g, s_g, g_g, n_boot=n_boot)
    return {
        "n": int(len(y_g)),
        "n_pos": int(y_g.sum()),
        "auc": round(float(point), 3),
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
    }


def cross_fitted_logit(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """OOF-вероятности логистической регрессии со стандартизацией внутри фолда."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        if len(np.unique(y[train])) < 2:
            continue
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"))
        model.fit(x[train], y[train])
        out[test] = model.predict_proba(x[test])[:, 1]
    return out


def combined_models(df: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                    width: np.ndarray) -> dict:
    """Угол сам по себе против «угол + признаки положения в кадре»."""
    result = {}
    folds = df["fold"].to_numpy()
    for suffix in ("true", "pred"):
        xs, _ = columns(df, suffix)
        angle = np.abs(df[f"angle_chord_{suffix}"].to_numpy(dtype=float))
        block = {
            "угол": np.column_stack([angle]),
            "угол + c_x + a_x": np.column_stack([angle, centre_offset(xs, width),
                                                 margin_asymmetry(xs, width)]),
        }
        for name, x in block.items():
            good = np.isfinite(x).all(axis=1) & np.isfinite(y)
            p = np.full(len(y), np.nan)
            p[good] = cross_fitted_logit(x[good], y[good].astype(int), folds[good])
            result[f"{suffix}: {name}"] = auc_with_ci(y, p, groups)
    return result


def disputed_positives(df: pd.DataFrame, y: np.ndarray, width: np.ndarray) -> list:
    """Положительные с углом < 5°: видно ли у них смещение или асимметрию полей."""
    xs, _ = columns(df, "true")
    c_x, a_x = centre_offset(xs, width), margin_asymmetry(xs, width)
    angle = np.abs(df["angle_chord_true"].to_numpy(dtype=float))
    rows = []
    for i in np.where(y == 1)[0]:
        rows.append({"angle_true": None if not np.isfinite(angle[i]) else round(float(angle[i]), 2),
                     "c_x": None if not np.isfinite(c_x[i]) else round(float(c_x[i]), 3),
                     "a_x": None if not np.isfinite(a_x[i]) else round(float(a_x[i]), 3)})
    # пороги «выраженности» — 75-й процентиль по отрицательным
    neg = y == 0
    thr_c = float(np.nanpercentile(c_x[neg], 75))
    thr_a = float(np.nanpercentile(a_x[neg], 75))
    for row in rows:
        row["выражено"] = bool((row["c_x"] or 0) > thr_c or (row["a_x"] or 0) > thr_a)
    return [{"порог_c_x": round(thr_c, 3), "порог_a_x": round(thr_a, 3)}] + rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", default="experiments/results/keypoints/oof_points.csv")
    ap.add_argument("--features", default="experiments/results/keypoints/oof_features.csv")
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--out", default="experiments/results/keypoints/axis_hypotheses.json")
    args = ap.parse_args()

    points = pd.read_csv(args.points)
    features = pd.read_csv(args.features)[["sop_uid", "y_axis", "angle_chord_pred", "angle_chord_true"]]
    index = pd.read_csv(args.index)[["sop_uid", "rows", "cols"]]
    df = points.merge(features, on="sop_uid").merge(index, on="sop_uid", how="left")

    y = df["y_axis"].to_numpy(dtype=float)
    groups = pd.factorize(df["study"], sort=True)[0]
    width = df["cols"].to_numpy(dtype=float)
    report: dict = {"n": int(len(df)), "n_pos": int(np.nansum(y)), "hypotheses": {}}

    for suffix in ("true", "pred"):
        xs, _ = columns(df, suffix)
        angle = np.abs(df[f"angle_chord_{suffix}"].to_numpy(dtype=float))
        candidates = {
            f"A1:c_x смещение центра ({suffix})": centre_offset(xs, width),
            f"A1:a_x асимметрия полей ({suffix})": margin_asymmetry(xs, width),
            f"F3:e трапеция кадра ({suffix})": np.abs(frame_trapezoid(xs, width)),
            f"F1:p разновысотность гребней ({suffix})": crest_height_difference(df, suffix),
        }
        for name, score in candidates.items():
            entry = auc_with_ci(y, score, groups)
            if entry is None:
                report["hypotheses"][name] = {"skipped": "мало данных или один класс"}
                continue
            both = np.isfinite(score) & np.isfinite(angle)
            if both.sum() > 3:
                entry["r_with_angle"] = round(float(np.corrcoef(score[both], angle[both])[0, 1]), 3)
            report["hypotheses"][name] = entry

    # A1 проверяется не сама по себе, а как прибавка к углу: критерий astra —
    # прирост OOF AUC >= 0,05 и объяснение хотя бы 3 из 5 положительных с углом < 5°.
    report["combined"] = combined_models(df, y, groups, width)
    report["disputed"] = disputed_positives(df, y, width)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"снимков {report['n']}, положительных по оси {report['n_pos']}\n")
    print(f"{'признак':44s} {'n':>4s} {'AUC':>6s} {'ДИ95':>16s} {'r с углом':>10s}")
    for name, entry in report["hypotheses"].items():
        if "auc" not in entry:
            print(f"{name:44s} {'—':>4s}  {entry.get('skipped', '')}")
            continue
        ci = f"[{entry['ci95'][0]:.2f}; {entry['ci95'][1]:.2f}]"
        r = entry.get("r_with_angle")
        print(f"{name:44s} {entry['n']:4d} {entry['auc']:6.3f} {ci:>16s} "
              f"{'' if r is None else f'{r:10.2f}'}")
    print("\nA1 как прибавка к углу (кросс-фиттинг по фолдам):")
    for name, entry in report["combined"].items():
        if entry is None:
            print(f"  {name:28s} — не посчитано")
            continue
        print(f"  {name:28s} AUC {entry['auc']:.3f} [{entry['ci95'][0]:.2f}; {entry['ci95'][1]:.2f}]")

    head, *rows = report["disputed"]
    disputed = [r for r in rows if r["angle_true"] is not None and r["angle_true"] < 5]
    explained = sum(1 for r in disputed if r["выражено"])
    print(f"\nПоложительные с углом < 5°: {len(disputed)}; из них со смещением или "
          f"асимметрией выше 75-го процентиля отрицательных: {explained}")
    print(f"  (пороги: c_x > {head['порог_c_x']}, a_x > {head['порог_a_x']})")

    print(f"\nЗаписано: {args.out}")


if __name__ == "__main__":
    main()
