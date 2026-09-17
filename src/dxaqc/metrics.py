"""Метрики и доверительные интервалы (план v2, раздел 2, пп. 4–5)."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float] | None = None) -> dict:
    """F1, Se, Sp, balanced accuracy — по выданному классу; ROC-AUC и AP — по вероятности."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    se = tp / (tp + fn) if tp + fn else math.nan
    sp = tn / (tn + fp) if tn + fp else math.nan
    out = {
        "n": int(y.size), "n_pos": int(y.sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "f1": float(f1_score(y, p, zero_division=0)) if y.size else math.nan,
        "sensitivity": se, "specificity": sp,
        "balanced_accuracy": (se + sp) / 2 if not (math.isnan(se) or math.isnan(sp)) else math.nan,
        "roc_auc": math.nan, "ap": math.nan,
    }
    if y_score is not None and 0 < y.sum() < y.size:  # AUC/AP определены только при обоих классах
        s = np.asarray(y_score, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
    return out


def macro_f1(per_violation: dict[str, tuple[Sequence[int], Sequence[int]]]) -> float:
    """Простое среднее F1 по нарушениям.

    per_violation = {нарушение: (y_true, y_pred)} — 5 нарушений (3 поясницы, 2 бедра), каждое на своей области.
    Организатор (Q&A): «F1 по макроагрегированной по отдельным нарушениям; пять типов нарушений».
    """
    scores = [f1_score(np.asarray(y, int), np.asarray(p, int), zero_division=0) for y, p in per_violation.values()]
    return float(np.mean(scores)) if scores else math.nan


def cluster_bootstrap_ci(
    groups: Sequence,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """95% ДИ кластерным бутстрэпом: ресэмплируются группы (исследования) с возвращением.

    statistic(indices) -> float; реплики, где статистика не определена (nan: нет положительных или
    отрицательных), пропускаются, их доля возвращается в skipped_share.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    values, skipped = [], 0
    for _ in range(n_boot):
        sample = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([index_by_group[g] for g in sample])
        v = statistic(idx)
        if v is None or math.isnan(v):
            skipped += 1
        else:
            values.append(v)
    point = statistic(np.arange(groups.size))
    if not values:
        return {"estimate": point, "low": math.nan, "high": math.nan, "skipped_share": 1.0}
    low, high = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": point, "low": float(low), "high": float(high), "skipped_share": skipped / n_boot}
