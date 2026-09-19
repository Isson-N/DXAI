#!/usr/bin/env python3
"""Быстрый взвешенный ROC-AUC и кластерный бутстрэп по исследованиям.

`roc_auc_score(..., sample_weight=...)` из sklearn на каждой реплике сортирует заново:
2000 реплик × десятки признаков считались минутами. Здесь порядок и группы одинаковых
значений считаются один раз, а каждая реплика — это несколько операций над массивами.
"""
import numpy as np


def prepare_auc(score):
    """Порядок по возрастанию и границы групп одинаковых значений (ничьи)."""
    order = np.argsort(score, kind="stable")
    sorted_score = np.asarray(score)[order]
    group = np.r_[0, np.cumsum(sorted_score[1:] != sorted_score[:-1])]
    return order, group, int(group[-1]) + 1 if len(group) else 0


def weighted_auc(y, weights, prepared):
    """AUC = доля пар (положительный, отрицательный) с верным порядком; ничьи считаются за половину."""
    order, group, n_groups = prepared
    y, weights = np.asarray(y)[order], np.asarray(weights)[order]
    pos = np.bincount(group, weights=weights * (y == 1), minlength=n_groups)
    neg = np.bincount(group, weights=weights * (y == 0), minlength=n_groups)
    total_pos, total_neg = pos.sum(), neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    below = np.r_[0.0, np.cumsum(neg)[:-1]]
    return float((pos * (below + 0.5 * neg)).sum() / (total_pos * total_neg))


def cluster_bootstrap_auc(y, score, groups, n_boot=2000, seed=11):
    """95% ДИ AUC при повторной выборке ИССЛЕДОВАНИЙ (снимки одного исследования зависимы)."""
    y = np.asarray(y, dtype=int)
    prepared = prepare_auc(np.asarray(score, dtype=float))
    keys = np.unique(groups)
    index = np.searchsorted(keys, groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        w = np.bincount(pick, minlength=len(keys))[index].astype(float)
        value = weighted_auc(y, w, prepared)
        if np.isfinite(value):
            values.append(value)
    if not values:
        return [None, None], 1.0
    return ([round(float(x), 3) for x in np.percentile(values, [2.5, 97.5])],
            round(1 - len(values) / n_boot, 3))


def paired_bootstrap_auc(y, score_a, score_b, groups, n_boot=2000, seed=11):
    """Разница AUC двух моделей на одних и тех же репликах исследований."""
    y = np.asarray(y, dtype=int)
    pa, pb = prepare_auc(np.asarray(score_a, float)), prepare_auc(np.asarray(score_b, float))
    keys = np.unique(groups)
    index = np.searchsorted(keys, groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        w = np.bincount(pick, minlength=len(keys))[index].astype(float)
        a, b = weighted_auc(y, w, pa), weighted_auc(y, w, pb)
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    if not diffs:
        return None
    diffs = np.array(diffs)
    return {"ci95": [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])],
            "p_a_better": round(float((diffs > 0).mean()), 3), "replicates": len(diffs)}
