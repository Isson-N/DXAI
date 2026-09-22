#!/usr/bin/env python3
"""Патч-классификатор предметов: порог и агрегация по ОБЪЕДИНЁННЫМ внутренним OOF.

Совет astra 22.09.2026. Порог, выбранный на одном внутреннем фолде (3–4 положительных),
делал F1 лотереей: 0,58–0,79 при стабильном AUC 0,86–0,88. Здесь для каждого внешнего
фолда модели обучаются на внутренних разбиениях его обучающей части, их предсказания
на отложенных внутренних фолдах объединяются, и уже на этом пуле выбираются агрегация
(заранее заданные два варианта: максимум и среднее трёх лучших окон) и порог. Затем
модели обучаются на всей обучающей части и применяются к внешнему фолду один раз.

    python training/train_foreign_nested.py --patches tr.npz --scan scan.npz --seeds 1,2,3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_foreign_patches import train_fold, ensemble_scores, f1_of

AGGREGATIONS = {
    "max": lambda s: float(np.max(s)),
    "top3": lambda s: float(np.mean(np.sort(s)[-3:])),
}


def image_level(scores, uids):
    """Оценки окон -> по снимку для каждой агрегации."""
    frame = pd.DataFrame({"uid": uids, "s": scores})
    return {name: frame.groupby("uid").s.apply(lambda v: fn(v.to_numpy())).to_dict()
            for name, fn in AGGREGATIONS.items()}


def best_threshold(scores, truth):
    best, best_score = 0.5, -1.0
    for candidate in np.unique(np.round(scores, 4)):
        value = f1_of(truth, (scores >= candidate).astype(int))
        if value > best_score:
            best, best_score = float(candidate), value
    return best, best_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patches", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--out", default="experiments/results/foreign_patch_nested")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", default="1,2,3")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    data = np.load(args.patches, allow_pickle=True)
    scan = np.load(args.scan, allow_pickle=True)
    index = pd.read_csv(args.index).set_index("sop_uid")

    patches = torch.from_numpy(data["patches"]).float().unsqueeze(1)
    labels = torch.from_numpy(data["labels"]).float()
    folds = np.asarray(data["fold"], dtype=int)
    studies = np.asarray([str(s) for s in data["study"]])
    counts = pd.Series(studies).value_counts()
    weights = torch.from_numpy(
        np.asarray([1.0 / counts[s] for s in studies], dtype=np.float32) * len(counts))

    scan_patches = torch.from_numpy(scan["patches"]).float().unsqueeze(1)
    scan_uids = np.asarray([str(u) for u in scan["sop_uid"]])
    scan_folds = np.asarray(scan["fold"], dtype=int)
    truth = {u: int(index.loc[u, "y_foreign"]) for u in np.unique(scan_uids)
             if u in index.index and pd.notna(index.loc[u, "y_foreign"])}

    def fit(mask, tag):
        return [train_fold(patches[mask], labels[mask], weights[mask], device,
                           args.epochs, seed * 1000 + tag) for seed in seeds]

    started, rows, choices = time.time(), [], []
    all_folds = sorted(set(folds))
    for outer in all_folds:
        # 1) внутренний OOF на обучающей части внешнего фолда
        pooled = {name: {} for name in AGGREGATIONS}
        for inner in [f for f in all_folds if f != outer]:
            models = fit((folds != outer) & (folds != inner), outer * 10 + inner)
            sel = scan_folds == inner
            per_image = image_level(ensemble_scores(models, scan_patches[sel], device), scan_uids[sel])
            for name in AGGREGATIONS:
                pooled[name].update(per_image[name])
        # 2) выбор агрегации и порога ТОЛЬКО на пуле внутренних OOF
        best = None
        for name, scores in pooled.items():
            uids = [u for u in scores if u in truth]
            th, score = best_threshold(np.asarray([scores[u] for u in uids]),
                                       np.asarray([truth[u] for u in uids]))
            if best is None or score > best[2]:
                best = (name, th, score)
        name, threshold, inner_f1 = best
        choices.append({"outer": outer, "aggregation": name, "threshold": threshold,
                        "inner_pooled_f1": round(inner_f1, 4)})
        # 3) модели на всей обучающей части -> внешний фолд, один раз
        models = fit(folds != outer, outer * 10 + 9)
        sel = scan_folds == outer
        per_image = image_level(ensemble_scores(models, scan_patches[sel], device), scan_uids[sel])[name]
        for uid, score in per_image.items():
            if uid in truth:
                rows.append({"sop_uid": uid, "fold": outer, "score": score, "threshold": threshold,
                             "aggregation": name, "y_true": truth[uid],
                             "y_pred": int(score >= threshold),
                             "study": str(index.loc[uid, "study"])})
        print(f"фолд {outer}: агрегация {name}, порог {threshold:.3f}, "
              f"F1 на внутреннем пуле {inner_f1:.3f}", flush=True)

    frame = pd.DataFrame(rows)
    score = f1_of(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "oof.csv", index=False)
    report = {"f1": round(score, 4), "choices": choices, "seeds": seeds,
              "tp": int(((frame.y_pred == 1) & (frame.y_true == 1)).sum()),
              "fp": int(((frame.y_pred == 1) & (frame.y_true == 0)).sum()),
              "fn": int(((frame.y_pred == 0) & (frame.y_true == 1)).sum()),
              "training_seconds": round(time.time() - started, 1)}
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nF1 на уровне снимка: {score:.4f}  (TP {report['tp']}, FP {report['fp']}, FN {report['fn']})")


if __name__ == "__main__":
    main()
