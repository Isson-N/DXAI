#!/usr/bin/env python3
"""Патч-классификатор посторонних предметов: обучение и оценка на уровне снимка.

Протокол зафиксирован до прогона: `docs/foreign_patch_protocol.md`. Ключевое отличие
от обычной классификации снимков — обучение идёт на окнах той же сетки, что и инференс,
а решение по снимку получается агрегацией окон (основной вариант — максимум).

Порог выбирается по F1 на полностью просканированных снимках ВНУТРЕННИХ фолдов
и никогда на фолде, где считается результат.

    python training/train_foreign_patches.py \
        --patches experiments/results/foreign_patches.npz \
        --out experiments/results/foreign_patch
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dxaqc.foreign_patch import PatchNet


'''class PatchNet(nn.Module):
    """Небольшая свёрточная сеть: 79 положительных окон не прокормят крупную модель."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, 1)

    def forward(self, x):
        return self.head(self.body(x).flatten(1)).squeeze(-1)
'''


def augment(batch: torch.Tensor) -> torch.Tensor:
    """Отражения и лёгкая фотометрия. Масштаб намеренно не трогаем: при известном
    шаге пикселя физический размер объекта — допустимый признак (замечание astra)."""
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[-1])
    if torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=[-2])
    if torch.rand(1).item() < 0.8:
        batch = batch * (0.85 + 0.3 * torch.rand(1, device=batch.device))
    if torch.rand(1).item() < 0.3:
        batch = batch + torch.randn_like(batch) * 0.02
    return batch.clamp(0, 1)


def train_fold(x, y, weights, device, epochs, seed, init_state=None):
    torch.manual_seed(seed)
    model = PatchNet().to(device)
    if init_state is not None:
        state = torch.load(init_state, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loader = DataLoader(TensorDataset(x, y, weights), batch_size=64, shuffle=True)
    # Положительных окон на порядок меньше — выравниваем вкладом в функцию потерь.
    positive_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=device)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y, batch_w in loader:
            batch_x = augment(batch_x.to(device))
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_y.to(device), pos_weight=positive_weight, reduction="none")
            loss = (loss * batch_w.to(device)).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        schedule.step()
    return model.eval()


def window_scores(model, x, device, batch=256) -> np.ndarray:
    """Оценка каждого окна: нужна не только для агрегации, но и для карт —
    по ним видно, ГДЕ модель ошибается, и куда разметчику смотреть."""
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            chunk = x[start:start + batch].to(device)
            scores.append(torch.sigmoid(model(chunk)).cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def ensemble_scores(models, x, device, batch=256) -> np.ndarray:
    """Среднее sigmoid по моделям ансамбля для каждого окна: гасит разброс между seed,
    который при 17 положительных (0,65-0,79) больше разницы между вариантами."""
    models = models if isinstance(models, (list, tuple)) else [models]
    return np.mean([window_scores(m, x, device, batch) for m in models], axis=0)


def image_scores(model, x, uids, device, batch=256) -> dict[str, float]:
    """Оценка снимка — максимум по его окнам (основной вариант протокола)."""
    scores = ensemble_scores(model, x, device, batch)
    result: dict[str, float] = {}
    for uid, score in zip(uids, scores):
        result[uid] = max(result.get(uid, 0.0), float(score))
    return result


def f1_of(truth: np.ndarray, guess: np.ndarray) -> float:
    tp = int(((guess == 1) & (truth == 1)).sum())
    fp = int(((guess == 1) & (truth == 0)).sum())
    fn = int(((guess == 0) & (truth == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_threshold(scores: np.ndarray, truth: np.ndarray) -> float:
    """Порог по максимуму F1 на просканированных снимках, не по патчам."""
    best, best_score = 0.5, -1.0
    for candidate in np.unique(np.round(scores, 4)):
        value = f1_of(truth, (scores >= candidate).astype(int))
        if value > best_score:
            best, best_score = float(candidate), value
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", default="experiments/results/foreign_patches.npz")
    parser.add_argument("--scan", default=None, help="NPZ полного сканирования для оценки")
    parser.add_argument("--index", default="data/index/images.csv")
    parser.add_argument("--out", default="experiments/results/foreign_patch")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="через запятую")
    parser.add_argument("--final-model", default=None)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init", default=None, help="state_dict для инициализации")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else [args.seed]

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    data = np.load(args.patches, allow_pickle=True)
    index = pd.read_csv(args.index).set_index("sop_uid")

    patches = torch.from_numpy(data["patches"]).float().unsqueeze(1)
    labels = torch.from_numpy(data["labels"]).float()
    uids = np.asarray([str(u) for u in data["sop_uid"]])
    folds = np.asarray(data["fold"], dtype=int)
    studies = np.asarray([str(s) for s in data["study"]])
    centre_x = np.asarray(data["center_x"], dtype=float) if "center_x" in data else np.zeros(len(uids))
    centre_y = np.asarray(data["center_y"], dtype=float) if "center_y" in data else np.zeros(len(uids))
    if args.scan:
        scan_data = np.load(args.scan, allow_pickle=True)
        scan_patches = torch.from_numpy(scan_data["patches"]).float().unsqueeze(1)
        scan_uids = np.asarray([str(u) for u in scan_data["sop_uid"]])
        scan_folds = np.asarray(scan_data["fold"], dtype=int)
        scan_cx = np.asarray(scan_data["center_x"], dtype=float) if "center_x" in scan_data else np.zeros(len(scan_uids))
        scan_cy = np.asarray(scan_data["center_y"], dtype=float) if "center_y" in scan_data else np.zeros(len(scan_uids))

    # Вклад исследования не должен зависеть от того, сколько окон оно дало.
    counts = pd.Series(studies).value_counts()
    weights = torch.from_numpy(
        np.asarray([1.0 / counts[s] for s in studies], dtype=np.float32) * len(counts))

    truth_by_uid = {u: int(index.loc[u, "y_foreign"]) for u in np.unique(uids)
                    if u in index.index and pd.notna(index.loc[u, "y_foreign"])}

    started = time.time()
    rows, window_rows, fold_thresholds = [], [], []
    for outer in sorted(set(folds)):
        test = folds == outer
        train = ~test
        # Внутренний фолд для выбора порога: соседний по кругу, из обучающей части.
        inner = folds == (outer + 1) % (max(folds) + 1)
        fit = train & ~inner

        # Раньше --seeds разбирался, но не использовался: обучалась одна модель.
        model = [train_fold(patches[fit], labels[fit], weights[fit],
                            device, args.epochs, seed * 100 + outer, args.init) for seed in seeds]

        eval_patches, eval_uids = (scan_patches[scan_folds == (outer + 1) % (max(folds) + 1)],
                                   scan_uids[scan_folds == (outer + 1) % (max(folds) + 1)]) if args.scan else (patches[inner], uids[inner])
        inner_scores = image_scores(model, eval_patches, eval_uids, device)
        inner_uids = [u for u in inner_scores if u in truth_by_uid]
        threshold = best_threshold(
            np.asarray([inner_scores[u] for u in inner_uids]),
            np.asarray([truth_by_uid[u] for u in inner_uids]))
        fold_thresholds.append(float(threshold))

        if args.scan:
            scan_test = scan_folds == outer
            raw = ensemble_scores(model, scan_patches[scan_test], device)
            win_uids, win_cx, win_cy = scan_uids[scan_test], scan_cx[scan_test], scan_cy[scan_test]
            test_scores = image_scores(model, scan_patches[scan_test], win_uids, device)
        else:
            raw = ensemble_scores(model, patches[test], device)
            win_uids, win_cx, win_cy = uids[test], centre_x[test], centre_y[test]
            test_scores = image_scores(model, patches[test], uids[test], device)
        window_rows.extend(
            {"sop_uid": u, "score": float(v), "fold": outer, "x": float(cx), "y": float(cy)}
            for u, v, cx, cy in zip(win_uids, raw, win_cx, win_cy))
        for uid, score in test_scores.items():
            if uid in truth_by_uid:
                rows.append({"sop_uid": uid, "fold": outer, "score": score,
                             "threshold": threshold, "y_true": truth_by_uid[uid],
                             "study": str(index.loc[uid, "study"])})
        print(f"фолд {outer}: обучено на {int(fit.sum())} окнах, "
              f"порог {threshold:.3f}, снимков в тесте {len(test_scores)}", flush=True)

    frame = pd.DataFrame(rows)
    frame["y_pred"] = (frame.score >= frame.threshold).astype(int)
    score = f1_of(frame.y_true.to_numpy(), frame.y_pred.to_numpy())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "oof.csv", index=False)
    pd.DataFrame(window_rows).to_csv(out / "windows.csv", index=False)
    report = {"f1": round(score, 4), "n": int(len(frame)),
              "n_positive": int(frame.y_true.sum()),
              "tp": int(((frame.y_pred == 1) & (frame.y_true == 1)).sum()),
              "fp": int(((frame.y_pred == 1) & (frame.y_true == 0)).sum()),
              "fn": int(((frame.y_pred == 0) & (frame.y_true == 1)).sum()),
              "epochs": args.epochs, "device": device, "seeds": seeds,
              "fold_thresholds": fold_thresholds,
              "training_seconds": round(time.time() - started, 1)}
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"\nF1 на уровне снимка: {score:.4f}  (TP {report['tp']}, FP {report['fp']}, FN {report['fn']})")
    print(f"сохранено: {out}")

    if args.final_model:
        # Финальный ансамбль: по модели на seed, обучение на ВСЕХ окнах. Порог — медиана
        # порогов, выбранных во внешних фолдах: на всех данных порог выбрать не на чем.
        final = [train_fold(patches, labels, weights, device, args.epochs, seed, args.init) for seed in seeds]
        payload = {"models": [{k: v.cpu() for k, v in m.state_dict().items()} for m in final],
                   "threshold": float(np.median(fold_thresholds)),
                   "size": args.size, "stride": args.stride,
                   "version": f"patch-ens{len(seeds)}-" + time.strftime("%Y%m%d")}
        Path(args.final_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.final_model)
        print(f"финальная модель: {args.final_model}, порог {payload['threshold']:.3f}, "
              f"моделей {len(final)}")


if __name__ == "__main__":
    main()
