#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Вложенная групповая валидация базовых моделей контроля качества DXA."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import importlib.metadata
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


HEADS = [
    "region", "spine_any", "spine_pos", "spine_axis", "spine_foreign",
    "hip_any", "hip_pos", "hip_roi",
]
VIOLATIONS = ["spine_pos", "spine_axis", "spine_foreign", "hip_pos", "hip_roi"]
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def seed_all(seed):
    seed = int(seed)  # numpy int64 не принимается random.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except (RuntimeError, TypeError) as exc:
        warnings.warn(f"Строгий детерминизм недоступен: {exc}")
        torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def boolean_column(series):
    values = series.fillna("").astype(str).str.strip().str.lower()
    unknown = ~values.isin(["true", "false", "1", "0", ""])
    if unknown.any():
        raise ValueError(f"Неизвестные булевы значения: {values[unknown].unique()}")
    return values.isin(["true", "1"]).to_numpy()


def binary_column(series):
    values = series.astype("string").str.strip().str.lower()
    values = values.replace({"true": "1", "false": "0", "": pd.NA})
    result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    if not np.isin(result[np.isfinite(result)], [0, 1]).all():
        raise ValueError(f"Небинарная метка в {series.name}")
    return result


def load_index(args, root):
    dtype = {"study": str, "study_n": str, "sop_uid": str}
    df = pd.read_csv(resolve(root, args.index), dtype=dtype)
    folds = pd.read_csv(resolve(root, args.folds), dtype=dtype)
    required = {
        "study_n", "study", "path", "sop_uid", "rows", "cols", "region",
        "side", "labeled", "quality_class", "y_pos", "y_axis", "y_foreign", "y_roi",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Нет колонок: {sorted(required - set(df.columns))}")
    if df["sop_uid"].isna().any() or df["sop_uid"].duplicated().any():
        raise ValueError("sop_uid должен быть заполнен и уникален")
    keys = ["study_n", "study"]
    if df[keys].isna().any().any() or folds[keys].isna().any().any():
        raise ValueError("Идентификаторы исследований не должны быть пустыми")
    if folds.duplicated(keys).any():
        raise ValueError("Дубли исследований в folds.csv")
    df = df.drop(columns=["fold"], errors="ignore").merge(
        folds[keys + ["fold"]], on=keys, how="left", validate="many_to_one",
        sort=False,
    )
    if df["fold"].isna().any() or not df["fold"].isin(range(5)).all():
        raise ValueError("Каждому изображению нужен фиксированный fold 0..4")
    df["fold"] = df["fold"].astype(int)
    if set(df["fold"]) != set(range(5)):
        raise ValueError("Нужны все пять внешних фолдов")
    if (df.groupby("study")["fold"].nunique() > 1).any():
        raise ValueError("Одно study встречается в разных фолдах")
    if not df["region"].isin(["spine", "hip"]).all():
        raise ValueError("Допустимые области: spine, hip")
    labeled = boolean_column(df["labeled"])
    y = np.full((len(df), len(HEADS)), np.nan, dtype=np.float32)
    y[:, 0] = (df["region"] == "hip").astype(float)
    mapping = {"any": "quality_class", "pos": "y_pos", "axis": "y_axis",
               "foreign": "y_foreign", "roi": "y_roi"}
    for j, head in enumerate(HEADS[1:], 1):
        region, target = head.split("_", 1)
        active = labeled & (df["region"].to_numpy() == region)
        values = binary_column(df[mapping[target]])
        if not np.isfinite(values[active]).all():
            raise ValueError(f"Пропущена применимая метка {head}")
        y[active, j] = values[active]
    for region, columns in [("spine", [2, 3, 4]), ("hip", [6, 7])]:
        active = labeled & (df["region"].to_numpy() == region)
        any_column = HEADS.index(region + "_any")
        if not np.array_equal(y[active, any_column], y[active][:, columns].max(axis=1)):
            raise ValueError(f"quality_class не совпадает с OR критериев: {region}")
    return df, y


def load_images(df, root, size=384, top_fraction=0.4):
    """Изображения, маска валидной области (не паддинг), верхняя полоса и простые признаки.

    Ресайз до 256 срезал тонкие дуги (косточки бюстгальтера шириной 1–3 px) — длинная
    сторона по умолчанию 384. Маска нужна, чтобы сеть отличала настоящий край кадра
    от нулевого паддинга: «обрезанное поле сканирования» — это метка по краям.
    """
    images = np.zeros((len(df), size, size), dtype=np.float32)
    masks = np.zeros((len(df), size, size), dtype=np.float32)
    tops = np.zeros((len(df), size, size), dtype=np.float32)
    features = np.empty((len(df), 14), dtype=np.float32)
    for i, row in df.iterrows():
        ds = pydicom.dcmread(resolve(root, row["path"]))
        raw = np.asarray(ds.pixel_array).squeeze()
        if raw.ndim != 2:
            raise ValueError(f"Ожидалось двумерное изображение: {row['path']}")
        raw = raw.astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            raw = float(2 ** int(ds.BitsStored) - 1) - raw
        if not np.isfinite(raw).all():
            raise ValueError(f"Некорректные пиксели: {row['path']}")
        if row["region"] == "hip" and str(row["side"]).strip().upper() == "L":
            raw = raw[:, ::-1].copy()
        h, w = raw.shape
        if (h, w) != (int(row["rows"]), int(row["cols"])):
            raise ValueError(f"Размер DICOM не совпадает с индексом: {row['path']}")
        lo, hi = np.percentile(raw, [0.5, 99.5])
        image = np.clip((raw - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(raw)
        image = image.astype(np.float32)
        # Признаки считаются до паддинга, на нормированной яркости.
        total = float(image.sum(dtype=np.float64))
        cx = float(image.sum(axis=0) @ np.linspace(0, 1, w) / total) if total else 0.5
        cy = float(image.sum(axis=1) @ np.linspace(0, 1, h) / total) if total else 0.5
        p5, p25, p75, p95 = np.percentile(image, [5, 25, 75, 95])
        features[i] = [
            h, w, h / w, np.mean(image != 0), image.mean(), np.median(image),
            image.std(), p5, p25, p75, p95, cx, cy,
            np.mean(image > 0.8 * image.max()),
        ]
        images[i], masks[i] = fit_canvas(image, size)
        # Верхняя полоса в собственном масштабе: там косточки бюстгальтера и застёжки.
        strip = image[:max(1, int(round(h * top_fraction)))]
        tops[i], _ = fit_canvas(strip, size)
    return images, masks, tops, features


def fit_canvas(image, size):
    """Вписывает изображение в квадрат size×size с сохранением пропорций; вторым — маска."""
    h, w = image.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = F.interpolate(
        torch.from_numpy(np.ascontiguousarray(image))[None, None], size=(nh, nw),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    canvas = np.zeros((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    mask[top:top + nh, left:left + nw] = 1.0
    return canvas, mask


LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]


class ImageDataset(Dataset):
    def __init__(self, images, indices, targets=None, augment=False, masks=None, channels="gray"):
        self.images, self.indices = images, np.asarray(indices)
        self.targets, self.augment = targets, augment
        self.masks, self.channels = masks, channels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = torch.from_numpy(self.images[i]).unsqueeze(0)
        if self.augment:
            x = x.clone()
            support = x != 0
            if random.random() < 0.8:
                x = ((x - x.mean()) * random.uniform(0.9, 1.1) + x.mean())
                x = x * random.uniform(0.9, 1.1)
                x = x.clamp(0, 1).pow(random.uniform(0.9, 1.1))
            if random.random() < 0.3:
                x = x + torch.randn_like(x) * random.uniform(0.002, 0.015)
            if random.random() < 0.2:
                sigma = random.uniform(0.3, 0.7)
                grid = torch.arange(-2, 3, dtype=x.dtype)
                kernel = torch.exp(-grid.square() / (2 * sigma ** 2))
                kernel = kernel / kernel.sum()
                kernel = (kernel[:, None] * kernel[None, :])[None, None]
                x = F.conv2d(x[None], kernel, padding=2)[0]
            x = x.clamp(0, 1) * support
        if self.channels == "physical":
            # Канал высоких частот подчёркивает тонкие дуги, канал маски отделяет
            # настоящий край кадра от паддинга; статистики ImageNet тут не применимы.
            high = F.conv2d(x[None], LAPLACIAN, padding=1)[0]
            mask = torch.from_numpy(self.masks[i])[None] if self.masks is not None else torch.ones_like(x)
            x = torch.cat([(x - 0.485) / 0.229, (high * 5).clamp(-3, 3), mask - 0.5], dim=0)
        else:
            x = (x.expand(3, -1, -1) - MEAN) / STD
        if self.targets is None:
            return x
        return x, torch.from_numpy(self.targets[i])


def make_loader(images, indices, args, seed, targets=None, train=False, masks=None, channels="gray"):
    # При spawn не передаём рабочим процессам копии всего кэша.
    import multiprocessing as mp
    workers = args.workers
    if workers and mp.get_start_method() != "fork":
        workers = 0
    return DataLoader(
        ImageDataset(images, indices, targets, train, masks, channels),
        batch_size=16, shuffle=train, num_workers=workers,
        pin_memory=args.device == "cuda", worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )


def create_encoder(pretrained):
    import timm
    try:
        return timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(f"Предобученные веса недоступны; случайная инициализация: {exc}")
        return timm.create_model(
            "resnet18", pretrained=False, num_classes=0, global_pool="avg"
        )


def embeddings(encoder, images, args, masks=None, channels="gray"):
    encoder = encoder.to(args.device).eval()
    result = np.empty((len(images), encoder.num_features), dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for x in make_loader(images, np.arange(len(images)), args, args.seed,
                             masks=masks, channels=channels):
            z = encoder(x.to(args.device, non_blocking=True)).float().cpu().numpy()
            result[offset:offset + len(z)] = z
            offset += len(z)
    encoder.cpu()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return result


def choose_threshold(y, probabilities, min_specificity=0.0):
    """Порог по максимуму F1; при min_specificity — только среди порогов с такой специфичностью.

    Порог «по максимуму F1» на редких головах ставит предупреждение почти всем снимкам
    (в аудите у B0 на hip_pos 102 ложных срабатывания на 114 отрицательных), поэтому
    в сервисе выбор ограничивается снизу по специфичности.
    """
    good = np.isfinite(y) & np.isfinite(probabilities)
    y, p = y[good].astype(int), probabilities[good]
    if not len(y):
        return 0.5, 0.0
    if not y.sum():
        return float(np.nextafter(1.0, np.inf)), 0.0
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p) - 1]
    tp = np.cumsum(y)[ends]
    scores = 2 * tp / (ends + 1 + y.sum())
    negatives = len(y) - y.sum()
    if min_specificity > 0 and negatives:
        specificity = 1 - (ends + 1 - tp) / negatives
        allowed = np.flatnonzero(specificity >= min_specificity - 1e-12)
        if len(allowed):
            masked = np.full_like(scores, -1.0)
            masked[allowed] = scores[allowed]
            scores = masked
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    thresholds = p[ends[best]]
    k = best[np.argmin(np.abs(thresholds - 0.5))]
    return float(p[ends[k]]), float(scores[k])


def fit_logistic(x, y, train, test, c, seed):
    train = train[np.isfinite(y[train])]
    if not len(train):
        warnings.warn("В обучающей части нет меток; используется вероятность 0.5")
        return np.full(len(test), 0.5)
    unique = np.unique(y[train])
    if len(unique) == 1:
        return np.full(len(test), float(unique[0]))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                           max_iter=3000, random_state=seed),
    )
    model.fit(x[train], y[train])
    return model.predict_proba(x[test])[:, 1]


def nested_logistic(x, y, folds, args, by_head=None):
    """by_head: голова → своя матрица признаков (остальные головы берут общую x)."""
    probabilities = np.full(y.shape, np.nan, dtype=np.float64)
    decisions = np.full(y.shape, np.nan)
    selection = {}
    for outer in range(5):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        selection[str(outer)] = {}
        for j, head in enumerate(HEADS):
            xh = (by_head or {}).get(head, x)
            best = None
            for c in CS:
                inner = np.full(len(y), np.nan)
                for val_fold in sorted(set(folds[train])):
                    tr = train[folds[train] != val_fold]
                    va = train[folds[train] == val_fold]
                    inner[va] = fit_logistic(xh, y[:, j], tr, va, c, args.seed)
                threshold, score = choose_threshold(y[train, j], inner[train], args.min_specificity)
                if best is None or score > best[0] + 1e-12:
                    best = (score, c, threshold)
            _, c, threshold = best
            p = fit_logistic(xh, y[:, j], train, test, c, args.seed)
            probabilities[test, j], decisions[test, j] = p, p >= threshold
            selection[str(outer)][head] = {
                "threshold": threshold, "C": c, "inner_oof_f1": best[0],
            }
        print(f"  Внешний фолд {outer}: готово", flush=True)
    return probabilities, decisions, selection


class MultiHeadCNN(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.region = nn.Linear(encoder.num_features, 2)
        self.quality = nn.Linear(encoder.num_features, len(HEADS) - 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.region(z), self.quality(z)


def cnn_loss(outputs, targets):
    region, quality = outputs
    region_target = F.one_hot(targets[:, 0].long(), 2).float()
    loss = F.binary_cross_entropy_with_logits(region, region_target)
    mask = torch.isfinite(targets[:, 1:])
    safe_targets = torch.nan_to_num(targets[:, 1:], nan=0.0)
    losses = F.binary_cross_entropy_with_logits(quality, safe_targets, reduction="none")
    return loss + (losses * mask).sum() / mask.sum().clamp_min(1)


def train_cnn(template, images, y, train, test, args, seed, masks=None):
    seed = int(seed)  # numpy int64 не принимается random.seed и torch.Generator
    seed_all(seed)
    model = MultiHeadCNN(copy.deepcopy(template)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = args.device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loader = make_loader(images, train, args, seed, targets=y, train=True,
                         masks=masks, channels=args.channels)
    for _ in range(args.epochs):
        model.train()
        for x, target in loader:
            x = x.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # Повтор шага сохраняет состояние BatchNorm и генераторов.
            bn_state = {n: b.clone() for n, b in model.named_buffers()}
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if amp else None
            try:
                with torch.autocast(device_type=args.device, enabled=amp):
                    loss = cnn_loss(model(x), target)
                scaler.scale(loss).backward()
            except RuntimeError as exc:
                if "determin" not in str(exc).lower():
                    raise
                warnings.warn(f"Операция без детерминированного ядра: {exc}")
                torch.use_deterministic_algorithms(True, warn_only=True)
                with torch.no_grad():
                    for n, b in model.named_buffers():
                        b.copy_(bn_state[n])
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device, enabled=amp):
                    loss = cnn_loss(model(x), target)
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for x in make_loader(images, test, args, seed, masks=masks, channels=args.channels):
            with torch.autocast(device_type=args.device, enabled=amp):
                region, quality = model(x.to(args.device, non_blocking=True))
            # Две независимые BCE-компоненты региона нормируются через softmax.
            p = torch.cat(
                [region.float().softmax(1)[:, 1:2], quality.float().sigmoid()], dim=1
            )
            predictions.append(p.cpu().numpy())
    result = np.concatenate(predictions).astype(np.float64)
    del model, optimizer, scheduler, scaler, loader
    if amp:
        torch.cuda.empty_cache()
    return result


def nested_cnn(template, images, y, folds, args, masks=None):
    probabilities = np.full(y.shape, np.nan, dtype=np.float64)
    decisions = np.full(y.shape, np.nan)
    selection = {}
    for outer in range(5):
        train, test = np.flatnonzero(folds != outer), np.flatnonzero(folds == outer)
        inner = np.full(y.shape, np.nan, dtype=np.float64)
        for val_fold in sorted(set(folds[train])):
            tr, va = train[folds[train] != val_fold], train[folds[train] == val_fold]
            inner[va] = train_cnn(
                template, images, y, tr, va, args, args.seed + 100 * outer + val_fold, masks
            )
            print(f"  Внешний {outer}, внутренний {val_fold}: готово", flush=True)
        selection[str(outer)] = {}
        thresholds = []
        for j, head in enumerate(HEADS):
            threshold, score = choose_threshold(y[train, j], inner[train, j], args.min_specificity)
            thresholds.append(threshold)
            selection[str(outer)][head] = {
                "threshold": threshold, "inner_oof_f1": score,
            }
        probabilities[test] = train_cnn(
            template, images, y, train, test, args, args.seed + 100 * outer + 99, masks
        )
        decisions[test] = probabilities[test] >= np.asarray(thresholds)
        print(f"  Внешний фолд {outer}: готово", flush=True)
    return probabilities, decisions, selection


def metric_values(y, p, pred, weights=None):
    weights = np.ones(len(y)) if weights is None else weights
    pos, neg = weights[y == 1].sum(), weights[y == 0].sum()
    tp = weights[(y == 1) & (pred == 1)].sum()
    tn = weights[(y == 0) & (pred == 0)].sum()
    fp, fn = neg - tn, pos - tp
    result = {
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "sensitivity": float(tp / pos) if pos else np.nan,
        "specificity": float(tn / neg) if neg else np.nan,
        "roc_auc": np.nan, "ap": np.nan,
    }
    if pos and neg:
        keep = weights > 0
        result["roc_auc"] = float(roc_auc_score(y[keep], p[keep], sample_weight=weights[keep]))
        result["ap"] = float(average_precision_score(
            y[keep], p[keep], sample_weight=weights[keep]
        ))
    return result


def interval(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    bounds = np.percentile(values[valid], [2.5, 97.5]).tolist() if valid.any() else [None, None]
    return {"ci95": bounds, "bootstrap_skipped_fraction": float(1 - valid.mean())}


def evaluate(df, y, probabilities, decisions, seed):
    groups, studies = pd.factorize(df["study"], sort=True)
    tasks = {}
    for j, head in enumerate(HEADS):
        active = np.isfinite(y[:, j])
        tasks[head] = (
            y[active, j].astype(int), probabilities[active, j],
            decisions[active, j].astype(int), groups[active],
        )
    # Общая оценка any использует истинную область, без каскада региональной головы.
    column = np.where(df["region"].to_numpy() == "spine", 1, 5)
    indices = np.arange(len(df))
    truth = y[indices, column]
    active = np.isfinite(truth)
    tasks["any_all"] = (
        truth[active].astype(int), probabilities[indices, column][active],
        decisions[indices, column][active].astype(int), groups[active],
    )
    results, replicas = {}, {}
    for head, (target, p, pred, _) in tasks.items():
        point = metric_values(target, p, pred)
        results[head] = {"n": len(target), "n_positive": int(target.sum()),
                         "n_negative": int(len(target) - target.sum())}
        results[head].update({name: {"value": value} for name, value in point.items()})
        replicas[head] = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    macro = []
    for _ in range(2000):
        counts = np.bincount(rng.integers(len(studies), size=len(studies)),
                             minlength=len(studies))
        f1s = {}
        for head, (target, p, pred, group) in tasks.items():
            values = metric_values(target, p, pred, counts[group])
            for name, value in values.items():
                replicas[head][name].append(value)
            f1s[head] = values["f1"]
        macro.append(np.mean([f1s[h] for h in VIOLATIONS]))
    for head in tasks:
        for name, values in replicas[head].items():
            results[head][name].update(interval(values))
    results["macro_violations"] = {
        "f1": {
            "value": float(np.mean([results[h]["f1"]["value"] for h in VIOLATIONS])),
            **interval(macro),
        },
        "heads": VIOLATIONS,
    }
    return results


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def environment():
    import platform
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["numpy", "pandas", "pydicom", "scikit-learn", "torch", "torchvision", "timm"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["cuda"] = torch.version.cuda
    versions["cudnn"] = torch.backends.cudnn.version()
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/index/images.csv")
    parser.add_argument("--folds", default="experiments/folds.csv")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", default="experiments/results")
    parser.add_argument("--models", default="b0,b1,b2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=384, help="длинная сторона после ресайза")
    parser.add_argument("--min-specificity", type=float, default=0.0,
                        help="нижняя граница специфичности при выборе порога (0 = только F1)")
    parser.add_argument("--top-fraction", type=float, default=0.4,
                        help="доля верхних строк для ветви «посторонние предметы»")
    parser.add_argument("--channels", choices=["gray", "physical"], default="physical",
                        help="physical: снимок + лапласиан + маска валидной области (для B2)")
    parser.add_argument("--foreign-branch", dest="foreign_branch", action="store_true", default=True,
                        help="B1: для головы «предметы» добавить эмбеддинг верхней полосы")
    parser.add_argument("--no-foreign-branch", dest="foreign_branch", action="store_false")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = parser.parse_args()
    models = list(dict.fromkeys(x.strip().lower() for x in args.models.split(",")))
    if not models or set(models) - {"b0", "b1", "b2"}:
        parser.error("--models: допустимы b0,b1,b2")
    if args.epochs < 1 or args.workers < 0:
        parser.error("--epochs >= 1, --workers >= 0")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    root = Path(args.root).resolve()
    out = resolve(root, args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    df, y = load_index(args, root)
    started = time.perf_counter()
    images, masks, tops, simple_features = load_images(df, root, args.size, args.top_fraction)
    preprocessing_seconds = time.perf_counter() - started
    folds = df["fold"].to_numpy()
    template, encoder_seconds, pretrained_loaded = None, 0.0, False
    if set(models) & {"b1", "b2"}:
        started = time.perf_counter()
        seed_all(args.seed)
        # Один неизменяемый шаблон на CPU для всех независимых запусков.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            template = create_encoder(args.pretrained).cpu()
        for warning in caught:
            warnings.warn(str(warning.message))
        pretrained_loaded = args.pretrained and not any(
            "Предобученные веса недоступны" in str(w.message) for w in caught
        )
        encoder_seconds = time.perf_counter() - started
    summary = []
    for name in models:
        print(f"{name.upper()} ({args.device})", flush=True)
        started = time.perf_counter()
        if name == "b0":
            p, pred, selection = nested_logistic(simple_features, y, folds, args)
        elif name == "b1":
            x = embeddings(template, images, args)
            by_head = {}
            if args.foreign_branch:
                # Верхняя полоса в своём масштабе: косточки бюстгальтера видны только там.
                xt = embeddings(template, tops, args)
                by_head["spine_foreign"] = np.concatenate([x, xt], axis=1)
            p, pred, selection = nested_logistic(x, y, folds, args, by_head)
            del x, by_head
        else:
            p, pred, selection = nested_cnn(template, images, y, folds, args, masks)
        training_seconds = time.perf_counter() - started
        if not np.isfinite(p).all():
            raise RuntimeError(f"{name}: не все внешние предсказания заполнены")
        # Неприменимые головы пусты; неизвестная истина не скрывает предсказание.
        for j, head in enumerate(HEADS[1:], 1):
            applicable = df["region"].to_numpy() == head.split("_", 1)[0]
            p[~applicable, j], pred[~applicable, j] = np.nan, np.nan
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        oof = df[["sop_uid", "study", "study_n", "region", "fold"]].copy()
        for j, head in enumerate(HEADS):
            oof[head + "_prob"] = p[:, j]
            oof[head + "_pred"] = pd.array(pred[:, j], dtype="Int64")
            oof[head + "_true"] = pd.array(y[:, j], dtype="Int64")
        oof.to_csv(folder / "oof.csv", index=False)
        started = time.perf_counter()
        metrics = evaluate(df, y, p, pred, args.seed)
        payload = {
            "model": name, "metrics": metrics, "outer_fold_selection": selection,
            "training_seconds": training_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "encoder_initialization_seconds": encoder_seconds if name != "b0" else 0,
            "evaluation_seconds": time.perf_counter() - started,
            "environment": environment(), "arguments": vars(args),
            "pretrained_loaded": pretrained_loaded if name != "b0" else None,
            "protocol": {
                "outer_folds": 5, "inner_folds": 4, "bootstrap_replicates": 2000,
                "bootstrap_cluster": "study", "region_positive": "hip",
                "threshold_rule": "probability >= threshold",
                "C_selection": "максимум внутреннего OOF F1 с подбором порога; при равенстве меньше C",
                "any_all": "объединение региональных any по истинной области",
                "undefined_f1": 0.0,
                "auc_ap": "не определены без одного из классов",
                "ci_scope": "кластерный бутстрэп фиксированных OOF, без переобучения",
                "features": "яркостные признаки до паддинга, после нормировки",
            },
        }
        with (folder / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(clean_json(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        for head, values in metrics.items():
            row = {"model": name, "head": head, "training_seconds": training_seconds}
            for key in ["n", "n_positive", "n_negative"]:
                row[key] = values.get(key)
            for metric in ["f1", "roc_auc", "ap", "sensitivity", "specificity"]:
                item = values.get(metric, {})
                row[metric] = item.get("value")
                ci = item.get("ci95", [None, None])
                row[metric + "_ci_low"], row[metric + "_ci_high"] = ci
                row[metric + "_bootstrap_skipped_fraction"] = item.get(
                    "bootstrap_skipped_fraction"
                )
            summary.append(row)
        del p, pred, oof
    table = pd.DataFrame(summary)
    table.to_csv(out / "summary.csv", index=False)
    print(table[["model", "head", "n_positive", "f1", "roc_auc", "ap",
                 "roc_auc_bootstrap_skipped_fraction"]].to_string(
        index=False, float_format=lambda value: f"{value:.3f}", na_rep="—"
    ))


if __name__ == "__main__":
    main()
