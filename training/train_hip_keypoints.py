import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pydicom
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxaqc.nets import ConvBlock, KeypointNet  # noqa: E402


POINTS = ["H", "B1", "B2", "T", "D", "D2"]
# У каждой точки бедра своё состояние, а не три общих признака кадра, как
# на пояснице. Головы присутствия обучаются на том же наборе состояний.
POINT_STATES = ["visible", "uncertain", "not_visible", "out_of_frame"]
STATE_NAMES = {name: POINT_STATES for name in POINTS}


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def vertebra_point(points, name):
    """Центр тела позвонка: ключ <V>_center; старый формат (<V>_top/_bottom) — середина."""
    c = points.get(f"{name}_center") or {}
    if "x" in c and "y" in c:
        return float(c["x"]), float(c["y"])
    t, b = points.get(f"{name}_top") or {}, points.get(f"{name}_bottom") or {}
    if "x" in t and "x" in b:
        return (float(t["x"]) + float(b["x"])) / 2, (float(t["y"]) + float(b["y"])) / 2
    return None


def read_annotations(paths):
    raw = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        for uid, item in obj.get("images", {}).items():
            raw.setdefault(uid, []).append(item)

    result = {}
    for uid, items in raw.items():
        # Схема dxa-hip-points/1: завершённые снимки помечены state == "done".
        # Скрытые повторы (uid#2) в обучение не берём — это тот же снимок.
        done = [x for x in items if x.get("state") == "done"]
        if not done or uid.endswith("#2"):
            continue

        rec = {"points": {}}
        for name in POINTS:
            coords, states = [], []
            for item in done:
                point = (item.get("points") or {}).get(name) or {}
                state = point.get("state")
                if state:
                    states.append(state)
                # Координату берём только у состояний, где она разрешена:
                # not_visible и out_of_frame её иметь не должны.
                if state in ("visible", "uncertain") and "x" in point and "y" in point:
                    coords.append((float(point["x"]), float(point["y"])))
            # Одиннадцать снимков размечены обоими: усредняем, как на пояснице.
            if coords:
                rec["points"][name] = {
                    "x": float(np.mean([c[0] for c in coords])),
                    "y": float(np.mean([c[1] for c in coords])),
                    "state": states[0] if states else "visible",
                }
            else:
                rec["points"][name] = {"state": states[0] if states else "not_visible"}
        result[uid] = rec
    return result


def load_image(path, size):
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        bits = int(getattr(ds, "BitsStored", 16))
        arr = (2 ** bits - 1) - arr
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = arr.shape[:2]
    scale = float(size) / max(h, w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    x = torch.from_numpy(arr)[None, None]
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)[0, 0]
    canvas = torch.zeros((size, size), dtype=torch.float32)
    oy = (size - nh) // 2
    ox = (size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = x
    return canvas, scale, float(ox), float(oy), w, h


def augment_image(x):
    if random.random() < 0.8:
        x = x * random.uniform(0.85, 1.15)
    if random.random() < 0.8:
        mean = x.mean()
        x = (x - mean) * random.uniform(0.85, 1.15) + mean
    if random.random() < 0.7:
        x = torch.clamp(x, 1e-5, 1)
        x = x.pow(random.uniform(0.8, 1.2))
    if random.random() < 0.25:
        x = x + torch.randn_like(x) * random.uniform(0.005, 0.03)
    if random.random() < 0.2:
        k = random.choice([3, 5])
        x = F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]
    return x.clamp(0, 1)


GRIDS = {}


def grid(size):
    """Сетка координат heatmap'а: строилась заново на каждую точку и упиралась в CPU (GPU простаивала)."""
    if size not in GRIDS:
        GRIDS[size] = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                                     torch.arange(size, dtype=torch.float32), indexing="ij")
    return GRIDS[size]


class KeypointDataset(Dataset):
    def __init__(self, rows, annotations, image_cache, size, train=False):
        self.rows = rows.reset_index(drop=True)
        self.annotations = annotations
        self.image_cache = image_cache
        self.size = size
        self.train = train
        self.hm_size = size // 2

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        uid = str(row.sop_uid)
        image, scale, ox, oy, _, _ = self.image_cache[uid]
        image = augment_image(image.clone()) if self.train else image.clone()
        image = image[None].repeat(3, 1, 1)

        ann = self.annotations.get(uid, {})
        target_xy = torch.zeros((len(POINTS), 2), dtype=torch.float32)
        mask = torch.zeros(len(POINTS), dtype=torch.float32)
        true_xy = torch.full((len(POINTS), 2), float("nan"))
        for j, name in enumerate(POINTS):
            p = (ann.get("points") or {}).get(name, {})
            if "x" in p and "y" in p:
                x = float(p["x"]) * scale + ox
                y = float(p["y"]) * scale + oy
                true_xy[j] = torch.tensor([float(p["x"]), float(p["y"])])
                # Доли входного квадрата: карта меньше входа, но координата нормирована одинаково.
                target_xy[j] = torch.tensor([x / self.size, y / self.size])
                mask[j] = 1.0

        # По одной голове состояния на каждую точку: видимость малого вертела
        # сама по себе может нести сигнал, поэтому её предсказываем явно.
        labels = []
        for name in POINTS:
            state = (ann.get("points") or {}).get(name, {}).get("state", "")
            labels.append(POINT_STATES.index(state) if state in POINT_STATES else -1)

        return {
            "image": image,
            "target_xy": target_xy,
            "mask": mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "true_xy": true_xy,
            "uid": uid,
        }


def target_kl(heatmaps, target_xy, sigma_norm):
    """KL(целевой гауссиан ‖ предсказанное распределение): центр верный, но карта размазана — штрафуем."""
    b, c, h, w = heatmaps.shape
    log_prob = (heatmaps.reshape(b, c, -1)).log_softmax(-1).reshape(b, c, h, w)
    ys = torch.linspace(0, 1, h, device=heatmaps.device).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w, device=heatmaps.device).view(1, 1, 1, w)
    d2 = (xs - target_xy[..., 0:1, None]) ** 2 + (ys - target_xy[..., 1:2, None]) ** 2
    g = torch.exp(-d2 / (2 * sigma_norm ** 2))
    g = g / g.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
    return (g * (torch.log(g.clamp_min(1e-12)) - log_prob)).sum(dim=(-1, -2))


def soft_argmax(heatmaps, temperature=1.0):
    """Пространственный softmax → ожидаемые координаты в долях карты [0, 1] и мера разброса.

    MSE по разреженному гауссиану на 60 обучающих снимках не сходится (карты выходят
    размытыми, медианная ошибка была ~100 мм). Здесь градиент идёт прямо в координату.
    """
    b, c, h, w = heatmaps.shape
    flat = (heatmaps.reshape(b, c, -1) / temperature).softmax(-1)
    prob = flat.reshape(b, c, h, w)
    ys = torch.linspace(0, 1, h, device=heatmaps.device).view(1, 1, h, 1)
    xs = torch.linspace(0, 1, w, device=heatmaps.device).view(1, 1, 1, w)
    ex = (prob * xs).sum(dim=(-1, -2))
    ey = (prob * ys).sum(dim=(-1, -2))
    spread = ((prob * (xs - ex[..., None, None]) ** 2).sum(dim=(-1, -2))
              + (prob * (ys - ey[..., None, None]) ** 2).sum(dim=(-1, -2)))
    return torch.stack([ex, ey], dim=-1), spread


def train_one(model, train_loader, val_loader, device, args):
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": encoder, "lr": args.encoder_lr},
                             {"params": rest, "lr": args.lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            target_xy = batch["target_xy"].to(device)   # доли карты [0, 1]
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                pred_hm, presence, regressed, pred_cls = model(image)
                coords, spread = soft_argmax(pred_hm)
                if args.head == "regress":
                    coords, spread = regressed, torch.zeros_like(spread)
                error = (coords - target_xy).abs().sum(-1)
                point_loss = (error * mask).sum() / mask.sum().clamp_min(1)
                # Штраф за размытость — только там, где точка есть.
                spread_loss = (spread * mask).sum() / mask.sum().clamp_min(1)
                if args.head == "softargmax":
                    kl = target_kl(pred_hm, target_xy, args.sigma / args.size)
                    map_loss = (kl * mask).sum() / mask.sum().clamp_min(1)
                else:
                    map_loss = torch.zeros((), device=device)
                presence_loss = F.binary_cross_entropy_with_logits(presence, mask)
                cls_loss = 0
                ncls = 0
                for k in range(3):
                    valid = labels[:, k] >= 0
                    if valid.any():
                        cls_loss = cls_loss + F.cross_entropy(pred_cls[valid, k], labels[valid, k])
                        ncls += 1
                cls_loss = cls_loss / max(ncls, 1)
                loss = (args.heatmap_weight * point_loss + args.spread_weight * spread_loss
                        + args.map_weight * map_loss + args.presence_weight * presence_loss
                        + args.class_weight * cls_loss)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sch.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                target_xy = batch["target_xy"].to(device)
                mask = batch["mask"].to(device)
                pred_hm, _, regressed, _ = model(image)
                coords, _ = soft_argmax(pred_hm)
                if args.head == "regress":
                    coords = regressed
                error = ((coords - target_xy).abs().sum(-1) * mask).sum() / mask.sum().clamp_min(1)
                vals.append(float(error))
        score = float(np.mean(vals)) if vals else float("inf")
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def line_angle(points):
    if len(points) < 2:
        return float("nan")
    p = np.asarray(points, dtype=float)
    a = np.polyfit(p[:, 1], p[:, 0], 1)[0]
    return math.degrees(math.atan(a))


def robust_angle(points):
    if len(points) < 2:
        return float("nan")
    p = np.asarray(points, dtype=float)
    y, x = p[:, 1], p[:, 0]
    A = np.column_stack([y, np.ones(len(y))])
    w = np.ones(len(y))
    beta = np.linalg.lstsq(A, x, rcond=None)[0]
    for _ in range(5):
        r = x - A @ beta
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-8))
        beta = np.linalg.lstsq(A * w[:, None], x * w, rcond=None)[0]
    return math.degrees(math.atan(beta[0]))


def geometric_features(xy):
    """Признаки ротации бедра по шести точкам. Порядок POINTS: H, B1, B2, T, D, D2.

    Главный признак — `f_simple` из experiments/hip_rotation_full_2026-09-21.md:
    отстояние малого вертела в медиальную сторону от оси диафиза, в ширинах шейки.
    Медиальное направление задаётся анатомией (головка всегда медиальнее диафиза),
    а не стороной из метаданных и не подбором знака по результату.
    """
    def point(i):
        q = xy[i] if i < len(xy) else None
        if q is None or not np.all(np.isfinite(q)):
            return None
        return np.array(q, dtype=float) * np.array([0.600, 0.606])

    H, B1, B2, T, D, D2 = (point(i) for i in range(6))
    out = {
        "f_simple": float("nan"),
        "f_with_axis": float("nan"),
        "neck_width_mm": float("nan"),
        "medial_sign": float("nan"),
        "n_points": sum(point(i) is not None for i in range(6)),
    }
    if B1 is not None and B2 is not None:
        out["neck_width_mm"] = float(np.linalg.norm(B1 - B2))
    if H is None or D is None:
        return out
    medial = 1.0 if H[0] > D[0] else -1.0
    out["medial_sign"] = medial
    width = out["neck_width_mm"]
    if T is not None and np.isfinite(width) and width > 1e-6:
        out["f_simple"] = float((T[0] - D[0]) * medial / width)
        # Вариант с осью диафиза D->D2 доступен реже: D2 часто вне кадра.
        if D2 is not None:
            axis = D2 - D
            length = np.linalg.norm(axis)
            if length > 1e-6:
                normal = np.array([-axis[1], axis[0]]) / length
                if normal[0] * medial < 0:
                    normal = -normal
                out["f_with_axis"] = float(np.dot(T - D, normal) / width)
    return out


def make_features(xy, ann):
    out = geometric_features(xy)
    for name in POINTS:
        out[name + "_state"] = (ann.get("points") or {}).get(name, {}).get("state", "")
    return out


def prepare_rows(index_path, folds_path):
    idx = pd.read_csv(index_path)
    idx = idx[idx["region"].astype(str).str.lower() == "hip"].copy()
    folds = pd.read_csv(folds_path)
    if "study_n" in folds.columns:
        idx = idx.merge(folds[["study_n", "fold"]], on="study_n", how="inner")
    else:
        idx = idx.merge(folds[["study", "fold"]], on="study", how="inner")
    return idx.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--folds", default="experiments/folds.csv")
    ap.add_argument("--annotations", nargs="+", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="experiments/results/keypoints")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=5.0, help="σ целевого пятна, px входа")
    ap.add_argument("--map-weight", type=float, default=0.02, help="вес KL к целевому пятну")
    ap.add_argument("--head", choices=["softargmax", "regress"], default="softargmax",
                    help="regress — контрольный baseline: координаты прямо из global-pool")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--point-threshold", type=float, default=0.5, help="порог головы присутствия точки")
    ap.add_argument("--only-fold", type=int, default=None, help="только один внешний фолд (дымовой тест)")
    ap.add_argument("--final-model", default=None, help="куда сохранить модель для сервиса")
    ap.add_argument("--heatmap-weight", type=float, default=1.0, help="вес L1 по координатам")
    ap.add_argument("--spread-weight", type=float, default=0.5, help="штраф за размытость карты")
    ap.add_argument("--presence-weight", type=float, default=0.3)
    ap.add_argument("--class-weight", type=float, default=1.0)
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    args = ap.parse_args()

    seed_all(args.seed)
    start_time = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    annotations = read_annotations(args.annotations)
    rows = prepare_rows(args.index, args.folds)
    image_cache = {}
    for _, r in rows.iterrows():
        uid = str(r.sop_uid)
        if uid not in image_cache:
            path = Path(str(r["path"]))
            if not path.is_absolute():
                path = Path(args.root) / path
            image_cache[uid] = load_image(str(path), args.size)

    oof_points, oof_features = [], []
    point_errors = {p: [] for p in POINTS}
    point_missing = {p: 0 for p in POINTS}
    # Ошибка признака важнее ошибки отдельных точек: именно её увидит модель.
    angle_errors = {x: [] for x in ["f_simple", "f_with_axis", "neck_width_mm"]}
    conf = {name: np.zeros((len(POINT_STATES), len(POINT_STATES)), dtype=int)
            for name in POINTS}

    folds_to_run = sorted(rows["fold"].unique())
    if args.only_fold is not None:
        folds_to_run = [f for f in folds_to_run if int(f) == args.only_fold]
    for fold in folds_to_run:
        test = rows[rows.fold == fold]
        train_pool = rows[rows.fold != fold]
        val_fold = int(train_pool.fold.min())
        train = train_pool[train_pool.fold != val_fold]
        val = train_pool[train_pool.fold == val_fold]

        tr_ds = KeypointDataset(train, annotations, image_cache, args.size, True)
        va_ds = KeypointDataset(val, annotations, image_cache, args.size, False)
        te_ds = KeypointDataset(test, annotations, image_cache, args.size, False)
        tr_loader = DataLoader(tr_ds, batch_size=8, shuffle=True, num_workers=args.workers)
        va_loader = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=args.workers)
        te_loader = DataLoader(te_ds, batch_size=8, shuffle=False, num_workers=args.workers)

        model = KeypointNet(args.pretrained, n_points=len(POINTS),
                            state_sizes=[len(POINT_STATES)] * len(POINTS)).to(device)
        model = train_one(model, tr_loader, va_loader, device, args)
        model.eval()

        with torch.no_grad():
            for batch in te_loader:
                hm, presence, regressed, logits = model(batch["image"].to(device))
                coords, _ = soft_argmax(hm)
                if args.head == "regress":
                    coords = regressed
                scores = torch.sigmoid(presence).cpu().numpy()
                coords = coords.cpu().numpy()
                cls_pred = logits.argmax(-1).cpu().numpy()
                true_xy = batch["true_xy"].numpy()

                for bi, uid in enumerate(batch["uid"]):
                    row = rows[rows.sop_uid.astype(str) == uid].iloc[0]
                    ann = annotations.get(uid, {})
                    pred_orig = []
                    true_orig = []
                    prow = {
                        "sop_uid": uid, "study": row["study"], "study_n": row["study_n"], "fold": int(fold)
                    }
                    for j, name in enumerate(POINTS):
                        if scores[bi, j] >= args.point_threshold:
                            _, scale, ox, oy, _, _ = image_cache[uid]
                            px = (coords[bi, j, 0] * args.size - ox) / scale
                            py = (coords[bi, j, 1] * args.size - oy) / scale
                            pred_orig.append([px, py])
                            prow[f"{name}_x_pred"] = px
                            prow[f"{name}_y_pred"] = py
                        else:
                            pred_orig.append(None)
                            prow[f"{name}_x_pred"] = np.nan
                            prow[f"{name}_y_pred"] = np.nan
                        prow[f"{name}_score"] = float(scores[bi, j])
                        if np.isfinite(true_xy[bi, j]).all():
                            true_orig.append(true_xy[bi, j].tolist())
                            prow[f"{name}_x_true"] = true_xy[bi, j, 0]
                            prow[f"{name}_y_true"] = true_xy[bi, j, 1]
                        else:
                            true_orig.append(None)
                            prow[f"{name}_x_true"] = np.nan
                            prow[f"{name}_y_true"] = np.nan
                        if true_orig[-1] is not None:
                            if pred_orig[-1] is None:
                                point_missing[name] += 1
                            else:
                                point_errors[name].append(
                                    float(np.linalg.norm((np.array(pred_orig[-1]) - np.array(true_orig[-1])) *
                                                         np.array([0.600, 0.606])))
                                )
                    oof_points.append(prow)

                    fp = make_features(pred_orig, ann)
                    ft = make_features(true_orig, ann)
                    fr = {"sop_uid": uid, "study": row["study"], "study_n": row["study_n"], "fold": int(fold)}
                    for k, v in fp.items():
                        fr[k + "_pred"] = v
                    for k, v in ft.items():
                        fr[k + "_true"] = v
                    fr["y_pos"] = row.get("y_pos", np.nan)
                    fr["y_axis"] = row.get("y_axis", np.nan)
                    oof_features.append(fr)

                    for key in angle_errors:
                        if np.isfinite(fp[key]) and np.isfinite(ft[key]):
                            angle_errors[key].append(abs(fp[key] - ft[key]))
                    for ki, k in enumerate(POINTS):
                        true_state = ft[k + "_state"]
                        pred_state = POINT_STATES[int(cls_pred[bi, ki])]
                        if true_state in POINT_STATES:
                            conf[k][POINT_STATES.index(true_state),
                                    POINT_STATES.index(pred_state)] += 1

    # Финальная модель для сервиса: учится на всех размеченных снимках (фолд 0 — валидация).
    if args.final_model:
        val = rows[rows.fold == rows.fold.min()]
        train = rows[rows.fold != rows.fold.min()]
        tr_loader = DataLoader(KeypointDataset(train, annotations, image_cache, args.size, True),
                               batch_size=8, shuffle=True, num_workers=args.workers)
        va_loader = DataLoader(KeypointDataset(val, annotations, image_cache, args.size, False),
                               batch_size=8, shuffle=False, num_workers=args.workers)
        model = KeypointNet(args.pretrained, n_points=len(POINTS),
                            state_sizes=[len(POINT_STATES)] * len(POINTS)).to(device)
        model = train_one(model, tr_loader, va_loader, device, args)
        target = Path(args.final_model)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "size": args.size, "points": POINTS, "states": STATE_NAMES,
                    "point_threshold": args.point_threshold, "seed": args.seed}, target)
        print(f"Финальная модель: {target}")

    pd.DataFrame(oof_points).to_csv(out_dir / "oof_points.csv", index=False)
    pd.DataFrame(oof_features).to_csv(out_dir / "oof_features.csv", index=False)

    metrics = {"points": {}, "angles": {}, "classification": {}, "training_seconds": time.time() - start_time}
    for p in POINTS:
        vals = point_errors[p]
        metrics["points"][p] = {
            "median_mm": float(np.median(vals)) if vals else None,
            "p90_mm": float(np.percentile(vals, 90)) if vals else None,
            "unpredicted_fraction": float(point_missing[p] / max(point_missing[p] + len(vals), 1)),
        }
    for k, vals in angle_errors.items():
        metrics["angles"][k] = float(np.median(vals)) if vals else None
    for k, cm in conf.items():
        metrics["classification"][k] = {
            "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
            "confusion_matrix": cm.tolist(),
        }
    metrics["environment"] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "device": str(device),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Точки:")
    for p in POINTS:
        print(p, metrics["points"][p]["median_mm"], metrics["points"][p]["p90_mm"])
    print("Углы:", {k: round(v, 3) if v is not None else None for k, v in metrics["angles"].items()})
    print("Состояния:", {k: round(v["accuracy"], 3) for k, v in metrics["classification"].items()})


if __name__ == "__main__":
    main()
