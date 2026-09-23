#!/usr/bin/env python3
"""Предобучение сервисного CNN-энкодера на синтетической ротации бедра."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_baselines import create_encoder, fit_canvas, make_loader, seed_all


class LazyCanvas:
    """Array-like адаптер: canvas строится только для запрошенного наблюдения."""

    def __init__(self, images, size, mask=False):
        self.images = images
        self.size = size
        self.mask = mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = fit_canvas(self.images[index], self.size)
        return mask if self.mask else image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="NPZ с images, theta, side, ct_id")
    parser.add_argument("--out", required=True, help="выходной .pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--channels", choices=["gray", "physical"], default="physical")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 1 or args.workers < 0:
        parser.error("--epochs и --size должны быть >= 1, --workers >= 0")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA недоступна")
    return args


def main():
    args = parse_args()
    seed_all(args.seed)
    with np.load(args.data, allow_pickle=False) as data:
        required = {"images", "theta", "side", "ct_id"}
        if not required.issubset(data.files):
            raise ValueError(f"Нет ключей NPZ: {sorted(required - set(data.files))}")
        images = np.asarray(data["images"], dtype=np.float32)
        theta = np.asarray(data["theta"], dtype=np.float32)
        side = np.asarray(data["side"])
        ct_id = np.asarray(data["ct_id"])
    if images.ndim != 3 or images.shape[1:] != (265, 300):
        raise ValueError("images должны иметь форму [N, 265, 300]")
    if any(len(value) != len(images) for value in (theta, side, ct_id)):
        raise ValueError("images, theta, side и ct_id должны иметь одинаковую длину")
    if not np.isfinite(images).all() or not np.isfinite(theta).all():
        raise ValueError("images и theta должны содержать только конечные значения")
    if images.size and (images.min() < 0 or images.max() > 1):
        raise ValueError("images должны быть нормированы в [0, 1]")

    ct_values = np.unique(ct_id)
    if len(ct_values) < 2:
        raise ValueError("Для группового разбиения нужны как минимум два ct_id")
    shuffled = np.random.default_rng(args.seed).permutation(ct_values)
    n_val_ct = min(len(ct_values) - 1, max(1, int(round(0.1 * len(ct_values)))))
    val_ct = shuffled[:n_val_ct]
    is_val = np.isin(ct_id, val_ct)
    train_indices = np.flatnonzero(~is_val)
    val_indices = np.flatnonzero(is_val)
    targets = (theta / 30.0).reshape(-1, 1).astype(np.float32)
    canvases = LazyCanvas(images, args.size)
    masks = LazyCanvas(images, args.size, mask=True)

    encoder = create_encoder(pretrained=True).to(args.device)
    head = nn.Linear(encoder.num_features, 1).to(args.device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()
    amp = args.device == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_loader = make_loader(
        canvases, train_indices, args, args.seed, targets=targets, train=True,
        masks=masks, channels=args.channels,
    )

    val_mae = val_r = float("nan")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        for x, target in train_loader:
            x = x.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, enabled=amp):
                prediction = head(encoder(x))
                loss = loss_fn(prediction, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        encoder.eval()
        head.eval()
        predictions = []
        with torch.inference_mode():
            loader = make_loader(
                canvases, val_indices, args, args.seed, masks=masks,
                channels=args.channels,
            )
            for x in loader:
                with torch.autocast(device_type=args.device, enabled=amp):
                    predictions.append(head(encoder(x.to(args.device))).float().cpu().numpy())
        predicted_theta = np.concatenate(predictions).ravel() * 30.0
        true_theta = theta[val_indices]
        val_mae = float(np.mean(np.abs(predicted_theta - true_theta)))
        val_r = float(np.corrcoef(predicted_theta, true_theta)[0, 1]) \
            if len(true_theta) > 1 and predicted_theta.std() > 0 and true_theta.std() > 0 \
            else float("nan")
        print(f"epoch {epoch:02d}: val MAE={val_mae:.3f} deg, r={val_r:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder": {name: value.detach().cpu() for name, value in encoder.state_dict().items()},
        "val_mae": val_mae, "val_r": val_r,
        "n_train": len(train_indices), "n_ct": len(ct_values),
    }, out)


if __name__ == "__main__":
    main()
