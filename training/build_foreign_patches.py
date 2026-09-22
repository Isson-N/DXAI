#!/usr/bin/env python3
"""Подготовка патчей для классификатора посторонних предметов."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


DEFAULT_ANNOTATIONS = (
    Path("data/annotations/foreign_boxes_isson.json"),
    Path("data/annotations/foreign_boxes_den.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64, help="сторона патча (по умолчанию: 64)")
    parser.add_argument(
        "--background", type=int, default=3,
        help="число фоновых патчей на отрицательный снимок (по умолчанию: 3)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/results/foreign_patches.npz"),
        help="выходной NPZ",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed генератора фоновых патчей")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size должен быть положительным")
    if args.background < 0:
        parser.error("--background не может быть отрицательным")
    return args


def load_annotations() -> dict[str, list[dict]]:
    images: dict[str, list[dict]] = defaultdict(list)
    for path in DEFAULT_ANNOTATIONS:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        schema = data.get("schema") or data.get("$schema")
        if schema is not None and schema != "dxa-foreign-boxes/1":
            raise ValueError(f"Неожиданная схема в {path}: {schema}")
        for uid, annotation in data.get("images", {}).items():
            # Суффикс обозначает скрытую копию, а не самостоятельный DICOM.
            if "#" in uid:
                continue
            images[uid].append(annotation)
    return images


def box_geometry(box: dict) -> tuple[float, float, float, tuple[float, float, float, float]]:
    shape = box.get("shape")
    if shape == "rect":
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
        return x + w / 2, y + h / 2, max(abs(w), abs(h)), (x, y, x + w, y + h)
    if shape == "line":
        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
        thickness = float(box.get("thickness", 0))
        half = thickness / 2
        return ((x1 + x2) / 2, (y1 + y2) / 2, np.hypot(x2 - x1, y2 - y1),
                (min(x1, x2) - half, min(y1, y2) - half,
                 max(x1, x2) + half, max(y1, y2) + half))
    raise ValueError(f"Неизвестная форма рамки: {shape!r}")


def read_image(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Ожидался двумерный DICOM, получена форма {arr.shape}: {path}")
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = (2 ** int(ds.BitsStored) - 1) - arr
    low, high = np.percentile(arr, (0.5, 99.5))
    if high > low:
        arr = np.clip((arr - low) / (high - low), 0, 1)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def extract_patch(image: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    # Округление центра определяет единственную целочисленную сетку без ресайза.
    x0 = int(np.floor(cx - size / 2))
    y0 = int(np.floor(cy - size / 2))
    x1, y1 = x0 + size, y0 + size
    top, left = max(0, -y0), max(0, -x0)
    bottom, right = max(0, y1 - image.shape[0]), max(0, x1 - image.shape[1])
    if top or bottom or left or right:
        # Последовательное дополнение поддерживает патчи больше самого изображения.
        padded = image
        pads = [top, bottom, left, right]
        while any(pads):
            t = min(pads[0], max(0, padded.shape[0] - 1))
            b = min(pads[1], max(0, padded.shape[0] - 1))
            l = min(pads[2], max(0, padded.shape[1] - 1))
            r = min(pads[3], max(0, padded.shape[1] - 1))
            if padded.shape[0] < 2 or padded.shape[1] < 2:
                raise ValueError("reflect padding невозможен для измерения короче 2 пикселей")
            padded = np.pad(padded, ((t, b), (l, r)), mode="reflect")
            pads = [pads[0] - t, pads[1] - b, pads[2] - l, pads[3] - r]
        return padded[y0 + top:y1 + top, x0 + left:x1 + left]
    return image[y0:y1, x0:x1]


def overlap_fraction(x0: int, y0: int, size: int, bounds: tuple[float, ...]) -> float:
    bx0, by0, bx1, by1 = bounds
    width = max(0.0, min(x0 + size, bx1) - max(x0, bx0))
    height = max(0.0, min(y0 + size, by1) - max(y0, by0))
    return width * height / (size * size)


def main() -> None:
    args = parse_args()
    annotations = load_annotations()
    index = pd.read_csv("data/index/images.csv", dtype={"sop_uid": str, "study": str})
    folds = pd.read_csv("experiments/folds.csv", dtype={"study": str})
    if index["sop_uid"].duplicated().any():
        raise ValueError("В images.csv обнаружены повторяющиеся sop_uid")
    fold_map = folds.set_index("study")["fold"].to_dict()
    image_map = index.set_index("sop_uid").to_dict("index")
    missing = sorted(set(annotations) - set(image_map))
    if missing:
        raise KeyError(f"В images.csv отсутствуют SOP UID ({len(missing)}): {missing[:3]}")

    patches, labels, uids, studies, out_folds, sources, box_sizes = ([] for _ in range(7))

    def append(patch: np.ndarray, label: int, uid: str, study: str,
               fold: int, source: str, box_size: float) -> None:
        patches.append(patch)
        labels.append(label)
        uids.append(uid)
        studies.append(study)
        out_folds.append(fold)
        sources.append(source)
        box_sizes.append(box_size)

    rng = np.random.default_rng(args.seed)
    for uid, versions in annotations.items():
        row = image_map[uid]
        study = str(row["study"])
        if study not in fold_map:
            raise KeyError(f"Для study {study} не найден fold")
        fold = int(fold_map[study])
        image = read_image(Path(row["path"]))
        all_boxes = [box for version in versions for box in version.get("boxes", [])]
        bounds = [box_geometry(box)[3] for box in all_boxes]

        for box in all_boxes:
            kind = box.get("kind")
            if kind not in {"object", "hard_negative"}:
                raise ValueError(f"Неизвестный kind для {uid}: {kind!r}")
            cx, cy, box_size, _ = box_geometry(box)
            append(extract_patch(image, cx, cy, args.size), int(kind == "object"),
                   uid, study, fold, kind, box_size)

        # При нескольких разметках требуем единогласную отрицательную метку.
        if args.background and all(v.get("y_foreign") == 0 for v in versions):
            height, width = image.shape
            accepted = 0
            attempts = 0
            max_attempts = max(1000, args.background * 1000)
            while accepted < args.background and attempts < max_attempts:
                attempts += 1
                x0 = int(rng.integers(0, max(1, width - args.size + 1)))
                y0 = int(rng.integers(0, max(1, height - args.size + 1)))
                if any(overlap_fraction(x0, y0, args.size, b) > 0.2 for b in bounds):
                    continue
                append(extract_patch(image, x0 + args.size / 2, y0 + args.size / 2,
                                     args.size), 0, uid, study, fold, "background", 0.0)
                accepted += 1
            if accepted < args.background:
                raise RuntimeError(f"Для {uid} найдено только {accepted} фоновых патчей")

    arrays = {
        "patches": np.stack(patches).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
        "sop_uid": np.asarray(uids, dtype=str),
        "study": np.asarray(studies, dtype=str),
        "fold": np.asarray(out_folds, dtype=np.int8),
        "source": np.asarray(sources, dtype=str),
        "box_size": np.asarray(box_sizes, dtype=np.float32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    print(f"Сохранено: {args.out} ({len(labels)} патчей)")
    print("Классы:", dict(sorted(Counter(labels).items())))
    print("Источники:", dict(sorted(Counter(sources).items())))
    print("Фолды:", dict(sorted(Counter(out_folds).items())))


if __name__ == "__main__":
    main()
