"""Measure CUDA inference time per DICOM study against the 180-second limit."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dxaqc.pipeline import run
from dxaqc.service_model import load


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="каталог или ZIP с DICOM")
    parser.add_argument("--models", default=Path("models"), type=Path)
    parser.add_argument("--limit-seconds", default=180.0, type=float)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        parser.error("PyTorch не установлен в текущем окружении")
    if not torch.cuda.is_available():
        parser.error("CUDA недоступна: нужен GPU, драйвер и сборка PyTorch с CUDA")
    if not args.input.exists():
        parser.error(f"вход не найден: {args.input}")

    started = time.perf_counter()
    model = load(args.models, "cuda")
    if any(part is None for part in (model.keypoints, model.cnn, model.foreign,
                                     model.hip_rotation)):
        parser.error("для измерения нужны все четыре модели")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    rows, errors = run(args.input, model, synchronize=torch.cuda.synchronize)
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started

    studies: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["processing_status"] == "Success":
            key = row["study_uid"] or row["path_to_study"]
            studies[key] += float(row["time_of_processing"])
    times = sorted(studies.values())
    result = {
        "device": torch.cuda.get_device_name(0),
        "images": len(rows),
        "successful_images": sum(row["processing_status"] == "Success" for row in rows),
        "errors_or_warnings": len(errors),
        "studies": len(times),
        "model_load_seconds": round(load_seconds, 3),
        "batch_wall_seconds": round(batch_seconds, 3),
        "mean_study_seconds": round(sum(times) / len(times), 3) if times else None,
        "max_study_seconds": round(max(times), 3) if times else None,
        "limit_seconds": args.limit_seconds,
        "within_limit": bool(times) and max(times) <= args.limit_seconds,
        "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["within_limit"] and result["successful_images"] == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
