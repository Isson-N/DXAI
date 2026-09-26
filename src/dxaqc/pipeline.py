"""Пакетная обработка: вход → строки результата и список ошибок."""
from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import Callable

from .io import NotAnImage, extract_if_archive, list_candidate_files, looks_like_dicom, read_image
from .labels import STATUS_FAIL, STATUS_OK

log = logging.getLogger("dxaqc")


def _cache_key(image, model_version: str) -> str:
    # План v2, этап 6: копии изображения с разными UID считаем один раз
    h = hashlib.sha256(image.pixels.tobytes())
    h.update(repr((image.pixels.shape, str(image.pixels.dtype), model_version)).encode())
    return h.hexdigest()


def run(input_path: Path, model, synchronize: Callable[[], None] | None = None) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    errors: list[dict] = []
    cache: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="dxaqc_") as tmp:
        root = extract_if_archive(input_path, Path(tmp))
        for path in list_candidate_files(root):
            rel = path.relative_to(root).as_posix()
            if not looks_like_dicom(path):
                errors.append({"path": rel, "stage": "discover", "message": "не DICOM, пропущен"})
                continue
            t0 = time.perf_counter()
            try:
                image = read_image(path, root)
            except NotAnImage as e:
                errors.append({"path": rel, "stage": "read", "message": str(e)})
                continue
            except Exception as e:  # любой сбой чтения фиксируется, обработка продолжается
                errors.append({"path": rel, "stage": "read", "message": f"{type(e).__name__}: {e}"})
                rows.append(_failure_row(rel, time.perf_counter() - t0))
                continue
            for note in image.warnings:
                errors.append({"path": rel, "stage": "read", "message": f"предупреждение: {note}"})
            try:
                key = _cache_key(image, model.version)
                if key not in cache:
                    cache[key] = model.predict(image)
                pred = cache[key]
                if synchronize is not None:
                    synchronize()
                rows.append({
                    "path_to_study": rel,
                    "study_uid": image.study_uid,
                    "image_uid": image.image_uid,
                    "anatomical_region": pred.region,
                    "quality_class": pred.quality_class,
                    "quality_prob": round(float(pred.quality_prob), 6),
                    "violation_type": ";".join(pred.violation_list),
                    "processing_status": STATUS_OK,
                    "time_of_processing": round(time.perf_counter() - t0, 4),
                })
            except Exception as e:
                errors.append({"path": rel, "stage": "predict", "message": f"{type(e).__name__}: {e}"})
                rows.append(_failure_row(rel, time.perf_counter() - t0, image.study_uid, image.image_uid))
    log.info("обработано строк: %d, ошибок/предупреждений: %d", len(rows), len(errors))
    return rows, errors


def _failure_row(rel: str, seconds: float, study_uid: str = "", image_uid: str = "") -> dict:
    return {
        "path_to_study": rel,
        "study_uid": study_uid,
        "image_uid": image_uid,
        "anatomical_region": "",
        "quality_class": "",
        "quality_prob": "",
        "violation_type": "",
        "processing_status": STATUS_FAIL,
        "time_of_processing": round(seconds, 4),
    }
