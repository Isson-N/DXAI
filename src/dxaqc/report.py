"""Таблица результатов в формате организатора (п. 2.5 ТЗ + quality_prob из разъяснений)."""
from __future__ import annotations

import csv
from pathlib import Path

from openpyxl import Workbook

COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "quality_prob",
    "violation_type",
    "processing_status",
    "time_of_processing",
]
ERROR_COLUMNS = ["path", "stage", "message"]


def write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".csv":
        with output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    elif suffix == ".xlsx":
        wb = Workbook()
        ws = wb.active
        ws.title = "results"
        ws.append(COLUMNS)
        for r in rows:
            ws.append([r[c] for c in COLUMNS])
        wb.save(output)
    else:
        raise ValueError(f"формат результата должен быть .csv или .xlsx: {output}")


def write_errors(errors: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ERROR_COLUMNS)
        w.writeheader()
        w.writerows(errors)
