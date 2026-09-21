"""Перевести D и D₂ из `not_visible` в `out_of_frame` там, где направляющая вне кадра.

Разметчику быстрее нажать одну клавишу, чем различать «линии не видно, потому что она
за нижним краем» и «линия в кадре, но кость на ней не различить». Разница смысловая:
первое — геометрия кадра, второе — качество изображения, и смешивать их нельзя, иначе
нельзя честно сказать, почему признак ротации недоступен на части снимков.

Решение объективное, а не на глаз: уровень направляющей считается как y(T) + 20 мм
или + 50 мм (при отсутствии T — от середины B₁B₂, как это делает сам разметчик),
переводится в пиксели через шаг 0,606 мм и сравнивается с высотой кадра.
Записи, где уровень внутри кадра, НЕ трогаются: там `not_visible` стоит по делу.

Запуск (по умолчанию ничего не пишет, только показывает):
    python tools/fix_guide_states.py data/annotations/hip_points_isson.json
    python tools/fix_guide_states.py data/annotations/hip_points_isson.json --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PIXEL_MM = 0.606          # шаг пикселя по вертикали, тот же, что в hip_annotator
OFFSETS = {"D": 20, "D2": 50}
PORT = 8767               # порт разметчика бедра


def guide_origin(points: dict) -> float | None:
    """Точка отсчёта направляющих: T, иначе середина B₁B₂ — правило hip_annotator."""
    t = points.get("T")
    if t and "y" in t and t.get("state") in ("visible", "uncertain"):
        return t["y"]
    b1, b2 = points.get("B1"), points.get("B2")
    if t and t.get("state") in ("not_visible", "out_of_frame", "uncertain") \
            and b1 and b2 and "y" in b1 and "y" in b2:
        return (b1["y"] + b2["y"]) / 2
    return None


def busy(port: int) -> bool:
    """Инструмент держит разметку в памяти: правка файла под ним будет затёрта."""
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("annotation", help="JSON с разметкой бедра")
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--apply", action="store_true", help="записать изменения")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    if args.apply and busy(args.port):
        sys.exit(f"Порт {args.port} занят: остановите разметчик, иначе правка будет затёрта.")

    path = Path(args.annotation)
    data = json.loads(path.read_text(encoding="utf-8"))
    index = pd.read_csv(args.index).set_index("sop_uid")

    # Записи лежат в двух местах: завершённые в images, незавершённые в session.drafts.
    groups = [data["images"], data.get("session", {}).get("drafts", {})]

    changed, kept, skipped = [], [], []
    for records in groups:
        for uid, record in records.items():
            points = record.get("points") or {}
            base = uid.split("#")[0]
            if base not in index.index:
                continue
            rows = int(index.loc[base, "rows"])
            origin = guide_origin(points)
            for key, offset in OFFSETS.items():
                point = points.get(key)
                if not point or point.get("state") != "not_visible":
                    continue
                if origin is None:
                    skipped.append(f"{uid[-8:]} {key}: уровень не вычислить (нет ни T, ни B₁B₂)")
                    continue
                level = origin + offset / PIXEL_MM
                if level >= rows or level < 0:
                    changed.append(f"{uid[-8:]} {key}: уровень {level:.0f} при высоте {rows}"
                                   f" → out_of_frame")
                    if args.apply:
                        point["state"] = "out_of_frame"
                else:
                    kept.append(f"{uid[-8:]} {key}: уровень {level:.0f} в кадре ({rows})"
                                f" → оставлено not_visible")

    for line in changed:
        print("  ИЗМЕНИТЬ  " + line)
    for line in kept:
        print("  оставить  " + line)
    for line in skipped:
        print("  пропуск   " + line)
    print(f"\nк изменению: {len(changed)}; оставлено как есть: {len(kept)};"
          f" не вычислено: {len(skipped)}")

    if not args.apply:
        print("Ничего не записано. Повторите с --apply, остановив разметчик.")
        return

    backup = path.with_suffix(f".before-guidefix-{datetime.now():%H%M%S}.json")
    shutil.copy(path, backup)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Записано. Копия прежнего файла: {backup}")


if __name__ == "__main__":
    main()
