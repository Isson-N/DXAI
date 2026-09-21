"""Наложить разметку на снимки и распечатать сводку — для проверки работы разметчика.

Читает файл любой из трёх схем (`dxa-foreign-boxes/1`, `dxa-hip-points/1`,
`dxa-axis-review/1`) и складывает картинки с наложением в каталог, чтобы глазами
увидеть, куда именно легли рамки, линии и точки.

Запуск:
    python tools/preview_annotation.py data/annotations/foreign_boxes_den.json
    python tools/preview_annotation.py data/annotations/hip_points_den.json --out /tmp/hip
    python tools/preview_annotation.py data/annotations/foreign_boxes_den.json --only-with-boxes
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image, ImageDraw

SCALE = 3                      # во сколько раз увеличить снимок: тонкие дуги иначе не видно
POINT_COLOURS = {
    "H": (255, 80, 80), "B1": (80, 200, 255), "B2": (80, 200, 255),
    "T": (255, 220, 60), "D": (140, 255, 140), "D2": (140, 255, 140),
}


def render(path: str) -> Image.Image:
    """DICOM → серое изображение с мягкой гаммой (слабый контраст иначе теряется)."""
    pixels = pydicom.dcmread(path).pixel_array.astype(float)
    low, high = np.percentile(pixels, [1, 99.7])
    norm = np.clip((pixels - low) / max(high - low, 1e-6), 0, 1) ** 0.75
    image = Image.fromarray((norm * 255).astype(np.uint8)).convert("RGB")
    return image.resize((image.width * SCALE, image.height * SCALE), Image.LANCZOS)


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: list) -> list[str]:
    """Рамки и линии посторонних предметов; возвращает строки для сводки."""
    notes = []
    for box in boxes:
        shape = box.get("shape", "rect")
        # Ловушки — синим, настоящие объекты — красным; неуверенные пунктиром по контуру.
        colour = (255, 70, 70) if box["kind"] == "object" else (90, 160, 255)
        if shape == "line":
            ends = [(box["x1"] * SCALE, box["y1"] * SCALE), (box["x2"] * SCALE, box["y2"] * SCALE)]
            draw.line(ends, fill=colour + (110,), width=int(box["thickness"] * SCALE))
            draw.line(ends, fill=colour + (255,), width=2)
            length = math.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"])
            ratio = length / max(box["thickness"], 1e-6)
            flag = "  ← полоса шире, чем нужно" if ratio < 3 else ""
            notes.append(f"линия {length:.0f}×{box['thickness']} px "
                         f"(отношение {ratio:.1f}:1) {box['class']}{flag}")
        else:
            x, y = box["x"] * SCALE, box["y"] * SCALE
            draw.rectangle([x, y, x + box["w"] * SCALE, y + box["h"] * SCALE],
                           outline=colour + (255,), width=2)
            notes.append(f"прямоугольник {box['w']}×{box['h']} px {box['class']}")
        if not box.get("sure", True):
            notes[-1] += "  [не уверен]"
    return notes


def draw_points(draw: ImageDraw.ImageDraw, points: dict) -> list[str]:
    """Ключевые точки бедра; невидимые точки попадают в сводку, рисовать нечего."""
    notes = []
    for name, point in points.items():
        state = point.get("state")
        if "x" not in point:
            notes.append(f"{name}: {state}")
            continue
        x, y = point["x"] * SCALE, point["y"] * SCALE
        colour = POINT_COLOURS.get(name, (255, 255, 255))
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=colour, width=3)
        draw.text((x + 9, y - 7), name, fill=colour)
        notes.append(f"{name}: ({point['x']:.0f}, {point['y']:.0f}) "
                     f"уверенность {point.get('confidence', '—')}")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("annotation", help="JSON с разметкой")
    ap.add_argument("--index", default="data/index/images.csv")
    ap.add_argument("--out", default="data/overview/preview")
    ap.add_argument("--only-with-boxes", action="store_true",
                    help="пропускать снимки без единого элемента разметки")
    args = ap.parse_args()

    data = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    schema = data.get("schema", "")
    records = data.get("images") or data.get("answers") or {}
    index = pd.read_csv(args.index).set_index("sop_uid")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"схема: {schema} | записей: {len(records)}")
    if schema.startswith("dxa-axis-review"):
        # У пересмотра «оси» рисовать нечего: там вердикты, а не геометрия.
        verdicts = {}
        for answer in records.values():
            verdicts[answer["verdict"]] = verdicts.get(answer["verdict"], 0) + 1
        print("вердикты:", verdicts)
        print("с причинами:", sum(1 for a in records.values() if a.get("reasons")))
        return

    written = 0
    for number, (uid, record) in enumerate(records.items(), 1):
        base_uid = uid.split("#")[0]
        if base_uid not in index.index:
            print(f"  {uid[-8:]}: нет в индексе, пропуск")
            continue
        boxes = record.get("boxes", [])
        points = {k: v for k, v in (record.get("points") or {}).items()}
        if args.only_with_boxes and not boxes and not points:
            continue
        image = render(index.loc[base_uid, "path"])
        draw = ImageDraw.Draw(image, "RGBA")
        notes = draw_boxes(draw, boxes) if boxes else draw_points(draw, points)
        name = f"{number:03d}_{record.get('state', '?')}_{uid[-8:]}.png"
        image.save(out / name)
        written += 1
        head = (f"метка={record.get('y_foreign', record.get('y_pos', '—'))} "
                f"состояние={record.get('state', '—')}")
        print(f"  {name}  {head}")
        for line in notes:
            print(f"      {line}")
        if record.get("comment"):
            print(f"      комментарий: {record['comment']}")

    print(f"\nсохранено картинок: {written} → {out}")


if __name__ == "__main__":
    main()
