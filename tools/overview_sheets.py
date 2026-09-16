#!/usr/bin/env python3
"""Контактные листы по классам разметки -> data/overview/*.png.

Нужен data/index/images.csv (tools/build_index.py). Запуск из корня проекта:
    python tools/overview_sheets.py
"""
import logging
from pathlib import Path

import pandas as pd
import pydicom
from PIL import Image, ImageDraw, ImageFont

logging.getLogger("pydicom").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "overview"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def sheet(sub: pd.DataFrame, name: str, title: str, ncol=8, cell=230):
    if sub.empty:
        return
    f_title, f_small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 11)
    nrow = (len(sub) + ncol - 1) // ncol
    head, lab = 32, 30
    canvas = Image.new("L", (ncol * cell, head + nrow * (cell + lab)), 0)
    dr = ImageDraw.Draw(canvas)
    dr.text((6, 6), f"{title}  (n={len(sub)})", fill=255, font=f_title)
    for i, (_, r) in enumerate(sub.iterrows()):
        im = Image.fromarray(pydicom.dcmread(ROOT / r.path).pixel_array)
        im.thumbnail((cell - 6, cell - 6))
        x, y = (i % ncol) * cell, head + (i // ncol) * (cell + lab)
        canvas.paste(im, (x + 2, y + lab))
        side = {"R": "правое", "L": "левое"}.get(r.side, "") if isinstance(r.side, str) else ""
        dr.text((x + 3, y + 1), f"№{r.study_n} {side} {r.rows}×{r.cols}", fill=255, font=f_small)
        comment = r.comment if isinstance(r.comment, str) else ""
        extra = r.violation_type if isinstance(r.violation_type, str) and ";" in r.violation_type else ""
        dr.text((x + 3, y + 14), (comment or extra)[:38], fill=190, font=f_small)
    canvas.save(OUT / f"{name}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data" / "index" / "images.csv")
    d["violation_type"] = d.violation_type.fillna("")
    lab = d[d.labeled]
    sp, hp = lab[lab.region == "spine"], lab[lab.region == "hip"]
    v = lambda df, s: df[df.violation_type.str.contains(s)]
    sheet(sp[sp.quality_class == 0], "spine_0_norma", "Поясница: норма")
    sheet(v(sp, "укладка"), "spine_1_ukladka", "Поясница: Некорректная укладка")
    sheet(v(sp, "ось"), "spine_2_os", "Поясница: Не выравнена ось позвоночника")
    sheet(v(sp, "посторонние"), "spine_3_predmety", "Поясница: Присутствуют посторонние предметы")
    sheet(sp[(sp.quality_class == 1) & (sp.violation_type == "")], "spine_4_itog1_bez_narusheniy",
          "Поясница: итог = 1, но ни одно нарушение не отмечено")
    sheet(hp[hp.quality_class == 0], "hip_0_norma", "Бедро: норма")
    sheet(v(hp, "укладка"), "hip_1_ukladka", "Бедро: Некорректная укладка")
    sheet(v(hp, "область"), "hip_2_oblast_interesa", "Бедро: Некорректная область интереса")
    sheet(d[~d.labeled], "ne_razmecheno", "Без разметки (эндопротезы)")


if __name__ == "__main__":
    main()
