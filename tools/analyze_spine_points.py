#!/usr/bin/env python3
"""Проверка ручной разметки поясниц: полнота, устойчивость угла, согласие с метками организатора.

    python tools/analyze_spine_points.py data/annotations/spine_points_<имя>.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERTEBRAE = ["Th12", "L1", "L2", "L3", "L4", "L5"]
SX, SY = 0.600, 0.606


def centers(points: dict) -> dict:
    """Центры тел: новый формат (<V>_center) и старый (две пластинки)."""
    out = {}
    for v in VERTEBRAE:
        c = points.get(f"{v}_center")
        if isinstance(c, dict) and "x" in c:
            out[v] = (c["x"], c["y"])
            continue
        t, b = points.get(f"{v}_top") or {}, points.get(f"{v}_bottom") or {}
        if "x" in t and "x" in b:
            out[v] = ((t["x"] + b["x"]) / 2, (t["y"] + b["y"]) / 2)
    return out


def angle(pts: list[tuple[float, float]]) -> float | None:
    """Угол прямой x = a·y + b к вертикали кадра, градусы."""
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    den = float(((y - y.mean()) ** 2).sum())
    if den < 1e-9:
        return None
    a = float(((y - y.mean()) * (x - x.mean())).sum() / den)
    return math.degrees(math.atan2(abs(a) * SX, SY))


def jitter_std(pts, sigma_px=2.5, n=200, seed=0) -> float:
    """Разброс угла при случайном сдвиге каждой точки на ±sigma пикселей."""
    rng = np.random.default_rng(seed)
    base = np.array(pts, float)
    vals = [angle(list(map(tuple, base + rng.normal(0, sigma_px, base.shape)))) for _ in range(n)]
    return float(np.std([v for v in vals if v is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--index", type=Path, default=ROOT / "data/index/images.csv")
    args = ap.parse_args()

    rows = []
    for f in args.files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for uid, ann in data["images"].items():
            c = centers(ann.get("points", {}))
            pts = list(c.values())
            ys = [p[1] for p in pts]
            rows.append({
                "annotator": data["annotator"], "sop_uid": uid, "study": ann["study"], "study_n": ann["study_n"],
                "complete": bool(ann.get("complete")), "skip": bool(ann.get("skip_image")),
                "seconds": ann.get("seconds_spent", 0.0), "n_centers": len(pts),
                "angle": angle(pts), "angle_std_jitter": jitter_std(pts) if len(pts) >= 3 else None,
                "angle_loo_max_shift": max_loo_shift(pts),
                "monotonic": ys == sorted(ys),
                "crest_left": (ann.get("crest_left") or {}).get("state"),
                "crest_right": (ann.get("crest_right") or {}).get("state"),
                "th12": ann.get("th12_half_visible"),
                "numbering_uncertain": bool(ann.get("numbering_uncertain")),
                "comment": (ann.get("comment") or "").strip(),
            })
    df = pd.DataFrame(rows)
    done = df[df.complete & ~df.skip]
    print(f"размечено: {len(df)}, завершено: {len(done)}, пропущено: {int(df.skip.sum())}")
    if len(done):
        print(f"время: медиана {done.seconds.median():.0f} с, всего {done.seconds.sum()/60:.0f} мин")
        print(f"центров на снимок: медиана {done.n_centers.median():.0f} (мин {done.n_centers.min()})")
        bad_order = done[~done.monotonic]
        print(f"нарушен порядок сверху вниз: {len(bad_order)} {list(bad_order.study_n)}")
        print(f"угол: медиана {done.angle.median():.2f}°, максимум {done.angle.max():.2f}°, "
              f"> 5°: {int((done.angle > 5).sum())}")
        print(f"устойчивость угла к сдвигу точек ±2,5 px: медиана σ = {done.angle_std_jitter.median():.2f}°, "
              f"макс {done.angle_std_jitter.max():.2f}°")
        print(f"сдвиг угла при удалении одного позвонка: медиана {done.angle_loo_max_shift.median():.2f}°, "
              f"макс {done.angle_loo_max_shift.max():.2f}°")
        print(f"неуверенная нумерация: {int(done.numbering_uncertain.sum())}, комментарии: {int((done.comment != '').sum())}")
        print("Th12:", done.th12.value_counts().to_dict())
        print("гребни слева:", done.crest_left.value_counts().to_dict(), "| справа:", done.crest_right.value_counts().to_dict())

    if args.index.exists():
        idx = pd.read_csv(args.index)
        m = done.merge(idx[["sop_uid", "y_axis", "y_pos", "y_foreign", "quality_class", "comment"]],
                       on="sop_uid", how="left", suffixes=("", "_label"))
        print("\nСогласие измерений с метками организатора (внешняя проверка, пороги ТЗ):")
        ax = m[m.y_axis.notna()]
        if len(ax):
            print("  ось: метка 1 →", ax.loc[ax.y_axis == 1, "angle"].round(2).tolist(),
                  "| метка 0 → медиана", round(ax.loc[ax.y_axis == 0, "angle"].median(), 2),
                  ", максимум", round(ax.loc[ax.y_axis == 0, "angle"].max(), 2))
            pred = (ax.angle > 5).astype(int)
            tp = int(((pred == 1) & (ax.y_axis == 1)).sum()); fp = int(((pred == 1) & (ax.y_axis == 0)).sum())
            fn = int(((pred == 0) & (ax.y_axis == 1)).sum())
            print(f"  правило «угол > 5°»: TP={tp} FP={fp} FN={fn}")
        po = m[m.y_pos.notna()]
        if len(po):
            crest_missing = po.crest_left.isin(["out_of_frame", "partial"]) | po.crest_right.isin(["out_of_frame", "partial"])
            th12_bad = po.th12.isin(["no"])
            rule = (crest_missing | th12_bad).astype(int)
            tp = int(((rule == 1) & (po.y_pos == 1)).sum()); fp = int(((rule == 1) & (po.y_pos == 0)).sum())
            fn = int(((rule == 0) & (po.y_pos == 1)).sum())
            print(f"  укладка «гребень не полностью в кадре или Th12 < половины»: TP={tp} FP={fp} FN={fn} "
                  f"(положительных меток {int(po.y_pos.sum())})")
        out = ROOT / "data/annotations/spine_points_analysis.csv"
        m.to_csv(out, index=False)
        print(f"\nтаблица по снимкам: {out}")


def max_loo_shift(pts) -> float | None:
    """Насколько меняется угол, если выбросить один позвонок (устойчивость к ошибке нумерации/точки)."""
    if len(pts) < 4:
        return None
    base = angle(pts)
    shifts = [abs(base - angle(pts[:i] + pts[i + 1:])) for i in range(len(pts))]
    return float(max(shifts))


if __name__ == "__main__":
    main()
