#!/usr/bin/env python3
"""Объединение разметки поясниц от двух разметчиков + согласие на общих снимках.

    python tools/merge_spine_points.py data/annotations/spine_points_a.json data/annotations/spine_points_b.json \
        --out data/annotations/spine_points_merged.json

В объединённом файле у каждого снимка список версий `annotations` (по одной на разметчика).
Согласие считается по снимкам, завершённым обоими: расстояние между точками (мм), разница угла оси,
совпадение состояний гребней и флага Th12.
"""
import argparse
import json
import math
from pathlib import Path

VERTEBRAE = ["Th12", "L1", "L2", "L3", "L4", "L5"]
SX, SY = 0.600, 0.606  # мм/пиксель выгруженных изображений (PLAN.md, решение 17.09.2026)


def normalize(points: dict) -> dict:
    """Приводит к центрам тел. Старый формат (две пластинки на позвонок) → середина между ними."""
    out = {}
    for v in VERTEBRAE:
        c = points.get(f"{v}_center")
        if isinstance(c, dict) and "x" in c:
            out[v] = {"x": c["x"], "y": c["y"]}
            continue
        t, b = points.get(f"{v}_top") or {}, points.get(f"{v}_bottom") or {}
        if "x" in t and "x" in b:
            out[v] = {"x": (t["x"] + b["x"]) / 2, "y": (t["y"] + b["y"]) / 2}
        elif "x" in t or "x" in b:
            q = t if "x" in t else b
            out[v] = {"x": q["x"], "y": q["y"], "half_only": True}
        else:
            state = (c or t or b or {}).get("state")
            if state:
                out[v] = {"state": state}
    return out


def centers(points: dict) -> list[tuple[float, float]]:
    return [(p["x"], p["y"]) for p in normalize(points).values() if "x" in p]


def axis_angle(points: dict) -> float | None:
    """Угол прямой x = a·y + b (МНК по центрам тел) к вертикали, градусы, с учётом масштаба пикселя."""
    c = centers(points)
    if len(c) < 3:
        return None
    my = sum(y for _, y in c) / len(c)
    mx = sum(x for x, _ in c) / len(c)
    syy = sum((y - my) ** 2 for _, y in c)
    if syy == 0:
        return None
    a = sum((x - mx) * (y - my) for x, y in c) / syy
    return math.degrees(math.atan2(abs(a) * SX, SY))


def agreement(a: dict, b: dict) -> dict:
    na, nb = normalize(a["points"]), normalize(b["points"])
    dists = []
    for key, pa in na.items():
        pb = nb.get(key, {})
        if "x" in pa and "x" in pb:
            dists.append(math.hypot((pa["x"] - pb["x"]) * SX, (pa["y"] - pb["y"]) * SY))
    angle_a, angle_b = axis_angle(a["points"]), axis_angle(b["points"])
    return {
        "mean_point_distance_mm": sum(dists) / len(dists) if dists else None,
        "max_point_distance_mm": max(dists) if dists else None,
        "angle_diff_deg": abs(angle_a - angle_b) if angle_a is not None and angle_b is not None else None,
        "crest_states_equal": all((a.get(k) or {}).get("state") == (b.get(k) or {}).get("state")
                                  for k in ("crest_left", "crest_right")),
        "th12_flag_equal": a.get("th12_half_visible") == b.get("th12_half_visible"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged: dict = {"schema": "dxa-spine-points-merged/1", "annotators": [], "images": {}}
    for f in args.files:
        data = json.loads(f.read_text(encoding="utf-8"))
        merged["annotators"].append({"annotator": data["annotator"], "part": data["part"]})
        for image_id, ann in data["images"].items():
            entry = merged["images"].setdefault(image_id, {"study": ann["study"], "study_n": ann["study_n"],
                                                           "annotations": []})
            entry["annotations"].append({"annotator": data["annotator"], **ann,
                                         "centers_px": normalize(ann.get("points", {}))})

    rows = []
    for image_id, entry in merged["images"].items():
        done = [x for x in entry["annotations"] if x.get("complete") and not x.get("skip_image")]
        if len(done) >= 2:
            rows.append({"study_n": entry["study_n"], **agreement(done[0], done[1])})
    merged["agreement"] = rows
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    complete = sum(any(x.get("complete") for x in e["annotations"]) for e in merged["images"].values())
    print(f"снимков: {len(merged['images'])}, завершено хотя бы одним: {complete}, общих завершённых: {len(rows)}")
    if rows:
        def mean(key):
            vals = [r[key] for r in rows if r[key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        print(f"согласие: точки {mean('mean_point_distance_mm'):.2f} мм в среднем, угол ±{mean('angle_diff_deg'):.2f}°, "
              f"гребни совпали {sum(r['crest_states_equal'] for r in rows)}/{len(rows)}, "
              f"Th12 совпал {sum(r['th12_flag_equal'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
