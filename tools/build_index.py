#!/usr/bin/env python3
"""Индекс обучающих данных: уникальные изображения + область + сторона + метки из разметка.xlsx.

Запуск из корня проекта:
    python tools/build_index.py            # -> data/index/images.csv, files.csv, labels_issues.csv

Правила (проверены на НД_для_обучения, 16.09.2026):
- область: Columns == 300 -> поясница, иначе -> бедро (совпало с разметкой во всех 100 исследованиях);
- сторона бедра: центр масс яркости верхней трети левее/правее нижней трети.
  Головка и таз медиальнее диафиза; правое бедро -> медиальная сторона справа на снимке
  (сверено с тестовыми файлами *_ППОБ / *_ЛПОБ);
- дубли: одинаковые пиксели (md5) внутри исследования считаются одним изображением.
"""
import hashlib
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

warnings.filterwarnings("ignore")
logging.getLogger("pydicom").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train"
OUT = ROOT / "data" / "index"

# формулировки из «Разъяснения по вопросам ЛЦТ_V2.docx»
SPINE = "Поясничный отдел позвоночника"
HIP = "Проксимальный отдел бедра"
V_POS = "Некорректная укладка"
V_AXIS = "Не выравнена ось позвоночника"
V_FOREIGN = "Присутствуют посторонние предметы"
V_ROI = "Некорректная область интереса"


def hip_side(px: np.ndarray) -> tuple[str, float]:
    a = px.astype(float)
    h, w = a.shape
    xs = np.arange(w)

    def cx(m):
        col = m.sum(0)
        return (col * xs).sum() / max(col.sum(), 1e-6)

    d = cx(a[: h // 3]) - cx(a[2 * h // 3 :])
    return ("R" if d > 0 else "L"), float(d)


def scan_files() -> pd.DataFrame:
    rows = []
    for f in sorted(TRAIN.glob("Исследования/**/*.dcm")):
        ds = pydicom.dcmread(f)
        px = ds.pixel_array
        region = "spine" if ds.Columns == 300 else "hip"
        side, score = hip_side(px) if region == "hip" else ("", np.nan)
        rows.append(dict(
            path=str(f.relative_to(ROOT)), study=f.relative_to(TRAIN).parts[1],
            study_uid_tag=ds.StudyInstanceUID, sop_uid=ds.SOPInstanceUID,
            instance=int(ds.get("InstanceNumber", 0) or 0), rows=ds.Rows, cols=ds.Columns,
            software=ds.get("SoftwareVersions"), exposed_area=str(ds.get("ExposedArea")),
            md5=hashlib.md5(px.tobytes()).hexdigest(), region=region, side=side, side_score=score,
        ))
    return pd.DataFrame(rows)


def read_labels() -> pd.DataFrame:
    raw = pd.read_excel(TRAIN / "разметка.xlsx", header=None, skiprows=2).iloc[:, :13]
    raw.columns = ["n", "study", "sp_pos", "sp_axis", "sp_art", "rh_pos", "rh_roi",
                   "lh_pos", "lh_roi", "tot_sp", "tot_rh", "tot_lh", "comment"]
    return raw[raw.study.notna()].set_index("study")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = scan_files()
    files["n_copies"] = files.groupby(["study", "md5"]).md5.transform("size")
    files.to_csv(OUT / "files.csv", index=False)

    img = files.sort_values(["study", "instance", "path"]).drop_duplicates(["study", "md5"]).copy()
    lab = read_labels()
    issues, out = [], []
    for _, r in img.iterrows():
        L = lab.loc[r.study]
        if r.region == "spine":
            parts = {V_POS: L.sp_pos, V_AXIS: L.sp_axis, V_FOREIGN: L.sp_art}
            total = L.tot_sp
        else:
            p = "rh" if r.side == "R" else "lh"
            parts = {V_POS: L[f"{p}_pos"], V_ROI: L[f"{p}_roi"]}
            total = L[f"tot_{p}"]
        labeled = pd.notna(total)
        viol = [k for k, v in parts.items() if v == 1] if labeled else []
        comps = int(max(parts.values())) if labeled else None
        consistent = (not labeled) or comps == int(total)
        if not consistent:
            issues.append(dict(n=L.n, study=r.study, region=r.region, side=r.side, **{k: v for k, v in parts.items()},
                               total=total, comment=L.comment))
        out.append(dict(
            study_n=int(L.n), study=r.study, path=r.path, sop_uid=r.sop_uid, n_copies=r.n_copies,
            rows=r.rows, cols=r.cols, region=r.region, side=r.side, side_score=r.side_score,
            anatomical_region=SPINE if r.region == "spine" else HIP,
            # Цель «есть нарушение» = OR отдельных критериев; столбец «Итог» не используется:
            # организатор на Q&A — «ориентируйтесь на столбцы с конкретными нарушениями, в Итоге есть ошибки»
            labeled=labeled, quality_class=comps if labeled else None,
            total_official=int(total) if labeled else None,
            violation_type=";".join(viol), label_consistent=consistent,
            **{f"y_{k}": (int(v) if labeled else None) for k, v in
               {"pos": parts[V_POS], "axis": parts.get(V_AXIS), "foreign": parts.get(V_FOREIGN),
                "roi": parts.get(V_ROI)}.items() if v is not None or not labeled},
            comment=L.comment if isinstance(L.comment, str) else "",
        ))
    images = pd.DataFrame(out)
    images.to_csv(OUT / "images.csv", index=False)
    pd.DataFrame(issues).to_csv(OUT / "labels_issues.csv", index=False)

    lab_img = images[images.labeled]
    print(f"файлов {len(files)}, уникальных изображений {len(images)}, размечено {len(lab_img)}")
    print(lab_img.groupby(["region", "side"]).quality_class.agg(["size", "sum"]))
    print("несогласованных строк разметки:", len(issues))


if __name__ == "__main__":
    main()
