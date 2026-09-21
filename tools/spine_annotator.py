# Запуск: python tools/spine_annotator.py --index data/index/images.csv --root . --annotator ivan --part 1
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import re
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pydicom
from PIL import Image

SCHEMA = "dxa-spine-points/1"
VERTEBRAE = ("Th12", "L1", "L2", "L3", "L4", "L5")
POINT_KEYS = tuple(f"{v}_center" for v in VERTEBRAE)
LEGACY_KEYS = tuple(f"{v}_{s}" for v in VERTEBRAE for s in ("top", "bottom"))
SPACING = {"x": 0.600, "y": 0.606}


def split_part(studies, part):
    """Вернуть исследования выбранной части в детерминированном порядке."""
    part = str(part)
    if part not in ("1", "2", "all"):
        raise ValueError("Часть должна быть 1, 2 или all.")
    unique = set(studies)
    digest = lambda text: hashlib.sha1(text.encode("utf-8")).hexdigest()
    ordered = sorted(unique, key=lambda s: (digest(s), s))
    if part == "all":
        return ordered
    overlap = set(sorted(unique, key=lambda s: (digest("overlap:" + s), s))[:10])
    parity = int(part) - 1
    return [s for i, s in enumerate(ordered) if i % 2 == parity or s in overlap]


def read_png(filename):
    try:
        ds = pydicom.dcmread(filename)
    except Exception:
        ds = pydicom.dcmread(filename, force=True)
    pixels = np.asarray(ds.pixel_array, dtype=np.float64)
    # Один кадр с явным измерением кадров также допустим.
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or min(pixels.shape) < 1:
        raise ValueError("Ожидалось однокадровое монохромное изображение.")
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        bits = int(getattr(ds, "BitsStored", 16))
        if not 1 <= bits <= 64:
            raise ValueError("Недопустимое значение BitsStored.")
        pixels = (2**bits - 1) - pixels
    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError("Изображение не содержит конечных значений.")
    low, high = float(pixels[finite].min()), float(pixels[finite].max())
    pixels = np.nan_to_num(pixels, nan=low, posinf=high, neginf=low)
    if high > low:
        pixels = np.clip((pixels - low) / (high - low) * 255, 0, 255)
    else:
        pixels = np.zeros_like(pixels)
    array = pixels.astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue(), int(array.shape[0]), int(array.shape[1])


def atomic_write(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=filename.parent,
            prefix="." + filename.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def has_coordinates(point):
    return isinstance(point, dict) and "x" in point and "y" in point


def migrate_legacy_points(annotation):
    """Convert the former top/bottom vertebra markup to one center per body."""
    points = annotation.get("points")
    if not isinstance(points, dict) or not (set(points) & set(LEGACY_KEYS)):
        return annotation, False

    migrated = {key: value for key, value in points.items() if key in POINT_KEYS}
    for vertebra in VERTEBRAE:
        center_key = f"{vertebra}_center"
        if center_key in migrated:
            continue
        top = points.get(f"{vertebra}_top")
        bottom = points.get(f"{vertebra}_bottom")
        if has_coordinates(top) and has_coordinates(bottom):
            migrated[center_key] = {
                "x": (top["x"] + bottom["x"]) / 2,
                "y": (top["y"] + bottom["y"]) / 2,
            }
        elif (
            isinstance(top, dict) and isinstance(bottom, dict)
            and top.get("state") == bottom.get("state")
            and top.get("state") in ("absent", "uncertain")
        ):
            migrated[center_key] = {"state": top["state"]}
        elif top is not None or bottom is not None:
            # One visible endplate is not enough to infer the body center reliably.
            migrated[center_key] = {"state": "uncertain"}

    result = dict(annotation)
    result["points"] = migrated
    return result, True


def derive(annotation):
    centers = {}
    points = annotation["points"]
    for vertebra in VERTEBRAE:
        center = points.get(vertebra + "_center")
        if has_coordinates(center):
            centers[vertebra] = {"x": center["x"], "y": center["y"]}
    annotation["centers"] = centers
    annotation["axis_angle_deg"] = None
    if len(centers) >= 3:
        x = np.array([p["x"] for p in centers.values()], dtype=float)
        y = np.array([p["y"] for p in centers.values()], dtype=float)
        denominator = float(np.sum((y - y.mean()) ** 2))
        if denominator > 1e-12:
            a = float(np.sum((y - y.mean()) * (x - x.mean())) / denominator)
            annotation["axis_angle_deg"] = math.degrees(
                math.atan2(abs(a) * SPACING["x"], SPACING["y"])
            )
    vertebrae_done = all(key in points for key in POINT_KEYS)
    crests_done = all(
        isinstance(annotation.get(key), dict)
        and (
            annotation[key].get("state") in ("out_of_frame", "partial")
            or (
                annotation[key].get("state") == "in_frame"
                and has_coordinates(annotation[key])
            )
        )
        for key in ("crest_left", "crest_right")
    )
    annotation["complete"] = bool(
        annotation["skip_image"]
        or (
            vertebrae_done and crests_done
            and annotation["th12_half_visible"] in ("yes", "no", "cannot")
        )
    )
    return annotation


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.filename = Path(args.out) / f"spine_points_{args.annotator}.json"
        self.items = {}
        self.order = []
        self.pngs = {}
        root = Path(args.root).resolve()
        with open(args.index, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {
                "study_n", "study", "path", "sop_uid", "region",
                "labeled", "y_pos", "y_axis", "y_foreign",
            }
            if not required.issubset(reader.fieldnames or []):
                missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
                raise ValueError("В CSV отсутствуют столбцы: " + missing)
            rows = [row for row in reader if row["region"].strip() == "spine"]
        studies = split_part((row["study"] for row in rows), args.part)
        study_rank = {study: i for i, study in enumerate(studies)}
        rows = [row for row in rows if row["study"] in study_rank]
        rows.sort(key=lambda row: (study_rank[row["study"]], row["sop_uid"]))
        print(f"Чтение изображений выбранной части: {len(rows)}…", flush=True)
        for row in rows:
            uid = row["sop_uid"].strip()
            if not uid:
                raise ValueError("В CSV найден пустой sop_uid.")
            if uid in self.items:
                old = self.items[uid]
                if old["study"] != row["study"]:
                    raise ValueError("Один sop_uid относится к разным исследованиям.")
                continue
            study_n = row["study_n"]
            try:
                study_n = int(study_n)
            except ValueError:
                pass
            item = {
                "image_id": uid, "study": row["study"], "study_n": study_n,
                "rows": 0, "cols": 0, "error": None,
            }
            try:
                relative = Path(row["path"])
                filename = (root / relative).resolve()
                if relative.is_absolute() or not filename.is_relative_to(root):
                    raise ValueError("Файл находится вне корневой папки.")
                png, height, width = read_png(filename)
                item.update(rows=height, cols=width)
                self.pngs[uid] = png
            except Exception as exc:
                # Не передаём исключение целиком: оно может содержать медицинский путь.
                item["error"] = (
                    "Не удалось прочитать DICOM. Проверьте файл и поддержку "
                    f"его формата ({type(exc).__name__})."
                )
                print(f"Ошибка изображения № {study_n}: {type(exc).__name__}", flush=True)
            self.items[uid] = item
            self.order.append(uid)

        self.data = {
            "schema": SCHEMA, "annotator": args.annotator, "part": str(args.part),
            "pixel_spacing_mm": SPACING, "images": {},
        }
        if self.filename.exists():
            with self.filename.open(encoding="utf-8") as stream:
                existing = json.load(stream)
            if (
                not isinstance(existing, dict) or existing.get("schema") != SCHEMA
                or existing.get("annotator") != args.annotator
                or not isinstance(existing.get("images"), dict)
            ):
                raise ValueError("Файл разметки имеет неподходящий формат или автора.")
            self.data = existing
            self.data["part"] = str(args.part)
            self.data["pixel_spacing_mm"] = SPACING
            legacy_migrated = False
            for uid in self.order:
                if uid in self.data["images"]:
                    previous = self.data["images"][uid]
                    previous, migrated = migrate_legacy_points(previous)
                    legacy_migrated = legacy_migrated or migrated
                    cleaned = self.validate(uid, previous)
                    cleaned["updated_at"] = previous.get("updated_at", "")
                    self.data["images"][uid] = cleaned
            if legacy_migrated:
                backup = self.filename.with_suffix(
                    f".legacy-{datetime.now():%Y%m%d-%H%M%S}.json"
                )
                backup.write_bytes(self.filename.read_bytes())
                atomic_write(self.filename, self.data)
                print(f"Старая разметка преобразована; исходный файл: {backup}", flush=True)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def blank(self, uid):
        item = self.items[uid]
        return derive({
            "study": item["study"], "study_n": item["study_n"],
            "rows": item["rows"], "cols": item["cols"], "points": {},
            "crest_left": None, "crest_right": None,
            "th12_half_visible": None, "comment": "",
            "skip_image": False, "skip_reason": "", "numbering_uncertain": False, "updated_at": "",
            "seconds_spent": 0.0,
        })

    def validate(self, uid, raw):
        if not isinstance(raw, dict):
            raise ValueError("Ожидался объект разметки.")
        allowed = {
            "study", "study_n", "rows", "cols", "points", "crest_left",
            "crest_right", "th12_half_visible", "comment", "skip_image",
            "skip_reason", "numbering_uncertain", "complete", "updated_at", "seconds_spent",
            "centers", "axis_angle_deg",
        }
        if set(raw) - allowed:
            raise ValueError("Объект разметки содержит неизвестные поля.")
        item = self.items[uid]
        result = self.blank(uid)

        def validate_point(point, crest=False):
            if not isinstance(point, dict):
                raise ValueError("Точка должна быть объектом.")
            if set(point) - {"x", "y", "state"}:
                raise ValueError("Неизвестные поля точки.")
            coordinates = "x" in point or "y" in point
            state = point.get("state")
            if "state" in point and not isinstance(state, str):
                raise ValueError("Состояние точки должно быть строкой.")
            if coordinates:
                if not all(numeric(point.get(key)) for key in ("x", "y")):
                    raise ValueError("Координаты x и y должны быть конечными числами.")
                x, y = float(point["x"]), float(point["y"])
                if item["error"]:
                    raise ValueError("Нельзя ставить точки на непрочитанном изображении.")
                if not (-5 <= x <= item["cols"] - 1 + 5
                        and -5 <= y <= item["rows"] - 1 + 5):
                    raise ValueError("Координаты выходят за изображение более чем на 5 пикселей.")
            if crest:
                # Черновики (только координаты или только состояние) сохраняются,
                # но не завершают шаг до появления обеих обязательных составляющих.
                if state not in (None, "in_frame", "partial", "out_of_frame"):
                    raise ValueError("Недопустимое состояние гребня.")
                if state == "out_of_frame" and coordinates:
                    raise ValueError("У гребня вне кадра не должно быть координат.")
                if not coordinates and state is None:
                    raise ValueError("Пустое описание гребня.")
            elif coordinates:
                if "state" in point:
                    raise ValueError("У поставленной точки не должно быть состояния N/U.")
            elif state not in ("absent", "uncertain"):
                raise ValueError("Укажите координаты точки или состояние absent/uncertain.")
            answer = {}
            if coordinates:
                answer.update(x=x, y=y)
            if state is not None:
                answer["state"] = state
            return answer

        points = raw.get("points", {})
        if not isinstance(points, dict) or set(points) - set(POINT_KEYS):
            raise ValueError("Недопустимые названия точек позвонков.")
        result["points"] = {key: validate_point(value) for key, value in points.items()}
        for key in ("crest_left", "crest_right"):
            if raw.get(key) is not None:
                result[key] = validate_point(raw[key], crest=True)
        half = raw.get("th12_half_visible")
        if half is not None and (not isinstance(half, str) or half not in ("yes", "no", "cannot")):
            raise ValueError("Недопустимое значение флага Th12.")
        result["th12_half_visible"] = half
        result["numbering_uncertain"] = bool(raw.get("numbering_uncertain", False))
        for key, maximum in (("comment", 20000), ("skip_reason", 2000)):
            value = raw.get(key, "")
            if not isinstance(value, str) or len(value) > maximum:
                raise ValueError(f"Поле {key}: ожидается текст длиной до {maximum} символов.")
            result[key] = value
        skip = raw.get("skip_image", False)
        if type(skip) is not bool:
            raise ValueError("Флаг skip_image должен быть логическим.")
        if skip and not result["skip_reason"].strip():
            raise ValueError("Для пропущенного снимка обязательна причина.")
        result["skip_image"] = skip
        seconds = raw.get("seconds_spent", 0)
        if not numeric(seconds) or seconds < 0:
            raise ValueError("Время работы должно быть конечным неотрицательным числом.")
        result["seconds_spent"] = round(float(seconds), 3)
        result["updated_at"] = datetime.now(timezone.utc).isoformat()
        return derive(result)

    def state(self):
        with self.lock:
            images = []
            for uid in self.order:
                item = self.items[uid]
                annotation = self.data["images"].get(uid, {})
                images.append({
                    **item, "complete": bool(annotation.get("complete")),
                    "has_comment": bool(annotation.get("comment", "").strip()),
                    "skip_image": bool(annotation.get("skip_image")),
                    "annotated": uid in self.data["images"],
                })
            return {
                "schema": SCHEMA, "annotator": self.args.annotator,
                "part": str(self.args.part), "pixel_spacing_mm": SPACING,
                "images": images, "first_run": not self.data["images"],
            }


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DXA — разметка позвоночника</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;font-size:14px}
*{box-sizing:border-box}
body{margin:0;background:#121820;color:#e8eef5}
button,select,input,textarea{font:inherit}
button,select{background:#263447;color:#edf3fa;border:1px solid #52647b;border-radius:5px;padding:7px 10px}
button{cursor:pointer}button:hover:not(:disabled){background:#354a63}
button:disabled{opacity:.4;cursor:default}
button.active{background:#185479;border-color:#62caff}
input,textarea{background:#172332;color:#fff;border:1px solid #51647b;border-radius:4px;padding:6px}
input[type=range]{width:115px;padding:0;vertical-align:middle}
textarea{width:100%;resize:vertical;min-height:58px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#1e2a3a}
#saveStatus{margin-left:auto;color:#9cdfb6}
nav{display:flex;gap:7px;align-items:center;padding:8px 12px;background:#182331;flex-wrap:wrap}
main{height:calc(100vh - 116px);min-height:520px;display:grid;grid-template-columns:minmax(350px,1fr) 385px}
#left{display:flex;flex-direction:column;min-width:0;min-height:0}
#tools{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
#stage{position:relative;flex:1;overflow:hidden;background:#080c11;min-height:240px;touch-action:none}
#stage canvas{position:absolute;left:0;top:0;width:100%;height:100%}
#overlay{cursor:crosshair}
#imageCanvas{pointer-events:none}
#imageMessage{position:absolute;top:20px;left:20px;right:20px;color:#ffd29b;pointer-events:none;white-space:pre-wrap}
#coordinates{padding:7px 12px;color:#b2c4d8;min-height:33px}
aside{padding:12px;overflow:auto;background:#1a2432;border-left:1px solid #364559}
#prompt{font-size:22px;font-weight:650;background:#253c53;padding:12px;border-radius:6px;min-height:96px}
.hint{font-size:12px;color:#b4c5d6;line-height:1.5}
.row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:8px 0}
#steps{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin:10px 0}
#steps button{font-size:12px;text-align:left;padding:6px}
#steps button.current{border:2px solid #ffd36e;padding:5px;background:#4d452c}
#steps button.done{color:#98e2b0}
#steps button.marked{color:#ffd28b}
section.controls{border-top:1px solid #3b4b60;padding:8px 0}
label.block{display:block;margin:8px 0 4px}
#skipReason{width:100%}
#instruction{max-width:1000px;margin:20px auto;padding:24px;background:#1d2a39;line-height:1.65}
#instruction svg{max-width:100%;display:block;margin:20px auto}
#instruction li{margin:12px 0}
.hidden{display:none!important}
#errorBanner{background:#743a32;padding:8px 14px;white-space:pre-wrap}
#angle{color:#8febd3}
a{color:#9dd9ff}
@media(max-width:800px){main{grid-template-columns:1fr;height:auto}#left{height:65vh}aside{max-height:none}}
</style>
</head>
<body>
<header>
<strong>DXA · позвоночник</strong>
<span id="identity"></span><span id="progress"></span><span id="study"></span>
<span id="saveStatus">Загрузка…</span><button id="retry" class="hidden">Повторить сохранение</button>
</header>
<nav>
<button id="editTab" class="active">Разметка</button>
<button id="viewTab">Просмотр</button>
<button id="helpTab">Инструкция</button>
<select id="filter" class="hidden" aria-label="Фильтр просмотра">
<option value="all">Все снимки</option><option value="unfinished">Незавершённые</option>
<option value="comment">С комментарием</option><option value="skipped">Пропущенные</option>
</select>
<button id="firstUnfinished">К первому незавершённому</button>
<a href="/api/export" download>Скачать JSON</a>
</nav>
<div id="errorBanner" class="hidden"></div>
<main id="workspace">
<div id="left">
<div id="tools">
<button id="fit">Вписать</button><button id="zoom3">×3</button><span id="zoomLabel">×3.00</span>
<label>Яркость <input id="brightness" type="range" min="30" max="250" value="100"></label>
<label>Контраст <input id="contrast" type="range" min="30" max="350" value="100"></label>
<button id="resetFilters">Сброс</button><span id="angle">Ось: —</span>
</div>
<div id="stage">
<canvas id="imageCanvas"></canvas><canvas id="overlay" tabindex="0" aria-label="Снимок для разметки"></canvas>
<div id="imageMessage"></div>
</div>
<div id="coordinates">Координаты: — · колесо: масштаб · ПКМ / пробел + мышь: панорама</div>
</div>
<aside>
<div id="prompt"></div>
<div id="steps"></div>
<section class="controls">
<div class="row">
<button id="absent" class="editControl">N · вне кадра / не видно</button>
<button id="uncertain" class="editControl">U · не определить</button>
</div>
<label class="editControl"><input type="checkbox" id="numberingUncertain"> Нумерация позвонков неуверенная</label>
<div class="hint">N/U относятся к текущему позвонку. Выберите строку шага для исправления.
Существующую точку можно перетащить.</div>
</section>
<section class="controls">
<strong>Текущий гребень</strong>
<div class="row" id="crestButtons">
<button data-crest="in_frame" class="editControl">1 · в кадре</button>
<button data-crest="partial" class="editControl">2 · частично</button>
<button data-crest="out_of_frame" class="editControl">3 · вне кадра</button>
</div>
<div class="hint">Для «в кадре» и «частично» нужны и точка, и состояние; порядок их выбора любой.</div>
</section>
<section class="controls">
<strong>Видна ли половина Th12?</strong>
<div class="row" id="halfButtons">
<button data-half="yes" class="editControl">Y · ≥ половины</button>
<button data-half="no" class="editControl">H · &lt; половины</button>
<button data-half="cannot" class="editControl">C · не определить</button>
</div>
</section>
<section class="controls">
<label class="block" for="comment">Комментарий (необязательно)</label>
<textarea id="comment" maxlength="20000" class="editControl"></textarea>
<label class="block"><input type="checkbox" id="skip" class="editControl"> Снимок нельзя разметить</label>
<label class="block" for="skipReason">Причина пропуска (обязательна при пропуске)</label>
<input id="skipReason" maxlength="2000" class="editControl" placeholder="Например: изображение повреждено">
</section>
<div class="row">
<button id="undo" class="editControl">Отменить · Z</button>
<button id="resetImage" class="editControl">Сбросить снимок</button>
<button id="back">← Назад</button><button id="next">Далее →</button>
</div>
<div id="position" class="hint"></div>
<div class="hint">Enter — далее · Y/H/C — флаг Th12 · Ctrl+Z — отмена.
В режиме просмотра изменения и учёт времени отключены.</div>
</aside>
</main>
<article id="instruction" class="hidden">
<h1>Инструкция разметчика</h1>
<p>Цель: разметить ориентиры для измерения оси позвоночника и проверки поля сканирования.</p>
<ol>
<li>Позвонки идут сверху вниз: Th12, L1–L5. Th12 — последний позвонок с рёбрами;
L5 — последний над крестцом; верхние края подвздошных костей обычно на уровне L4–L5.
Если нумерация неочевидна (переходный позвонок, 6 поясничных), отметьте в комментарии
и размечайте по лучшему пониманию, спорную точку — «U».</li>
<li>Центр тела позвонка — середина тела (не остистого и не поперечных отростков):
на глаз середина между верхней и нижней замыкательными пластинками и между левым и правым краем тела.
Точность до 2–3 пикселей достаточна: по центрам строится только ось.</li>
<li>Позвонок частично за кадром: если тело не видно — «N». Если видно, но центр определить
нельзя (наложение, артефакт) — «U».</li>
<li>Гребни: самая верхняя точка гребня подвздошной кости слева и справа на экране.
Если вершина срезана краем кадра — «частично» (клавиша 2) <b>без точки</b>, не ставьте её на границе кадра.
«Вне кадра» (3) — гребня не видно совсем.</li>
<li>Если не уверены, какой позвонок Th12 (переходный позвонок, необычная анатомия) — поставьте галочку
«Нумерация позвонков неуверенная». Точки всё равно ставьте: для угла оси названия позвонков не нужны,
важен только порядок сверху вниз.</li>
<li>Th12: «видна ≥ половины» — в кадре не меньше половины высоты тела Th12;
«&lt; половины»; «не определить» — Th12 не идентифицируется.</li>
<li>Сколиоз, перелом, металл не мешают разметке: ставьте точки по фактическому положению тел.
Посторонние предметы не размечаем.</li>
<li>Сомневаетесь — комментарий. Лучше «U», чем угадывать.</li>
</ol>
<svg viewBox="0 0 660 230" width="660" role="img" aria-label="Центр тела позвонка">
<path d="M90 50 Q170 25 250 50 L260 180 Q170 200 80 180 Z"
 fill="#40566e" stroke="#d4e1ef" stroke-width="3"/>
<path d="M90 50 Q170 25 250 50 M80 180 Q170 200 260 180" fill="none" stroke="#ffdb80" stroke-width="4"/>
<circle cx="170" cy="114" r="9" fill="#ffda69"/>
<path d="M158 114 L182 114 M170 102 L170 126" stroke="#83ffe0" stroke-width="3"/>
<g stroke="#b6cadc"><path d="M186 114 L305 114"/></g>
<g fill="#eef4fa" font-size="17" font-family="sans-serif">
<text x="315" y="120">один клик: центр тела позвонка</text>
</g>
</svg>
<p>Последовательно: 6 центров тел позвонков → два гребня с состояниями → флаг Th12.
Координаты сохраняются в пикселях исходного снимка. Гребни именуются по сторонам
экрана, а не по анатомической стороне пациента.</p>
<p>Масштаб по умолчанию ×3. Колесо — приближение под курсором (×1…×10);
правая кнопка или пробел с перетаскиванием — перемещение. «Вписать» может уменьшить
изображение ниже ×1. Яркость и контраст не меняют сохранённое изображение.</p>
<p>Каждое изменение автоматически сохраняется. Не закрывайте страницу до появления
«сохранено ✓». Время учитывается только при видимой вкладке и активном окне разметки.
Открывайте одну вкладку разметки на одного разметчика.</p>
<button id="start">К разметке</button>
</article>
<script>
"use strict";
const $ = id => document.getElementById(id);
const vertebrae = ["Th12","L1","L2","L3","L4","L5"];
const keys = vertebrae.map(v => v+"_center").concat(["crest_left","crest_right","th12_half_visible"]);
const NV = vertebrae.length;            // 6 точек-центров
const CREST0 = NV, FLAG = NV+2, TOTAL = NV+3;
const copy = value => JSON.parse(JSON.stringify(value));
let state, items=[], current=null, ann=null, image=null, index=-1, step=0;
let mode="edit", help=false, loading=false, history=[], zoom=3, offsetX=0, offsetY=0;
let cursor=null, gesture=null, space=false, width=1, height=1;
let saveQueue=Promise.resolve(), saveOK=true, pending=0, loadNumber=0;
let clock=performance.now(), timeDirty=false, lastFocus=document.hasFocus();
const base=$("imageCanvas"), overlay=$("overlay");
const ctx=overlay.getContext("2d"), imageCtx=base.getContext("2d");

async function api(url, options={}) {
    let response;
    try { response=await fetch(url,{cache:"no-store",...options}); }
    catch(e) { throw new Error("Нет связи с сервером."); }
    if(!response.ok) {
        let message="Ошибка сервера: "+response.status;
        try {message=(await response.json()).error || message;} catch(e) {}
        throw new Error(message);
    }
    return response.json();
}
function error(message="") {
    $("errorBanner").textContent=message;
    $("errorBanner").classList.toggle("hidden",!message);
}
function editable() {return !!ann && mode==="edit" && !help && !loading;}
function timed() {return editable() && !document.hidden && lastFocus;}
function account() {
    const now=performance.now();
    if(timed()) {
        const seconds=Math.max(0,(now-clock)/1000);
        ann.seconds_spent=(ann.seconds_spent||0)+seconds;
        if(seconds>0) timeDirty=true;
    }
    clock=now;
}
function point(key) {
    if(!ann) return null;
    return key.startsWith("crest_") ? ann[key] : ann.points[key];
}
function coords(p) {return p && Number.isFinite(p.x) && Number.isFinite(p.y);}
function done(i) {
    if(!ann) return false;
    const key=keys[i];
    if(i<NV) return !!ann.points[key];
    if(i<FLAG) {
        const p=ann[key];
        return !!p && (["out_of_frame","partial"].includes(p.state) ||
            (coords(p) && p.state==="in_frame"));
    }
    return ["yes","no","cannot"].includes(ann.th12_half_visible);
}
function complete() {return !!ann && (ann.skip_image || keys.every((k,i)=>done(i)));}
function firstMissing() {
    const found=keys.findIndex((key,i)=>!done(i));
    return found<0 ? TOTAL : found;
}
function nextMissing(after) {
    for(let i=after+1;i<TOTAL;i++) if(!done(i)) return i;
    return firstMissing();
}
function label(key) {
    if(key==="crest_left") return "гребень Л";
    if(key==="crest_right") return "гребень П";
    if(key==="th12_half_visible") return "Половина Th12";
    return key.replace("_center","");
}
function localMetadata() {
    if(!current || !ann) return;
    current.complete=complete();
    current.has_comment=!!ann.comment.trim();
    current.skip_image=ann.skip_image;
}
function saveIndicator() {
    $("saveStatus").textContent=pending ? "сохранение…" : saveOK ? "сохранено ✓" : "ошибка сохранения";
    $("saveStatus").style.color=saveOK ? "#9cdfb6" : "#ffada5";
    $("retry").classList.toggle("hidden",saveOK || !!pending);
}
function enqueueSave() {
    if(!ann || !current) return;
    account();
    const id=current.image_id, snapshot=copy(ann);
    timeDirty=false;
    pending++; saveIndicator();
    saveQueue=saveQueue.then(async()=>{
        try {
            const saved=await api("/api/annotation/"+encodeURIComponent(id),{
                method:"PUT",headers:{"Content-Type":"application/json"},
                body:JSON.stringify(snapshot)
            });
            saveOK=true;
            const item=items.find(it=>it.image_id===id);
            if(item) item.annotated=true;
            if(current && current.image_id===id) ann.updated_at=saved.updated_at;
            error("");
        } catch(e) {
            saveOK=false;
            error(e.message+" Изменения остаются в этой вкладке. Нажмите «Повторить сохранение».");
        } finally {
            pending--; saveIndicator();
        }
    });
}
async function flush() {
    account();
    if(timeDirty && mode==="edit") enqueueSave();
    await saveQueue;
    return saveOK;
}
function mutate(fn, advance=false) {
    if(!editable()) return;
    account();
    history.push({annotation:copy(ann),step});
    if(history.length>250) history.shift();
    const previous=step;
    fn();
    if(advance && done(previous)) step=nextMissing(previous);
    ann.complete=complete();
    localMetadata();
    enqueueSave();
    render();
}
function undo() {
    if(!editable() || !history.length) return;
    account();
    const elapsed=ann.seconds_spent;
    const previous=history.pop();
    ann=previous.annotation;
    ann.seconds_spent=elapsed;
    step=previous.step;
    localMetadata(); enqueueSave(); syncFields(); render();
}
function chosenList() {
    if(mode!=="view") return items;
    const filter=$("filter").value;
    return items.filter(it=>filter==="all" ||
        filter==="unfinished" && !it.complete ||
        filter==="comment" && it.has_comment ||
        filter==="skipped" && it.skip_image);
}
function syncFields() {
    $("comment").value=ann ? ann.comment : "";
    $("numberingUncertain").checked=!!ann?.numbering_uncertain;
    $("skip").checked=!!ann?.skip_image;
    $("skipReason").value=ann ? ann.skip_reason : "";
}
function render() {
    $("identity").textContent=state ? `Разметчик: ${state.annotator} · часть ${state.part}` : "";
    $("progress").textContent=`${items.filter(it=>it.complete).length} завершено / ${items.length} всего`;
    $("study").textContent=current ? "study_n: "+current.study_n : "";
    $("filter").classList.toggle("hidden",mode!=="view");
    $("workspace").classList.toggle("hidden",help);
    $("instruction").classList.toggle("hidden",!help);
    $("editTab").classList.toggle("active",mode==="edit"&&!help);
    $("viewTab").classList.toggle("active",mode==="view"&&!help);
    $("helpTab").classList.toggle("active",help);
    let prompt="Нет снимков для выбранного фильтра.";
    if(loading) prompt="Загрузка снимка…";
    else if(ann) {
        if(mode==="view") prompt="Просмотр · только чтение";
        else if(ann.skip_image) prompt="Снимок пропущен. Причина сохранена.";
        else if(step<NV) {
            const [v,side]=keys[step].split("_");
            prompt=`Кликните ЦЕНТР тела позвонка ${v}`;
        } else if(step<FLAG) {
            const p=ann[keys[step]];
            prompt=`Гребень ${step===CREST0?"СЛЕВА":"СПРАВА"} на экране: `;
            prompt+=!coords(p) ? "кликните вершину и выберите состояние (1/2/3)" :
                "выберите состояние (1/2/3)";
        } else if(step===FLAG) prompt="Видна ли ≥ половины тела Th12? Y / H / C";
        else prompt="Разметка завершена. Можно перейти далее.";
    }
    $("prompt").textContent=prompt;
    $("steps").replaceChildren();
    keys.forEach((key,i)=>{
        const button=document.createElement("button");
        const p=i<FLAG ? point(key) : null;
        let status=done(i)?"✓":"○";
        if(p?.state==="absent") status="N";
        if(p?.state==="uncertain") status="U";
        if(p?.state==="out_of_frame") status="3";
        button.textContent=`${status} ${label(key)}`;
        if(i>=CREST0 && i<FLAG && p?.state)
            button.textContent+=" · "+({in_frame:"в кадре",partial:"частично",out_of_frame:"вне кадра"}[p.state]);
        if(i===FLAG && ann?.th12_half_visible)
            button.textContent+=" · "+({yes:"≥½",no:"<½",cannot:"?"}[ann.th12_half_visible]);
        button.className=(done(i)?"done ":"")+(p?.state?"marked ":"")+(i===step?"current":"");
        button.disabled=!ann || loading || mode!=="edit";
        button.onclick=()=>{step=i;render();};
        $("steps").append(button);
    });
    document.querySelectorAll(".editControl").forEach(el=>el.disabled=!editable());
    $("absent").disabled=$("uncertain").disabled=!editable()||step>=NV||ann.skip_image;
    document.querySelectorAll("[data-crest]").forEach(el=>{
        el.disabled=!editable()||step<CREST0||step>=FLAG||ann.skip_image;
        el.classList.toggle("active",!!ann && step>=CREST0 && step<FLAG && ann[keys[step]]?.state===el.dataset.crest);
    });
    document.querySelectorAll("[data-half]").forEach(el=>{
        el.disabled=!editable()||step!==FLAG||ann.skip_image;
        el.classList.toggle("active",ann?.th12_half_visible===el.dataset.half);
    });
    $("undo").disabled=!editable()||!history.length;
    const list=chosenList(), position=current ? list.indexOf(current) : -1;
    $("back").disabled=loading||position<=0;
    $("next").disabled=loading||position<0||position>=list.length-1||
        (mode==="edit"&&!complete());
    $("position").textContent=position<0 ? `В выборке: ${list.length}` :
        `Снимок ${position+1} / ${list.length}${current.error?" · ошибка DICOM":""}`;
    $("firstUnfinished").disabled=loading||!items.some(it=>!it.complete);
    draw();
}
function resize() {
    const box=$("stage").getBoundingClientRect();
    if(!box.width || !box.height) return;
    width=box.width; height=box.height;
    const dpr=window.devicePixelRatio||1;
    for(const canvas of [base,overlay]) {
        canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr);
        canvas.getContext("2d").setTransform(dpr,0,0,dpr,0,0);
    }
    draw();
}
function screen(p) {return {x:p.x*zoom+offsetX,y:p.y*zoom+offsetY};}
function native(p) {return {x:(p.x-offsetX)/zoom,y:(p.y-offsetY)/zoom};}
function bounded(p) {
    return {x:Math.max(0,Math.min(current.cols-1,p.x)),y:Math.max(0,Math.min(current.rows-1,p.y))};
}
function placements() {
    if(!ann) return [];
    return keys.slice(0,FLAG).map(key=>({key,p:point(key)})).filter(it=>coords(it.p));
}
function centers() {
    if(!ann) return [];
    return vertebrae.map(v=>{
        const c=ann.points[v+"_center"];
        return coords(c) ? {x:c.x,y:c.y,v} : null;
    }).filter(Boolean);
}
function draw() {
    imageCtx.clearRect(0,0,width,height); ctx.clearRect(0,0,width,height);
    if(image) {
        imageCtx.imageSmoothingEnabled=false;
        imageCtx.drawImage(image,offsetX,offsetY,image.naturalWidth*zoom,image.naturalHeight*zoom);
    }
    $("zoomLabel").textContent="×"+zoom.toFixed(2);
    $("angle").textContent="Ось: —";
    const mid=centers();
    if(mid.length>=3) {
        const ym=mid.reduce((s,p)=>s+p.y,0)/mid.length;
        const xm=mid.reduce((s,p)=>s+p.x,0)/mid.length;
        const den=mid.reduce((s,p)=>s+(p.y-ym)**2,0);
        if(den>1e-12) {
            const a=mid.reduce((s,p)=>s+(p.y-ym)*(p.x-xm),0)/den, b=xm-a*ym;
            const lo=Math.min(...mid.map(p=>p.y)), hi=Math.max(...mid.map(p=>p.y));
            const first=screen({x:a*lo+b,y:lo}), last=screen({x:a*hi+b,y:hi});
            ctx.strokeStyle="#64efd1"; ctx.lineWidth=2; ctx.setLineDash([7,4]);
            ctx.beginPath();ctx.moveTo(first.x,first.y);ctx.lineTo(last.x,last.y);ctx.stroke();
            ctx.setLineDash([]);
            const angle=Math.atan2(Math.abs(a)*0.600,0.606)*180/Math.PI;
            $("angle").textContent=`Ось: ${angle.toFixed(2)}° к вертикали`;
        }
    }
    for(const p of mid) {
        const q=screen(p);ctx.strokeStyle="#70ffcf";ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(q.x-5,q.y);ctx.lineTo(q.x+5,q.y);
        ctx.moveTo(q.x,q.y-5);ctx.lineTo(q.x,q.y+5);ctx.stroke();
    }
    for(const {key,p} of placements()) {
        const q=screen(p);
        const selected=keys[step]===key;
        ctx.beginPath();ctx.arc(q.x,q.y,selected?6:4.5,0,Math.PI*2);
        ctx.fillStyle=key.startsWith("crest")?"#ff91d2":"#ffdb75";ctx.fill();
        ctx.strokeStyle="#10151d";ctx.lineWidth=1.5;ctx.stroke();
        ctx.font="bold 12px system-ui";
        ctx.lineWidth=3;ctx.strokeStyle="#10151d";
        ctx.strokeText(label(key),q.x+9,q.y-7);
        ctx.fillStyle=selected?"#fff":"#ffebad";ctx.fillText(label(key),q.x+9,q.y-7);
    }
    if(cursor) {
        ctx.strokeStyle="#ffffff77";ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(cursor.x,0);ctx.lineTo(cursor.x,height);
        ctx.moveTo(0,cursor.y);ctx.lineTo(width,cursor.y);ctx.stroke();
        const p=native(cursor);
        $("coordinates").textContent=`x: ${p.x.toFixed(1)} · y: ${p.y.toFixed(1)} пикс. · колесо: масштаб · ПКМ / пробел: панорама`;
    } else $("coordinates").textContent="Координаты: — · колесо: масштаб · ПКМ / пробел + мышь: панорама";
}
function fitScale() {
    // Весь снимок в окне (Th12 вверху кадра сразу видна), но не мельче ×1 и не крупнее ×10
    if(!image) return 3;
    return Math.max(1,Math.min((width-20)/image.naturalWidth,(height-20)/image.naturalHeight,10));
}
function centerImage(scale=3) {
    zoom=scale;
    offsetX=(width-(image?.naturalWidth||0)*zoom)/2;
    offsetY=(height-(image?.naturalHeight||0)*zoom)/2;
    draw();
}
function imagePoint(p) {
    const q=native(p);
    if(!image || q.x<0 || q.y<0 || q.x>=current.cols || q.y>=current.rows) return null;
    return bounded(q);
}
function applyPoint(p) {
    if(!editable()||ann.skip_image||step>=FLAG) return;
    const key=keys[step];
    mutate(()=>{
        if(step<NV) ann.points[key]=p;
        else {
            const old=ann[key]?.state;
            ann[key]={...p};
            if(old==="in_frame"||old==="partial") ann[key].state=old;
        }
    },true);
}
function setPointState(value) {
    if(!editable()||ann.skip_image||step>=NV) return;
    mutate(()=>{ann.points[keys[step]]={state:value};},true);
}
function setCrestState(value) {
    if(!editable()||ann.skip_image||step<CREST0||step>=FLAG) return;
    const key=keys[step];
    mutate(()=>{
        const old=ann[key];
        ann[key]=value==="out_of_frame" ? {state:value} :
            {...(coords(old)?{x:old.x,y:old.y}:{}),state:value};
    },true);
}
function setHalf(value) {
    if(!editable()||ann.skip_image||step!==FLAG) return;
    mutate(()=>{ann.th12_half_visible=value;},true);
}
async function loadItem(item) {
    if(loading || gesture) return;
    if(!await flush()) return;
    account(); loading=true; render();
    const serial=++loadNumber;
    image=null; cursor=null; history=[];
    current=item||null; ann=null; syncFields();
    if(!item) {
        loading=false;$("imageMessage").textContent="Нет снимков.";render();return;
    }
    $("imageMessage").textContent="Загрузка…";
    try {
        const loaded=await api("/api/annotation/"+encodeURIComponent(item.image_id));
        if(serial!==loadNumber) return;
        ann=loaded;step=firstMissing();index=items.indexOf(item);
        if(!item.error) {
            const loadedImage=new Image();
            await new Promise((resolve,reject)=>{
                loadedImage.onload=resolve;
                loadedImage.onerror=()=>reject(new Error("Не удалось загрузить PNG."));
                loadedImage.src="/api/image/"+encodeURIComponent(item.image_id)+".png";
            });
            image=loadedImage;
        }
        $("imageMessage").textContent=item.error||"";
    } catch(e) {
        $("imageMessage").textContent=e.message;
        if(!ann) error(e.message);
    } finally {
        loading=false;clock=performance.now();timeDirty=false;
        syncFields();render();resize();centerImage(fitScale());
    }
}
async function navigate(direction) {
    if(loading || gesture) return;
    if(direction>0 && mode==="edit" && !complete()) return;
    const list=chosenList(), pos=list.indexOf(current), destination=list[pos+direction];
    if(destination) await loadItem(destination);
}
async function setMode(value) {
    if(loading || gesture || !await flush()) return;
    account();mode=value;help=false;clock=performance.now();
    render();resize();
    const list=chosenList();
    if(!list.includes(current)) await loadItem(list[0]);
}
async function showHelp() {
    if(loading || gesture || !await flush()) return;
    account();help=true;render();
}
function pointerPosition(e) {
    const box=overlay.getBoundingClientRect();
    return {x:e.clientX-box.left,y:e.clientY-box.top};
}
overlay.addEventListener("contextmenu",e=>e.preventDefault());
overlay.addEventListener("pointerdown",e=>{
    if(loading || gesture) return;
    overlay.focus({preventScroll:true});
    const p=pointerPosition(e);
    if(e.button===2 || (e.button===0&&space)) {
        gesture={type:"pan",start:p,x:offsetX,y:offsetY,pointer:e.pointerId};
    } else if(e.button===0) {
        if(!editable()||ann.skip_image||!image) return;
        const hits=placements().map(it=>({...it,q:screen(it.p)}))
            .filter(it=>Math.hypot(it.q.x-p.x,it.q.y-p.y)<=9)
            .sort((a,b)=>Math.hypot(a.q.x-p.x,a.q.y-p.y)-Math.hypot(b.q.x-p.x,b.q.y-p.y));
        if(hits.length) {
            account();
            gesture={type:"point",key:hits[0].key,original:copy(ann),oldStep:step,start:p,
                pointer:e.pointerId,moved:false};
            step=keys.indexOf(hits[0].key);
            render();
        } else {
            const q=imagePoint(p);
            if(q) applyPoint(q);
        }
    }
    if(gesture) overlay.setPointerCapture(e.pointerId);
    e.preventDefault();
});
overlay.addEventListener("pointermove",e=>{
    cursor=pointerPosition(e);
    if(gesture?.type==="pan") {
        offsetX=gesture.x+cursor.x-gesture.start.x;
        offsetY=gesture.y+cursor.y-gesture.start.y;
    } else if(gesture?.type==="point") {
        if(Math.hypot(cursor.x-gesture.start.x,cursor.y-gesture.start.y)>2) gesture.moved=true;
        if(gesture.moved) {
            const p=bounded(native(cursor)), key=gesture.key;
            if(key.startsWith("crest")) ann[key]={...ann[key],...p};
            else ann.points[key]=p;
        }
    }
    draw();
});
function endGesture(e,cancel=false) {
    if(!gesture) return;
    const g=gesture;gesture=null;
    if(overlay.hasPointerCapture(g.pointer)) overlay.releasePointerCapture(g.pointer);
    if(g.type==="point") {
        account();
        if(cancel) {
            const elapsed=ann.seconds_spent;ann=g.original;ann.seconds_spent=elapsed;
            step=g.oldStep;
        } else if(g.moved) {
            history.push({annotation:g.original,step:g.oldStep});
            localMetadata();enqueueSave();
        }
    }
    render();
}
overlay.addEventListener("pointerup",e=>endGesture(e));
overlay.addEventListener("pointercancel",e=>endGesture(e,true));
overlay.addEventListener("lostpointercapture",e=>{if(gesture) endGesture(e,true);});
overlay.addEventListener("pointerleave",()=>{if(!gesture){cursor=null;draw();}});
overlay.addEventListener("wheel",e=>{
    e.preventDefault();
    if(!image) return;
    const p=pointerPosition(e), nativePoint=native(p);
    zoom=Math.max(1,Math.min(10,zoom*Math.exp(-e.deltaY*0.0015)));
    offsetX=p.x-nativePoint.x*zoom;offsetY=p.y-nativePoint.y*zoom;draw();
},{passive:false});
$("fit").onclick=()=>{
    if(image) centerImage(fitScale());
};
$("zoom3").onclick=()=>centerImage(3);
function filters() {
    base.style.filter=`brightness(${$("brightness").value}%) contrast(${$("contrast").value}%)`;
}
$("brightness").oninput=filters;$("contrast").oninput=filters;
$("resetFilters").onclick=()=>{$("brightness").value=100;$("contrast").value=100;filters();};
$("absent").onclick=()=>setPointState("absent");
$("uncertain").onclick=()=>setPointState("uncertain");
document.querySelectorAll("[data-crest]").forEach(el=>el.onclick=()=>setCrestState(el.dataset.crest));
document.querySelectorAll("[data-half]").forEach(el=>el.onclick=()=>setHalf(el.dataset.half));
$("comment").oninput=()=>mutate(()=>{ann.comment=$("comment").value;});
$("numberingUncertain").onchange=()=>mutate(()=>{ann.numbering_uncertain=$("numberingUncertain").checked;});
$("skipReason").oninput=()=>{
    // При удалении последнего символа причины одновременно снимаем пропуск,
    // поэтому на сервер никогда не попадает пропуск без причины.
    mutate(()=>{
        ann.skip_reason=$("skipReason").value;
        if(!ann.skip_reason.trim()) ann.skip_image=false;
    });
    $("skip").checked=ann.skip_image;
};
$("skip").onchange=()=>{
    const checked=$("skip").checked;
    let reason=$("skipReason").value;
    if(checked&&!reason.trim()) {
        reason=window.prompt("Почему снимок нельзя разметить?","");
        if(!reason?.trim()) {$("skip").checked=false;return;}
        if(reason.length>2000) {error("Причина должна быть не длиннее 2000 символов.");$("skip").checked=false;return;}
    }
    mutate(()=>{ann.skip_image=checked;ann.skip_reason=reason;});
    syncFields();
};
$("undo").onclick=undo;
$("resetImage").onclick=()=>{
    // Полный сброс текущего снимка: одно действие в истории, отменяется через Z
    if(!editable() || !ann) return;
    if(!window.confirm("Стереть всю разметку этого снимка?")) return;
    account();
    const elapsed=ann.seconds_spent;
    history.push({annotation:copy(ann),step});
    ann={...ann, points:{}, crest_left:null, crest_right:null, th12_half_visible:null,
         comment:"", skip_image:false, skip_reason:"", numbering_uncertain:false, seconds_spent:elapsed};
    step=0;
    localMetadata(); enqueueSave(); syncFields(); render();
};
$("back").onclick=()=>navigate(-1);$("next").onclick=()=>navigate(1);
$("retry").onclick=()=>enqueueSave();
$("editTab").onclick=()=>setMode("edit");$("viewTab").onclick=()=>setMode("view");
$("helpTab").onclick=showHelp;$("start").onclick=()=>setMode("edit");
$("filter").onchange=async()=>{const list=chosenList();await loadItem(list[0]);render();};
$("firstUnfinished").onclick=async()=>{
    const item=items.find(it=>!it.complete);
    if(!item) return;
    if(!await flush()) return;
    account();help=false;
    if(mode==="view") $("filter").value="unfinished";
    render();resize();await loadItem(item);
};
document.addEventListener("keydown",e=>{
    if(e.target.closest("input,textarea,select") || e.altKey || e.metaKey) return;
    if(e.code==="Space") {space=true;e.preventDefault();return;}
    if(help||loading||gesture) return;
    if(e.ctrlKey && e.code!=="KeyZ") return;
    const actions={
        KeyN:()=>setPointState("absent"),KeyU:()=>setPointState("uncertain"),
        Digit1:()=>setCrestState("in_frame"),Digit2:()=>setCrestState("partial"),
        Digit3:()=>setCrestState("out_of_frame"),KeyY:()=>setHalf("yes"),
        KeyH:()=>setHalf("no"),KeyC:()=>setHalf("cannot"),KeyZ:undo,
        ArrowLeft:()=>navigate(-1),ArrowRight:()=>navigate(1),Enter:()=>navigate(1)
    };
    if(actions[e.code]) {e.preventDefault();if(!e.repeat) actions[e.code]();}
});
document.addEventListener("keyup",e=>{if(e.code==="Space") space=false;});
document.addEventListener("visibilitychange",()=>{
    // На момент события document.hidden уже обновлён: учитываем предыдущий интервал вручную.
    const now=performance.now();
    if(document.hidden && editable() && lastFocus) {
        ann.seconds_spent+=(now-clock)/1000;timeDirty=true;
    }
    clock=now;
    if(document.hidden && timeDirty && !gesture) enqueueSave();
});
window.addEventListener("blur",()=>{
    account();lastFocus=false;space=false;
    if(gesture) endGesture(null,true);
    if(timeDirty) enqueueSave();
});
window.addEventListener("focus",()=>{lastFocus=true;clock=performance.now();});
window.addEventListener("beforeunload",e=>{
    account();
    if(pending || !saveOK || timeDirty) {
        if(timeDirty&&!pending) enqueueSave();
        e.preventDefault();e.returnValue="";
    }
});
setInterval(()=>{
    if(!gesture) {
        account();
        if(timeDirty&&editable()) enqueueSave();
    }
},15000);
new ResizeObserver(resize).observe($("stage"));
async function boot() {
    try {
        state=await api("/api/state");items=state.images;
        help=state.first_run;
        render();resize();
        await loadItem(items.find(it=>!it.complete)||items[0]);
        saveIndicator();
    } catch(e) {error(e.message);$("saveStatus").textContent="Ошибка загрузки";}
}
boot();
</script>
</body>
</html>
"""


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        super().__init__(address, Handler)
        self.app = app


class Handler(BaseHTTPRequestHandler):
    server_version = "SpineAnnotator/1"

    def log_message(self, fmt, *args):
        # Не журналируем URL с идентификаторами медицинских изображений.
        pass

    def send_bytes(self, status, body, content_type, download=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        if download:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="spine_points_{self.server.app.args.annotator}.json"',
            )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, value, download=False):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8", download)

    def fail(self, status, message):
        self.send_json(status, {"error": message})

    def identifier(self, route, prefix, suffix=""):
        value = route[len(prefix):]
        if suffix:
            if not value.endswith(suffix):
                return None
            value = value[:-len(suffix)]
        value = unquote(value)
        if value not in self.server.app.items:
            return None
        return value

    def do_GET(self):
        app = self.server.app
        route = urlsplit(self.path).path
        try:
            if route == "/":
                self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/state":
                self.send_json(200, app.state())
            elif route == "/api/export":
                with app.lock:
                    self.send_json(200, app.data, download=True)
            elif route.startswith("/api/image/"):
                uid = self.identifier(route, "/api/image/", ".png")
                if uid is None:
                    self.fail(404, "Неизвестный идентификатор изображения.")
                elif uid not in app.pngs:
                    self.fail(422, app.items[uid]["error"])
                else:
                    self.send_bytes(200, app.pngs[uid], "image/png")
            elif route.startswith("/api/annotation/"):
                uid = self.identifier(route, "/api/annotation/")
                if uid is None:
                    self.fail(404, "Неизвестный идентификатор изображения.")
                else:
                    with app.lock:
                        self.send_json(200, app.data["images"].get(uid, app.blank(uid)))
            else:
                self.fail(404, "Страница не найдена.")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.fail(500, "Внутренняя ошибка сервера.")

    def do_PUT(self):
        app = self.server.app
        route = urlsplit(self.path).path
        if not route.startswith("/api/annotation/"):
            self.fail(404, "Страница не найдена.")
            return
        uid = self.identifier(route, "/api/annotation/")
        if uid is None:
            self.fail(404, "Неизвестный идентификатор изображения.")
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.fail(403, "Сохранение разрешено только с этой страницы.")
            return
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.fail(403, "Межсайтовое сохранение запрещено.")
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Ожидается Content-Type: application/json.")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Передавайте JSON с известной длиной.")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ValueError("Некорректная длина запроса.") from None
            if length <= 0 or length > 1_000_000:
                raise ValueError("Пустой или слишком большой запрос.")
            self.connection.settimeout(30)
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("Запрос получен не полностью.")
            try:
                raw = json.loads(content.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError("Некорректный JSON.") from None
            with app.lock:
                cleaned = app.validate(uid, raw)
                previous = app.data["images"].get(uid)
                if previous:
                    cleaned["seconds_spent"] = max(
                        cleaned["seconds_spent"], previous.get("seconds_spent", 0)
                    )
                updated = dict(app.data)
                updated["images"] = {**app.data["images"], uid: cleaned}
                try:
                    atomic_write(app.filename, updated)
                except OSError:
                    self.fail(500, "Не удалось записать файл разметки. Проверьте место и права доступа.")
                    return
                app.data = updated
            self.send_json(200, cleaned)
        except ValueError as exc:
            self.fail(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError:
            self.fail(400, "Истекло время получения запроса.")
        except Exception:
            self.fail(500, "Внутренняя ошибка сохранения.")


def main():
    parser = argparse.ArgumentParser(
        description="Локальная ручная разметка ориентиров позвоночника на DXA."
    )
    parser.add_argument("--index", required=True, help="CSV-индекс изображений")
    parser.add_argument("--root", default=".", help="Корень относительных путей DICOM")
    parser.add_argument("--annotator", required=True, help="Имя разметчика: латиница, цифры, _ и -")
    parser.add_argument("--part", required=True, choices=("1", "2", "all"), help="Часть исследований")
    parser.add_argument("--port", type=int, default=8765, help="Порт сервера (8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Адрес сервера (127.0.0.1)")
    parser.add_argument("--out", default="data/annotations", help="Папка результатов")
    parser.add_argument("--reset", action="store_true",
                        help="Начать разметку заново: старый файл переименовывается в *.bak-<дата>")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.annotator):
        parser.error("--annotator: разрешены только латинские буквы, цифры, _ и -.")
    if not 1 <= args.port <= 65535:
        parser.error("--port должен быть от 1 до 65535.")
    if args.host not in ("127.0.0.1", "localhost"):
        print("ВНИМАНИЕ: адрес не является стандартным loopback; не открывайте медицинские данные наружу.")
    if args.reset:
        existing = pathlib.Path(args.out) / f"spine_points_{args.annotator}.json"
        if existing.exists():
            backup = existing.with_suffix(f".bak-{datetime.now():%Y%m%d-%H%M%S}.json")
            existing.rename(backup)
            print(f"Старая разметка сохранена в {backup}")
    try:
        app = Application(args)
        server = Server((args.host, args.port), app)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Ошибка запуска: {exc}\n")
    print(f"\nАдрес: http://{args.host}:{args.port}", flush=True)
    print("При удалённой работе откройте через проброс порта VS Code.", flush=True)
    print("Используйте одну вкладку на разметчика. Остановка: Ctrl+C.", flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
