# tools/hip_annotator.py
#
# pip install pydicom pillow numpy
#
# python tools/hip_annotator.py \
#   --index data/index/images.csv --root . --annotator ivan
#
# python tools/hip_annotator.py \
#   --report data/annotations/hip_points_ivan.json
#
# Незавершённые снимки хранятся отдельно: session.drafts.
# images содержит только done / skipped.
# Очередь и текущая позиция сохраняются в session; пути и study не сохраняются.
#
# Масштаб протокола: x = 0.600, y = 0.606 мм/пиксель.
# Изображение не отражается и не поворачивается; "ниже" = увеличение y.
#
# Повторяемость:
# SD = sqrt(sum(||repeat - original||^2) / (2*n)).
# Это парная внутрисубъектная 2D SD (technical error of measurement),
# без вычитания систематического сдвига.
# Максимум = максимальное расстояние между двумя постановками.
# Для приёмки SD сравнивается с заданным порогом; максимум информативный.

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import tempfile
import threading
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


SCHEMA = "dxa-hip-points/1"
POINTS = ("H", "B1", "B2", "T", "D", "D2")
STATES = ("visible", "not_visible", "out_of_frame", "uncertain")
MM_X = 0.600
MM_Y = 0.606


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def binary_label(value):
    """В CSV pandas пишет 0.0 / 1.0; пустая клетка — неизвестная метка."""
    text = str(value or "").strip()
    if not text:
        return None
    number = float(text)
    if number not in (0.0, 1.0):
        raise ValueError(f"Недопустимая бинарная метка: {text!r}")
    return int(number)


def digest(*parts):
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def atomic_write(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=filename.parent,
            prefix="." + filename.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2,
                      allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
        temporary = None
        # На POSIX закрепляем также замену записи каталога.
        if os.name == "posix":
            fd = os.open(str(filename.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def read_json(filename):
    def reject_constant(value):
        raise ValueError(f"Недопустимое JSON-число: {value}")
    with open(filename, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def safe_path(root, relative):
    rel = Path(relative)
    resolved = (root / rel).resolve()
    if rel.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("Путь DICOM находится вне --root.")
    return resolved


def read_dicom(filename, rows, cols):
    # Импорты здесь: --report не требует DICOM-зависимостей.
    import numpy as np
    import pydicom
    from PIL import Image

    try:
        ds = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        ds = pydicom.dcmread(filename, force=True)

    pixels = np.asarray(ds.pixel_array, dtype=np.float64)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise ValueError("Ожидался один монохромный кадр.")
    if pixels.shape != (rows, cols):
        raise ValueError(
            f"Фактический размер {pixels.shape} не совпадает "
            f"с CSV ({rows}, {cols})."
        )
    photo = str(getattr(ds, "PhotometricInterpretation", ""))
    if photo not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Поддерживаются только MONOCHROME1/MONOCHROME2.")

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Некорректный RescaleSlope/Intercept.")
    pixels = pixels * slope + intercept
    mask = np.isfinite(pixels)
    if not mask.any():
        raise ValueError("Нет конечных значений пикселей.")
    low, high = float(pixels[mask].min()), float(pixels[mask].max())
    pixels = np.nan_to_num(pixels, nan=low, posinf=high, neginf=low)
    if high > low:
        pixels = (pixels - low) / (high - low)
    else:
        pixels = np.zeros_like(pixels)
    if photo == "MONOCHROME1":
        pixels = 1 - pixels
    pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def select_images(index):
    required = {
        "sop_uid", "study", "path", "rows", "cols",
        "region", "side", "y_pos",
    }
    rows = []
    seen = set()
    with open(index, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Нет колонок: " + ", ".join(sorted(missing)))
        for line, raw in enumerate(reader, 2):
            if raw["region"].strip() != "hip":
                continue
            uid = raw["sop_uid"].strip()
            study = raw["study"].strip()
            if not uid or "#" in uid or not study:
                raise ValueError(f"Строка {line}: неверный sop_uid/study.")
            if uid in seen:
                raise ValueError(f"Повтор sop_uid в CSV: {uid}")
            seen.add(uid)
            side = raw["side"].strip()
            if side not in ("L", "R"):
                raise ValueError(f"Строка {line}: side должен быть L/R.")
            dimensions = []
            for name in ("rows", "cols"):
                number = float(raw[name])
                if not math.isfinite(number) or number < 1 or not number.is_integer():
                    raise ValueError(f"Строка {line}: неверный {name}.")
                dimensions.append(int(number))
            rows.append({
                "uid": uid,
                "study": study,
                "path": raw["path"].strip(),
                "rows": dimensions[0],
                "cols": dimensions[1],
                "side": side,
                "y_pos": binary_label(raw["y_pos"]),
            })

    positive_studies = {r["study"] for r in rows if r["y_pos"] == 1}
    paired = [r for r in rows if r["study"] in positive_studies]
    counts = Counter(r["study"] for r in paired)
    positive_counts = Counter(
        r["study"] for r in paired if r["y_pos"] == 1
    )
    distribution = Counter(positive_counts.values())
    # Фактический состав (проверен 20.09.2026): 24 исследования с положительной
    # меткой, 47 снимков. В 23 из них по два снимка бедра (12 с двумя
    # положительными, 11 с одним), и одно исследование с единственным снимком —
    # у него пары нет, сравнивать левое с правым там не получится.
    singles = [study for study, n in counts.items() if n == 1]
    if (
        len(positive_studies) != 24
        or len(paired) != 47
        or any(n not in (1, 2) for n in counts.values())
        or len(singles) != 1
        or distribution != Counter({2: 12, 1: 12})
    ):
        raise ValueError(
            "Состав положительных исследований не соответствует протоколу: "
            f"исследований={len(positive_studies)}, снимков={len(paired)}, "
            f"распределение числа положительных={dict(distribution)}. "
            "Ожидаются 24 исследования (47 снимков): 23 по два снимка — "
            "12 с двумя положительными и 11 с одним — плюс одно исследование "
            "с единственным снимком. Проверьте CSV."
        )
    for study in positive_studies:
        sides = {r["side"] for r in paired if r["study"] == study}
        if len(sides) != sum(1 for r in paired if r["study"] == study):
            raise ValueError(
                "В положительном исследовании стороны снимков совпадают: "
                "ожидается либо пара L/R, либо единственный снимок."
            )

    negatives = [
        r for r in rows
        if r["study"] not in positive_studies and r["y_pos"] == 0
    ]
    if len(negatives) < 25:
        raise ValueError("Недостаточно отрицательных снимков: нужно 25.")
    # Воспроизводимая псевдослучайная выборка, независимая от порядка CSV.
    negatives.sort(key=lambda r: (
        digest("hip-negative-sample-v1", r["study"], r["uid"]), r["uid"]
    ))
    selected = paired + negatives[:25]
    selected.sort(key=lambda r: (
        digest(r["study"], r["uid"]), r["uid"]
    ))
    return selected


def make_order(rows):
    base = [r["uid"] for r in rows]
    half = len(base) // 2
    sources = [base[i] for i in (6, 13, 20, 27, 34)]
    # Случайные, но воспроизводимые промежутки второй половины.
    slots = list(range(half, len(base)))
    seed = digest(*base)
    slots.sort(key=lambda i: digest("hip-repeat-slot-v1", seed, str(i)))
    sources.sort(key=lambda uid: digest("hip-repeat-source-v1", seed, uid))
    insertions = dict(zip(slots[:5], sources))
    order = base[:half]
    for i in range(half, len(base)):
        if i in insertions:
            order.append(insertions[i] + "#2")
        order.append(base[i])
    order.extend(uid + "#2" for uid in base[:5])
    # 10 повторов: 5 вразнобой во второй половине и 5 первых снимков в конце.
    expected = len(base) + 10
    assert len(order) == expected and len(set(order)) == expected
    return order


def validate_points(raw, rows, cols):
    if not isinstance(raw, dict) or set(raw) - set(POINTS):
        raise ValueError("Некорректный набор точек.")
    result = {}
    for name, point in raw.items():
        if not isinstance(point, dict):
            raise ValueError("Точка должна быть объектом.")
        if set(point) - {"x", "y", "state", "confidence"}:
            raise ValueError("Неизвестное поле точки.")
        state = point.get("state")
        confidence = point.get("confidence")
        if state not in STATES:
            raise ValueError("Неизвестное состояние точки.")
        if type(confidence) is not int or confidence not in (1, 2, 3):
            raise ValueError("Уверенность должна быть 1–3.")
        has_x, has_y = "x" in point, "y" in point
        if has_x != has_y:
            raise ValueError("Координаты задаются парой.")
        has_coordinates = has_x and has_y
        if state == "visible" and not has_coordinates:
            raise ValueError("Для visible нужны координаты.")
        if state in ("not_visible", "out_of_frame") and has_coordinates:
            raise ValueError("Для невидимой точки координаты запрещены.")
        clean = {"state": state, "confidence": confidence}
        if has_coordinates:
            x, y = point["x"], point["y"]
            if (
                not finite(x) or not finite(y)
                or not 0 <= x < cols or not 0 <= y < rows
            ):
                raise ValueError("Координаты вне исходного изображения.")
            clean.update(x=float(x), y=float(y))
        result[name] = clean
    return result


class Conflict(Exception):
    pass


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.filename = Path(args.out) / f"hip_points_{args.annotator}.json"
        self.rows = select_images(args.index)
        part = str(getattr(args, "part", "all"))
        if part != "all":
            # Делим по исследованиям: иначе левое и правое бедро одного
            # пациента уедут к разным разметчикам и сравнение пары превратится
            # в сравнение двух людей.
            from annotation_split import describe as describe_split, split_studies
            all_studies = [r["study"] for r in self.rows]
            summary = describe_split(all_studies, part, overlap=6, salt="hip")
            mine = split_studies(all_studies, part, overlap=6, salt="hip")
            self.rows = [r for r in self.rows if r["study"] in mine]
            positives = sum(1 for r in self.rows if r["y_pos"] == 1)
            print(f"{summary}; снимков {len(self.rows)}, "
                  f"из них положительных {positives}", flush=True)
        self.items = {r["uid"]: r for r in self.rows}
        self.order = make_order(self.rows)
        self.pngs = {}
        root = Path(args.root).resolve()

        # Проверяем каждый выбранный DICOM до открытия браузера.
        # Ошибку не превращаем молча в "пропущен".
        for row in self.rows:
            try:
                self.pngs[row["uid"]] = read_dicom(
                    safe_path(root, row["path"]), row["rows"], row["cols"]
                )
            except Exception as exc:
                raise ValueError(
                    f"DICOM {row['uid']}: {type(exc).__name__}: {exc}"
                ) from exc

        # Только хеш конфигурации; ни study, ни пути в JSON не попадают.
        fingerprint = digest(
            "hip-protocol-v1",
            *[
                digest(
                    r["uid"], r["study"], str(r["rows"]), str(r["cols"]),
                    r["side"], str(r["y_pos"])
                )
                for r in self.rows
            ],
        )
        self.data = {
            "schema": SCHEMA,
            "annotator": args.annotator,
            "updated": now_iso(),
            "images": {},
            "session": {
                "order": self.order,
                "fingerprint": fingerprint,
                "cursor": 0,
                "revision": 0,
                "drafts": {},
            },
        }

        if self.filename.exists():
            loaded = read_json(self.filename)
            if (
                not isinstance(loaded, dict)
                or loaded.get("schema") != SCHEMA
                or loaded.get("annotator") != args.annotator
                or not isinstance(loaded.get("images"), dict)
            ):
                raise ValueError("Неподходящий файл разметки.")
            session = loaded.get("session")
            if not isinstance(session, dict):
                raise ValueError("Нет сохранённой сессии; файл не перезаписан.")
            if (
                session.get("order") != self.order
                or session.get("fingerprint") != fingerprint
            ):
                raise ValueError(
                    "Изменились индекс/очередь/протокол. "
                    "Используйте другую папку --out; старый файл не изменён."
                )
            cursor = session.get("cursor")
            revision = session.get("revision")
            if (
                type(cursor) is not int or not 0 <= cursor < len(self.order)
                or type(revision) is not int or revision < 0
                or not isinstance(session.get("drafts"), dict)
            ):
                raise ValueError("Повреждённая сессия.")
            if set(loaded["images"]) & set(session["drafts"]):
                raise ValueError("Снимок одновременно в images и drafts.")
            for records, completed in (
                (loaded["images"], True), (session["drafts"], False)
            ):
                for key, annotation in records.items():
                    if key not in self.order:
                        raise ValueError("Неизвестный ключ в файле разметки.")
                    clean = self.validate(key, annotation, completed)
                    # Проверка не должна менять исторический timestamp.
                    if clean != annotation:
                        raise ValueError("Неподходящий формат сохранённой записи.")
            self.data = loaded
        else:
            atomic_write(self.filename, self.data)

        repeats = sum(1 for key in self.order if key.endswith("#2"))
        print(f"Основных снимков: {len(self.order) - repeats}; "
              f"дополнительных показов: {repeats}.", flush=True)
        print(f"Сохранение: {self.filename}", flush=True)

    def item(self, key):
        return self.items[key.split("#", 1)[0]]

    def blank(self, key):
        item = self.item(key)
        return {
            "side": item["side"],
            "y_pos": item["y_pos"],
            "points": {},
            "comment": "",
            "seconds": 0.0,
            "updated": now_iso(),
        }

    def validate(self, key, raw, completed):
        if not isinstance(raw, dict):
            raise ValueError("Разметка должна быть объектом.")
        allowed = {
            "state", "side", "y_pos", "points", "comment", "seconds", "updated"
        }
        if set(raw) - allowed:
            raise ValueError("Неизвестные поля разметки.")
        item = self.item(key)
        if raw.get("side") != item["side"] or raw.get("y_pos") != item["y_pos"]:
            raise ValueError("side/y_pos не совпадают с CSV.")
        result = self.blank(key)
        result["points"] = validate_points(
            raw.get("points"), item["rows"], item["cols"]
        )
        comment = raw.get("comment")
        seconds = raw.get("seconds")
        if not isinstance(comment, str) or len(comment) > 20000:
            raise ValueError("Комментарий должен быть строкой до 20000 символов.")
        if not finite(seconds) or seconds < 0:
            raise ValueError("Некорректное время.")
        updated = raw.get("updated", now_iso())
        if not isinstance(updated, str):
            raise ValueError("Некорректный timestamp.")
        if completed:
            state = raw.get("state")
            if state not in ("done", "skipped"):
                raise ValueError("Завершённый снимок: done или skipped.")
            if state == "done" and set(result["points"]) != set(POINTS):
                raise ValueError("Для done нужны состояния всех шести точек.")
            result["state"] = state
        elif "state" in raw:
            raise ValueError("У черновика нет состояния done/skipped.")
        result.update(comment=comment, seconds=float(seconds), updated=updated)
        return result

    def view(self, index):
        with self.lock:
            key = self.order[index]
            item = self.item(key)
            annotation = self.data["images"].get(
                key, self.data["session"]["drafts"].get(key, self.blank(key))
            )
            return {
                "index": index,
                "total": len(self.order),
                "rows": item["rows"],
                "cols": item["cols"],
                "side": item["side"],
                # Не показываем метку, study, uid или признак повтора в UI.
                "annotation": copy.deepcopy(annotation),
                "finished": len(self.data["images"]),
                "revision": self.data["session"]["revision"],
            }

    def save(self, raw):
        with self.lock:
            if not isinstance(raw, dict):
                raise ValueError("Ожидается JSON-объект.")
            if raw.get("revision") != self.data["session"]["revision"]:
                raise Conflict(
                    "Сессия изменена в другой вкладке. "
                    "Перезагрузите страницу; несохранённые изменения "
                    "сначала скопируйте."
                )
            index = raw.get("index")
            cursor = raw.get("cursor", index)
            for value in (index, cursor):
                if type(value) is not int or not 0 <= value < len(self.order):
                    raise ValueError("Неверная позиция очереди.")
            key = self.order[index]
            mode = raw.get("mode")
            if mode not in ("draft", "done", "skipped"):
                raise ValueError("Неизвестное действие.")
            annotation = copy.deepcopy(raw.get("annotation"))
            if not isinstance(annotation, dict):
                raise ValueError("Нет разметки.")
            annotation.pop("state", None)
            if mode != "draft":
                annotation["state"] = mode
            clean = self.validate(key, annotation, mode != "draft")
            old = self.data["images"].get(
                key, self.data["session"]["drafts"].get(key)
            )
            if old:
                clean["seconds"] = max(clean["seconds"], old["seconds"])
            clean["updated"] = now_iso()

            updated = copy.deepcopy(self.data)
            updated["images"].pop(key, None)
            updated["session"]["drafts"].pop(key, None)
            destination = (
                updated["session"]["drafts"] if mode == "draft"
                else updated["images"]
            )
            destination[key] = clean
            updated["session"]["cursor"] = cursor
            updated["session"]["revision"] += 1
            updated["updated"] = now_iso()
            atomic_write(self.filename, updated)
            self.data = updated
            return {
                "revision": updated["session"]["revision"],
                "finished": len(updated["images"]),
            }


def report(filename):
    data = read_json(filename)
    if data.get("schema") != SCHEMA or not isinstance(data.get("images"), dict):
        raise ValueError("Не тот формат файла.")
    images = data["images"]
    base = {k: v for k, v in images.items() if "#" not in k}
    repeat = {k: v for k, v in images.items() if k.endswith("#2")}

    def counts(records):
        return Counter(v.get("state") for v in records.values())

    a, b = counts(base), counts(repeat)
    print(f"Основные: размечено {a['done']}, пропущено {a['skipped']}.")
    print(f"Повторы: размечено {b['done']}, пропущено {b['skipped']}.")
    print("Черновиков:", len(data.get("session", {}).get("drafts", {})))
    print("\nДоли состояний: только основные снимки со state=done.")
    print("Повторы не удваивают статистику классов; пропущенные исключены.")
    for label, title in ((1, "Положительные"), (0, "Отрицательные")):
        records = [
            v for v in base.values()
            if v.get("state") == "done" and v.get("y_pos") == label
        ]
        print(f"\n{title}, n={len(records)}")
        for point in POINTS:
            c = Counter(
                v.get("points", {}).get(point, {}).get("state", "missing")
                for v in records
            )
            values = [
                f"{state}={c[state]}/{len(records)} "
                f"({100*c[state]/len(records):.1f}%)"
                if records else f"{state}=—"
                for state in STATES
            ]
            print(f"  {point:2}: " + "; ".join(values))
            if c["missing"]:
                print(f"      ВНИМАНИЕ: нет состояния у {c['missing']} записей.")
    unknown = sum(
        v.get("state") == "done" and v.get("y_pos") not in (0, 1)
        for v in base.values()
    )
    if unknown:
        print(f"\nНеизвестный y_pos: {unknown}; в доли классов не включены.")

    pairs = []
    for key, second in repeat.items():
        first = images.get(key[:-2])
        if (
            first and first.get("state") == "done"
            and second.get("state") == "done"
        ):
            pairs.append((first, second))
    print(f"\nЗавершённых пар повторов: {len(pairs)}/10.")
    print(
        "SD = sqrt(sum(dx_mm² + dy_mm²)/(2*n)); "
        "max — максимальное парное расстояние."
    )
    print("Координаты сравниваются только при visible в обоих показах.")
    failed = False
    insufficient = False
    for name in POINTS:
        distances = []
        state_disagreements = 0
        for first, second in pairs:
            p = first.get("points", {}).get(name, {})
            q = second.get("points", {}).get(name, {})
            if p.get("state") != q.get("state"):
                state_disagreements += 1
            if p.get("state") != "visible" or q.get("state") != "visible":
                continue
            if not all(finite(t.get(axis)) for t in (p, q) for axis in ("x", "y")):
                continue
            distances.append(math.hypot(
                (q["x"] - p["x"]) * MM_X,
                (q["y"] - p["y"]) * MM_Y,
            ))
        limit = 3.0 if name == "T" else 1.5
        n = len(distances)
        if n:
            sd = math.sqrt(sum(d*d for d in distances) / (2*n))
            maximum = max(distances)
            if n < 2:
                verdict = "недостаточно пар для приёмки"
                insufficient = True
            else:
                verdict = "проходит" if sd <= limit else "не проходит"
            if sd > limit:
                failed = True
                verdict = "не проходит"
            print(
                f"  {name:2}: n={n}, SD={sd:.3f} мм, max={maximum:.3f} мм, "
                f"порог SD≤{limit:g} мм — {verdict}; "
                f"расхождений состояния={state_disagreements}"
            )
        else:
            insufficient = True
            print(
                f"  {name:2}: n=0, SD/max=—, порог SD≤{limit:g} мм — "
                f"недостаточно данных; расхождений состояния={state_disagreements}"
            )

    if failed:
        print(
            "\nВНИМАНИЕ: порог не пройден. Разметку стоит остановить "
            "и уточнить правила."
        )
    elif len(pairs) < 10 or insufficient:
        print("\nПроверка пока неполная; общая приёмка не подтверждена.")
    else:
        print("\nВсе оцениваемые точки проходят заданные пороги SD.")
    print("Невидимость T не заменяется вымышленной координатой.")


HTML = r"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DXA · точки бедра</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#101820;color:#e8eff8}
header{padding:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button,input,textarea,select{font:inherit}
button,select{padding:7px;background:#253c51;color:white;border:1px solid #607990;
border-radius:4px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
input,textarea{background:#192b3b;color:white;border:1px solid #607990}
textarea{width:100%;min-height:70px}
main{display:grid;grid-template-columns:minmax(300px,1fr) 370px;height:calc(100vh - 65px)}
section{display:flex;flex-direction:column;min-height:0;min-width:0}
#tools{padding:8px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
#stage{position:relative;flex:1;min-height:200px;background:#030609;overflow:hidden}
canvas{position:absolute;width:100%;height:100%;touch-action:none;cursor:crosshair}
aside{padding:12px;overflow:auto;background:#1a2939}
#side{font-size:38px;font-weight:bold}
#points button{display:block;width:100%;text-align:left;margin:7px 0}
#points button.active{outline:2px solid #ffe274;background:#3e5160}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}
.small{font-size:12px;line-height:1.5;color:#bfd1df}
#error{color:#ffd0bf;white-space:pre-wrap}
#coords{padding:6px;color:#bacddd}
#save{margin-left:auto}
input[type=range]{width:100px}
#veil{position:absolute;inset:0;background:#101820c0;display:grid;place-items:center}
#veil[hidden]{display:none}
@media(max-width:800px){main{display:flex;flex-direction:column;height:auto}
section{height:65vh}aside{overflow:visible}}
</style>
<header>
<b>DXA · точки бедра</b>
<span id="progress"></span><span id="timer"></span><span id="save"></span>
<button id="retry" hidden>Повторить сохранение</button>
</header>
<main>
<section>
<div id="tools">
<button id="fit">Вписать</button><button id="zoom3">×3</button>
<label>Яркость <input id="brightness" type="range" min="30" max="250" value="100"></label>
<label>Контраст <input id="contrast" type="range" min="30" max="350" value="100"></label>
<button id="reset">Сброс</button>
</div>
<div id="stage"><canvas id="canvas"></canvas><div id="veil">Загрузка…</div></div>
<div id="coords">Координаты исходного DICOM</div>
</section>
<aside>
<div id="side"></div>
<div id="position"></div>
<div id="error"></div>
<div class="row">
<button id="prev">P · Назад</button>
<button id="next">N · Далее</button>
<button id="skip">S · Пропустить</button>
</div>
<div id="points"></div>
<div id="description" class="small"></div>
<div class="row">
<button data-state="visible">V · visible</button>
<button data-state="not_visible">X · not visible</button>
<button data-state="out_of_frame">O · вне кадра</button>
<button data-state="uncertain">? · uncertain</button>
</div>
<label>Уверенность:
<select id="confidence">
<option value="1">1 — низкая</option>
<option value="2">2 — средняя</option>
<option value="3" selected>3 — высокая</option>
</select></label>
<div class="row"><button id="delete">D · Удалить точку</button></div>
<div id="guideInfo" class="small"></div>
<p class="small">
Координаты невидимых точек не угадывайте. X/O удаляют координату.
Uncertain допускает координату, только если есть обоснованная постановка.
</p>
<label>Комментарий<textarea id="comment" maxlength="20000"></textarea></label>
<p class="small">
1–6 — точка. Shift+1–3 или NumPad1–3 — уверенность.<br>
ЛКМ — поставить; затем выбирается следующая точка.<br>
Колесо — зум; ПКМ или Space+мышь — панорама.<br>
D/D₂ ставятся на соответствующей направляющей, y привязывается к линии.<br>
N завершает снимок, если заданы все шесть точек; иначе сохраняет черновик.
S сохраняет пропуск и переходит дальше. Пропуск сохраняет уже введённые точки.<br>
Изображение не отражается. Сторона L/R сама по себе не определяет,
где на экране медиальная сторона: ориентируйтесь на анатомию.<br>
Таймер учитывает время, пока вкладка видима и активна.
</p>
</aside>
</main>
<script>
"use strict";
const TOKEN=__TOKEN__;
const $=id=>document.getElementById(id);
const names=["H","B1","B2","T","D","D2"];
const labels=["H","B₁","B₂","T","D","D₂"];
const descriptions=[
 "Центр головки бедренной кости.",
 "Верхний край шейки в самом узком её месте.",
 "Нижний край шейки в том же сечении, что B₁.",
 "Наиболее медиально выступающая точка контура малого вертела.",
 "Середина между контурами диафиза на линии +20 мм.",
 "Середина между контурами диафиза на линии +50 мм."
];
let current=null,ann=null,image=null,index=0,selected=0;
let preferredState="visible",preferredConfidence=3;
let W=1,H=1,zoom=1,ox=0,oy=0,gesture=null,space=false;
let revision=0,finished=0,pending=0,failed=false,busy=true;
let tail=Promise.resolve(),last=performance.now(),dirty=false;
let conflict=false;

function clone(v){return JSON.parse(JSON.stringify(v))}
async function api(url, options={}){
 const r=await fetch(url,{cache:"no-store",...options,
  headers:{"X-Hip-Token":TOKEN,"Content-Type":"application/json",
   ...(options.headers||{})}});
 let j;
 try{j=await r.json()}catch(e){throw Error("Некорректный ответ сервера")}
 if(!r.ok){
  const e=Error(j.error||`HTTP ${r.status}`);e.conflict=r.status===409;throw e
 }
 return j
}
function error(text=""){$("error").textContent=text}
function active(){return !!ann&&!busy&&!failed}
function account(){
 const now=performance.now();
 if(ann&&!busy&&!document.hidden&&document.hasFocus()){
  const dt=Math.max(0,(now-last)/1000);
  ann.seconds+=dt;
  if(dt>0)dirty=true;
 }
 last=now;
}
function mode(){
 if(ann.state==="skipped")return "skipped";
 return names.every(k=>ann.points[k])?"done":"draft";
}
function autoQueue(){
 // Автосохранение сохраняет ЧЕРНОВИК: раньше оно писало «готово» сразу после
 // шестой точки, и снимок считался завершённым до того, как человек его
 // проверил и нажал N (находка astra, 21.09.2026). Уже завершённые записи
 // статус сохраняют.
 const keep=(ann&&(ann.state==="done"||ann.state==="skipped"))?ann.state:"draft";
 return queueSave(index,keep)
}
function saveStatus(){
 $("save").textContent=failed?"НЕ СОХРАНЕНО":pending?"Сохранение…":
  dirty?"Изменения в памяти":"Сохранено ✓";
 $("retry").hidden=!failed||conflict;
}
function queueSave(cursor=index, requestedMode=null){
 if(!ann)return Promise.resolve(false);
 account();
 const snapshot=clone(ann),capturedIndex=index;
 const capturedMode=requestedMode||mode();
 pending++;saveStatus();
 tail=tail.then(async()=>{
  if(failed)return false;
  try{
   const result=await api("/api/save",{
    method:"POST",
    body:JSON.stringify({index:capturedIndex,cursor,mode:capturedMode,
     annotation:snapshot,revision})
   });
   revision=result.revision;finished=result.finished;
   return true;
  }catch(e){
   failed=true;conflict=!!e.conflict;
   error(e.message+" Изменения сохранены в памяти этой вкладки. Не закрывайте её.");
   return false;
  }
 }).finally(()=>{
  pending--;
  if(!pending&&!failed)dirty=false;
  saveStatus();renderHeader();
 });
 return tail;
}
function mutate(fn){
 if(!active())return;
 account();fn();delete ann.state;dirty=true;
 render();autoQueue();
}
function choose(i){
 selected=i;
 const p=ann?.points[names[i]];
 preferredState=p?.state||"visible";preferredConfidence=p?.confidence||3;
 render();
}
function setState(state){
 if(!active())return;
 preferredState=state;
 const key=names[selected],p=ann.points[key];
 if(state==="visible"&&!p?.hasOwnProperty("x")){
  // visible без координат — только режим следующего клика, не фиктивная точка.
  error("Поставьте координату кликом по снимку.");render();return
 }
 mutate(()=>{
  const q={state,confidence:p?.confidence||preferredConfidence};
  if((state==="visible"||state==="uncertain")&&p?.hasOwnProperty("x")){
   q.x=p.x;q.y=p.y
  }
  ann.points[key]=q;
 });
}
function setConfidence(value){
 if(!active())return;
 preferredConfidence=value;
 if(ann.points[names[selected]]){
  mutate(()=>ann.points[names[selected]].confidence=value);
 }else render();
}
function guide(){
 if(!ann)return null;
 const p=ann.points,t=p.T;
 if(t&&Number.isFinite(t.y)&&["visible","uncertain"].includes(t.state))
  return {y:t.y,source:"T"};
 if(t&&["not_visible","out_of_frame","uncertain"].includes(t.state)
    &&Number.isFinite(p.B1?.y)&&Number.isFinite(p.B2?.y))
  return {y:(p.B1.y+p.B2.y)/2,source:"середины B₁B₂"};
 return null;
}
function level(key){
 const g=guide();
 return g?g.y+(key==="D"?20:50)/0.606:null;
}
function renderHeader(){
 $("progress").textContent=current?`Завершено ${finished}/${current.total}`:"";
 $("position").textContent=current?`Показ ${index+1} из ${current.total}`:"";
 $("timer").textContent=ann?`${Math.floor(ann.seconds)} с`:"";
}
function render(){
 renderHeader();saveStatus();
 $("side").textContent=current?`${current.side} · ${current.side==="L"?"левое":"правое"} бедро`:"";
 $("description").textContent=descriptions[selected];
 $("confidence").value=preferredConfidence;
 const container=$("points");container.replaceChildren();
 names.forEach((name,i)=>{
  const p=ann?.points[name],b=document.createElement("button");
  b.className=selected===i?"active":"";
  b.textContent=`${i+1} · ${labels[i]} — ${p?p.state+" · "+p.confidence:"не задана"}`;
  b.onclick=()=>choose(i);container.append(b)
 });
 const g=guide();
 $("guideInfo").textContent=g?
  `Направляющие от ${g.source}: D y=${level("D").toFixed(1)}, `+
  `D₂ y=${level("D2").toFixed(1)} пикс. Если линия ниже кадра — out_of_frame.`:
  "Для направляющих поставьте T; если T нельзя поставить, укажите её состояние и поставьте B₁/B₂.";
 document.querySelectorAll("[data-state]").forEach(b=>{
  b.style.outline=b.dataset.state===preferredState?"2px solid #ffe274":"none";
 });
 draw()
}
function draw(){
 const c=$("canvas").getContext("2d");
 c.clearRect(0,0,W,H);
 if(!image)return;
 c.save();
 c.filter=`brightness(${$("brightness").value}%) contrast(${$("contrast").value}%)`;
 c.imageSmoothingEnabled=false;
 c.drawImage(image,ox,oy,current.cols*zoom,current.rows*zoom);
 c.restore();
 c.font="bold 13px system-ui";
 if(guide()){
  for(const key of ["D","D2"]){
   const y=level(key),sy=oy+y*zoom;
   if(y<0||y>=current.rows)continue;
   c.strokeStyle=key==="D"?"#5cd8ff":"#bdadff";
   c.fillStyle=c.strokeStyle;c.setLineDash([7,5]);c.lineWidth=1;
   c.beginPath();c.moveTo(ox,sy);c.lineTo(ox+current.cols*zoom,sy);c.stroke();
   c.setLineDash([]);
   c.fillText(key==="D"?"D · +20 мм":"D₂ · +50 мм",Math.max(4,ox+6),sy-5)
  }
 }
 if(!ann)return;
 names.forEach((key,i)=>{
  const p=ann.points[key];if(!Number.isFinite(p?.x))return;
  const x=ox+p.x*zoom,y=oy+p.y*zoom;
  c.strokeStyle=i===selected?"#fff077":p.state==="uncertain"?"#ff8ea4":"#71ffc4";
  c.fillStyle=c.strokeStyle;c.lineWidth=2;
  c.beginPath();c.arc(x,y,5,0,2*Math.PI);c.stroke();
  c.beginPath();c.moveTo(x-9,y);c.lineTo(x+9,y);
  c.moveTo(x,y-9);c.lineTo(x,y+9);c.stroke();
  c.fillText(labels[i],x+10,y-7);
 });
 const b1=ann.points.B1,b2=ann.points.B2;
 if(Number.isFinite(b1?.x)&&Number.isFinite(b2?.x)){
  c.strokeStyle="#a8b8c8";c.lineWidth=1;c.beginPath();
  c.moveTo(ox+b1.x*zoom,oy+b1.y*zoom);
  c.lineTo(ox+b2.x*zoom,oy+b2.y*zoom);c.stroke()
 }
}
function resize(){
 const r=$("stage").getBoundingClientRect(),d=devicePixelRatio||1,c=$("canvas");
 W=r.width;H=r.height;c.width=Math.round(W*d);c.height=Math.round(H*d);
 c.getContext("2d").setTransform(d,0,0,d,0,0);draw()
}
function center(z){
 if(!current)return;
 zoom=z;ox=(W-current.cols*z)/2;oy=(H-current.rows*z)/2;draw()
}
function fit(){
 if(current)center(Math.max(.01,Math.min((W-20)/current.cols,(H-20)/current.rows)))
}
function pointer(e){
 const r=$("canvas").getBoundingClientRect();
 return {x:e.clientX-r.left,y:e.clientY-r.top}
}
function native(p){return {x:(p.x-ox)/zoom,y:(p.y-oy)/zoom}}
function put(p){
 if(!active()||!image)return;
 const q=native(p),key=names[selected];
 if(q.x<0||q.y<0||q.x>=current.cols||q.y>=current.rows)return;
 if(["not_visible","out_of_frame"].includes(preferredState)){
  error("Для координаты выберите V или ?.");return
 }
 if(key==="D"||key==="D2"){
  const y=level(key);
  if(y===null){error("Сначала задайте источник направляющих.");return}
  if(y<0||y>=current.rows){error("Уровень вне кадра: используйте O.");return}
  q.y=y;
 }
 error();
 mutate(()=>{
  ann.points[key]={x:q.x,y:q.y,state:preferredState,confidence:preferredConfidence};
 });
 choose((selected+1)%6)
}
async function load(i){
 busy=true;$("veil").hidden=false;image=null;gesture=null;draw();
 try{
  const data=await api("/api/item/"+i);
  const img=new Image();
  await new Promise((resolve,reject)=>{
   img.onload=resolve;img.onerror=()=>reject(Error("Не удалось загрузить PNG."));
   img.src="/api/png/"+i+"?t="+encodeURIComponent(TOKEN)
  });
  if(img.naturalWidth!==data.cols||img.naturalHeight!==data.rows)
   throw Error("Размер PNG не совпадает с DICOM.");
  current=data;index=i;ann=data.annotation;revision=data.revision;
  finished=data.finished;image=img;dirty=false;
  $("comment").value=ann.comment;
  $("brightness").value=100;$("contrast").value=100;
  busy=false;last=performance.now();$("veil").hidden=true;
  choose(Math.max(0,names.findIndex(k=>!ann.points[k])));resize();fit();
 }catch(e){error(e.message);$("veil").textContent="Ошибка загрузки. Перезагрузите страницу."}
}
async function go(delta,skip=false){
 if(!active())return;
 account();busy=true;
 const target=Math.max(0,Math.min(current.total-1,index+delta));
 const requested=skip?"skipped":mode();
 if(skip)ann.state="skipped";
 const ok=await queueSave(target,requested);
 if(!ok||failed){busy=false;return}
 if(target===index){
  busy=false;
  if(requested!=="draft")ann.state=requested;
  render();
  error(requested==="draft"?"Черновик сохранён. Задайте оставшиеся точки.":"Сохранено. Это край очереди.");
  return
 }
 await load(target);
}
$("canvas").oncontextmenu=e=>e.preventDefault();
$("canvas").onpointerdown=e=>{
 if(!active()||!image)return;
 const p=pointer(e);
 if(e.button===2||(e.button===0&&space))
  gesture={kind:"pan",p,ox,oy,id:e.pointerId};
 else if(e.button===0)gesture={kind:"point",p,id:e.pointerId};
 if(gesture){$("canvas").setPointerCapture(e.pointerId);e.preventDefault()}
};
$("canvas").onpointermove=e=>{
 const p=pointer(e),q=native(p);
 $("coords").textContent=`Исходный DICOM: x=${q.x.toFixed(1)}, y=${q.y.toFixed(1)}`;
 if(gesture?.kind==="pan"){
  ox=gesture.ox+p.x-gesture.p.x;oy=gesture.oy+p.y-gesture.p.y;draw()
 }
};
$("canvas").onpointerup=e=>{
 if(!gesture)return;
 const g=gesture;gesture=null;
 if(g.kind==="point"){
  const p=pointer(e);
  if(Math.hypot(p.x-g.p.x,p.y-g.p.y)<6)put(p)
 }
};
$("canvas").onpointercancel=()=>{gesture=null};
$("canvas").addEventListener("wheel",e=>{
 e.preventDefault();if(!image)return;
 const p=pointer(e),q=native(p);
 zoom=Math.max(.01,Math.min(30,zoom*Math.exp(-e.deltaY*.0015)));
 ox=p.x-q.x*zoom;oy=p.y-q.y*zoom;draw()
},{passive:false});
$("prev").onclick=()=>go(-1);
$("next").onclick=()=>go(1);
$("skip").onclick=()=>go(1,true);
$("delete").onclick=()=>mutate(()=>{delete ann.points[names[selected]]});
$("fit").onclick=fit;$("zoom3").onclick=()=>center(3);
$("brightness").oninput=draw;$("contrast").oninput=draw;
$("reset").onclick=()=>{$("brightness").value=100;$("contrast").value=100;draw()};
$("confidence").onchange=()=>setConfidence(Number($("confidence").value));
document.querySelectorAll("[data-state]").forEach(b=>
 b.onclick=()=>setState(b.dataset.state));
$("comment").oninput=()=>mutate(()=>{ann.comment=$("comment").value});
$("retry").onclick=async()=>{
 if(conflict||pending||!ann)return;
 failed=false;error();await autoQueue();render()
};
document.onkeydown=e=>{
 if(e.target.matches("input,textarea,select")||e.ctrlKey||e.altKey||e.metaKey)return;
 if(e.code==="Space"){space=true;e.preventDefault();return}
 if(e.repeat)return;
 if(/^Numpad[1-3]$/.test(e.code)||(e.shiftKey&&/^Digit[1-3]$/.test(e.code))){
  e.preventDefault();setConfidence(Number(e.code.slice(-1)));return
 }
 if(!e.shiftKey&&/^Digit[1-6]$/.test(e.code)){
  e.preventDefault();choose(Number(e.code.slice(-1))-1);return
 }
 if(e.key==="?"){e.preventDefault();setState("uncertain");return}
 const actions={
  KeyV:()=>setState("visible"),KeyX:()=>setState("not_visible"),
  KeyO:()=>setState("out_of_frame"),KeyN:()=>go(1),KeyP:()=>go(-1),
  KeyS:()=>go(1,true),KeyD:()=>$("delete").click()
 };
 if(actions[e.code]){e.preventDefault();actions[e.code]()}
};
document.onkeyup=e=>{if(e.code==="Space")space=false};
window.addEventListener("blur",()=>{
 account();space=false;gesture=null;
 if(ann&&!busy&&!failed)autoQueue()
});
window.addEventListener("focus",()=>{last=performance.now()});
document.addEventListener("visibilitychange",()=>{
 // Периодический учёт ограничивает возможную потерю времени до секунды.
 last=performance.now();
 if(document.hidden&&ann&&!busy&&!failed)autoQueue()
});
window.addEventListener("beforeunload",e=>{
 if(pending||dirty||failed){e.preventDefault();e.returnValue=""}
});
new ResizeObserver(resize).observe($("stage"));
setInterval(()=>{
 account();renderHeader();
 if(!pending&&!failed)saveStatus();
},1000);
setInterval(()=>{if(active())autoQueue()},10000);
async function boot(){
 try{
  const s=await api("/api/session");revision=s.revision;
  await load(s.cursor);
 }catch(e){error(e.message)}
}
boot();
</script>
</html>
"""


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "HipAnnotator/1"

    def log_message(self, fmt, *args):
        pass

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, value):
        self.send_bytes(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def fail(self, status, message):
        self.send_json(status, {"error": message})

    def authorized(self, png=False):
        from urllib.parse import parse_qs
        supplied = self.headers.get("X-Hip-Token", "")
        if png:
            supplied = parse_qs(urlsplit(self.path).query).get("t", [""])[0]
        return secrets.compare_digest(supplied, self.server.app.token)

    def valid_host(self):
        # Защита локального HTTP-сервера от DNS rebinding.
        host = self.headers.get("Host", "")
        return host in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def route_index(self, route, prefix):
        text = route[len(prefix):]
        if not re.fullmatch(r"\d+", text):
            raise ValueError("Некорректная позиция.")
        index = int(text)
        if not 0 <= index < len(self.server.app.order):
            raise ValueError("Позиция вне очереди.")
        return index

    def do_GET(self):
        if not self.valid_host():
            self.fail(403, "Недопустимый Host.")
            return
        app = self.server.app
        route = urlsplit(self.path).path
        try:
            if route == "/":
                page = HTML.replace("__TOKEN__", json.dumps(app.token))
                self.send_bytes(200, page.encode("utf-8"),
                                "text/html; charset=utf-8")
                return
            if not self.authorized(png=route.startswith("/api/png/")):
                self.fail(403, "Нет токена сессии. Перезагрузите страницу.")
                return
            if route == "/api/session":
                with app.lock:
                    session = app.data["session"]
                    self.send_json(200, {
                        "cursor": session["cursor"],
                        "revision": session["revision"],
                    })
            elif route.startswith("/api/item/"):
                index = self.route_index(route, "/api/item/")
                self.send_json(200, app.view(index))
            elif route.startswith("/api/png/"):
                index = self.route_index(route, "/api/png/")
                uid = app.order[index].split("#", 1)[0]
                self.send_bytes(200, app.pngs[uid], "image/png")
            else:
                self.fail(404, "Не найдено.")
        except ValueError as exc:
            self.fail(400, str(exc))
        except Exception:
            self.fail(500, "Ошибка чтения сессии.")

    def do_POST(self):
        if not self.valid_host() or not self.authorized():
            self.fail(403, "Нет доступа к сессии.")
            return
        if urlsplit(self.path).path != "/api/save":
            self.fail(404, "Не найдено.")
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Ожидается application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 200_000:
                raise ValueError("Неверный размер запроса.")
            self.connection.settimeout(30)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Запрос получен не полностью.")

            def reject(value):
                raise ValueError(f"Недопустимое число: {value}")

            raw = json.loads(body.decode("utf-8"), parse_constant=reject)
            self.send_json(200, self.server.app.save(raw))
        except Conflict as exc:
            self.fail(409, str(exc))
        except (ValueError, UnicodeError) as exc:
            self.fail(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"Ошибка записи: {type(exc).__name__}: {exc}", flush=True)
            self.fail(500, "Ошибка атомарной записи. Проверьте диск и права доступа.")


def main():
    parser = argparse.ArgumentParser(
        description="Разметка шести ключевых точек бедра на DXA."
    )
    parser.add_argument("--index", default="data/index/images.csv")
    parser.add_argument("--root", default=".")
    parser.add_argument("--annotator")
    parser.add_argument("--out", default="data/annotations")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--part", default="all", choices=("1", "2", "all"),
                        help="часть работы: 1 или 2 (делится по исследованиям, "
                             "6 общих для оценки согласия), all — всё")
    parser.add_argument("--report", metavar="JSON")
    args = parser.parse_args()

    if args.report:
        try:
            report(args.report)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            parser.exit(1, f"Ошибка отчёта: {exc}\n")
        return

    if not args.annotator or not re.fullmatch(r"[A-Za-z0-9_-]+", args.annotator):
        parser.error("--annotator: латинские буквы, цифры, _ и -.")
    if not 1 <= args.port <= 65535:
        parser.error("--port: 1–65535.")

    try:
        app = Application(args)
        server = Server(("127.0.0.1", args.port), app)
    except (OSError, ValueError, ImportError, csv.Error) as exc:
        parser.exit(1, f"Ошибка запуска: {exc}\n")

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n{url}\nОстановка: Ctrl+C.", flush=True)
    print("Используйте одну вкладку на файл разметки.", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nСервер остановлен.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
