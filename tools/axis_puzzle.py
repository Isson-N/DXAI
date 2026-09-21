# tools/axis_puzzle.py
#
# pip install pydicom pillow numpy
#
# python tools/axis_puzzle.py \
#   --index data/index/images.csv --root . --annotator ivan
#
# Инструмент разбора оси позвоночника на DXA-снимках.
# Очередь берётся из data/index/axis_puzzle_queue.csv.
# Служебные метаданные снимка намеренно не передаются в браузер и не
# сохраняются в файле разметки.

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


SCHEMA = "dxa-axis-puzzle/1"

VERTEBRAE = ("Th12", "L1", "L2", "L3", "L4", "L5")
LEVELS = ("top", "bottom")
STATES = ("visible", "not_visible", "out_of_frame", "uncertain")
VERDICTS = ("axis_deviated", "normal", "cannot_decide")

FEATURES = (
    "whole_column_tilted",
    "lumbar_only_tilted",
    "pelvis_tilted",
    "column_shifted_sideways",
    "vertebrae_rotated",
    "column_cut_off",
    "uncertain",
)

FEATURE_LABELS = {
    "whole_column_tilted": "наклонён весь столб",
    "lumbar_only_tilted": "наклонён только поясничный отдел",
    "pelvis_tilted": "таз перекошен",
    "column_shifted_sideways": "столб смещён вбок",
    "vertebrae_rotated": "позвонки повёрнуты вокруг своей оси",
    "column_cut_off": "кадр обрезает столб",
    "uncertain": "сомневаюсь",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def atomic_write(filename, value):
    """Атомарно записывает JSON-файл с принудительной синхронизацией."""
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
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, filename)
        temporary = None

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
    relative_path = Path(relative)
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / relative_path).resolve()

    if relative_path.is_absolute() or not resolved.is_relative_to(resolved_root):
        raise ValueError("Путь изображения находится вне --root.")

    return resolved


def read_dicom(filename, rows, cols):
    """Читает один монохромный DICOM и возвращает PNG."""
    import numpy as np
    import pydicom
    from PIL import Image

    try:
        dataset = pydicom.dcmread(filename)
    except pydicom.errors.InvalidDicomError:
        dataset = pydicom.dcmread(filename, force=True)

    pixels = np.asarray(dataset.pixel_array, dtype=np.float64)

    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]

    if pixels.ndim != 2:
        raise ValueError("Ожидался один монохромный кадр.")

    if pixels.shape != (rows, cols):
        raise ValueError("Размер изображения не совпадает с индексом.")

    photometric = str(
        getattr(dataset, "PhotometricInterpretation", "")
    )
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Поддерживаются только монохромные изображения.")

    slope = float(getattr(dataset, "RescaleSlope", 1))
    intercept = float(getattr(dataset, "RescaleIntercept", 0))

    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Некорректное преобразование интенсивности.")

    pixels = pixels * slope + intercept
    mask = np.isfinite(pixels)

    if not mask.any():
        raise ValueError("В изображении нет конечных значений.")

    low = float(pixels[mask].min())
    high = float(pixels[mask].max())

    pixels = np.nan_to_num(
        pixels,
        nan=low,
        posinf=high,
        neginf=low,
    )

    if high > low:
        pixels = (pixels - low) / (high - low)
    else:
        pixels = np.zeros_like(pixels)

    if photometric == "MONOCHROME1":
        pixels = 1 - pixels

    pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def read_index(filename):
    """
    Читает только нужные поля images.csv.
    Никакие дополнительные поля не интерпретируются.
    """
    required = {"sop_uid", "path", "rows", "cols"}
    result = {}
    order = []

    with open(filename, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        available = set(reader.fieldnames or ())
        missing = required - available
        if missing:
            raise ValueError(
                "В images.csv отсутствуют колонки: "
                + ", ".join(sorted(missing))
            )

        for line_number, row in enumerate(reader, 2):
            uid = str(row["sop_uid"] or "").strip()
            path = str(row["path"] or "").strip()

            if not uid or "#" in uid:
                raise ValueError(
                    f"Строка {line_number}: некорректный sop_uid."
                )
            if not path:
                raise ValueError(
                    f"Строка {line_number}: пустой путь изображения."
                )
            if uid in result:
                raise ValueError(
                    f"Строка {line_number}: повтор sop_uid."
                )

            dimensions = []
            for name in ("rows", "cols"):
                try:
                    number = float(row[name])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Строка {line_number}: некорректный {name}."
                    )

                if (
                    not math.isfinite(number)
                    or number < 1
                    or not number.is_integer()
                ):
                    raise ValueError(
                        f"Строка {line_number}: некорректный {name}."
                    )

                dimensions.append(int(number))

            record = {
                "uid": uid,
                "path": path,
                "rows": dimensions[0],
                "cols": dimensions[1],
            }
            result[uid] = record
            order.append(uid)

    return result, order


def read_queue(filename, images):
    """Читает ровно одну колонку uid и сохраняет порядок строк."""
    queue = []

    with open(filename, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["uid"]:
            raise ValueError(
                "Очередь должна содержать единственную колонку uid."
            )

        for line_number, row in enumerate(reader, 2):
            uid = str(row.get("uid") or "").strip()

            if not uid or "#" in uid:
                raise ValueError(
                    f"Строка {line_number}: некорректный uid очереди."
                )
            if uid not in images:
                raise ValueError(
                    f"Строка {line_number}: снимок отсутствует в images.csv."
                )
            if uid in queue:
                raise ValueError(
                    f"Строка {line_number}: повтор uid в очереди."
                )

            queue.append(uid)

    if len(queue) != 42:
        raise ValueError(
            f"В очереди должно быть 42 снимка, найдено {len(queue)}."
        )

    return queue


def blank_points():
    return {
        vertebra: {
            level: {
                "state": "not_visible",
                "confidence": 1,
            }
            for level in LEVELS
        }
        for vertebra in VERTEBRAE
    }


def blank_annotation():
    return {
        "points": blank_points(),
        "comment": "",
        "verdict": "cannot_decide",
        "features": [],
        "status": "draft",
        "seconds": 0.0,
        "updated": now_iso(),
    }


def validate_points(raw):
    if not isinstance(raw, dict):
        raise ValueError("points должны быть объектом.")

    if set(raw) != set(VERTEBRAE):
        raise ValueError("Некорректный набор позвонков.")

    clean = {}

    for vertebra in VERTEBRAE:
        value = raw.get(vertebra)
        if not isinstance(value, dict) or set(value) != set(LEVELS):
            raise ValueError("У каждого позвонка нужны top и bottom.")

        clean[vertebra] = {}

        for level in LEVELS:
            point = value[level]
            if not isinstance(point, dict):
                raise ValueError("Точка должна быть объектом.")

            allowed = {"x", "y", "state", "confidence"}
            if set(point) - allowed:
                raise ValueError("Неизвестное поле точки.")

            state = point.get("state")
            confidence = point.get("confidence")

            if state not in STATES:
                raise ValueError("Неизвестное состояние точки.")

            if type(confidence) is not int or confidence not in (1, 2, 3):
                raise ValueError("Уверенность должна быть 1–3.")

            # Прежняя версия писала «x: null» у непоставленных точек, поэтому
            # судим по значению, а не по наличию ключа: иначе файл не читается.
            has_x = point.get("x") is not None
            has_y = point.get("y") is not None

            if has_x != has_y:
                raise ValueError(
                    "Координаты точки должны задаваться парой."
                )

            if state in ("not_visible", "out_of_frame") and (
                has_x or has_y
            ):
                raise ValueError(
                    "Для not_visible и out_of_frame координаты запрещены."
                )

            if state == "visible" and not (has_x and has_y):
                raise ValueError(
                    "Для visible нужны координаты."
                )

            result = {
                "state": state,
                "confidence": confidence,
            }

            if has_x and has_y:
                x = point["x"]
                y = point["y"]

                if (
                    not finite(x)
                    or not finite(y)
                    or not 0 <= x <= 1
                    or not 0 <= y <= 1
                ):
                    raise ValueError(
                        "Нормированные координаты должны быть в диапазоне 0..1."
                    )

                result["x"] = float(x)
                result["y"] = float(y)

            clean[vertebra][level] = result

    return clean


def validate_annotation(raw, status=None):
    if not isinstance(raw, dict):
        raise ValueError("Разметка должна быть объектом.")

    allowed = {
        "points",
        "comment",
        "verdict",
        "features",
        "status",
        "seconds",
        "updated",
    }
    if set(raw) - allowed:
        raise ValueError("Неизвестные поля разметки.")

    clean = blank_annotation()
    clean["points"] = validate_points(raw.get("points"))

    comment = raw.get("comment")
    if not isinstance(comment, str) or len(comment) > 20000:
        raise ValueError(
            "Комментарий должен быть строкой длиной до 20000 символов."
        )
    clean["comment"] = comment

    verdict = raw.get("verdict")
    # Прежняя версия называла этот вердикт «deviated» — принимаем оба написания.
    if verdict == "deviated":
        verdict = "axis_deviated"
    # Черновик без вердикта — нормальное состояние: разметчик ещё не решил.
    # Обязательность вердикта проверяется ниже, при переводе снимка в «done».
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError("Некорректный вердикт.")
    clean["verdict"] = verdict

    features = raw.get("features")
    if not isinstance(features, list):
        raise ValueError("features должны быть списком.")

    if len(set(features)) != len(features):
        raise ValueError("Признаки не должны повторяться.")

    if any(feature not in FEATURES for feature in features):
        raise ValueError("Неизвестный признак.")
    clean["features"] = list(features)

    seconds = raw.get("seconds", 0.0)
    if not finite(seconds) or seconds < 0:
        raise ValueError("Некорректное время.")
    clean["seconds"] = float(seconds)

    updated = raw.get("updated", now_iso())
    if not isinstance(updated, str):
        raise ValueError("Некорректная дата изменения.")
    clean["updated"] = updated

    final_status = status if status is not None else raw.get("status")
    if final_status not in ("draft", "done", "skipped"):
        raise ValueError("Некорректный статус.")

    if final_status == "done":
        for vertebra in VERTEBRAE:
            for level in LEVELS:
                if not clean["points"][vertebra][level].get("state"):
                    raise ValueError("Для done нужны все состояния точек.")
        # Вердикт — главный результат задачи, без него снимок не закрывается.
        if clean["verdict"] is None:
            raise ValueError("Снимок нельзя завершить без вердикта.")

    clean["status"] = final_status
    return clean


class Conflict(Exception):
    pass


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)

        self.filename = (
            Path(args.out) / f"axis_puzzle_{args.annotator}.json"
        )

        self.images, image_order = read_index(args.index)

        queue_file = Path("data/index/axis_puzzle_queue.csv")
        self.order = read_queue(queue_file, self.images)

        self.pngs = {}
        root = Path(args.root).resolve()

        for uid in self.order:
            record = self.images[uid]
            try:
                self.pngs[uid] = read_dicom(
                    safe_path(root, record["path"]),
                    record["rows"],
                    record["cols"],
                )
            except Exception as exc:
                raise ValueError(
                    f"Ошибка чтения изображения: {type(exc).__name__}: {exc}"
                ) from exc

        self.data = {
            "schema": SCHEMA,
            "session_revision": 0,
            "items": {},
            "session": {
                "order": list(self.order),
                "cursor": 0,
            },
        }

        if self.filename.exists():
            loaded = read_json(self.filename)
            self.data = self.load_existing(loaded)
        else:
            atomic_write(self.filename, self.data)

        print(f"Снимков в очереди: {len(self.order)}.", flush=True)
        print(f"Сохранение: {self.filename}", flush=True)

    def load_existing(self, loaded):
        if not isinstance(loaded, dict):
            raise ValueError("Файл разметки должен быть объектом.")

        if loaded.get("schema") != SCHEMA:
            raise ValueError("Неподходящая схема файла разметки.")

        items = loaded.get("items")
        if not isinstance(items, dict):
            raise ValueError("В файле нет items.")

        revision = loaded.get("session_revision", 0)
        if type(revision) is not int or revision < 0:
            # Прежняя версия инструмента писала ревизию строкой-токеном.
            # Ревизия защищает только от второй вкладки внутри сеанса, поэтому
            # непонятное значение — повод начать счёт заново, а не терять файл.
            revision = 0

        session = loaded.get("session", {})
        if not isinstance(session, dict):
            raise ValueError("Некорректная сессия.")

        order = session.get("order", self.order)
        cursor = session.get("cursor", 0)

        if order != self.order:
            raise ValueError("Изменилась очередь снимков.")

        if (
            type(cursor) is not int
            or not 0 <= cursor < len(self.order)
        ):
            raise ValueError("Некорректная позиция в очереди.")

        clean_items = {}

        for uid, value in items.items():
            if uid not in self.order:
                raise ValueError("В файле есть снимок вне очереди.")

            # UID используется только как внутренний ключ файла и никогда
            # не возвращается браузеру.
            clean_items[uid] = validate_annotation(value)

        return {
            "schema": SCHEMA,
            "session_revision": revision,
            "items": clean_items,
            "session": {
                "order": list(self.order),
                "cursor": cursor,
            },
        }

    def annotation(self, uid):
        raw = self.data["items"].get(uid)
        if raw is None:
            return blank_annotation()
        return copy.deepcopy(raw)

    def view(self, index):
        with self.lock:
            uid = self.order[index]
            image = self.images[uid]
            return {
                "index": index,
                "total": len(self.order),
                "rows": image["rows"],
                "cols": image["cols"],
                "annotation": self.annotation(uid),
                "finished": sum(
                    value.get("status") in ("done", "skipped")
                    for value in self.data["items"].values()
                ),
                "revision": self.data["session_revision"],
            }

    def save(self, raw):
        with self.lock:
            if not isinstance(raw, dict):
                raise ValueError("Ожидался JSON-объект.")

            revision = raw.get("revision")
            if revision != self.data["session_revision"]:
                raise Conflict(
                    "Сессия изменена в другой вкладке. "
                    "Перезагрузите страницу."
                )

            index = raw.get("index")
            cursor = raw.get("cursor", index)

            for value in (index, cursor):
                if (
                    type(value) is not int
                    or not 0 <= value < len(self.order)
                ):
                    raise ValueError("Некорректная позиция очереди.")

            mode = raw.get("mode")
            if mode not in ("draft", "done", "skipped"):
                raise ValueError("Некорректное действие сохранения.")

            annotation = copy.deepcopy(raw.get("annotation"))
            if not isinstance(annotation, dict):
                raise ValueError("Нет разметки.")

            annotation["status"] = mode
            clean = validate_annotation(annotation, mode)

            uid = self.order[index]
            old = self.data["items"].get(uid)
            if old:
                clean["seconds"] = max(
                    clean["seconds"],
                    float(old.get("seconds", 0)),
                )

            clean["updated"] = now_iso()

            updated = copy.deepcopy(self.data)
            updated["items"][uid] = clean
            updated["session"]["cursor"] = cursor
            updated["session_revision"] += 1

            atomic_write(self.filename, updated)
            self.data = updated

            return {
                "revision": updated["session_revision"],
                "finished": sum(
                    value.get("status") in ("done", "skipped")
                    for value in updated["items"].values()
                ),
            }


HTML = r"""<!doctype html>
<html lang="ru">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DXA · ось позвоночника</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#101820;color:#e8eff8}
header{padding:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button,input,textarea,select{font:inherit}
button,select{padding:7px;background:#253c51;color:white;border:1px solid #607990;border-radius:4px;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
input,textarea{background:#192b3b;color:white;border:1px solid #607990}
textarea{width:100%;min-height:80px}
main{display:grid;grid-template-columns:minmax(300px,1fr) 380px;height:calc(100vh - 65px)}
section{display:flex;flex-direction:column;min-height:0;min-width:0}
#tools{padding:8px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
#stage{position:relative;flex:1;min-height:200px;background:#030609;overflow:hidden}
canvas{position:absolute;width:100%;height:100%;touch-action:none;cursor:crosshair}
aside{padding:12px;overflow:auto;background:#1a2939}
#progress{font-weight:bold}
#position{font-size:16px;font-weight:bold}
#points button{display:block;width:100%;text-align:left;margin:5px 0}
#points button.active{outline:2px solid #ffe274;background:#3e5160}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:9px 0}
.small{font-size:12px;line-height:1.5;color:#bfd1df}
#error{color:#ffd0bf;white-space:pre-wrap}
#coords{padding:6px;color:#bacddd}
#save{margin-left:auto}
input[type=range]{width:100px}
#veil{position:absolute;inset:0;background:#101820c0;display:grid;place-items:center}
#veil[hidden]{display:none}
fieldset{border:1px solid #506d83;margin:10px 0;padding:8px}
fieldset label{display:block;margin:6px 0}
@media(max-width:800px){
 main{display:flex;flex-direction:column;height:auto}
 section{height:65vh}
 aside{overflow:visible}
}
</style>

<header>
<b>DXA · ось позвоночника</b>
<span id="progress"></span>
<span id="timer"></span>
<span id="save"></span>
<button id="retry" hidden>Повторить сохранение</button>
</header>

<main>
<section>
<div id="tools">
<button id="fit">Вписать</button>
<button id="zoom3">×3</button>
<label>Яркость <input id="brightness" type="range" min="30" max="250" value="100"></label>
<label>Контраст <input id="contrast" type="range" min="30" max="350" value="100"></label>
<button id="reset">Сброс</button>
</div>
<div id="stage">
<canvas id="canvas"></canvas>
<div id="veil">Загрузка…</div>
</div>
<div id="coords">Координаты исходного изображения</div>
</section>

<aside>
<div id="position"></div>
<div id="error"></div>

<div class="row">
<button id="prev">P · Назад</button>
<button id="next">N · Далее</button>
<button id="skip">S · Пропустить</button>
</div>

<div id="points"></div>

<div class="row">
<button data-state="visible">V · видна</button>
<button data-state="not_visible">X · не видна</button>
<button data-state="out_of_frame">O · вне кадра</button>
<button data-state="uncertain">? · не уверена</button>
</div>

<label>Уверенность:
<select id="confidence">
<option value="1" selected>1 — сомневаюсь</option>
<option value="2">2 — так себе</option>
<option value="3">3 — уверен</option>
</select>
</label>

<div class="row">
<button id="delete">D · Удалить точку</button>
</div>

<fieldset>
<legend>Вердикт</legend>
<label><input type="radio" name="verdict" value="axis_deviated"> ось отклонена</label>
<label><input type="radio" name="verdict" value="normal"> норма</label>
<label><input type="radio" name="verdict" value="cannot_decide"> не могу решить</label>
</fieldset>

<fieldset>
<legend>Признаки</legend>
<div id="features"></div>
</fieldset>

<label>Комментарий
<textarea id="comment" maxlength="20000"
 placeholder="Что именно выглядит наклонённым и относительно чего?"></textarea>
</label>

<p class="small">
1–6 — выбрать позвонок. T/B — верхняя или нижняя точка.
Shift+1–3 или NumPad1–3 — уверенность.
ЛКМ — поставить точку; следующая точка выбирается автоматически.
Уже поставленную точку можно перетащить мышью.
<br>
Колесо — зум; ПКМ или Space+мышь — панорама.
<br>
V — видна, X — не видна, O — вне кадра, ? — не уверена.
D — удалить точку, P — предыдущий снимок, N — следующий,
S — пропустить.
<br>
После последней точки выбор не зацикливается.
Координаты при X и O запрещены.
Тонкие вертикаль и горизонталь показывают центр кадра.
</p>
</aside>
</main>

<script>
"use strict";

const TOKEN=__TOKEN__;
const vertebrae=["Th12","L1","L2","L3","L4","L5"];
const levels=["top","bottom"];
const pointNames=[];
for(const v of vertebrae)for(const l of levels)pointNames.push(v+"_"+l);

let current=null, ann=null, image=null;
let index=0, selected=0, revision=0, finished=0;
let W=1,H=1,zoom=1,ox=0,oy=0;
let gesture=null,space=false,busy=true;
let pending=0,failed=false,dirty=false,conflict=false;
let tail=Promise.resolve(),last=performance.now();

const $=id=>document.getElementById(id);
const clone=value=>JSON.parse(JSON.stringify(value));

function error(text=""){ $("error").textContent=text; }

async function api(url,options={}){
 const controller=new AbortController();
 const timer=setTimeout(()=>controller.abort(),15000);
 try{
  const response=await fetch(url,{
   cache:"no-store",
   ...options,
   signal:controller.signal,
   headers:{
    "X-Axis-Token":TOKEN,
    "Content-Type":"application/json",
    ...(options.headers||{})
   }
  });
  let data;
  try{data=await response.json()}
  catch(e){throw Error("Некорректный ответ сервера")}
  if(!response.ok){
   const e=Error(data.error||`HTTP ${response.status}`);
   e.conflict=response.status===409;
   throw e;
  }
  return data;
 }catch(e){
  if(e.name==="AbortError")throw Error("Запрос превысил таймаут 15 секунд.");
  throw e;
 }finally{
  clearTimeout(timer);
 }
}

function account(){
 const now=performance.now();
 if(ann&&!busy&&!document.hidden&&document.hasFocus()){
  const dt=Math.max(0,(now-last)/1000);
  ann.seconds+=dt;
  if(dt>0)dirty=true;
 }
 last=now;
}

function statusMode(){
 if(ann.status==="skipped")return "skipped";
 return pointNames.every(name=>{
  const [v,l]=name.split("_");
  return ann.points[v][l].state;
 }) ? "done" : "draft";
}

function saveStatus(){
 $("save").textContent=failed?"НЕ СОХРАНЕНО":
   pending?"Сохранение…":dirty?"Изменения в памяти":"Сохранено ✓";
 $("retry").hidden=!failed||conflict;
}

function queueSave(cursor=index,requestedMode=null){
 if(!ann)return Promise.resolve(false);

 account();
 const snapshot=clone(ann);
 const capturedIndex=index;
 const capturedMode=requestedMode||statusMode();

 pending++;
 saveStatus();

 tail=tail.then(async()=>{
  if(failed)return false;
  try{
   const result=await api("/api/save",{
    method:"POST",
    body:JSON.stringify({
     index:capturedIndex,
     cursor,
     mode:capturedMode,
     annotation:snapshot,
     revision
    })
   });
   revision=result.revision;
   finished=result.finished;
   return true;
  }catch(e){
   failed=true;
   conflict=!!e.conflict;
   error(e.message+" Изменения этой вкладки сохранены в памяти.");
   return false;
  }
 }).finally(()=>{
  pending--;
  if(!pending&&!failed)dirty=false;
  saveStatus();
  renderHeader();
 });

 return tail;
}

function autoQueue(){
 const mode=(ann.status==="done"||ann.status==="skipped")
  ?ann.status:"draft";
 return queueSave(index,mode);
}

function mutate(fn){
 if(!ann||busy||failed)return;
 account();
 fn();
 if(ann.status!=="done"&&ann.status!=="skipped")ann.status="draft";
 dirty=true;
 render();
 autoQueue();
}

function selectedParts(){
 return pointNames[selected].split("_");
}

function choose(number){
 selected=Math.max(0,Math.min(pointNames.length-1,number));
 render();
}

function pointValue(){
 const [v,l]=selectedParts();
 return ann.points[v][l];
}

function setState(state){
 if(!ann||busy||failed)return;

 const point=pointValue();
 if(state==="visible"&&!Number.isFinite(point.x)){
  error("Для visible поставьте координату щелчком по изображению.");
  render();
  return;
 }

 mutate(()=>{
  const result={
   state,
   confidence:point.confidence||Number($("confidence").value)||1
  };
  if(
   (state==="visible"||state==="uncertain") &&
   Number.isFinite(point.x)&&Number.isFinite(point.y)
  ){
   result.x=point.x;
   result.y=point.y;
  }

  const [v,l]=selectedParts();
  ann.points[v][l]=result;
 });
}

function setConfidence(value){
 if(!ann||busy||failed)return;

 const [v,l]=selectedParts();
 if(ann.points[v][l]){
  mutate(()=>ann.points[v][l].confidence=value);
 }
}

function renderHeader(){
 $("progress").textContent=current?
  `Завершено ${finished}/${current.total}`:"";
 $("position").textContent=current?
  `Показ ${index+1} из ${current.total}`:"";
 $("timer").textContent=ann?
  `${Math.floor(ann.seconds)} с`:"";
}

function renderPoints(){
 const container=$("points");
 container.replaceChildren();

 pointNames.forEach((name,i)=>{
  const [v,l]=name.split("_");
  const p=ann.points[v][l];
  const button=document.createElement("button");
  button.className=i===selected?"active":"";
  button.textContent=
   `${i<12?Math.floor(i/2)+1:""} · ${v} ${l} — `+
   `${p.state} · ${p.confidence}`;
  button.onclick=()=>choose(i);
  container.append(button);
 });
}

function renderFeatures(){
 const container=$("features");
 container.replaceChildren();

 for(const feature of [
  "whole_column_tilted",
  "lumbar_only_tilted",
  "pelvis_tilted",
  "column_shifted_sideways",
  "vertebrae_rotated",
  "column_cut_off",
  "uncertain"
 ]){
  const label=document.createElement("label");
  const input=document.createElement("input");
  input.type="checkbox";
  input.value=feature;
  input.checked=ann.features.includes(feature);
  input.onchange=()=>{
   mutate(()=>{
    const values=new Set(ann.features);
    if(input.checked)values.add(feature);
    else values.delete(feature);
    ann.features=[...values];
   });
  };
  label.append(input," ",{
   whole_column_tilted:"наклонён весь столб",
   lumbar_only_tilted:"наклонён только поясничный отдел",
   pelvis_tilted:"таз перекошен",
   column_shifted_sideways:"столб смещён вбок",
   vertebrae_rotated:"позвонки повёрнуты вокруг своей оси",
   column_cut_off:"кадр обрезает столб",
   uncertain:"сомневаюсь"
  }[feature]);
  container.append(label);
 }
}

function render(){
 renderHeader();
 saveStatus();
 renderPoints();
 renderFeatures();

 $("confidence").value=pointValue().confidence||1;
 document.querySelectorAll("[data-state]").forEach(button=>{
  button.style.outline=
   button.dataset.state===pointValue().state
    ?"2px solid #ffe274":"none";
 });

 document.querySelectorAll("[name=verdict]").forEach(input=>{
  input.checked=input.value===ann.verdict;
 });

 $("comment").value=ann.comment;
 draw();
}

function draw(){
 const context=$("canvas").getContext("2d");
 context.clearRect(0,0,W,H);
 if(!image)return;

 context.save();
 context.filter=
  `brightness(${$("brightness").value}%) `+
  `contrast(${$("contrast").value}%)`;
 context.imageSmoothingEnabled=false;
 context.drawImage(
  image,ox,oy,current.cols*zoom,current.rows*zoom
 );
 context.restore();

 // Центральные вертикаль и горизонталь кадра.
 const centerX=ox+current.cols*zoom/2;
 const centerY=oy+current.rows*zoom/2;

 context.strokeStyle="#91a9b8";
 context.lineWidth=1;
 context.setLineDash([5,5]);
 context.beginPath();
 context.moveTo(centerX,oy);
 context.lineTo(centerX,oy+current.rows*zoom);
 context.moveTo(ox,centerY);
 context.lineTo(ox+current.cols*zoom,centerY);
 context.stroke();
 context.setLineDash([]);

 if(!ann)return;

 context.font="bold 13px system-ui";

 pointNames.forEach((name,i)=>{
  const [v,l]=name.split("_");
  const p=ann.points[v][l];
  if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return;

  const x=ox+p.x*current.cols*zoom;
  const y=oy+p.y*current.rows*zoom;

  context.strokeStyle=i===selected?
   "#fff077":p.state==="uncertain"?"#ff8ea4":"#71ffc4";
  context.fillStyle=context.strokeStyle;
  context.lineWidth=2;

  context.beginPath();
  context.arc(x,y,5,0,2*Math.PI);
  context.stroke();

  context.beginPath();
  context.moveTo(x-9,y);context.lineTo(x+9,y);
  context.moveTo(x,y-9);context.lineTo(x,y+9);
  context.stroke();

  context.fillText(`${v} ${l}`,x+10,y-7);
 });
}

function resize(){
 const rect=$("stage").getBoundingClientRect();
 const d=devicePixelRatio||1;
 const canvas=$("canvas");

 W=rect.width;H=rect.height;
 canvas.width=Math.round(W*d);
 canvas.height=Math.round(H*d);
 canvas.getContext("2d").setTransform(d,0,0,d,0,0);
 draw();
}

function center(scale){
 if(!current)return;
 zoom=scale;
 ox=(W-current.cols*zoom)/2;
 oy=(H-current.rows*zoom)/2;
 draw();
}

function fit(){
 if(!current)return;
 center(Math.max(
  .01,
  Math.min(
   (W-20)/current.cols,
   (H-20)/current.rows
  )
 ));
}

function pointer(event){
 const rect=$("canvas").getBoundingClientRect();
 return {
  x:event.clientX-rect.left,
  y:event.clientY-rect.top
 };
}

function nativePoint(point){
 return {
  x:(point.x-ox)/(current.cols*zoom),
  y:(point.y-oy)/(current.rows*zoom)
 };
}

function put(point){
 if(!ann||busy||failed||!image)return;

 const q=nativePoint(point);
 if(q.x<0||q.x>1||q.y<0||q.y>1)return;

 const [v,l]=selectedParts();
 const currentPoint=ann.points[v][l];
 const state=currentPoint.state;
 const confidence=Number($("confidence").value)||1;

 if(state==="not_visible"||state==="out_of_frame"){
  error("Для постановки координаты выберите V или ?.");
  return;
 }

 error();

 mutate(()=>{
  ann.points[v][l]={
   x:q.x,
   y:q.y,
   state:state==="uncertain"?"uncertain":"visible",
   confidence
  };
 });

 if(selected<pointNames.length-1)choose(selected+1);
}

function pointAt(point){
 if(!ann)return -1;

 let best=-1;
 let distance=10;

 pointNames.forEach((name,i)=>{
  const [v,l]=name.split("_");
  const p=ann.points[v][l];
  if(!Number.isFinite(p.x)||!Number.isFinite(p.y))return;

  const x=ox+p.x*current.cols*zoom;
  const y=oy+p.y*current.rows*zoom;
  const d=Math.hypot(x-point.x,y-point.y);

  if(d<distance){
   distance=d;
   best=i;
  }
 });

 return best;
}

function dragTo(point){
 if(!ann||busy||failed)return;

 const q=nativePoint(point);
 if(q.x<0||q.x>1||q.y<0||q.y>1)return;

 const [v,l]=selectedParts();
 const p=ann.points[v][l];

 if(!Number.isFinite(p.x))return;

 p.x=q.x;
 p.y=q.y;
 dirty=true;
 draw();
}

async function load(target){
 busy=true;
 $("veil").hidden=false;
 image=null;
 gesture=null;
 draw();

 try{
  const data=await api("/api/item/"+target);
  const picture=new Image();

  await new Promise((resolve,reject)=>{
   picture.onload=resolve;
   picture.onerror=()=>reject(Error("Не удалось загрузить изображение."));
   picture.src="/api/png/"+target+"?t="+encodeURIComponent(TOKEN);
  });

  if(
   picture.naturalWidth!==data.cols||
   picture.naturalHeight!==data.rows
  ){
   throw Error("Размер изображения не совпадает с индексом.");
  }

  current=data;
  index=target;
  ann=data.annotation;
  revision=data.revision;
  finished=data.finished;
  image=picture;
  failed=false;
  dirty=false;
  busy=false;
  last=performance.now();
  $("veil").hidden=true;

  const firstUnset=pointNames.findIndex(name=>{
   const [v,l]=name.split("_");
   return !Number.isFinite(ann.points[v][l].x);
  });

  selected=firstUnset<0?pointNames.length-1:firstUnset;
  resize();
  fit();
  render();
 }catch(e){
  error(e.message);
  $("veil").textContent="Ошибка загрузки. Перезагрузите страницу.";
 }
}

async function go(delta,skip=false){
 if(!ann||busy||failed)return;

 account();
 busy=true;

 const target=Math.max(
  0,
  Math.min(current.total-1,index+delta)
 );
 const requested=skip?"skipped":statusMode();

 if(skip)ann.status="skipped";

 const ok=await queueSave(target,requested);

 if(!ok||failed){
  busy=false;
  return;
 }

 if(target===index){
  busy=false;
  ann.status=requested;
  render();
  error(
   requested==="draft"
    ?"Черновик сохранён."
    :"Сохранено. Это край очереди."
  );
  return;
 }

 await load(target);
}

$("canvas").oncontextmenu=event=>event.preventDefault();

$("canvas").onpointerdown=event=>{
 if(!ann||busy||failed||!image)return;

 const point=pointer(event);

 if(event.button===2||(event.button===0&&space)){
  gesture={
   kind:"pan",
   point,
   ox,
   oy,
   id:event.pointerId
  };
 }else if(event.button===0){
  const hit=pointAt(point);
  if(hit>=0){
   choose(hit);
   gesture={kind:"drag",point,id:event.pointerId};
  }else{
   gesture={kind:"point",point,id:event.pointerId};
  }
 }

 if(gesture){
  $("canvas").setPointerCapture(event.pointerId);
  event.preventDefault();
 }
};

$("canvas").onpointermove=event=>{
 const point=pointer(event);
 const q=nativePoint(point);

 $("coords").textContent=
  `Нормированные координаты: x=${q.x.toFixed(3)}, y=${q.y.toFixed(3)}`;

 if(gesture?.kind==="pan"){
  ox=gesture.ox+point.x-gesture.point.x;
  oy=gesture.oy+point.y-gesture.point.y;
  draw();
 }else if(gesture?.kind==="drag"){
  dragTo(point);
 }
};

$("canvas").onpointerup=event=>{
 if(!gesture)return;

 const activeGesture=gesture;
 gesture=null;

 if(activeGesture.kind==="point"){
  const point=pointer(event);
  if(
   Math.hypot(
    point.x-activeGesture.point.x,
    point.y-activeGesture.point.y
   )<6
  ){
   put(point);
  }
 }else if(activeGesture.kind==="drag"){
  dragTo(pointer(event));
  mutate(()=>{});
 }
};

$("canvas").onpointercancel=()=>{gesture=null};

$("canvas").addEventListener("wheel",event=>{
 event.preventDefault();
 if(!image)return;

 const point=pointer(event);
 const before=nativePoint(point);

 zoom=Math.max(
  .01,
  Math.min(30,zoom*Math.exp(-event.deltaY*.0015))
 );

 ox=point.x-before.x*current.cols*zoom;
 oy=point.y-before.y*current.rows*zoom;
 draw();
},{passive:false});

$("prev").onclick=()=>go(-1);
$("next").onclick=()=>go(1);
$("skip").onclick=()=>go(1,true);

$("delete").onclick=()=>{
 mutate(()=>{
  const [v,l]=selectedParts();
  ann.points[v][l]={
   state:"not_visible",
   confidence:Number($("confidence").value)||1
  };
 });
};

$("fit").onclick=fit;
$("zoom3").onclick=()=>center(3);

$("brightness").oninput=draw;
$("contrast").oninput=draw;

$("reset").onclick=()=>{
 $("brightness").value=100;
 $("contrast").value=100;
 draw();
};

$("confidence").onchange=()=>{
 setConfidence(Number($("confidence").value));
};

document.querySelectorAll("[data-state]").forEach(button=>{
 button.onclick=()=>setState(button.dataset.state);
});

document.querySelectorAll("[name=verdict]").forEach(input=>{
 input.onchange=()=>{
  mutate(()=>{ann.verdict=input.value});
 };
});

$("comment").oninput=()=>{
 mutate(()=>{ann.comment=$("comment").value});
};

$("retry").onclick=async()=>{
 if(conflict||pending||!ann)return;
 failed=false;
 error();
 await autoQueue();
 render();
};

document.onkeydown=event=>{
 if(
  event.target.matches("input,textarea,select")||
  event.ctrlKey||event.altKey||event.metaKey
 )return;

 if(event.code==="Space"){
  space=true;
  event.preventDefault();
  return;
 }

 if(event.repeat)return;

 if(
  /^Numpad[1-3]$/.test(event.code)||
  (event.shiftKey&&/^Digit[1-3]$/.test(event.code))
 ){
  event.preventDefault();
  setConfidence(Number(event.code.slice(-1)));
  return;
 }

 if(!event.shiftKey&&/^Digit[1-6]$/.test(event.code)){
  event.preventDefault();
  choose((Number(event.code.slice(-1))-1)*2);
  return;
 }

 if(event.code==="KeyT"){
  event.preventDefault();
  choose(Math.floor(selected/2)*2);
  return;
 }

 if(event.code==="KeyB"){
  event.preventDefault();
  choose(Math.floor(selected/2)*2+1);
  return;
 }

 if(event.key==="?"){
  event.preventDefault();
  setState("uncertain");
  return;
 }

 const actions={
  KeyV:()=>setState("visible"),
  KeyX:()=>setState("not_visible"),
  KeyO:()=>setState("out_of_frame"),
  KeyN:()=>go(1),
  KeyP:()=>go(-1),
  KeyS:()=>go(1,true),
  KeyD:()=>$("delete").click()
 };

 if(actions[event.code]){
  event.preventDefault();
  actions[event.code]();
 }
};

document.onkeyup=event=>{
 if(event.code==="Space")space=false;
};

window.addEventListener("blur",()=>{
 account();
 space=false;
 gesture=null;
 if(ann&&!busy&&!failed)autoQueue();
});

window.addEventListener("focus",()=>{
 last=performance.now();
});

document.addEventListener("visibilitychange",()=>{
 last=performance.now();
 if(document.hidden&&ann&&!busy&&!failed)autoQueue();
});

window.addEventListener("beforeunload",event=>{
 if(pending||dirty||failed){
  event.preventDefault();
  event.returnValue="";
 }
});

new ResizeObserver(resize).observe($("stage"));

setInterval(()=>{
 account();
 renderHeader();
 if(!pending&&!failed)saveStatus();
},1000);

setInterval(()=>{
 if(ann&&!busy&&!failed)autoQueue();
},10000);

async function boot(){
 try{
  const session=await api("/api/session");
  revision=session.revision;
  await load(session.cursor);
 }catch(e){
  error(e.message);
 }
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
    server_version = "AxisPuzzle/1"

    def log_message(self, fmt, *args):
        pass

    def valid_host(self):
        host = self.headers.get("Host", "")
        return host in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }

    def authorized(self, png=False):
        supplied = self.headers.get("X-Axis-Token", "")

        if png:
            from urllib.parse import parse_qs
            supplied = parse_qs(
                urlsplit(self.path).query
            ).get("t", [""])[0]

        return secrets.compare_digest(
            supplied,
            self.server.app.token,
        )

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
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def fail(self, status, message):
        self.send_json(status, {"error": message})

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

        route = urlsplit(self.path).path
        app = self.server.app

        try:
            if route == "/":
                page = HTML.replace(
                    "__TOKEN__",
                    json.dumps(app.token),
                )
                self.send_bytes(
                    200,
                    page.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if not self.authorized(
                png=route.startswith("/api/png/")
            ):
                self.fail(403, "Нет токена сессии.")
                return

            if route == "/api/session":
                with app.lock:
                    self.send_json(
                        200,
                        {
                            "cursor": app.data["session"]["cursor"],
                            "revision": app.data["session_revision"],
                        },
                    )
                return

            if route.startswith("/api/item/"):
                index = self.route_index(route, "/api/item/")
                self.send_json(200, app.view(index))
                return

            if route.startswith("/api/png/"):
                index = self.route_index(route, "/api/png/")
                uid = app.order[index]
                self.send_bytes(200, app.pngs[uid], "image/png")
                return

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
            if not 0 < length <= 300_000:
                raise ValueError("Некорректный размер запроса.")

            self.connection.settimeout(30)
            body = self.rfile.read(length)

            if len(body) != length:
                raise ValueError("Запрос получен не полностью.")

            def reject_constant(value):
                raise ValueError(f"Недопустимое число: {value}")

            raw = json.loads(
                body.decode("utf-8"),
                parse_constant=reject_constant,
            )

            self.send_json(200, self.server.app.save(raw))

        except Conflict as exc:
            self.fail(409, str(exc))
        except (ValueError, UnicodeError) as exc:
            self.fail(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(
                f"Ошибка записи: {type(exc).__name__}: {exc}",
                flush=True,
            )
            self.fail(
                500,
                "Ошибка атомарной записи. Проверьте диск и права доступа.",
            )


def main():
    parser = argparse.ArgumentParser(
        description="Разметка оси позвоночника на DXA-снимках."
    )
    parser.add_argument(
        "--index",
        default="data/index/images.csv",
    )
    parser.add_argument(
        "--root",
        default=".",
    )
    parser.add_argument(
        "--annotator",
    )
    parser.add_argument(
        "--out",
        default="data/annotations",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8769,
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.annotator or not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        args.annotator,
    ):
        parser.error(
            "--annotator: латинские буквы, цифры, _ и -."
        )

    if not 1 <= args.port <= 65535:
        parser.error("--port: 1–65535.")

    try:
        app = Application(args)
        server = Server(("127.0.0.1", args.port), app)
    except (
        OSError,
        ValueError,
        ImportError,
        csv.Error,
    ) as exc:
        parser.exit(1, f"Ошибка запуска: {exc}\n")

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n{url}\nОстановка: Ctrl+C.", flush=True)
    print(
        "Используйте одну вкладку на файл разметки.",
        flush=True,
    )

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