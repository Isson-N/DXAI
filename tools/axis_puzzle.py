#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Четвёртый инструмент разметки: точки замыкательных пластинок Th12--L5.

Запуск:
    python tools/axis_puzzle.py --annotator ivan
    python tools/axis_puzzle.py --annotator ivan --port 8769
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pydicom
from PIL import Image, ImageEnhance
from io import BytesIO


ROOT = Path(__file__).resolve().parents[1]
QUEUE_CSV = ROOT / "data/index/axis_puzzle_queue.csv"
IMAGES_CSV = ROOT / "data/index/images.csv"
ANNOTATIONS_DIR = ROOT / "data/annotations"

SCHEMA = "dxa-axis-puzzle/1"
VERTEBRAE = ("Th12", "L1", "L2", "L3", "L4", "L5")
POINT_NAMES = ("top", "bottom")
STATES = ("visible", "not_visible", "out_of_frame", "uncertain")
VERDICTS = ("deviated", "normal", "cannot_decide")
FEATURES = (
    "whole_column_tilted",
    "lumbar_only_tilted",
    "pelvis_oblique",
    "column_shifted_sideways",
    "vertebrae_rotated",
    "column_cut_by_frame",
    "uncertain",
)


def parse_number(value):
    """
    Разбор чисел из CSV.

    В частности, корректно обрабатывает значения pandas вида "0.0"/"1.0"
    и пустые клетки.
    """
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_queue():
    result = []
    with QUEUE_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("uid") or "").strip()
            if uid:
                result.append(uid)
    return result


def read_images():
    result = {}
    with IMAGES_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("sop_uid") or "").strip()
            path = (row.get("path") or "").strip()
            if not uid or not path:
                continue
            # Поля rows/cols читаются намеренно только как числа. Они не
            # возвращаются браузеру и не участвуют в слепом интерфейсе.
            result[uid] = {
                "path": path,
                "rows": parse_number(row.get("rows")),
                "cols": parse_number(row.get("cols")),
            }
    return result


def empty_point():
    # Уверенность 3 — «уверен», 1 — «сомневаюсь». По умолчанию 1 (решение
    # разметчика): уверенность повышается осознанно, а не достаётся даром.
    # Поэтому «1» у точки с координатой читается как «поставил, но сомневаюсь»,
    # а не как «значение не трогали».
    return {"x": None, "y": None, "state": "not_visible", "confidence": 1}


def empty_item():
    return {
        "points": {
            v: {p: empty_point() for p in POINT_NAMES}
            for v in VERTEBRAE
        },
        "comment": "",
        "verdict": None,
        "features": [],
        "status": "draft",
    }


def normalize_point(point):
    if not isinstance(point, dict):
        return empty_point()

    state = point.get("state")
    if state not in STATES:
        state = "not_visible"

    try:
        confidence = int(point.get("confidence", 1))
    except (TypeError, ValueError):
        confidence = 1
    confidence = max(1, min(3, confidence))

    x = point.get("x")
    y = point.get("y")

    if state in ("not_visible", "out_of_frame"):
        # Обязательное правило: эти состояния не могут иметь координату.
        x = None
        y = None
    else:
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            x = None
            y = None

        if x is None or y is None or not (0 <= x <= 1 and 0 <= y <= 1):
            x = None
            y = None

    return {
        "x": x,
        "y": y,
        "state": state,
        "confidence": confidence,
    }


def normalize_item(value):
    base = empty_item()
    if not isinstance(value, dict):
        return base

    points = value.get("points", {})
    for vertebra in VERTEBRAE:
        for point_name in POINT_NAMES:
            base["points"][vertebra][point_name] = normalize_point(
                points.get(vertebra, {}).get(point_name)
                if isinstance(points, dict)
                else None
            )

    comment = value.get("comment", "")
    if not isinstance(comment, str):
        comment = str(comment)
    base["comment"] = comment[:20000]

    verdict = value.get("verdict")
    base["verdict"] = verdict if verdict in VERDICTS else None

    features = value.get("features", [])
    if not isinstance(features, list):
        features = []
    base["features"] = [
        x for x in features if isinstance(x, str) and x in FEATURES
    ]

    if value.get("status") == "done":
        base["status"] = "done"

    return base


def load_annotation_file(filename):
    if not filename.exists():
        return {
            "schema": SCHEMA,
            "session_revision": secrets.token_urlsafe(24),
            "items": {},
        }

    try:
        with filename.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        # Не выдаём браузеру путь или подробности ошибки.
        data = {}

    items = data.get("items", {})
    if not isinstance(items, dict):
        items = {}

    return {
        "schema": SCHEMA,
        "session_revision": str(data.get("session_revision") or
                                secrets.token_urlsafe(24)),
        "items": items,
    }


def atomic_write(filename, data):
    """
    Атомарная запись: временный файл рядом с целевым + os.replace().
    Вызывается только под file_lock.
    """
    filename.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix="." + filename.name + ".",
        suffix=".tmp",
        dir=str(filename.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, filename)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


class App:
    def __init__(self, annotator):
        self.annotator = annotator
        self.filename = ANNOTATIONS_DIR / f"axis_puzzle_{annotator}.json"
        self.queue = read_queue()
        self.images = read_images()
        self.uid_by_index = self.queue[:]
        self.index_by_uid = {uid: i for i, uid in enumerate(self.queue)}

        self.file_lock = threading.RLock()
        self.data = load_annotation_file(self.filename)

        # Ревизия сессии выдаётся вкладке. Получение новой ревизии
        # инвалидирует предыдущую вкладку и предотвращает молчаливое
        # затирание файла.
        self.session_lock = threading.Lock()
        self.active_session = None

        # Защита от прихода старого асинхронного save после нового save.
        self.last_client_seq = {}

    def new_session(self):
        token = secrets.token_urlsafe(24)
        with self.session_lock:
            self.active_session = token
        return token

    def check_session(self, token):
        with self.session_lock:
            return bool(token) and token == self.active_session

    def get_item(self, index):
        if index < 0 or index >= len(self.uid_by_index):
            raise ValueError
        uid = self.uid_by_index[index]
        return normalize_item(self.data["items"].get(uid))

    def save_item(self, index, item, token, client_seq):
        if index < 0 or index >= len(self.uid_by_index):
            raise ValueError
        if not self.check_session(token):
            raise PermissionError

        try:
            client_seq = int(client_seq)
        except (TypeError, ValueError):
            raise ValueError

        with self.file_lock:
            old_seq = self.last_client_seq.get(index, -1)
            if client_seq < old_seq:
                return False

            checked = normalize_item(item)
            uid = self.uid_by_index[index]

            # Загружаем актуальный файл под блокировкой и меняем только UID
            # текущей очереди. Все записи вне этой очереди сохраняются.
            current = load_annotation_file(self.filename)
            current["schema"] = SCHEMA
            current["items"][uid] = checked
            current["session_revision"] = self.data.get(
                "session_revision", current.get("session_revision")
            )
            atomic_write(self.filename, current)
            self.data = current
            self.last_client_seq[index] = client_seq
            return True

    def image_bytes(self, index):
        if index < 0 or index >= len(self.uid_by_index):
            raise ValueError

        uid = self.uid_by_index[index]
        info = self.images.get(uid)
        if not info:
            raise FileNotFoundError

        ds = pydicom.dcmread(info["path"])
        array = ds.pixel_array.astype(np.float32)

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            array = np.max(array) - array

        finite = np.isfinite(array)
        if not finite.any():
            array = np.zeros_like(array)
        else:
            lo, hi = np.percentile(array[finite], (0.5, 99.5))
            if hi <= lo:
                lo = float(np.min(array[finite]))
                hi = float(np.max(array[finite]))
            if hi <= lo:
                array = np.zeros_like(array)
            else:
                array = np.clip((array - lo) / (hi - lo), 0, 1) * 255

        image = Image.fromarray(array.astype(np.uint8), mode="L")
        out = BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Разбор снимков</title>
<style>
html,body{margin:0;height:100%;font-family:Arial,sans-serif;background:#202124;color:#eee}
#layout{display:flex;height:100vh;overflow:hidden}
#left{flex:1;position:relative;background:#111;overflow:hidden}
canvas{display:block;width:100%;height:100%;cursor:crosshair}
#right{width:370px;box-sizing:border-box;overflow:auto;padding:14px;background:#292a2d}
label{display:block;margin:8px 0 4px}
button,input,select,textarea{font:inherit}
button{margin:2px;padding:6px 9px;background:#444;color:#fff;border:1px solid #777;border-radius:3px}
button.active{background:#1769aa}
textarea{width:100%;box-sizing:border-box;background:#1d1e20;color:#fff;border:1px solid #777}
textarea{height:125px;resize:vertical}
fieldset{border:1px solid #666;margin:10px 0;padding:8px}
.small{font-size:12px;color:#bbb}
#message{min-height:18px;color:#ffcc66}
#counter{font-weight:bold}
.pointrow{display:flex;gap:4px;align-items:center;margin:3px 0}
.pointrow span{width:48px}
.hint{line-height:1.4;font-size:12px;color:#ccc}
hr{border:0;border-top:1px solid #555}
input[type=range]{width:100%}
</style>
</head>
<body>
<div id="layout">
<div id="left"><canvas id="canvas"></canvas></div>
<div id="right">
  <div id="counter"></div>
  <div id="message"></div>

  <fieldset>
    <legend>Точка</legend>
    <div id="selected"></div>
    <div id="pointButtons"></div>
    <div>
      Состояние:
      <button data-state="visible">видимая</button>
      <button data-state="uncertain">сомнительная</button>
      <button data-state="not_visible">не видна</button>
      <button data-state="out_of_frame">за кадром</button>
    </div>
    <div>
      Уверенность:
      <button data-confidence="3">3 — уверен</button>
      <button data-confidence="2">2 — так себе</button>
      <button data-confidence="1">1 — сомневаюсь</button>
    </div>
    <button id="deletePoint">Удалить координату</button>
  </fieldset>

  <fieldset>
    <legend>Вердикт</legend>
    <label><input type="radio" name="verdict" value="deviated"> ось отклонена</label>
    <label><input type="radio" name="verdict" value="normal"> норма</label>
    <label><input type="radio" name="verdict" value="cannot_decide"> не могу решить</label>
  </fieldset>

  <fieldset>
    <legend>Признаки</legend>
    <div id="features"></div>
  </fieldset>

  <label for="comment">Комментарий</label>
  <textarea id="comment" maxlength="20000"
    placeholder="Что именно выглядит наклонённым, относительно чего и что мешает решить?"></textarea>

  <div>
    <button id="prev">← Предыдущий</button>
    <button id="next">Следующий →</button>
    <button id="done">Завершить явно</button>
    <button id="draft">Оставить черновиком</button>
  </div>

  <fieldset>
    <legend>Изображение</legend>
    <label>Яркость <input id="brightness" type="range" min="0.2" max="2.5" step="0.05" value="1"></label>
    <label>Контраст <input id="contrast" type="range" min="0.2" max="3" step="0.05" value="1"></label>
    <div class="small">Колесо — зум; ПКМ или Space+ЛКМ — панорама.</div>
  </fieldset>

  <fieldset class="hint">
    <legend>Клавиши</legend>
    1–6 — выбрать Th12/L1/L2/L3/L4/L5;<br>
    T — верхняя пластинка, B — нижняя;<br>
    V — видимая, U — сомнительная,<br>
    N — не видна, O — за кадром;<br>
    1/2/3 в режиме уверенности — уверенность;<br>
    Delete/Backspace — удалить координату;<br>
    ←/→ — предыдущий/следующий снимок;<br>
    +/− — зум; Space — панорама;<br>
    Ctrl+Enter — завершить явно.
  </fieldset>
</div>
</div>
<script>
"use strict";

const features = [
  ["whole_column_tilted","наклонён весь столб"],
  ["lumbar_only_tilted","наклонён только поясничный отдел"],
  ["pelvis_oblique","таз перекошен"],
  ["column_shifted_sideways","столб смещён вбок"],
  ["vertebrae_rotated","позвонки повёрнуты вокруг своей оси"],
  ["column_cut_by_frame","кадр обрезает столб"],
  ["uncertain","сомневаюсь"]
];
const vertebrae = ["Th12","L1","L2","L3","L4","L5"];
const pointNames = ["top","bottom"];

let index = 0, total = 0, item = null, sessionRevision = null;
let selectedV = 0, selectedP = "top", clientSeq = 0;
let image = new Image(), imageReady = false;
// Один флаг на панораму и на перетаскивание точки приводил к тому, что захват
// точки двигал всё изображение. Режим жеста теперь различается явно.
let zoom = 1, panX = 0, panY = 0, dragMode = null, dragX = 0, dragY = 0;
let brightness = 1, contrast = 1, commentTimer = null;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const $ = id => document.getElementById(id);

function timeoutFetch(url, options={}) {
  // У каждого запроса есть собственный AbortController и таймаут 15 секунд.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  options.signal = controller.signal;
  return fetch(url, options).finally(() => clearTimeout(timer));
}

function showMessage(s) { $("message").textContent = s || ""; }

async function getJSON(url, options={}) {
  const r = await timeoutFetch(url, options);
  if (!r.ok) throw new Error("network");
  return r.json();
}

function selectedPoint() {
  return item.points[vertebrae[selectedV]][selectedP];
}

function pointLabel(v, p) {
  return vertebrae[v] + " " + (p === "top" ? "верх" : "низ");
}

function renderControls() {
  $("selected").textContent = "Выбрано: " + pointLabel(selectedV, selectedP);
  document.querySelectorAll("[data-state]").forEach(b => {
    b.classList.toggle("active", b.dataset.state === selectedPoint().state);
  });
  document.querySelectorAll("[data-confidence]").forEach(b => {
    b.classList.toggle("active", Number(b.dataset.confidence) === selectedPoint().confidence);
  });

  document.querySelectorAll("input[name=verdict]").forEach(x => {
    x.checked = x.value === item.verdict;
  });

  const box = $("features");
  box.innerHTML = "";
  features.forEach(([key, text]) => {
    const lab = document.createElement("label");
    lab.innerHTML = `<input type="checkbox" data-feature="${key}"> ${text}`;
    lab.querySelector("input").checked = item.features.includes(key);
    box.appendChild(lab);
  });
  $("comment").value = item.comment || "";
  $("counter").textContent =
    `Снимок ${index + 1} из ${total}` +
    (item.status === "done" ? " — завершён" : " — черновик");
}

function imageTransform() {
  const iw = image.naturalWidth || 1, ih = image.naturalHeight || 1;
  const scale = Math.min(canvas.width / iw, canvas.height / ih) * zoom;
  return {
    scale,
    ox: (canvas.width - iw * scale) / 2 + panX,
    oy: (canvas.height - ih * scale) / 2 + panY
  };
}

function imageToScreen(x, y) {
  const t = imageTransform();
  return [t.ox + x * image.naturalWidth * t.scale,
          t.oy + y * image.naturalHeight * t.scale];
}

function screenToImage(x, y) {
  const t = imageTransform();
  return [
    (x - t.ox) / (image.naturalWidth * t.scale),
    (y - t.oy) / (image.naturalHeight * t.scale)
  ];
}

function draw() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!imageReady) return;

  const t = imageTransform();
  ctx.save();
  ctx.filter = `brightness(${brightness}) contrast(${contrast})`;
  ctx.drawImage(image, t.ox, t.oy,
    image.naturalWidth * t.scale, image.naturalHeight * t.scale);
  ctx.restore();

  // Тонкие вертикальная и горизонтальная линии края кадра.
  ctx.strokeStyle = "rgba(80,220,255,.75)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(t.ox, 0); ctx.lineTo(t.ox, h);
  ctx.moveTo(t.ox + image.naturalWidth*t.scale, 0);
  ctx.lineTo(t.ox + image.naturalWidth*t.scale, h);
  ctx.moveTo(0, t.oy); ctx.lineTo(w, t.oy);
  ctx.moveTo(0, t.oy + image.naturalHeight*t.scale);
  ctx.lineTo(w, t.oy + image.naturalHeight*t.scale);
  ctx.stroke();

  const pts = [];
  vertebrae.forEach((v, vi) => pointNames.forEach(p => {
    const q = item.points[v][p];
    if (q.x !== null && q.y !== null &&
        (q.state === "visible" || q.state === "uncertain")) {
      const [sx, sy] = imageToScreen(q.x, q.y);
      pts.push([sx, sy, pointLabel(vi, p), vi, p]);
    }
  }));

  // Линии соединяют поставленные точки в порядке сверху вниз.
  ctx.strokeStyle = "#ffdf4d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  pts.forEach(([x,y]) => {
    if (!started) { ctx.moveTo(x,y); started = true; }
    else ctx.lineTo(x,y);
  });
  ctx.stroke();

  pts.forEach(([x,y,label,vi,p]) => {
    const selected = vi === selectedV && p === selectedP;
    ctx.fillStyle = selected ? "#ff3333" : "#00ff88";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(x,y,selected ? 7 : 5,0,Math.PI*2);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "13px Arial";
    ctx.fillText(label, x+8, y-7);
  });
}

function resize() {
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  // После задания CSS-пиксельного transform используем CSS-размеры.
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  draw();
}
window.addEventListener("resize", resize);

async function loadIndex(n) {
  if (n < 0) n = 0;
  if (n >= total) n = total - 1;
  index = n;
  const data = await getJSON("/api/state/" + index);
  item = data.item;
  imageReady = false;
  image = new Image();
  image.onload = () => { imageReady = true; resize(); };
  image.onerror = () => showMessage("Не удалось загрузить изображение");
  image.src = "/api/image/" + index;
  renderControls();
  draw();
}

function snapshot() {
  return JSON.parse(JSON.stringify(item));
}

function save(reason="") {
  const capturedIndex = index;
  const capturedRevision = sessionRevision;
  const capturedItem = snapshot();
  const capturedSeq = ++clientSeq;

  // Все значения, включая индекс снимка и ревизию, захвачены до async-запроса.
  // Поэтому переход на другой снимок не может отправить туда старую разметку.
  timeoutFetch("/api/save", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      index: capturedIndex,
      revision: capturedRevision,
      seq: capturedSeq,
      item: capturedItem
    })
  }).then(r => {
    if (!r.ok) throw new Error("save");
    return r.json();
  }).catch(e => {
    showMessage(e.name === "AbortError" ? "Сохранение прервано по таймауту"
                                        : "Сохранение не выполнено");
  });
}

function mutate(fn) {
  fn();
  renderControls();
  draw();
  save();
}

document.querySelectorAll("[data-state]").forEach(b => {
  b.onclick = () => mutate(() => {
    const q = selectedPoint();
    q.state = b.dataset.state;
    if (q.state === "not_visible" || q.state === "out_of_frame") {
      q.x = null; q.y = null;
    }
  });
});
document.querySelectorAll("[data-confidence]").forEach(b => {
  b.onclick = () => mutate(() => {
    selectedPoint().confidence = Number(b.dataset.confidence);
  });
});
$("deletePoint").onclick = () => mutate(() => {
  const q = selectedPoint();
  q.x = null; q.y = null;
  q.state = "not_visible";
});

document.querySelectorAll("input[name=verdict]").forEach(x => {
  x.onchange = () => mutate(() => { item.verdict = x.value; });
});
$("features").onclick = e => {
  if (!e.target.dataset.feature) return;
  mutate(() => {
    const key = e.target.dataset.feature;
    item.features = e.target.checked
      ? [...new Set([...item.features, key])]
      : item.features.filter(x => x !== key);
  });
};
$("comment").oninput = () => {
  item.comment = $("comment").value.slice(0,20000);
  clearTimeout(commentTimer);
  commentTimer = setTimeout(() => save(), 250);
};

$("prev").onclick = () => loadIndex(index - 1);
$("next").onclick = () => loadIndex(index + 1);
$("done").onclick = () => mutate(() => { item.status = "done"; });
$("draft").onclick = () => mutate(() => { item.status = "draft"; });

$("brightness").oninput = e => { brightness = Number(e.target.value); draw(); };
$("contrast").oninput = e => { contrast = Number(e.target.value); draw(); };

canvas.oncontextmenu = e => e.preventDefault();
canvas.onmousedown = e => {
  if (e.button === 2 || e.button === 0 && keys.has(" ")) {
    dragMode = "pan"; dragX = e.clientX; dragY = e.clientY;
    return;
  }
  if (e.button !== 0 || !imageReady) return;

  const r = canvas.getBoundingClientRect();
  const sx = e.clientX-r.left, sy = e.clientY-r.top;

  // Поиск ближайшей точки явно начинается с -1. Пустой набор возвращает
  // "ничего", а не первый элемент.
  let best = -1, bestDist = Infinity;
  vertebrae.forEach((v, vi) => pointNames.forEach(p => {
    const q = item.points[v][p];
    if (q.x === null || q.y === null) return;
    const [x,y] = imageToScreen(q.x,q.y);
    const d = Math.hypot(x-sx,y-sy);
    if (d < bestDist) {
      bestDist = d;
      best = vi*2 + (p === "bottom" ? 1 : 0);
    }
  }));
  if (best >= 0 && bestDist <= 10) {
    selectedV = Math.floor(best/2);
    selectedP = best % 2 ? "bottom" : "top";
    renderControls(); draw();
    dragMode = "point"; dragX=e.clientX; dragY=e.clientY;
    return;
  }

  const q = selectedPoint();
  if (q.state !== "visible" && q.state !== "uncertain") return;
  const [x,y] = screenToImage(sx,sy);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  mutate(() => { q.x=x; q.y=y; });
};
window.onmouseup = () => {
  // Перенос точки записывается один раз, в конце жеста, а не на каждом движении.
  if (dragMode === "point") save();
  dragMode = null;
};
window.onmousemove = e => {
  if (!dragMode) return;
  if (dragMode === "pan") {
    panX += e.clientX-dragX; panY += e.clientY-dragY;
    dragX=e.clientX; dragY=e.clientY; draw();
    return;
  }
  const q = selectedPoint();
  if (!q || q.state !== "visible" && q.state !== "uncertain") return;
  const r = canvas.getBoundingClientRect();
  const [x,y] = screenToImage(e.clientX-r.left, e.clientY-r.top);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  q.x = x; q.y = y; draw();
};
canvas.onwheel = e => {
  e.preventDefault();
  zoom = Math.max(.2, Math.min(12, zoom * (e.deltaY < 0 ? 1.12 : .89)));
  draw();
};

let keys = new Set();
window.onkeydown = e => {
  keys.add(e.key);
  if (e.key === " " || e.key === "ArrowLeft" || e.key === "ArrowRight")
    e.preventDefault();

  if (/^[1-6]$/.test(e.key)) {
    selectedV = Number(e.key)-1;
    // Цифры выбирают позвонок; состояние уверенности меняется кнопками
    // и не смешивается с выбором позвонка.
    renderControls(); draw();
  } else if (e.key.toLowerCase() === "t") {
    selectedP="top"; renderControls(); draw();
  } else if (e.key.toLowerCase() === "b") {
    selectedP="bottom"; renderControls(); draw();
  } else if (e.key.toLowerCase() === "v") {
    document.querySelector('[data-state="visible"]').click();
  } else if (e.key.toLowerCase() === "u") {
    document.querySelector('[data-state="uncertain"]').click();
  } else if (e.key.toLowerCase() === "n") {
    document.querySelector('[data-state="not_visible"]').click();
  } else if (e.key.toLowerCase() === "o") {
    document.querySelector('[data-state="out_of_frame"]').click();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    $("deletePoint").click();
  } else if (e.key === "ArrowLeft") {
    loadIndex(index-1);
  } else if (e.key === "ArrowRight") {
    loadIndex(index+1);
  } else if (e.key === "+" || e.key === "=") {
    zoom=Math.min(12,zoom*1.12); draw();
  } else if (e.key === "-" || e.key === "_") {
    zoom=Math.max(.2,zoom/1.12); draw();
  } else if (e.key === "Enter" && e.ctrlKey) {
    $("done").click();
  }
};
window.onkeyup = e => keys.delete(e.key);

async function start() {
  try {
    const s = await getJSON("/api/session");
    sessionRevision = s.revision;
    total = s.total;
    await loadIndex(0);
  } catch(e) {
    showMessage("Не удалось открыть очередь");
  }
}
start();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AxisPuzzle/1"

    def log_message(self, fmt, *args):
        # Не печатаем UID, пути, группы или другие сведения о снимках.
        pass

    @property
    def app(self):
        return self.server.app

    def send_json(self, code, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def generic_error(self, code=HTTPStatus.BAD_REQUEST):
        # Ошибки намеренно не содержат UID, путь, метку, угол или группу.
        self.send_json(code, {"error": "request rejected"})

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 2_000_000:
                raise ValueError
            return json.loads(self.rfile.read(length))
        except Exception:
            raise ValueError

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/session":
            token = self.app.new_session()
            self.send_json(200, {
                "revision": token,
                "total": len(self.app.uid_by_index),
            })
            return

        if path.startswith("/api/state/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                self.send_json(200, {"item": self.app.get_item(index)})
            except Exception:
                self.generic_error(404)
            return

        if path.startswith("/api/image/"):
            try:
                index = int(path.rsplit("/", 1)[1])
                raw = self.app.image_bytes(index)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except Exception:
                self.generic_error(404)
            return

        self.generic_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            self.generic_error(404)
            return

        try:
            body = self.read_json()
            index = int(body["index"])
            revision = str(body["revision"])
            seq = int(body["seq"])
            item = body["item"]

            accepted = self.app.save_item(index, item, revision, seq)
            self.send_json(200, {"saved": bool(accepted)})
        except PermissionError:
            self.generic_error(409)
        except Exception:
            self.generic_error(400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()

    # Имя разметчика используется только для имени файла аннотаций.
    # В HTML/API оно не передаётся.
    app = App(args.annotator)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.app = app

    print(f"http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()