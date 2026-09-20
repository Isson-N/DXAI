#!/usr/bin/env python3
"""
Blind DXA axis review and agreement report.

Dependencies:
    pip install numpy Pillow pydicom

Review:
    python tools/axis_review.py \
        --index data/index/images.csv \
        --reviewer reviewerA \
        --out data/annotations \
        --seed 20260920 \
        --clean-pixels-confirmed

Report:
    python tools/axis_review.py \
        --index data/index/images.csv \
        --report 'data/annotations/axis_review_*.json'

Image paths:
    absolute paths are used as-is;
    relative paths are resolved from CWD first, then CSV directory.

The web interface receives neither SOP UID nor original labels.
The server listens on 127.0.0.1 only.

"revised" is sticky and becomes true when:
  * a saved verdict changes, even during the same visit;
  * on a subsequent visit, any substantive answer field changes.

Initial completion of reasons/confidence during the first visit does
not itself count as revision. This is a flag, not a full event history.

Answers are saved automatically once verdict AND confidence are selected.
N/P/S also flush any complete current answer before navigation.
An incomplete answer blocks N/P; S discards its incomplete draft.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit
import webbrowser

import numpy as np


SCHEMA = "dxa-axis-review/1"
VERDICTS = ("deviated", "normal", "unsure")
LABELS = {
    "deviated": "отклонена",
    "normal": "норма",
    "unsure": "не могу решить",
}
REASONS = (
    "наклон всего позвоночника",
    "изгиб (сколиотическая дуга)",
    "поворот таза",
    "смещение позвоночника относительно центра кадра",
    "позвоночник непараллелен краю кадра",
    "плохое качество изображения",
    "другое",
)
NOTE_LIMIT = 240
# Скрытые повторы: тот же снимок показывается второй раз под ключом «uid#2».
# Нужны, чтобы отличить «метка шумная» от «разметчик нестабилен» (требование
# astra и fable, 20.09.2026): без них κ внешнего согласия неинтерпретируем.
REPEAT_SUFFIX = "#2"
# Перекрытие частей здесь больше, чем в остальных инструментах: общие снимки
# дают МЕЖэкспертное согласие, а оно ценнее внутриэкспертного (обе модели,
# 20.09.2026). Внутриэкспертное меряется скрытыми повторами.
AXIS_OVERLAP = 25


def slot_uid(key):
    """Ключ очереди → идентификатор снимка (у повтора отбрасывается суффикс)."""
    return key.split(REPEAT_SUFFIX)[0]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def read_index(filename):
    filename = Path(filename).resolve()
    records = {}
    seen = set()
    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sop_uid", "study", "path", "rows", "cols", "y_axis", "region"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("В CSV отсутствуют колонки: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            uid = (row["sop_uid"] or "").strip()
            if not uid:
                raise ValueError(f"Пустой sop_uid в строке {line}")
            if uid in seen:
                raise ValueError(f"Повторный sop_uid в CSV: {uid}")
            seen.add(uid)

            # pandas пишет метку как «0.0»/«1.0», поэтому разбираем через float.
            value = (row["y_axis"] or "").strip()
            if not value:
                continue
            try:
                number = float(value)
            except ValueError:
                raise ValueError(f"y_axis должен быть 0/1/пусто, строка {line}") from None
            if number not in (0.0, 1.0):
                raise ValueError(f"y_axis должен быть 0/1/пусто, строка {line}")
            value = str(int(number))

            raw_path = (row["path"] or "").strip()
            if not raw_path:
                raise ValueError(f"Пустой path, строка {line}")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                cwd_candidate = Path.cwd() / path
                path = (
                    cwd_candidate
                    if cwd_candidate.exists()
                    else filename.parent / path
                )

            # Original label stays server-side; it is never part of API data.
            records[uid] = {
                "y": int(value),
                "study": (row["study"] or "").strip(),
                "path": path.resolve(),
            }

    if not records:
        raise ValueError("В индексе нет снимков с непустым y_axis")
    return records


def validate_answer(answer):
    if not isinstance(answer, dict):
        raise ValueError("Ответ должен быть объектом")
    if answer.get("verdict") not in VERDICTS:
        raise ValueError("Некорректный verdict")
    if type(answer.get("confident")) is not bool:
        raise ValueError("confident должен быть boolean")
    reasons = answer.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("reasons должен быть списком")
    if any(not isinstance(r, str) or r not in REASONS for r in reasons):
        raise ValueError("Неизвестная причина")
    if len(reasons) != len(set(reasons)):
        raise ValueError("Повторяющиеся причины")
    note = answer.get("note")
    if not isinstance(note, str) or len(note) > NOTE_LIMIT:
        raise ValueError(f"note должен быть строкой длиной не более {NOTE_LIMIT}")
    if note.strip() and "другое" not in reasons:
        raise ValueError("Текст допустим только с причиной «другое»")
    if "seconds" in answer:
        seconds = answer["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Некорректный seconds")
    if "revised" in answer and type(answer["revised"]) is not bool:
        raise ValueError("revised должен быть boolean")


def substantive(answer):
    if answer is None:
        return None
    return (
        answer["verdict"],
        answer["confident"],
        tuple(sorted(answer["reasons"])),
        answer["note"],
    )


def load_review(filename):
    with Path(filename).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"{filename}: неизвестная схема")
    if not isinstance(data.get("reviewer"), str):
        raise ValueError(f"{filename}: отсутствует reviewer")
    if type(data.get("seed")) is not int:
        raise ValueError(f"{filename}: некорректный seed")
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(f"{filename}: отсутствует answers")
    for uid, answer in answers.items():
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"{filename}: некорректный sop_uid")
        validate_answer(answer)
        if "seconds" not in answer or "revised" not in answer:
            raise ValueError(f"{filename}: отсутствуют seconds/revised")
    return data


def atomic_json(filename, data):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=filename.name + ".", suffix=".tmp", dir=str(filename.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
        # Directory fsync is not available on every OS/filesystem.
        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(str(filename.parent), flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_dicom(path):
    # Imported only in review mode: reporting does not need a DICOM decoder.
    import pydicom
    from PIL import Image
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import (
            apply_modality_lut,
            apply_voi_lut,
        )

    ds = pydicom.dcmread(str(path))
    photo = str(getattr(ds, "PhotometricInterpretation", ""))
    if photo not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Поддерживаются только монохромные DICOM")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("Многокадровый DICOM не поддерживается")

    pixels = np.asarray(ds.pixel_array)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise ValueError("Ожидалось двумерное изображение")

    # Exclude padding from contrast estimation.
    valid = np.isfinite(pixels)
    padding = getattr(ds, "PixelPaddingValue", None)
    if padding is not None:
        limit = getattr(ds, "PixelPaddingRangeLimit", padding)
        lo_pad, hi_pad = sorted((float(padding), float(limit)))
        valid &= ~((pixels >= lo_pad) & (pixels <= hi_pad))

    image = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float64)
    has_voi = (
        bool(getattr(ds, "VOILUTSequence", None))
        or ("WindowCenter" in ds and "WindowWidth" in ds)
    )
    if has_voi:
        image = np.asarray(apply_voi_lut(image, ds), dtype=np.float64)

    valid &= np.isfinite(image)
    if not np.any(valid):
        raise ValueError("Нет пригодных пикселей")
    values = image[valid]
    if has_voi:
        low, high = float(values.min()), float(values.max())
    else:
        low, high = map(float, np.percentile(values, [0.5, 99.5]))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0

    image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    if photo == "MONOCHROME1":
        image = 1.0 - image
    image[~valid] = 0.0

    # No metadata, DICOM overlays, filenames or annotations are copied.
    output = io.BytesIO()
    Image.fromarray(np.rint(image * 255).astype(np.uint8)).save(
        output, format="PNG"
    )
    return output.getvalue()


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Пересмотр оси</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
body { margin: 0; background: #17191c; color: #eee; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 370px;
       min-height: 100vh; }
#viewer { background: #000; display: flex; align-items: center;
          justify-content: center; min-width: 0; }
#image { max-width: 100%; max-height: 98vh; object-fit: contain; }
aside { padding: 20px; }
h1 { font-size: 21px; margin-top: 0; }
button, textarea { font: inherit; }
button { background: #303640; color: #fff; border: 1px solid #687080;
         border-radius: 5px; padding: 10px; cursor: pointer; }
button.selected { background: #174c73; border-color: #6dccff; }
button:disabled { opacity: .5; cursor: default; }
.stack { display: grid; gap: 8px; }
.row { display: flex; gap: 8px; flex-wrap: wrap; }
fieldset { border: 0; margin: 18px 0; padding: 0; }
legend { margin-bottom: 9px; }
label { display: block; margin: 10px 0; }
textarea { width: 100%; box-sizing: border-box; min-height: 70px; }
small { color: #bac1cc; }
#status { min-height: 3em; margin: 14px 0; }
.error { color: #ffb6b6; }
[hidden] { display: none !important; }
@media (max-width: 850px) {
  main { display: block; }
  #image { max-height: 65vh; }
}
</style>
</head>
<body>
<main>
<div id="viewer"><img id="image" alt="Изображение для пересмотра"></div>
<aside>
<h1>Ось позвоночника отклонена?</h1>
<div id="form">
  <div id="verdicts" class="stack">
    <button type="button" data-verdict="deviated">1 — ось отклонена</button>
    <button type="button" data-verdict="normal">2 — норма</button>
    <button type="button" data-verdict="unsure">0 — не могу решить</button>
  </div>
  <fieldset>
    <legend>Уверенность</legend>
    <div id="confidence" class="row">
      <button type="button" data-confidence="true">Q — уверен</button>
      <button type="button" data-confidence="false">W — сомневаюсь</button>
    </div>
  </fieldset>
  <fieldset id="reasonBox" hidden>
    <legend>Причины — можно несколько</legend>
    <div id="reasons"></div>
    <div id="noteBox" hidden>
      <textarea id="note" maxlength="240"
                placeholder="Коротко: другая причина"></textarea>
    </div>
    <small>Причины: клавиши A, D, F, G, H, J, K.</small>
  </fieldset>
</div>
<div id="status" role="status" aria-live="polite"></div>
<div class="row">
  <button id="previous" type="button">P — назад</button>
  <button id="next" type="button">N — дальше</button>
  <button id="skip" type="button">S — пропустить</button>
</div>
<p><small>
Ответ сохраняется автоматически после выбора вердикта и уверенности.
Пропуск не равен ответу «не могу решить» и не удаляет сохранённый ответ.
</small></p>
</aside>
</main>
<script>
"use strict";
const BASE = "";
const TOKEN = __BASE__;
const REASONS = __REASONS__;
const reasonKeys = ["a","d","f","g","h","j","k"];
const el = id => document.getElementById(id);
let current = null;
let draft = emptyDraft();
let loaded = false;
let navigating = false;
let failed = false;
let queue = Promise.resolve();
let lastQueued = null;

function emptyDraft() {
  return {verdict:null, confident:null, reasons:[], note:""};
}
function message(text, error=false) {
  el("status").textContent = text;
  el("status").className = error ? "error" : "";
}
async function api(route, data) {
  const response = await fetch(BASE + route, {
    method:"POST",
    headers:{"Content-Type":"application/json", "X-Axis-Review":TOKEN},
    body:JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Ошибка запроса");
  return result;
}
function schedule(task) {
  queue = queue.then(async () => {
    if (failed) return;
    try { await task(); }
    catch (error) {
      failed = true;
      navigating = false;
      message(
        "Не удалось сохранить или загрузить данные. Не продолжайте. " +
        "Проверьте сервер, затем перезагрузите страницу.", true
      );
      console.error("Review request failed");
    }
  });
  return queue;
}
function ready() {
  return loaded && !navigating && !failed && current && !current.done;
}
function complete() {
  return draft.verdict !== null && typeof draft.confident === "boolean";
}
function snapshot() {
  return {
    verdict:draft.verdict, confident:draft.confident,
    reasons:[...draft.reasons], note:draft.note
  };
}
function syncUI() {
  document.querySelectorAll("[data-verdict]").forEach(b => {
    b.classList.toggle("selected", b.dataset.verdict === draft.verdict);
  });
  document.querySelectorAll("[data-confidence]").forEach(b => {
    b.classList.toggle(
      "selected",
      draft.confident === (b.dataset.confidence === "true")
    );
  });
  document.querySelectorAll("[data-reason]").forEach(box => {
    box.checked = draft.reasons.includes(REASONS[Number(box.dataset.reason)]);
  });
  el("reasonBox").hidden = draft.verdict === null;
  el("noteBox").hidden = !draft.reasons.includes("другое");
  if (el("note").value !== draft.note) el("note").value = draft.note;
}
function saveIfComplete(force=false) {
  if (!current || current.done || !complete()) return;
  const answer = snapshot();
  const signature = JSON.stringify(answer);
  if (!force && signature === lastQueued) return;
  lastQueued = signature;
  const visit = current.visit;
  message("Сохранение…");
  schedule(async () => {
    await api("/save", {visit, answer});
    if (current && current.visit === visit) message("Сохранено");
  });
}
function changed() {
  syncUI();
  if (complete()) saveIfComplete();
  else message("Выберите вердикт и уверенность.");
}
function selectVerdict(value) {
  if (!ready()) return;
  draft.verdict = value;
  changed();
}
function selectConfidence(value) {
  if (!ready()) return;
  draft.confident = value;
  changed();
}
function toggleReason(index) {
  if (!ready() || draft.verdict === null) return;
  const reason = REASONS[index];
  if (draft.reasons.includes(reason)) {
    draft.reasons = draft.reasons.filter(r => r !== reason);
  } else {
    draft.reasons.push(reason);
  }
  draft.reasons = REASONS.filter(r => draft.reasons.includes(r));
  if (!draft.reasons.includes("другое")) draft.note = "";
  changed();
}
async function show(result) {
  loaded = false;
  current = result;
  lastQueued = null;
  el("previous").disabled = !result.can_previous;
  el("next").disabled = result.done;
  el("skip").disabled = result.done;
  el("form").hidden = result.done;

  if (result.done) {
    el("image").removeAttribute("src");
    message(
      "Конец очереди. Сохранённые ответы не изменены. " +
      "К пропускам можно вернуться кнопкой «назад»; " +
      "после перезапуска откроется первый пропуск."
    );
    navigating = false;
    return;
  }

  draft = result.answer ? {
    verdict:result.answer.verdict,
    confident:result.answer.confident,
    reasons:[...result.answer.reasons],
    note:result.answer.note
  } : emptyDraft();
  syncUI();

  await new Promise(resolve => {
    el("image").onload = () => {
      loaded = true;
      navigating = false;
      message(result.answer ? "Сохранённый ответ" : "Выберите ответ.");
      resolve();
    };
    el("image").onerror = () => {
      navigating = false;
      message(
        "Изображение не удалось загрузить. Можно пропустить его; " +
        "ответ не будет создан.", true
      );
      resolve();
    };
    el("image").src = BASE + "/image/" + result.image;
  });
}
function navigate(delta, skip=false) {
  if (failed || navigating || !current) return;
  if (current.done && delta > 0) return;
  if (!current.done) {
    const partial = draft.verdict !== null || draft.confident !== null;
    if (!skip && partial && !complete()) {
      message("Завершите ответ или нажмите S для пропуска.", true);
      return;
    }
    if (!skip && !loaded && delta > 0) {
      message("Изображение недоступно. Используйте S для пропуска.", true);
      return;
    }
    // Flush timing and the last complete answer, including pending note edits.
    if (complete()) saveIfComplete(true);
  }
  navigating = true;
  schedule(async () => {
    const result = await api("/enter", {delta});
    await show(result);
  });
}

document.querySelectorAll("[data-verdict]").forEach(b => {
  b.onclick = () => selectVerdict(b.dataset.verdict);
});
document.querySelectorAll("[data-confidence]").forEach(b => {
  b.onclick = () => selectConfidence(b.dataset.confidence === "true");
});
REASONS.forEach((reason, index) => {
  const label = document.createElement("label");
  const box = document.createElement("input");
  box.type = "checkbox";
  box.dataset.reason = String(index);
  box.onchange = () => {
    toggleReason(index);
    syncUI();
  };
  label.append(box, document.createTextNode(
    " " + reasonKeys[index].toUpperCase() + " — " + reason
  ));
  el("reasons").append(label);
});
el("note").oninput = () => {
  if (!ready()) return;
  draft.note = el("note").value;
  saveIfComplete();
};
el("previous").onclick = () => navigate(-1);
el("next").onclick = () => navigate(1);
el("skip").onclick = () => navigate(1, true);

document.addEventListener("keydown", event => {
  if (event.ctrlKey || event.metaKey || event.altKey || event.repeat) return;
  const target = event.target;
  if (target && (
      target.tagName === "TEXTAREA" ||
      (target.tagName === "INPUT" && target.type !== "checkbox") ||
      target.isContentEditable
  )) return;

  // Physical Latin key positions also work with a Russian keyboard layout.
  let key = event.code.startsWith("Key")
    ? event.code.slice(3).toLowerCase()
    : event.code.startsWith("Digit")
      ? event.code.slice(5)
      : event.code.startsWith("Numpad")
        ? event.code.slice(6)
        : event.key.toLowerCase();

  const actions = {
    "1":()=>selectVerdict("deviated"),
    "2":()=>selectVerdict("normal"),
    "0":()=>selectVerdict("unsure"),
    "q":()=>selectConfidence(true),
    "w":()=>selectConfidence(false),
    "n":()=>navigate(1),
    "p":()=>navigate(-1),
    "s":()=>navigate(1,true)
  };
  if (actions[key]) {
    event.preventDefault();
    actions[key]();
  } else if (reasonKeys.includes(key)) {
    event.preventDefault();
    toggleReason(reasonKeys.indexOf(key));
  }
});
window.addEventListener("beforeunload", event => {
  if (failed || el("status").textContent === "Сохранение…") {
    event.preventDefault();
    event.returnValue = "";
  }
});
navigating = true;
schedule(async () => show(await api("/enter", {delta:0})));
</script>
</body>
</html>
"""


class ReviewApp:
    def __init__(self, records, args, outfile):
        self.records = records
        self.outfile = outfile
        # Адрес — обычный http://127.0.0.1:<порт>/, чтобы его можно было
        # набрать руками. От посторонних запросов к localhost защищает не
        # секретный путь, а заголовок X-Axis-Review со случайным токеном.
        self.base = ""
        self.token = secrets.token_urlsafe(24)
        self.order = sorted(records)
        random.Random(args.seed).shuffle(self.order)
        # Explicitly avoid the exact CSV order, including very small cohorts.
        if len(self.order) > 1 and self.order == list(records):
            self.order = self.order[1:] + self.order[:1]

        # Повторы берём из первой половины очереди и расставляем во второй,
        # равномерно и не подряд; разметчику они ничем не отличаются от новых.
        repeats = min(args.duplicates, len(self.order) // 4)
        if repeats > 0:
            rng = random.Random(args.seed ^ 0x5EED)
            half = len(self.order) // 2
            step = max(1, half // repeats)
            picked = [self.order[i * step] for i in range(repeats)]
            for shift, uid in enumerate(picked):
                low = half + shift + 1
                high = len(self.order) + shift
                self.order.insert(rng.randint(low, high), uid + REPEAT_SUFFIX)

        if outfile.exists():
            self.data = load_review(outfile)
            if (
                self.data["reviewer"] != args.reviewer
                or self.data["seed"] != args.seed
            ):
                raise ValueError("reviewer/seed сохранённого файла не совпадают")
            unknown = {key for key in self.data["answers"]
                       if slot_uid(key) not in records}
            if unknown:
                raise ValueError(
                    "Сохранённый файл содержит снимки вне текущей выборки. "
                    "Используйте исходный индекс."
                )
            # Whitelist persisted fields: never propagate arbitrary metadata.
            self.data = {
                "schema": SCHEMA,
                "reviewer": args.reviewer,
                "seed": args.seed,
                "updated": self.data.get("updated", utc_now()),
                "answers": {
                    uid: {
                        key: answer[key]
                        for key in (
                            "verdict", "confident", "reasons",
                            "note", "seconds", "revised"
                        )
                    }
                    for uid, answer in self.data["answers"].items()
                },
            }
        else:
            self.data = {
                "schema": SCHEMA,
                "reviewer": args.reviewer,
                "seed": args.seed,
                "updated": utc_now(),
                "answers": {},
            }

        self.position = next(
            (i for i, uid in enumerate(self.order)
             if uid not in self.data["answers"]),
            len(self.order),
        )
        self.visit = None
        self.baseline = None
        self.last_save_time = None
        self.image_token = None
        self.image_cache = None

    def enter(self, delta):
        if type(delta) is not int or delta not in (-1, 0, 1):
            raise ValueError("Некорректная навигация")
        self.position = max(
            0, min(len(self.order), self.position + delta)
        )
        self.visit = secrets.token_urlsafe(20)
        self.image_token = None
        self.image_cache = None

        if self.position == len(self.order):
            return {"done": True, "can_previous": self.position > 0}

        uid = self.order[self.position]
        answer = self.data["answers"].get(uid)
        self.baseline = substantive(answer)
        self.last_save_time = time.monotonic()
        self.image_token = secrets.token_urlsafe(20)
        # Deliberately omit UID, label, study, path, row number, etc.
        return {
            "done": False,
            "visit": self.visit,
            "image": self.image_token,
            "can_previous": self.position > 0,
            "answer": (
                {k: answer[k] for k in ("verdict", "confident", "reasons", "note")}
                if answer else None
            ),
        }

    def save(self, payload):
        if (
            self.position >= len(self.order)
            or payload.get("visit") != self.visit
        ):
            raise ValueError("Устаревшая сессия снимка")
        answer = payload.get("answer")
        validate_answer(answer)
        # Rebuild rather than persisting any client-supplied extra fields.
        answer = {
            "verdict": answer["verdict"],
            "confident": answer["confident"],
            "reasons": [r for r in REASONS if r in answer["reasons"]],
            "note": answer["note"],
        }
        uid = self.order[self.position]
        previous = self.data["answers"].get(uid)
        now = time.monotonic()
        elapsed = max(0.0, now - self.last_save_time)

        revised = bool(previous and previous["revised"])
        if previous and previous["verdict"] != answer["verdict"]:
            revised = True
        if self.baseline is not None and self.baseline != substantive(answer):
            revised = True

        answer["seconds"] = round(
            (previous["seconds"] if previous else 0.0) + elapsed, 3
        )
        answer["revised"] = revised
        updated = dict(self.data)
        updated["answers"] = dict(self.data["answers"])
        updated["answers"][uid] = answer
        updated["updated"] = utc_now()

        # Update in-memory state only after a successful atomic write.
        atomic_json(self.outfile, updated)
        self.data = updated
        self.last_save_time = now
        return {"ok": True}

    def image(self, token):
        if not self.image_token or token != self.image_token:
            return None
        if self.image_cache is None:
            uid = slot_uid(self.order[self.position])
            self.image_cache = render_dicom(self.records[uid]["path"])
        return self.image_cache


def handler_factory(app):
    page = HTML.replace("__BASE__", json.dumps(app.token)).replace(
        "__REASONS__", json.dumps(REASONS, ensure_ascii=False)
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "AxisReview"

        def log_message(self, format_string, *args):
            # Do not log image URLs, capability tokens or request bodies.
            pass

        def reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; img-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'none'"
            )
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, status, obj):
            self.reply(
                status,
                json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def valid_host(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return self.headers.get("Host") == expected

        def do_GET(self):
            if not self.valid_host():
                self.reply(403, b"Forbidden", "text/plain")
                return
            path = urlsplit(self.path).path
            if path in ("", "/"):
                self.reply(200, page, "text/html; charset=utf-8")
                return
            prefix = app.base + "/image/"
            if path.startswith(prefix):
                try:
                    image = app.image(path[len(prefix):])
                    if image is None:
                        self.reply(404, b"Not found", "text/plain")
                    else:
                        self.reply(200, image, "image/png")
                except Exception:
                    # Do not leak DICOM tags or paths in browser errors.
                    print(
                        "Не удалось декодировать текущее изображение. "
                        "Проверьте DICOM/поддержку transfer syntax.",
                        file=sys.stderr,
                    )
                    self.reply(422, b"Image unavailable", "text/plain")
                return
            self.reply(404, b"Not found", "text/plain")

        def do_POST(self):
            if (
                not self.valid_host()
                or not secrets.compare_digest(
                    self.headers.get("X-Axis-Review", ""), app.token)
                or self.headers.get("Content-Type", "").split(";")[0]
                != "application/json"
            ):
                self.json_reply(403, {"error": "Запрос запрещён"})
                return
            origin = self.headers.get("Origin")
            expected_origin = f"http://127.0.0.1:{self.server.server_port}"
            if origin is not None and origin != expected_origin:
                self.json_reply(403, {"error": "Запрос запрещён"})
                return

            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Некорректный размер запроса")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Ожидался объект")
                if path == app.base + "/enter":
                    result = app.enter(payload.get("delta"))
                elif path == app.base + "/save":
                    result = app.save(payload)
                else:
                    self.json_reply(404, {"error": "Не найдено"})
                    return
                self.json_reply(200, result)
            except (ValueError, TypeError, KeyError):
                self.json_reply(400, {"error": "Некорректный запрос"})
            except Exception:
                print(
                    "Ошибка обработки/сохранения. Проверьте доступ к каталогу "
                    "и свободное место.",
                    file=sys.stderr,
                )
                self.json_reply(500, {"error": "Ошибка сохранения"})

    return Handler


def run_review(records, args):
    if not args.reviewer or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.reviewer):
        raise ValueError("--reviewer: латинская буква, затем латиница/цифры/_/-")
    if not args.clean_pixels_confirmed:
        raise ValueError(
            "Сначала проверьте пиксели на отсутствие идентификаторов, старых "
            "меток и иных подсказок; затем укажите --clean-pixels-confirmed."
        )
    part = str(getattr(args, "part", "all"))
    if part != "all":
        from annotation_split import describe as describe_split, split_studies
        all_studies = [r["study"] for r in records.values()]
        print(describe_split(all_studies, part, overlap=AXIS_OVERLAP, salt="axis"))
        mine = split_studies(all_studies, part, overlap=AXIS_OVERLAP, salt="axis")
        records = {uid: r for uid, r in records.items() if r["study"] in mine}
        if not records:
            raise ValueError("В этой части не осталось снимков.")

    outfile = Path(args.out) / f"axis_review_{args.reviewer}_{args.seed}.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile = outfile.with_suffix(outfile.suffix + ".lock")
    try:
        lock_fd = os.open(str(lockfile), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"Есть блокировка {lockfile}. Возможно, этот файл уже открыт "
            "другим сервером. После аварии удалите блокировку вручную, "
            "убедившись, что прежний процесс завершён."
        )
    server = None
    try:
        with os.fdopen(lock_fd, "w", encoding="ascii") as f:
            f.write(str(os.getpid()))
        app = ReviewApp(records, args, outfile)
        server = HTTPServer(("127.0.0.1", args.port), handler_factory(app))
        url = f"http://127.0.0.1:{server.server_port}{app.base}/"
        print(f"Открыть: {url}")
        print(f"Сохранение: {outfile}")
        print("Используйте одну вкладку. Завершение сервера: Ctrl+C.")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен; подтверждённые сохранения на диске.")
    finally:
        if server is not None:
            server.server_close()
        lockfile.unlink(missing_ok=True)


# ------------------------------- Reporting -------------------------------

def binary_verdict(answer):
    value = answer["verdict"]
    if value == "unsure":
        return None
    return int(value == "deviated")


def cluster_key(records, uid):
    study = records[uid]["study"]
    # Missing study IDs must not merge unrelated images into one cluster.
    return ("study", study) if study else ("sop", uid)


def confusion(pairs):
    """Rows: first rater 0/1; columns: second rater 0/1."""
    table = np.zeros((2, 2), dtype=np.int64)
    for _, first, second in pairs:
        table[first, second] += 1
    return table


def kappa(table):
    n = float(table.sum())
    if n == 0:
        return float("nan")
    observed = float(np.trace(table)) / n
    expected = float(np.dot(table.sum(axis=1), table.sum(axis=0))) / (n * n)
    if abs(1.0 - expected) < 1e-14:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def bootstrap_ci(pairs, repeats, seed):
    if not pairs:
        return None, 0, 0
    grouped = {}
    for cluster, first, second in pairs:
        if cluster not in grouped:
            grouped[cluster] = np.zeros((2, 2), dtype=np.int64)
        grouped[cluster][first, second] += 1
    blocks = np.stack(list(grouped.values()))
    cluster_count = len(blocks)
    if cluster_count < 2:
        return None, 0, cluster_count

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indexes = rng.integers(0, cluster_count, size=cluster_count)
        value = kappa(blocks[indexes].sum(axis=0))
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return None, len(values), cluster_count
    low, high = np.percentile(values, [2.5, 97.5])
    return (float(low), float(high)), len(values), cluster_count


def exact_mcnemar(b, c):
    """Two-sided exact conditional binomial test; no scipy dependency."""
    n = b + c
    if n == 0:
        return 1.0
    limit = min(b, c)
    log_factorial_n = math.lgamma(n + 1)
    terms = (
        math.exp(
            log_factorial_n - math.lgamma(i + 1)
            - math.lgamma(n - i + 1) - n * math.log(2.0)
        )
        for i in range(limit + 1)
    )
    return min(1.0, 2.0 * math.fsum(terms))


def percent(numerator, denominator):
    return f"{100.0 * numerator / denominator:.1f}%" if denominator else "NA"


def number(value):
    return f"{value:.4f}" if math.isfinite(value) else "NA"


def print_metrics(pairs, args, full=True):
    table = confusion(pairs)
    n = int(table.sum())
    tn, fp, fn, tp = map(int, table.ravel())
    value = kappa(table)
    ci, valid, clusters = bootstrap_ci(
        pairs, args.bootstrap, args.bootstrap_seed
    )
    interval = f"[{ci[0]:.4f}; {ci[1]:.4f}]" if ci else "NA"
    print(f"  Бинарных пар: {n}; исследований/кластеров: {clusters}")
    print(f"  Cohen's κ = {number(value)}; 95% bootstrap ДИ = {interval}")
    if clusters >= 2:
        print(f"  Определённых bootstrap-реплик: {valid}/{args.bootstrap}")
        if valid < args.bootstrap:
            print(
                "  Внимание: вырожденные реплики с неопределённой κ исключены; "
                "ДИ следует интерпретировать осторожно."
            )
    else:
        print("  Для bootstrap ДИ нужно не менее двух кластеров.")
    print(f"  Простое согласие: {percent(tp + tn, n)}")
    print(
        "  Положительное согласие 2TP/(2TP+FP+FN): "
        + percent(2 * tp, 2 * tp + fp + fn)
    )
    if full:
        print(
            f"  Положительных: исходно {tp + fn}/{n}, "
            f"при пересмотре {tp + fp}/{n}"
        )
        print(
            f"  Макнемар, точный двусторонний: "
            f"0→1 = {fp}, 1→0 = {fn}, p = {exact_mcnemar(fp, fn):.6g}"
        )
        print(
            "  Изменение доли положительных: "
            + (f"{100 * (fp - fn) / n:+.1f} п.п." if n else "NA")
        )


def reason_distribution(title, selected):
    print(f"\n{title}: n = {len(selected)}")
    counts = Counter(
        reason
        for answer in selected
        for reason in set(answer["reasons"])
    )
    for reason in REASONS:
        count = counts[reason]
        print(f"  {reason}: {count} ({percent(count, len(selected))})")
    no_reasons = sum(not a["reasons"] for a in selected)
    print(f"  Без причины: {no_reasons} ({percent(no_reasons, len(selected))})")
    notes = Counter(a["note"].strip() for a in selected if a["note"].strip())
    if notes:
        print("  Тексты «другое»:")
        for note, count in notes.most_common():
            print(f"    {json.dumps(note, ensure_ascii=False)}: {count}")


def printable(value):
    """Keep each discrepancy on one TSV line."""
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def print_repeatability(repeats, answers):
    """Скрытые повторы: насколько разметчик воспроизводит сам себя."""
    pairs = [(answers[slot_uid(key)]["verdict"], answer["verdict"])
             for key, answer in repeats.items() if slot_uid(key) in answers]
    if not pairs:
        print("Скрытых повторов в файле нет (или на них нет ответов).")
        return
    same = sum(first == second for first, second in pairs)
    binary = [(a, b) for a, b in pairs if "unsure" not in (a, b)]
    same_binary = sum(a == b for a, b in binary)
    print(
        f"Скрытые повторы: {len(pairs)}; совпало вердиктов: {same}/{len(pairs)}"
        + (f"; среди бинарных пар {same_binary}/{len(binary)}" if binary else "")
    )
    if len(pairs) and same < len(pairs):
        print("  расхождения сам с собой: "
              + "; ".join(f"{a}→{b}" for a, b in pairs if a != b))
    print("  Внутренняя согласованность ограничивает внешнюю: разметчик не может "
          "согласиться с чужой меткой лучше, чем с собственной.")


def report_one(filename, data, records, args):
    all_answers = data["answers"]
    # Повторные показы считаются отдельно: они измеряют стабильность разметчика,
    # а не согласие с исходной меткой, и в основную таблицу попадать не должны.
    repeats = {key: a for key, a in all_answers.items()
               if key.endswith(REPEAT_SUFFIX) and slot_uid(key) in records}
    answers = {uid: a for uid, a in all_answers.items()
               if not uid.endswith(REPEAT_SUFFIX) and uid in records}
    unknown = len(all_answers) - len(answers) - len(repeats)
    total = len(records)
    missing = total - len(answers)
    unsure = sum(a["verdict"] == "unsure" for a in answers.values())
    revised = sum(a["revised"] for a in answers.values())

    print("\n" + "=" * 78)
    print(f"Файл: {filename}")
    print(f"Reviewer: {data['reviewer']}; seed: {data['seed']}")
    print(f"Ответов в выборке: {len(answers)}/{total}; без ответа: {missing}")
    if unknown:
        print(f"ВНИМАНИЕ: {unknown} ответов вне текущего индекса исключены.")
    print(
        f"«Не могу решить»: {unsure}; "
        f"{percent(unsure, len(answers))} от ответов; "
        f"{percent(unsure, total)} от всей выборки"
    )
    print(f"Изменённых ответов (revised): {revised}/{len(answers)}")
    print_repeatability(repeats, answers)

    table = np.zeros((2, 3), dtype=np.int64)
    columns = {"normal": 0, "deviated": 1, "unsure": 2}
    pairs = []
    groups = {"tp": [], "fp": [], "fn": [], "diff": [], "unsure": []}
    for uid, answer in answers.items():
        original = records[uid]["y"]
        table[original, columns[answer["verdict"]]] += 1
        new = binary_verdict(answer)
        if new is None:
            groups["unsure"].append(answer)
            continue
        pairs.append((cluster_key(records, uid), original, new))
        if original == 1 and new == 1:
            groups["tp"].append(answer)
        elif original == 0 and new == 1:
            groups["fp"].append(answer)
        elif original == 1 and new == 0:
            groups["fn"].append(answer)
        if original != new:
            groups["diff"].append(answer)

    print("\nИсходная метка × новый вердикт:")
    print("                 норма   отклонена   не могу решить   без ответа")
    for original in (0, 1):
        absent = sum(
            records[uid]["y"] == original and uid not in answers
            for uid in records
        )
        print(
            f"  исходно {original}  "
            f"{table[original, 0]:9d} {table[original, 1]:11d} "
            f"{table[original, 2]:16d} {absent:12d}"
        )
    print("\nМетрики: только пары с двумя бинарными ответами.")
    print_metrics(pairs, args)

    print("\nРасхождения и неопределённые ответы (TSV):")
    print("sop_uid\tисходная\tновый вердикт\tуверен\tпричины\tтекст\trevised")
    found = False
    for uid in sorted(answers):
        answer = answers[uid]
        original = records[uid]["y"]
        new = binary_verdict(answer)
        if new is not None and new == original:
            continue
        found = True
        print("\t".join(map(printable, [
            uid, original, LABELS[answer["verdict"]],
            "да" if answer["confident"] else "нет",
            "; ".join(answer["reasons"]), answer["note"],
            "да" if answer["revised"] else "нет",
        ])))
    if not found:
        print("  Нет.")

    print(
        "\nTP/FP/FN ниже — только относительно исходной метки, "
        "не доказанная клиническая истина."
    )
    print("Причины: множественный выбор; проценты могут суммироваться более 100%.")
    reason_distribution("TP: исходно 1, пересмотр 1", groups["tp"])
    reason_distribution("FP: исходно 0, пересмотр 1", groups["fp"])
    reason_distribution("FN: исходно 1, пересмотр 0", groups["fn"])
    reason_distribution("Бинарные расхождения: FP + FN", groups["diff"])
    reason_distribution("Неопределённые ответы", groups["unsure"])


def expand_report_files(patterns):
    paths = []
    seen = set()
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"Не найдены файлы: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def run_report(records, args):
    files = expand_report_files(args.report)
    reviews = [(path, load_review(path)) for path in files]

    print("95% ДИ: percentile bootstrap с повторной выборкой исследований.")
    print("Исходный CSV должен быть тем же зафиксированным индексом, что при пересмотре.")
    print("Пропуски и unsure не превращаются в отрицательные ответы.")
    print("κ и согласие не являются доказательством потолка F1.")
    if any(not item["study"] for item in records.values()):
        print(
            "ВНИМАНИЕ: для пустых study каждый SOP принят за отдельный кластер."
        )

    for filename, data in reviews:
        report_one(filename, data, records, args)

    if len(reviews) >= 2:
        print("\n" + "=" * 78)
        print("ПОПАРНОЕ СОГЛАСИЕ МЕЖДУ ПЕРЕСМОТРАМИ")
        for (path_a, data_a), (path_b, data_b) in itertools.combinations(reviews, 2):
            common = sorted(
                set(data_a["answers"]) & set(data_b["answers"]) & set(records)
            )
            pairs = []
            unsure_pairs = 0
            for uid in common:
                first = binary_verdict(data_a["answers"][uid])
                second = binary_verdict(data_b["answers"][uid])
                if first is None or second is None:
                    unsure_pairs += 1
                else:
                    pairs.append((cluster_key(records, uid), first, second))
            print(f"\n{path_a.name} × {path_b.name}")
            print(
                f"  Совместно отвеченных: {len(common)}; "
                f"хотя бы один unsure: {unsure_pairs} "
                f"({percent(unsure_pairs, len(common))})"
            )
            print_metrics(pairs, args, full=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Слепой пересмотр метки оси DXA и расчёт согласованности."
    )
    parser.add_argument("--index", default="data/index/images.csv")
    parser.add_argument("--reviewer")
    parser.add_argument("--out", default="data/annotations")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--part", default="all", choices=("1", "2", "all"),
                        help="часть работы: 1 или 2 (делится по исследованиям, "
                             "25 общих — по ним считается согласие двух "
                             "разметчиков), all — всё")
    parser.add_argument("--duplicates", type=int, default=10,
                        help="сколько снимков показать повторно (скрытая проверка "
                             "повторяемости; 0 — выключить)")
    parser.add_argument("--port", type=int, default=8768,
                        help="Локальный порт; 0 — свободный порт автоматически")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--clean-pixels-confirmed", action="store_true",
        help="Подтверждаю отсутствие идентификаторов/подсказок в пикселях"
    )
    parser.add_argument(
        "--report", nargs="+", metavar="FILE_OR_GLOB",
        help="Печатать отчёт без сервера и браузера"
    )
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Число bootstrap-реплик, по умолчанию 10000")
    parser.add_argument("--bootstrap-seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap должен быть не менее 100")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed должен быть неотрицательным")
    if not 0 <= args.port <= 65535:
        parser.error("Некорректный --port")
    return args


def main():
    args = parse_args()
    try:
        records = read_index(args.index)
        if args.report:
            run_report(records, args)
        else:
            run_review(records, args)
    except (OSError, ValueError, ImportError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
