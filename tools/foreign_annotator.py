# tools/foreign_annotator.py
# Запуск:
# python tools/foreign_annotator.py \
#   --index data/index/images.csv --root . --annotator ivan

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
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pydicom
from PIL import Image


SCHEMA = "dxa-foreign-boxes/1"

OBJECT_CLASSES = (
    "молния",
    "пуговица/кнопка",
    "монета",
    "застёжка/пряжка",
    "украшение/цепочка",
    "провод/трубка",
    "другое",
)

HARD_CLASSES = (
    "яркая линия у края кадра",
    "край кости",
    "подпись/маркер аппарата",
    "артефакт",
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(filename: Path, value):
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
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
        temporary = None
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def binary_label(value):
    """Метка из CSV: pandas пишет «0.0»/«1.0», пустая клетка означает «не размечено»."""
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number in (0.0, 1.0) else None


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_dicom(filename: Path, expected_rows: int, expected_cols: int):
    try:
        ds = pydicom.dcmread(filename)
    except Exception:
        ds = pydicom.dcmread(filename, force=True)

    pixels = np.asarray(ds.pixel_array, dtype=np.float64)
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]

    if pixels.ndim != 2:
        raise ValueError("Ожидалось однокадровое изображение.")

    actual_rows, actual_cols = map(int, pixels.shape)
    if actual_rows != expected_rows or actual_cols != expected_cols:
        raise ValueError(
            f"Размер изображения {actual_rows}x{actual_cols} "
            f"не совпадает с CSV {expected_rows}x{expected_cols}."
        )

    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        bits = int(getattr(ds, "BitsStored", 16))
        if not 1 <= bits <= 64:
            raise ValueError("Недопустимое BitsStored.")
        pixels = (2**bits - 1) - pixels

    finite = np.isfinite(pixels)
    if not finite.any():
        raise ValueError("Изображение не содержит конечных значений.")

    low = float(pixels[finite].min())
    high = float(pixels[finite].max())
    pixels = np.nan_to_num(pixels, nan=low, posinf=high, neginf=low)

    if high > low:
        pixels = np.clip((pixels - low) / (high - low) * 255, 0, 255)
    else:
        pixels = np.zeros_like(pixels)

    image = Image.fromarray(pixels.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def safe_path(root: Path, relative: str):
    rel = Path(relative)
    result = (root / rel).resolve()
    if rel.is_absolute() or not result.is_relative_to(root):
        raise ValueError("Файл находится вне корневой папки.")
    return result


class Application:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.filename = (
            Path(args.out) / f"foreign_boxes_{args.annotator}.json"
        )
        self.items = {}
        self.order = []
        self.pngs = {}

        root = Path(args.root).resolve()

        with open(args.index, newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {
                "sop_uid",
                "study",
                "path",
                "rows",
                "cols",
                "y_foreign",
                "region",
            }
            fields = set(reader.fieldnames or [])
            missing = required - fields
            if missing:
                raise ValueError(
                    "В CSV отсутствуют столбцы: " + ", ".join(sorted(missing))
                )

            rows = []
            for row in reader:
                label = binary_label(row["y_foreign"])
                if label is None:
                    continue
                if not row["sop_uid"].strip():
                    raise ValueError("В CSV найден пустой sop_uid.")
                rows.append(row)

        # Стабильный порядок, не зависящий от порядка строк CSV.
        rows.sort(
            key=lambda row: (
                hashlib.sha1(
                    (row["study"].strip() + "\0" + row["sop_uid"].strip())
                    .encode("utf-8")
                ).hexdigest(),
                row["study"].strip(),
                row["sop_uid"].strip(),
            )
        )

        for row in rows:
            uid = row["sop_uid"].strip()
            if uid in self.items:
                continue

            try:
                expected_rows = int(row["rows"])
                expected_cols = int(row["cols"])
                if expected_rows < 1 or expected_cols < 1:
                    raise ValueError("Некорректный размер изображения.")

                filename = safe_path(root, row["path"])
                png = read_dicom(filename, expected_rows, expected_cols)
                error = None
            except Exception as exc:
                png = None
                error = (
                    "Не удалось прочитать изображение или размер не совпадает "
                    f"с CSV ({type(exc).__name__})."
                )
                print(
                    f"Ошибка изображения {uid}: {type(exc).__name__}",
                    flush=True,
                )

            self.items[uid] = {
                "image_id": uid,
                "study": row["study"].strip(),
                "rows": int(row["rows"]),
                "cols": int(row["cols"]),
                "y_foreign": binary_label(row["y_foreign"]),
                "region": row["region"].strip(),
                "error": error,
            }
            self.order.append(uid)
            if png is not None:
                self.pngs[uid] = png

        self.data = {
            "schema": SCHEMA,
            "annotator": args.annotator,
            "updated": now_iso(),
            "images": {},
        }

        if self.filename.exists():
            with self.filename.open(encoding="utf-8") as stream:
                loaded = json.load(stream)

            if (
                not isinstance(loaded, dict)
                or loaded.get("schema") != SCHEMA
                or loaded.get("annotator") != args.annotator
                or not isinstance(loaded.get("images"), dict)
            ):
                raise ValueError("Файл разметки имеет неподходящий формат.")

            self.data = loaded
            self.data["schema"] = SCHEMA
            self.data["annotator"] = args.annotator

            for uid in list(self.data["images"]):
                if uid not in self.items:
                    del self.data["images"][uid]

            for uid, annotation in list(self.data["images"].items()):
                self.data["images"][uid] = self.validate(uid, annotation)

        self.filename.parent.mkdir(parents=True, exist_ok=True)
        print(f"Изображений для разметки: {len(self.order)}", flush=True)

    def blank(self, uid):
        item = self.items[uid]
        return {
            "state": "done",
            "y_foreign": item["y_foreign"],
            "boxes": [],
            "comment": "",
            "seconds": 0.0,
            "updated": now_iso(),
        }

    def validate(self, uid, raw):
        if uid not in self.items:
            raise ValueError("Неизвестный sop_uid.")
        if not isinstance(raw, dict):
            raise ValueError("Разметка должна быть объектом.")

        item = self.items[uid]
        result = self.blank(uid)

        state = raw.get("state", "done")
        if state not in ("done", "skipped", "empty_confirmed"):
            raise ValueError("Недопустимое состояние снимка.")

        y_foreign = raw.get("y_foreign", item["y_foreign"])
        if y_foreign not in (0, 1):
            raise ValueError("y_foreign должен быть 0 или 1.")
        if y_foreign != item["y_foreign"]:
            raise ValueError("y_foreign не совпадает с CSV.")

        boxes = raw.get("boxes", [])
        if not isinstance(boxes, list) or len(boxes) > 1000:
            raise ValueError("Некорректный список рамок.")

        cleaned_boxes = []
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError("Рамка должна быть объектом.")

            allowed = {"x", "y", "w", "h", "kind", "class", "sure"}
            if set(box) - allowed:
                raise ValueError("Неизвестное поле рамки.")

            for key in ("x", "y", "w", "h"):
                if not finite_number(box.get(key)):
                    raise ValueError("Координаты рамки должны быть числами.")

            x = int(round(box["x"]))
            y = int(round(box["y"]))
            w = int(round(box["w"]))
            h = int(round(box["h"]))

            if w < 1 or h < 1:
                raise ValueError("Размер рамки должен быть положительным.")
            if x < 0 or y < 0 or x + w > item["cols"] or y + h > item["rows"]:
                raise ValueError("Рамка выходит за пределы изображения.")

            kind = box.get("kind")
            if kind not in ("object", "hard_negative"):
                raise ValueError("Недопустимый тип рамки.")

            class_name = box.get("class")
            valid_classes = OBJECT_CLASSES if kind == "object" else HARD_CLASSES
            if class_name not in valid_classes:
                raise ValueError("Недопустимый класс рамки.")

            sure = box.get("sure", True)
            if type(sure) is not bool:
                raise ValueError("sure должен быть логическим значением.")

            cleaned_boxes.append(
                {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "kind": kind,
                    "class": class_name,
                    "sure": sure,
                }
            )

        comment = raw.get("comment", "")
        if not isinstance(comment, str) or len(comment) > 20000:
            raise ValueError("Комментарий должен быть строкой до 20000 символов.")

        seconds = raw.get("seconds", 0.0)
        if not finite_number(seconds) or seconds < 0:
            raise ValueError("Некорректное время работы.")

        if state == "empty_confirmed" and boxes:
            raise ValueError("empty_confirmed не может содержать рамки.")

        if state == "skipped" and boxes:
            raise ValueError("Пропущенный снимок не должен содержать рамки.")

        result.update(
            state=state,
            y_foreign=y_foreign,
            boxes=cleaned_boxes,
            comment=comment,
            seconds=round(float(seconds), 3),
            updated=now_iso(),
        )
        return result

    def state(self):
        with self.lock:
            images = []
            for uid in self.order:
                item = self.items[uid]
                annotation = self.data["images"].get(uid)
                images.append(
                    {
                        **item,
                        "annotated": annotation is not None,
                        "state": annotation.get("state") if annotation else None,
                        "complete": annotation is not None
                        and annotation.get("state")
                        in ("done", "skipped", "empty_confirmed"),
                    }
                )

            return {
                "schema": SCHEMA,
                "annotator": self.args.annotator,
                "images": images,
                "first_run": not bool(self.data["images"]),
            }

    def save(self, uid, raw):
        with self.lock:
            cleaned = self.validate(uid, raw)
            previous = self.data["images"].get(uid)
            if previous:
                cleaned["seconds"] = max(
                    cleaned["seconds"],
                    float(previous.get("seconds", 0)),
                )

            updated = dict(self.data)
            updated["updated"] = now_iso()
            updated["images"] = {
                **self.data["images"],
                uid: cleaned,
            }
            atomic_write(self.filename, updated)
            self.data = updated
            return cleaned


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DXA — посторонние предметы</title>
<style>
:root{color-scheme:dark;font:14px system-ui,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#111923;color:#e9f0f7}
button,select,input,textarea{font:inherit}
button,select{background:#26384d;color:#fff;border:1px solid #58708a;border-radius:5px;padding:7px 10px}
button{cursor:pointer}
button:hover:not(:disabled){background:#36516d}
button:disabled{opacity:.45;cursor:default}
input,textarea{background:#172433;color:#fff;border:1px solid #526980;border-radius:4px;padding:6px}
header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#1d2b3b}
#save{margin-left:auto;color:#9fe4b5}
#error{background:#713c35;padding:8px 14px;white-space:pre-wrap}
nav{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:8px 12px;background:#182432}
main{height:calc(100vh - 100px);display:grid;grid-template-columns:minmax(400px,1fr) 390px}
#left{display:flex;flex-direction:column;min-width:0;min-height:0}
#tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:8px}
#stage{position:relative;flex:1;overflow:hidden;background:#05080c;touch-action:none}
canvas{position:absolute;inset:0;width:100%;height:100%}
#overlay{cursor:crosshair}
#message{position:absolute;left:20px;top:20px;color:#ffd49d}
#coords{padding:7px 12px;color:#abc0d3}
aside{overflow:auto;padding:12px;background:#1a2635;border-left:1px solid #3a4b60}
h3{margin:8px 0}
.panel{border-top:1px solid #3b4d62;padding:10px 0}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:7px 0}
#mode button.active,#classes button.active{background:#17628d;border-color:#7bd4ff}
#boxes{display:flex;flex-direction:column;gap:7px}
.boxrow{background:#26384b;border-radius:5px;padding:8px}
.boxrow.selected{outline:2px solid #ffd369}
.boxrow button{padding:4px 7px}
.small{font-size:12px;color:#b9c9d8;line-height:1.45}
textarea{width:100%;min-height:65px;resize:vertical}
input[type=range]{width:110px;padding:0}
@media(max-width:800px){main{grid-template-columns:1fr;height:auto}#left{height:65vh}}
</style>
</head>
<body>
<header>
<strong>DXA · посторонние предметы</strong>
<span id="who"></span>
<span id="progress"></span>
<span id="timer"></span>
<span id="save">Загрузка…</span>
<button id="retry" hidden>Повторить сохранение</button>
</header>
<div id="error" hidden></div>
<nav>
<button id="prev">P · Предыдущий</button>
<button id="next">N · Следующий</button>
<button id="skip">S · Пропустить</button>
<button id="empty">Enter · Предметов нет</button>
<a href="/api/export" download>Скачать JSON</a>
</nav>
<main>
<section id="left">
<div id="tools">
<button id="fit">Вписать</button>
<button id="zoom3">×3</button>
<label>Яркость <input id="brightness" type="range" min="30" max="250" value="100"></label>
<label>Контраст <input id="contrast" type="range" min="30" max="350" value="100"></label>
<button id="reset">Сброс</button>
</div>
<div id="stage">
<canvas id="image"></canvas>
<canvas id="overlay"></canvas>
<div id="message"></div>
</div>
<div id="coords">x: — · y: —</div>
</section>
<aside>
<div id="info"></div>

<div class="panel">
<strong>Режим рамки</strong>
<div id="mode" class="row">
<button data-mode="object" class="active">Предмет</button>
<button data-mode="hard_negative">H · Ловушка</button>
</div>
<div class="small">
H переключает режим hard negative. Рамки-ловушки нужно ставить и на
отрицательных снимках: линии, края костей, подписи, маркеры и артефакты.
</div>
</div>

<div class="panel">
<strong>Класс</strong>
<div id="classes" class="row"></div>
<div class="small">1–7 — классы предметов. Для ловушек выбирается отдельный подтип.</div>
</div>

<div class="panel">
<button id="sure">U · Не уверен</button>
<span id="sureText">Новые рамки: уверенные</span>
</div>

<div class="panel">
<h3>Рамки</h3>
<div id="boxes"></div>
</div>

<div class="panel">
<label>Комментарий</label>
<textarea id="comment" maxlength="20000"></textarea>
</div>

<div class="panel small">
ЛКМ перетащить — новая рамка.<br>
Клик по рамке — выбрать и перетащить.<br>
D — удалить последнюю, Delete — удалить выбранную.<br>
Колесо — зум, ПКМ или Space + мышь — панорамирование.<br>
U помечает выбранную рамку как неуверенную; если рамка не выбрана,
U включается для следующей рамки.
</div>
</aside>
</main>

<script>
"use strict";

const $=id=>document.getElementById(id);
const OBJECTS=[
"молния","пуговица/кнопка","монета","застёжка/пряжка",
"украшение/цепочка","провод/трубка","другое"];
const HARD=[
"яркая линия у края кадра","край кости",
"подпись/маркер аппарата","артефакт"];

let state=null,items=[],current=null,ann=null,index=-1,imageObj=null;
let mode="object",className=OBJECTS[0],sure=true,selected=-1;
let zoom=1,ox=0,oy=0,W=1,H=1,gesture=null,space=false;
let last=performance.now(),pending=false,saveOK=true,saveQueue=Promise.resolve();

function copy(x){return JSON.parse(JSON.stringify(x))}
function api(url,opt={}){
 return fetch(url,{cache:"no-store",...opt}).then(async r=>{
  if(!r.ok){let j={};try{j=await r.json()}catch(e){}
   throw Error(j.error||("Ошибка сервера: "+r.status))}
  return r.json()
 })
}
function showError(text=""){$("error").textContent=text;$("error").hidden=!text}
function active(){return !!ann&&!!current}
function account(){
 if(!active())return;
 let now=performance.now();
 ann.seconds=(ann.seconds||0)+Math.max(0,(now-last)/1000);
 last=now;
}
function status(){
 $("save").textContent=pending?"сохранение…":saveOK?"сохранено ✓":"ошибка сохранения";
 $("save").style.color=saveOK?"#9fe4b5":"#ffaaa0";
 $("retry").hidden=saveOK||pending;
}
function enqueue(){
 if(!active())return;
 account();
 let snapshot=copy(ann);
 pending++;status();
 saveQueue=saveQueue.then(()=>api(
  "/api/annotation/"+encodeURIComponent(current.image_id),{
   method:"PUT",headers:{"Content-Type":"application/json"},
   body:JSON.stringify(snapshot)
  }).then(saved=>{
   saveOK=true;ann=saved;current.annotated=true;showError("");
  }).catch(e=>{
   saveOK=false;showError(e.message+
    " Изменения остаются в браузере. Нажмите «Повторить сохранение».")
  }).finally(()=>{pending--;status()}));
}
function mutate(fn){
 if(!active())return;
 account();fn();ann.updated=new Date().toISOString();enqueue();render();
}
function native(p){return{x:(p.x-ox)/zoom,y:(p.y-oy)/zoom}}
function screen(p){return{x:p.x*zoom+ox,y:p.y*zoom+oy}}
function clampBox(x,y,w,h){
 x=Math.max(0,Math.min(current.cols-1,x));
 y=Math.max(0,Math.min(current.rows-1,y));
 w=Math.max(1,Math.min(current.cols-x,w));
 h=Math.max(1,Math.min(current.rows-y,h));
 return {x:Math.round(x),y:Math.round(y),w:Math.round(w),h:Math.round(h)}
}
function draw(){
 let b=$("image").getContext("2d"),c=$("overlay").getContext("2d");
 b.clearRect(0,0,W,H);c.clearRect(0,0,W,H);
 if(imageObj){
  b.imageSmoothingEnabled=false;
  b.drawImage(imageObj,ox,oy,imageObj.naturalWidth*zoom,imageObj.naturalHeight*zoom)
 }
 if(!ann)return;
 ann.boxes.forEach((q,i)=>{
  let p=screen(q),sel=i===selected;
  c.strokeStyle=sel?"#fff27b":q.kind==="object"?"#ffbf55":"#ff78a8";
  c.lineWidth=sel?3:2;c.strokeRect(p.x,p.y,q.w*zoom,q.h*zoom);
  c.fillStyle=c.strokeStyle;c.font="bold 12px system-ui";
  c.fillText((i+1)+" "+q.class+(q.sure?"":" · U"),p.x+4,p.y+15)
 });
}
function resize(){
 let r=$("stage").getBoundingClientRect(),d=devicePixelRatio||1;
 W=r.width;H=r.height;
 for(let x of [$("image"),$("overlay")]){
  x.width=Math.round(W*d);x.height=Math.round(H*d);
  x.getContext("2d").setTransform(d,0,0,d,0,0)
 }
 draw()
}
function center(scale){
 zoom=scale;ox=(W-(current?.cols||0)*zoom)/2;
 oy=(H-(current?.rows||0)*zoom)/2;draw()
}
function fit(){
 if(!current)return;
 center(Math.max(1,Math.min(10,(W-20)/current.cols,(H-20)/current.rows)))
}
function selectedBoxAt(p){
 if(!ann)return -1;
 return ann.boxes.map((q,i)=>{
  let s=screen(q);
  return {i,d:(p.x>=s.x&&p.x<=s.x+q.w*zoom&&
   p.y>=s.y&&p.y<=s.y+q.h*zoom)?0:Infinity}
 }).sort((a,b)=>a.d-b.d)[0]?.i??-1
}
function renderClasses(){
 let box=$("classes");box.replaceChildren();
 let list=mode==="object"?OBJECTS:HARD;
 list.forEach((x,i)=>{
  let b=document.createElement("button");
  b.textContent=`${i+1} · ${x}`;
  b.className=x===className?"active":"";
  b.onclick=()=>{className=x;renderClasses()};
  box.append(b)
 })
}
function renderBoxes(){
 let box=$("boxes");box.replaceChildren();
 if(!ann||!ann.boxes.length){box.textContent="Пока нет рамок.";return}
 ann.boxes.forEach((q,i)=>{
  let d=document.createElement("div");d.className="boxrow"+(i===selected?" selected":"");
  d.innerHTML=`<b>${i+1}. ${q.kind==="object"?"предмет":"ловушка"}</b>
   · ${q.class}<br><span class="small">
   x=${q.x}, y=${q.y}, ${q.w}×${q.h}${q.sure?"":" · НЕ УВЕРЕН"}
   </span>`;
  let row=document.createElement("div");row.className="row";
  let select=document.createElement("button");select.textContent="Выбрать";
  select.onclick=()=>{selected=i;render()};
  let del=document.createElement("button");del.textContent="Удалить";
  del.onclick=()=>mutate(()=>{ann.boxes.splice(i,1);selected=-1});
  let u=document.createElement("button");u.textContent=q.sure?"U":"Уверенно";
  u.onclick=()=>mutate(()=>{q.sure=!q.sure});
  row.append(select,u,del);d.append(row);box.append(d)
 })
}
function render(){
 $("who").textContent=state?"Разметчик: "+state.annotator:"";
 let done=items.filter(x=>x.annotated).length;
 $("progress").textContent=`Размечено ${done} из ${items.length}`;
 $("info").innerHTML=current?
  `<b>${current.y_foreign?"Положительный":"Отрицательный"} снимок</b>
   · ${index+1}/${items.length}<br>
   Размер: ${current.cols}×${current.rows}`:"";
 $("sureText").textContent=sure?"Новые рамки: уверенные":"Новые рамки: НЕ УВЕРЕН";
 $("sure").classList.toggle("active",!sure);
 $("mode").querySelectorAll("button").forEach(b=>
  b.classList.toggle("active",b.dataset.mode===mode));
 renderClasses();renderBoxes();draw();status()
}
async function load(item){
 if(!item)return;
 account();enqueue();await saveQueue;
 current=item;index=items.indexOf(item);selected=-1;imageObj=null;
 $("message").textContent="Загрузка…";render();
 try{
  ann=await api("/api/annotation/"+encodeURIComponent(item.image_id));
  if(!item.error){
   imageObj=new Image();
   await new Promise((ok,no)=>{
    imageObj.onload=ok;imageObj.onerror=no;
    imageObj.src="/api/image/"+encodeURIComponent(item.image_id)+".png"
   })
  }
  $("message").textContent=item.error||"";
 }catch(e){$("message").textContent=e.message}
 last=performance.now();render();resize();fit()
}
function go(delta){
 let i=index+delta;
 if(i>=0&&i<items.length)load(items[i])
}
function pointer(e){
 let r=$("overlay").getBoundingClientRect();
 return{x:e.clientX-r.left,y:e.clientY-r.top}
}
$("overlay").oncontextmenu=e=>e.preventDefault();
$("overlay").onpointerdown=e=>{
 let p=pointer(e);
 if(e.button===2||(e.button===0&&space)){
  gesture={type:"pan",p,ox,oy,id:e.pointerId};
 }else if(e.button===0&&active()&&!current.error){
  let hit=selectedBoxAt(p);
  if(hit>=0){
   selected=hit;
   gesture={type:"move",p,old:copy(ann.boxes[hit]),id:e.pointerId}
  }else{
   let n=native(p);
   gesture={type:"new",p,n,id:e.pointerId}
  }
 }
 if(gesture)$("overlay").setPointerCapture(e.pointerId);
 e.preventDefault();render()
};
$("overlay").onpointermove=e=>{
 let p=pointer(e),q=native(p);
 $("coords").textContent=`x: ${q.x.toFixed(1)} · y: ${q.y.toFixed(1)}`;
 if(!gesture){draw();return}
 if(gesture.type==="pan"){
  ox=gesture.ox+p.x-gesture.p.x;oy=gesture.oy+p.y-gesture.p.y
 }else if(gesture.type==="move"){
  let old=gesture.old,n=native(p);
  ann.boxes[selected]=clampBox(n.x-(gesture.p.x- screen({x:old.x,y:old.y}).x)/zoom,
   n.y-(gesture.p.y- screen({x:old.x,y:old.y}).y)/zoom,old.w,old.h)
 }else if(gesture.type==="new"){
  let x=Math.min(gesture.n.x,q.x),y=Math.min(gesture.n.y,q.y);
  let w=Math.abs(q.x-gesture.n.x),h=Math.abs(q.y-gesture.n.y);
  draw();
  let s=screen(clampBox(x,y,w,h)),c=$("overlay").getContext("2d");
  c.strokeStyle="#fff";c.setLineDash([5,4]);c.strokeRect(s.x,s.y,s.w*zoom,s.h*zoom)
 }
 draw()
};
function end(e){
 if(!gesture)return;
 let g=gesture;gesture=null;
 if(g.type==="move")enqueue();
 if(g.type==="new"){
  let p=pointer(e),n=native(p),x=Math.min(g.n.x,n.x),y=Math.min(g.n.y,n.y);
  let q=clampBox(x,y,Math.abs(n.x-g.n.x),Math.abs(n.y-g.n.y));
  if(q.w>=2&&q.h>=2)mutate(()=>{ann.boxes.push({
   ...q,kind:mode,class:className,sure});selected=ann.boxes.length-1
  })
 }
 render()
}
$("overlay").onpointerup=end;$("overlay").onpointercancel=end;
$("overlay").onwheel=e=>{
 e.preventDefault();if(!imageObj)return;
 let p=pointer(e),n=native(p);
 zoom=Math.max(1,Math.min(12,zoom*Math.exp(-e.deltaY*.0015)));
 ox=p.x-n.x*zoom;oy=p.y-n.y*zoom;draw()
};
$("prev").onclick=()=>go(-1);$("next").onclick=()=>go(1);
$("fit").onclick=fit;$("zoom3").onclick=()=>center(3);
$("retry").onclick=enqueue;
$("sure").onclick=()=>{sure=!sure;render()};
$("skip").onclick=()=>{
 if(!active())return;
 mutate(()=>{ann.state="skipped";ann.boxes=[];selected=-1})
};
$("empty").onclick=()=>{
 if(!active()||current.y_foreign)return;
 if(confirm("Подтвердить, что предметов нет?"))
  mutate(()=>{ann.state="empty_confirmed";ann.boxes=[];selected=-1})
};
$("comment").oninput=()=>mutate(()=>{ann.comment=$("comment").value});
$("brightness").oninput=()=>{$("image").style.filter=
 `brightness(${$("brightness").value}%) contrast(${$("contrast").value}%)`};
$("contrast").oninput=$("brightness").oninput;
$("reset").onclick=()=>{
 $("brightness").value=100;$("contrast").value=100;
 $("image").style.filter="none"
};
document.querySelectorAll("[data-mode]").forEach(b=>b.onclick=()=>{
 mode=b.dataset.mode;className=mode==="object"?OBJECTS[0]:HARD[0];render()
});
document.onkeydown=e=>{
 if(e.target.matches("input,textarea,select"))return;
 if(e.code==="Space"){space=true;e.preventDefault();return}
 if(e.key==="Delete"&&selected>=0)
  mutate(()=>{ann.boxes.splice(selected,1);selected=-1});
 if(e.code==="KeyD"&&ann?.boxes.length)
  mutate(()=>{ann.boxes.pop();selected=-1});
 if(e.code==="KeyU"){
  if(selected>=0)mutate(()=>{ann.boxes[selected].sure=!ann.boxes[selected].sure});
  else{sure=!sure;render()}
 }
 if(e.code==="KeyH"){mode=mode==="hard_negative"?"object":"hard_negative";
  className=mode==="object"?OBJECTS[0]:HARD[0];render()}
 if(e.code==="KeyN"||e.key==="ArrowRight")go(1);
 if(e.code==="KeyP"||e.key==="ArrowLeft")go(-1);
 if(e.code==="Enter"&&!current?.y_foreign)$("empty").click();
 if(e.code==="KeyS")$("skip").click();
 if(/^Digit[1-7]$/.test(e.code)){
  let list=mode==="object"?OBJECTS:HARD,i=Number(e.code.slice(-1))-1;
  if(i<list.length){className=list[i];render()}
 }
};
document.onkeyup=e=>{if(e.code==="Space")space=false};
window.onresize=resize;
window.onblur=()=>{account();enqueue()};
setInterval(()=>{if(active())enqueue()},15000);

async function boot(){
 try{
  state=await api("/api/state");items=state.images;
  render();resize();await load(items.find(x=>!x.annotated)||items[0])
 }catch(e){showError(e.message)}
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
    server_version = "ForeignAnnotator/1"

    def log_message(self, fmt, *args):
        pass

    def send_bytes(self, status, body, content_type, download=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if download:
            name = (
                f"foreign_boxes_{self.server.app.args.annotator}.json"
            )
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{name}"',
            )
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, value, download=False):
        body = json.dumps(
            value, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        self.send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            download,
        )

    def fail(self, status, message):
        self.send_json(status, {"error": message})

    def uid_from_route(self, route, prefix, suffix=""):
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
                self.send_bytes(
                    200,
                    HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return

            if route == "/api/state":
                self.send_json(200, app.state())
                return

            if route == "/api/export":
                with app.lock:
                    self.send_json(200, app.data, download=True)
                return

            if route.startswith("/api/image/"):
                uid = self.uid_from_route(
                    route, "/api/image/", ".png"
                )
                if uid is None:
                    self.fail(404, "Неизвестный идентификатор изображения.")
                elif uid not in app.pngs:
                    self.fail(422, app.items[uid]["error"])
                else:
                    self.send_bytes(200, app.pngs[uid], "image/png")
                return

            if route.startswith("/api/annotation/"):
                uid = self.uid_from_route(
                    route, "/api/annotation/"
                )
                if uid is None:
                    self.fail(404, "Неизвестный идентификатор изображения.")
                else:
                    with app.lock:
                        self.send_json(
                            200,
                            app.data["images"].get(uid, app.blank(uid)),
                        )
                return

            self.fail(404, "Страница не найдена.")
        except Exception:
            self.fail(500, "Внутренняя ошибка сервера.")

    def do_PUT(self):
        app = self.server.app
        route = urlsplit(self.path).path

        if not route.startswith("/api/annotation/"):
            self.fail(404, "Страница не найдена.")
            return

        uid = self.uid_from_route(route, "/api/annotation/")
        if uid is None:
            self.fail(404, "Неизвестный идентификатор изображения.")
            return

        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Ожидается Content-Type: application/json.")

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Некорректный размер запроса.")

            self.connection.settimeout(30)
            raw_bytes = self.rfile.read(length)
            if len(raw_bytes) != length:
                raise ValueError("Запрос получен не полностью.")

            raw = json.loads(raw_bytes.decode("utf-8"))
            cleaned = app.save(uid, raw)
            self.send_json(200, cleaned)

        except ValueError as exc:
            self.fail(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.fail(500, "Внутренняя ошибка сохранения.")


def main():
    parser = argparse.ArgumentParser(
        description="Разметка рамок посторонних предметов на DXA."
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--out", default="data/annotations")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.annotator):
        parser.error(
            "--annotator: разрешены латинские буквы, цифры, _ и -."
        )

    if not 1 <= args.port <= 65535:
        parser.error("--port должен быть от 1 до 65535.")

    if args.reset:
        filename = (
            Path(args.out)
            / f"foreign_boxes_{args.annotator}.json"
        )
        if filename.exists():
            backup = filename.with_suffix(
                f".bak-{datetime.now():%Y%m%d-%H%M%S}.json"
            )
            filename.rename(backup)
            print(f"Старая разметка сохранена в {backup}")

    try:
        app = Application(args)
        server = Server((args.host, args.port), app)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Ошибка запуска: {exc}\n")

    url = f"http://{args.host}:{args.port}"
    print(f"\nАдрес: {url}", flush=True)
    print("Остановка: Ctrl+C.", flush=True)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
