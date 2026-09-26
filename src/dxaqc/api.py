"""Local HTTP API for batch inference over mounted DICOM inputs."""
from __future__ import annotations

import json
import logging
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .pipeline import run
from .report import write_errors, write_results

log = logging.getLogger("dxaqc.api")
MAX_REQUEST_BYTES = 65536


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def predict_batch(request: dict, model, input_root: Path, output_root: Path) -> dict:
    """Run a mounted folder/ZIP and persist one organizer-format result per image."""
    if not isinstance(request, dict):
        raise RequestError("тело запроса должно быть объектом JSON")
    name = request.get("input")
    if not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute():
        raise RequestError("input должен быть относительным путём внутри входного каталога")
    fmt = request.get("format", "csv")
    if fmt not in ("csv", "xlsx"):
        raise RequestError("format должен быть csv или xlsx")

    root = input_root.resolve(strict=True)
    source = (root / name).resolve()
    if not source.is_relative_to(root):
        raise RequestError("input выходит за пределы входного каталога")
    if not source.exists() or not (source.is_dir() or source.is_file()):
        raise RequestError("входной набор не найден", 404)
    if source.is_file() and not zipfile.is_zipfile(source):
        raise RequestError("вход должен быть каталогом или ZIP-архивом")

    started = time.perf_counter()
    rows, errors = run(source, model)
    job_id = uuid.uuid4().hex
    destination = output_root.resolve() / job_id
    results_path = destination / f"results.{fmt}"
    errors_path = destination / "errors.csv"
    write_results(rows, results_path)
    write_errors(errors, errors_path)
    return {
        "job_id": job_id,
        "results_file": str(results_path),
        "errors_file": str(errors_path),
        "processed": len(rows),
        "failed": sum(row["processing_status"] == "Failure" for row in rows),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "rows": rows,
        "errors": errors,
    }


def make_handler(model, input_root: Path, output_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                log.warning("клиент закрыл соединение до получения ответа")

        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/health":
                self._json(404, {"error": "маршрут не найден"})
                return
            self._json(200, {"status": "ok", "model_version": model.version})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/predict":
                self._json(404, {"error": "маршрут не найден"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "ожидается Content-Type: application/json"})
                return
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                size = -1
            if not 0 < size <= MAX_REQUEST_BYTES:
                self._json(413, {"error": "размер JSON-запроса должен быть от 1 до 65536 байт"})
                return
            try:
                request = json.loads(self.rfile.read(size))
                response = predict_batch(request, model, input_root, output_root)
            except json.JSONDecodeError:
                self._json(400, {"error": "некорректный JSON"})
            except RequestError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception:
                log.exception("ошибка пакетной обработки")
                self._json(500, {"error": "ошибка обработки; детали в журнале сервера"})
            else:
                self._json(200, response)

    return Handler


def serve(host: str, port: int, model, input_root: Path, output_root: Path) -> None:
    if not input_root.is_dir():
        raise ValueError(f"входной каталог не найден: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    with HTTPServer((host, port), make_handler(model, input_root, output_root)) as server:
        log.info("API слушает %s:%d", host, port)
        server.serve_forever()
