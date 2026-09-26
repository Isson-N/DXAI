"""Поиск и чтение DICOM-файлов."""
from __future__ import annotations

import logging
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

logging.getLogger("pydicom").setLevel(logging.ERROR)
# UID организатора с ведущими нулями и длинные LO нарушают VR — читаем без проверки, это не ошибка файла
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

# SOP-классы без изображения: пропускаются, в errors.csv попадает запись
NON_IMAGE_SOP_PREFIXES = (
    "1.2.840.10008.1.3.10",       # Media Storage Directory (DICOMDIR)
    "1.2.840.10008.5.1.4.1.1.88",  # Structured Report
    "1.2.840.10008.5.1.4.1.1.11",  # Presentation State
)


@dataclass
class DicomImage:
    path: str                     # путь относительно корня входа
    study_uid: str
    image_uid: str
    pixels: np.ndarray            # 2D float32, яркая кость = большие значения
    columns: int
    rows: int
    warnings: list[str] = field(default_factory=list)


class NotAnImage(Exception):
    """Файл — DICOM, но без изображения (DICOMDIR, SR и т.п.)."""


def extract_if_archive(src: Path, workdir: Path) -> Path:
    """Папку возвращает как есть, zip распаковывает во временный каталог."""
    if src.is_dir():
        return src
    if zipfile.is_zipfile(src):
        dst = workdir / "input"
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                name = _zip_name(info)
                target = (dst / name).resolve()
                if not target.is_relative_to(dst.resolve()):
                    continue  # защита от путей вида ../
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(info))
        return dst
    raise ValueError(f"вход не папка и не zip-архив: {src}")


def _zip_name(info: zipfile.ZipInfo) -> str:
    # zip из Windows часто хранит кириллицу в cp866 без флага UTF-8
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp866")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


def list_candidate_files(root: Path) -> list[Path]:
    """Все файлы, отсортированные по пути (детерминированный порядок)."""
    return sorted(p for p in root.rglob("*") if p.is_file())


def looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() in {".dcm", ".dicom"}:
        return True
    try:
        with path.open("rb") as f:
            head = f.read(132)
        return len(head) == 132 and head[128:132] == b"DICM"
    except OSError:
        return False


def read_image(path: Path, root: Path) -> DicomImage:
    """Читает DICOM и возвращает изображение. Исключение — если прочитать нельзя."""
    rel = path.relative_to(root).as_posix()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ds = pydicom.dcmread(path)
        except InvalidDicomError:
            ds = pydicom.dcmread(path, force=True)
        sop_class = str(ds.get("SOPClassUID", ""))
        if not sop_class:
            raise ValueError("повреждённый DICOM: нет SOPClassUID")
        if sop_class.startswith(NON_IMAGE_SOP_PREFIXES) or "PixelData" not in ds:
            raise NotAnImage(f"нет изображения (SOPClassUID={sop_class})")
        arr = ds.pixel_array

    notes: list[str] = []
    photometric = str(ds.get("PhotometricInterpretation", "MONOCHROME2"))
    if arr.ndim == 3 and photometric not in ("RGB", "YBR_FULL", "YBR_FULL_422"):
        notes.append(f"многокадровое изображение ({arr.shape[0]} кадров), взят первый кадр")
        arr = arr[0]
    if arr.ndim == 3:  # цветное: берём яркость
        arr = arr[..., :3].astype(np.float32).mean(axis=-1)
    arr = arr.astype(np.float32)
    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if photometric == "MONOCHROME1":  # инверсия относительно максимума разрядности, не содержимого снимка
        arr = ((2 ** int(ds.get("BitsStored", 8)) - 1) * slope + intercept) - arr

    return DicomImage(
        path=rel,
        study_uid=str(ds.get("StudyInstanceUID", "")),
        image_uid=str(ds.get("SOPInstanceUID", "")),
        pixels=arr,
        columns=int(arr.shape[1]),
        rows=int(arr.shape[0]),
        warnings=notes,
    )
