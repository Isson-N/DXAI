"""Деление работы между двумя разметчиками — общее для всех инструментов.

Повторяет схему `spine_annotator.py`, где она уже отработала на позвоночнике:
исследования раскладываются в детерминированном псевдослучайном порядке и
делятся по чётности, а несколько исследований отдаются обоим — на них считается
согласие разметчиков.

Деление идёт по ИССЛЕДОВАНИЯМ, а не по снимкам: иначе два снимка бедра одного
пациента попадут к разным людям, и сравнение левого с правым станет сравнением
двух разных разметчиков.
"""
from __future__ import annotations

import hashlib

PARTS = ("1", "2", "all")


def _digest(*values: str) -> str:
    return hashlib.sha1("\0".join(values).encode("utf-8")).hexdigest()


def split_studies(studies, part, overlap: int = 10, salt: str = "") -> set[str]:
    """Исследования, которые достаются указанной части.

    `part` — «1», «2» или «all»; `overlap` — сколько исследований получают оба
    разметчика; `salt` разводит деление между инструментами, чтобы общие
    исследования в разных задачах были разными.
    """
    part = str(part)
    if part not in PARTS:
        raise ValueError("Часть должна быть 1, 2 или all.")
    unique = sorted(set(studies))
    ordered = sorted(unique, key=lambda study: (_digest(salt, study), study))
    if part == "all":
        return set(ordered)
    shared = set(sorted(unique, key=lambda s: (_digest("overlap", salt, s), s))[:overlap])
    parity = int(part) - 1
    return {study for index, study in enumerate(ordered)
            if index % 2 == parity or study in shared}


def describe(studies, part, overlap: int = 10, salt: str = "") -> str:
    """Строка для вывода при запуске: сколько исследований и сколько общих."""
    chosen = split_studies(studies, part, overlap, salt)
    if str(part) == "all":
        return f"часть all: все {len(chosen)} исследований"
    other = split_studies(studies, "2" if str(part) == "1" else "1", overlap, salt)
    return (f"часть {part}: {len(chosen)} исследований, "
            f"из них общих с другой частью — {len(chosen & other)}")
