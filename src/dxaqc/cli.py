"""Командная строка: dxaqc predict --input <папка|zip> --output results.csv|xlsx"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .model import StubModel
from . import service_model
from .pipeline import run
from .report import write_errors, write_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dxaqc", description="Контроль качества DXA-исследований")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="пакетная обработка папки или zip-архива")
    p.add_argument("--input", required=True, type=Path, help="папка или zip с DICOM")
    p.add_argument("--output", required=True, type=Path, help="файл результата .csv или .xlsx")
    p.add_argument("--errors", type=Path, help="файл ошибок (по умолчанию errors.csv рядом с результатом)")
    p.add_argument("--models", type=Path, default=Path("models"),
                   help="каталог с весами (spine_keypoints.pt, quality_cnn.pt)")
    p.add_argument("--device", default="cpu", help="cpu или cuda")
    p.add_argument("--stub", action="store_true", help="заглушка вместо моделей (проверка конвейера)")
    parser.add_argument("--version", action="version", version=f"dxaqc {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.input.exists():
        logging.error("вход не найден: %s", args.input)
        return 2
    if args.stub:
        model = StubModel()
        logging.warning("используется заглушка модели (%s): результаты не являются прогнозом", model.version)
    else:
        model = service_model.load(args.models, args.device)
        for note in model.notes:
            logging.warning("%s", note)
        if model.keypoints is None and model.cnn is None:
            logging.error("в каталоге %s нет ни одной модели; запустите с --stub для проверки конвейера",
                          args.models)
            return 2
    rows, errors = run(args.input, model)
    write_results(rows, args.output)
    write_errors(errors, args.errors or args.output.with_name("errors.csv"))
    logging.info("результат: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
