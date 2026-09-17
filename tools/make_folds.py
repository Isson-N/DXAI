#!/usr/bin/env python3
"""Фиксирует фолды, политику спорных меток и таблицу выполнимости в experiments/.

Нужен data/index/images.csv (tools/build_index.py). Запуск из корня проекта:
    .venv/bin/python tools/make_folds.py
"""
from pathlib import Path

from dxaqc import cv

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments"


def main():
    OUT.mkdir(exist_ok=True)
    df = cv.load_labeled(ROOT / "data" / "index" / "images.csv")
    policy = cv.label_policy(df)
    seed = cv.choose_seed(df)
    folds = cv.assign_folds(df, seed)

    per_study = df.assign(fold=folds)[["study_n", "study", "fold"]].drop_duplicates().sort_values("study_n")
    assert per_study.study.is_unique, "исследование попало в разные фолды"
    per_study.to_csv(OUT / "folds.csv", index=False)
    policy.to_csv(OUT / "label_policy.csv", index=False)

    feas = cv.feasibility(df, folds, policy)
    feas.to_csv(OUT / "feasibility.csv", index=False)

    inner = feas[feas.part == "inner_train"]
    problems = inner[(inner.pos_images == 0) | (inner.neg_images == 0)]
    summary = (feas[feas.part == "outer_test"].groupby("target")[["pos_studies", "neg_studies"]]
               .agg(["min", "max"]))
    print(f"сид разбиения: {seed}")
    print("положительные/отрицательные исследования во внешних тестовых фолдах (min–max):")
    print(summary.to_string())
    print(f"внутренних обучающих частей без одного из классов: {len(problems)}")
    if len(problems):
        print(problems.to_string(index=False))


if __name__ == "__main__":
    main()
