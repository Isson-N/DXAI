"""Разбиение на фолды, политика спорных меток и проверка выполнимости (план v2, раздел 2, пп. 1, 2, 6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5

# цель обучения: (область из images.csv, колонка метки)
TARGETS = {
    "spine_any": ("spine", "quality_class"),
    "spine_positioning": ("spine", "y_pos"),
    "spine_axis": ("spine", "y_axis"),
    "spine_foreign": ("spine", "y_foreign"),
    "hip_any": ("hip", "quality_class"),
    "hip_positioning": ("hip", "y_pos"),
    "hip_roi": ("hip", "y_roi"),
}
# порядок редкости для страты: самое редкое нарушение изображения определяет его страту
RARITY_ORDER = ["spine_positioning", "hip_roi", "spine_axis", "spine_foreign", "hip_positioning"]

# Расхождения «Итог» ↔ критерии: по указанию организатора (Q&A) цель «есть нарушение» = OR критериев,
# поэтому в обучении ничего не маскируется; файл политики фиксирует решение для прозрачности.
DISPUTED = [
    {"study_n": 6, "region": "spine", "target": "spine_any",
     "reason": "Итог = 1 при всех критериях 0 (сколиоз); цель = OR критериев = 0; сколиоз не нарушение (Q&A)"},
    {"study_n": 11, "region": "spine", "target": "spine_any",
     "reason": "Итог = 1 при всех критериях 0 (сколиоз); цель = OR критериев = 0; сколиоз не нарушение (Q&A)"},
    {"study_n": 35, "region": "spine", "target": "spine_any",
     "reason": "Итог = 0 при оси = 1 (перелом); цель = OR критериев = 1; перелом не отменяет критерий (Q&A)"},
]


def load_labeled(index_csv: str) -> pd.DataFrame:
    df = pd.read_csv(index_csv)
    return df[df["labeled"]].reset_index(drop=True)


def label_policy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in DISPUTED:
        hit = df[(df.study_n == d["study_n"]) & (df.region == d["region"])]
        for _, r in hit.iterrows():
            region, column = TARGETS[d["target"]]
            rows.append({  # без пути к файлу: в именах папок организатора есть номера обращений
                "study_n": d["study_n"], "study": r.study, "region": r.region, "side": _side(r),
                "target": d["target"],
                "total_official": int(r.total_official), "target_label": int(r[column]),
                "train_mask": 0, "reason": d["reason"],
            })
    return pd.DataFrame(rows)


def strata(df: pd.DataFrame) -> np.ndarray:
    out = []
    for _, r in df.iterrows():
        label = f"{r.region}_ok"
        for t in RARITY_ORDER:
            region, column = TARGETS[t]
            if r.region == region and r[column] == 1:
                label = t
                break
        out.append(label)
    return np.asarray(out)


def assign_folds(df: pd.DataFrame, seed: int) -> pd.Series:
    folds = pd.Series(-1, index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(sgkf.split(df, strata(df), groups=df.study)):
        folds.iloc[test_idx] = k
    return folds


def _side(r) -> str:
    return r.side if isinstance(r.side, str) else ""


def image_key(df: pd.DataFrame) -> pd.Series:
    """Идентификатор изображения без пути: исследование + область + сторона."""
    side = df["side"].where(df["side"].apply(lambda x: isinstance(x, str)), "")
    return df["study"] + "|" + df["region"] + "|" + side


def _masked(df: pd.DataFrame, policy: pd.DataFrame, target: str) -> pd.Series:
    """True для изображений, чья метка цели маскируется в обучении (train_mask = 1)."""
    if not len(policy):
        return pd.Series(False, index=df.index)
    p = policy[(policy.target == target) & (policy.get("train_mask", 1) == 1)].fillna({"side": ""})
    keys = set(p["study"] + "|" + p["region"] + "|" + p["side"])
    return image_key(df).isin(keys)


def feasibility(df: pd.DataFrame, folds: pd.Series, policy: pd.DataFrame) -> pd.DataFrame:
    """Число положительных/отрицательных исследований и изображений по целям:
    во внешнем тесте и во внутренних обучающих частях (внешнее обучение без одного внутреннего фолда)."""
    rows = []
    for target, (region, column) in TARGETS.items():
        sub = df[(df.region == region) & ~_masked(df, policy, target)]
        f = folds.loc[sub.index]
        for k in range(N_FOLDS):
            test = sub[f == k]
            rows.append(_counts(target, k, "outer_test", None, test, column))
            for j in (x for x in range(N_FOLDS) if x != k):
                inner_train = sub[(f != k) & (f != j)]
                rows.append(_counts(target, k, "inner_train", j, inner_train, column))
    return pd.DataFrame(rows)


def _counts(target, outer, part, inner, d, column) -> dict:
    pos_studies = d.loc[d[column] == 1, "study"].nunique()
    neg_studies = d.loc[d[column] == 0, "study"].nunique()
    return {"target": target, "outer_fold": outer, "part": part, "inner_fold": inner,
            "pos_images": int((d[column] == 1).sum()), "neg_images": int((d[column] == 0).sum()),
            "pos_studies": int(pos_studies), "neg_studies": int(neg_studies)}


def balance_score(df: pd.DataFrame, folds: pd.Series) -> float:
    """Чем меньше, тем ровнее распределены положительные исследования редких целей по внешним фолдам."""
    score = 0.0
    for t in RARITY_ORDER:
        region, column = TARGETS[t]
        sub = df[(df.region == region) & (df[column] == 1)]
        counts = np.array([sub.loc[folds.loc[sub.index] == k, "study"].nunique() for k in range(N_FOLDS)])
        score += counts.std() / max(counts.mean(), 1e-9)
    return score


def choose_seed(df: pd.DataFrame, candidates=range(200)) -> int:
    """Выбор сида только по распределению меток (не по результатам моделей)."""
    return min(candidates, key=lambda s: (balance_score(df, assign_folds(df, s)), s))
