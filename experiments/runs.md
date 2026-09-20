# Журнал запусков: команда → результат

Одна строка на запуск. Команды приводятся так, чтобы их можно было повторить дословно.
Развилки и причины выбора — в `docs/decisions.md`.

| дата | что | команда | результат | итог |
|---|---|---|---|---|
| 19.09 | бейзлайны, 256 px | `training/train_baselines.py --models b0,b1,b2 --epochs 30` | `experiments/results/{b0,b1,b2}/` | macro-F1 0,27 / 0,30 / 0,44; область — shortcut по размеру кадра |
| 19.09 | аудит OOF | `training/audit_oof.py experiments/results/{b0,b1,b2}/oof.csv` | `experiments/results/audit_b0_b1_b2.json`, `experiments/audit_2026-09-19.md` | значимы только hip_pos и hip_any (B2); по пояснице — ничего |
| 20.09 | модель точек, тепловые карты | `training/train_keypoints.py --epochs 60 --size 384` | — | ✖ не обучилось: ошибка 83–142 мм |
| 20.09 | модель точек, soft-argmax | `training/train_keypoints.py --epochs 150 --size 384 --final-model models/spine_keypoints.pt` | `experiments/results/keypoints/` | точки 3–5 мм, угол 0,65°; укладка F1 0,909, ось AUC 0,881 |
| 20.09 | бейзлайны 384 + маска | `training/train_baselines.py --models b0,b1,b2 --epochs 30 --size 384 --final-model` | `experiments/results384/` | B1 предметы F1 0,75 (AUC не изменился); B2 бедро 0,62 / 0,57 |
| 20.09 | сквозной прогон сервиса | `dxaqc predict --input data/train --models models --device cpu` | — | 499/499 Success, 0,068 с на снимок |

Планируемые запуски (вариант из `docs/decisions.md` → команда):

- 1б «центр плато»: `training/train_baselines.py --models b1,b2 --threshold-tolerance 0.02 --out experiments/results/plateau`
- 1в «выше специфичность»: `training/train_baselines.py --models b1,b2 --min-specificity 0.9 --out experiments/results/spec90`
- 2в «исходное разрешение»: `--size 405` (паддинг до самого большого кадра)
- 4б «отдельные модели по областям»: ветка `alt/per-region-models`
