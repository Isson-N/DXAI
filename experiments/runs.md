# Журнал запусков: команда → результат

Одна строка на запуск. Команды приводятся так, чтобы их можно было повторить дословно.
Развилки и причины выбора — в `docs/decisions.md`.

| дата | что | команда | результат | итог |
|---|---|---|---|---|
| 19.09 | бейзлайны, 256 px | `training/train_baselines.py --models b0,b1,b2 --epochs 30` | `experiments/results/{b0,b1,b2}/` | macro-F1 0,27 / 0,30 / 0,44; область — shortcut по размеру кадра |
| 19.09 | аудит OOF | `training/audit_oof.py experiments/results/{b0,b1,b2}/oof.csv` | `experiments/results/audit_b0_b1_b2.json`, `experiments/audit_2026-09-19.md` | значимы только hip_pos и hip_any (B2); по пояснице — ничего |
| 20.09 | модель точек, 384 px | `training/train_keypoints.py --annotations ... --epochs 60 --size 384` | `experiments/results/keypoints/` | считается |

Планируемые запуски (вариант из `docs/decisions.md` → команда):

- 1б «центр плато»: `training/train_baselines.py --models b1,b2 --threshold-tolerance 0.02 --out experiments/results/plateau`
- 1в «выше специфичность»: `training/train_baselines.py --models b1,b2 --min-specificity 0.9 --out experiments/results/spec90`
- 2в «исходное разрешение»: `--size 405` (паддинг до самого большого кадра)
- 4б «отдельные модели по областям»: ветка `alt/per-region-models`
