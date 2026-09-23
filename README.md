# dxaqc — контроль качества DXA-исследований

> **Кто подхватывает проект — начните с [HANDOFF.md](HANDOFF.md):** что изменилось 21–23.09.2026, итоговые метрики, где веса, что осталось сделать.


Сервис по DICOM определяет анатомическую область (поясница / проксимальный отдел бедра), класс качества,
вероятность нарушения и тип нарушения. **Сейчас (этап 1) модель — заглушка**: область по размеру кадра,
нарушений нет. Полное описание по п. 5 ТЗ — на этапе 6 (`PLAN.md`).

## Быстрый старт

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-analysis.txt -e ".[dev]"
.venv/bin/python -m pytest                                        # тесты
.venv/bin/dxaqc predict --input data/test --output out/results.xlsx   # папка или zip → xlsx/csv + errors.csv

bash docker/build.sh                                              # образ dxaqc:0.1.0
bash docker/run.sh data/test out csv                              # запуск без сети
bash docker/smoke_test.sh                                         # сборка + запуск + проверка формата
```

## Структура

| Путь | Что |
|---|---|
| `src/dxaqc/` | сервис: `io` (чтение DICOM), `model` (интерфейс, заглушка), `pipeline`, `report` (формат выхода), `cli`; `metrics`, `cv` — протокол валидации |
| `tools/` | разбор данных (`build_index.py`, `overview_sheets.py`), фолды (`make_folds.py`), разметка поясниц (`spine_annotator.py`, `merge_spine_points.py`; см. `docs/annotation.md`) |
| `experiments/` | зафиксированные фолды, политика спорных меток, проверка выполнимости |
| `docker/` | Dockerfile и скрипты сборки/запуска/smoke-теста |
| `docs/questions_to_organizer.md` | вопросы организатору |
| `PLAN.md`, `NOTES.md`, `DISCUSSION.md` | план, журнал, переписка команды |

Данные организатора и внешние данные в git не хранятся (`data/README.md`, `extra-data/README.md`).
