# Что посмотреть, чтобы вкатиться в задачу «ИИ-контроль качества денситометрии»

Собрано 16.09.2026. Все ссылки открывались, названия, длительность и содержание видео
проверены (у ключевых — по субтитрам, отсюда и таймкоды).
У англоязычных видео есть автосубтитры, их можно автоматически перевести на русский.

Суть задачи в одну строку: на входе DICOM с DXA (поясница / проксимальный отдел бедра),
надо сказать: снимок годный или нет, и что именно не так — укладка (Th12…гребни
подвздошных костей), наклон оси > 5°, металл/артефакты, ротация бедра (малый вертел),
поля вокруг ROI (3 см / 2 см). Оценка по F1 и ROC-AUC с 95% ДИ.

---

## Итоговый топ по важности

| # | Что | Длина | Зачем именно для соревнования |
|---|---|---|---|
| 1 | D. Krueger — DXA Quality Matters | 28 мин | ровно наша задача глазами эксперта ISCD |
| 2 | ISCD DXA Atlas (галерея ошибок) | листать 1–2 ч | «насмотренность» на плохие снимки |
| 3 | StatQuest: матрица ошибок → Se/Sp → ROC-AUC → бутстрэп → ДИ | ~50 мин | метрики, по которым судят |
| 4 | Deep Learning School: CNN + сегментация (короткие лекции) | ~2 ч | быстрый вход в CV |
| 5 | Обучение YOLO-pose (ключевые точки) | 1 ч | Th12, гребни, ось позвоночника, угол |
| 6 | Son Dang — DXA Interpretation (фрагменты) | ~20 мин | разбор реальных кривых сканов |
| 7 | Статья Banks et al. 2023, Part 1 | 1 ч чтения | укладка и ROI по пунктам |
| 8 | Митап НПКЦ ДиТ по денситометрии (фрагменты) | ~25 мин | русская терминология, школа организатора |
| 9 | DICOM Explained + pydicom | ~35 мин | чтение входных данных |
| 10 | Разинков — CV using DL (лекции 1, 2, 5) | ~3 ч | глубже по CV, на русском |
| 11 | Varoquaux & Cheplygina 2022 | 40 мин чтения | утечка данных, честная валидация |
| 12 | U-Net за 10 минут + обучение U-Net | 10 мин + 1,7 ч | маски позвонков/бедра |
| 13 | Дисбаланс классов | 14 мин | «плохих» снимков будет мало |
| 14 | Grad-CAM | ~15 мин | тепловые карты для доп. серии |
| 15 | FastAPI + Docker | 19 мин | контейнер и API обязательны |
| 16 | CS231n 2017 (выборочно) | ~7,5 ч | если хочется фундамента |

Дальше расписано по блокам с ссылками.

---

## 1. Денситометрия и контроль качества (самое важное — это предметная область)

**1. DXA Quality Matters — Diane Krueger (ISCD), 28 мин** — главное видео.
https://www.youtube.com/watch?v=wBgfGPwh4Os
Частота ошибок в реальных центрах (30–50% сканов с ошибками), что должно быть на снимке
позвоночника и бедра, металл и артефакты, маркеры межпозвонковых промежутков, как узнать
L1–L5 по форме, как ставится neck box у GE и Hologic, центрирование и ротация бедра.
- 02:00–06:00 — статистика ошибок укладки и анализа
- 06:00–12:00 — позвоночник: артефакты, маркеры позвонков, края
- 13:00–17:00 — бедро: ротация, захват анатомии, neck box, внешние артефакты
- 21:00 — итоговый чек-лист проверки скана

**2. ISCD DXA Atlas — галерея снимков с ошибками** (бесплатно).
https://iscd.org/dxaatlas/hip-rotation/
С этой страницы открываются разделы: Positioning and Analysis Issues, Internal Artifacts,
External Artifacts, Normal Scans, Anatomical Variants. Просто листать и смотреть на
картинки: это ровно те классы, которые будет искать модель. (curl получает 403,
в браузере открывается.)

**6. DXA Interpretation — Son Dang, 1:11 (смотреть фрагменты).**
https://www.youtube.com/watch?v=TTnxMX37mAg
- 09:00–13:00 — укладка бедра, внутренняя ротация, что должно попасть в скан
- 16:00–30:00 — разбор неправильных сканов: разметка по Th12 вместо L1, кальцинаты,
  металл, неверные межпозвонковые линии, бедро без вертлужной впадины, повторяемость измерений

**7. Banks K. et al. Improving DXA Quality by Avoiding Common Technical and Diagnostic
Pitfalls: Part 1. J Nucl Med Technol, 2023** (полный текст бесплатно).
https://tech.snmjournals.org/content/51/3/167
Укладка позвоночника, бедра и предплечья, где должны стоять ROI, контроль качества.
Хорошо заходит после видео Krueger, пригодится для слайда «клиническая логика».

**8. Митап по остеопорозу и денситометрии — НПКЦ ДиТ ДЗМ, 2:17 (смотреть фрагменты).**
https://www.youtube.com/watch?v=3_nFi1PqfC0
Модератор — А. В. Петряйкин (НПКЦ ДиТ). Большая часть про клинику остеопороза.
- 1:04–1:15 — как устроена DXA, поясница L1–L4, исключение позвонков из-за артефактов,
  что должно быть в отчёте (ROI, по которым видна правильность укладки)
- 2:09–2:16 — типичные ошибки: разметка «на позвонок выше», повторное исследование
  с другой укладкой, позиционирование бедра

Тот же митап для рентгенолаборантов (содержание почти совпадает):
https://www.youtube.com/watch?v=tq-QrcFrRoM

Коротко и дополнительно:
- Common pitfalls in DXA scanning of the spine, 8 мин — https://www.youtube.com/watch?v=kfqKuZEB6MM
- DXA Tech Training (Hologic), 23 мин, главы 12:24–15:15 — как выглядят правильно
  снятые бедро и позвоночник — https://www.youtube.com/watch?v=PGwxJsGdars
- APO DXA Hologic Tech Training, 52 мин, 09:00–11:00 бедро/позвоночник, 14:00–19:00 примеры
  оптимальных и неоптимальных сканов (гребень подвздошной кости, обрезанный L4) —
  https://www.youtube.com/watch?v=VDcfbCrYNHk

Справочники, открывать по мере надобности:
- Обновлённое руководство EANM по DXA, 2024 (PMC, бесплатно) —
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11732917/
- ISCD Best Practices for DXA (PDF) —
  https://iscd.org/wp-content/uploads/2021/08/Best-Practices-DXA-Article.pdf
- Методрекомендации НПКЦ ДиТ «Остеоденситометрия» (Годзенко, Петряйкин и др., 2017) —
  https://telemedai.ru/biblioteka-dokumentov/osteodensitometriya

> Организаторы в п. 6 ссылаются на «методические рекомендации, разделы 2.6, 2.7», но не
> называют их. В открытом доступе документ с такими разделами найти не удалось:
> «Остеоденситометрия» 2017, «Московский стандарт…» 2024 и МР по КТ-скринингу 2023 не
> подходят. Скорее всего, его дадут вместе с данными — там будут точные критерии разметки.

---

## 2. Метрики и статистика (по ним будут судить)

**3. StatQuest, всего ~50 минут, смотреть подряд:**
1. Confusion Matrix, 7 мин — https://www.youtube.com/watch?v=Kdsp6soqA7o
2. Sensitivity and Specificity, 12 мин — https://www.youtube.com/watch?v=vP06aMoz4v8
3. ROC and AUC, 16 мин — https://www.youtube.com/watch?v=4jRBRDbJemM
4. Bootstrapping Main Ideas, 9 мин — https://www.youtube.com/watch?v=Xz0x-8-cgaQ
   (так и считают 95% ДИ для F1/AUC)
5. Confidence Intervals, 7 мин — https://www.youtube.com/watch?v=TqOeMYtOc1w

Дополнительно про F1:
- Precision, Recall & F1 Intuitively Explained, 9 мин — https://www.youtube.com/watch?v=8d3JbbSj-I8

**13. Дисбаланс классов — Emma Ding, 14 мин.**
https://www.youtube.com/watch?v=GR-OW5asKlk

**11. Varoquaux G., Cheplygina V. Machine learning for medical imaging: methodological
failures and recommendations for the future. npj Digit Med, 2022.**
https://www.nature.com/articles/s41746-022-00592-y
Почему модели в медицинской визуализации «хорошие на бумаге»: утечка данных (снимки
одного пациента в train и test), маленький test, подгонка под бенчмарк. В критериях
жюри (п. 8.1) прямо стоит «предотвращение утечки данных», так что это надо знать.

Для презентации — чек-лист CLAIM 2024, по нему видно, что обычно спрашивают:
https://pubs.rsna.org/doi/10.1148/ryai.240300

---

## 3. Computer vision — быстрый вход

**4. Deep Learning School (ФПМИ МФТИ) — короткие лекции, ~2 ч на всё:**
1. Лекция: свёрточные нейросети, 50 мин — https://www.youtube.com/watch?v=HpKGv-kYurk
2. Семинар: свёрточные сети, 22 мин — https://www.youtube.com/watch?v=xgVr9vISm8w
3. Лекция: архитектуры CNN, 50 мин — https://www.youtube.com/watch?v=TcUPuKpIlhQ
4. Semantic Segmentation: Introduction, 19 мин — https://www.youtube.com/watch?v=tIqndofykgc

Весь курс (бесплатно, со Stepik и домашками): https://stepik.org/course/272073/promo

**5. Практика из плейлиста «Компьютерное зрение с нуля до профи» (Д. Колесников),
на русском.** Весь плейлист не нужен, только это:
- Обучение YOLO-pose для Pose Estimation, 1:04 — https://www.youtube.com/watch?v=Oa9TufXt8GE
  → ключевые точки: верх Th12/L1, гребни подвздошных костей, центры позвонков → угол оси,
  малый/большой вертел, седалищная кость. Скорее всего, это самый прямой путь к половине
  критериев.
- Training U-Net for Semantic Segmentation in PyTorch, 1:42 — https://www.youtube.com/watch?v=zpyzBR3MuT0
- Аугментация данных, 1:44 — https://www.youtube.com/watch?v=xtcS6xexLu8
- Разметка данных в CVAT, 1:33 — https://www.youtube.com/watch?v=2XPsU-GlAcw
  (если экспертной разметки не хватит и точки придётся ставить самим)

Документация к YOLO-pose: https://docs.ultralytics.com/tasks/pose/

**10. Евгений Разинков — Computer Vision using DL (2019), на русском.**
https://www.youtube.com/playlist?list=PL6-BrcpR2C5RfrBJHr8WVyPK0WypMU9_j
Смотреть лекции 1 (введение, 48 мин), 2 (локализация объектов, 1:08), 5 (сегментация,
U-Net, 1:13). Детекцию (3, 4) — по желанию.

**12. U-Net:**
- The U-Net (actually) explained in 10 minutes — https://www.youtube.com/watch?v=NhdzGfB1q74

**14. Grad-CAM, 15 мин** — тепловые карты «куда смотрела модель», как раз для доп. серии
с визуализацией нарушения (п. 2.6).
https://www.youtube.com/watch?v=_QiebC9WxOc

**16. Stanford CS231n (2017) — фундамент, если есть время.**
https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk
Выборочно: 5 (CNN), 6–7 (обучение сетей), 9 (архитектуры), 11 (детекция и сегментация),
12 (визуализация, CAM). Около 7,5 ч.

Альтернатива для тех, кто любит «сначала сделать»: fast.ai, уроки 1–2 —
https://course.fast.ai/

---

## 4. Медицинские данные и инженерия

**9. DICOM:**
- What is DICOM | DICOM Explained, 10 мин — https://www.youtube.com/watch?v=uIy2Zp0QDSU
- DICOM in Python (pydicom), 22 мин — https://www.youtube.com/watch?v=To7v7i7eB0A
- highdicom — сборка DICOM SR и доп. серий (пп. 2.6, опционально) —
  https://highdicom.readthedocs.io/

**15. FastAPI + Docker, 19 мин** — контейнер, API пакетной обработки, закреплённые версии.
https://www.youtube.com/watch?v=h5wLuVDr0oc

---

## Порядок, если времени мало (~1 рабочий день)

1. Krueger (28 мин) → полистать ISCD Atlas (30 мин)
2. StatQuest ×5 (50 мин)
3. DLS: CNN + сегментация (~1,5 ч)
4. YOLO-pose (1 ч)
5. DICOM Explained (10 мин)
6. Когда придут данные — прочитать методрекомендации организатора (разделы 2.6–2.7)
   и Banks 2023.
