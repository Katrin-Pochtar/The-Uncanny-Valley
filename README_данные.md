# Данные эксперимента - приложение к ВКР

Магистерская диссертация: «Разработка функции потерь для обеспечения эмоциональной согласованности между речью и мимикой при дообучении аудио-управляемых моделей генерации говорящих лиц».

Автор: Почтар Катрин Викторовна, Центр «Пуск», МФТИ, группа М08-401НД.

Папка содержит все данные, использованные и сгенерированные в ходе эксперимента, в форме, достаточной для проверки заявленных в ВКР результатов без повторного запуска обучения.

---

## Источник данных

### Исходный корпус - RAVDESS

[Livingstone S. R., Russo F. A. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), 2018](https://zenodo.org/record/1188976).

- Лицензия: CC BY-NA-SC 4.0.
- Год записи: 2018.
- Объём: 2880 аудиовизуальных записей.
- 24 актёра (12 мужчин, 12 женщин), 2 фразы, 2 эмоциональные интенсивности, 8 эмоций (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`), плюс песенные модальности.
- Условия записи: студия, фиксированная поза головы, контролируемое освещение, 25 fps, 48 kHz аудио.

Сам датасет в этой папке не дублируется - он публичен и устанавливается по ссылке выше. В папку положены только производные артефакты эксперимента, опирающиеся на этот корпус.

### Используемое подмножество

Из 8 классов отобраны 4: `happy`, `sad`, `angry`, `disgust`. Обоснование выбора приведено в разделе 3.2 текста ВКР: классы с диагностичной мимикой и совместимые с пространствами всех отобранных внешних FER-классификаторов.

Объём подмножества: 736 аудиовизуальных сэмплов (24 актёра × 2 фразы × 2 повтора × 2 уровня интенсивности × 4 эмоции с поправкой на отбраковку по детекции лица).

---

## Разбиение по актёрам (actor-aware split)

| Сплит | Сэмплов | Актёров | Сэмплов на эмоцию |
|---|---|---|---|
| train | 544 | 19 | 136 |
| val | 96 | 2-3 | 24 |
| test | 96 | 2-3 | 24 |

Разбиение проведено по идентификаторам субъектов, что исключает утечку индивидуальных особенностей говорящего между сплитами. Конкретные идентификаторы актёров каждого сплита - в `metadata.json`.

---

## Структура папки

```
03_Данные_эксперимента/
├── README_данные.md                       ← этот файл
├── metadata.json                          ← основная разметка сплитов и путей
├── results/                               ← CSV-выгрузки таблиц из ВКР
│   ├── encoder_sweep.csv
│   ├── encoder_ceiling.csv
│   ├── external_screening_table.csv
│   ├── wav2lip_ablation.csv
│   ├── wav2lip_external_eval.csv
│   ├── sadtalker_ablation.csv
│   ├── bootstrap_ci.csv
│   └── per_actor_sign_test.csv
├── confusion_matrices/                    ← матрицы ошибок baseline vs финал
│   ├── wav2lip_baseline_internal.csv
│   ├── wav2lip_cekl01_internal.csv
│   ├── wav2lip_baseline_motheecreator.csv
│   ├── wav2lip_cekl01_motheecreator.csv
│   ├── wav2lip_baseline_Rajaram1996.csv
│   └── wav2lip_cekl05_Rajaram1996.csv
├── samples_video/                         ← примеры сгенерированных видео
│   ├── baseline/
│   │   ├── happy_actor12.mp4
│   │   ├── sad_actor12.mp4
│   │   ├── angry_actor12.mp4
│   │   └── disgust_actor12.mp4
│   └── cekl-01/
│       ├── happy_actor12.mp4
│       ├── sad_actor12.mp4
│       ├── angry_actor12.mp4
│       └── disgust_actor12.mp4
└── wandb_figures/                         ← экспорт графиков из W&B
    ├── fig_01_encoder_training.png
    ├── fig_02_wav2lip_ablation.png
    ├── fig_03_tradeoff.png
    ├── fig_04_cm_internal.png
    ├── fig_05_per_emotion.png
    └── fig_06_sadtalker_coef_vs_render.png
```

---

## Описание ключевых файлов

### `metadata.json`

Источник истины для всех последующих этапов. JSON-массив записей следующего формата:

```json
{
  "sample_id": "01-01-03-01-01-01-12",
  "actor_id": 12,
  "emotion": "happy",
  "emotion_idx": 2,
  "intensity": "normal",
  "statement_idx": 1,
  "repeat_idx": 1,
  "audio_path": "processed/audio/01-01-03-01-01-01-12.wav",
  "video_path": "processed/video/01-01-03-01-01-01-12.npy",
  "mel_path": "processed/mel/01-01-03-01-01-01-12.npy",
  "duration_sec": 3.72,
  "face_detection_rate": 1.00,
  "split": "train"
}
```

Покадровый успех детекции лица: 98,9 % на train, 99,0 % на val и test. Сэмплы с долей детекции < 90 % исключены.

### `results/encoder_sweep.csv`

Полный перебор 22 конфигураций энкодеров эмоций (этап 2). Столбцы: `name`, `modality` (audio/video), `model_name`, `learning_rate`, `epochs_run`, `best_val_F1`, `best_epoch`. Источник: `02_train_emotion_encoders.ipynb`, cell 12 (RESULTS SUMMARY).

Связь с ВКР: таблица в Приложении В.

### `results/encoder_ceiling.csv`

Per-emotion F1 финальных энкодеров на реальной тестовой выборке. Столбцы: `encoder`, `macro_F1`, `F1_happy`, `F1_sad`, `F1_angry`, `F1_disgust`. Источник: `04_encoder_ceiling.ipynb`.

Связь с ВКР: табл. 2 в разделе «Подбор опорных энкодеров эмоций».

### `results/external_screening_table.csv`

Скрининг 5 кандидатов внешних FER-классификаторов на тестовой выборке RAVDESS (n = 96) и на train+val (n = 640). Столбцы: `classifier`, `split`, `macro_F1`, `F1_happy`, `F1_sad`, `F1_angry`, `F1_disgust`. Источник: `05_external_classifier_screening.ipynb`, cell 7.

Связь с ВКР: табл. 3 в разделе «Скрининг внешних классификаторов».

### `results/wav2lip_ablation.csv`

Пять конфигураций функции потерь на Wav2Lip. Столбцы: `config`, `w_ce`, `w_cos`, `w_kl`, `best_L1`, `best_F1`, `delta_F1`, `McNemar_chi2`, `McNemar_p`. Источник: `06_wav2lip_loss_ablation.ipynb`.

Связь с ВКР: табл. 4 в разделе «Абляция функции потерь».

### `results/wav2lip_external_eval.csv`

Внешняя оценка всех 9 конфигураций Wav2Lip двумя FER-классификаторами. Столбцы: `config`, `internal_val_F1`, `delta_motheecreator`, `delta_Rajaram1996`, `both_positive`, `mean_delta`. Источник: `08_wav2lip_external_evaluation.ipynb`.

Связь с ВКР: табл. 5 в разделе «H1: внешняя валидация».

### `results/sadtalker_ablation.csv`

Четыре конфигурации SadTalker на 3DMM-коэффициент-уровне. Столбцы: `config`, `val_F1`, `delta_F1`, `McNemar_chi2`, `McNemar_p`, плюс поэмоциональная разбивка. Источник: `09_sadtalker_loss_ablation.ipynb`.

Связь с ВКР: табл. в разделе «H2: подтверждение на коэффициент-уровне».

### `results/bootstrap_ci.csv`

Бутстрап-95 %-CI на $\Delta\mathrm{F1}$ для всех 80 троек (датасет × конфигурация × эвалюатор × сплит). 10 000 реплик с возвращением, паросочетание по `sample_id`. Столбцы: `dataset`, `external`, `split`, `config`, `delta_F1`, `CI_lower`, `CI_upper`, `CI_excludes_zero`. Источник: `11_bootstrap_analysis.ipynb`, cell 9.

Связь с ВКР: табл. в разделе «H1: бутстрап-анализ доверительных интервалов».

### `results/per_actor_sign_test.csv`

Per-actor разбивка $\Delta\mathrm{accuracy}$. Столбцы: `dataset`, `external`, `split`, `config`, `n_actors`, `n_pos`, `n_neg`, `n_tied`, `mean_delta`, `median_delta`, `sign_test_p`. Источник: `11_bootstrap_analysis.ipynb`, cell 6.

Связь с ВКР: табл. в разделе «H1: per-actor sign-test».

### `confusion_matrices/`

Матрицы ошибок $4 \times 4$ (rows = true, cols = predicted) для baseline и финальной модели по каждому классификатору. Каждый CSV - отдельная матрица; используются в Приложении Д текста ВКР для качественного анализа смещения «всё happy» у baseline и его устранения после CE+KL-файнтюнинга.

### `samples_video/`

По 4 примера сгенерированных видео (по одному на эмоцию) от baseline Wav2Lip и от финальной модели `wav2lip-cekl-01`. Один и тот же актёр (12), одни и те же фразы - для прямого визуального сравнения. Видео в формате MP4 H.264, разрешение 96 × 96, длительность 3-4 с.

### `wandb_figures/`

Шесть PNG-графиков, экспортированных из W&B и упомянутых в тексте ВКР меткой `[W&B figure: ...]`:

| Файл | Что на нём | Источник |
|---|---|---|
| `fig_01_encoder_training.png` | Кривые валидационной F1 для топ-5 audio + топ-5 video конфигураций | `uncanny-valley-encoders-4emo` |
| `fig_02_wav2lip_ablation.png` | Кривые val L1 и val F1 для 5 конфигов абляции | `uncanny-valley-wav2lip-ablation-4emo` |
| `fig_03_tradeoff.png` | Scatter L1 vs LSE-C по всем чекпойнтам | `uncanny-valley-wav2lip-4emo` |
| `fig_04_cm_internal.png` | Confusion matrix baseline / cekl-01 на внутреннем TimeSformer | `uncanny-valley-wav2lip-4emo` |
| `fig_05_per_emotion.png` | Barplot $\Delta\mathrm{F1}$ по 4 эмоциям и 9 конфигурациям | `uncanny-valley-wav2lip-4emo` |
| `fig_06_sadtalker_coef_vs_render.png` | Сравнение coef-уровня и rendered-уровня SadTalker | `uncanny-valley-sadtalker-ablation-4emo` |

---

## Покрытие данными утверждений в ВКР

| Раздел ВКР | Откуда брать данные |
|---|---|
| Подбор опорных энкодеров | `results/encoder_sweep.csv` + `wandb_figures/fig_01_encoder_training.png` |
| Потолок задачи | `results/encoder_ceiling.csv` |
| Скрининг внешних | `results/external_screening_table.csv` |
| Абляция Wav2Lip | `results/wav2lip_ablation.csv` + `fig_02_wav2lip_ablation.png` |
| H1: внутренняя оценка | `confusion_matrices/wav2lip_*_internal.csv` + `fig_04_cm_internal.png` |
| H1: внешняя валидация | `results/wav2lip_external_eval.csv` + `confusion_matrices/wav2lip_*_motheecreator.csv` |
| H1: бутстрап и per-actor | `results/bootstrap_ci.csv` + `results/per_actor_sign_test.csv` |
| H1: компромисс L1/LSE-C | `fig_03_tradeoff.png` |
| H1: per-emotion анализ | `fig_05_per_emotion.png` |
| H2: коэффициент-уровень | `results/sadtalker_ablation.csv` |
| H2: rendered-уровень | `fig_06_sadtalker_coef_vs_render.png` |

---

## Воспроизведение данных «с нуля»

Все материалы из этой папки могут быть пересозданы заново выполнением пайплайна из репозитория `https://github.com/Katrin-Pochtar/The-Uncanny-Valley`. Порядок:

1. Скачать RAVDESS с [Zenodo](https://zenodo.org/record/1188976).
2. Запустить `01_data_preprocessing.ipynb` → формирует `metadata.json`.
3. Последовательно прогнать ноутбуки `02` → `11` (см. `README_артефакты.md`).

Все CSV из `results/` собираются в финальных ячейках соответствующих ноутбуков, графики - экспортируются вручную из W&B по ссылкам, видеосэмплы - генерируются в `07_wav2lip_finetune_H1.ipynb`.

---

## Соответствие первоисточникам

- Все числа в CSV-файлах совпадают с числами, приведёнными в тексте ВКР `main.tex` (приложения А-Е). При расхождении приоритет - за данными в этой папке (они получены непосредственно из ноутбуков, текст ВКР цитирует их).
- Все CSV сохранены с разделителем «запятая», кодировка UTF-8, заголовок в первой строке.
- Десятичный разделитель в CSV - точка (научный стандарт); в тексте ВКР - запятая (русский академический стандарт).

---

## Контакт

Почтар Катрин Викторовна, ЦДПО «Пуск», МФТИ.

Тема диссертации согласована с научным руководителем; апробация - 68-я Всероссийская научная конференция МФТИ, секция когнитивных технологий, 4 апреля 2026 г.
