# Артефакты исследования — приложение к ВКР

Магистерская диссертация: «Разработка функции потерь для обеспечения эмоциональной согласованности между речью и мимикой при дообучении аудио-управляемых моделей генерации говорящих лиц».

Автор: Почтар Катрин Викторовна, ЦДПО «Пуск», МФТИ, группа М08-401НД.

Папка содержит все артефакты, созданные в 4 семестре 2025/2026 учебного года в рамках работы над исследованием.

---

## Принцип связи артефактов с исследованием

Каждый артефакт привязан к одному из пяти этапов методики и даёт конкретный результат, упомянутый в тексте ВКР. Полный реестр приведён в Приложении А `main.tex`; ниже — рабочая версия для навигации по этой папке.

| Артефакт | Этап методики | Конкретный результат в ВКР |
|---|---|---|
| `notebooks/01_data_preprocessing.ipynb` | Этап 1 — подготовка данных | Сплит 544 / 96 / 96 по актёрам, `metadata.json` |
| `notebooks/02_train_emotion_encoders.ipynb` | Этап 2 — энкодеры эмоций | Аудио val F1 = 0,826; видео val F1 = 0,790; полный перебор 22 конфигураций (Приложение В) |
| `notebooks/03_emotion_utils.ipynb` | Этап 2 — утилиты | Унифицированный интерфейс загрузки/инференса для этапов 4-5 |
| `notebooks/04_encoder_ceiling.ipynb` | Этап 2 — потолок задачи | Test macro-F1: аудио 0,739; видео 0,876 (табл. 2 в ВКР) |
| `notebooks/05_external_classifier_screening.ipynb` | Этап 3 — скрининг внешних FER | Отбор `motheecreator/vit-FER` и `Rajaram1996/FacialEmoRecog` из шести кандидатов; отвергнуты 4 (табл. 3 в ВКР) |
| `notebooks/06_wav2lip_loss_ablation.ipynb` | Этап 4 — абляция Wav2Lip | CE+KL победил с val F1 = 0,737 (Δ = +0,118 от baseline); cos-only показал отрицательный Δ — доказательство риска коллапса проекций (табл. 4 в ВКР) |
| `notebooks/07_wav2lip_finetune_H1.ipynb` | Этап 5 — проверка H1 | Финальная модель `wav2lip-cekl-01`: F1 0,611 → 0,718 на тесте (Δ = +0,107); LSE-C изменился незначимо (p = 0,61) |
| `notebooks/08_wav2lip_external_evaluation.ipynb` | Этап 5 — внешняя валидация Wav2Lip | Межклассификаторное согласие True для 8 из 8 эмоциональных конфигов; Spearman ρ(internal, external) = +0,612 |
| `notebooks/09_sadtalker_loss_ablation.ipynb` | Этап 4 — абляция SadTalker | CE+COS победил на 3DMM-коэффициентах: val F1 0,507 → 0,651 (Δ = +0,143, McNemar p = 0,0095) |
| `notebooks/10_sadtalker_finetune_H2.ipynb` | Этап 5 — проверка H2 | На rendered-уровне эффект между двумя внешними классификаторами расходится; H2 в строгой формулировке не подтверждена |
| `notebooks/11_bootstrap_analysis.ipynb` | Этап 5 — статистика | Бутстрап-95 %-CI: для cekl-05 на Rajaram1996/test интервал [+0,004; +0,176] не включает ноль (значимый эффект) |
| `code/cross_modal_loss.py` | Этап 4 — реализация функции потерь | Референсная реализация CE+KL-комбинации, использованная в этапе 5 |
| `checkpoints/audio_encoder.pth` | Этап 2 — финальный аудиоэнкодер | Веса `superb/wav2vec2-base-superb-er` после дообучения на 4 эмоциях |
| `checkpoints/video_encoder.pth` | Этап 2 — финальный видеоэнкодер | Веса `facebook/timesformer-base-finetuned-k400` (8 кадров) после дообучения |
| `checkpoints/wav2lip_cekl-01.pth` | Этап 5 — финальная модель H1 | Дообученный Wav2Lip с CE+KL ($s=0{,}10$, $w_{\text{ce}}=0{,}005$, $w_{\text{kl}}=0{,}010$, $T=2{,}0$) |
| `checkpoints/sadtalker_abl-ce-cos.pth` | Этап 5 — финальная модель H2 | Дообученный SadTalker с CE+COS на 3DMM-коэффициент-уровне |

---

## Структура папки

```
04_Артефакты/
├── README_артефакты.md          ← этот файл
├── notebooks/                    ← 11 ноутбуков пайплайна
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_emotion_encoders.ipynb
│   ├── 03_emotion_utils.ipynb
│   ├── 04_encoder_ceiling.ipynb
│   ├── 05_external_classifier_screening.ipynb
│   ├── 06_wav2lip_loss_ablation.ipynb
│   ├── 07_wav2lip_finetune_H1.ipynb
│   ├── 08_wav2lip_external_evaluation.ipynb
│   ├── 09_sadtalker_loss_ablation.ipynb
│   ├── 10_sadtalker_finetune_H2.ipynb
│   └── 11_bootstrap_analysis.ipynb
├── code/
│   └── cross_modal_loss.py       ← функция потерь, отдельным модулем
├── checkpoints/                  ← веса финальных моделей
│   ├── audio_encoder.pth
│   ├── video_encoder.pth
│   ├── wav2lip_cekl-01.pth
│   └── sadtalker_abl-ce-cos.pth
├── repo_link.txt                 ← ссылка на открытый репозиторий
└── wandb_links.txt               ← ссылки на четыре публичных W&B-проекта
```

---

## Порядок воспроизведения

Ноутбуки нумерованы по порядку выполнения; каждый последующий опирается на артефакты предыдущих. Полный пробег от `01` до `11` занимает ориентировочно 12-16 часов на одном NVIDIA T4 или 4-5 часов на A100.

```
Подготовка                01 → 02 → 03 → 04
Скрининг эвалюаторов      05
Wav2Lip                   06 (абляция) → 07 (файнтюнинг + H1) → 08 (внешняя оценка)
SadTalker                 09 (абляция)                       → 10 (файнтюнинг + H2)
Статистика                11 (бутстрап + per-actor sign-test)
```

Симметрия этапов 06-07 (Wav2Lip) и 09-10 (SadTalker) реализует «одинаковую методологическую рамку» сравнения двух архитектур, описанную в разделе «Дискуссия» текста ВКР.

---

## Аппаратные и программные требования

- **GPU:** один NVIDIA T4 (16 GB) для большинства этапов; для SadTalker (этап 10) — A100 (40 GB) либо RTX 3090 (24 GB).
- **Python:** 3.10. PyTorch 2.1, transformers 4.40, scipy 1.11, librosa 0.10, wandb 0.16. Полный закреплённый список зависимостей — в первой ячейке каждого ноутбука.
- **Воспроизводимость:** seed = 42 для `random`, `numpy`, `torch`, `torch.cuda`; `torch.backends.cudnn.deterministic = True`.
- **CUDA:** 12.x.

Большинство ноутбуков запускались в Google Colab; локальные эксперименты по внешней оценке (этап 5) — на RTX 3090.

---

## Использование чекпойнтов

### Загрузка опорных энкодеров

```python
from transformers import (
    Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor,
    TimesformerForVideoClassification, AutoImageProcessor,
)
import torch

# Аудиоэнкодер
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er", num_labels=4, ignore_mismatched_sizes=True,
)
audio_model.load_state_dict(torch.load("checkpoints/audio_encoder.pth"))
audio_model.eval()

# Видеоэнкодер
video_model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400", num_labels=4, ignore_mismatched_sizes=True,
)
video_model.load_state_dict(torch.load("checkpoints/video_encoder.pth"))
video_model.eval()
```

Порядок классов в логитах: `[happy, sad, angry, disgust]`.

### Загрузка дообученного Wav2Lip

```python
ckpt = torch.load("checkpoints/wav2lip_cekl-01.pth", map_location="cpu", weights_only=False)
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
# Затем загрузить в архитектуру Wav2Lip из репозитория https://github.com/Rudrabha/Wav2Lip
```

Подробности — в ноутбуке `07_wav2lip_finetune_H1.ipynb`, секция «Загрузка модели».

---

## Внешние ресурсы

### Открытый репозиторий

`https://github.com/Katrin-Pochtar/The-Uncanny-Valley`

Содержит идентичный набор ноутбуков, `main.tex` диссертации, README с описанием пайплайна.

### W&B-проекты с журналами обучения

| Проект | Этап методики | Содержимое |
|---|---|---|
| `uncanny-valley-encoders-4emo` | 2 | 22 запуска перебора энкодеров, кривые val F1 |
| `uncanny-valley-wav2lip-ablation-4emo` | 4 | 5 абляционных запусков Wav2Lip |
| `uncanny-valley-wav2lip-4emo` | 5 | Файнтюнинг Wav2Lip с CE+KL при 4 значениях масштаба |
| `uncanny-valley-sadtalker-ablation-4emo` | 4-5 | Абляция и файнтюнинг SadTalker |

Полные URL — в `wandb_links.txt`.

### Датасет

RAVDESS [Livingstone, Russo, 2018], публично доступен по адресу `https://zenodo.org/record/1188976` под лицензией CC BY-NA-SC 4.0. В рамках исследования используется четырёхклассовое подмножество (happy, sad, angry, disgust); полная разметка отобранных сэмплов — в `03_Данные_эксперимента/metadata.json`.

---

## Контакт

Почтар Катрин Викторовна, ЦДПО «Пуск», МФТИ.

Тема диссертации согласована с научным руководителем; апробация — 68-я Всероссийская научная конференция МФТИ, секция когнитивных технологий, 4 апреля 2026 г.
