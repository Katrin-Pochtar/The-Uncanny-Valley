"""
Кросс-модальная функция потерь для дообучения аудио-управляемых моделей
генерации говорящих лиц.

Магистерская диссертация Почтар К. В., МФТИ, 2026 г.
Тема: "Разработка функции потерь для обеспечения эмоциональной согласованности
       между речью и мимикой при дообучении аудио-управляемых моделей
       генерации говорящих лиц."

Полная формализация:
    L = L_recon + s * (w_ce * L_CE + w_cos * L_COS + w_kl * L_KL)

где
    L_recon  --- L1-реконструкция нижней половины кадра (Wav2Lip-наследие);
    L_CE     --- перекрёстная энтропия логитов видеоэнкодера по меткам эмоций;
    L_COS    --- косинусное сближение проекций аудио- и видеопризнаков;
    L_KL     --- KL-дистилляция распределения от аудиоэнкодера к видеоэнкодеру
                 при температуре T;
    s        --- общий масштабный коэффициент;
    w_*      --- покомпонентные веса.

Победившая в абляции конфигурация (этап 4 методики):
    Wav2Lip:   w_ce=0,05, w_cos=0,0,  w_kl=0,10, T=2,0 (CE+KL)
    SadTalker: w_ce=0,05, w_cos=0,20, w_kl=0,0,  T=2,0 (CE+COS)

См. Приложение Б текста ВКР, листинг lst:loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    """Конфигурация кросс-модальной функции потерь.

    Attributes
    ----------
    scale : float
        Общий масштабный коэффициент `s`. Контролирует силу эмоционального
        сигнала относительно реконструкции. Выбран по результатам поиска
        в `07_wav2lip_finetune_H1.ipynb`.
    w_ce : float
        Вес перекрёстной энтропии по меткам эмоций.
    w_cos : float
        Вес косинусного сближения проекций. Для CE+KL-конфигурации = 0.
    w_kl : float
        Вес KL-дистилляции. Для CE+COS-конфигурации = 0.
    temperature : float
        Температура softmax при KL-дистилляции (стандартное значение из
        работы Hinton et al., 2015).
    """

    scale: float = 0.10
    w_ce: float = 0.05
    w_cos: float = 0.0
    w_kl: float = 0.10
    temperature: float = 2.0


# Конфигурации из ВКР, табл. 4 (Wav2Lip) и табл. в разделе H2 (SadTalker)
CONFIG_WAV2LIP_CEKL_01 = LossConfig(
    scale=0.10, w_ce=0.05, w_cos=0.0, w_kl=0.10, temperature=2.0,
)
CONFIG_SADTALKER_CE_COS = LossConfig(
    scale=0.25, w_ce=0.05, w_cos=0.20, w_kl=0.0, temperature=2.0,
)
CONFIG_BASELINE = LossConfig(
    scale=0.0, w_ce=0.0, w_cos=0.0, w_kl=0.0, temperature=2.0,
)


def cross_modal_emotion_loss(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    audio_logits: torch.Tensor,
    video_logits: torch.Tensor,
    emotion_label: torch.Tensor,
    audio_proj: Optional[torch.Tensor] = None,
    video_proj: Optional[torch.Tensor] = None,
    config: LossConfig = CONFIG_WAV2LIP_CEKL_01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Вычисление кросс-модальной функции потерь.

    Parameters
    ----------
    pred_video : Tensor, shape (B, C, H, W) или (B, T, C, H, W)
        Сгенерированный декодером кадр (нижняя половина лица для Wav2Lip).
    target_video : Tensor, той же формы
        Эталонный кадр для L1-реконструкции.
    audio_logits : Tensor, shape (B, 4)
        Логиты эмоций от зафиксированного аудиоэнкодера. Градиент отсекается
        внутри функции.
    video_logits : Tensor, shape (B, 4)
        Логиты эмоций от зафиксированного видеоэнкодера на pred_video.
    emotion_label : Tensor, shape (B,), dtype=int64
        Истинные метки эмоций (0=happy, 1=sad, 2=angry, 3=disgust).
    audio_proj, video_proj : Tensor, shape (B, D), optional
        Проекции в общее пространство для косинусного члена.
        Обязательны, если config.w_cos > 0.
    config : LossConfig
        Веса и температура функции потерь.

    Returns
    -------
    loss : Tensor (скаляр)
        Полная функция потерь для обратного распространения.
    components : dict[str, Tensor]
        Покомпонентные значения (recon, ce, kl, cos, emo_total)
        для логирования в W&B.
    """
    # 1. L1-реконструкция (Wav2Lip-наследие)
    loss_recon = F.l1_loss(pred_video, target_video)

    # 2. Перекрёстная энтропия: явная супервизия по меткам
    if config.w_ce > 0:
        loss_ce = F.cross_entropy(video_logits, emotion_label)
    else:
        loss_ce = torch.zeros((), device=pred_video.device)

    # 3. KL-дистилляция: перенос softmax-распределения от аудио к видео.
    #    Градиент идёт через video_logits (student); audio_logits заморожены.
    if config.w_kl > 0:
        T = config.temperature
        student = F.log_softmax(video_logits / T, dim=-1)
        teacher = F.softmax(audio_logits.detach() / T, dim=-1)
        loss_kl = F.kl_div(student, teacher, reduction="batchmean") * (T ** 2)
    else:
        loss_kl = torch.zeros((), device=pred_video.device)

    # 4. Косинусное сближение проекций в общем пространстве
    if config.w_cos > 0:
        if audio_proj is None or video_proj is None:
            raise ValueError(
                "audio_proj и video_proj обязательны при w_cos > 0"
            )
        cos_sim = F.cosine_similarity(video_proj, audio_proj.detach(), dim=-1)
        loss_cos = (1.0 - cos_sim).mean()
    else:
        loss_cos = torch.zeros((), device=pred_video.device)

    # 5. Итоговая функция потерь
    loss_emo = (
        config.w_ce * loss_ce
        + config.w_kl * loss_kl
        + config.w_cos * loss_cos
    )
    loss = loss_recon + config.scale * loss_emo

    components = {
        "recon": loss_recon.detach(),
        "ce": loss_ce.detach(),
        "kl": loss_kl.detach(),
        "cos": loss_cos.detach(),
        "emo_total": loss_emo.detach(),
        "total": loss.detach(),
    }
    return loss, components


class CrossModalEmotionLoss(nn.Module):
    """Модульная обёртка над `cross_modal_emotion_loss` для совместимости
    с трекерами параметров nn.Module.

    Пример использования:
        >>> criterion = CrossModalEmotionLoss(CONFIG_WAV2LIP_CEKL_01)
        >>> loss, comp = criterion(pred_v, gt_v, a_logits, v_logits, labels)
        >>> loss.backward()
    """

    def __init__(self, config: LossConfig = CONFIG_WAV2LIP_CEKL_01) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        audio_logits: torch.Tensor,
        video_logits: torch.Tensor,
        emotion_label: torch.Tensor,
        audio_proj: Optional[torch.Tensor] = None,
        video_proj: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return cross_modal_emotion_loss(
            pred_video=pred_video,
            target_video=target_video,
            audio_logits=audio_logits,
            video_logits=video_logits,
            emotion_label=emotion_label,
            audio_proj=audio_proj,
            video_proj=video_proj,
            config=self.config,
        )


__all__ = [
    "LossConfig",
    "CONFIG_WAV2LIP_CEKL_01",
    "CONFIG_SADTALKER_CE_COS",
    "CONFIG_BASELINE",
    "cross_modal_emotion_loss",
    "CrossModalEmotionLoss",
]
