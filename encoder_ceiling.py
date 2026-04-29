"""Encoder ceiling on real RAVDESS test split.

Establishes the upper bound for any downstream classifier on Wav2Lip-generated
frames: the F1 of the frozen audio (HuBERT) and video (TimeSformer) encoders
themselves on the real, ungenerated test data. The ceiling for `evaluate` in
04_finetune_wav2lip.ipynb is this video-encoder F1 — generated frames cannot
be classified more accurately than the encoder classifies real ones.

Usage (Colab / local with the same paths as the notebooks):
    python encoder_ceiling.py
        [--metadata /content/processed_data/metadata.json]
        [--audio-path /content/trained_encoders_6emotions/6emo-hubert-er-lr3e5-nf]
        [--video-path /content/trained_encoders_6emotions/6emo-tsf-lr3e5-16f-nf]
        [--out-dir /content/encoder_ceiling]

Outputs:
    encoder_ceiling.csv          per-encoder, per-emotion F1 + macro
    encoder_ceiling_audio.png    audio confusion matrix
    encoder_ceiling_video.png    video confusion matrix
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_fscore_support,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

EMOTIONS = ["happy", "sad", "angry", "fearful", "disgust", "surprised"]
EXCLUDE = {0, 1}  # neutral, calm
REMAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
WAV2LIP_TO_ENCODER = [2, 3, 4, 5, 6, 7]
NUM_EMO = len(EMOTIONS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default="/content/processed_data/metadata.json")
    p.add_argument("--audio-path",
                   default="/content/trained_encoders_6emotions/6emo-hubert-er-lr3e5-nf")
    p.add_argument("--video-path",
                   default="/content/trained_encoders_6emotions/6emo-tsf-lr3e5-16f-nf")
    p.add_argument("--out-dir", default="/content/encoder_ceiling")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-audio-s", type=float, default=3.0,
                   help="max audio length passed to HuBERT (matches training)")
    return p.parse_args()


def load_emotion_utils():
    """Import emotion_utils as the notebooks do — caller is responsible for cwd."""
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, "/content")
    from emotion_utils import (
        DifferentiableVideoPreprocess,
        load_frozen_audio_encoder,
        load_frozen_video_encoder,
    )
    return (DifferentiableVideoPreprocess,
            load_frozen_audio_encoder,
            load_frozen_video_encoder)


def remap_logits(logits, head_labels):
    """Match WAV2LIP_TO_ENCODER mapping used in 04 / 04b."""
    if head_labels == NUM_EMO:
        return logits
    return logits[:, WAV2LIP_TO_ENCODER]


def adapt_frames_to_n(frames_thwc, target_t):
    """frames_thwc: (T, H, W, 3) numpy. Returns (target_t, 3, H, W) tensor."""
    T = frames_thwc.shape[0]
    if T == target_t:
        idx = np.arange(target_t)
    else:
        idx = np.linspace(0, T - 1, target_t).round().astype(int)
    sampled = frames_thwc[idx]                                  # (target_t, H, W, 3)
    return torch.from_numpy(sampled).permute(0, 3, 1, 2).float()  # (T, 3, H, W)


@torch.no_grad()
def predict_video(video_enc, video_preprocess, frames_thwc, target_t, device):
    """Returns argmax label index in EMOTIONS space."""
    frames = adapt_frames_to_n(frames_thwc, target_t).to(device)  # (T, 3, H, W)
    pv = video_preprocess(frames.unsqueeze(0))                    # (1, T, 3, h, w)
    out = video_enc(pixel_values=pv)
    head_labels = int(getattr(video_enc.config, "num_labels", out.logits.shape[-1]))
    logits = remap_logits(out.logits, head_labels)
    return int(logits.argmax(dim=-1).item())


@torch.no_grad()
def predict_audio(audio_enc, audio_proc, wav_1d, max_audio_s, device):
    sr = getattr(audio_proc, "sampling_rate", 16000)
    enc = audio_proc([wav_1d.numpy()], sampling_rate=sr, return_tensors="pt",
                     padding="max_length", truncation=True, max_length=int(max_audio_s * sr))
    kwargs = {"input_values": enc["input_values"].to(device)}
    if "attention_mask" in enc:
        kwargs["attention_mask"] = enc["attention_mask"].to(device)
    out = audio_enc(**kwargs)
    head_labels = int(getattr(audio_enc.config, "num_labels", out.logits.shape[-1]))
    logits = remap_logits(out.logits, head_labels)
    return int(logits.argmax(dim=-1).item())


def report_and_plot(name, labels, preds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    macro_f1 = float(f1_score(labels, preds,
                               labels=list(range(NUM_EMO)), average="macro", zero_division=0))
    prec, rec, per_f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_EMO)), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_EMO)))

    print(f"\n=== {name} encoder on REAL test frames/audio ===")
    print(f"  macro F1: {macro_f1:.4f}")
    rows = []
    for i, e in enumerate(EMOTIONS):
        rows.append({"emotion": e, "P": prec[i], "R": rec[i], "F1": per_f1[i], "support": int(sup[i])})
    per_df = pd.DataFrame(rows)
    print(per_df.to_string(index=False))

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS).to_string())

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_EMO))
    ax.set_yticks(range(NUM_EMO))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{name} encoder — macro F1 = {macro_f1:.3f} (REAL test)")
    for i in range(NUM_EMO):
        for j in range(NUM_EMO):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=9, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_path = out_dir / f"encoder_ceiling_{name}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  -> saved {fig_path}")

    return macro_f1, per_f1


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DifferentiableVideoPreprocess, load_frozen_audio_encoder, load_frozen_video_encoder = \
        load_emotion_utils()

    print(f"Device: {args.device}")
    print(f"Audio encoder: {args.audio_path}")
    print(f"Video encoder: {args.video_path}")

    audio_enc, audio_proc = load_frozen_audio_encoder(args.audio_path, args.device)
    video_enc = load_frozen_video_encoder(args.video_path, args.device)
    video_preprocess = DifferentiableVideoPreprocess(224).to(args.device)
    target_t = int(getattr(video_enc.config, "num_frames", 8))
    print(f"Video encoder expects {target_t} frames per clip.")

    with open(args.metadata) as f:
        meta = json.load(f)
    samples = [s for s in meta if s["split"] == args.split and s["emotion_idx"] not in EXCLUDE]
    print(f"Real {args.split} samples: {len(samples)}")

    audio_labels, audio_preds = [], []
    video_labels, video_preds = [], []

    for s in tqdm(samples, desc="real test"):
        true = REMAP[s["emotion_idx"]]

        wav, _ = torchaudio.load(s["audio_path"])
        wav_1d = wav.squeeze(0)
        audio_labels.append(true)
        audio_preds.append(predict_audio(audio_enc, audio_proc, wav_1d,
                                          args.max_audio_s, args.device))

        frames = np.load(s["frames_path"]).astype(np.float32) / 255.0  # (T, H, W, 3)
        video_labels.append(true)
        video_preds.append(predict_video(video_enc, video_preprocess,
                                          frames, target_t, args.device))

    audio_labels = np.array(audio_labels)
    audio_preds = np.array(audio_preds)
    video_labels = np.array(video_labels)
    video_preds = np.array(video_preds)

    audio_macro, audio_per = report_and_plot("audio", audio_labels, audio_preds, out_dir)
    video_macro, video_per = report_and_plot("video", video_labels, video_preds, out_dir)

    summary = pd.DataFrame([
        {
            "encoder": "audio (HuBERT)",
            "macro_F1": audio_macro,
            **{f"F1_{e}": audio_per[i] for i, e in enumerate(EMOTIONS)},
        },
        {
            "encoder": "video (TimeSformer)",
            "macro_F1": video_macro,
            **{f"F1_{e}": video_per[i] for i, e in enumerate(EMOTIONS)},
        },
    ])
    csv_path = out_dir / "encoder_ceiling.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary -> {csv_path}")
    print(summary.to_string(index=False))

    print("\nThesis interpretation:")
    print("  Generated-frame F1 in 04 cannot exceed the video-encoder ceiling above.")
    print("  Audio F1 is informative for H2 (SadTalker conditioning) and the KL")
    print("  reference distribution used in 04's CE+KL loss.")


if __name__ == "__main__":
    main()
