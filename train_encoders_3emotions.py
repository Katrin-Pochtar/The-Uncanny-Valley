#!/usr/bin/env python3
"""
Train audio/video emotion encoders on three classes only: happy, angry, disgust.
Logic mirrors 02_train_encoders.ipynb; metadata must use RAVDESS emotion_idx as in 01_data_preprocessing.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    HubertForSequenceClassification,
    TimesformerForVideoClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

warnings.filterwarnings("ignore")

# Same ordering as 01_data_preprocessing.ipynb EMOTION_TO_IDX
EMOTION_NAME_TO_IDX = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}

EMOTIONS = ["happy", "angry", "disgust"]
ALLOWED_EMOTION_IDX = {EMOTION_NAME_TO_IDX[e] for e in EMOTIONS}
# happy -> 0, angry -> 1, disgust -> 2 (order in EMOTIONS list)
REMAP = {EMOTION_NAME_TO_IDX[e]: i for i, e in enumerate(EMOTIONS)}
NUM_EMOTIONS = len(EMOTIONS)
LABEL_SMOOTHING = 0.1


class EmotionDataset(Dataset):
    def __init__(self, metadata_path: str | Path, split: str, modality: str):
        with open(metadata_path) as f:
            data = json.load(f)
        self.samples = [
            s
            for s in data
            if s["split"] == split and s["emotion_idx"] in ALLOWED_EMOTION_IDX
        ]
        self.modality = modality

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {"emotion": REMAP[s["emotion_idx"]]}
        if self.modality == "audio":
            wav, _ = torchaudio.load(s["audio_path"])
            item["audio"] = wav.squeeze(0)
        else:
            frames = np.load(s["frames_path"])
            item["video"] = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return item


def collate_fn(batch):
    out = {"emotion": torch.tensor([b["emotion"] for b in batch])}
    if "audio" in batch[0]:
        out["audio"] = [b["audio"] for b in batch]
    if "video" in batch[0]:
        out["video"] = torch.stack([b["video"] for b in batch])
    return out


def crop_audio(wav, sr, duration, train):
    L = int(round(duration * sr))
    n = wav.numel()
    if n <= L:
        return torch.nn.functional.pad(wav, (0, L - n))
    start = torch.randint(0, n - L + 1, ()).item() if train else (n - L) // 2
    return wav[start : start + L]


def crop_video(video, n_frames, train):
    T = video.shape[0]
    if T <= n_frames:
        idx = torch.linspace(0, T - 1, n_frames).round().long()
        return video[idx]
    start = torch.randint(0, T - n_frames + 1, ()).item() if train else (T - n_frames) // 2
    return video[start : start + n_frames]


def prepare_audio(batch, processor, window_s, device, train=True):
    sr = 16000
    wavs = [crop_audio(a, sr, window_s, train).numpy() for a in batch["audio"]]
    enc = processor(
        wavs,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(window_s * sr),
    )
    kwargs = {"input_values": enc["input_values"].to(device)}
    if "attention_mask" in enc:
        kwargs["attention_mask"] = enc["attention_mask"].to(device)
    return kwargs, batch["emotion"].to(device)


def prepare_video(batch, processor, n_frames, device, train=True):
    clips = []
    for v in batch["video"]:
        clip = crop_video(v, n_frames, train)
        clips.append([clip[i].permute(1, 2, 0).numpy() for i in range(clip.shape[0])])
    enc = processor(clips, return_tensors="pt", do_rescale=False)
    return {"pixel_values": enc["pixel_values"].to(device)}, batch["emotion"].to(device)


def train_one_epoch(model, loader, prep_fn, optimizer, scaler, loss_fn, device):
    model.train()
    total_loss, preds, labels = 0.0, [], []
    for batch in tqdm(loader, leave=False):
        kwargs, y = prep_fn(batch, train=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=device == "cuda"):
            logits = model(**kwargs).logits
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds.extend(logits.argmax(1).detach().cpu().tolist())
        labels.extend(y.cpu().tolist())
    return {
        "loss": total_loss / len(loader),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


@torch.no_grad()
def evaluate(model, loader, prep_fn, loss_fn, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for batch in tqdm(loader, leave=False):
        kwargs, y = prep_fn(batch, train=False)
        with autocast("cuda", enabled=device == "cuda"):
            logits = model(**kwargs).logits
            loss = loss_fn(logits, y)
        total_loss += loss.item()
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    return {
        "loss": total_loss / len(loader),
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    cfg: dict,
    metadata: str,
    out_dir: Path,
    device: str,
    use_wandb: bool,
    wandb_project: str,
):
    seed_all()
    name, modality = cfg["name"], cfg["modality"]
    print(f"{'=' * 60}\n{name}\n{'=' * 60}")

    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            name=name,
            group=modality,
            config={**cfg, "emotions": EMOTIONS},
            reinit=True,
        )

    train_ds = EmotionDataset(metadata, "train", modality)
    val_ds = EmotionDataset(metadata, "val", modality)
    bs = cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_fn)

    if modality == "audio":
        model_cls = HubertForSequenceClassification if "hubert" in cfg["model"].lower() else Wav2Vec2ForSequenceClassification
        model = model_cls.from_pretrained(cfg["model"], num_labels=NUM_EMOTIONS, ignore_mismatched_sizes=True)
        processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg["model"])
        prep_fn = partial(prepare_audio, processor=processor, window_s=cfg.get("window_s", 3.0), device=device)
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
    else:
        model = TimesformerForVideoClassification.from_pretrained(
            cfg["model"], num_labels=NUM_EMOTIONS, ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(cfg["model"])
        prep_fn = partial(prepare_video, processor=processor, n_frames=cfg.get("n_frames", 8), device=device)
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False

    model.to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scaler = GradScaler(enabled=device == "cuda")
    scheduler = None

    best_f1, patience_cnt = 0.0, 0
    save_path = out_dir / name

    for epoch in range(cfg["epochs"]):
        freeze_ep = cfg.get("freeze_epochs", 2)
        if epoch == freeze_ep:
            for p in model.parameters():
                p.requires_grad = True
            unfreeze_lr = cfg["lr"] if freeze_ep == 0 else cfg["lr"] * 0.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=unfreeze_lr)
            scaler = GradScaler(enabled=device == "cuda")
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"] - epoch, eta_min=1e-7)

        t = train_one_epoch(model, train_loader, prep_fn, optimizer, scaler, loss_fn, device)
        v = evaluate(model, val_loader, prep_fn, loss_fn, device)

        if scheduler:
            scheduler.step()

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": t["loss"],
                    "train/acc": t["acc"],
                    "train/f1": t["f1"],
                    "val/loss": v["loss"],
                    "val/acc": v["acc"],
                    "val/f1": v["f1"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        print(f"  [{epoch + 1:2d}/{cfg['epochs']}] " f"t_f1={t['f1']:.3f} v_f1={v['f1']:.3f} v_loss={v['loss']:.3f}")

        if v["f1"] > best_f1:
            best_f1 = v["f1"]
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_path))
            processor.save_pretrained(str(save_path))
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.get("patience", 5):
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if use_wandb:
        import wandb

        wandb.log({"best_val_f1": best_f1})
        wandb.finish()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"  Best F1: {best_f1:.4f} -> {save_path}\n")
    return {"name": name, "best_f1": best_f1, "path": str(save_path), "modality": modality}


def default_experiments():
    """Same grid as 02_train_encoders.ipynb; run names prefixed to avoid overwriting 6-class runs."""
    p = "3emo-"
    return [
        {"name": f"{p}w2v2-er-lr3e5", "modality": "audio", "model": "superb/wav2vec2-base-superb-er", "lr": 3e-5, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}w2v2-er-lr5e5", "modality": "audio", "model": "superb/wav2vec2-base-superb-er", "lr": 5e-5, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}w2v2-er-lr1e4", "modality": "audio", "model": "superb/wav2vec2-base-superb-er", "lr": 1e-4, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}w2v2-er-lr5e5-w5", "modality": "audio", "model": "superb/wav2vec2-base-superb-er", "lr": 5e-5, "window_s": 5.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}hubert-er-lr3e5", "modality": "audio", "model": "superb/hubert-base-superb-er", "lr": 3e-5, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}hubert-er-lr5e5", "modality": "audio", "model": "superb/hubert-base-superb-er", "lr": 5e-5, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}hubert-er-lr1e4", "modality": "audio", "model": "superb/hubert-base-superb-er", "lr": 1e-4, "window_s": 3.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}hubert-er-lr5e5-w5", "modality": "audio", "model": "superb/hubert-base-superb-er", "lr": 5e-5, "window_s": 5.0, "batch_size": 8, "epochs": 50, "freeze_epochs": 2, "patience": 8},
        {"name": f"{p}w2v2-lg-lr2e5", "modality": "audio", "model": "facebook/wav2vec2-large", "lr": 2e-5, "window_s": 3.0, "batch_size": 4, "epochs": 40, "freeze_epochs": 3, "patience": 7},
        {"name": f"{p}hubert-lg-lr2e5", "modality": "audio", "model": "facebook/hubert-large-ll60k", "lr": 2e-5, "window_s": 3.0, "batch_size": 4, "epochs": 40, "freeze_epochs": 3, "patience": 7},
        {"name": f"{p}tsf-lr1e5-16f", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 1e-5, "n_frames": 16, "batch_size": 2, "epochs": 30, "freeze_epochs": 1, "patience": 7},
        {"name": f"{p}tsf-lr3e5-16f", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 3e-5, "n_frames": 16, "batch_size": 2, "epochs": 30, "freeze_epochs": 1, "patience": 7},
        {"name": f"{p}tsf-lr5e5-16f", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 5e-5, "n_frames": 16, "batch_size": 2, "epochs": 30, "freeze_epochs": 1, "patience": 7},
        {"name": f"{p}tsf-lr3e5-16f-nf", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 3e-5, "n_frames": 16, "batch_size": 2, "epochs": 30, "freeze_epochs": 0, "patience": 7},
        {"name": f"{p}tsf-lr3e5-8f", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 3e-5, "n_frames": 8, "batch_size": 4, "epochs": 30, "freeze_epochs": 1, "patience": 7},
        {"name": f"{p}tsf-lr1e5-16f-f3", "modality": "video", "model": "facebook/timesformer-base-finetuned-k400", "lr": 1e-5, "n_frames": 16, "batch_size": 2, "epochs": 30, "freeze_epochs": 3, "patience": 7},
    ]


def main():
    parser = argparse.ArgumentParser(description="Train encoders on happy, angry, disgust only.")
    parser.add_argument(
        "--metadata",
        type=str,
        default="processed_data/metadata.json",
        help="Path to metadata.json from preprocessing",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="trained_encoders_3emotions",
        help="Directory for saved models",
    )
    parser.add_argument("--wandb-project", type=str, default="uncanny-valley-encoders-3emo")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).resolve()
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata not found: {metadata_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

        wandb.login()

    seed_all(args.seed)
    print(f"Device: {device}")
    print(f"Classes: {NUM_EMOTIONS} — {EMOTIONS}")
    print(f"RAVDESS emotion_idx used: {sorted(ALLOWED_EMOTION_IDX)} -> labels {list(range(NUM_EMOTIONS))}")
    print(f"Metadata: {metadata_path}")
    print(f"Output:   {out_dir}\n")

    experiments = default_experiments()
    results = []
    for exp in experiments:
        results.append(
            run_experiment(
                exp,
                str(metadata_path),
                out_dir,
                device,
                use_wandb,
                args.wandb_project,
            )
        )

    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Name':35s} {'Modality':>8s} {'Best Val F1':>12s}")
    print("-" * 58)
    for r in sorted(results, key=lambda x: -x["best_f1"]):
        print(f"{r['name']:35s} {r['modality']:>8s} {r['best_f1']:12.4f}")

    best_audio = max((r for r in results if r["modality"] == "audio"), key=lambda x: x["best_f1"])
    best_video = max((r for r in results if r["modality"] == "video"), key=lambda x: x["best_f1"])
    print(f"\nBest audio: {best_audio['name']} (F1={best_audio['best_f1']:.4f})")
    print(f"Best video: {best_video['name']} (F1={best_video['best_f1']:.4f})")


if __name__ == "__main__":
    main()
