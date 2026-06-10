"""Train the BYOL-A encoder self-supervised on FMA audio (no labels).

Log-mel spectrograms are cached to datasets/byol_mel_cache/ on first epoch so
later epochs skip librosa entirely.

Usage:
  python train_byol_a.py [--epochs N] [--max-steps N] [--no-wandb]
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from abstraction.utils.checkpoints import save_checkpoint
from abstraction.utils.config import load_config
from abstraction.utils.wandb_utils import init_wandb

from byol_a_model import BYOLA, N_MELS, TARGET_FRAMES, Augmenter

APPROACH = "03_byol_a_encoder"
SR = 16000


def log_mel(audio_path, cache_dir):
    cache = Path(cache_dir) / (Path(audio_path).stem + ".npy")
    if cache.exists():
        return np.load(cache)
    import librosa

    y, _ = librosa.load(audio_path, sr=SR, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=1024, hop_length=160)
    mel = np.log(mel + 1e-8).astype(np.float32)
    np.save(cache, mel)
    return mel


class MelDataset(Dataset):
    def __init__(self, fma_dir, cache_dir):
        self.files = []
        for genre in os.listdir(fma_dir):
            genre_path = os.path.join(fma_dir, genre)
            if os.path.isdir(genre_path):
                self.files += [os.path.join(genre_path, f) for f in os.listdir(genre_path)
                               if f.endswith((".mp3", ".wav"))]
        if not self.files:
            raise FileNotFoundError(f"No audio files under {fma_dir}")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = log_mel(self.files[idx], self.cache_dir)
        # random 1s-ish crop, per-sample standardization
        if mel.shape[1] > TARGET_FRAMES:
            start = random.randint(0, mel.shape[1] - TARGET_FRAMES)
            mel = mel[:, start:start + TARGET_FRAMES]
        else:
            mel = np.pad(mel, ((0, 0), (0, TARGET_FRAMES - mel.shape[1])))
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        return torch.from_numpy(mel).unsqueeze(0)  # [1, N_MELS, T]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    byol_cfg = config["byol_a"]
    epochs = args.epochs or byol_cfg["num_epochs"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["system"]["seed"])
    print(f"Using device: {device}")

    dataset = MelDataset(config["paths"]["fma_audio"], config["paths"]["mel_cache"])
    loader = DataLoader(dataset, batch_size=byol_cfg["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0)
    print(f"Found {len(dataset)} audio files")

    model = BYOLA(embed_dim=config["cond"]["dim"], ema_decay=byol_cfg["ema_decay"]).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=float(byol_cfg["learning_rate"]))
    augment = Augmenter(mixup_alpha=byol_cfg["mixup_alpha"])
    ckpt_path = config["paths"]["byol_checkpoint"]

    run = None if args.no_wandb else init_wandb(config, APPROACH, "train")

    model.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            view1 = torch.stack([augment(x) for x in batch]).to(device)
            view2 = torch.stack([augment(x) for x in batch]).to(device)

            optimizer.zero_grad()
            loss = model(view1, view2)
            loss.backward()
            optimizer.step()
            model.update_target()

            epoch_loss += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(model.online_encoder, ckpt_path)
                print(f"[smoke] stopped after {step} steps, encoder saved to {ckpt_path}")
                return

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | byol loss {avg:.4f}", flush=True)
        if run:
            run.log({"epoch": epoch + 1, "loss": avg})
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            save_checkpoint(model.online_encoder, ckpt_path)
            print(f"--- encoder checkpoint saved at epoch {epoch + 1} ---", flush=True)

    save_checkpoint(model.online_encoder, ckpt_path)
    print(f"Saved BYOL-A encoder to {ckpt_path}")


if __name__ == "__main__":
    main()
