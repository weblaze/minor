"""S4: CAV-MAE embeddings of every track and every painting.

Audio: the same windows as the MERT extraction (`audio.n_windows` contiguous windows of
`audio.window_s` seconds from the centre of the track), resampled to 16 kHz, 128-bin filterbank,
padded to 1024 frames. One embedding per window, stored as (n_windows, 768) per track in
`<embeddings>/audio/shard_*.npz`. The extraction resumes from the shards already written.

Images: shorter side scaled to 224 (bicubic), centre crop, ImageNet normalisation. One embedding
per painting in `<embeddings>/art.npz` (names, features).

Embeddings are the token means after the joint layers and are stored without normalisation.

Usage:
  python methods/s4/extract.py --source datasets/fma_small [--limit N]
"""
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from abstraction.data.audio import iter_audio, load_windows
from abstraction.data.datasets import IMAGE_EXTENSIONS
from abstraction.data.embeddings import done_ids, save_shard
from abstraction.encoders.cav_mae import (AUDIO_FRAMES, AUDIO_SAMPLE_RATE, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD,
                                          MEL_BINS, fbank_of, load_cav_mae, smoke_cav_mae)
from abstraction.pipeline.cli import setup
from abstraction.utils.runlog import RunLog, set_seed

from common import SMOKE_TRACKS, base_parser, run_name, smoke_image_names


class TrackWindows(Dataset):
    """Filterbanks of the windows of one track: (track id, tensor (n_windows, frames, bins) or None)."""

    def __init__(self, items, n_windows, window_s):
        self.items, self.n_windows, self.window_s = items, n_windows, window_s

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        track_id, loadable = self.items[index]
        try:
            windows = load_windows(loadable, AUDIO_SAMPLE_RATE, self.n_windows, self.window_s)
        except Exception as error:
            return track_id, None, type(error).__name__
        if windows is None:
            return track_id, None, "too_short"
        return track_id, torch.stack([fbank_of(torch.from_numpy(w)) for w in windows]), None


class SyntheticWindows(Dataset):
    """Random filterbanks with one offset per track. For smoke runs only."""

    def __init__(self, n_tracks, n_windows):
        g = torch.Generator().manual_seed(0)
        self.offsets = torch.randn(n_tracks, 1, 1, MEL_BINS, generator=g)
        self.n_windows = n_windows

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(index)
        fbank = self.offsets[index] + 0.3 * torch.randn(self.n_windows, AUDIO_FRAMES, MEL_BINS, generator=g)
        return f"synthetic_{index:05d}", fbank, None


class Paintings(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.names = sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.directory, self.names[index]))
        return self.transform(ImageOps.exif_transpose(image).convert("RGB"))


class SyntheticPaintings(Dataset):
    """Random low-frequency colour fields. For smoke runs only."""

    def __init__(self):
        self.names = smoke_image_names().tolist()
        self.low = torch.rand(len(self.names), 3, 4, 4, generator=torch.Generator().manual_seed(1)) * 2 - 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        return torch.nn.functional.interpolate(self.low[index:index + 1], size=IMAGE_SIZE, mode="bicubic")[0]


def collate_tracks(batch):
    return batch


@torch.no_grad()
def extract_audio(model, dataset, out_dir, hp, device, shard_size=1000):
    _, shard_index = done_ids(out_dir)
    ids, rows, skipped = [], [], []
    loader = DataLoader(dataset, batch_size=hp["tracks_per_batch"], num_workers=hp["workers"],
                        collate_fn=collate_tracks)

    def flush():
        nonlocal shard_index
        if ids:
            save_shard(out_dir, shard_index, ids, np.stack(rows))
            shard_index += 1
            ids.clear()
            rows.clear()

    n = 0
    for batch in tqdm(loader):
        good = [(track_id, fbank) for track_id, fbank, _ in batch if fbank is not None]
        skipped += [f"{track_id}\t{error}" for track_id, fbank, error in batch if fbank is None]
        if not good:
            continue
        fbank = torch.stack([f for _, f in good]).to(device)
        embedded = model.embed_audio(fbank.flatten(0, 1)).view(len(good), fbank.shape[1], -1)
        ids.extend(track_id for track_id, _ in good)
        rows.extend(embedded.float().cpu().numpy())
        n += len(good)
        if len(ids) >= shard_size:
            flush()
    flush()
    if skipped:
        with open(out_dir / "skipped.tsv", "a") as f:
            f.write("\n".join(skipped) + "\n")
    return n, len(skipped)


@torch.no_grad()
def extract_images(model, dataset, batch_size, workers, device):
    out = [model.embed_image(images.to(device)).float().cpu()
           for images in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=workers))]
    return torch.cat(out).numpy()


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--source", default=None, help="audio directory or zip; default paths.fma_audio")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N tracks only")
    parser.add_argument("--images", default=None, help="embed only this image folder (e.g. a GAN pool) into --out")
    parser.add_argument("--out", default=None, help="npz for --images")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    hp = config["cav_mae"]
    if args.images:
        model = load_cav_mae(args.checkpoint or hp["checkpoint"], device)
        images = Paintings(args.images)
        features = extract_images(model, images, hp["image_batch_size"], 0, device)
        np.savez(args.out, names=np.asarray(images.names), features=features)
        print(f"{len(features)} images embedded into {args.out}")
        return
    log = RunLog(config["paths"]["runs_dir"], run_name("extract", args.smoke), config, seed)

    if args.smoke:
        model = smoke_cav_mae(device)
        out_dir = log.dir / "embeddings"
        tracks = SyntheticWindows(SMOKE_TRACKS, config["audio"]["n_windows"])
        paintings = SyntheticPaintings()
        hp = {**hp, "workers": 0}
    else:
        model = load_cav_mae(args.checkpoint or hp["checkpoint"], device)
        out_dir = Path(args.embeddings or hp["embeddings"])
        finished, _ = done_ids(out_dir / "audio")
        items = list(iter_audio(args.source or config["paths"]["fma_audio"]))[:args.limit]
        items = [item for item in items if item[0] not in finished]
        tracks = TrackWindows(items, config["audio"]["n_windows"], config["audio"]["window_s"])
        paintings = Paintings(config["paths"]["abstract_art"])
    out_dir.mkdir(parents=True, exist_ok=True)

    features = extract_images(model, paintings, hp["image_batch_size"], hp["workers"], device)
    np.savez(out_dir / "art.npz", names=np.asarray(paintings.names), features=features)
    n, n_skipped = extract_audio(model, tracks, out_dir / "audio", hp, device)
    result = {"n_paintings": len(features), "n_tracks_embedded": n, "n_tracks_skipped": n_skipped,
              "embedding_dim": int(features.shape[1]), "embeddings": str(out_dir)}
    log.save_json("result.json", result)
    print(result)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
