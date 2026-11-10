"""Extract MERT embeddings for every track of an audio collection.

For each track, `audio.n_windows` contiguous windows of `audio.window_s` seconds
are cut from the centre, resampled to 24 kHz and passed through the frozen
MERT-v1-95M. The hidden states of all 13 layers are averaged over time and
stored as float16, shape (n_windows, 13, 768) per track. Layer weighting and
normalisation are applied later, at training time, so that every method reads
the same files.

Usage:
  python scripts/extract_mert.py --source datasets/fma_small [--limit N]
  python scripts/extract_mert.py --source fma_medium.zip --out datasets/mert_embeddings
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from abstraction.data.audio import iter_audio, load_windows
from abstraction.data.embeddings import done_ids, save_shard
from abstraction.encoders.mert import SAMPLE_RATE, MertEncoder
from abstraction.utils.config import load_config


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=config["paths"]["fma_audio"])
    parser.add_argument("--out", default=config["paths"]["mert_embeddings"])
    parser.add_argument("--revision", default=config["mert"]["revision"])
    parser.add_argument("--tracks-per-batch", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    audio = config["audio"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    encoder = MertEncoder(device, revision=args.revision)

    finished, shard_index = done_ids(args.out)
    ids, rows, pending_ids, pending = [], [], [], []
    skipped = []

    def flush_batch():
        if not pending:
            return
        emb = encoder.embed(np.concatenate(pending))
        emb = emb.view(len(pending), audio["n_windows"], *emb.shape[1:]).numpy().astype(np.float16)
        ids.extend(pending_ids)
        rows.extend(emb)
        pending.clear()
        pending_ids.clear()

    def flush_shard():
        nonlocal shard_index
        if ids:
            save_shard(args.out, shard_index, ids, np.stack(rows))
            shard_index += 1
            ids.clear()
            rows.clear()

    n = 0
    for track_id, loadable in tqdm(iter_audio(args.source)):
        if track_id in finished:
            continue
        if args.limit is not None and n >= args.limit:
            break
        try:
            windows = load_windows(loadable, SAMPLE_RATE, audio["n_windows"], audio["window_s"])
        except Exception as error:
            skipped.append(f"{track_id}\t{type(error).__name__}")
            continue
        if windows is None:
            skipped.append(f"{track_id}\ttoo_short")
            continue
        pending_ids.append(track_id)
        pending.append(windows)
        n += 1
        if len(pending) == args.tracks_per_batch:
            flush_batch()
        if len(ids) >= args.shard_size:
            flush_shard()
    flush_batch()
    flush_shard()

    if skipped:
        with open(f"{args.out}/skipped.tsv", "a") as f:
            f.write("\n".join(skipped) + "\n")
    print(f"embedded {n} tracks, skipped {len(skipped)}; hidden-state route: {encoder.route}")


if __name__ == "__main__":
    main()
