from pathlib import Path

import numpy as np


def save_shard(out_dir, index, ids, array):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"shard_{index:05d}.tmp.npz"
    np.savez(tmp, ids=np.asarray(ids, dtype=str), data=array)
    tmp.replace(out_dir / f"shard_{index:05d}.npz")


def load_shards(directory):
    """Load every shard written by an extraction script.

    Returns (ids, data): ids is a list of strings, data an array whose first
    axis matches ids.
    """
    paths = sorted(p for p in Path(directory).glob("shard_*.npz") if ".tmp" not in p.name)
    if not paths:
        raise FileNotFoundError(f"no shards in {directory}")
    ids, data = [], []
    for path in paths:
        with np.load(path) as shard:
            ids.extend(shard["ids"].tolist())
            data.append(shard["data"])
    return ids, np.concatenate(data)


def done_ids(directory):
    directory = Path(directory)
    if not directory.exists():
        return set(), 0
    paths = sorted(p for p in directory.glob("shard_*.npz") if ".tmp" not in p.name)
    ids = set()
    for path in paths:
        with np.load(path) as shard:
            ids.update(shard["ids"].tolist())
    return ids, len(paths)
