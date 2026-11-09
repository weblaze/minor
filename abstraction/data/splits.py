import hashlib

SPLITS = ("train", "val", "test")


def split_of(item_id, val_percent=10, test_percent=10):
    """Deterministic split assignment from the MD5 hash of an identifier.

    The assignment depends only on the identifier, so it is identical across
    machines, runs and subsets of the data.
    """
    bucket = int(hashlib.md5(str(item_id).encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + val_percent:
        return "val"
    return "train"


def split_indices(item_ids, val_percent=10, test_percent=10):
    out = {name: [] for name in SPLITS}
    for i, item_id in enumerate(item_ids):
        out[split_of(item_id, val_percent, test_percent)].append(i)
    return out
