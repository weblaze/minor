from pathlib import Path

import torch


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def warm_start(model, path, device="cpu"):
    """Load a checkpoint with strict=False, reporting key mismatches.

    Used when a model adds parameters on top of a previously trained one
    (e.g. ConditionedImageVAE warm-started from a plain ImageVAE checkpoint).
    """
    state = torch.load(path, map_location=device, weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"[warm_start] new (untrained) keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[warm_start] ignored checkpoint keys: {result.unexpected_keys}")
    return model
