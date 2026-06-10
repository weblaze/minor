import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(approach_config=None):
    """Load configs/base.yaml, deep-merged with an optional per-approach YAML.

    Relative entries under `paths` are resolved against the repo root, so
    scripts work no matter which directory they are launched from.
    """
    with open(BASE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    if approach_config is not None:
        with open(approach_config, "r") as f:
            override = yaml.safe_load(f) or {}
        config = _deep_merge(config, override)

    for key, value in config.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        config["paths"][key] = str(path)

    return config
