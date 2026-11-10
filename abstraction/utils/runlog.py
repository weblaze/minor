import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import REPO_ROOT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RunLog:
    """One directory per run: config.yaml, meta.json, metrics.jsonl.

    meta.json records the git commit, whether the working tree had uncommitted
    changes, the seed, the software versions and the device, so that a run can
    be traced to the exact code that produced it.
    """

    def __init__(self, root, name, config, seed):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = config.get("system", {}).get("tag")
        if tag:
            name = f"{name}_{tag}"
        self.dir = Path(root) / f"{name}_seed{seed}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=False)
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        status = _git("status", "--porcelain")
        meta = {
            "name": name,
            "seed": seed,
            "started_utc": stamp,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": None if status is None else bool(status),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
        }
        with open(self.dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        self._start = time.time()

    def log(self, **values):
        row = {"elapsed_s": round(time.time() - self._start, 1)}
        for key, value in values.items():
            row[key] = value.item() if hasattr(value, "item") else value
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_json(self, filename, obj):
        with open(self.dir / filename, "w") as f:
            json.dump(obj, f, indent=2)
