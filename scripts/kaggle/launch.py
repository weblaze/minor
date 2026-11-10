"""Push a job to Kaggle, query its state, or download its output.

A job is a YAML file (scripts/kaggle/jobs/) with the keys
  name       kernel title; the kernel slug is derived from it
  branch     git branch whose pushed head commit is run
  gpu        true or false
  datasets   Kaggle datasets to mount (owner/slug)
  kernels    Kaggle kernels whose output is mounted (owner/slug; `{user}` is replaced)
  inputs     NAME: file or directory name searched under /kaggle/input;
             `{NAME}` in a command is replaced by the path found
  commands   shell commands run in order from the repository root

The commit is read from origin/<branch>, so the branch has to be pushed first.
The Kaggle user is taken from --user, KAGGLE_USERNAME, or ~/.kaggle/kaggle.json.
Requires the `kaggle` command-line client and an API token.

Usage:
  python scripts/kaggle/launch.py push   scripts/kaggle/jobs/<job>.yaml [--dry-run DIR]
  python scripts/kaggle/launch.py status scripts/kaggle/jobs/<job>.yaml
  python scripts/kaggle/launch.py fetch  scripts/kaggle/jobs/<job>.yaml   (output to runs/kaggle/<slug>/)
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

from abstraction.utils.config import REPO_ROOT

REPOSITORY = "https://github.com/weblaze/minor.git"
KERNEL = Path(__file__).with_name("kernel.py")


def kaggle_user(explicit):
    if explicit:
        return explicit
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.exists():
        with open(token) as f:
            return json.load(f)["username"]
    raise SystemExit("Kaggle user unknown: pass --user or set KAGGLE_USERNAME")


def slug_of(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def push(job, user, dry_run=None):
    git("fetch", "--quiet", "origin", job["branch"])
    commit = git("rev-parse", f"origin/{job['branch']}")
    payload = {"name": job["name"], "repository": REPOSITORY, "branch": job["branch"], "commit": commit,
               "inputs": job.get("inputs", {}), "commands": job["commands"]}
    metadata = {
        "id": f"{user}/{slug_of(job['name'])}",
        "title": job["name"],
        "code_file": "job.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if job.get("gpu") else "false",
        "enable_internet": "true",
        "dataset_sources": job.get("datasets", []),
        "kernel_sources": [k.format(user=user) for k in job.get("kernels", [])],
        "competition_sources": [],
        "model_sources": [],
    }
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(dry_run or folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "job.py", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"JOB = {payload!r}\n" + KERNEL.read_text(encoding="utf-8"))
        with open(folder / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        if dry_run:
            print(f"wrote {folder} for {metadata['id']} at {job['branch']} {commit}; nothing pushed")
            return
        subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)], check=True)
    print(f"pushed {metadata['id']} at {job['branch']} {commit}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("push", "status", "fetch"))
    parser.add_argument("job")
    parser.add_argument("--user", default=None)
    parser.add_argument("--dry-run", default=None, metavar="DIR",
                        help="with push: write job.py and kernel-metadata.json to DIR and do not contact Kaggle")
    args = parser.parse_args()
    with open(args.job) as f:
        job = yaml.safe_load(f)
    user = kaggle_user(args.user)
    kernel_id = f"{user}/{slug_of(job['name'])}"

    if args.action == "push":
        push(job, user, args.dry_run)
    elif args.action == "status":
        subprocess.run(["kaggle", "kernels", "status", kernel_id], check=True)
    else:
        out = REPO_ROOT / "runs" / "kaggle" / slug_of(job["name"])
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(out)], check=True)
        print(f"output in {out}")


if __name__ == "__main__":
    main()
