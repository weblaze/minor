"""Run the smoke pipeline of every method present on the current branch.

Each method is run end to end with `--smoke` (synthetic data, randomly initialised feature
extractor, a few steps). A step that depends on an earlier output receives the newest matching
run directory. The script stops at the first failure and exits non-zero.

Usage:
  python scripts/smoke.py [--methods s1 q1 f1]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

# (script, arguments); "{name}" is replaced by the newest runs/<name>*/ directory
PIPELINES = {
    "s1": [("train_vae.py", []),
           ("train_mapper.py", ["--vae", "{s1_vae_smoke}/vae.pt"]),
           ("evaluate.py", ["--vae", "{s1_vae_smoke}/vae.pt", "--mapper", "{s1_mapper_s1_smoke}/mapper.pt"]),
           ("evaluate.py", ["--vae", "{s1_vae_smoke}/vae.pt"])],
    "q1": [("diagnostics.py", []),
           ("train_vae.py", []),
           ("gate.py", ["--vae", "{q1_vae_smoke}/vae.pt"]),
           ("baselines.py", ["--vae", "{q1_vae_smoke}/vae.pt"]),
           ("train_mapper.py", ["--vae", "{q1_vae_smoke}/vae.pt", "--ignore-gate"]),
           ("evaluate.py", ["--vae", "{q1_vae_smoke}/vae.pt", "--mapper", "{q1_mapper_q1_smoke}/mapper.pt"])],
    "f1": [("train_mapper.py", []),
           ("evaluate.py", ["--mapper", "{f1_mapper_f1_smoke}/mapper.pt"]),
           ("evaluate.py", []),
           ("diagnose.py", []),
           ("realism.py", ["--only", "swd", "soft_0.02_swd"]),
           ("orientation.py", ["--seeds", "2"])],
}


def newest(prefix):
    matches = sorted(RUNS.glob(prefix + "*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"no run directory matching {prefix}*")
    return str(matches[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args()
    methods = args.methods or [m for m in PIPELINES if (ROOT / "methods" / m).is_dir()]
    if not methods:
        raise SystemExit("no method directory on this branch")
    summary = []
    for method in methods:
        for script, arguments in PIPELINES[method]:
            path = ROOT / "methods" / method / script
            if not path.exists():
                summary.append((method, script, "absent", 0.0))
                continue
            resolved = [a if "{" not in a else newest(a[1:a.index("}")]) + a[a.index("}") + 1:] for a in arguments]
            start = time.time()
            code = subprocess.run([sys.executable, str(path), "--smoke", "--seed", "0", *resolved], cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            seconds = time.time() - start
            summary.append((method, script, "ok" if code.returncode == 0 else "FAILED", seconds))
            if code.returncode != 0:
                for row in summary:
                    print("%-3s %-18s %-7s %6.1f s" % row)
                print(code.stderr[-2000:])
                raise SystemExit(1)
    for row in summary:
        print("%-3s %-18s %-7s %6.1f s" % row)


if __name__ == "__main__":
    main()
