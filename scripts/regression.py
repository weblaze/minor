"""Numerical regression check over the smoke pipelines of every method.

Runs the command lists of scripts/smoke.py (seed 0, synthetic data) on the CPU, collects every
numeric value from the result files the runs write, and either saves them as a baseline or
compares them against a saved baseline. A refactoring step must leave every value unchanged.

  results.json, diagnosis.json, gate.json, diagnostics.json, result.json   every number, recursively
  metrics.jsonl                                                            the last line

Values are keyed by method, step number, script, run name (without the time stamp), file and
position in the file. Timing fields (elapsed_s) are ignored. The CPU is forced (CUDA_VISIBLE_DEVICES
is set to -1) because GPU kernels are not bit-reproducible between runs.

Usage:
  python scripts/regression.py --save runs/regression_baseline.json [--methods s1 q1 f1]
  python scripts/regression.py --compare runs/regression_baseline.json [--rtol 1e-5]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke import PIPELINES, ROOT, RUNS, newest  # noqa: E402

RESULT_FILES = ("results.json", "diagnosis.json", "gate.json", "diagnostics.json", "result.json")
IGNORED_KEYS = {"elapsed_s"}
STAMP = re.compile(r"_\d{8}T\d{6}Z$")


def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in IGNORED_KEYS:
                flatten(value, f"{prefix}.{key}", out)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            flatten(value, f"{prefix}[{i}]", out)
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)


def values_of(run, prefix):
    out = {}
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run).as_posix()
        if path.name in RESULT_FILES:
            with open(path) as f:
                flatten(json.load(f), f"{prefix}/{relative}", out)
        elif path.name == "metrics.jsonl":
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            if lines:
                flatten(json.loads(lines[-1]), f"{prefix}/{relative}:last", out)
    return out


def run_pipelines(methods):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(ROOT)}
    RUNS.mkdir(exist_ok=True)
    values = {}
    for method in methods:
        for step, (script, arguments) in enumerate(PIPELINES[method]):
            path = ROOT / "methods" / method / script
            if not path.exists():
                continue
            resolved = [a if "{" not in a else newest(a[1:a.index("}")]) + a[a.index("}") + 1:] for a in arguments]
            before = set(RUNS.iterdir())
            start = time.time()
            code = subprocess.run([sys.executable, str(path), "--smoke", "--seed", "0", *resolved], cwd=ROOT,
                                  env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if code.returncode != 0:
                print(code.stderr[-2000:])
                raise SystemExit(f"{method} {script} failed")
            new = sorted(set(RUNS.iterdir()) - before)
            for run in new:
                values.update(values_of(run, f"{method}/{step:02d}_{script}/{STAMP.sub('', run.name)}"))
            print(f"{method} {script:18s} {time.time() - start:6.1f} s  {len(new)} run(s)", flush=True)
    return values


def same(a, b, rtol):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= rtol * max(abs(a), abs(b)) or a == b


def compare(baseline, current, rtol):
    differences = 0
    for key in sorted(set(baseline) | set(current)):
        if key not in current:
            print(f"missing  {key} (baseline {baseline[key]!r})")
        elif key not in baseline:
            print(f"new      {key} = {current[key]!r}")
        elif not same(baseline[key], current[key], rtol):
            print(f"changed  {key}: {baseline[key]!r} -> {current[key]!r}")
        else:
            continue
        differences += 1
    shared = len(set(baseline) & set(current))
    print(f"{shared} values compared, {differences} differences")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--save", metavar="FILE")
    mode.add_argument("--compare", metavar="FILE")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        methods = args.methods or baseline["methods"]
    else:
        methods = args.methods or [m for m in PIPELINES if (ROOT / "methods" / m).is_dir()]
    current = run_pipelines(methods)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump({"methods": methods, "values": current}, f, indent=1, sort_keys=True)
        print(f"{len(current)} values -> {args.save}")
        return
    expected = {k: v for k, v in baseline["values"].items() if k.split("/")[0] in methods}
    if compare(expected, current, args.rtol):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
