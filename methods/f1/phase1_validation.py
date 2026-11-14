"""F1: combination sweep with selection on the validation split.

The batch of 2026-11-14 (`phase1_batch.py`) selected the combinations to rerun by their scores on
the test tracks. This queue repeats the sweep with validation scores recorded, selects on the
validation tracks, and reruns the selected combinations with further seeds; test scores of the
selected combinations are then free of selection on test data.

Units, in order:
  valsweep_<code>_<n>     realism.py --grid sweep, 25 configurations per unit, seed 0
  valrerun_<code>_<seed>  the five best combinations per code and the baseline, seeds 1 to 4.
                          Selection rule, on validation tracks only: at least 300 distinct
                          paintings, Frechet distance of the painting output to validation art at
                          most 450, ranked by r of the painting output

The state directory holds one `.done` marker and one log per unit, `selection.json` and
`progress.txt`. Restarting the script skips finished units.

Usage:
  python methods/f1/phase1_validation.py [--state runs/f1_validation_phase1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from phase1_batch import BASELINE, CHUNK, CODES, run_units  # noqa: E402
from realism import sweep_grid  # noqa: E402


def sweep_units():
    names = list(sweep_grid())
    chunks = [names[i:i + CHUNK] for i in range(0, len(names), CHUNK)]
    return [(f"valsweep_{code}_{n}", "realism.py",
             ["--grid", "sweep", "--code", code, "--tag", f"valsweep_{code}_{n}", "--seed", "0", "--only", *chunk])
            for code in CODES for n, chunk in enumerate(chunks)]


def rerun_units(state):
    selection = {}
    for code in CODES:
        rows = {}
        for path in sorted((ROOT / "runs").glob(f"f1_realism_valsweep_{code}_*_seed0_*/results.json")):
            rows.update(json.load(open(path)))
        eligible = [(row["val_painting_r"], name) for name, row in rows.items()
                    if row["val_distinct"] >= 300 and row["val_painting_fd"] <= 450
                    and row["val_painting_r"] == row["val_painting_r"]]
        best = [name for _, name in sorted(eligible, reverse=True)[:5]]
        selection[code] = sorted(set(best + [BASELINE]))
    json.dump(selection, open(state / "selection.json", "w"), indent=2)
    return [(f"valrerun_{code}_{seed}", "realism.py",
             ["--grid", "sweep", "--code", code, "--tag", f"valrerun_{code}", "--seed", str(seed),
              "--only", *selection[code]])
            for code in CODES for seed in range(1, 5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=str(ROOT / "runs" / "f1_validation_phase1"))
    args = parser.parse_args()
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    run_units(sweep_units(), state)
    run_units(rerun_units(state), state)
    with open(state / "progress.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} queue finished\n")


if __name__ == "__main__":
    main()
