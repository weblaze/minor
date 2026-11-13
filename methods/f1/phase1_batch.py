"""F1 batch for the end of phase 1: unattended queue of mapping runs on one machine.

Units run one after another. Each unit is one invocation of realism.py or orientation.py. A unit
that has finished leaves `<unit>.done` in the state directory and is skipped when the queue is
started again, so an interrupted queue resumes where it stopped. A failing unit is recorded in
`failed.txt` and the queue continues.

Units, in order:
  vote_<code>_<n>      orientation.py, configuration swd, 10 seeds per unit, seeds 0 to 49, for the
                       mean and the standardised code (orientation majority, loss per orientation,
                       training history per seed)
  sweep_<code>_<n>     realism.py --grid sweep, 25 configurations per unit, all 100 combinations of
                       output head, distribution term and structure term, seed 0
  rerun_<code>_<seed>  the five best sweep configurations per code and the baseline, seeds 1 to 4.
                       Selection rule, fixed here: at least 300 distinct paintings and Frechet
                       distance of the painting output at most 450; ranked by r of the painting output
  arch_<h>x<d>_<n>     orientation.py on the mean code with other sizes of the mapping network,
                       20 seeds each
  votesoft_<n>         orientation.py, soft retrieval head, mean code, seeds 0 to 19

Usage:
  python methods/f1/phase1_batch.py [--state runs/f1_batch_phase1] [--list]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from realism import sweep_grid  # noqa: E402

CODES = ("mean", "standardised")
CHUNK = 25
BASELINE = "free__swd1__global"


def static_units():
    names = list(sweep_grid())
    chunks = [names[i:i + CHUNK] for i in range(0, len(names), CHUNK)]
    units = []
    for n in range(5):
        units.append((f"vote_mean_{n}", "orientation.py",
                      ["--code", "mean", "--tag", f"vote_mean_{n}", "--configs", "swd", "--seeds", "10",
                       "--first-seed", str(10 * n)]))
    for code in CODES:
        for n, chunk in enumerate(chunks):
            units.append((f"sweep_{code}_{n}", "realism.py",
                          ["--grid", "sweep", "--code", code, "--tag", f"sweep_{code}_{n}", "--seed", "0",
                           "--only", *chunk]))
    for n in range(5):
        units.append((f"vote_standardised_{n}", "orientation.py",
                      ["--code", "standardised", "--tag", f"vote_standardised_{n}", "--configs", "swd",
                       "--seeds", "10", "--first-seed", str(10 * n)]))
    return units


def late_units():
    units = []
    for hidden, depth in ((256, 3), (1024, 1), (2048, 3)):
        for n in range(2):
            units.append((f"arch_{hidden}x{depth}_{n}", "orientation.py",
                          ["--code", "mean", "--tag", f"arch_{hidden}x{depth}_{n}", "--configs", "swd",
                           "--hidden", str(hidden), "--depth", str(depth), "--seeds", "10",
                           "--first-seed", str(10 * n)]))
    for n in range(2):
        units.append((f"votesoft_{n}", "orientation.py",
                      ["--code", "mean", "--tag", f"votesoft_{n}", "--configs", "soft_0.02_swd", "--seeds", "10",
                       "--first-seed", str(10 * n)]))
    return units


def rerun_units(state):
    """Selects configurations from the finished sweep; writes the selection to selection.json."""
    selection = {}
    for code in CODES:
        rows = {}
        for path in sorted((ROOT / "runs").glob(f"f1_realism_sweep_{code}_*_seed0_*/results.json")):
            rows.update(json.load(open(path)))
        eligible = [(row["painting_r"], name) for name, row in rows.items()
                    if row["distinct"] >= 300 and row["painting_fd"] <= 450 and row["painting_r"] == row["painting_r"]]
        best = [name for _, name in sorted(eligible, reverse=True)[:5]]
        selection[code] = sorted(set(best + [BASELINE]))
    json.dump(selection, open(state / "selection.json", "w"), indent=2)
    units = []
    for code in CODES:
        for seed in range(1, 5):
            units.append((f"rerun_{code}_{seed}", "realism.py",
                          ["--grid", "sweep", "--code", code, "--tag", f"rerun_{code}", "--seed", str(seed),
                           "--only", *selection[code]]))
    return units


def run_units(units, state):
    for name, script, arguments in units:
        done = state / f"{name}.done"
        if done.exists():
            continue
        start = time.time()
        with open(state / f"{name}.log", "w") as log:
            code = subprocess.run([sys.executable, str(HERE / script), *arguments], cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - start) / 60
        with open(state / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} exit {code} {minutes:.1f} min\n")
        if code == 0:
            done.write_text(f"{minutes:.1f} min\n")
        else:
            with open(state / "failed.txt", "a") as f:
                f.write(name + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=str(ROOT / "runs" / "f1_batch_phase1"))
    parser.add_argument("--list", action="store_true", help="print the static units and exit")
    args = parser.parse_args()
    if args.list:
        for name, script, arguments in static_units() + late_units():
            print(name, script, " ".join(arguments)[:100])
        return
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        # holds off idle sleep for as long as this process lives; no system setting is changed
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    run_units(static_units(), state)
    run_units(rerun_units(state), state)
    run_units(late_units(), state)
    with open(state / "progress.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} queue finished\n")


if __name__ == "__main__":
    main()
