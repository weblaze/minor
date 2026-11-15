import glob
import json
import sys

import numpy as np

ROOT = sys.argv[1]
KEYS = ["painting_r", "painting_knn", "painting_fd", "distinct",
        "val_painting_r", "val_painting_knn", "val_painting_fd", "val_distinct"]


def load(pattern):
    rows = {}
    for path in sorted(glob.glob(f"{ROOT}/runs/{pattern}/results.json")):
        rows.update(json.load(open(path)))
    return rows


selection = json.load(open(f"{ROOT}/runs/f1_validation_phase1/selection.json"))
for code in ["mean", "standardised"]:
    sweep = load(f"f1_realism_valsweep_{code}_*_seed0_*")
    print(f"== {code}: {len(sweep)} combinations; keys {sorted(next(iter(sweep.values())))}")
    names = list(sweep)
    val = np.array([sweep[n]["val_painting_r"] for n in names])
    test = np.array([sweep[n]["painting_r"] for n in names])
    ok = np.isfinite(val) & np.isfinite(test)
    print(f"val/test r over combinations: Pearson {np.corrcoef(val[ok], test[ok])[0, 1]:.3f}, n {ok.sum()}")
    order_val = [names[i] for i in np.argsort(-np.where(ok, val, -9))]
    order_test = [names[i] for i in np.argsort(-np.where(ok, test, -9))]
    print("top 10 by val r (unfiltered):", order_val[:10])
    print("top 10 by test r (unfiltered):", order_test[:10])
    eligible_val = [n for n in names if sweep[n]["val_distinct"] >= 300 and sweep[n]["val_painting_fd"] <= 450]
    print(f"eligible on validation: {len(eligible_val)}")
    for name in selection[code]:
        per_seed = [sweep[name]]
        for seed in range(1, 5):
            rows = load(f"f1_realism_valrerun_{code}_seed{seed}_*")
            per_seed.append(rows[name])
        line = [name, f"test-rank {order_test.index(name) + 1}", f"val-rank {order_val.index(name) + 1}"]
        for key in KEYS:
            values = np.array([row.get(key, np.nan) for row in per_seed], dtype=float)
            line.append(f"{key} {values.mean():.3f} ({values.std(ddof=1):.3f})")
        print(" | ".join(line))
