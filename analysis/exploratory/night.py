import json, glob, os, sys
import numpy as np
from collections import defaultdict
R = "runs/"
def seeds_of(pattern, config):
    rows = []
    for p in sorted(glob.glob(R + pattern + "/results.json")):
        rows += json.load(open(p))[config]["seeds"]
    return rows
def vote(label, pattern, config="swd"):
    rows = seeds_of(pattern, config)
    c = np.array([r["corr"] for r in rows])
    major = np.sign(np.median(c[:, 0]))
    grp = np.sign(c[:, 0]) == major
    def first(hist, thr):
        for step, corr, swd in hist:
            if corr >= thr: return step
        return np.nan
    stats = lambda k, m: (np.mean([r[k] for r, g in zip(rows, grp) if g == m]) if (grp == m).any() else np.nan)
    h = lambda m, f: np.nanmean([f(r["history"]) for r, g in zip(rows, grp) if g == m]) if (grp == m).any() else np.nan
    at = lambda step: (lambda hist: [x[1] for x in hist if x[0] == step][0])
    swd_at = lambda step: (lambda hist: [x[2] for x in hist if x[0] == step][0])
    print(f"{label}: n {len(rows)} | axis1 |corr| {np.abs(c[:,0]).min():.2f}-{np.abs(c[:,0]).max():.2f} | majority {int(grp.sum())}/{len(rows)} | minority seeds {[r['seed'] for r, g in zip(rows, grp) if not g]}"
          f" | axis2 sign split {int((c[:,1] > 0).sum())}/{int((c[:,1] <= 0).sum())} | axis3 {int((c[:,2] > 0).sum())}/{int((c[:,2] <= 0).sum())}")
    for name, m in (("majority", True), ("minority", False)):
        if not (grp == m).any(): continue
        print(f"   {name}: r_painting {stats('r_painting', m):.3f} val_swd {stats('val_swd', m):.4f} val_energy {stats('val_energy', m):.4f} val_struct {stats('val_struct', m):.4f}"
              f" | batch corr at step 100 {h(m, at(100)):.3f}, 300 {h(m, at(300)):.3f}, 1000 {h(m, at(1000)):.3f}, 3900 {h(m, at(3900)):.3f} | steps to 0.8 {h(m, lambda x: first(x, 0.8)):.0f}, to 0.9 {h(m, lambda x: first(x, 0.9)):.0f} | batch swd at 1000 {h(m, swd_at(1000)):.4f}, 3900 {h(m, swd_at(3900)):.4f}")
    rp = np.array([r["r_painting"] for r in rows]); print(f"   r_painting over seeds: mean {rp.mean():.3f} sd {rp.std():.3f} min {rp.min():.3f} max {rp.max():.3f}")
print("== direction votes")
vote("mean, swd, 1024x3", "f1_orientation_vote_mean_*")
vote("standardised, swd, 1024x3", "f1_orientation_vote_standardised_*")
for a in ["256x3", "1024x1", "2048x3"]:
    vote(f"mean, swd, {a}", f"f1_orientation_arch_{a}_*")
vote("mean, soft_0.02_swd", "f1_orientation_votesoft_*", "soft_0.02_swd")
# same seed across settings
def signs(pattern, config="swd"):
    rows = seeds_of(pattern, config); c = np.array([r["corr"][0] for r in rows]); return {r["seed"]: int(np.sign(x) == np.sign(np.median(c))) for r, x in zip(rows, c)}
a, b = signs("f1_orientation_vote_mean_*"), signs("f1_orientation_vote_standardised_*")
print("same seed, mean vs standardised: both majority %d, both minority %d, differ %d" % (sum(a[s] and b[s] for s in a), sum((not a[s]) and (not b[s]) for s in a), sum(a[s] != b[s] for s in a)))
for arch in ["256x3", "1024x1", "2048x3"]:
    c = signs(f"f1_orientation_arch_{arch}_*"); print(f"same seed, 1024x3 vs {arch}: agree {sum(a[s] == c[s] for s in c)}/{len(c)}")

print("== sweep")
sel = json.load(open(R + "f1_batch_phase1/selection.json"))
for code in ["mean", "standardised"]:
    rows = {}
    for p in sorted(glob.glob(R + f"f1_realism_sweep_{code}_*_seed0_*/results.json")): rows.update(json.load(open(p)))
    px = lambda x: np.mean([x[k] for k in x if k.startswith("pixel_")])
    base = rows["free__swd1__global"]
    print(f"-- {code}: {len(rows)} configs | baseline r {base['painting_r']:.3f} knn {base['painting_knn']:.3f} FD {base['painting_fd']:.0f} distinct {base['distinct']} pixel {px(base):.3f} view {base['view_r']:.3f}")
    ok = {n: x for n, x in rows.items() if x["painting_r"] == x["painting_r"]}
    for title, key in (("top 8 by painting r (all)", lambda n: -ok[n]["painting_r"]), ("top 5 by kNN", lambda n: -ok[n]["painting_knn"]), ("top 5 by pixel r", lambda n: -px(ok[n])), ("lowest 5 FD", lambda n: ok[n]["painting_fd"])):
        print("  ", title)
        for n in sorted(ok, key=key)[: 8 if "top 8" in title else 5]:
            x = ok[n]; print(f"      {n:42s} r {x['painting_r']:.3f} knn {x['painting_knn']:.3f} FD {x['painting_fd']:.0f} distinct {x['distinct']} disp {x['displacement']:.3f} pixel {px(x):.3f} view {x['view_r']:.3f}")
    for i, part in enumerate(("head", "dist", "struct")):
        agg = defaultdict(list)
        for n, x in ok.items(): agg[n.split("__")[i]].append(x)
        print("   mean by", part, "| " + " | ".join(f"{k}: r {np.mean([x['painting_r'] for x in v]):.3f} knn {np.mean([x['painting_knn'] for x in v]):.3f} FD {np.mean([x['painting_fd'] for x in v]):.0f} dist {np.mean([x['distinct'] for x in v]):.0f} px {np.mean([px(x) for x in v]):.3f}" for k, v in agg.items()))
    print("   collapsed (distinct < 100):", sum(x["distinct"] < 100 for x in rows.values()), "| eligible (distinct>=300, FD<=450):", sum(x["distinct"] >= 300 and x["painting_fd"] <= 450 for x in ok.values()))
    print("   reruns, seeds 0-4 (mean, sd): ")
    for n in sel[code]:
        vals = [rows[n]]
        for s in range(1, 5):
            for p in glob.glob(R + f"f1_realism_rerun_{code}_seed{s}_*/results.json"):
                d = json.load(open(p))
                if n in d: vals.append(d[n])
        f = lambda k: (np.mean([v[k] for v in vals]), np.std([v[k] for v in vals]))
        print(f"      {n:42s} n {len(vals)} r {f('painting_r')[0]:.3f} +- {f('painting_r')[1]:.3f} | knn {f('painting_knn')[0]:.3f} +- {f('painting_knn')[1]:.3f} | FD {f('painting_fd')[0]:.0f} +- {f('painting_fd')[1]:.0f} | distinct {f('distinct')[0]:.0f}")
