"""Exploratory: is the genre -> painting-style correspondence the same across mapper seeds?"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "methods/f1")
import common as f1, realism
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix

dev = torch.device("cuda")
cfg = load_config("methods/f1/config.yaml")
ids, layers, splits = f1.load_layers(cfg)
names, art, art_splits = f1.load_art(cfg, dev)
gal_np, gal_names = art[art_splits == "train"], names[art_splits == "train"]
gallery = torch.from_numpy(gal_np).to(dev)
unit = gal_np / np.linalg.norm(gal_np, axis=1, keepdims=True)
tracks = pd.read_csv(sys.argv[1] + "/tracks.csv", index_col=0, header=[0, 1])
genre = tracks.loc[[int(i) for i in ids[splits == "test"]], ("track", "genre_top")].astype(str).values
styles = np.array([n.split("__")[0] for n in gal_names])
calm = np.isin(styles, ["Color_Field_Painting", "Minimalism"])
up = lambda m: m[np.triu_indices(len(m), 1)]
for code in ["mean", "standardised"]:
    codes, _ = f1.codes_of(layers, splits, code, 128)
    train = torch.from_numpy(codes[splits == "train"]).to(dev)
    test = torch.from_numpy(codes[splits == "test"].mean(axis=1)).to(dev)
    da = up(distance_matrix(test.cpu().numpy()))
    picks = []
    for seed in range(5):
        fwd = realism.fit({**realism.DEFAULT}, cfg, train, [], gallery, dev, cfg["mapper"]["steps"], seed)
        with torch.no_grad():
            out, q = fwd(test)
            near = (torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(gallery, dim=1).T).argmax(1).cpu().numpy()
        picks.append(near)
        share = pd.Series(calm[near]).groupby(genre).mean()
        r = float(np.corrcoef(da, up(distance_matrix(gal_np[near])))[0, 1])
        print(code, "seed", seed, "r %.3f" % r, "| share of Colour Field + Minimalism by genre:", {k: round(v, 2) for k, v in share.items()}, flush=True)
    sims, same = [], []
    rng = np.random.default_rng(0)
    for a in range(5):
        for b in range(a + 1, 5):
            sims.append(float((unit[picks[a]] * unit[picks[b]]).sum(1).mean()))
            same.append(float((picks[a] == picks[b]).mean()))
    base = float((unit[picks[0]] * unit[rng.permutation(picks[1])]).sum(1).mean())
    print(code, "between seeds: mean cosine similarity of the two paintings of a track %.3f (shuffled tracks %.3f); same painting rate %.3f"
          % (np.mean(sims), base, np.mean(same)), flush=True)
