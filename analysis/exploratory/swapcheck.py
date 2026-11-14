"""Exploratory: structure of the Q1 mean-code outputs under features other than DINOv2."""
import sys, glob, importlib.util
import numpy as np, torch
SRC = sys.argv[1]; REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
spec = importlib.util.spec_from_file_location("abstraction.models.vector_vae", SRC + "/vector_vae.py")
mod = importlib.util.module_from_spec(spec); sys.modules["abstraction.models.vector_vae"] = mod; spec.loader.exec_module(mod)
sys.path.insert(0, SRC)
import common as q1
from data import apply_code_transform, load_layers, transform_from_state
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix
import importlib
f1 = importlib.machinery.SourceFileLoader("f1common", REPO + r"\methods\f1\common.py").load_module()
dev = torch.device("cuda")
cfg = load_config(SRC + "/config.yaml")
ids, layers, splits = load_layers(cfg)
dino = q1.build_dino(cfg, dev, False)
K = REPO + r"\runs\kaggle"
vae = q1.load_vae(glob.glob(K + r"\abstraction-q1-vae\runs\q1_vae_seed0_*\vae.pt")[0], cfg, dev)
up = lambda m: m[np.triu_indices(len(m), 1)]
for label, path in [("q1 mean code", K + r"\abstraction-code-swap\runs\q1_mapper_q1_meancode_*\mapper.pt"),
                    ("q1 whitened code", K + r"\abstraction-q1-seed0-vae-pair\runs\q1_mapper_q1_kl0.02_*\mapper.pt")]:
    st = torch.load(glob.glob(path)[0], map_location=dev, weights_only=True)
    codes = apply_code_transform(layers[splits == "test"], transform_from_state(st["code_transform"])).mean(axis=1)
    mapper = q1.build_mapper(cfg, codes.shape[1]).to(dev).eval(); mapper.load_state_dict(st["model"])
    imgs, feats = [], []
    with torch.no_grad():
        for s in range(0, len(codes), 64):
            x = q1.generate(mapper, vae, torch.from_numpy(codes[s:s+64]).to(dev), st["code_mean"].to(dev), st["code_std"].to(dev))
            x = x[0] if isinstance(x, tuple) else x
            imgs.append(x.cpu().float()); feats.append(dino(x).float().cpu())
    imgs = torch.cat(imgs); feats = torch.cat(feats).numpy()
    da = up(distance_matrix(codes))
    row = {"dinov2": float(np.corrcoef(da, up(distance_matrix(feats)))[0, 1])}
    for k, v in f1.image_statistics(imgs).items():
        row[k] = float(np.corrcoef(da, up(distance_matrix(v, metric="euclidean")))[0, 1])
    print(label, {k: round(v, 3) for k, v in row.items()}, flush=True)
    if label == "q1 mean code":
        from torchvision.utils import save_image
        save_image(imgs[:64] * 0.5 + 0.5, sys.argv[2], nrow=8)
