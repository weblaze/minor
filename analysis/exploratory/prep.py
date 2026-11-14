"""Exploratory, not part of the repository protocol. Regenerates test images of S1 s1 and Q1 (kl 0.02),
computes DINOv2 features of all art, and saves everything to one npz."""
import sys, glob
import numpy as np, torch
REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
OUT = sys.argv[1]; ART = sys.argv[2]
sys.path.insert(0, REPO + r"\methods\q1")
from abstraction.utils.config import load_config
from abstraction.encoders.dino import DinoFeatures
from abstraction.models.mapper import Mapper
sys.path.insert(0, __import__('os').path.dirname(__file__)); from spatial_vae import SpatialVAE
from abstraction.data.datasets import ImageDataset
import common as q1
from data import apply_code_transform, load_layers, transform_from_state

dev = torch.device("cuda")
cfg = load_config(REPO + r"\methods\q1\config.yaml")
cfg["paths"]["abstract_art"] = ART
cfg["paths"]["mert_embeddings"] = REPO + r"\runs\kaggle\abstraction-features\mert_embeddings"
dino = DinoFeatures(cfg["dino"]["size"], cfg["dino"]["pooling"]).to(dev).eval()
out = {}
with torch.no_grad():
    for split in ["train", "val", "test"]:
        ds = ImageDataset(ART, size=cfg["image_size"], train=False, split=split)
        out[f"art_{split}"] = q1.features_of_dataset(dino, ds, dev).numpy()
        print(split, out[f"art_{split}"].shape, flush=True)
    K = REPO + r"\runs\kaggle"
    # Q1, kl 0.02
    ids, layers, splits = load_layers(cfg)
    vae = q1.load_vae(glob.glob(K + r"\abstraction-q1-vae\runs\q1_vae_seed0_*\vae.pt")[0], cfg, dev)
    st = torch.load(glob.glob(K + r"\abstraction-q1-seed0-vae-pair\runs\q1_mapper_q1_kl0.02_*\mapper.pt")[0], map_location=dev, weights_only=True)
    tr = transform_from_state(st["code_transform"])
    codes = apply_code_transform(layers[splits == "test"], tr).mean(axis=1)
    mapper = q1.build_mapper(cfg, codes.shape[1]).to(dev).eval(); mapper.load_state_dict(st["model"])
    imgs, feats = [], []
    for s in range(0, len(codes), 64):
        x = q1.generate(mapper, vae, torch.from_numpy(codes[s:s+64]).to(dev), st["code_mean"].to(dev), st["code_std"].to(dev))
        x = x[0] if isinstance(x, tuple) else x
        imgs.append(x.cpu()); feats.append(dino(x).float().cpu())
    out["q1_images"] = torch.cat(imgs).numpy().astype(np.float16); out["q1_feats"] = torch.cat(feats).numpy(); out["q1_codes"] = codes
    out["ids"] = np.array(ids)[splits == "test"]
    for s in ["train", "val", "test"]:
        out[f"white_{s}"] = apply_code_transform(layers[splits == s], tr)
        out[f"mean_{s}"] = layers[splits == s].astype(np.float32).mean(axis=2)
    # S1 s1
    s1cfg = load_config(REPO + r"\methods\q1\config.yaml")
    sv = torch.load(glob.glob(K + r"\abstraction-s1-vae\runs\s1_vae_seed0_*\vae.pt")[0], map_location=dev, weights_only=True)
    svae = SpatialVAE(4, 64).to(dev).eval(); svae.load_state_dict(sv["model"]); lstd = float(sv["latent_std"])
    sm = torch.load(glob.glob(K + r"\abstraction-s1-lambda\runs\s1_mapper_s1_lam1.0_*\mapper.pt")[0], map_location=dev, weights_only=True)
    m = Mapper(768, 4 * 16 * 16, 1024, 3).to(dev).eval(); m.load_state_dict(sm["model"])
    mc = out["mean_test"].mean(axis=1)
    imgs, feats = [], []
    for s in range(0, len(mc), 64):
        c = torch.from_numpy(mc[s:s+64]).to(dev)
        z = m((c - sm["code_mean"].to(dev)) / sm["code_std"].to(dev)).view(-1, 4, 16, 16)
        x = svae.decode(z * lstd)
        imgs.append(x.cpu()); feats.append(dino(x).float().cpu())
    out["s1_images"] = torch.cat(imgs).numpy().astype(np.float16); out["s1_feats"] = torch.cat(feats).numpy()
    ds = ImageDataset(ART, size=cfg["image_size"], train=False, split="test")
    out["art_test_images"] = torch.stack([ds[i] for i in range(len(ds))]).numpy().astype(np.float16)
np.savez(OUT, **out)
print("saved")
