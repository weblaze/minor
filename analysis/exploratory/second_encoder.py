"""Exploratory. Stages: gen_s1q1 | gen_f1 [--code mean] | eval.  Run from the repository root.
Images of every set are cached in cache/<set>.npz (float16, [-1, 1], 128 px) with the audio codes used for A."""
import sys, os, glob, json, argparse, importlib.util, importlib.machinery
import numpy as np, torch

REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
K = REPO + r"\runs\kaggle"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPLORE = os.path.dirname(HERE)
Q1SRC = os.path.dirname(EXPLORE) + r"\q1src"
CACHE = HERE + r"\cache"
os.makedirs(CACHE, exist_ok=True)
MERT = REPO + r"\runs\kaggle\abstraction-features\mert_embeddings"
dev = torch.device("cuda")

Q1_SETS = {
    "q1_meancode_seed0": K + r"\abstraction-code-swap\runs\q1_mapper_q1_meancode_seed0_*",
    "q1_meancode_seed1": K + r"\abstraction-q1-mean-seeds12\runs\q1_mapper_q1_meancode_seed1_*",
    "q1_meancode_seed2": K + r"\abstraction-q1-mean-seeds12\runs\q1_mapper_q1_meancode_seed2_*",
    "q1_shuffled_meancode": K + r"\abstraction-q1-mean-controls\runs\q1_mapper_shuffled_meancode_seed0_*",
    "q1_no_feat_meancode": K + r"\abstraction-q1-mean-controls\runs\q1_mapper_no_feat_meancode_seed0_*",
    "q1_stdcode": K + r"\abstraction-std-code\runs\q1_mapper_q1_stdcode_seed0_*",
}
S1_SETS = {
    "s1_s1": K + r"\abstraction-s1-lambda\runs\s1_mapper_s1_lam1.0_seed0_*",
    "s1_s0": K + r"\abstraction-s1-seed0\runs\s1_mapper_s0_seed0_*",
    "s1_shuffled": K + r"\abstraction-s1-seed0\runs\s1_mapper_shuffled_seed0_*",
}


def save(name, images, codes, extra=None):
    np.savez(f"{CACHE}\\{name}.npz", images=images.astype(np.float16), codes=np.asarray(codes, dtype=np.float32), **(extra or {}))
    print("cached", name, images.shape, flush=True)


def gen_s1q1():
    spec = importlib.util.spec_from_file_location("abstraction.models.vector_vae", Q1SRC + "/vector_vae.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["abstraction.models.vector_vae"] = mod; spec.loader.exec_module(mod)
    sys.path.insert(0, Q1SRC)
    import common as q1
    from data import apply_code_transform, load_layers, transform_from_state
    from abstraction.utils.config import load_config
    from abstraction.models.mapper import Mapper
    sys.path.insert(0, EXPLORE); from spatial_vae import SpatialVAE
    cfg = load_config(Q1SRC + "/config.yaml")
    cfg["paths"]["mert_embeddings"] = MERT
    ids, layers, splits = load_layers(cfg)
    layers_test = layers[splits == "test"]
    print("test tracks", len(layers_test), flush=True)
    vae = q1.load_vae(glob.glob(K + r"\abstraction-q1-vae\runs\q1_vae_seed0_*\vae.pt")[0], cfg, dev)
    with torch.no_grad():
        for name, pattern in Q1_SETS.items():
            if os.path.exists(f"{CACHE}\\{name}.npz"):
                continue
            st = torch.load(glob.glob(pattern + r"\mapper.pt")[0], map_location=dev, weights_only=True)
            codes = apply_code_transform(layers_test, transform_from_state(st["code_transform"])).mean(axis=1)
            mapper = q1.build_mapper(cfg, codes.shape[1]).to(dev).eval(); mapper.load_state_dict(st["model"])
            imgs = []
            for s in range(0, len(codes), 64):
                x = q1.generate(mapper, vae, torch.from_numpy(codes[s:s + 64]).to(dev), st["code_mean"].to(dev), st["code_std"].to(dev))
                x = x[0] if isinstance(x, tuple) else x
                imgs.append(x.float().cpu())
            save(name, torch.cat(imgs).numpy(), codes)
        sv = torch.load(glob.glob(K + r"\abstraction-s1-vae\runs\s1_vae_seed0_*\vae.pt")[0], map_location=dev, weights_only=True)
        svae = SpatialVAE(4, 64).to(dev).eval(); svae.load_state_dict(sv["model"]); lstd = float(sv["latent_std"])
        mc = layers_test.astype(np.float32).mean(axis=2).mean(axis=1)
        for name, pattern in S1_SETS.items():
            if os.path.exists(f"{CACHE}\\{name}.npz"):
                continue
            sm = torch.load(glob.glob(pattern + r"\mapper.pt")[0], map_location=dev, weights_only=True)
            m = Mapper(768, 4 * 16 * 16, 1024, 3).to(dev).eval(); m.load_state_dict(sm["model"])
            imgs = []
            for s in range(0, len(mc), 64):
                c = torch.from_numpy(mc[s:s + 64]).to(dev)
                z = m((c - sm["code_mean"].to(dev)) / sm["code_std"].to(dev)).view(-1, 4, 16, 16)
                imgs.append(svae.decode(z * lstd).float().cpu())
            save(name, torch.cat(imgs).numpy(), mc)


def gen_f1(code_variant):
    f1dir = REPO + r"\methods\f1"
    sys.path.insert(0, f1dir)
    import realism
    common = sys.modules["common"]
    from abstraction.utils.config import load_config
    config = load_config(f1dir + r"\config.yaml")
    config["paths"]["mert_embeddings"] = MERT
    print("image_size", config["image_size"], "steps", config["mapper"]["steps"], flush=True)
    seed = 0
    _, layers, splits = common.load_layers(config)
    art_names, art, art_splits = common.load_art(config, dev)
    gallery = torch.from_numpy(art[art_splits == "train"]).to(dev)
    gallery_names = art_names[art_splits == "train"]
    code = config["audio_code"]
    codes, _ = common.codes_of(layers, splits, code_variant, code["whiten_dim"], code["layers"])
    from realism import VIEWS
    view_codes = [common.codes_of(layers, splits, "standardised", code["whiten_dim"], [i])[0] for i in VIEWS]
    to = lambda x: torch.from_numpy(x).to(dev)
    train, test_codes = to(codes[splits == "train"]), codes[splits == "test"].mean(axis=1)
    test = to(test_codes)
    train_views = [to(v[splits == "train"]) for v in view_codes]
    grid, sweep = realism.GRID, realism.sweep_grid()
    jobs = {"f1_swd": (grid["swd"], config["mapper"]["steps"]),
            "f1_soft02_swd1_chamfer01_global3": (sweep["soft02__swd1_chamfer01__global3"], config["mapper"]["steps"]),
            "f1_untrained": (grid["swd"], 0)}
    for name, (g, steps) in jobs.items():
        spec = {**realism.DEFAULT, **g}
        forward = realism.fit(spec, config, train, train_views, gallery, dev, steps, seed)
        with torch.no_grad():
            out, query = forward(test)
            nearest = (torch.nn.functional.normalize(query, dim=1) @ torch.nn.functional.normalize(gallery, dim=1).T).argmax(1).cpu().numpy()
        imgs = common.load_images(config, gallery_names[nearest]).numpy()
        print(name, "distinct", len(np.unique(nearest)), flush=True)
        save(f"{name}__{code_variant}", imgs, test_codes)
    test_names = art_names[art_splits == "test"]
    np.savez(f"{CACHE}\\art_test.npz", images=common.load_images(config, test_names).numpy().astype(np.float16))


class Enc:
    def __init__(self, name):
        self.name = name
        from transformers import ViTModel, ViTMAEModel
        if name == "dinov2":
            from abstraction.encoders.dino import DinoFeatures
            self.m = DinoFeatures(112, "cls").to(dev).eval()
        elif name == "mae":
            self.m = ViTMAEModel.from_pretrained("facebook/vit-mae-base", mask_ratio=0.0).to(dev).eval()
        elif name == "dino_v1":
            self.m = ViTModel.from_pretrained("facebook/dino-vitb16", add_pooling_layer=False, revision="refs/pr/1").to(dev).eval()
        else:
            raise ValueError(name)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images):
        out = []
        for s in range(0, len(images), 32):
            x = torch.from_numpy(images[s:s + 32].astype(np.float32)).to(dev)
            if self.name == "dinov2":
                f = self.m(x)
            else:
                x = torch.nn.functional.interpolate((x + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
                h = self.m(pixel_values=(x - self.mean) / self.std).last_hidden_state
                f = h[:, 1:].mean(1) if self.name == "mae" else h[:, 0]
            out.append(f.float().cpu())
        return torch.cat(out).numpy()


KNOWN = {"s1_s1": 0.861, "s1_s0": 0.083, "s1_shuffled": 0.140, "q1_meancode_seed0": 0.825, "q1_meancode_seed1": 0.815,
         "q1_meancode_seed2": 0.818, "q1_shuffled_meancode": 0.209, "q1_no_feat_meancode": 0.869, "q1_stdcode": 0.797}


def evaluate(f1_code):
    from abstraction.metrics.structure import distance_matrix, knn_overlap
    from abstraction.metrics.geometry import participation_ratio
    up = lambda m: m[np.triu_indices(len(m), 1)]
    sets = {n: dict(np.load(f"{CACHE}\\{n}.npz")) for n in list(S1_SETS) + list(Q1_SETS)}
    for n in ["f1_swd", "f1_soft02_swd1_chamfer01_global3", "f1_untrained"]:
        sets[n] = dict(np.load(f"{CACHE}\\{n}__{f1_code}.npz"))
    if os.path.exists(CACHE + "\\f1_untrained__standardised.npz"):
        sets["f1_untrained_standardised_code"] = dict(np.load(CACHE + "\\f1_untrained__standardised.npz"))
        KNOWN["f1_untrained_standardised_code"] = 0.066
    KNOWN.update({"f1_swd": 0.645, "f1_soft02_swd1_chamfer01_global3": 0.737, "f1_untrained": 0.07})
    TOL = {"f1_soft02_swd1_chamfer01_global3": 0.02, "f1_untrained": 0.02, "f1_untrained_standardised_code": 0.02}
    art_test = np.load(f"{CACHE}\\art_test.npz")["images"]
    dA = {n: distance_matrix(s["codes"]) for n, s in sets.items()}
    results = {n: {"known_dinov2": KNOWN[n]} for n in sets}
    pr = {}
    ok = {}
    for enc_name in ["dinov2", "mae", "dino_v1"]:
        try:
            enc = Enc(enc_name)
        except Exception as e:
            print("FAILED to load", enc_name, repr(e)[:300], flush=True)
            for n in sets: results[n][enc_name] = {"error": repr(e)[:200]}
            pr[enc_name] = None
            continue
        for n, s in sets.items():
            if enc_name != "dinov2" and not ok[n]:
                continue
            f = enc(s["images"])
            d = distance_matrix(f)
            r = float(np.corrcoef(up(dA[n]), up(d))[0, 1]); kn = float(knn_overlap(dA[n], d, 10)[0])
            results[n][enc_name] = {"r": r, "knn": kn}
            if enc_name == "dinov2":
                ok[n] = abs(r - KNOWN[n]) <= TOL.get(n, 0.01)
                print(f"sanity {n}: r={r:.3f} known={KNOWN[n]} {'OK' if ok[n] else 'MISMATCH'}", flush=True)
            else:
                print(enc_name, n, f"r={r:.3f} knn={kn:.3f}", flush=True)
        pr[enc_name] = participation_ratio(enc(art_test))
        print(enc_name, "PR test art", pr[enc_name], flush=True)
        del enc; torch.cuda.empty_cache()
    json.dump({"f1_code": f1_code, "results": results, "participation_ratio_test_art": pr}, open(HERE + r"\results.json", "w"), indent=1)
    cell = lambda e, k: "n/a" if not isinstance(e, dict) or k not in e else f"{e[k]:.3f}"
    lines = ["| image set | known DINOv2 r | DINOv2 r | DINOv2 kNN | MAE r | MAE kNN | DINO v1 r | DINO v1 kNN |", "|---|---|---|---|---|---|---|---|"]
    for n, row in results.items():
        lines.append(f"| {n} | {row['known_dinov2']} | " + " | ".join(cell(row.get(e), k) for e in ["dinov2", "mae", "dino_v1"] for k in ["r", "knn"]) + " |")
    lines += ["", "Participation ratio of features on the 595 test paintings: " + ", ".join(f"{k} {v:.1f}" if v else f"{k} n/a" for k, v in pr.items())]
    open(HERE + r"\results.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("stage"); p.add_argument("--code", default="mean")
    a = p.parse_args()
    {"gen_s1q1": gen_s1q1, "gen_f1": lambda: gen_f1(a.code), "eval": lambda: evaluate(a.code)}[a.stage]()
