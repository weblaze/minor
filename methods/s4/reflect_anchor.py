"""B-anchor by code reflection: choose the orientation of an already trained drawing mapper with CAV-MAE.

The mapper input is the standardised audio code x = (code - code_mean) / code_std. Reflecting x along a
leading principal axis u of the training codes (x' = x - 2 (x . u) u) keeps every distance between tracks,
so the music structure the mapper is given is unchanged; only which way round the axis is drawn changes.
For each of the 2^k sign patterns of the top `--axes` axes, every track gets one image (mean of its window
codes) and CAV-MAE scores how well images pair with their own sound among all tracks of the split
(as methods/s4/bridge_identify.py; centres: mean CAV-MAE embeddings of the training tracks and paintings).
The pattern is chosen on validation tracks (largest own-minus-other similarity) and reported on test tracks.

A reflection is linear, so it is folded into the first layer of the mapper: W' = W R. The chosen pattern is
saved as `mapper.pt` in the format of methods/s1 (methods/s1/evaluate.py and methods/s3/second_reader.py
read it unchanged). Also reported: skew of the training codes along each axis (a reflection along a skewed
axis moves inputs to regions the mapper never saw).

Usage:
  python methods/s4/reflect_anchor.py --vae <gan run>/generator.pt --run runs/<trained mapper run> [--axes 3]
"""
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.encoders.cav_mae import IMAGE_MEAN, IMAGE_STD, load_cav_mae
from abstraction.generators import add_generator_argument, load_generator
from abstraction.models.mapper import Mapper
from abstraction.pipeline.cli import setup
from abstraction.utils.runlog import RunLog, set_seed

from bridge import to_cav
from common import aligned_codes, base_parser, cav_inputs, image_centre, run_name


def load_mapper(state, device):
    weights = state["model"]
    first = weights["blocks.0.net.0.weight"]
    depth = sum(1 for k in weights if k.startswith("blocks.") and k.endswith("net.0.weight"))
    mapper = Mapper(first.shape[1], weights["out.weight"].shape[0], first.shape[0], depth).to(device).eval()
    mapper.load_state_dict(weights)
    return mapper


def main():
    parser = add_generator_argument(base_parser(__doc__), "gan")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run", required=True, help="run directory with mapper.pt (S1 format)")
    parser.add_argument("--axes", type=int, default=3)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    state = torch.load(Path(args.run) / "mapper.pt", map_location=device, weights_only=True)
    config["audio_code"]["variant"] = state["code"]
    embeddings = args.embeddings or config["cav_mae"]["embeddings"]
    config["run"] = {"mapper_run": args.run, "vae": str(args.vae), "axes": args.axes}
    log = RunLog(config["paths"]["runs_dir"], run_name("reflect_anchor", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    cav = load_cav_mae(args.checkpoint or config["cav_mae"]["checkpoint"], device)
    cav_audio, cav_images, cav_names = cav_inputs(embeddings, args.smoke, cav)
    mapper = load_mapper(state, device)
    code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
    mean = torch.tensor(IMAGE_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGE_STD, device=device)[None, :, None, None]

    train_codes, train_audio = aligned_codes(config, "train", cav_audio, False)
    x_train = ((torch.from_numpy(train_codes.mean(axis=1)).to(device) - code_mean) / code_std).double().cpu().numpy()
    centred = x_train - x_train.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(centred, rowvar=False))
    axes = eigvec[:, np.argsort(eigval)[::-1][:args.axes]]
    proj = centred @ axes
    skew = ((proj - proj.mean(0)) ** 3).mean(0) / proj.std(0) ** 3
    centre_a = torch.from_numpy(train_audio.mean(axis=(0, 1))).float().to(device)
    centre_v = torch.from_numpy(image_centre(config, cav_images, cav_names, False)).to(device)

    def reflection(signs):
        r = np.eye(axes.shape[0])
        for k, s in enumerate(signs):
            if s < 0:
                r -= 2 * np.outer(axes[:, k], axes[:, k])
        return torch.from_numpy(r).float().to(device)

    split_data = {split: aligned_codes(config, split, cav_audio, False) for split in ("val", "test")}

    @torch.no_grad()
    def judge(split, r):
        codes, audio = split_data[split]
        u = F.normalize(torch.from_numpy(audio.mean(axis=1)).to(device) - centre_a, dim=1)
        x = (torch.from_numpy(codes.mean(axis=1)).to(device) - code_mean) / code_std
        v = []
        for start in range(0, len(x), 32):
            z = mapper(x[start:start + 32] @ r.T)
            images = generator.decode(z.view(-1, *generator.latent_shape) * generator.latent_scale)
            v.append(F.normalize(cav.embed_image(to_cav(images, mean, std)).float() - centre_v, dim=1))
        sims = torch.cat(v) @ u.T
        n = len(sims)
        own = sims.diagonal()
        rank = (sims > own[:, None]).sum(1) + 1
        other = (sims.sum() - own.sum()) / (n * n - n)
        return {"gap": float(own.mean() - other), "median_rank": float(rank.float().median()),
                "top10": float((rank <= 10).float().mean()), "n": n}

    patterns = list(itertools.product((1, -1), repeat=args.axes))
    rows = []
    for signs in patterns:
        r = reflection(signs)
        row = {"signs": list(signs), "val": judge("val", r), "test": judge("test", r)}
        rows.append(row)
        print(signs, "val gap {:.4f} rank {:.0f} | test gap {:.4f} rank {:.0f}".format(
            row["val"]["gap"], row["val"]["median_rank"], row["test"]["gap"], row["test"]["median_rank"]), flush=True)
    best = max(range(len(rows)), key=lambda i: rows[i]["val"]["gap"])
    val_gaps = [row["val"]["gap"] for row in rows]
    test_gaps = [row["test"]["gap"] for row in rows]
    results = {"mapper_run": args.run, "skew_of_axes": skew.tolist(), "patterns": rows,
               "chosen": rows[best]["signs"], "original_is_chosen": best == 0,
               "val_test_gap_corr": float(np.corrcoef(val_gaps, test_gaps)[0, 1]),
               "test_rank_of_chosen": int(sorted(test_gaps, reverse=True).index(test_gaps[best]) + 1)}
    log.save_json("results.json", results)

    chosen = {**state, "model": dict(state["model"])}
    chosen["model"]["blocks.0.net.0.weight"] = state["model"]["blocks.0.net.0.weight"].to(device) @ reflection(rows[best]["signs"])
    chosen["reflection_signs"] = rows[best]["signs"]
    torch.save(chosen, log.dir / "mapper.pt")
    print({k: v for k, v in results.items() if k != "patterns"})
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
