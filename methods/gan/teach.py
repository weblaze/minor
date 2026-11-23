"""Picker to drawer: train a mapper (audio code -> G-GAN latent) to draw what a picker describes.

Project owner, 2026-11-22: contenders must draw the image; pickers (F1, S2) become teachers. The teacher's
description of every window comes from `teacher_targets.py` (F1: a DINOv2 feature, and its soft DINO v1
reading). The generator, DINOv2 and DINO v1 are frozen.

  L = (1 - cos(DINOv2(aug(G(M(a)))), t2)) + w_v1 (1 - cos(DINOv1(aug(G(M(a)))), t1))
      + w_lat L_lat + w_feat L_feat [+ w_struct L_struct]

  t2, t1     teacher description of the window (DINOv2, DINO v1)
  L_lat      sliced Wasserstein distance between M(a) and generator prior samples (as S3)
  L_feat     sliced Wasserstein distance between DINOv2 features of the images and of real art (as S3)
  L_struct   S1 structure term on the DINOv2 features (off by default)
Two encoders read the image so that a match cannot rest on detail only one of them sees (reach test,
2026-11-22: a DINOv2-only search gained little under DINO v1). The S3 augmentation is applied before both.

Variants:
  teach      as above
  shuffled   each window gets the description of another track of the batch (new permutation every step)

Test split (all windows): cosine of the drawn image to the teacher's description under DINOv2 and DINO v1,
and the same for the teacher's own pick (nearest training painting), as references. Writes mapper.pt in the
format of methods/s1 (methods/s1/evaluate.py scores structure, realism and images) and results.json.

Usage:
  python methods/gan/teach.py --vae <gan run>/generator.pt --targets <dir>/targets.npz --variant teach [--hours 4]
"""
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.data.datasets import ImageDataset
from abstraction.generators import load_generator
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import art_dataset
from abstraction.pipeline.features import build_dino, features_of_dataset
from abstraction.utils.runlog import RunLog, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s3"))
from common import TwinPenalty, augment, base_parser, build_mapper, generate, load_codes  # noqa: E402
from second_reader import DinoV1  # noqa: E402

VARIANTS = ("teach", "shuffled")


def cos_loss(a, b):
    return 1 - F.cosine_similarity(a.float(), b.float(), dim=-1).mean()


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True, help="generator checkpoint")
    parser.add_argument("--targets", required=True, help="targets.npz from teacher_targets.py")
    parser.add_argument("--variant", choices=VARIANTS, default="teach")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None, help="stop and save after this many hours")
    parser.add_argument("--w-v1", type=float, default=1.0)
    parser.add_argument("--w-struct", type=float, default=0.0)
    parser.add_argument("--w-div", type=float, default=0.0,
                        help="repetition penalty (s3 common.TwinPenalty)")
    parser.add_argument("--div-margin", type=float, default=0.8)
    parser.add_argument("--div-bank", type=int, default=1024)
    args = parser.parse_args()
    config, device = setup(args)
    hp, s3 = config["mapper"], config["s3"]
    seed = config["system"]["seed"]
    steps = args.steps or hp["steps"]
    batch_size = args.batch or hp["batch_size"]
    set_seed(seed)
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "targets": str(args.targets), "steps": steps,
                     "batch": batch_size, "w_v1": args.w_v1, "w_struct": args.w_struct,
                     "w_div": args.w_div, "div_margin": args.div_margin, "code": config["audio_code"]["variant"], "generator": args.generator}
    log = RunLog(config["paths"]["runs_dir"], f"teach_{args.variant}", config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, False)
    dino_v1 = DinoV1(device)
    for p in list(dino.parameters()) + list(dino_v1.model.parameters()):
        p.requires_grad_(False)

    with np.load(args.targets, allow_pickle=False) as data:
        row = {str(i): k for k, i in enumerate(data["ids"])}
        t2_all, t1_all = data["dinov2"], data["dino_v1"]
        pick_all, train_names = data["pick"], data["train_names"]

    def split_data(split):
        ids, codes = load_codes(config, split)
        rows = np.array([row[str(i)] for i in ids])
        return (torch.from_numpy(codes).to(device), torch.from_numpy(t2_all[rows]).float().to(device),
                torch.from_numpy(t1_all[rows]).float().to(device), rows)

    train_codes, train_t2, train_t1, _ = split_data("train")
    code_mean = train_codes.mean(dim=(0, 1))
    code_std = train_codes.std(dim=(0, 1)) + 1e-6

    prior = torch.Generator().manual_seed(seed)
    real_latents = (generator.prior_sample(20000, prior).flatten(1) / generator.latent_scale).to(device)
    real_features = features_of_dataset(dino, art_dataset(config, "train", False), device).to(device)

    mapper = build_mapper(config, generator).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator().manual_seed(seed)
    n_tracks, n_windows = train_codes.shape[:2]
    started = time.time()
    twin = TwinPenalty(args.div_bank, args.div_margin) if args.w_div > 0 else None

    step = 0
    for step in range(steps):
        track = torch.randint(n_tracks, (batch_size,), generator=g).to(device)
        window = torch.randint(n_windows, (batch_size,), generator=g).to(device)
        codes, t2, t1 = train_codes[track, window], train_t2[track, window], train_t1[track, window]
        if args.variant == "shuffled":
            permute = torch.randperm(batch_size, generator=g).to(device)
            t2, t1 = t2[permute], t1[permute]
        images, z = generate(mapper, generator, codes, code_mean, code_std)
        seen = augment(images, s3["aug"], g)
        clean = dino(images).float()
        index = torch.randint(real_latents.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        feat_index = torch.randint(real_features.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        terms = {"dinov2": cos_loss(dino(seen), t2), "dino_v1": cos_loss(dino_v1(seen), t1),
                 "lat": sliced_wasserstein(z.float().flatten(1), real_latents[index], hp["swd_projections"]),
                 "feat": sliced_wasserstein(clean, real_features[feat_index], hp["swd_projections"])}
        loss = (terms["dinov2"] + args.w_v1 * terms["dino_v1"] + s3["w_lat"] * terms["lat"]
                + s3["w_feat"] * terms["feat"])
        if args.w_struct > 0:
            terms["struct"] = distance_correlation_loss(pairwise_distances(codes), pairwise_distances(clean))
            loss = loss + args.w_struct * terms["struct"]
        if args.w_div > 0:
            terms["twin"], terms["twin_rate"] = twin(clean)
            loss = loss + args.w_div * terms["twin"]
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        if (step + 1) % 100 == 0 or step == 0:
            log.log(split="train", step=step + 1, loss=float(loss), **{k: float(v) for k, v in terms.items()})
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}: " + " ".join(f"{k} {float(v):.3f}" for k, v in terms.items()), flush=True)
        if args.hours and time.time() - started > args.hours * 3600:
            print(f"time limit reached at step {step + 1}", flush=True)
            break
    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "variant": args.variant, "seed": seed, "steps": step + 1,
                "code": config["audio_code"]["variant"]}, log.dir / "mapper.pt")

    del train_codes, train_t2, train_t1, real_latents, real_features
    mapper.eval()
    test_codes, test_t2, test_t1, test_rows = split_data("test")
    codes = test_codes.flatten(0, 1)
    t2, t1 = test_t2.flatten(0, 1), test_t1.flatten(0, 1)
    dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    row_of = {m: i for i, m in enumerate(dataset.image_files)}
    picks = pick_all[test_rows].reshape(-1)
    drawn = {"dinov2": [], "dino_v1": []}
    picked = {"dinov2": [], "dino_v1": []}
    with torch.no_grad():
        for start in range(0, len(codes), 32):
            sl = slice(start, start + 32)
            images, _ = generate(mapper, generator, codes[sl], code_mean, code_std)
            paintings = torch.stack([dataset[row_of[str(train_names[p])]] for p in picks[sl]]).to(device)
            for out, imgs in ((drawn, images), (picked, paintings)):
                out["dinov2"].append(F.cosine_similarity(dino(imgs).float(), t2[sl], dim=-1))
                out["dino_v1"].append(F.cosine_similarity(dino_v1(imgs).float(), t1[sl], dim=-1))
    results = {"variant": args.variant, "steps": step + 1, "n_test_windows": len(codes)}
    for name, block in (("drawn", drawn), ("teacher_pick", picked)):
        results[name] = {k: float(torch.cat(v).mean()) for k, v in block.items()}
    log.save_json("results.json", results)
    print(results, flush=True)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
