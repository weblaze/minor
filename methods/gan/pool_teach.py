"""Drawer taught by S2 over a G-GAN pool: every target is an image the generator has already drawn.

The F1-taught drawer (teach.py) failed mainly because the G-GAN cannot draw what a picked painting looks
like (reach test 2026-11-22: DINO v1 0.41 against 0.56). Here the targets are generator latents, so every
target is reachable by construction, and each training track gets a different one.

  pool      `--n` latents z ~ N(0, I) from a fixed seed; near-blank images dropped (rule of
            methods/s2/gan_pool.py); DINOv2 features of the kept images
  assign    S2 out of sample on the TRAINING tracks only, fixed once: Gromov-Wasserstein between
            `--n-match` random training tracks and `--n-gallery` pool images (track codes: mean of the window
            codes), barycentric targets, every training track placed by ridge regression from its code
            (methods/s2/oos.py, linear), then all training tracks assigned to pool images jointly, each image
            at most once (Hungarian). Variant `shuffled`: a random one-to-one assignment instead.
  regress   mapper M (audio window code -> latent) trained with mean squared error to the assigned latent of
            its track; the generator is not in the loop.

Test tracks are never used for the assignment; they are drawn by M alone, so the same track always gives
the same image. Writes `assignment.npz`, `mapper.pt` (format of methods/s1; methods/s1/evaluate.py and
methods/s3/second_reader.py read it) and results.json (assignment structure: Mantel-style correlation of
track code distances and assigned-image feature distances on training tracks; train/val regression error).

Usage:
  python methods/gan/pool_teach.py --vae <gan run>/generator.pt [--variant teach|shuffled] [--n 12000]
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.pipeline.cli import setup
from abstraction.pipeline.features import build_dino
from abstraction.generators import load_generator
from abstraction.utils.config import load_config
from abstraction.utils.runlog import RunLog, set_seed

METHODS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHODS / "s2"))
from common import cosine_distances, gromov_wasserstein  # noqa: E402  (methods/s2/common.py)
from gan_pool import blank  # noqa: E402
from oos import joint, place_linear  # noqa: E402

sys.path.insert(0, str(METHODS / "s3"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("s3_common", METHODS / "s3" / "common.py")
s3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s3)


class BlankRule:
    min_std, grey_std, min_sat = 0.06, 0.10, 0.08


def upper(d):
    i, j = np.triu_indices(len(d), 1)
    return d[i, j]


def main():
    parser = s3.base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--variant", choices=("teach", "shuffled"), default="teach")
    parser.add_argument("--n", type=int, default=12000, help="latents drawn for the pool")
    parser.add_argument("--n-match", type=int, default=3000)
    parser.add_argument("--n-gallery", type=int, default=6000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    config, device = setup(args)
    s2 = load_config(METHODS / "s2" / "config.yaml")
    hp = {**s2["match"], **s2["oos"]}
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "n": args.n, "n_match": args.n_match,
                     "n_gallery": args.n_gallery, "steps": args.steps, "batch": args.batch,
                     "code": config["audio_code"]["variant"], "match": hp}
    log = RunLog(config["paths"]["runs_dir"], f"pool_teach_{args.variant}", config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, False)
    torch.manual_seed(seed)
    latents, features = [], []
    with torch.no_grad():
        for start in range(0, args.n, 100):
            z = torch.randn(min(100, args.n - start), *generator.latent_shape, device=device)
            images = generator.decode(z * generator.latent_scale).clamp(-1, 1)
            keep = ~blank(images, BlankRule)
            latents.append(z[keep].flatten(1).cpu())
            features.append(dino(images[keep]).float().cpu())
    latents, features = torch.cat(latents).numpy(), torch.cat(features).numpy()
    print(f"pool: kept {len(latents)} of {args.n}", flush=True)

    ids, codes = s3.load_codes(config, "train")
    tracks = codes.mean(axis=1)
    n_train = len(tracks)
    if len(latents) < n_train:
        raise SystemExit(f"pool too small: {len(latents)} kept images for {n_train} training tracks")
    if args.variant == "teach":
        matched_idx = rng.choice(n_train, size=min(args.n_match, n_train), replace=False)
        gallery_idx = rng.choice(len(features), size=min(args.n_gallery, len(features)), replace=False)
        d_a = cosine_distances(torch.as_tensor(tracks[matched_idx], dtype=torch.float32, device=device))
        d_b = cosine_distances(torch.as_tensor(features[gallery_idx], dtype=torch.float32, device=device))
        coupling, gw = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"],
                                          torch.Generator().manual_seed(seed), eps_start=hp["epsilon_start"],
                                          decay=hp["epsilon_decay"])
        weights = (coupling / coupling.sum(dim=1, keepdim=True)).cpu().numpy()
        targets = weights @ features[gallery_idx]
        placed = place_linear(tracks[matched_idx], targets, tracks, hp["alpha"])
        picks = joint(placed, features)
        del d_a, d_b, coupling
    else:
        gw = None
        picks = rng.permutation(len(latents))[:n_train]

    d_code = upper(cosine_distances(torch.as_tensor(tracks[:2000], dtype=torch.float32)).numpy())
    d_img = upper(cosine_distances(torch.as_tensor(features[picks[:2000]], dtype=torch.float32)).numpy())
    assign_r = float(np.corrcoef(d_code, d_img)[0, 1])
    print(f"assignment: GW {gw}, distinct {len(np.unique(picks))}, train structure r {assign_r:.3f}", flush=True)
    np.savez(log.dir / "assignment.npz", track_ids=ids, picks=picks, latents=latents, features=features)

    train_codes = torch.from_numpy(codes).float().to(device)
    target_z = torch.from_numpy(latents[picks]).float().to(device)
    code_mean = train_codes.mean(dim=(0, 1))
    code_std = train_codes.std(dim=(0, 1)) + 1e-6
    _, val_codes = s3.load_codes(config, "val")
    mapper = s3.build_mapper(config, generator).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=config["mapper"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.steps)
    g = torch.Generator().manual_seed(seed)
    n_windows = train_codes.shape[1]
    for step in range(args.steps):
        track = torch.randint(n_train, (args.batch,), generator=g).to(device)
        window = torch.randint(n_windows, (args.batch,), generator=g).to(device)
        pred = mapper((train_codes[track, window] - code_mean) / code_std)
        loss = F.mse_loss(pred, target_z[track])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        scheduler.step()
        if (step + 1) % 1000 == 0:
            log.log(split="train", step=step + 1, loss=float(loss))
            print(f"step {step + 1}: mse {float(loss):.4f}", flush=True)
    mapper.eval()
    with torch.no_grad():
        train_pred = mapper((train_codes.mean(dim=1) - code_mean) / code_std)
        val_pred = mapper((torch.from_numpy(val_codes).float().to(device).mean(dim=1) - code_mean) / code_std)
    results = {"variant": args.variant, "pool_kept": int(len(latents)), "pool_drawn": args.n, "gw": gw,
               "distinct": int(len(np.unique(picks))), "assignment_train_r": assign_r,
               "train_mse_track_mean": float(F.mse_loss(train_pred, target_z)),
               "train_cos_track_mean": float(F.cosine_similarity(train_pred, target_z).mean()),
               "val_latent_norm": float(val_pred.norm(dim=1).mean()),
               "prior_latent_norm": float(target_z.norm(dim=1).mean())}
    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "variant": f"pool_{args.variant}", "seed": seed, "steps": args.steps,
                "code": config["audio_code"]["variant"]}, log.dir / "mapper.pt")
    log.save_json("results.json", results)
    print(results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
