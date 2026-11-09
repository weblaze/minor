"""Stage B of S1: train the mapping M from audio codes to image latents.

  L = L_dist + lambda_struct * L_struct + mu_cycle * L_cycle

  L_dist    sliced Wasserstein distance between mapped latents and latents of real art
  L_struct  1 - Pearson(D_a, D_v); D_a: cosine distances between the audio codes of a batch,
            D_v: cosine distances between the DINOv2 features of the decoded images
  L_cycle   mean squared error of an inverse network that predicts the audio code from the latent

Variants:
  s1        L_dist + lambda * L_struct
  s0        L_dist + mu * L_cycle                  (objective of the 2025 version)
  s1_cycle  L_dist + lambda * L_struct + mu * L_cycle
  shuffled  as s1, but D_a is computed after permuting the tracks of the batch,
            so the structure term carries no information about the input

A batch holds one window from each of `batch_size` distinct tracks. The VAE and
DINOv2 are frozen. The checkpoint of the last step is kept; no checkpoint
selection is performed.

Usage:
  python methods/s1/train_mapper.py --vae runs/<vae run>/vae.pt --variant s1 --seed 0
"""
import math

import torch
import torch.nn.functional as F

from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.models.mapper import Mapper
from abstraction.utils.runlog import RunLog, set_seed

from common import (CODE_DIM, base_parser, build_dino, build_mapper, encode_images, generate, latent_dim,
                    load_vae, setup)
from data import art_dataset, load_codes, sample_windows

VARIANTS = {
    "s1": (True, False, False),
    "s0": (False, True, False),
    "s1_cycle": (True, True, False),
    "shuffled": (True, False, True),
}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="s1")
    parser.add_argument("--lambda-struct", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    hp = config["mapper"]
    if args.lambda_struct is not None:
        hp["lambda_struct"] = args.lambda_struct
    seed = config["system"]["seed"]
    steps = args.steps or (30 if args.smoke else hp["steps"])
    use_struct, use_cycle, shuffle = VARIANTS[args.variant]
    set_seed(seed)
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "steps": steps}
    log = RunLog(config["paths"]["runs_dir"], f"s1_mapper_{args.variant}" + ("_smoke" if args.smoke else ""),
                 config, seed)

    vae, latent_std = load_vae(args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    _, train_codes = load_codes(config, "train", args.smoke)
    _, val_codes = load_codes(config, "val", args.smoke)
    train_codes = torch.from_numpy(train_codes).to(device)
    val_codes = torch.from_numpy(val_codes).to(device)
    code_mean = train_codes.mean(dim=(0, 1))
    code_std = train_codes.std(dim=(0, 1)) + 1e-6

    real = encode_images(vae, art_dataset(config, "train", False, args.smoke), device) / latent_std
    real_val = encode_images(vae, art_dataset(config, "val", False, args.smoke), device) / latent_std
    real, real_val = real.to(device), real_val.to(device)

    mapper = build_mapper(config).to(device)
    inverse = Mapper(latent_dim(config), CODE_DIM, hp["hidden"], hp["depth"]).to(device) if use_cycle else None
    params = list(mapper.parameters()) + (list(inverse.parameters()) if use_cycle else [])
    optimiser = torch.optim.AdamW(params, lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    use_amp = hp["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    batch_size = min(hp["batch_size"], train_codes.shape[0])
    sampler = torch.Generator().manual_seed(seed)
    log.save_json("data.json", {"n_train_tracks": train_codes.shape[0], "n_val_tracks": val_codes.shape[0],
                                "n_real_latents": real.shape[0], "latent_std": latent_std,
                                "batch_size": batch_size})

    def losses(codes, real_latents, permute):
        """Returns the three loss terms; a term that the variant does not use is still
        computed when `permute` is None (validation), so that all variants report all terms."""
        evaluate_all = permute is None
        with torch.autocast(device.type, enabled=use_amp):
            images, z = generate(mapper, vae, latent_std, codes, code_mean, code_std, config)
            features = dino(images).float() if (use_struct or evaluate_all) else None
        z = z.float()
        l_dist = sliced_wasserstein(z, real_latents, hp["swd_projections"])
        l_struct = torch.zeros((), device=device)
        if features is not None:
            source = codes[permute] if permute is not None and shuffle else codes
            l_struct = distance_correlation_loss(pairwise_distances(source), pairwise_distances(features))
        l_cycle = torch.zeros((), device=device)
        if use_cycle:
            l_cycle = F.mse_loss(inverse(z.flatten(1)), (codes - code_mean) / code_std)
        return l_dist, l_struct, l_cycle

    def validate():
        mapper.eval()
        g = torch.Generator().manual_seed(0)
        totals = torch.zeros(3)
        n_batches = max(1, val_codes.shape[0] // batch_size)
        with torch.no_grad():
            for _ in range(n_batches):
                codes = sample_windows(val_codes, batch_size, g)
                index = torch.randint(real_val.shape[0], (hp["real_batch_size"],), generator=g)
                totals += torch.stack(losses(codes, real_val[index.to(device)], None)).cpu()
        mapper.train()
        return (totals / n_batches).tolist()

    for step in range(steps):
        codes = sample_windows(train_codes, batch_size, sampler)
        index = torch.randint(real.shape[0], (hp["real_batch_size"],), generator=sampler)
        permute = torch.randperm(codes.shape[0], generator=sampler).to(device)
        l_dist, l_struct, l_cycle = losses(codes, real[index.to(device)], permute)
        loss = l_dist
        if use_struct:
            loss = loss + hp["lambda_struct"] * l_struct
        if use_cycle:
            loss = loss + hp["mu_cycle"] * l_cycle
        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        if step % 20 == 0:
            log.log(split="train", step=step, loss=loss, l_dist=l_dist, l_struct=l_struct, l_cycle=l_cycle)
        if (step + 1) % hp["val_every"] == 0 or step + 1 == steps:
            v_dist, v_struct, v_cycle = validate()
            log.log(split="val", step=step + 1, l_dist=v_dist, l_struct=v_struct, l_cycle=v_cycle)

    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "variant": args.variant, "seed": seed, "steps": steps}, log.dir / "mapper.pt")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
