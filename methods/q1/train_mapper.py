"""Stage B of Q1: train the mapping M from audio codes to image latents.

  L = w_feat * L_feat + w_lat * L_lat + w_global * L_global + w_local * L_local

  L_feat    sliced Wasserstein distance between DINOv2 features of the generated images
            and DINOv2 features of real training art
  L_lat     sliced Wasserstein distance between mapped latents and samples of the
            aggregate posterior of real training art (mu + sigma * eps)
  L_global  1 - Pearson(D_a, D_v); D_a: cosine distances between the audio codes of a batch,
            D_v: cosine distances between the DINOv2 features of the generated images
  L_local   KL divergence between the row-wise neighbour distributions
            softmax(-D / (mean(D) * temperature)) of D_a and D_v

Variants:
  q1         all four terms
  no_local   w_local = 0
  no_global  w_global = 0
  no_feat    w_feat = 0
  shuffled   as q1, but D_a is computed after permuting the tracks of the batch,
             so the structure terms carry no information about the input

A batch holds one window from each of `batch_size` distinct tracks. The audio-code
transform is fitted on training tracks and stored in the checkpoint. The VAE and
DINOv2 are frozen. The checkpoint of the last step is kept; no checkpoint selection
is performed.

The script stops if gate.json next to the VAE checkpoint is missing or reports
passed = false. --ignore-gate overrides this and is recorded in the run config.

Usage:
  python methods/q1/train_mapper.py --vae runs/<vae run>/vae.pt --variant q1 --seed 0
"""
import math

import torch

from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, neighbour_kl_loss, pairwise_distances
from abstraction.utils.runlog import RunLog, set_seed

from common import (base_parser, build_dino, build_mapper, code_statistics, encode_images, features_of_dataset,
                    gate_passed, generate, load_vae, run_name, setup)
from data import (apply_code_transform, art_dataset, fit_code_transform, load_layers, sample_windows,
                  transform_to_state)

TERMS = ("feat", "lat", "global", "local")
VARIANTS = {
    "q1": {},
    "no_local": {"local": 0.0},
    "no_global": {"global": 0.0},
    "no_feat": {"feat": 0.0},
    "shuffled": {},
}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="q1")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ignore-gate", action="store_true")
    args = parser.parse_args()
    config, device = setup(args)
    hp = config["mapper"]
    seed = config["system"]["seed"]
    steps = args.steps or (30 if args.smoke else hp["steps"])
    weights = {term: hp[f"w_{term}"] for term in TERMS}
    weights.update(VARIANTS[args.variant])
    shuffle = args.variant == "shuffled"

    gate = gate_passed(args.vae)
    if gate is not True and not args.ignore_gate:
        state = "missing" if gate is None else "failed"
        raise SystemExit(f"gate.json next to {args.vae} is {state}; run gate.py or pass --ignore-gate")
    set_seed(seed)
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "steps": steps, "weights": weights,
                     "gate_passed": gate, "ignore_gate": args.ignore_gate}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"mapper_{args.variant}", args.smoke), config, seed)

    vae = load_vae(args.vae, config, device)
    dino = build_dino(config, device, args.smoke)

    _, layers, splits = load_layers(config, args.smoke)
    transform = fit_code_transform(layers[splits == "train"], config["audio_code"]["variant"],
                                   config["audio_code"]["whiten_dim"])
    train_codes = torch.from_numpy(apply_code_transform(layers[splits == "train"], transform)).to(device)
    val_codes = torch.from_numpy(apply_code_transform(layers[splits == "val"], transform)).to(device)
    del layers
    code_mean, code_std = code_statistics(train_codes)

    art = {}
    for split in ("train", "val"):
        dataset = art_dataset(config, split, False, args.smoke)
        mu, logvar = encode_images(vae, dataset, device)
        art[split] = (mu.to(device), logvar.to(device), features_of_dataset(dino, dataset, device).to(device))

    mapper = build_mapper(config, train_codes.shape[-1]).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    batch_size = min(hp["batch_size"], train_codes.shape[0], 32 if args.smoke else hp["batch_size"])
    sampler = torch.Generator().manual_seed(seed)
    log.save_json("data.json", {"n_train_tracks": train_codes.shape[0], "n_val_tracks": val_codes.shape[0],
                                "n_train_art": art["train"][0].shape[0], "n_val_art": art["val"][0].shape[0],
                                "code_dim": train_codes.shape[-1], "batch_size": batch_size})

    def losses(codes, split, generator, permute=None):
        mu, logvar, real_features = art[split]
        index = torch.randint(mu.shape[0], (hp["real_batch_size"],), generator=generator).to(device)
        noise = torch.randn(len(index), mu.shape[1], generator=generator).to(device)
        real_latents = mu[index] + (0.5 * logvar[index]).exp() * noise
        images, z = generate(mapper, vae, codes, code_mean, code_std)
        features = dino(images)
        source = codes if permute is None else codes[permute]
        d_audio, d_image = pairwise_distances(source), pairwise_distances(features)
        return {
            "feat": sliced_wasserstein(features, real_features[index], hp["swd_projections"]),
            "lat": sliced_wasserstein(z, real_latents, hp["swd_projections"]),
            "global": distance_correlation_loss(d_audio, d_image),
            "local": neighbour_kl_loss(d_audio, d_image, hp["local_temperature"]),
        }

    def validate():
        mapper.eval()
        g = torch.Generator().manual_seed(0)
        totals = {term: 0.0 for term in TERMS}
        n_batches = max(1, val_codes.shape[0] // batch_size)
        with torch.no_grad():
            for _ in range(n_batches):
                terms = losses(sample_windows(val_codes, batch_size, g), "val", g)
                for term in TERMS:
                    totals[term] += terms[term].item() / n_batches
        mapper.train()
        return totals

    for step in range(steps):
        codes = sample_windows(train_codes, batch_size, sampler)
        permute = torch.randperm(codes.shape[0], generator=sampler).to(device) if shuffle else None
        terms = losses(codes, "train", sampler, permute)
        loss = sum(weights[term] * terms[term] for term in TERMS)
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}: {[(k, v.item()) for k, v in terms.items()]}")
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        if step % 20 == 0:
            log.log(split="train", step=step, loss=loss, **{f"l_{term}": terms[term] for term in TERMS})
        if (step + 1) % hp["val_every"] == 0 or step + 1 == steps:
            log.log(split="val", step=step + 1, **{f"l_{term}": value for term, value in validate().items()})

    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "code_transform": transform_to_state(transform), "variant": args.variant, "seed": seed,
                "steps": steps}, log.dir / "mapper.pt")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
