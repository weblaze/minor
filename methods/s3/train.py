"""S3: train a mapper M (audio code -> generator latent) and a reader R together, so that the track can be
identified from its image alone. The generator (G-GAN by default) and DINOv2 are frozen.

  L = L_info + w_lat * L_lat + w_feat * L_feat  [+ w_struct * L_struct]

  L_info    symmetric InfoNCE between R(DINOv2(aug(G(M(a))))) and the audio side of R, over the batch
  L_lat     sliced Wasserstein distance between M(a) and samples of the generator prior
  L_feat    sliced Wasserstein distance between DINOv2 features of the (unaugmented) images and of real art
  L_struct  the S1 structure term (s3_struct only)

Variants:
  s3         as above
  s3_noaug   no augmentation before DINOv2
  s3_struct  plus the S1 structure term
  shuffled   each image is paired with the code of another track of the batch (new permutation every step),
             so L_info carries no information about the input
  random_m   M stays at its random initialisation; only R is trained

Writes mapper.pt in the format of methods/s1 (so methods/s1/evaluate.py scores structure, realism and
images) and reader.pt for methods/s3/identify.py.

Usage:
  python methods/s3/train.py --vae <gan run>/generator.pt --variant s3 --seed 0 [--hours 9]
"""
import math
import time

import torch

from abstraction.generators import load_generator
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import art_dataset, sample_windows
from abstraction.pipeline.features import build_dino, features_of_dataset
from abstraction.utils.runlog import RunLog, set_seed

from common import Reader, augment, base_parser, build_mapper, code_dim, generate, info_nce, load_codes, run_name

VARIANTS = ("s3", "s3_noaug", "s3_struct", "shuffled", "random_m")


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True, help="generator checkpoint")
    parser.add_argument("--variant", choices=VARIANTS, default="s3")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None, help="stop and save after this many hours")
    args = parser.parse_args()
    config, device = setup(args)
    hp, s3 = config["mapper"], config["s3"]
    seed = config["system"]["seed"]
    steps = args.steps or (30 if args.smoke else hp["steps"])
    use_aug = args.variant != "s3_noaug"
    set_seed(seed)
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "steps": steps,
                     "code": config["audio_code"]["variant"], "generator": args.generator}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"train_{args.variant}", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    _, train_codes = load_codes(config, "train", args.smoke)
    _, val_codes = load_codes(config, "val", args.smoke)
    train_codes = torch.from_numpy(train_codes).to(device)
    val_codes = torch.from_numpy(val_codes).to(device)
    code_mean = train_codes.mean(dim=(0, 1))
    code_std = train_codes.std(dim=(0, 1)) + 1e-6

    prior = torch.Generator().manual_seed(seed)
    real_latents = (generator.prior_sample(20000 if not args.smoke else 512, prior).flatten(1)
                    / generator.latent_scale).to(device)
    real_features = features_of_dataset(dino, art_dataset(config, "train", False, args.smoke), device).to(device)

    mapper = build_mapper(config, generator).to(device)
    reader = Reader(code_dim(config), s3["embed_dim"], s3["reader_hidden"]).to(device)
    groups = [{"params": list(reader.parameters()), "lr": s3["reader_lr"]}]
    if args.variant != "random_m":
        groups.append({"params": list(mapper.parameters()), "lr": hp["lr"]})
    else:
        mapper.requires_grad_(False)
    params = [p for g in groups for p in g["params"]]
    optimiser = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    batch_size = min(hp["batch_size"], train_codes.shape[0])
    sampler = torch.Generator().manual_seed(seed)
    started = time.time()

    def losses(codes, g, permute):
        normed = (codes - code_mean) / code_std
        images, z = generate(mapper, generator, codes, code_mean, code_std)
        z = z.float().flatten(1)
        clean = dino(images).float()
        seen = dino(augment(images, s3["aug"], g)).float() if use_aug else clean
        audio = normed[permute] if permute is not None else normed
        image_embed, audio_embed = reader.embed_images(seen), reader.embed_audio(audio)
        index = torch.randint(real_latents.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        feat_index = torch.randint(real_features.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        terms = {
            "info": info_nce(image_embed, audio_embed, s3["temperature"]),
            "lat": sliced_wasserstein(z, real_latents[index], hp["swd_projections"]),
            "feat": sliced_wasserstein(clean, real_features[feat_index], hp["swd_projections"]),
            "struct": distance_correlation_loss(pairwise_distances(codes), pairwise_distances(clean)),
        }
        top1 = (image_embed @ audio_embed.T).argmax(1).eq(torch.arange(len(codes), device=device)).float().mean()
        return terms, top1

    def total(terms):
        loss = terms["info"]
        if args.variant != "random_m":
            loss = loss + s3["w_lat"] * terms["lat"] + s3["w_feat"] * terms["feat"]
        if args.variant == "s3_struct":
            loss = loss + s3["w_struct"] * terms["struct"]
        return loss

    def validate():
        mapper.eval(), reader.eval()
        g = torch.Generator().manual_seed(0)
        n_batches = max(1, val_codes.shape[0] // batch_size)
        sums = {}
        with torch.no_grad():
            for _ in range(n_batches):
                terms, top1 = losses(sample_windows(val_codes, batch_size, g), g, None)
                for key, value in {**terms, "top1": top1}.items():
                    sums[key] = sums.get(key, 0.0) + float(value) / n_batches
        mapper.train(), reader.train()
        return sums

    step = 0
    for step in range(steps):
        codes = sample_windows(train_codes, batch_size, sampler)
        permute = torch.randperm(len(codes), generator=sampler).to(device) if args.variant == "shuffled" else None
        terms, top1 = losses(codes, sampler, permute)
        loss = total(terms)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        scheduler.step()
        if step % 20 == 0:
            log.log(split="train", step=step, loss=loss, top1=top1, **{f"l_{k}": v for k, v in terms.items()})
        if (step + 1) % hp["val_every"] == 0 or step + 1 == steps:
            log.log(split="val", step=step + 1, **validate())
        if args.hours and time.time() - started > args.hours * 3600:
            print(f"time limit reached at step {step + 1}", flush=True)
            break

    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "variant": args.variant, "seed": seed, "steps": step + 1,
                "code": config["audio_code"]["variant"]}, log.dir / "mapper.pt")
    torch.save({"model": reader.state_dict(), "embed_dim": s3["embed_dim"], "hidden": s3["reader_hidden"]},
               log.dir / "reader.pt")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
