"""S4 bridge: a mapper M (MERT code -> generator latent) trained so that CAV-MAE, a model pretrained on
natural video without text, pairs each generated image with the sound it was made for.

  audio window w --MERT--> code --M--> latent --G (frozen G-GAN)--> image --CAV-MAE image encoder--> v
  the same window w --CAV-MAE audio encoder (precomputed)--> u

  L = L_av + w_lat * L_lat + w_feat * L_feat

  L_av    symmetric InfoNCE over the batch between centred, normalised v and u: the image has to be the
          one CAV-MAE rates closest to its own sound among the batch, not merely an image CAV-MAE likes
          for every sound. u is fixed; only M is trained. Centres: mean CAV-MAE embedding of the training
          windows (audio) and of the training paintings (image), as in the D-S4 and B-anchor judges.
  L_lat   sliced Wasserstein distance between M(code) and samples of the generator prior
  L_feat  sliced Wasserstein distance between DINOv2 features of the images and of real training art

Variants:
  s4        as above
  shuffled  each image is paired with the CAV-MAE audio embedding of another track of the batch (new
            permutation every step), so L_av carries no information about the input

Writes mapper.pt in the format of methods/s1 (methods/s1/evaluate.py scores structure, realism and images).

Usage:
  python methods/s4/bridge.py --vae <gan run>/generator.pt --checkpoint models/cav_mae/audio_model.21.pth
"""
import math
import time

import torch
import torch.nn.functional as F

from abstraction.encoders.cav_mae import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, load_cav_mae, smoke_cav_mae
from abstraction.generators import add_generator_argument, load_generator
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.models.mapper import Mapper
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import art_dataset
from abstraction.pipeline.features import build_dino, features_of_dataset
from abstraction.utils.runlog import RunLog, set_seed

from common import aligned_codes, base_parser, cav_inputs, image_centre, run_name


def to_cav(images, mean, std):
    """Generator images in [-1, 1] to CAV-MAE input (224 px, ImageNet normalisation); differentiable."""
    x = F.interpolate(images * 0.5 + 0.5, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
    return (x - mean) / std


def main():
    parser = add_generator_argument(base_parser(__doc__), "gan")
    parser.add_argument("--vae", required=True, help="generator checkpoint")
    parser.add_argument("--checkpoint", default=None, help="CAV-MAE checkpoint; overrides cav_mae.checkpoint")
    parser.add_argument("--variant", choices=("s4", "shuffled"), default="s4")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="overrides mapper.batch_size")
    args = parser.parse_args()
    config, device = setup(args)
    hp, bridge = config["mapper"], config["bridge"]
    if args.batch_size:
        hp["batch_size"] = args.batch_size
    config["audio_code"]["variant"] = bridge["code"]
    seed = config["system"]["seed"]
    steps = args.steps or (30 if args.smoke else hp["steps"])
    set_seed(seed)
    embeddings = args.embeddings or config["cav_mae"]["embeddings"]
    config["run"] = {"variant": args.variant, "vae": str(args.vae), "steps": steps, "code": bridge["code"],
                     "generator": args.generator, "embeddings": embeddings}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"bridge_{args.variant}", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    cav = smoke_cav_mae(device) if args.smoke else load_cav_mae(args.checkpoint or config["cav_mae"]["checkpoint"], device)
    cav_audio, cav_images, cav_names = cav_inputs(embeddings, args.smoke, cav)
    train_codes, train_audio = aligned_codes(config, "train", cav_audio, args.smoke)
    val_codes, val_audio = aligned_codes(config, "val", cav_audio, args.smoke)
    audio_centre = train_audio.reshape(-1, train_audio.shape[-1]).mean(axis=0)
    centre_v = torch.from_numpy(image_centre(config, cav_images, cav_names, args.smoke)).to(device)
    centre_a = torch.from_numpy(audio_centre).to(device)
    to = lambda x: torch.from_numpy(x).to(device)
    train_codes, train_audio, val_codes, val_audio = map(to, (train_codes, train_audio, val_codes, val_audio))
    code_mean, code_std = train_codes.mean(dim=(0, 1)), train_codes.std(dim=(0, 1)) + 1e-6
    mean = torch.tensor(IMAGE_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGE_STD, device=device)[None, :, None, None]

    prior = torch.Generator().manual_seed(seed)
    real_latents = (generator.prior_sample(512 if args.smoke else 20000, prior).flatten(1)
                    / generator.latent_scale).to(device)
    real_features = features_of_dataset(dino, art_dataset(config, "train", False, args.smoke), device).to(device)

    mapper = Mapper(train_codes.shape[-1], math.prod(generator.latent_shape), hp["hidden"], hp["depth"]).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    batch_size = min(hp["batch_size"], train_codes.shape[0])
    sampler = torch.Generator().manual_seed(seed)
    started = time.time()

    def batch(codes, audio, g):
        tracks = torch.randperm(codes.shape[0], generator=g)[:batch_size].to(device)
        windows = torch.randint(codes.shape[1], (len(tracks),), generator=g).to(device)
        return codes[tracks, windows], audio[tracks, windows]

    def losses(codes, audio, g, permute):
        z = mapper((codes - code_mean) / code_std)
        images = generator.decode(z.view(-1, *generator.latent_shape) * generator.latent_scale)
        v = F.normalize(cav.embed_image(to_cav(images, mean, std)).float() - centre_v, dim=1)
        u = F.normalize(audio - centre_a, dim=1)
        if permute is not None:
            u = u[permute]
        logits = v @ u.T / bridge["temperature"]
        target = torch.arange(len(logits), device=device)
        index = torch.randint(real_latents.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        feat_index = torch.randint(real_features.shape[0], (hp["real_batch_size"],), generator=g).to(device)
        terms = {"av": 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)),
                 "lat": sliced_wasserstein(z.float(), real_latents[index], hp["swd_projections"]),
                 "feat": sliced_wasserstein(dino(images).float(), real_features[feat_index], hp["swd_projections"])}
        top1 = logits.argmax(1).eq(target).float().mean()
        return terms, top1

    def validate():
        mapper.eval()
        g = torch.Generator().manual_seed(0)
        n_batches = max(1, val_codes.shape[0] // batch_size)
        sums = {}
        with torch.no_grad():
            for _ in range(n_batches):
                terms, top1 = losses(*batch(val_codes, val_audio, g), g, None)
                for key, value in {**terms, "top1": top1}.items():
                    sums[key] = sums.get(key, 0.0) + float(value) / n_batches
        mapper.train()
        return sums

    step = 0
    for step in range(steps):
        codes, audio = batch(train_codes, train_audio, sampler)
        permute = torch.randperm(len(codes), generator=sampler).to(device) if args.variant == "shuffled" else None
        terms, top1 = losses(codes, audio, sampler, permute)
        loss = terms["av"] + bridge["w_lat"] * terms["lat"] + bridge["w_feat"] * terms["feat"]
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
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
                "variant": args.variant, "seed": seed, "steps": step + 1, "code": bridge["code"],
                "centre_audio": centre_a.cpu(), "centre_image": centre_v.cpu()}, log.dir / "mapper.pt")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
