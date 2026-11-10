"""Stage A of Q1: train the vector-latent image VAE on the art collection.

Objective: L1 reconstruction + kl_weight * KL (mean over latent dimensions,
linear warm-up from zero) + dino_weight * mean squared error between the
DINOv2 tokens of the input and of the reconstruction.

The checkpoint with the lowest validation L1 is kept. The VAE has to pass the
sampleability gate (gate.py) before a mapping is trained on it.

Usage:
  python methods/q1/train_vae.py [--seed N] [--epochs N] [--kl-weight X] [--smoke]
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_dino, build_vae, run_name, setup
from data import art_dataset


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--kl-weight", type=float, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    hp = config["vae"]
    if args.kl_weight is not None:
        hp["kl_weight"] = args.kl_weight
    seed = config["system"]["seed"]
    epochs = args.epochs or (2 if args.smoke else hp["epochs"])
    set_seed(seed)
    log = RunLog(config["paths"]["runs_dir"], run_name("vae", args.smoke), config, seed)

    train_set = art_dataset(config, "train", True, args.smoke)
    val_set = art_dataset(config, "val", False, args.smoke)
    loader = DataLoader(train_set, batch_size=hp["batch_size"], shuffle=True, drop_last=True,
                        num_workers=0 if args.smoke else 2)
    val_loader = DataLoader(val_set, batch_size=hp["batch_size"])

    vae = build_vae(config).to(device)
    dino = build_dino(config, device, args.smoke)
    optimiser = torch.optim.AdamW(vae.parameters(), lr=hp["lr"])
    total_steps = epochs * len(loader)

    def lr_factor(step):
        if step < hp["warmup_steps"]:
            return (step + 1) / hp["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_factor)
    use_amp = hp["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log.save_json("data.json", {"n_train": len(train_set), "n_val": len(val_set),
                                "n_parameters": sum(p.numel() for p in vae.parameters())})

    step, best = 0, float("inf")
    for epoch in range(epochs):
        vae.train()
        for images in loader:
            images = images.to(device)
            kl_weight = hp["kl_weight"] * min(1.0, step / hp["kl_warmup_steps"])
            with torch.autocast(device.type, enabled=use_amp):
                recon, mu, logvar = vae(images)
                l1 = F.l1_loss(recon, images)
                mu, logvar = mu.float(), logvar.float()
                kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).mean()
                with torch.no_grad():
                    target = dino(images, "tokens")
                feat = F.mse_loss(dino(recon, "tokens").float(), target.float())
                loss = l1 + kl_weight * kl + hp["dino_weight"] * feat
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            if step % 50 == 0:
                log.log(split="train", step=step, epoch=epoch, loss=loss, l1=l1, kl=kl, dino=feat,
                        kl_weight=kl_weight, lr=scheduler.get_last_lr()[0])
            step += 1

        vae.eval()
        total, total_kl, count, n_images = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                mu, logvar = vae.encode(images)
                recon = vae.decode(mu)
                total += F.l1_loss(recon, images, reduction="sum").item()
                total_kl += 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum().item()
                count += images.numel()
                n_images += len(images)
        val_l1 = total / count
        log.log(split="val", step=step, epoch=epoch, l1=val_l1, kl_nats_per_image=total_kl / n_images)
        if val_l1 < best:
            best = val_l1
            torch.save({"model": vae.state_dict(), "epoch": epoch, "val_l1": val_l1}, log.dir / "vae.pt")
            save_image(torch.cat([images[:8], recon[:8]]) * 0.5 + 0.5, log.dir / "reconstructions.png", nrow=8)

    log.save_json("result.json", {"best_val_l1": best, "kl_weight": hp["kl_weight"]})
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
