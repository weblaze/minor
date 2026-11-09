"""Stage A of S1: train the spatial image VAE on the art collection.

Objective: L1 reconstruction + kl_weight * KL (linear warm-up from zero)
+ dino_weight * mean squared error between the DINOv2 tokens of the input and
of the reconstruction. No label-supervised perceptual network is used.

The checkpoint with the lowest validation L1 is kept. It stores the standard
deviation of the posterior means of the training images; mapped latents are
expressed in units of this value.

Usage:
  python methods/s1/train_vae.py [--seed N] [--epochs N] [--smoke]
"""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_dino, build_vae, encode_images, setup
from data import art_dataset


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    hp = config["vae"]
    seed = config["system"]["seed"]
    epochs = args.epochs or (2 if args.smoke else hp["epochs"])
    set_seed(seed)
    log = RunLog(config["paths"]["runs_dir"], "s1_vae" + ("_smoke" if args.smoke else ""), config, seed)

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
        total, count = 0.0, 0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                recon = vae.decode(vae.encode(images)[0])
                total += F.l1_loss(recon, images, reduction="sum").item()
                count += images.numel()
        val_l1 = total / count
        log.log(split="val", step=step, epoch=epoch, l1=val_l1)
        if val_l1 < best:
            best = val_l1
            torch.save({"model": vae.state_dict(), "latent_std": 1.0, "epoch": epoch, "val_l1": val_l1},
                       log.dir / "vae.pt")
            save_image(torch.cat([images[:8], recon[:8]]) * 0.5 + 0.5, log.dir / "reconstructions.png", nrow=8)

    state = torch.load(log.dir / "vae.pt", map_location=device, weights_only=True)
    vae.load_state_dict(state["model"])
    vae.eval()
    state["latent_std"] = encode_images(vae, art_dataset(config, "train", False, args.smoke), device).std().item()
    torch.save(state, log.dir / "vae.pt")
    log.save_json("result.json", {"best_val_l1": best, "best_epoch": state["epoch"],
                                  "latent_std": state["latent_std"]})
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
