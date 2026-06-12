"""Two-stage training that stamps the original autoencoder approach complete.

Stage pretrain — plain ImageVAE on the abstract-art set (MSE + VGG16 style
loss + KL with warm-up). This is the original v0.2 trainer, modularized.

Stage finetune — ConditionedImageVAE warm-started from the pretrain checkpoint
(only cond_proj is new). The decoder learns to use CLAP audio embeddings from
the pairs file, with conditioning dropped at cond_dropout so unconditioned
decoding keeps working. 10-20 epochs is enough — see the 2026 stress-test doc.

Usage:
  python train.py --stage pretrain [--epochs N] [--max-steps N] [--no-wandb]
  python train.py --stage finetune [--epochs N] [--max-steps N] [--no-wandb]
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from abstraction.data.datasets import ClapImageDataset, ImageDataset
from abstraction.models.image_vae import ConditionedImageVAE, ImageVAE
from abstraction.utils.checkpoints import save_checkpoint, warm_start
from abstraction.utils.config import load_config
from abstraction.utils.wandb_utils import init_wandb

APPROACH = "01_conditioned_autoencoder"

VGG_LAYERS = [2, 7, 12, 21, 30]  # conv1_2, conv2_2, conv3_3, conv4_3, conv5_3
vgg_preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram.div(c * h * w)


def style_loss(vgg, x, recon_x):
    # Concatenate target and reconstruction along the batch dimension to process in a single forward pass.
    # Target x is detached.
    combined = torch.cat([x.detach(), recon_x], dim=0)
    combined_vgg = vgg_preprocess(combined)
    loss = 0
    curr = combined_vgg
    for i, layer in enumerate(vgg):
        curr = layer(curr)
        if i in VGG_LAYERS:
            x_feat, recon_feat = torch.chunk(curr, 2, dim=0)
            loss += nn.functional.mse_loss(gram_matrix(x_feat), gram_matrix(recon_feat))
    return loss


def vae_loss(vgg, recon_x, x, mu, logvar, beta, style_weight):
    mse = nn.functional.mse_loss(recon_x, x, reduction="mean")
    style = style_loss(vgg, x, recon_x)
    mu_c = torch.clamp(mu, -10, 10)
    logvar_c = torch.clamp(logvar, -10, 10)
    kl = -0.5 * torch.mean(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    return mse + style_weight * style + beta * kl, mse, style, kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    vae_cfg = config["image_vae"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["system"]["seed"])
    print(f"Using device: {device}")

    finetune = args.stage == "finetune"
    epochs = args.epochs or (vae_cfg["finetune_epochs"] if finetune else vae_cfg["pretrain_epochs"])
    base_beta = vae_cfg["beta"]
    warmup = vae_cfg["kl_warmup_epochs"]
    style_weight = vae_cfg["style_weight"]
    lr = float(vae_cfg["learning_rate"])

    if finetune:
        dataset = ClapImageDataset(
            pairs_file=config["paths"]["pairs_file"],
            clap_dir=config["paths"]["clap_features"],
            image_dir=config["paths"]["abstract_art"],
            size=config["image_size"],
        )
        model = ConditionedImageVAE(latent_channels=vae_cfg["latent_channels"]).to(device)
        warm_start(model, config["paths"]["image_vae_checkpoint"], device=device)
        ckpt_path = config["paths"]["conditioned_checkpoint"]

        # Freeze encoder parameters so the latent space does not adapt to bypass conditioning
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.fc_mu.parameters():
            p.requires_grad = False
        for p in model.fc_logvar.parameters():
            p.requires_grad = False
    else:
        dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"])
        model = ImageVAE(latent_channels=vae_cfg["latent_channels"]).to(device)
        ckpt_path = config["paths"]["image_vae_checkpoint"]
        if Path(ckpt_path).exists():
            print(f"Resuming pretrain from {ckpt_path}")
            warm_start(model, ckpt_path, device=device)

    loader = DataLoader(dataset, batch_size=vae_cfg["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0, pin_memory=True)
    print(f"Loaded {len(dataset)} samples for stage {args.stage}")

    print("Loading VGG16 for style loss...")
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
    for m in vgg.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    for p in vgg.parameters():
        p.requires_grad = False

    # Filter optimizer parameters to train only parameters that require grad (e.g. cond_proj/decoder during finetuning)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.5, 0.999))
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    run = None if args.no_wandb else init_wandb(config, APPROACH, args.stage)

    model.train()
    step = 0
    for epoch in range(epochs):
        beta = 0.0 if epoch < warmup else min(base_beta, base_beta * (epoch - warmup + 1) / warmup)
        totals = {"loss": 0.0, "mse": 0.0, "style": 0.0, "kl": 0.0}

        for batch in loader:
            if finetune:
                images = batch["image"].to(device)
                cond = batch["audio_emb"].to(device)
                drop = torch.rand(cond.shape[0], device=device) < vae_cfg["cond_dropout"]
                cond = torch.where(drop[:, None], torch.zeros_like(cond), cond)
                
                # Latent dropout: with 50% probability, we zero out z to force decoder to rely on cond
                mu, logvar, _ = model.encode(images)
                z = model.reparameterize(mu, logvar)
                z_drop = torch.rand(z.shape[0], device=device) < 0.5
                z = torch.where(z_drop[:, None, None, None], torch.zeros_like(z), z)
                
                recon = model.decode(z, cond=cond)
            else:
                images = batch.to(device)
                recon, mu, logvar = model(images)

            optimizer.zero_grad()
            loss, mse, style, kl = vae_loss(vgg, recon, images, mu, logvar, beta, style_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals["loss"] += loss.item()
            totals["mse"] += mse.item()
            totals["style"] += style.item()
            totals["kl"] += kl.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(model, ckpt_path)
                print(f"[smoke] stopped after {step} steps, checkpoint saved to {ckpt_path}")
                return

        lr_sched.step()
        n = len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | loss {totals['loss']/n:.4f} | "
              f"mse {totals['mse']/n:.4f} | style {totals['style']/n:.4f} | kl {totals['kl']/n:.4f}",
              flush=True)
        if run:
            run.log({"epoch": epoch + 1, "beta": beta,
                     **{k: v / n for k, v in totals.items()}})

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            save_checkpoint(model, ckpt_path)
            print(f"--- checkpoint saved at epoch {epoch + 1} ---", flush=True)

    save_checkpoint(model, ckpt_path)
    print(f"Saved {args.stage} checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
