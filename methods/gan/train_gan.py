"""G-GAN: unconditional generator of abstract art (128 px, vector latent).

Generator: latent vector -> 4 x 4 feature map -> five upsampling blocks -> image in [-1, 1].
Discriminator: mirrored, spectral normalisation. Hinge loss. Differentiable augmentation
(colour, translation, cutout) is applied to real and generated images before the discriminator.
An exponential moving average of the generator weights is kept and is the model that is saved
and evaluated.

Every `eval_every` steps: Frechet and kernel distance (DINOv2 features) of `n_eval_samples`
samples of the averaged generator against the test split of the art collection, and a sample
grid. The run stops at `gan.steps` or when `--hours` of wall-clock time have passed.

Usage:
  python methods/gan/train_gan.py --seed 0 [--channels 64] [--hours 9]
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from abstraction.data.datasets import ImageDataset
from abstraction.encoders.dino import DinoFeatures
from abstraction.metrics.distribution import frechet_distance, kernel_distance
from abstraction.utils.config import load_config
from abstraction.utils.runlog import RunLog, set_seed

CONFIG_PATH = Path(__file__).with_name("config.yaml")


class Generator(nn.Module):
    def __init__(self, latent_dim, ch):
        super().__init__()
        widths = [ch * 8, ch * 8, ch * 4, ch * 2, ch, ch]
        self.first = nn.Linear(latent_dim, widths[0] * 16)
        blocks = []
        for a, b in zip(widths[:-1], widths[1:]):
            blocks += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(a, b, 3, padding=1),
                       nn.BatchNorm2d(b), nn.LeakyReLU(0.2), nn.Conv2d(b, b, 3, padding=1),
                       nn.BatchNorm2d(b), nn.LeakyReLU(0.2)]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv2d(widths[-1], 3, 3, padding=1)
        self.widths = widths

    def forward(self, z):
        x = self.first(z).view(len(z), self.widths[0], 4, 4)
        return torch.tanh(self.out(self.blocks(x)))


class Discriminator(nn.Module):
    def __init__(self, ch):
        super().__init__()
        sn = nn.utils.spectral_norm
        widths = [ch, ch, ch * 2, ch * 4, ch * 8, ch * 8]
        layers = [sn(nn.Conv2d(3, widths[0], 3, padding=1)), nn.LeakyReLU(0.2)]
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [sn(nn.Conv2d(a, b, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
                       sn(nn.Conv2d(b, b, 3, padding=1)), nn.LeakyReLU(0.2)]
        self.features = nn.Sequential(*layers)
        self.out = sn(nn.Linear(widths[-1] * 16, 1))

    def forward(self, x):
        return self.out(self.features(x).flatten(1)).squeeze(1)


def augment(x, policy):
    """Differentiable augmentation (Zhao et al. 2020): colour, translation by up to 1/8, cutout of 1/2."""
    if "color" in policy:
        x = x + (torch.rand(len(x), 1, 1, 1, device=x.device) - 0.5)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) * (torch.rand(len(x), 1, 1, 1, device=x.device) * 2) + mean
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * (torch.rand(len(x), 1, 1, 1, device=x.device) + 0.5) + mean
    n, _, h, w = x.shape
    if "translation" in policy:
        sx = torch.randint(-(w // 8), w // 8 + 1, (n, 1, 1), device=x.device)
        sy = torch.randint(-(h // 8), h // 8 + 1, (n, 1, 1), device=x.device)
        gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
        gx = (gx.unsqueeze(0) + sx).clamp(-1, w) + 1
        gy = (gy.unsqueeze(0) + sy).clamp(-1, h) + 1
        padded = F.pad(x, (1, 1, 1, 1))
        x = padded[torch.arange(n, device=x.device).view(n, 1, 1), :, gy, gx].permute(0, 3, 1, 2)
    if "cutout" in policy:
        ch, cw = h // 2, w // 2
        oy = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=x.device)
        ox = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=x.device)
        gy, gx = torch.meshgrid(torch.arange(ch, device=x.device), torch.arange(cw, device=x.device), indexing="ij")
        gy = (gy.unsqueeze(0) + oy - ch // 2).clamp(0, h - 1)
        gx = (gx.unsqueeze(0) + ox - cw // 2).clamp(0, w - 1)
        mask = torch.ones(n, h, w, device=x.device)
        mask[torch.arange(n, device=x.device).view(n, 1, 1), gy, gx] = 0
        x = x * mask.unsqueeze(1)
    return x


class SyntheticImages(Dataset):
    """Random low-frequency colour fields. For smoke runs only."""

    def __init__(self, n, size, seed):
        g = torch.Generator().manual_seed(seed)
        low = torch.rand(n, 3, 4, 4, generator=g) * 2 - 1
        self.images = F.interpolate(low, size=(size, size), mode="bicubic").clamp(-1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


@torch.no_grad()
def features_of(dino, batches, device):
    return torch.cat([dino(x.to(device)).float().cpu() for x in batches]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="appended to the run directory name")
    parser.add_argument("--channels", type=int, default=None, help="overrides gan.base_channels")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--hours", type=float, default=None, help="stop after this much wall-clock time")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic images, randomly initialised DINOv2, few steps; checks the code path only")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["system"]["seed"] = args.seed
    if args.tag:
        config["system"]["tag"] = args.tag
    hp = config["gan"]
    if args.channels:
        hp["base_channels"] = args.channels
    seed = config["system"]["seed"]
    steps = args.steps or (40 if args.smoke else hp["steps"])
    eval_every = 20 if args.smoke else hp["eval_every"]
    n_eval = 64 if args.smoke else hp["n_eval_samples"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    config["run"] = {"steps": steps, "hours": args.hours}
    log = RunLog(config["paths"]["runs_dir"], "gan" + ("_smoke" if args.smoke else ""), config, seed)

    size = config["image_size"]
    if args.smoke:
        train_set, test_set = SyntheticImages(256, size, 0), SyntheticImages(96, size, 1)
        dino = DinoFeatures(config["dino"]["size"], config["dino"]["pooling"], pretrained=False, depth=2).to(device)
        hp = {**hp, "batch_size": 16, "base_channels": 16}
    else:
        split = dict(val_percent=config["split"]["val_percent"], test_percent=config["split"]["test_percent"])
        train_set = ImageDataset(config["paths"]["abstract_art"], size=size, train=True, split="train", **split)
        test_set = ImageDataset(config["paths"]["abstract_art"], size=size, train=False, split="test", **split)
        dino = DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)
    loader = DataLoader(train_set, batch_size=hp["batch_size"], shuffle=True, drop_last=True, num_workers=0)
    test_features = features_of(dino, DataLoader(test_set, batch_size=64), device)

    generator = Generator(hp["latent_dim"], hp["base_channels"]).to(device)
    discriminator = Discriminator(hp["base_channels"]).to(device)
    average = copy.deepcopy(generator).eval()
    for p in average.parameters():
        p.requires_grad = False
    opt_g = torch.optim.Adam(generator.parameters(), lr=hp["lr"], betas=tuple(hp["betas"]))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=hp["lr"], betas=tuple(hp["betas"]))
    log.save_json("data.json", {"n_train": len(train_set), "n_test": len(test_set),
                                "generator_parameters": sum(p.numel() for p in generator.parameters()),
                                "discriminator_parameters": sum(p.numel() for p in discriminator.parameters())})
    fixed = torch.randn(64, hp["latent_dim"], generator=torch.Generator().manual_seed(seed)).to(device)
    policy = hp["augment"]

    def evaluate(step):
        g = torch.Generator().manual_seed(1000 + seed)
        batches = (average(torch.randn(min(64, n_eval - i), hp["latent_dim"], generator=g).to(device))
                   for i in range(0, n_eval, 64))
        feats = features_of(dino, batches, device)
        kid, kid_std = kernel_distance(feats, test_features, subset_size=min(1000, len(feats), len(test_features)),
                                       seed=seed)
        log.log(split="eval", step=step, frechet_distance=frechet_distance(feats, test_features),
                kernel_distance=kid, kernel_distance_std=kid_std)
        with torch.no_grad():
            save_image(average(fixed) * 0.5 + 0.5, log.dir / f"samples_{step:06d}.png", nrow=8)
        torch.save({"model": average.state_dict(), "latent_dim": hp["latent_dim"],
                    "base_channels": hp["base_channels"], "step": step, "seed": seed}, log.dir / "generator.pt")

    start, step, done = time.time(), 0, False
    while not done:
        for real in loader:
            real = real.to(device)
            z = torch.randn(len(real), hp["latent_dim"], device=device)
            fake = generator(z)
            real_in = augment(real, policy)
            if hp["r1_weight"] > 0:
                real_in.requires_grad_(True)
            d_real, d_fake = discriminator(real_in), discriminator(augment(fake.detach(), policy))
            loss_d = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            if hp["r1_weight"] > 0:
                grad = torch.autograd.grad(d_real.sum(), real_in, create_graph=True)[0]
                loss_d = loss_d + 0.5 * hp["r1_weight"] * grad.pow(2).sum(dim=(1, 2, 3)).mean()
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            loss_g = -discriminator(augment(fake, policy)).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            with torch.no_grad():
                decay = min(hp["ema"], (1 + step) / (10 + step))
                for a, p in zip(average.parameters(), generator.parameters()):
                    a.lerp_(p, 1 - decay)
                for a, b in zip(average.buffers(), generator.buffers()):
                    a.copy_(b)

            if step % hp["log_every"] == 0:
                log.log(split="train", step=step, loss_d=loss_d, loss_g=loss_g, d_real=d_real.mean(),
                        d_fake=d_fake.mean())
            step += 1
            out_of_time = args.hours is not None and time.time() - start > args.hours * 3600
            if step % eval_every == 0 or step == steps or out_of_time:
                evaluate(step)
            if step == steps or out_of_time:
                done = True
                break
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
