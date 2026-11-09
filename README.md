# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status (v0.4.0)

No model in this repository has yet produced an image that depends on its audio input.

- v0.1 (2025) trained a mapping network with a mean-squared error between audio and image latents that were paired by dataset index, i.e. at random. Such an objective converges to the mean image latent.
- v0.2 and v0.3 (2026) added a latent-diffusion scaffold conditioned on CLAP, with pseudo-pairs built from mood words. This used a text-supervised encoder and was never trained.

v0.4.0 removes the code that used text-supervised models, the code with known defects, the plotting scripts that produced simulated curves, and the stale checkpoints. All of it remains in the git history (last commit before the cleanup: `ff46080`). The list of removals is in [CHANGELOG.md](CHANGELOG.md).

## What is in the repository

```
abstraction/
  encoders/   frozen MERT-v1-95M (audio) and DINOv2 ViT-S/14 (image, differentiable)
  data/       audio windows, hash-based splits, embedding shards, ImageDataset
  models/     Mapper (MLP), ImageVAE, AudioVAE, ConditionalUNet
  losses/     distance-correlation loss, sliced Wasserstein distance
  metrics/    Mantel test, kNN overlap, retrieval, Frechet and kernel distances, participation ratio
  utils/      config loader, checkpoints, run log (config, git commit, seed, metrics)
scripts/
  extract_mert.py   per-track MERT embeddings, shape (3 windows, 13 layers, 768)
  extract_dino.py   DINOv2 features of the art images
approaches/
  03_byol_a_encoder/   self-supervised audio encoder (BYOL-A); untrained
visualization/
  export_wandb_curves.py   exports logged training curves from wandb
configs/base.yaml
```

ImageVAE, AudioVAE and ConditionalUNet are untrained scaffolds from earlier versions. They are kept for reuse, not as results.

## Work in progress

Two structure-transfer methods are developed on separate branches from a shared base:

| Branch | Method |
|---|---|
| `base` | Shared data loading, frozen encoders (MERT, DINOv2), evaluation, logging |
| `s1-structure` | S1: a mapping trained so that pairwise distances between songs match pairwise distances between generated images |
| `q1-structure` | Q1: S1 with a vector-latent generator, a feature-space realism term, a local structure term and pre-training diagnostics |

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Datasets are not tracked. See [datasets/README.md](datasets/README.md).
