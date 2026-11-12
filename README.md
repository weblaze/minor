# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status (v0.4.0)

The `main` branch holds the cleaned code base and no trained model. The methods under development are on the branches listed under "Work in progress"; their runs are recorded in [docs/run_log.md on `base`](https://github.com/weblaze/minor/blob/base/docs/run_log.md).

State of the experiments on 2026-11-13 (FMA-small, 7994 tracks; 5832 abstract paintings from WikiArt; one seed; 789 test tracks):

| Finding | Evidence |
|---|---|
| A mapping trained with the structure term transfers the distance structure of an audio code to generated images on unseen tracks | S1: Mantel r 0.861 against 0.083 without the term, 0.140 with shuffled audio distances, 0.055 untrained |
| The audio code determines the structure score | S1 and Q1: 0.83 to 0.86 on the mean code, 0.26 to 0.27 on the whitened code |
| The method determines the distance of the outputs to the art distribution | Frechet distance (DINOv2 features, training art 113): Q1 402 to 425, S1 1173 |
| Most of the measured structure is specific to the DINOv2 feature space | r 0.02 to 0.31 under colour, layout and power-spectrum features of the same images |
| Real paintings assigned by a learned mapping carry part of the structure | F1: r 0.57 to 0.64 with nearest training painting; 0.05 to 0.12 for retrieval without a learned mapping |

These are single-seed results and no human evaluation has been made.

History of the code base:

- v0.1 (2025) trained a mapping network with a mean-squared error between audio and image latents that were paired by dataset index, i.e. at random. Such an objective converges to the mean image latent.
- v0.2 and v0.3 (2026) added a latent-diffusion scaffold conditioned on CLAP, with pseudo-pairs built from mood words. This used a text-supervised encoder and was never trained.

v0.4.0 removes the code that used text-supervised models, the code with known defects, the plotting scripts that produced simulated curves, and the stale checkpoints. All of it remains in the git history (last commit before the cleanup: `ff46080`). The list of removals is in [CHANGELOG.md](CHANGELOG.md).

## What is in the repository

```
abstraction/
  models/     ImageVAE (spatial latent, 128 px), AudioVAE, ConditionalUNet
  data/       ImageDataset
  utils/      config loader, checkpoint helpers, wandb run naming
approaches/
  03_byol_a_encoder/   self-supervised audio encoder (BYOL-A); untrained
visualization/
  export_wandb_curves.py   exports logged training curves from wandb
configs/base.yaml
```

The model files are untrained scaffolds. They are kept for reuse, not as results.

## Work in progress

Structure-transfer methods are developed on separate branches from a shared base:

| Branch | Method |
|---|---|
| `base` | Shared data loading, frozen encoders (MERT, DINOv2), evaluation, logging |
| `s1-structure` | S1: a mapping trained so that pairwise distances between songs match pairwise distances between generated images |
| `q1-structure` | Q1: S1 with a vector-latent generator, a feature-space realism term, a local structure term and pre-training diagnostics |
| `f1-feature` | F1: a mapping from audio codes to DINOv2 image features with nearest-painting retrieval and no image decoder; used to locate where structure is lost |

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Datasets are not tracked. See [datasets/README.md](datasets/README.md).
