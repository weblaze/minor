# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status

The `main` branch holds the cleaned code base and no trained model. Methods are developed on the branches listed under "Work in progress". Every run on real data is recorded in [docs/run_log.md on `base`](https://github.com/weblaze/minor/blob/base/docs/run_log.md), including runs with negative outcomes.

Data in use: FMA-small (music) and abstract paintings from four WikiArt styles. Encoders: MERT-v1-95M (audio) and DINOv2 ViT-S/14 (image), both frozen and both trained without text.

Progress as of 2026-11-13:

- Feature extraction, training and evaluation run end to end on free GPU sessions through a job runner (`scripts/kaggle/`).
- S1 and Q1 have been trained and evaluated on held-out tracks for one seed, with their controls (no structure term, shuffled audio distances, untrained mapping) and two non-learned baselines.
- The structure objective transfers the distance structure of the audio representation to the outputs on held-out tracks; the controls do not.
- Open problems identified by these runs: the distribution term does not keep generated images close to the art distribution; the choice of audio representation changes the outcome more than the choice of method; the measured structure depends on the image feature space used to measure it.
- F1 was added to separate these effects: it maps into image-feature space and assigns real paintings, without an image generator.

All results are from a single seed and no human evaluation has been made. They are preliminary and are not reported here as findings.

History of the code base:

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
  build_art_set.py  art collection from WikiArt style folders, 128 px, with manifest
  kaggle/           job launcher and runner for Kaggle kernels (see scripts/kaggle/README.md)
approaches/
  03_byol_a_encoder/   self-supervised audio encoder (BYOL-A); untrained
visualization/
  export_wandb_curves.py   exports logged training curves from wandb
configs/base.yaml
```

ImageVAE, AudioVAE and ConditionalUNet are untrained scaffolds from earlier versions. They are kept for reuse, not as results.

## Work in progress

Structure-transfer methods are developed on separate branches from a shared base:

| Branch | Method |
|---|---|
| `base` | Shared data loading, frozen encoders (MERT, DINOv2), evaluation, logging |
| `s1-structure` | S1: a mapping trained so that pairwise distances between songs match pairwise distances between generated images |
| `q1-structure` | Q1: S1 with a vector-latent generator, a feature-space realism term, a local structure term and pre-training diagnostics |
| `f1-feature` | F1: a mapping from audio codes to DINOv2 image features with nearest-painting retrieval and no image decoder; used to locate where structure is lost |

## Versioning

Releases of `main` are tagged `vX.Y.Z` (see [CHANGELOG.md](CHANGELOG.md)). Snapshots of a method branch are tagged `<method>-vX.Y.Z` (`base-`, `s1-`, `q1-`, `f1-`) and published as pre-releases; each run in the run log names the commit it was made from.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Datasets are not tracked. See [datasets/README.md](datasets/README.md).

## Authors and citation

Arunav Ganguly, Aman Dhingra, Sourabh Singh Verma. Citation metadata is in [CITATION.cff](CITATION.cff).

## Licence

No licence has been granted yet. The code is published for reading and review; reuse requires permission from the authors.
