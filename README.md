# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status

The `main` branch holds the cleaned code base and no trained model. Methods are developed on the branches listed under "Work in progress". Every run on real data is recorded in [docs/run_log.md on `base`](https://github.com/weblaze/minor/blob/base/docs/run_log.md), including runs with negative outcomes.

Data in use: FMA-small (music) and abstract paintings from four WikiArt styles. Encoders: MERT-v1-95M (audio) and DINOv2 ViT-S/14 (image), both frozen and both trained without text.

Progress as of 2026-11-14:

- Feature extraction, training and evaluation run end to end on free GPU sessions through a job runner (`scripts/kaggle/`).
- S1 and Q1 have been trained and evaluated on held-out tracks for one seed, with their controls (no structure term, shuffled audio distances, untrained mapping) and two non-learned baselines.
- The structure objective transfers the distance structure of the audio representation to the outputs on held-out tracks; the controls do not.
- Open problems identified by these runs: the distribution term does not keep generated images close to the art distribution; the choice of audio representation changes the outcome more than the choice of method; the measured structure depends on the image feature space used to measure it.
- F1 was added to separate these effects: it maps into image-feature space and assigns real paintings, without an image generator.
- With real paintings, the leading contrast of the audio representation is mapped to the leading contrast of the art collection in every seed; the orientation of that correspondence and all finer detail are not determined by distance matching alone.
- Retrieval of real paintings is limited by the sparsity of a finite collection; structure and closeness to the art distribution trade off along one curve, on which the generated-image results also lie.

Q1 on the mean code has three seeds and the F1 retrieval results have five to fifty; all other results are from a single seed. No human evaluation has been made. The results are preliminary and are not reported here as findings.

## How the system works

```
audio track -> middle 30 s -> three windows of 10 s
            -> MERT-v1-95M (frozen, trained without text) -> 13 layers x 768 values per window
            -> audio code (mean, standardised or whitened combination of the layers)
audio code  -> mapping network (the only trained part of a bridge)
            -> image latent  -> generator -> image            (S1, Q1)
            -> image feature -> nearest real painting         (F1)
image       -> DINOv2 ViT-S/14 (frozen, trained without text) -> feature used for training and evaluation
```

The music collection and the image collection are unrelated: no track is paired with any image. A bridge is trained with two terms. The structure term asks that tracks which are close in the audio code receive images which are close in image-feature space. The distribution term asks that the outputs, as a population, resemble the art collection. Every bridge is evaluated on held-out tracks against controls (no structure term, permuted audio distances, untrained mapping) and non-learned baselines.

The components are separate and replaceable: encoder, audio code, bridge, generator, evaluation.

## Research plan

| Phase | Content |
|---|---|
| 1 | Structure-transfer bridges (S0, S1, Q1, F1, S2) with a VAE generator and with retrieval of real paintings; a closing sweep over combinations of mapping techniques |
| 2 | Bridges that add a signal beyond distances without human involvement: S4 (how sound and sight co-occur in real-world video), then S3 (a visual language invented by the generator). Each with a GAN first, then diffusion; comparison of VAE, GAN and diffusion |
| 2.5 | Settling the orientation of the mapping by machine-side tests, and aligning clusters of tracks to clusters of paintings, applied to all methods |
| 3 | Bridges with human involvement: S5 (human-made co-occurrence), S6 (human preference); use of the temporal sequence within a track |
| After all methods | Other text-free encoders; validation against emotion ratings, psychophysics and colour psychology; creative extensions |

Human judging closes each phase. The full plan and the meaning of every identifier are in [docs/roadmap.md on `base`](https://github.com/weblaze/minor/blob/base/docs/roadmap.md).

## Documentation

| Document | Branch | Content |
|---|---|---|
| [docs/roadmap.md](https://github.com/weblaze/minor/blob/base/docs/roadmap.md) | `base` | Phases, identifiers of bridges, generators and tests |
| [docs/run_log.md](https://github.com/weblaze/minor/blob/base/docs/run_log.md) | `base` | Every run on real data: job, commit, hardware, results, deviations, negative results |
| [docs/results.md](https://github.com/weblaze/minor/blob/base/docs/results.md) | `base` | One generated table of all evaluated runs of all methods |
| [methods/s1/README.md](https://github.com/weblaze/minor/blob/s1-structure/methods/s1/README.md) | `s1-structure` | S1: method, variants, hypotheses, protocol |
| [methods/q1/README.md](https://github.com/weblaze/minor/blob/q1-structure/methods/q1/README.md) | `q1-structure` | Q1: method, diagnostics, gate, baselines, protocol |
| [methods/f1/README.md](https://github.com/weblaze/minor/blob/f1-feature/methods/f1/README.md) | `f1-feature` | F1: method, staged diagnosis, experiment grid |
| [scripts/kaggle/README.md](https://github.com/weblaze/minor/blob/base/scripts/kaggle/README.md) | `base` | Job runner for free GPU sessions |
| [CHANGELOG.md](CHANGELOG.md) | `main` | Versions and additions per branch |

## Reproducibility

- Every run writes a directory with its configuration, git commit, seed, software versions and metrics.
- Train, validation and test splits are fixed by a hash of the track identifier or file name.
- `python scripts/smoke.py` (branch `base` and method branches) runs each method end to end on synthetic data to verify the code path.
- `python scripts/results.py table` rebuilds the results table from the run directories; `python scripts/results.py agreement` compares runs on the same tracks.
- Protocol rules are fixed before runs; where a run departs from them the run log states the deviation.

## History of the code base


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
