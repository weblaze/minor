# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status (2026-11-27)

The `main` branch holds the cleaned code base and no trained model. Since 2026-11-16 all methods are developed together on the branch [`pipeline`](https://github.com/weblaze/minor/tree/pipeline). Every run on real data, including runs with negative outcomes, is recorded in [docs/run_log.md on `base`](https://github.com/weblaze/minor/blob/base/docs/run_log.md).

Data: FMA-small (8,000 music clips of 30 s) and about 5,800 abstract paintings from four WikiArt styles at 128 px. Encoders, all frozen and none trained with text: MERT-v1-95M (audio), DINOv2 ViT-S/14 (image), CAV-MAE-Scale++ (audio-visual, pretrained on video without text or labels).

| Phase | State |
|---|---|
| 1: structure transfer with a VAE and with retrieval of real paintings (S0, S1, Q1, F1, S2) | Experiments complete; write-up in progress |
| 2: S4 and S3, with a GAN first, then other generators | GAN stage complete; other generator families are being tried now |
| 2.5 and 3: orientation, clusters, time, human-made signals | Orientation and cluster alignment partly tested; the rest not started |

Since 2026-11-22 a candidate system must draw its image. Methods that pick an existing painting are kept as references and teachers, not as candidates.

## Findings so far

The findings below held across seeds or were far larger than their controls. All are machine measurements on held-out tracks. No human evaluation has been completed, so none of them is yet a claim about what people perceive.

- Without paired data and without text, a trained mapping can carry the distance structure of a music collection onto generated images: tracks that are close in the audio representation receive images that are close in image-feature space, on tracks never seen in training. Controls with permuted audio distances or an untrained mapping do not.
- Generated images can carry enough of a track to identify it. A reader trained on the images names the correct held-out track far above chance, and a fresh reader built on a different image encoder, which played no part in training, still does. Identifiability and similarity structure can be obtained in the same system.
- The leading contrast of the music (roughly acoustic against electronic and high-energy) aligns with the leading contrast of the art collection in every seed. Which way round it aligns is not fixed by distance matching alone.
- The choice of audio representation changes the outcome more than the choice of method.
- Faithfulness to the music's structure and closeness to real art trade off against each other. Picking existing paintings is limited by how sparsely a finite collection covers the space.
- Every system that draws through the current GAN produces many more near-identical images than real art contains. A repetition penalty reduces this but does not close the gap; it is the main open problem of the generator stage.
- Some routes do not work: a drawing network trained to reproduce the generator settings of a chosen image, grouping both collections into clusters before matching them, and a video-trained audio-visual model used on its own as the bridge.
- Generation is deterministic: the same audio always gives the same image.

## Preliminary observations

From one or two seeds, or from a single check:

- Without any genre or style labels, genre and painting style line up: electronic tracks go mostly to Abstract Expressionism, folk mostly to Colour Field and Minimalism. This follows from the alignment of the leading contrasts.
- Longer training improves structure up to a point; beyond it the images adapt to the reader they were trained with rather than carrying more of the music.
- Matching the shape of the music collection to the shape of the art collection, without any training, gives strong structure.
- How much of the measured structure is specific to the image encoder used to measure it is not settled.

## Not yet shown

- Whether people perceive the correspondence. A small pilot (4 judges) was too small to detect it either way.
- Whether the images read as art. The best drawing systems are judged "partly" so far.
- Whether the mapping carries more than energy and genre.

## How the system works

```
audio track -> middle 30 s -> three windows of 10 s
            -> MERT-v1-95M (frozen, trained without text) -> 13 layers x 768 values per window
            -> audio code (mean, standardised or whitened combination of the layers)
audio code  -> mapping network (the only trained part of a bridge)
            -> image latent  -> generator (VAE or GAN) -> image   (S1, Q1)
            -> image feature -> nearest real painting             (F1)
audio codes + art features -> shape matching, no training -> painting per track   (S2)
audio track + painting -> CAV-MAE (frozen, pretrained on video) -> similarity      (S4)
audio code  -> mapping network -> GAN -> image; a reader learns to name the track from the image   (S3)
image       -> DINOv2 ViT-S/14 (frozen, trained without text) -> feature used for training and evaluation
```

The music collection and the image collection are unrelated: no track is paired with any image. A bridge is trained with two terms. The structure term asks that tracks which are close in the audio code receive images which are close in image-feature space. The distribution term asks that the outputs, as a population, resemble the art collection. Every bridge is evaluated on held-out tracks against controls (no structure term, permuted audio distances, untrained mapping) and non-learned baselines.

The components are separate and replaceable: encoder, audio code, bridge, generator, evaluation.

## Research plan

Order set on 2026-11-24:

| Step | Content |
|---|---|
| Now | Other generator techniques (GAN losses and architectures, diffusion and flow models); if they need more data, the data step moves up |
| Next | Orientation anchoring and cluster alignment on all candidate systems, then use of the time sequence within a track on the best ones |
| Then | Larger and more varied music and art data; retraining of the candidates and of methods that may have been held back by data |
| Last stand | S5 (human-made co-occurrence), S6 (human preference), D-human, D-complexity |
| End of research | Higher resolution, multi-seed runs and human judging of the winning systems |
| After all methods | Other text-free encoders; validation against emotion ratings, psychophysics and colour psychology; creative extensions |

The meaning of every identifier is in [docs/roadmap.md on `base`](https://github.com/weblaze/minor/blob/base/docs/roadmap.md).

## Documentation

| Document | Branch | Content |
|---|---|---|
| [docs/roadmap.md](https://github.com/weblaze/minor/blob/base/docs/roadmap.md) | `base` | Phases, identifiers of bridges, generators and tests |
| [docs/run_log.md](https://github.com/weblaze/minor/blob/base/docs/run_log.md) | `base` | Every run on real data: job, commit, hardware, results, deviations, negative results |
| [docs/results.md](https://github.com/weblaze/minor/blob/base/docs/results.md) | `base` | One generated table of all evaluated runs of all methods |
| [methods/s1/README.md](https://github.com/weblaze/minor/blob/s1-structure/methods/s1/README.md) | `s1-structure` | S1: method, variants, hypotheses, protocol |
| [methods/q1/README.md](https://github.com/weblaze/minor/blob/q1-structure/methods/q1/README.md) | `q1-structure` | Q1: method, diagnostics, gate, baselines, protocol |
| [methods/f1/README.md](https://github.com/weblaze/minor/blob/f1-feature/methods/f1/README.md) | `f1-feature` | F1: method, staged diagnosis, experiment grid |
| [docs/pipeline.md](https://github.com/weblaze/minor/blob/pipeline/docs/pipeline.md) | `pipeline` | Code layout, running a stage, adding an audio code, bridge, generator or evaluation, Kaggle jobs, regression check |
| [methods/gan/README.md](https://github.com/weblaze/minor/blob/pipeline/methods/gan/README.md) | `pipeline` | G-GAN: architecture, regularisers, training |
| [methods/s2/README.md](https://github.com/weblaze/minor/blob/pipeline/methods/s2/README.md) | `pipeline` | S2: blind matching by Gromov-Wasserstein |
| [methods/s4/README.md](https://github.com/weblaze/minor/blob/pipeline/methods/s4/README.md) | `pipeline` | S4: CAV-MAE probe and the D-S4 orientation test |
| [scripts/kaggle/README.md](https://github.com/weblaze/minor/blob/base/scripts/kaggle/README.md) | `base` | Job runner for free GPU sessions |
| [CHANGELOG.md](CHANGELOG.md) | `main` | Versions and additions per branch |

## Reproducibility

- Every run writes a directory with its configuration, git commit, seed, software versions and metrics.
- Train, validation and test splits are fixed by a hash of the track identifier or file name.
- `python scripts/smoke.py` runs each method end to end on synthetic data to verify the code path; `python scripts/regression.py --compare <baseline>` (branch `pipeline`) checks that a change leaves every smoke result unchanged.
- Training and feature extraction run as jobs on free Kaggle GPUs; a job runs the pushed head of its branch and records the commit.
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

All methods are developed on one branch since 2026-11-16; the earlier method branches are kept as they were:

| Branch | Content |
|---|---|
| `pipeline` | Working branch: every method below plus S2, S4 and G-GAN, the shared front end, the judging material |
| `base` | Shared data loading, frozen encoders (MERT, DINOv2), evaluation, logging; the run log, roadmap and paper draft |
| `s1-structure` | S1: a mapping trained so that pairwise distances between songs match pairwise distances between generated images |
| `q1-structure` | Q1: S1 with a vector-latent generator, a feature-space realism term, a local structure term and pre-training diagnostics |
| `f1-feature` | F1: a mapping from audio codes to DINOv2 image features with nearest-painting retrieval and no image decoder; used to locate where structure is lost |
| `g-gan` | G-GAN as first developed (merged into `pipeline`) |

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
