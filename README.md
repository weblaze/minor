# Abstraction

Research code for music-to-abstract-image generation under two constraints:

1. No text at any point in training (no captions, no text-supervised encoders such as CLAP or CLIP).
2. No paired data and no human labels as a training signal. Human judgement is used for evaluation only.

## Status

The `main` branch holds the cleaned code base and no trained model. Since 2026-11-16 all methods are developed together on the branch [`pipeline`](https://github.com/weblaze/minor/tree/pipeline) (see "Work in progress"). Every run on real data is recorded in [docs/run_log.md on `base`](https://github.com/weblaze/minor/blob/base/docs/run_log.md), including runs with negative outcomes.

Data in use: FMA-small (music) and abstract paintings from four WikiArt styles. Encoders: MERT-v1-95M (audio) and DINOv2 ViT-S/14 (image), both frozen and both trained without text. From phase 2 also CAV-MAE-Scale++, an audio-visual model pretrained on video without text or labels.

| Phase | State on 2026-11-19 |
|---|---|
| 1: structure transfer with a VAE (S0, S1, Q1, F1, S2) | S0, S1, Q1, F1 and S2 run and evaluated; a paper draft exists; human judging page built |
| 2: S4, then S3, with G-GAN and then diffusion | G-GAN trained in three runs and still improving; S4 probe run; the GAN can paint for S1 and Q1; S3 and diffusion not started |
| 2.5 and 3 | not started |

Progress of phase 1 (as of 2026-11-14):

- Feature extraction, training and evaluation run end to end on free GPU sessions through a job runner (`scripts/kaggle/`).
- S1 and Q1 have been trained and evaluated on held-out tracks for one seed, with their controls (no structure term, shuffled audio distances, untrained mapping) and two non-learned baselines.
- The structure objective transfers the distance structure of the audio representation to the outputs on held-out tracks; the controls do not.
- Open problems identified by these runs: the distribution term does not keep generated images close to the art distribution; the choice of audio representation changes the outcome more than the choice of method; the measured structure depends on the image feature space used to measure it.
- F1 was added to separate these effects: it maps into image-feature space and assigns real paintings, without an image generator.
- With real paintings, the leading contrast of the audio representation is mapped to the leading contrast of the art collection in every seed; the orientation of that correspondence and all finer detail are not determined by distance matching alone.
- Retrieval of real paintings is limited by the sparsity of a finite collection; structure and closeness to the art distribution trade off along one curve, on which the generated-image results also lie.

Progress since 2026-11-14:

- The methods were merged into one tree with a shared front end (`abstraction/pipeline/`), a single entry point (`scripts/run.py`), smoke pipelines and a numerical regression check that every refactoring must pass with zero changed values. Code replaced by the shared front end is kept in `legacy/` folders.
- G-GAN, an unconditional image generator trained on the art collection, went through three runs. The first collapsed; with an R1 penalty and a residual architecture the second learns, and with mixed precision the third trains about three times faster. Its distance to real art (Frechet distance on DINOv2 features) is below that of the prior samples of every VAE and still falls; it is not yet as close as the VAE reconstructions.
- A GAN has no encoder. S1 and Q1 now accept it as the painter and then compare their mapped latents with samples of the GAN input distribution instead of latents of real paintings.
- S4 probe: CAV-MAE embeds all tracks and paintings. It orders tracks partly as MERT does and paintings largely as DINOv2 does, and its sound-image similarity has a part that is specific to the pair and stable across windows of a track. Used directly to pick a painting per track, it carries little of the distance structure of the music. The same similarity scores the two orientations of a mapping (D-S4).
- S2 matches the shape of the music collection to the shape of the art collection with Gromov-Wasserstein, without pairs and without training. On the mean audio code it gives every test track a different real painting and keeps the distance structure of the music more strongly than the trained F1 mappings; it is solved on the test tracks themselves. Out of sample (the shape match uses training tracks only, and each test track is placed from its audio code alone) the structure holds at Mantel r 0.72 on the mean code, but the tracks crowd onto about 250 paintings; handing out the paintings jointly (each used once) restores a different painting for every track at r 0.71 and Frechet distance 225. On the standardised code the music fits a structureless reference cloud better than it fits the art.
- Human judging pilot (4 judges, 124 answers): judges do not yet match tracks to their images above chance for S1, Q1 or F1; the pilot is too small to detect modest effects. By eye, the images generated through the VAE read as blobs and greys, and the R1 GAN is the only generator judged to look like art.
- Orientation: which way round the leading contrast of the music maps onto the art is not fixed by distance matching. Three machine-side judges now agree on it for the leading axis: the majority of fifty randomly initialised F1 mappings, the score of the video-trained model (D-S4), and the S2 shape match.
- A judging page for the human pilot of phase 1 is built: judges hear a clip and pick the picture that fits it, and pair clips and pictures by similarity.

Q1 on the mean code has three seeds and the F1 retrieval results have five to fifty; all other results are from a single seed. No human evaluation has been made yet. The results are preliminary and are not reported here as findings.

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
