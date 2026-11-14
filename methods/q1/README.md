# Q1: structure transfer with a sampleable generator and feature-space realism

Status: implemented and smoke-tested on synthetic data. No training run on real data has been made. This document contains no results.

## 1. Problem

Same as S1: given a collection of music tracks and an unrelated collection of abstract images, learn a mapping from a track to an image without paired examples, text, or labels. Q1 keeps the structure-transfer objective of S1 and changes the parts of S1 that the 2026-11 audit identified as likely failure points.

## 2. Differences from S1

| Component | S1 | Q1 | Reason |
|---|---|---|---|
| Image latent | spatial, 4 x 16 x 16 | vector, 128-d | a single vector output of an MLP has no spatial prior; a vector latent with a stronger KL term can be sampled |
| Generator check | none | sampleability gate before any mapping is trained | a mapping can only reach images the decoder produces away from training encodings |
| Realism term | latent sliced Wasserstein | sliced Wasserstein on DINOv2 features of decoded images (main), latent sliced Wasserstein (auxiliary) | matching latent marginals does not constrain decoded images |
| Structure term | global distance correlation | global distance correlation and a local neighbour-distribution term | Pearson correlation is dominated by large distances |
| Audio code | mean of 13 MERT layers | per-layer standardisation, centring, optional PCA whitening | layer scales differ; uncentred codes occupy a narrow cone |
| Before training | none | diagnostics D1 to D5 | checks that both geometries carry usable structure and sets the baseline scores |

## 3. Method

```
track -> 3 windows of 10 s -> MERT-v1-95M (frozen) -> 13 x 768 per window
      -> code transform (fitted on training tracks) -> a in R^d
a -> M (MLP) -> z in R^128 -> VAE decoder (frozen) -> image x (128 px)
x -> DINOv2 ViT-S/14 (frozen) -> v in R^384
```

Audio code variants (`audio_code.variant`):

- `mean`: unweighted mean of the 13 layer vectors (d = 768). Identical to S1.
- `standardised`: each layer z-scored per dimension with training statistics, then averaged (d = 768).
- `whitened`: `standardised`, projected on the leading `whiten_dim` principal axes and scaled to unit variance per axis (d = 128).

Generator: convolutional VAE with a 128-d vector latent (`train_vae.py`). Objective: L1 reconstruction + `kl_weight` x KL (mean over latent dimensions, linear warm-up) + `dino_weight` x mean squared error between DINOv2 tokens of input and reconstruction.

Mapping `M`: MLP d -> 1024 -> 1024 -> 1024 -> 128 with LayerNorm, GELU and residual connections; deterministic. Inputs are standardised with training statistics.

Objective for `M`, on a batch of B tracks (one window per track):

```
L = w_feat L_feat + w_lat L_lat + w_global L_global + w_local L_local

L_feat   = SWD({v_i}, DINOv2 features of real training art)
L_lat    = SWD({z_i}, samples mu + sigma * eps of the posteriors of real training art)
L_global = 1 - Pearson(D_a, D_v) over the B(B-1)/2 pairs
L_local  = mean_i KL(P_i || Q_i)
           P_i = softmax_j(-D_a[i,j] / (mean(D_a) * T)),  Q_i likewise from D_v,  j != i
D_a[i,j] = cosine distance between a_i and a_j
D_v[i,j] = cosine distance between v_i and v_j
```

SWD: sliced Wasserstein distance with 256 random projections. Only `M` receives gradient updates. Training uses float32; float16 autocast gave non-finite gradients through the decoder and DINOv2 in S1.

## 4. Diagnostics (run before training a mapping)

| Id | Script | Measures |
|---|---|---|
| D1 | `diagnostics.py` | same-track window retrieval and distance spread for each audio-code variant; selects the variant without labels |
| D2 | `diagnostics.py` | augmented-view retrieval, distance spread and participation ratio of DINOv2 features of the art |
| D3 | `diagnostics.py` | normalised distance quantiles, Wasserstein-1 distance between the two distance distributions, variance spectra |
| D4 | `gate.py` | Frechet and kernel distance to test art of decoded reconstructions, prior samples, interpolations and Gaussian-mixture samples; active latent dimensions |
| D5 | `baselines.py` | structure and distribution scores of the PCA-alignment and retrieval baselines |

Gate rule, fixed before any run: the VAE passes if FD(prior samples) <= `gate.max_fd_ratio` x FD(reconstructions), both against test art. `train_mapper.py` stops without a passing `gate.json` unless `--ignore-gate` is given; the override is recorded in the run config. If the gate fails, `kl_weight` is raised and the VAE retrained.

If D1 retrieval is at chance for every variant, or D2 retrieval is at chance, the corresponding encoder does not provide usable structure and no mapping is trained.

## 5. Variants and controls

| Name | Objective | Purpose |
|---|---|---|
| `q1` | all four terms | the method |
| `no_local` | w_local = 0 | contribution of the local term |
| `no_global` | w_global = 0 | contribution of the global term |
| `no_feat` | w_feat = 0 | contribution of feature-space realism |
| `shuffled` | as `q1`, D_a computed on a permutation of the batch | same loss terms, structure target uninformative |
| random M | untrained mapping (`evaluate.py` without `--mapper`) | distance preservation of a random network |
| `pca_alignment` | linear map between principal axes, no training | structure obtainable from second-order statistics alone |
| `retrieval` | nearest real training artwork under the same linear map | structure obtainable without generating images |

## 6. Hypotheses (stated before any run on real data)

- H1. On held-out tracks, `q1` yields a higher Mantel correlation and k-nearest-neighbour overlap between audio and image distances than `shuffled`, random M and `pca_alignment`, for each of three seeds.
- H2. The Frechet distance (DINOv2 features) of `q1` outputs to held-out art is lower than that of `no_feat` and within a factor of two of the VAE prior samples measured by the gate.
- H3. `q1` yields higher k-nearest-neighbour overlap than `no_local` at equal or higher Mantel correlation.
- H4. Images generated for the same track by mappings trained with different seeds are not more similar than images of different tracks (`same_track_cosine` ~ `other_track_cosine`), while the Mantel correlation between seeds is positive. Prediction: the objective determines relative structure and not the images; any perceptual probe of specific audio-visual correspondences (e.g. tempo to density) is predicted at chance.

A positive H1 shows that the constraint was optimised and generalises to unseen tracks. It does not show that the correspondence is perceptually meaningful. H4, if confirmed, is a negative result about identifiability and is reported as such.

## 7. Protocol

1. `python scripts/extract_mert.py --source <fma zip or directory>`
2. `python methods/q1/diagnostics.py`; set `audio_code.variant` to the selected variant
3. `python methods/q1/train_vae.py --seed 0`
4. `python methods/q1/gate.py --vae <vae.pt>`; on failure raise `vae.kl_weight` and repeat from 3
5. `python methods/q1/baselines.py --vae <vae.pt>`
6. `python methods/q1/train_mapper.py --vae <vae.pt> --variant <name> --seed <0|1|2>` for every variant
7. `python methods/q1/evaluate.py --vae <vae.pt> --mapper <mapper.pt>`; `python methods/q1/evaluate.py --vae <vae.pt> --seed <0|1|2>` for random M
8. `python methods/q1/compare_seeds.py <eval run> <eval run> <eval run> --out <file>` for the three `q1` seeds

Splits: tracks and images are assigned to train (80 %), validation (10 %) and test (10 %) by the MD5 hash of the track identifier or file name. All transforms and statistics are fitted on the training split. Loss weights are selected on validation losses only. The checkpoint of the final step is evaluated; there is no checkpoint selection. D1 to D3 and the gate read the test split of each collection for measurement; no parameter of `M` depends on them, but the choice of code variant and of `kl_weight` does, and this is reported with the results.

Every script accepts `--smoke`: synthetic codes and images, randomly initialised DINOv2, few steps. Smoke runs verify the code path and produce no interpretable numbers.

## 8. Outputs

Each run writes a directory under `runs/` with `config.yaml`, `meta.json` (git commit, seed, versions, device) and `metrics.jsonl`. In addition:

| Script | Files |
|---|---|
| `diagnostics.py` | `diagnostics.json` |
| `train_vae.py` | `vae.pt`, `reconstructions.png`, `result.json` |
| `gate.py` | `gate.json` (also copied next to `vae.pt`), one image grid per latent set |
| `train_mapper.py` | `mapper.pt` (weights, input statistics, code transform) |
| `evaluate.py` | `results.json`, `features.npz`, `samples.png` |
| `baselines.py` | `results.json`, `pca_alignment.png`, `retrieval.png` |

## 9. Not implemented

- DistanceGAN-style adversarial ablation.
- Evaluation with an encoder pair independent of MERT and DINOv2. All structure metrics are computed in the feature spaces that the losses optimise.
- DEAM valence/arousal analysis, descriptor probes, human evaluation.
- Stochastic mapping (one track to several images).

## 10. Limitations

- Structure is measured with the encoders used for training; agreement in these spaces is not evidence of perceptual agreement.
- The PCA-alignment baseline matches axes by variance rank with arbitrary signs; it is one member of a family of 2^k linear maps with identical latent distances.
- When two code variants tie in D1, the first in the order mean, standardised, whitened is reported as selected.
- The gate threshold (`max_fd_ratio` = 2) and all loss weights are initial values that no run has validated.
- Images are 128 px; DINOv2 is applied at 112 px (8 x 8 patches).
