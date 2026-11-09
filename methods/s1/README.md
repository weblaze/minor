# S1: structure-preserving unpaired mapping

Status: implemented and smoke-tested on synthetic data. No training run on real data has been made. This document contains no results.

## 1. Problem

Given a collection of music tracks and an unrelated collection of abstract images, learn a mapping from a track to an image without paired examples, text, or labels. S1 tests one hypothesis about what can constrain such a mapping: the pairwise distance structure of the music collection.

## 2. Method

```
track -> 3 windows of 10 s -> MERT-v1-95M (frozen) -> a in R^768
a -> M (MLP) -> z in R^(4 x 16 x 16) -> VAE decoder (frozen) -> image x (128 px)
x -> DINOv2 ViT-S/14 (frozen) -> v in R^384
```

- Audio code `a`: unweighted mean of the 13 MERT hidden layers, each averaged over time. One code per window.
- Generator: a convolutional VAE with a spatial latent, trained on the image collection only (`train_vae.py`). Objective: L1 reconstruction, KL term with linear warm-up, and mean squared error between DINOv2 tokens of input and reconstruction.
- Mapping `M`: MLP 768 -> 1024 -> 1024 -> 1024 -> 1024 with LayerNorm, GELU and residual connections; deterministic. Inputs are standardised with training-set statistics. Outputs are latents in units of the standard deviation of the training latents.

Objective for `M`, computed on a batch of B tracks (one window per track):

```
L = L_dist + lambda * L_struct (+ mu * L_cycle)

L_dist   = sliced Wasserstein distance between {z_i} and latents of real images
L_struct = 1 - Pearson(D_a, D_v) over the B(B-1)/2 pairs
           D_a[i,j] = cosine distance between a_i and a_j
           D_v[i,j] = cosine distance between v_i and v_j
L_cycle  = || I(z_i) - a_i ||^2 for an inverse network I (ablation only)
```

Only `M` (and `I`, if used) receives gradient updates. MERT, the VAE and DINOv2 are frozen.

## 3. Variants and controls

| Name | Objective | Purpose |
|---|---|---|
| `s1` | L_dist + lambda L_struct | the method |
| `s0` | L_dist + mu L_cycle | objective of the 2025 version; no structure term |
| `s1_cycle` | L_dist + lambda L_struct + mu L_cycle | tests whether the cycle term adds anything |
| `shuffled` | as `s1`, with D_a computed on a permutation of the batch | same loss terms, structure target uninformative |
| random M | untrained mapping (`evaluate.py` without `--mapper`) | distance preservation of a random network |

## 4. Hypotheses

- H1. On held-out tracks, `s1` yields a higher Mantel correlation and k-nearest-neighbour overlap between audio and image distances than `s0`, `shuffled` and random M, for each of three seeds.
- H2. The Frechet distance (DINOv2 features) of `s1` outputs to held-out art is not substantially larger than that of `s0`.

A positive H1 shows that the constraint was optimised and generalises to unseen tracks. It does not show that the correspondence is perceptually meaningful; that requires the human evaluation, which is not part of this code.

## 5. Protocol

1. `python scripts/extract_mert.py --source <fma zip or directory>`
2. `python methods/s1/train_vae.py --seed 0` (one VAE, shared by all mapping runs)
3. `python methods/s1/train_mapper.py --vae <vae.pt> --variant <name> --seed <0|1|2>` for every variant
4. `python methods/s1/evaluate.py --vae <vae.pt> --mapper <mapper.pt>`; `python methods/s1/evaluate.py --vae <vae.pt> --seed <0|1|2>` for random M

Splits: tracks and images are assigned to train (80 %), validation (10 %) and test (10 %) by the MD5 hash of the track identifier or file name. `lambda` is selected from {0.1, 0.3, 1, 3} using validation L_struct and validation L_dist only. The checkpoint of the final step is evaluated; there is no checkpoint selection. Test data are used by `evaluate.py` only.

Every run writes a directory under `runs/` with `config.yaml`, `meta.json` (git commit, uncommitted-changes flag, seed, software versions, device), `metrics.jsonl` and, for evaluation, `results.json` and `features.npz`.

`--smoke` replaces the data with synthetic codes and images and DINOv2 with a randomly initialised network. It verifies that the code runs; its numbers have no meaning.

## 6. Evaluation (test split, one image per track from the mean of its window codes)

- Mantel correlation between D_a and D_v, Pearson and Spearman, with a one-sided permutation p-value (9,999 permutations).
- k-nearest-neighbour overlap (k = 10) with its chance level k/(n-1); trustworthiness (Venna and Kaski, 2001).
- Frechet distance and kernel distance between DINOv2 features of generated images and of test-split art. Reference values: training art against test art (lower bound set by sampling), VAE reconstructions against test art (bound set by the generator).
- Mean pairwise distance and participation ratio of the generated features (collapse indicators).
- Genre leakage (optional): cross-validated accuracy of a linear probe for genre on image features and on audio codes.

## 7. Differences from `S1_PLAN.md`

- Splits are hash-based instead of the official FMA split, so that the same code applies to any audio collection.
- Mixed precision is disabled for the mapping stage. With float16, gradients through the decoder and DINOv2 were non-finite in the smoke run.
- Not implemented: the neighbourhood-loss variant, single-layer MERT variants, the AudioVAE ablation, the DEAM valence/arousal analysis, the audio/image descriptor probes, and the human study.

## 8. Known limitations

- L_struct can be satisfied by variation along a single image attribute. The participation ratio of the generated features is reported to detect this.
- The audio and image distances used for evaluation come from the same encoders as the training loss. Agreement measured with them is not independent evidence.
- On a 4 GB GPU a mapping step with batch size 64 takes about 3 s (measured on a GTX 1650 in the smoke run); the planned number of steps requires a larger GPU.
