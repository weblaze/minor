# Paper 1: result tables

All numbers are copied from `docs/run_log.md`. Each table names its source entry by heading. Unless stated otherwise: 789 test tracks; r is the Pearson correlation between cosine distances of audio codes and of image features (DINOv2 CLS); Mantel tests use 9999 permutations; kNN overlap uses k = 10 (chance 0.0127); FD is the Frechet distance on DINOv2 features against 595 test images (training art against the same reference: 113.3); PR is the participation ratio. Code names: mean, standardised, whitened (see `methods.md`).

Code labels of the S1 and Q1 rows follow the run log. In `docs/results.md` the S1 `s0` row reads Mantel r 0.082; the S1 seed-0 entry of the run log reads 0.083 (see `missing.md`).

---

## (a) S1: lambda selection, variants and controls

### (a1) Lambda sweep (validation split)

Caption: S1 variant `s1`, seed 0 (one seed), mean code, 20000 steps, 797 validation tracks. Source: "2026-11-10 — S1 lambda sweep".

| `lambda_struct` | Val L_dist, step 20000 | Val L_struct, step 20000 | Val L_struct, step 5500 | Train L_struct, last batch |
|---|---|---|---|---|
| 0.1 | 0.0769 | 0.2221 | 0.2110 | 0.0614 |
| 0.3 | 0.0777 | 0.1989 | 0.1689 | 0.0401 |
| 1.0 | 0.0836 | 0.1395 | 0.1164 | 0.0328 |
| 3.0 | 0.0988 | 0.1067 | 0.0926 | 0.0276 |

Selected: `lambda_struct` = 1.0 by a rule stated after seeing these values (see `deviations_and_limitations.md`).

### (a2) Variants and controls on the test split

Caption: S1, seed 0 (one seed), mean code, 20000 steps, `lambda_struct` 1.0. FD of the S1 VAE reconstructions: 163.0. Mean pairwise feature distance of test art 0.672; PR of test art 44.8. Source: "2026-11-12 — S1 seed 0, all variants, test split".

| Variant | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Mean pairwise distance | PR |
|---|---|---|---|---|---|---|---|
| `s1` | 0.861 | 0.853 | 0.268 | 0.932 | 1172.8 | 0.342 | 14.7 |
| `s1_cycle` | 0.839 | 0.829 | 0.250 | 0.920 | 1198.2 | 0.326 | 14.4 |
| `s0` | 0.083 | 0.093 | 0.085 | 0.694 | 1108.7 | 0.289 | 15.7 |
| `shuffled` | 0.140 | 0.104 | 0.049 | 0.583 | 1571.7 | 0.156 | 15.0 |
| random M | 0.055 | 0.061 | 0.029 | 0.574 | 1877.5 | 0.118 | 18.1 |

Mantel p = 0.0001 for all trained variants; random M: p = 0.0092 (Pearson), 0.0019 (Spearman).

---

## (b) Q1: generator gate, baselines, seeds and controls

### (b1) Audio-code and image-feature diagnostics (D1 to D3)

Caption: no training; 789 test tracks (D1), 595 test images (D2). Source: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".

| Variant | Dimension | Same-track window retrieval (chance 0.0013) | CV of distances | PR |
|---|---|---|---|---|
| mean | 768 | 0.914 | 0.336 | 25.7 |
| standardised | 768 | 0.942 | 0.152 | 38.8 |
| whitened | 128 | 0.957 | 0.090 | 104.7 |

D2, image features: augmented-view retrieval 0.908 (chance 0.0017), CV of distances 0.201, PR 44.8. D3 (standardised code against image features): Wasserstein-1 distance between mean-normalised pairwise-distance distributions 0.041.

### (b2) VAE KL-weight sweep and gate (D4)

Caption: Q1 VAE seed 0 (one seed), 80 epochs; FD against 595 test images. Gate: pass if FD(prior) / FD(reconstructions) <= 2.0. Sources: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing" (row 0.02), "2026-11-10 — Q1 KL-weight sweep" (other rows).

| `kl_weight` | Val L1 | KL (nats/image) | Active dims | FD reconstructions | FD interpolations | FD mixture | FD prior | Ratio | Gate |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.155 | 240.1 | 128 | 208.5 | 456.2 | 601.3 | 974.7 | 4.68 | fail |
| 0.1 | 0.176 | 70.3 | 128 | 307.5 | 556.7 | 557.0 | 685.2 | 2.23 | fail |
| 0.3 | 0.213 | 21.5 | 126 | 576.8 | 708.5 | 681.8 | 773.2 | 1.34 | pass |
| 1.0 | 0.265 | 7.0 | 115 | 980.5 | 1053.2 | 1013.1 | 1078.2 | 1.10 | pass |
| 3.0 | 0.332 | 2.2 | 2 | 1284.4 | 1297.9 | 1290.5 | 1261.4 | 0.98 | pass |

Kernel distances at `kl_weight` 0.02 (only setting with logged kernel distances): training art 0.008, reconstructions 3.37, interpolations 11.56, Gaussian mixture 17.44, prior 43.84. Best validation L1 at 0.02: 0.1552.

### (b3) Non-learned baselines, standardised code (D5)

Caption: no training; standardised code; VAE `kl_weight` 0.02; 789 test tracks; 9999 permutations; all four Mantel p = 0.0001. Source: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".

| Baseline | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | PR |
|---|---|---|---|---|---|---|
| PCA alignment (32 axes) | 0.420 | 0.412 | 0.238 | 0.900 | 539.4 | 16.3 |
| Retrieval | 0.115 | 0.111 | 0.044 | 0.624 | 248.3 | 41.1 |

### (b4) Q1 on two VAEs, whitened code, with baselines

Caption: Q1 variant `q1`, seed 0 (one seed), whitened code, 20000 steps. VAE 0.02 failed the gate and was used with `--ignore-gate`. Baselines: whitened code. Source: "2026-11-12 — Q1 seed 0 on two VAEs, test split".

| VAE `kl_weight` | Output | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Mean pairwise distance | PR |
|---|---|---|---|---|---|---|---|---|
| 0.3 | `q1` mapper | 0.179 | 0.144 | 0.101 | 0.704 | 522.0 | 0.568 | 20.5 |
| 0.3 | PCA alignment | 0.180 | | 0.130 | 0.732 | 742.0 | 0.299 | 11.1 |
| 0.3 | retrieval | 0.056 | | 0.038 | 0.575 | 228.7 | 0.643 | 42.5 |
| 0.02 | `q1` mapper | 0.259 | 0.210 | 0.152 | 0.774 | 402.1 | 0.614 | 28.9 |
| 0.02 | PCA alignment | 0.179 | | 0.137 | 0.746 | 541.7 | 0.366 | 17.4 |
| 0.02 | retrieval | 0.049 | | 0.044 | 0.578 | 250.6 | 0.661 | 41.8 |

FD of VAE reconstructions: 576.8 (0.3), 208.5 (0.02). Final validation L_global: 0.824 (0.3), 0.768 (0.02). Mantel rho of the baselines is not in the log.

### (b5) Q1 on the mean code: three seeds and controls

Caption: Q1, mean code, 20000 steps, VAE `kl_weight` 0.02 (gate overridden). `q1`: seeds 0, 1, 2 (three seeds); `shuffled` and `no_feat`: seed 0 only. Sources: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; seed 0 of `q1` from "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code".

| Variant | Seed | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | PR | Mean pairwise distance |
|---|---|---|---|---|---|---|---|---|
| `q1` | 0 | 0.825 | 0.840 | 0.282 | 0.915 | 425.4 | 24.3 | 0.601 |
| `q1` | 1 | 0.815 | 0.834 | 0.285 | 0.920 | 409.0 | 24.9 | 0.600 |
| `q1` | 2 | 0.818 | 0.836 | 0.290 | 0.922 | 409.7 | 23.7 | 0.598 |
| `shuffled` | 0 | 0.209 | 0.208 | 0.163 | 0.823 | 405.5 | 30.1 | 0.657 |
| `no_feat` | 0 | 0.869 | 0.873 | 0.392 | 0.965 | 675.2 | 21.6 | 0.305 |

Agreement between the three `q1` seeds: Pearson correlation between image distance matrices 0.847 to 0.854; cosine similarity between image features of the same track 0.415 to 0.528, against 0.400 to 0.410 for shuffled tracks. Not run on the mean code: random M, PCA alignment, retrieval, `no_local`, `no_global`.

---

## (c) Audio-code comparison across S1 and Q1

### (c1) Both methods under three codes

Caption: seed 0 (one seed per cell), 20000 steps; Q1 on VAE `kl_weight` 0.02 (gate overridden). Sources: "2026-11-13 — S1 and Q1 on the standardised audio code" (table assembled there from "2026-11-12 — S1 seed 0, all variants, test split", "2026-11-12 — Q1 seed 0 on two VAEs, test split" and "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code").

| Method | Audio code | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | PR |
|---|---|---|---|---|---|---|---|
| S1 `s1` | mean | 0.861 | 0.853 | 0.268 | 0.932 | 1172.8 | 14.7 |
| S1 `s1` | standardised | 0.847 | 0.835 | 0.410 | 0.969 | 1364.0 | 16.1 |
| S1 `s1` | whitened | 0.272 | 0.226 | 0.178 | 0.791 | 1172.9 | 29.7 |
| Q1 `q1` | mean | 0.825 | 0.840 | 0.282 | 0.915 | 425.4 | 24.3 |
| Q1 `q1` | standardised | 0.797 | 0.786 | 0.351 | 0.955 | 422.1 | 20.9 |
| Q1 `q1` | whitened | 0.259 | 0.210 | 0.152 | 0.774 | 402.1 | 28.9 |

### (c2) Cross-code evaluation

Caption: seed 0 (one seed); post hoc; Pearson r between cosine distances of image features and of each audio code; same 789 tracks. Correlation between the two audio distance matrices (mean, whitened): 0.272. Source: "2026-11-12 — comparison of S1 and Q1 under a common audio code".

| Run (training code) | r against mean code | r against whitened code |
|---|---|---|
| S1 `s1` (mean) | 0.861 | 0.155 |
| S1 `s1_cycle` (mean) | 0.839 | 0.148 |
| S1 `s0` (mean) | 0.082 | 0.098 |
| S1 `shuffled` (mean) | 0.140 | 0.046 |
| S1 random M (mean) | 0.055 | 0.035 |
| Q1 `q1`, VAE 0.02 (whitened) | 0.160 | 0.259 |
| Q1 `q1`, VAE 0.3 (whitened) | 0.131 | 0.179 |

### (c3) Leading axes of the mean code and S1 structure

Caption: S1 `s1`, seed 0 (one seed), mean code; principal axes fitted on training windows; exploratory, script not in the repository. Variance fractions of the first six axes: 0.117, 0.073, 0.056, 0.048, 0.034, 0.031. Source: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part B.

| Axes removed from the mean code | r with S1 image distances | r with whitened-code distances |
|---|---|---|
| 0 | 0.861 | 0.272 |
| 1 | 0.369 | 0.546 |
| 2 | 0.289 | 0.580 |
| 5 | 0.112 | 0.637 |
| 10 | 0.027 | 0.670 |

Distances on the first 10 axes alone correlate 0.891 with full mean-code distances and 0.844 with S1 image distances.

---

## (d) F1 diagnosis stages

### (d1) Precursor: direct mapping to DINOv2 features

Caption: seed 0 (one seed), exploratory, script not in the repository; same mapper architecture, 4000 steps, batch 256, no decoder. FD against test art (training art 113.3); PR of test art 44.8. Source: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part C.

| Code | Distribution term | lambda | Direct: r | Direct: kNN | Direct: FD | Direct: PR | Nearest painting: r | kNN | FD | Distinct paintings |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | sliced Wasserstein | 0 | 0.493 | 0.302 | 782 | 24.8 | 0.363 | 0.079 | 256 | 503 |
| mean | sliced Wasserstein | 1 | 0.945 | 0.698 | 853 | 25.4 | 0.637 | 0.123 | 369 | 353 |
| mean | energy distance | 1 | 0.928 | 0.588 | 819 | 24.8 | 0.648 | 0.127 | 287 | 453 |
| mean | minibatch optimal transport | 1 | 0.488 | 0.202 | 578 | 9.5 | 0.380 | 0.115 | 300 | 392 |
| standardised | sliced Wasserstein | 1 | 0.988 | 0.818 | 889 | 27.1 | 0.583 | 0.133 | 340 | 390 |
| standardised | energy distance | 1 | 0.972 | 0.739 | 821 | 24.9 | 0.579 | 0.143 | 323 | 407 |
| standardised | minibatch optimal transport | 1 | 0.506 | 0.255 | 575 | 9.3 | 0.389 | 0.140 | 297 | 382 |
| whitened | sliced Wasserstein | 1 | 0.966 | 0.796 | 1205 | 108.1 | 0.066 | 0.039 | 541 | 263 |

### (d2) Staged diagnosis per code

Caption: F1 variant `f1`, seed 0 (one seed), 4000 steps, sliced Wasserstein term, `lambda_struct` 1; gallery 4666 training paintings; run before commit `aa6783c` with an unclean working tree. FD of the direct outputs: 854 to 1197 for all codes. Source: "2026-11-13 — F1 diagnosis (local)".

| Code | PR | CV | r train direct | r test direct | r test painting | kNN test painting | r gallery 500 | r gallery 2000 | r oracle painting | FD test painting | Distinct paintings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 25.7 | 0.336 | 0.965 | 0.945 | 0.644 | 0.129 | 0.597 | 0.611 | 0.650 | 386 | 338 |
| standardised | 38.8 | 0.152 | 0.991 | 0.988 | 0.572 | 0.135 | 0.519 | 0.548 | 0.571 | 343 | 388 |
| whitened | 104.7 | 0.090 | 0.980 | 0.966 | 0.066 | 0.033 | 0.047 | 0.061 | 0.085 | 546 | 266 |
| layer 0, standardised | 28.8 | 0.180 | 0.996 | 0.995 | 0.548 | 0.115 | 0.482 | 0.526 | 0.536 | 321 | 403 |
| layer 3, standardised | 32.0 | 0.165 | 0.993 | 0.990 | 0.610 | 0.150 | 0.541 | 0.593 | 0.602 | 337 | 388 |
| layer 6, standardised | 36.2 | 0.158 | 0.992 | 0.989 | 0.595 | 0.154 | 0.535 | 0.566 | 0.587 | 334 | 393 |
| layer 9, standardised | 34.8 | 0.163 | 0.991 | 0.989 | 0.615 | 0.139 | 0.554 | 0.595 | 0.627 | 334 | 401 |
| layer 12, standardised | 58.0 | 0.128 | 0.987 | 0.982 | 0.516 | 0.139 | 0.447 | 0.496 | 0.520 | 347 | 393 |

The gallery-1000 stage is configured in `methods/f1/config.yaml` but its values are not in the log.

### (d3) Direct assignment

Caption: seed 0 (one seed); exploratory, script not in the repository; soft assignment of 789 test tracks to 4666 paintings, optimised on 1 - r with temperature annealing, then hardened. Source: "2026-11-13 — F1 diagnosis (local)".

| Code | r after hardening |
|---|---|
| mean | 0.545 |
| standardised | 0.547 |
| whitened | 0.230 |

---

## (e) F1 variants and controls

Caption: F1, seed 0 (one seed), standardised code, 4000 steps; Mantel p = 0.0001 for every row. Source: "2026-11-13 — F1 variants and controls, seed 0 (local)".

| Variant | Output | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Distinct paintings |
|---|---|---|---|---|---|---|---|
| `f1` | direct | 0.988 | 0.987 | 0.819 | 0.999 | 902.8 | |
| `f1` | painting | 0.578 | 0.572 | 0.134 | 0.862 | 342.8 | 385 |
| `f0` | direct | 0.621 | 0.596 | 0.446 | 0.966 | 784.4 | |
| `f0` | painting | 0.303 | 0.288 | 0.091 | 0.746 | 254.1 | 494 |
| `shuffled` | direct | 0.558 | 0.534 | 0.389 | 0.952 | 852.5 | |
| `shuffled` | painting | 0.279 | 0.265 | 0.085 | 0.748 | 281.9 | 459 |
| random M | direct | 0.734 | 0.703 | 0.513 | 0.979 | 1785.6 | |
| random M | painting | 0.066 | 0.039 | 0.037 | 0.603 | 1336.9 | 151 |

---

## (f) F1 experiment grid, combination sweep and reruns

### (f1) Grid of 27 configurations: selected rows

Caption: F1, seed 0 (one seed), 4000 steps, test tracks; run on a working tree with `realism.py` uncommitted. Displacement: mean cosine distance between output and nearest painting. r views: mean r of the painting output against five single-layer codes. r pixel: mean r under four pixel-based feature sets. `n/a`: all tracks received the same painting. The full 27 rows per code are in the source entry. Source: "2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)".

Mean code:

| Configuration | r direct | FD direct | r painting | kNN painting | FD painting | Displacement | Distinct | r views | r pixel |
|---|---|---|---|---|---|---|---|---|---|
| `swd` | 0.945 | 854 | 0.645 | 0.131 | 384 | 0.360 | 340 | 0.445 | 0.123 |
| `struct_only` | 0.952 | 2435 | 0.139 | 0.056 | 1630 | 0.883 | 104 | 0.126 | 0.086 |
| `critic_1` | 0.824 | 1451 | 0.583 | 0.101 | 426 | 0.387 | 295 | 0.251 | 0.112 |
| `chamfer_1` | 0.516 | 2070 | 0.523 | 0.024 | 2051 | 0.038 | 62 | 0.056 | 0.489 |
| `pca64_swd` | 0.943 | 507 | 0.605 | 0.107 | 275 | 0.310 | 488 | 0.365 | 0.095 |
| `soft_0.02_swd` | 0.832 | 443 | 0.707 | 0.134 | 341 | 0.473 | 375 | 0.392 | 0.067 |
| `soft_0.02` | 0.935 | 1106 | 0.354 | 0.081 | 805 | 0.834 | 157 | 0.187 | 0.046 |
| `st_0.05_swd` | 0.696 | 736 | 0.696 | 0.077 | 736 | 0.354 | 116 | 0.275 | 0.285 |
| `st_0.02` | 0.046 | 2624 | n/a | 0.012 | 2624 | 0.312 | 1 | n/a | n/a |
| `st_0.1` | 0.644 | 2562 | 0.644 | 0.026 | 2562 | 0.287 | 27 | 0.122 | 0.531 |
| `swd_local` | 0.946 | 853 | 0.639 | 0.119 | 396 | 0.361 | 326 | 0.441 | 0.130 |
| `views_swd` | 0.854 | 862 | 0.589 | 0.130 | 354 | 0.362 | 381 | 0.503 | 0.109 |
| `noise16_swd` | 0.944 | 863 | 0.645 | 0.117 | 336 | 0.374 | 380 | 0.405 | 0.104 |

Standardised code:

| Configuration | r direct | FD direct | r painting | kNN painting | FD painting | Displacement | Distinct | r views | r pixel |
|---|---|---|---|---|---|---|---|---|---|
| `swd` | 0.988 | 897 | 0.573 | 0.136 | 344 | 0.375 | 386 | 0.503 | 0.172 |
| `struct_only` | 0.996 | 1670 | 0.160 | 0.064 | 397 | 0.844 | 539 | 0.138 | 0.026 |
| `critic_1` | 0.849 | 1545 | 0.364 | 0.134 | 374 | 0.437 | 345 | 0.309 | 0.153 |
| `chamfer_1` | 0.429 | 1323 | 0.402 | 0.042 | 1323 | 0.032 | 53 | 0.353 | 0.336 |
| `pca64_swd` | 0.986 | 518 | 0.524 | 0.120 | 254 | 0.331 | 546 | 0.455 | 0.109 |
| `soft_0.02_swd` | 0.785 | 442 | 0.612 | 0.171 | 427 | 0.625 | 328 | 0.535 | 0.139 |
| `soft_0.02` | 0.973 | 1335 | 0.063 | 0.024 | 2969 | 0.954 | 18 | 0.053 | 0.050 |
| `st_0.05_swd` | 0.628 | 454 | 0.628 | 0.151 | 454 | 0.516 | 283 | 0.557 | 0.241 |
| `st_0.02` | 0.295 | 3079 | 0.295 | 0.022 | 3079 | 0.632 | 2 | 0.254 | 0.295 |
| `st_0.1` | 0.643 | 831 | 0.643 | 0.151 | 831 | 0.423 | 223 | 0.569 | 0.169 |
| `swd_local` | 0.988 | 874 | 0.570 | 0.133 | 342 | 0.383 | 377 | 0.500 | 0.173 |
| `views_swd` | 0.982 | 883 | 0.577 | 0.139 | 354 | 0.372 | 373 | 0.518 | 0.178 |
| `noise16_swd` | 0.988 | 907 | 0.564 | 0.141 | 352 | 0.378 | 382 | 0.496 | 0.153 |

### (f2) Combination sweep: means over combinations sharing a component

Caption: F1, seed 0 (one seed), 100 combinations per code (5 heads x 5 distribution terms x 4 structure terms); each row is the mean over the 20 or 25 combinations sharing that component; painting output. Collapsed combinations (fewer than 100 distinct paintings): 7 (mean code), 5 (standardised). Combinations meeting the selection rule (at least 300 distinct paintings, FD at most 450): 58 (mean), 54 (standardised). Source: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part B.

| Component | Mean: r | kNN | FD | Distinct | Pixel r | Standardised: r | kNN | FD | Distinct | Pixel r |
|---|---|---|---|---|---|---|---|---|---|---|
| head free | 0.626 | 0.126 | 405 | 347 | 0.148 | 0.556 | 0.137 | 406 | 356 | 0.174 |
| head pca64 | 0.604 | 0.115 | 300 | 475 | 0.111 | 0.547 | 0.138 | 300 | 480 | 0.129 |
| head soft05 | 0.592 | 0.128 | 655 | 173 | 0.075 | 0.575 | 0.162 | 712 | 170 | 0.164 |
| head soft02 | 0.675 | 0.135 | 401 | 325 | 0.058 | 0.604 | 0.169 | 456 | 280 | 0.157 |
| head st05 | 0.693 | 0.110 | 498 | 268 | 0.124 | 0.611 | 0.147 | 374 | 348 | 0.186 |
| dist swd1 | 0.622 | 0.121 | 451 | 334 | 0.128 | 0.575 | 0.146 | 416 | 367 | 0.159 |
| dist swd5 | 0.643 | 0.127 | 426 | 328 | 0.082 | 0.591 | 0.151 | 442 | 333 | 0.159 |
| dist energy1 | 0.633 | 0.125 | 420 | 336 | 0.074 | 0.583 | 0.154 | 433 | 324 | 0.160 |
| dist swd1_energy1 | 0.636 | 0.125 | 419 | 340 | 0.077 | 0.581 | 0.155 | 433 | 329 | 0.161 |
| dist swd1_chamfer01 | 0.656 | 0.116 | 543 | 250 | 0.156 | 0.562 | 0.147 | 524 | 281 | 0.171 |
| struct global | 0.654 | 0.118 | 461 | 306 | 0.108 | 0.586 | 0.150 | 459 | 319 | 0.166 |
| struct global_local | 0.646 | 0.125 | 425 | 331 | 0.102 | 0.569 | 0.155 | 428 | 336 | 0.153 |
| struct global_views | 0.597 | 0.129 | 443 | 329 | 0.083 | 0.585 | 0.147 | 470 | 311 | 0.161 |
| struct global3 | 0.655 | 0.119 | 479 | 305 | 0.120 | 0.574 | 0.149 | 442 | 341 | 0.169 |

Other extremes at seed 0: lowest FD of the painting output 234 with 555 distinct paintings (`pca64__swd1_energy1__global`, mean code, r 0.611); highest kNN overlap 0.191 (`soft05__swd1_chamfer01__global3`, standardised code, r 0.596, FD 323).

### (f3) Reruns of the selected combinations

Caption: F1, five seeds (0 to 4), mean and standard deviation over seeds; selection on the seed-0 sweep by the rule in `phase1_batch.py` (at least 300 distinct paintings, FD at most 450, ranked by r of the painting output); test tracks. Source: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part C.

| Code | Combination | r painting | kNN overlap | FD | Distinct |
|---|---|---|---|---|---|
| mean | `free__swd1__global` (baseline) | 0.641 (0.013) | 0.122 (0.006) | 346 (20) | 371 |
| mean | `st05__swd1_chamfer01__global` | 0.747 (0.007) | 0.079 (0.003) | 375 (28) | 330 |
| mean | `st05__swd1_chamfer01__global_local` | 0.743 (0.023) | 0.090 (0.004) | 364 (43) | 338 |
| mean | `soft02__swd1_chamfer01__global3` | 0.737 (0.012) | 0.145 (0.004) | 340 (19) | 359 |
| mean | `st05__swd1_energy1__global` | 0.717 (0.010) | 0.128 (0.004) | 342 (9) | 358 |
| mean | `st05__energy1__global_local` | 0.704 (0.018) | 0.128 (0.004) | 371 (11) | 323 |
| standardised | `free__swd1__global` (baseline) | 0.556 (0.011) | 0.137 (0.006) | 338 (11) | 394 |
| standardised | `st05__swd1_chamfer01__global3` | 0.641 (0.006) | 0.132 (0.006) | 393 (8) | 389 |
| standardised | `st05__energy1__global` | 0.623 (0.013) | 0.164 (0.006) | 300 (3) | 387 |
| standardised | `st05__swd1_energy1__global` | 0.620 (0.007) | 0.163 (0.003) | 293 (6) | 399 |
| standardised | `st05__energy1__global_views` | 0.615 (0.014) | 0.162 (0.004) | 298 (6) | 385 |
| standardised | `soft02__swd1_chamfer01__global3` | 0.614 (0.012) | 0.181 (0.005) | 274 (16) | 455 |

---

## (g) Orientation across seeds and network sizes

### (g1) Ten seeds

Caption: F1, ten seeds per row (50 mappings), test tracks; correlation between test tracks' scores on principal axis k of the audio code and their paintings' scores on principal axis k of the art features. Skewness of the leading three axes: mean code 0.24, -0.22, 0.57; standardised code -0.27, -0.58, 0.45; art features -0.06, -0.03, 0.16. Source: "2026-11-13 — F1 orientation across ten seeds (local)".

| Code | Configuration | Abs. correlation, axis 1 (range) | Seeds with the minority sign, axis 1 | Sign split, axis 2 | r painting (range) | Painting similarity between seeds | Same orientation only | Shuffled tracks |
|---|---|---|---|---|---|---|---|---|
| mean | `swd` | 0.87 to 0.92 | 1 of 10 (seed 4) | 5 / 4, one near zero | 0.613 to 0.676 | 0.558 | 0.591 | 0.490 |
| mean | `swd`, weight 5 | 0.85 to 0.91 | 2 of 10 (seeds 4, 8) | 5 / 5 | 0.608 to 0.689 | 0.508 | 0.563 | 0.465 |
| mean | `swd`, weight 20 | 0.80 to 0.90 | 2 of 10 (seeds 4, 8) | 5 / 5 | 0.617 to 0.674 | 0.500 | 0.550 | 0.456 |
| mean | `soft_0.02_swd` | 0.25 to 0.85 | 1 of 10 (seed 4) | 5 / 5 | 0.676 to 0.714 | 0.378 | 0.387 | 0.344 |
| standardised | `swd` | 0.88 to 0.90 | 1 of 10 (seed 4) | 5 / 5 | 0.541 to 0.573 | 0.527 | 0.560 | 0.474 |

Validation distribution terms of the minority-sign seed (mean code, `swd`): sliced Wasserstein 0.0206 for seed 4 against 0.0175 to 0.0212; energy distance 0.248 against 0.202 to 0.259.

### (g2) Orientation votes over random initialisations (D-vote)

Caption: F1, configuration `swd` unless stated; seed counts per row as listed; counts of the negative sign of the audio-painting correlation on principal axis k. Source: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part A.

| Code | Mapping network | Seeds | Axis 1 negative | Axis 2 negative | Axis 3 negative | Mean abs. correlation, axes 1, 2, 3 | r painting, mean (sd) |
|---|---|---|---|---|---|---|---|
| mean | 1024 x 3 | 50 | 40 | 26 | 7 | 0.89, 0.52, 0.38 | 0.646 (0.015) |
| mean | 256 x 3 | 20 | 11 | 10 | 2 | 0.84, 0.32, 0.34 | 0.639 (0.015) |
| mean | 1024 x 1 | 20 | 14 | 11 | 0 | 0.85, 0.42, 0.32 | 0.636 (0.015) |
| mean | 2048 x 3 | 20 | 13 | 12 | 2 | 0.88, 0.53, 0.40 | 0.642 (0.017) |
| mean, `soft_0.02_swd` | 1024 x 3 | 20 | 14 | 11 | 0 | 0.65, 0.43, 0.35 | 0.703 (0.014) |
| standardised | 1024 x 3 | 50 | 16 | 26 | 24 | 0.89, 0.52, 0.35 | 0.556 (0.010) |

Pooled over the five mean-code settings: axis 1 negative in 92 of 130 runs (71 %), axis 2 in 70 of 130 (54 %), axis 3 positive in 119 of 130 (92 %). Standardised code: majority sign of axis 1 in 34 of 50 runs (68 %). The same seed gives the majority or minority sign under both codes in 44 of 50 cases; between network sizes the same seed agrees in 9 to 12 of 20 cases.

### (g3) Majority against minority orientation (D-ease, D-loss)

Caption: F1, mean code, network 1024 x 3, 40 majority and 10 minority seeds. Source: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part A.

| Quantity | Majority | Minority |
|---|---|---|
| r painting | 0.647 | 0.643 |
| Validation sliced Wasserstein | 0.0190 | 0.0188 |
| Validation energy distance | 0.2265 | 0.2249 |
| Validation structure loss | 0.0523 | 0.0525 |
| Batch distance correlation at step 100 / 300 / 1000 / 3900 | 0.935 / 0.958 / 0.979 / 0.987 | 0.934 / 0.955 / 0.979 / 0.988 |
| Batch sliced Wasserstein at step 1000 / 3900 | 0.0374 / 0.0270 | 0.0364 / 0.0268 |

The log states that the same comparison for the standardised code and other network sizes shows differences of the same size or smaller; those values are not tabled.

### (g4) Orientation in Q1

Caption: Q1, mean code, VAE 0.02; three seeds of `q1`, seed 0 of the controls. Correlation between test tracks' scores on principal axis 1 of the audio code and their images' scores on principal axis 1 of the art features. Source: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".

| Run | Axis-1 correlation |
|---|---|
| `q1`, seeds 0, 1, 2 | -0.43, -0.85, -0.91 |
| `shuffled`, seed 0 | -0.04 |
| `no_feat`, seed 0 | 0.00 |

The log does not state which of the three values belongs to which seed beyond the listed order.

---

## (h) The retrieval limit

Caption for (h1) to (h4): F1 configuration `swd`, seed 0 (one seed); exploratory, script not in the repository. Source: "2026-11-14 — F1: what limits the painting output (local, exploratory)".

### (h1) Interpolation between direct output (t = 0) and nearest painting (t = 1)

| t | Mean code: r | kNN | FD | Standardised code: r | kNN | FD |
|---|---|---|---|---|---|---|
| 0 | 0.945 | 0.696 | 854 | 0.988 | 0.817 | 897 |
| 0.25 | 0.929 | 0.621 | 651 | 0.949 | 0.703 | 663 |
| 0.5 | 0.868 | 0.369 | 484 | 0.837 | 0.424 | 471 |
| 0.75 | 0.765 | 0.221 | 396 | 0.696 | 0.223 | 367 |
| 1 | 0.645 | 0.132 | 384 | 0.573 | 0.136 | 344 |

### (h2) Gallery size (Gaussian model of the art features)

FD of 4666 Gaussian samples to test art: 123; training art: 113.

| Gallery | Mean code: displacement | r | FD | Standardised code: displacement | r | FD |
|---|---|---|---|---|---|---|
| 4666 real paintings | 0.360 | 0.645 | 384 | 0.375 | 0.573 | 344 |
| 4666 Gaussian samples | 0.383 | 0.643 | 390 | 0.395 | 0.564 | 358 |
| 46660 Gaussian samples | 0.358 | 0.697 | 330 | 0.369 | 0.623 | 310 |
| 466600 Gaussian samples | 0.338 | 0.724 | 333 | 0.349 | 0.665 | 315 |

### (h3) Mean of the k nearest paintings, displacement, collection density

| Quantity | Mean code | Standardised code |
|---|---|---|
| r, mean of k nearest paintings, k = 2 / 4 / 8 / 16 | 0.686 / 0.700 / 0.706 / 0.704 | 0.643 / 0.676 / 0.689 / 0.693 |
| FD of the k-nearest means (range over k) | 427 to 663 | 402 to 642 |
| Displacement length relative to output length | 0.84 | 0.86 |
| Cosine between displacement and output | -0.44 | -0.44 |
| r with displacement replaced by a random vector of equal length (art covariance) | 0.707 | 0.555 |

Collection density (cosine distance): test painting to nearest training painting 0.250 (quartiles 0.195, 0.251, 0.313); training painting to nearest other training painting 0.255; mean distance between two training paintings 0.687. Outputs of the mapping are at 0.36 to 0.38 from their nearest painting. Grid entry: mean cosine distance between two test-set paintings 0.672.

### (h4) Structure of the painting output in leading art axes

Caption: F1, seed 0 (one seed); exploratory, script not in the repository. Euclidean distances on projected, centred features. Cumulative variance of art features: 0.21 at 4 axes, 0.32 at 8, 0.45 at 16, 0.74 at 64. Source: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)", part A.

| Code | Configuration | Full features (cosine) | 2 axes | 4 axes | 8 axes | 16 axes | 64 axes | Residual beyond 16 axes |
|---|---|---|---|---|---|---|---|---|
| mean | `swd` | 0.645 | 0.428 | 0.536 | 0.583 | 0.593 | 0.617 | 0.383 |
| mean | `soft_0.02_swd` | 0.707 | 0.353 | 0.517 | 0.577 | 0.599 | 0.654 | 0.466 |
| standardised | `swd` | 0.573 | 0.505 | 0.558 | 0.588 | 0.583 | 0.562 | 0.120 |
| standardised | `soft_0.02_swd` | 0.612 | 0.429 | 0.575 | 0.637 | 0.634 | 0.615 | 0.129 |

---

## (i) Structure under pixel-based features

### (i1) S1 and Q1, regenerated test images

Caption: seed 0 (one seed); exploratory, scripts not in the repository; Euclidean distances between image statistics, r against the code each mapper was trained on. Sources: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part A (first two columns); "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code" (third column).

| Feature set | S1 `s1` (mean code) | Q1 `q1`, VAE 0.02 (whitened code) | Q1 `q1`, VAE 0.02 (mean code) |
|---|---|---|---|
| DINOv2 CLS (training target) | 0.861 | 0.259 | 0.825 |
| Colour mean and standard deviation (6-d) | 0.256 | 0.102 | 0.233 |
| Colour histogram (64-d) | 0.228 | 0.079 | 0.023 |
| 8 x 8 layout (192-d) | 0.280 | 0.115 | 0.307 |
| Radial power spectrum (63-d) | 0.144 | 0.078 | 0.097 |

Spread of the radial power spectrum relative to test art: S1 0.42, Q1 (whitened code) 0.66.

### (i2) F1 painting output

Caption: F1, seed 0 (one seed), standardised code, retrieved paintings. Source: "2026-11-13 — F1 variants and controls, seed 0 (local)".

| Feature set | `f1` | `f0` |
|---|---|---|
| Colour mean and standard deviation | 0.169 | 0.100 |
| Colour histogram | 0.193 | 0.109 |
| 8 x 8 layout | 0.153 | 0.092 |
| Power spectrum | 0.183 | 0.082 |

Grid entry: collapsed solutions (2 to 63 paintings) give r under pixel-based features of 0.3 to 0.55, against 0.06 to 0.19 for configurations with several hundred paintings ("2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)").

---

## (j) Audio-code axes against FMA metadata

Caption: no mapping involved; 7994 tracks (all splits), window-averaged codes, principal axes fitted on training tracks; Echo Nest attributes for 1294 tracks; metadata used for interpretation only; scripts not in the repository. Source: "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)".

### (j1) Nearest-neighbour agreement (cosine, k = 10; genre chance 0.125)

| Code | Same-genre rate | Same-artist rate |
|---|---|---|
| mean | 0.457 | 0.127 |
| standardised | 0.528 | 0.159 |
| whitened | 0.493 | 0.165 |

### (j2) Leading axes of the mean code

eta^2: fraction of axis variance explained by genre. R^2: 5-fold ridge regression from 100 low-level descriptors. rho: Spearman correlation.

| Axis | Variance fraction | Genre eta^2 | R^2 from low-level descriptors | Strongest Echo Nest attributes (rho) |
|---|---|---|---|---|
| 1 | 0.130 | 0.31 | 0.82 | acousticness +0.68, energy -0.51, danceability -0.47 |
| 2 | 0.078 | 0.19 | 0.62 | danceability +0.53, speechiness +0.31, instrumentalness -0.28 |
| 3 | 0.060 | 0.09 | 0.62 | speechiness +0.14 |
| 4 | 0.050 | 0.09 | 0.65 | energy -0.32, instrumentalness +0.22 |
| 5 | 0.036 | 0.14 | 0.61 | instrumentalness -0.18 |
| 6 | 0.032 | 0.09 | 0.45 | speechiness +0.27 |

Genre means on axis 1 (axis standard deviation 0.52): Folk +0.49, Instrumental +0.31, International +0.09, Experimental +0.05, Pop -0.02, Rock -0.11, Hip-Hop -0.39, Electronic -0.40. Strongest single low-level correlates of axis 1: spectral-contrast bands (rho up to +0.63). Standardised code, axis 1: acousticness -0.65, danceability +0.55, energy +0.50; genre eta^2 0.33; R^2 0.77.

Note: the variance fractions here (axis 1: 0.130) differ from those in (c3) (axis 1: 0.117); (c3) fits axes on training windows, (j2) on window-averaged training tracks.

---

## (k) Genre against painting style

### (k1) Seed 0

Caption: F1 configuration `swd`, seed 0 (one seed), mean / standardised code; row fractions; exploratory, script not in the repository. Gallery shares: Abstract Expressionism 0.47, Colour Field 0.27, Minimalism 0.23, Action painting 0.02. Source: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)", part B.

| Genre | Abstract Expressionism (mean / standardised) | Colour Field | Minimalism |
|---|---|---|---|
| Electronic | 0.85 / 0.87 | 0.11 / 0.09 | 0.03 / 0.04 |
| Hip-Hop | 0.82 / 0.84 | 0.11 / 0.11 | 0.04 / 0.02 |
| Rock | 0.67 / 0.71 | 0.24 / 0.24 | 0.07 / 0.06 |
| Pop | 0.63 / 0.57 | 0.27 / 0.26 | 0.10 / 0.14 |
| International | 0.48 / 0.37 | 0.35 / 0.56 | 0.15 / 0.04 |
| Experimental | 0.47 / 0.44 | 0.41 / 0.42 | 0.10 / 0.13 |
| Instrumental | 0.34 / 0.36 | 0.35 / 0.36 | 0.30 / 0.27 |
| Folk | 0.16 / 0.10 | 0.44 / 0.52 | 0.39 / 0.37 |

Cramer V: 0.285 (mean code), 0.326 (standardised code).

### (k2) Five seeds: share of Colour Field and Minimalism paintings per genre

Caption: F1 configuration `swd`, seeds 0 to 4 (five seeds); seeds share the initialisation across codes. Source: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)", part C.

| Code | Seed | r painting | Electronic | Hip-Hop | Rock | Pop | International | Experimental | Instrumental | Folk |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0 | 0.645 | 0.14 | 0.15 | 0.30 | 0.37 | 0.51 | 0.51 | 0.65 | 0.83 |
| mean | 1 | 0.659 | 0.10 | 0.20 | 0.39 | 0.37 | 0.48 | 0.60 | 0.72 | 0.84 |
| mean | 2 | 0.624 | 0.20 | 0.16 | 0.30 | 0.34 | 0.53 | 0.55 | 0.74 | 0.85 |
| mean | 3 | 0.648 | 0.14 | 0.16 | 0.43 | 0.32 | 0.49 | 0.52 | 0.65 | 0.87 |
| mean | 4 | 0.628 | 0.80 | 0.68 | 0.51 | 0.43 | 0.34 | 0.39 | 0.15 | 0.09 |
| standardised | 0 | 0.573 | 0.13 | 0.13 | 0.29 | 0.40 | 0.60 | 0.55 | 0.64 | 0.89 |
| standardised | 1 | 0.560 | 0.15 | 0.10 | 0.48 | 0.39 | 0.51 | 0.53 | 0.72 | 0.80 |
| standardised | 2 | 0.549 | 0.14 | 0.14 | 0.39 | 0.41 | 0.44 | 0.60 | 0.77 | 0.84 |
| standardised | 3 | 0.559 | 0.14 | 0.16 | 0.48 | 0.46 | 0.62 | 0.57 | 0.69 | 0.92 |
| standardised | 4 | 0.541 | 0.75 | 0.79 | 0.53 | 0.57 | 0.35 | 0.42 | 0.21 | 0.05 |

Between seeds, the two paintings assigned to the same track have mean cosine similarity 0.536 (mean code) and 0.508 (standardised), against 0.490 and 0.474 for shuffled tracks; the same painting is assigned in 1.6 % and 0.8 % of cases.
