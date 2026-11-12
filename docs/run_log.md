# Run log

Chronological record of runs on real data. One entry per job. Raw outputs are kept in the Kaggle kernel named in each entry and, when fetched, under `runs/kaggle/<slug>/` (not tracked).

## 2026-11-10 — feature extraction

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/features.yaml` |
| Kernel | `arunavganguly/abstraction-features`, version 2 |
| Commit | `7f190ac` (`base`) |
| Hardware | Tesla T4, 15360 MiB |
| Software | torch 2.11.0+cu130, transformers 5.20.2 |
| Result | completed |

Art collection (`build_art_set.py`, source `steubk/wikiart`, 128 x 128 px):

| Style | Images |
|---|---|
| Abstract_Expressionism | 2782 |
| Color_Field_Painting | 1615 |
| Minimalism | 1337 |
| Action_painting | 98 |
| Total | 5832 |

Unreadable files: 0. Duration: 606 s. No de-duplication was performed.

MERT embeddings (`extract_mert.py`, source `imsparsh/fma-free-music-archive-small-medium`, directory `fma_small`):

- mp3 files found: 8000.
- Tracks embedded: 7994; array per track (3 windows, 13 layers, 768), float16; 8 shards, 479 MB.
- Tracks skipped: 6. `098565`, `098567`, `098569` shorter than 30 s; `099134`, `108925`, `133297` not decodable. These are the six files listed as defective in the FMA documentation for `fma_small`.
- Hidden states were read with forward hooks (`route = hooks`); the model output carried `hidden_states = None` under transformers 5.20.2.
- Duration: 3052 s.

Earlier attempts:

- Version 1 (commit `afec30a`): art collection built; extraction failed with `TypeError` on `hidden_states = None`. Fixed in `7f190ac`.
- `arunavganguly/abstraction-mert-check-cpu` (commit `7f190ac`, CPU): 16 tracks embedded, output shape (16, 3, 13, 768), all values finite.

## 2026-11-10 — S1 VAE and mapper timing

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/s1_vae.yaml` |
| Kernel | `arunavganguly/abstraction-s1-vae`, version 1 |
| Commit | `1c4709e` (`s1-structure`) |
| Hardware | 2 x Tesla T4, 15360 MiB; one used |
| Result | completed |

VAE (`train_vae.py --seed 0`, 80 epochs, 11600 steps, 5.84 M parameters):

- Images: 4666 train, 571 validation.
- Best validation L1: 0.0763 (epoch 78). Final training KL (mean over latent elements): 4.39 at weight 0.001.
- Standard deviation of training latents: 1.093.
- Duration: 3889 s.

Mapper (`train_mapper.py --variant s1 --seed 0 --steps 200`, timing run only; not a result):

- Tracks: 6408 train, 797 validation. Batch 64.
- 0.49 s per step (steps 20 to 180). Projected duration of 20000 steps: 2.7 h.
- Validation at step 200: L_dist 0.218, L_struct 0.271.

## 2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/q1_vae.yaml` |
| Kernel | `arunavganguly/abstraction-q1-vae`, version 1 |
| Commit | `43a2389` (`q1-structure`) |
| Hardware | 2 x Tesla T4, 15360 MiB; one used |
| Result | completed; gate failed |

D1, audio code variants (789 test tracks, same-track window retrieval, chance 0.0013):

| Variant | Dimension | Retrieval | CV of distances | Participation ratio |
|---|---|---|---|---|
| mean | 768 | 0.914 | 0.336 | 25.7 |
| standardised | 768 | 0.942 | 0.152 | 38.8 |
| whitened | 128 | 0.957 | 0.090 | 104.7 |

Selected by D1: `whitened`. The configuration of this job was `standardised`; the VAE and gate do not depend on the variant, the baselines and the mapper timing run do.

D2, image features (595 test images): augmented-view retrieval 0.908 (chance 0.0017), CV of distances 0.201, participation ratio 44.8.

D3 (`standardised` codes against image features): Wasserstein-1 distance between mean-normalised pairwise-distance distributions 0.041.

VAE (`train_vae.py --seed 0`, `kl_weight` 0.02, 80 epochs, 11600 steps, 15.66 M parameters):

- Best validation L1: 0.1552. KL at the last validation: 240.1 nats per image (1.88 per latent dimension).
- Duration: 3908 s.

D4, gate (DINOv2 features, reference 595 test images):

| Set | n | Frechet distance | Kernel distance |
|---|---|---|---|
| Training art | 4666 | 113.3 | 0.008 |
| Reconstructions | 4666 | 208.5 | 3.37 |
| Interpolations | 2000 | 456.2 | 11.56 |
| Gaussian mixture (10 components) | 2000 | 601.3 | 17.44 |
| Prior | 2000 | 974.7 | 43.84 |

- Aggregate posterior: mean |mean of mu| 0.133, mean variance of mu 0.847, active dimensions 128 of 128.
- FD(prior) / FD(reconstructions) = 4.68; threshold 2.0. Gate failed.

D5, baselines (`standardised` codes, 789 test tracks, 9999 permutations, k = 10, chance overlap 0.0127):

| Baseline | Mantel r (Pearson) | Mantel rho (Spearman) | kNN overlap | Trustworthiness | Frechet distance | Participation ratio |
|---|---|---|---|---|---|---|
| PCA alignment (32 axes) | 0.420 | 0.412 | 0.238 | 0.900 | 539.4 | 16.3 |
| Retrieval | 0.115 | 0.111 | 0.044 | 0.624 | 248.3 | 41.1 |

All four Mantel p-values equal 0.0001, the minimum for 9999 permutations. Retrieval used 595 distinct artworks.

Mapper (`train_mapper.py --variant q1 --seed 0 --steps 200 --ignore-gate`, timing run only; not a result):

- 0.49 s per step (steps 20 to 180). Projected duration of 20000 steps: 2.7 h.

Consequences, as fixed in `methods/q1/README.md`:

- `audio_code.variant` is set to `whitened`; the baselines are to be recomputed with it.
- `vae.kl_weight` is raised and the VAE retrained until the gate passes.

## 2026-11-10 — Q1 KL-weight sweep

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/q1_kl_sweep.yaml` |
| Kernel | `arunavganguly/abstraction-q1-kl-sweep`, version 1 |
| Commit | `7c5da53` (`q1-structure`) |
| Hardware | 2 x Tesla T4, one chain per GPU |
| Result | completed; duration 7605 s |

VAE seed 0, 80 epochs, followed by the gate. Frechet distances (FD) on DINOv2 features against 595 test images; FD of training art against the same reference is 113.3. The row for 0.02 is from the previous entry.

| `kl_weight` | Val L1 | KL (nats/image) | Active dims | FD reconstructions | FD interpolations | FD mixture | FD prior | Ratio | Gate |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.155 | 240.1 | 128 | 208.5 | 456.2 | 601.3 | 974.7 | 4.68 | fail |
| 0.1 | 0.176 | 70.3 | 128 | 307.5 | 556.7 | 557.0 | 685.2 | 2.23 | fail |
| 0.3 | 0.213 | 21.5 | 126 | 576.8 | 708.5 | 681.8 | 773.2 | 1.34 | pass |
| 1.0 | 0.265 | 7.0 | 115 | 980.5 | 1053.2 | 1013.1 | 1078.2 | 1.10 | pass |
| 3.0 | 0.332 | 2.2 | 2 | 1284.4 | 1297.9 | 1290.5 | 1261.4 | 0.98 | pass |

Observations:

- The lowest weight that passes the gate is 0.3. By the rule in `methods/q1/README.md` this VAE is selected.
- The ratio falls because FD(reconstructions) rises (208.5 to 576.8 between 0.02 and 0.3), not because FD(prior) falls to the level of reconstructions at 0.02. The lowest FD(prior) in the sweep is 685.2 (weight 0.1); no setting brings prior samples within a factor of six of training art.
- At weight 3.0 the posterior collapses (2 active dimensions of 128).
- The mapper is trained against posterior samples, not prior samples. FD(interpolations) is lowest at weight 0.02 (456.2).

Limitation of the gate rule exposed by this sweep: a ratio criterion can be met by degrading reconstructions. The rule is kept as written; the selected VAE (0.3) and the VAE with the best reconstructions (0.02, gate overridden) are both carried to a seed-0 mapper run, and the second is reported as a deviation from the protocol.

## 2026-11-10 — S1 lambda sweep

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/s1_lambda.yaml` |
| Kernel | `arunavganguly/abstraction-s1-lambda`, version 1 |
| Commit | `e040aba` (`s1-structure`) |
| Hardware | 2 x Tesla T4, one chain per GPU |
| Result | completed; duration 18953 s |

Variant `s1`, seed 0, 20000 steps, VAE of `arunavganguly/abstraction-s1-vae`. Validation losses on 797 tracks; training values are the last logged batch.

| `lambda_struct` | Val L_dist, step 20000 | Val L_struct, step 20000 | Val L_struct, step 5500 | Train L_struct, last batch |
|---|---|---|---|---|
| 0.1 | 0.0769 | 0.2221 | 0.2110 | 0.0614 |
| 0.3 | 0.0777 | 0.1989 | 0.1689 | 0.0401 |
| 1.0 | 0.0836 | 0.1395 | 0.1164 | 0.0328 |
| 3.0 | 0.0988 | 0.1067 | 0.0926 | 0.0276 |

Observations:

- Validation L_struct decreases and validation L_dist increases monotonically with lambda; no value dominates another.
- For every lambda, validation L_struct is lower at step 5500 than at step 20000 while training L_struct keeps falling. The structure term overfits the 6408 training tracks. The protocol evaluates the final step; this is not changed.
- Selection rule, stated after seeing these values: the largest lambda whose validation L_dist is within 10 % of the lowest validation L_dist in the sweep. Lowest 0.0769; lambda 1.0 gives 0.0836 (+8.7 %), lambda 3.0 gives 0.0988 (+28.5 %). Selected: `lambda_struct` = 1.0, the configured value. The run at lambda 1.0 is the seed-0 `s1` run.
- Mean step time 0.46 s; 2.6 h per run.

## 2026-11-12 — S1 seed 0, all variants, test split

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/s1_seed0.yaml` |
| Kernel | `arunavganguly/abstraction-s1-seed0`, version 1 |
| Commit | `8feff77` (`s1-structure`) |
| Hardware | 2 x Tesla T4, one chain per GPU |
| Result | completed; duration 12163 s |

Seed 0, 20000 steps, `lambda_struct` 1.0, audio code: mean of 13 layers (768-d). The `s1` mapper is the lambda 1.0 run of `abstraction-s1-lambda`. 789 test tracks, 9999 permutations, k = 10 (chance overlap 0.0127). Frechet distance (FD) against 595 test images; training art 113.3, VAE reconstructions 163.0. Mean pairwise feature distance of test art 0.672; participation ratio of test art 44.8.

| Variant | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Mean pairwise distance | Participation ratio |
|---|---|---|---|---|---|---|---|
| `s1` | 0.861 | 0.853 | 0.268 | 0.932 | 1172.8 | 0.342 | 14.7 |
| `s1_cycle` | 0.839 | 0.829 | 0.250 | 0.920 | 1198.2 | 0.326 | 14.4 |
| `s0` | 0.083 | 0.093 | 0.085 | 0.694 | 1108.7 | 0.289 | 15.7 |
| `shuffled` | 0.140 | 0.104 | 0.049 | 0.583 | 1571.7 | 0.156 | 15.0 |
| random M | 0.055 | 0.061 | 0.029 | 0.574 | 1877.5 | 0.118 | 18.1 |

Mantel p = 0.0001 for all trained variants (minimum for 9999 permutations); random M: p = 0.0092 (Pearson), 0.0019 (Spearman).

Observations:

- H1 holds for seed 0: `s1` exceeds `s0`, `shuffled` and random M on Mantel correlation and kNN overlap.
- The cycle term does not improve any structure score.
- Outputs of every variant are far from the art distribution: FD of 1109 to 1878 against 163 for VAE reconstructions, and about half the feature spread of test art. `s0`, which has only the latent distribution term, is at 1109; the latent sliced Wasserstein term does not place outputs on the art distribution. Sample grids show texture without spatial composition.
- One seed. Seeds 1 and 2 are not run.

## 2026-11-12 — Q1 seed 0 on two VAEs, test split

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/q1_seed0_vae_pair.yaml` |
| Kernel | `arunavganguly/abstraction-q1-seed0-vae-pair`, version 1 |
| Commit | `eaffe89` (`q1-structure`) |
| Hardware | 2 x Tesla T4, one chain per GPU |
| Result | completed; duration 10910 s |

Variant `q1`, seed 0, 20000 steps, audio code: whitened (128-d). VAE `kl_weight` 0.3 passed the gate; `kl_weight` 0.02 failed it and was used with `--ignore-gate` (deviation from the protocol). Same test tracks, permutations and references as above.

| VAE `kl_weight` | Output | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Mean pairwise distance | Participation ratio |
|---|---|---|---|---|---|---|---|---|
| 0.3 | `q1` mapper | 0.179 | 0.144 | 0.101 | 0.704 | 522.0 | 0.568 | 20.5 |
| 0.3 | PCA alignment | 0.180 | | 0.130 | 0.732 | 742.0 | 0.299 | 11.1 |
| 0.3 | retrieval | 0.056 | | 0.038 | 0.575 | 228.7 | 0.643 | 42.5 |
| 0.02 | `q1` mapper | 0.259 | 0.210 | 0.152 | 0.774 | 402.1 | 0.614 | 28.9 |
| 0.02 | PCA alignment | 0.179 | | 0.137 | 0.746 | 541.7 | 0.366 | 17.4 |
| 0.02 | retrieval | 0.049 | | 0.044 | 0.578 | 250.6 | 0.661 | 41.8 |

FD of VAE reconstructions: 576.8 (0.3), 208.5 (0.02). Final validation losses: L_global 0.824 (0.3), 0.768 (0.02).

Observations:

- On the gate-selected VAE (0.3) the mapper equals PCA alignment on Mantel r and is below it on kNN overlap. On the 0.02 VAE the mapper exceeds PCA alignment on every structure score and on FD. The VAE that fails the gate gives the better result on all measures.
- Mapper FD is below the FD of the gate's prior samples for both VAEs (522.0 against 773.2; 402.1 against 974.7); the second clause of H2 holds for both. The comparison with `no_feat` is not run.
- PCA alignment with the whitened code gives Mantel r 0.18; with the standardised code (entry of 2026-11-10) it gave 0.42. The score depends on the audio code.
- One seed; the ablations `no_local`, `no_global`, `no_feat` and `shuffled` are not run.

## 2026-11-12 — comparison of S1 and Q1 under a common audio code

Computed from the `features.npz` files of the two entries above (same 789 test tracks, same order). Pearson correlation between cosine distances of image features and cosine distances of audio codes. Post hoc analysis, not part of either protocol.

Correlation between the two audio distance matrices (mean code, whitened code): 0.272.

| Run | r against mean code | r against whitened code |
|---|---|---|
| S1 `s1` | 0.861 | 0.155 |
| S1 `s1_cycle` | 0.839 | 0.148 |
| S1 `s0` | 0.082 | 0.098 |
| S1 `shuffled` | 0.140 | 0.046 |
| S1 random M | 0.055 | 0.035 |
| Q1 `q1`, VAE 0.02 | 0.160 | 0.259 |
| Q1 `q1`, VAE 0.3 | 0.131 | 0.179 |

Observations:

- Each method preserves the distance structure of the code it was trained on and little of the other. The Mantel scores of S1 and Q1 are not comparable as reported in their own entries.
- The mean code has a participation ratio of 25.7 and the whitened code 104.7 (D1). The structure of the mean code is concentrated in fewer directions.
- S1 reaches high structure preservation with outputs far from the art distribution (FD 1173); Q1 reaches outputs closer to the art distribution (FD 402) with low structure preservation.

## 2026-11-12 — exploratory analysis of the seed-0 outputs (local)

Not part of either protocol. Run on the laptop (GTX 1650) from the fetched checkpoints and embeddings; scripts are not in the repository. One seed throughout. 789 test tracks; r is the Pearson correlation between distance matrices.

A. Structure measured with features other than DINOv2 (Euclidean distances between simple image statistics of the regenerated test images, against the audio code each mapper was trained on):

| Feature set | S1 `s1` (mean code) | Q1 `q1`, VAE 0.02 (whitened code) |
|---|---|---|
| DINOv2 CLS (training target) | 0.861 | 0.259 |
| Colour mean and standard deviation (6-d) | 0.256 | 0.102 |
| Colour histogram (64-d) | 0.228 | 0.079 |
| 8 x 8 layout (192-d) | 0.280 | 0.115 |
| Radial power spectrum (63-d) | 0.144 | 0.078 |

Spread of the radial power spectrum relative to test art: S1 0.42, Q1 0.66. Most of the structure reported for S1 is specific to the DINOv2 feature space.

B. Mean code, leading principal axes (fitted on training windows). Variance fractions of the first six axes: 0.117, 0.073, 0.056, 0.048, 0.034, 0.031.

| Axes removed from the mean code | r with S1 image distances | r with whitened-code distances |
|---|---|---|
| 0 | 0.861 | 0.272 |
| 1 | 0.369 | 0.546 |
| 2 | 0.289 | 0.580 |
| 5 | 0.112 | 0.637 |
| 10 | 0.027 | 0.670 |

Distances computed on the first 10 axes alone correlate 0.891 with the full mean-code distances and 0.844 with S1 image distances. The structure S1 preserves lies in about ten directions of the mean code. What these directions encode is not established.

C. Mapping audio codes directly to DINOv2 features (same mapper architecture, 4000 steps, batch 256, no decoder), with the output either used directly or replaced by the nearest training painting (cosine). FD against test art features; training art 113.3; participation ratio (PR) of test art 44.8.

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

Observations:

- No batch-level distribution term tested places the direct outputs on the art distribution (FD 575 to 1205 against 113). The outputs occupy fewer directions than art (PR about 25 against 44.8) or, for the whitened code, more (108). The weakness of the distribution term is not specific to the S1 latent space.
- Replacing each output by the nearest training painting gives real images (FD 287 to 369; plain retrieval baseline 229 to 251) with r 0.58 to 0.65 for the mean and standardised codes. The plain retrieval baseline gave r 0.05 to 0.12 and PCA alignment 0.18 to 0.42.
- The whitened code loses its structure when outputs are restricted to real paintings (r 0.066). Its distances are nearly uniform (coefficient of variation 0.090, D1) and its participation ratio (104.7) exceeds that of art features (44.8). D1 selects the code by same-track retrieval; this criterion does not test whether the code's geometry can be represented in the image feature space.

## 2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)

Not part of either protocol; metadata are used for interpretation only and enter no training step. Source: `fma_metadata.zip` (`tracks.csv`, `features.csv`, `echonest.csv`). 7994 tracks (all splits), window-averaged codes, principal axes fitted on training tracks. Echo Nest attributes exist for 1294 of the tracks. Scripts are not in the repository.

Nearest-neighbour agreement (cosine, k = 10; genre chance 0.125, 8 balanced genres):

| Code | Same-genre rate | Same-artist rate |
|---|---|---|
| mean | 0.457 | 0.127 |
| standardised | 0.528 | 0.159 |
| whitened | 0.493 | 0.165 |

Leading axes of the mean code. eta^2: fraction of axis variance explained by genre. R^2: 5-fold ridge regression from 100 low-level descriptors (mean and standard deviation of RMS energy, spectral centroid, bandwidth, roll-off, zero-crossing rate, spectral contrast, MFCC, chroma, tonnetz). rho: Spearman correlation.

| Axis | Variance fraction | Genre eta^2 | R^2 from low-level descriptors | Strongest Echo Nest attributes (rho) |
|---|---|---|---|---|
| 1 | 0.130 | 0.31 | 0.82 | acousticness +0.68, energy -0.51, danceability -0.47 |
| 2 | 0.078 | 0.19 | 0.62 | danceability +0.53, speechiness +0.31, instrumentalness -0.28 |
| 3 | 0.060 | 0.09 | 0.62 | speechiness +0.14 |
| 4 | 0.050 | 0.09 | 0.65 | energy -0.32, instrumentalness +0.22 |
| 5 | 0.036 | 0.14 | 0.61 | instrumentalness -0.18 |
| 6 | 0.032 | 0.09 | 0.45 | speechiness +0.27 |

Genre means on axis 1 (axis standard deviation 0.52): Folk +0.49, Instrumental +0.31, International +0.09, Experimental +0.05, Pop -0.02, Rock -0.11, Hip-Hop -0.39, Electronic -0.40. The strongest single low-level correlates of axis 1 are spectral-contrast bands (rho up to +0.63); RMS energy is not among the three strongest for axes 1 to 4.

The standardised code has the same leading contrast (axis 1: acousticness -0.65, danceability +0.55, energy +0.50; genre eta^2 0.33; R^2 0.77).

Observations:

- The first axis of the mean code separates acoustic from electronic, high-energy material and is largely predictable from spectral descriptors. It is a timbre and production contrast, not loudness.
- The axes that S1 preserves (exploratory entry above, part B) are therefore musically interpretable, and coarse: they relate to genre-level differences.
- The standardised code has the highest same-genre neighbour rate of the three.

## 2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/code_swap.yaml` |
| Kernel | `arunavganguly/abstraction-code-swap`, version 1 |
| Commit | `76b055e` (`q1-structure`); S1 from a second clone at `7a17f65` (`s1-structure`) |
| Hardware | 2 x Tesla T4, one method per GPU |
| Result | completed; duration 9407 s |

Seed 0, 20000 steps, 789 test tracks. Q1 on the VAE with `kl_weight` 0.02 (gate overridden). Rows marked "earlier" are from the entries of 2026-11-12 above.

| Method | Audio code | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Participation ratio |
|---|---|---|---|---|---|---|---|
| S1 `s1` (earlier) | mean | 0.861 | 0.853 | 0.268 | 0.932 | 1172.8 | 14.7 |
| S1 `s1` | whitened | 0.272 | 0.226 | 0.178 | 0.791 | 1172.9 | 29.7 |
| Q1 `q1` (earlier) | whitened | 0.259 | 0.210 | 0.152 | 0.774 | 402.1 | 28.9 |
| Q1 `q1` | mean | 0.825 | 0.840 | 0.282 | 0.915 | 425.4 | 24.3 |

Observations:

- The audio code determines the structure score: both methods reach 0.83 to 0.86 on the mean code and 0.26 to 0.27 on the whitened code.
- The method determines the distance to the art distribution: Q1 gives FD 402 to 425 under either code, S1 gives 1173 under either code.
- Q1 on the mean code has the highest structure score among runs with FD below 500.

Structure of the Q1 outputs under pixel-based feature sets (local, exploratory; test images regenerated from the checkpoints; Pearson r against the code each mapper was trained on):

| Feature set | Q1, mean code | Q1, whitened code |
|---|---|---|
| DINOv2 CLS (training target) | 0.825 | 0.259 |
| Colour mean and standard deviation | 0.233 | 0.102 |
| Colour histogram | 0.023 | 0.079 |
| 8 x 8 layout | 0.307 | 0.115 |
| Radial power spectrum | 0.097 | 0.078 |

As for S1, most of the structure measured in DINOv2 features is not present in these feature sets.

## 2026-11-13 — F1 diagnosis (local)

| Field | Value |
|---|---|
| Script | `methods/f1/diagnose.py --seed 0` |
| Commit | `aa6783c` (`f1-feature`); run before the commit, working tree not clean |
| Hardware | GTX 1650, 4 GB |
| Result | completed |

One mapping per audio code (variant `f1`, 4000 steps, sliced Wasserstein term, `lambda_struct` 1). r: Pearson correlation between cosine distances of audio codes and of image features, 789 tracks. Stages are defined in `methods/f1/README.md`. Gallery: 4666 training paintings. PR: participation ratio of the code; CV: coefficient of variation of its pairwise distances.

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

FD of the direct outputs: 854 to 1197 for all codes.

Observations:

- The mapping generalises: r on test tracks is within 0.02 of r on training tracks for every code. The overfitting seen in S1 (validation L_struct rising after step 5500) does not appear when the mapping is trained without a decoder and feature extractor in the loop.
- Structure is lost at the restriction to real paintings: r falls from 0.95 to 0.99 (direct) to 0.52 to 0.64 (painting), and kNN overlap from 0.70 to 0.89 to 0.12 to 0.15.
- The loss depends weakly on the number of paintings: 500 paintings give r within 0.05 to 0.07 of 4666 paintings.
- The loss is not a generalisation effect: a mapping trained on the test tracks themselves gives the same r after retrieval (oracle column).
- The direct outputs are far from the art distribution (FD 854 or more); the retrieval step moves each output to the art collection and this displacement is where the distance structure is distorted.
- Single layers differ: layers 3 and 9 retain more structure after retrieval than layers 0 and 12; no single layer exceeds the mean code.

Direct assignment (exploratory, script not in the repository): a soft assignment of the 789 test tracks to the 4666 paintings, optimised by gradient descent on 1 - r with temperature annealing and then hardened, gave r 0.545 (mean code), 0.547 (standardised) and 0.230 (whitened). This optimiser did not exceed map-then-retrieve for the mean and standardised codes; it is not a proof of an upper bound.
