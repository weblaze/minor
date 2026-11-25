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

## 2026-11-13 — F1 variants and controls, seed 0 (local)

| Field | Value |
|---|---|
| Scripts | `methods/f1/train_mapper.py`, `methods/f1/evaluate.py` |
| Commit | `aa6783c` (`f1-feature`) |
| Hardware | GTX 1650, 4 GB |
| Result | completed |

Standardised code, 4000 steps, 789 test tracks, 9999 permutations, k = 10 (chance overlap 0.0127). Mantel p = 0.0001 for every row.

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

`f1`, painting output, r under pixel-based feature sets of the retrieved paintings: colour mean and standard deviation 0.169, colour histogram 0.193, 8 x 8 layout 0.153, power spectrum 0.183 (`f0`: 0.100, 0.109, 0.092, 0.082).

Observations:

- An untrained mapping already gives r 0.734 on its direct outputs: a smooth network preserves distances of its input. The direct output is therefore not evidence of learning; the painting output is.
- On the painting output, `f1` (0.578) exceeds `f0` (0.303), `shuffled` (0.279) and random M (0.066). The structure term contributes 0.28 over the distribution term alone.
- `f0` and `shuffled` retain r about 0.3 after retrieval: a mapping that only matches the art distribution, being continuous, sends similar tracks to similar paintings.

## 2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)

| Field | Value |
|---|---|
| Script | `methods/f1/realism.py --seed 0 --code <mean, standardised>` |
| Commit | grid as committed in `f1-feature` after the run; run made on a working tree with `realism.py` uncommitted |
| Hardware | GTX 1650, 4 GB |
| Result | completed; 27 configurations per code |

One mapping per configuration, 4000 steps, 789 test tracks. Configurations are defined in `GRID` of `realism.py`; the default is head `free`, sliced Wasserstein term, global structure term, batch 256. Columns: r and FD of the direct output; r, kNN overlap and FD of the nearest training painting; displacement (mean cosine distance between output and its nearest painting); number of distinct paintings; mean r of the painting output against five single-layer codes; mean r under four pixel-based feature sets of the paintings. `n/a`: all tracks received the same painting.

Mean code:

| Configuration | r direct | FD direct | r painting | kNN painting | FD painting | Displacement | Distinct | r views | r pixel (mean of 4) |
|---|---|---|---|---|---|---|---|---|---|
| `swd` | 0.945 | 854 | 0.645 | 0.131 | 384 | 0.360 | 340 | 0.445 | 0.123 |
| `struct_only` | 0.952 | 2435 | 0.139 | 0.056 | 1630 | 0.883 | 104 | 0.126 | 0.086 |
| `swd_batch1024` | 0.943 | 779 | 0.636 | 0.135 | 355 | 0.357 | 358 | 0.440 | 0.116 |
| `critic_0.3` | 0.886 | 1590 | 0.574 | 0.094 | 405 | 0.423 | 341 | 0.224 | 0.093 |
| `critic_1` | 0.824 | 1451 | 0.583 | 0.101 | 426 | 0.387 | 295 | 0.251 | 0.112 |
| `critic_3` | 0.778 | 1288 | 0.415 | 0.114 | 432 | 0.364 | 262 | 0.325 | 0.087 |
| `chamfer_1` | 0.516 | 2070 | 0.523 | 0.024 | 2051 | 0.038 | 62 | 0.056 | 0.489 |
| `chamfer_3` | 0.504 | 1479 | 0.480 | 0.030 | 1538 | 0.055 | 62 | 0.071 | 0.451 |
| `chamfer_10` | 0.207 | 1519 | 0.181 | 0.041 | 1585 | 0.053 | 57 | 0.074 | 0.123 |
| `critic_chamfer` | 0.595 | 1454 | 0.586 | 0.024 | 1353 | 0.020 | 63 | 0.055 | 0.517 |
| `pca32_swd` | 0.938 | 665 | 0.618 | 0.109 | 289 | 0.283 | 481 | 0.372 | 0.087 |
| `pca64_swd` | 0.943 | 507 | 0.605 | 0.107 | 275 | 0.310 | 488 | 0.365 | 0.095 |
| `soft_0.05_swd` | 0.827 | 571 | 0.618 | 0.155 | 462 | 0.387 | 269 | 0.451 | 0.050 |
| `soft_0.02_swd` | 0.832 | 443 | 0.707 | 0.134 | 341 | 0.473 | 375 | 0.392 | 0.067 |
| `soft_0.02` | 0.935 | 1106 | 0.354 | 0.081 | 805 | 0.834 | 157 | 0.187 | 0.046 |
| `soft_0.01` | 0.912 | 1597 | 0.279 | 0.057 | 2211 | 0.777 | 25 | 0.225 | 0.175 |
| `st_0.05_swd` | 0.696 | 736 | 0.696 | 0.077 | 736 | 0.354 | 116 | 0.275 | 0.285 |
| `st_0.05` | 0.648 | 1988 | 0.648 | 0.029 | 1988 | 0.371 | 14 | 0.159 | 0.425 |
| `st_0.02` | 0.046 | 2624 | n/a | 0.012 | 2624 | 0.312 | 1 | n/a | n/a |
| `st_0.1` | 0.644 | 2562 | 0.644 | 0.026 | 2562 | 0.287 | 27 | 0.122 | 0.531 |
| `swd_local` | 0.946 | 853 | 0.639 | 0.119 | 396 | 0.361 | 326 | 0.441 | 0.130 |
| `st_0.05_local` | 0.606 | 2181 | 0.606 | 0.024 | 2181 | 0.322 | 11 | 0.079 | 0.554 |
| `st_0.05_local_only` | 0.519 | 3065 | 0.519 | 0.030 | 3065 | 0.430 | 6 | 0.233 | 0.476 |
| `views_swd` | 0.854 | 862 | 0.589 | 0.130 | 354 | 0.362 | 381 | 0.503 | 0.109 |
| `st_0.05_views` | 0.566 | 2377 | 0.566 | 0.024 | 2377 | 0.385 | 9 | 0.246 | 0.388 |
| `noise16_swd` | 0.944 | 863 | 0.645 | 0.117 | 336 | 0.374 | 380 | 0.405 | 0.104 |
| `noise16_critic` | 0.849 | 1453 | 0.496 | 0.100 | 446 | 0.347 | 264 | 0.257 | 0.129 |

Standardised code:

| Configuration | r direct | FD direct | r painting | kNN painting | FD painting | Displacement | Distinct | r views | r pixel (mean of 4) |
|---|---|---|---|---|---|---|---|---|---|
| `swd` | 0.988 | 897 | 0.573 | 0.136 | 344 | 0.375 | 386 | 0.503 | 0.172 |
| `struct_only` | 0.996 | 1670 | 0.160 | 0.064 | 397 | 0.844 | 539 | 0.138 | 0.026 |
| `swd_batch1024` | 0.989 | 844 | 0.579 | 0.132 | 335 | 0.374 | 396 | 0.507 | 0.166 |
| `critic_0.3` | 0.939 | 961 | 0.386 | 0.129 | 309 | 0.337 | 421 | 0.338 | 0.087 |
| `critic_1` | 0.849 | 1545 | 0.364 | 0.134 | 374 | 0.437 | 345 | 0.309 | 0.153 |
| `critic_3` | 0.755 | 1678 | 0.411 | 0.114 | 568 | 0.387 | 211 | 0.365 | 0.119 |
| `chamfer_1` | 0.429 | 1323 | 0.402 | 0.042 | 1323 | 0.032 | 53 | 0.353 | 0.336 |
| `chamfer_3` | 0.410 | 1334 | 0.389 | 0.035 | 1370 | 0.032 | 47 | 0.341 | 0.328 |
| `chamfer_10` | 0.394 | 1335 | 0.375 | 0.026 | 1373 | 0.031 | 44 | 0.332 | 0.314 |
| `critic_chamfer` | 0.065 | 1501 | 0.046 | 0.049 | 1433 | 0.017 | 55 | 0.037 | 0.052 |
| `pca32_swd` | 0.969 | 703 | 0.552 | 0.132 | 269 | 0.310 | 509 | 0.477 | 0.155 |
| `pca64_swd` | 0.986 | 518 | 0.524 | 0.120 | 254 | 0.331 | 546 | 0.455 | 0.109 |
| `soft_0.05_swd` | 0.765 | 591 | 0.627 | 0.176 | 627 | 0.474 | 226 | 0.548 | 0.163 |
| `soft_0.02_swd` | 0.785 | 442 | 0.612 | 0.171 | 427 | 0.625 | 328 | 0.535 | 0.139 |
| `soft_0.02` | 0.973 | 1335 | 0.063 | 0.024 | 2969 | 0.954 | 18 | 0.053 | 0.050 |
| `soft_0.01` | 0.964 | 1307 | 0.065 | 0.028 | 2481 | 0.935 | 40 | 0.050 | 0.046 |
| `st_0.05_swd` | 0.628 | 454 | 0.628 | 0.151 | 454 | 0.516 | 283 | 0.557 | 0.241 |
| `st_0.05` | 0.416 | 2497 | 0.416 | 0.036 | 2497 | 0.404 | 5 | 0.369 | 0.364 |
| `st_0.02` | 0.295 | 3079 | 0.295 | 0.022 | 3079 | 0.632 | 2 | 0.254 | 0.295 |
| `st_0.1` | 0.643 | 831 | 0.643 | 0.151 | 831 | 0.423 | 223 | 0.569 | 0.169 |
| `swd_local` | 0.988 | 874 | 0.570 | 0.133 | 342 | 0.383 | 377 | 0.500 | 0.173 |
| `st_0.05_local` | 0.551 | 1707 | 0.551 | 0.092 | 1707 | 0.438 | 59 | 0.489 | 0.249 |
| `st_0.05_local_only` | 0.344 | 2677 | 0.344 | 0.032 | 2677 | 0.446 | 4 | 0.305 | 0.296 |
| `views_swd` | 0.982 | 883 | 0.577 | 0.139 | 354 | 0.372 | 373 | 0.518 | 0.178 |
| `st_0.05_views` | 0.352 | 3423 | 0.352 | 0.020 | 3423 | 0.374 | 2 | 0.317 | 0.352 |
| `noise16_swd` | 0.988 | 907 | 0.564 | 0.141 | 352 | 0.378 | 382 | 0.496 | 0.153 |
| `noise16_critic` | 0.869 | 1242 | 0.378 | 0.140 | 408 | 0.330 | 342 | 0.327 | 0.113 |

Observations:

- Without a distribution term (`struct_only`, `soft_0.02`, `soft_0.01`) the painting output loses the structure (r 0.06 to 0.35). A distribution term is necessary.
- No configuration raises r of the painting output by more than 0.07 over the baseline while keeping the number of distinct paintings: best `soft_0.02_swd` on the mean code (0.707 against 0.645, 375 paintings, FD 341) and `st_0.1` / `st_0.05_swd` / `soft_0.05_swd` on the standardised code (0.63 to 0.64 against 0.573).
- The critic lowers r and does not lower the FD of the direct output in 4000 steps. The local term, the noise input, the larger batch and the multi-layer term change r by less than 0.01 to 0.06; the multi-layer term raises the agreement with single-layer codes (mean code: 0.445 to 0.503) at a cost in r.
- The PCA heads give the lowest FD of the painting output (254 to 289) and the most distinct paintings (481 to 546) with r 0.03 to 0.05 below the baseline.
- Chamfer terms and straight-through retrieval without a distribution term collapse to 2 to 63 paintings. In these collapsed solutions r under pixel-based feature sets is 0.3 to 0.55, against 0.06 to 0.19 for the configurations with several hundred paintings. `st_0.1` on the mean code reaches the DINOv2 r of the baseline (0.644) with 27 paintings.
- Displacement of the baseline is 0.36 to 0.38; the mean cosine distance between two paintings of the test set is 0.672. The nearest painting is more than half a typical inter-painting distance away from the output.

## 2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)

Scripts are not in the repository. Configuration `swd` of `realism.py` unless stated. FMA genre labels and WikiArt style names (from the painting file names) are used for interpretation only.

A. Structure of the painting output measured in the leading principal axes of the training-art features (Euclidean distances on the projected, centred features; r against audio distances):

| Code | Configuration | Full features (cosine) | 2 axes | 4 axes | 8 axes | 16 axes | 64 axes | Residual beyond 16 axes |
|---|---|---|---|---|---|---|---|---|
| mean | `swd` | 0.645 | 0.428 | 0.536 | 0.583 | 0.593 | 0.617 | 0.383 |
| mean | `soft_0.02_swd` | 0.707 | 0.353 | 0.517 | 0.577 | 0.599 | 0.654 | 0.466 |
| standardised | `swd` | 0.573 | 0.505 | 0.558 | 0.588 | 0.583 | 0.562 | 0.120 |
| standardised | `soft_0.02_swd` | 0.612 | 0.429 | 0.575 | 0.637 | 0.634 | 0.615 | 0.129 |

Cumulative variance of art features: 0.21 at 4 axes, 0.32 at 8, 0.45 at 16, 0.74 at 64. Most of the structure of the painting output is present in the first 4 to 8 axes. Restricting the measurement to leading axes does not raise r above about 0.65; the hypothesis that uncontrolled variation in the remaining axes sets the ceiling is not supported.

B. Genre of the test track against the WikiArt style of its painting (seed 0). Row fractions; share of each style in the gallery: Abstract Expressionism 0.47, Colour Field 0.27, Minimalism 0.23, Action painting 0.02.

| Genre | Abstract Expressionism (mean / standardised code) | Colour Field | Minimalism |
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

C. Five mapper seeds (0 to 4), share of Colour Field and Minimalism paintings per genre:

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

Observations:

- r of the painting output is stable across seeds: 0.624 to 0.659 (mean code), 0.541 to 0.573 (standardised).
- In all five seeds the genre ordering along the acoustic-to-electronic contrast of the audio code coincides with the ordering of the paintings along the contrast between Colour Field and Minimalism on one side and Abstract Expressionism on the other. The leading contrast of the audio code is mapped to the leading contrast of the art features.
- The orientation is not fixed: seed 4 gives the mirror image (Folk to Abstract Expressionism, Electronic to Colour Field and Minimalism) under both codes. The objective depends on distances only and is invariant to this reflection; the mapping is identified up to symmetries of this kind. Seeds share the initialisation across codes, so the two codes are not independent trials of the orientation.
- The assignment of individual paintings is not reproducible between seeds; the coarse correspondence is, up to orientation.

## 2026-11-13 — S1 and Q1 on the standardised audio code

| Field | Value |
|---|---|
| Job | `scripts/kaggle/jobs/std_code.yaml` |
| Kernel | `arunavganguly/abstraction-std-code`, version 1 |
| Commit | `6bb84a7` (`q1-structure`); S1 from a second clone at `d3b327b` (`s1-structure`) |
| Hardware | 2 x Tesla T4, one method per GPU |
| Result | completed; duration 11089 s |

Seed 0, 20000 steps, 789 test tracks. Q1 on the VAE with `kl_weight` 0.02 (gate overridden). Rows for the mean and whitened codes are from the entries of 2026-11-12.

| Method | Audio code | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Participation ratio |
|---|---|---|---|---|---|---|---|
| S1 `s1` | mean | 0.861 | 0.853 | 0.268 | 0.932 | 1172.8 | 14.7 |
| S1 `s1` | standardised | 0.847 | 0.835 | 0.410 | 0.969 | 1364.0 | 16.1 |
| S1 `s1` | whitened | 0.272 | 0.226 | 0.178 | 0.791 | 1172.9 | 29.7 |
| Q1 `q1` | mean | 0.825 | 0.840 | 0.282 | 0.915 | 425.4 | 24.3 |
| Q1 `q1` | standardised | 0.797 | 0.786 | 0.351 | 0.955 | 422.1 | 20.9 |
| Q1 `q1` | whitened | 0.259 | 0.210 | 0.152 | 0.774 | 402.1 | 28.9 |

Observations:

- The standardised code gives Mantel r within 0.03 of the mean code for both methods and the highest kNN overlap and trustworthiness of the three codes (S1 0.410 against 0.268; Q1 0.351 against 0.282).
- The distance to the art distribution is again set by the method: Q1 422, S1 1364.
- Among the generated-image runs, Q1 on the mean or standardised code combines Mantel r of about 0.8 with FD of about 420. These are the runs with the best combination of structure and distance to the art distribution so far.

## 2026-11-13 — F1 orientation across ten seeds (local)

| Field | Value |
|---|---|
| Script | `methods/f1/orientation.py --seeds 10` |
| Commit | `f1-feature`, script committed after the run |
| Hardware | GTX 1650, 4 GB |
| Result | completed; 50 mappings |

For each seed: correlation between the test tracks' scores on principal axis k of the audio code and the scores of their paintings on principal axis k of the art features (k = 1, 2, 3); r of the painting output; distribution terms on validation tracks against validation art. Signs of principal axes are arbitrary, so only agreement between seeds is meaningful.

Skewness of the leading three axes: mean code 0.24, -0.22, 0.57; standardised code -0.27, -0.58, 0.45; art features -0.06, -0.03, 0.16.

| Code | Configuration | Abs. correlation, axis 1 (range) | Seeds with the minority sign, axis 1 | Sign split, axis 2 | r painting (range) | Painting similarity between seeds | Same orientation only | Shuffled tracks |
|---|---|---|---|---|---|---|---|---|
| mean | `swd` | 0.87 to 0.92 | 1 of 10 (seed 4) | 5 / 4, one near zero | 0.613 to 0.676 | 0.558 | 0.591 | 0.490 |
| mean | `swd`, weight 5 | 0.85 to 0.91 | 2 of 10 (seeds 4, 8) | 5 / 5 | 0.608 to 0.689 | 0.508 | 0.563 | 0.465 |
| mean | `swd`, weight 20 | 0.80 to 0.90 | 2 of 10 (seeds 4, 8) | 5 / 5 | 0.617 to 0.674 | 0.500 | 0.550 | 0.456 |
| mean | `soft_0.02_swd` | 0.25 to 0.85 | 1 of 10 (seed 4) | 5 / 5 | 0.676 to 0.714 | 0.378 | 0.387 | 0.344 |
| standardised | `swd` | 0.88 to 0.90 | 1 of 10 (seed 4) | 5 / 5 | 0.541 to 0.573 | 0.527 | 0.560 | 0.474 |

Validation distribution terms of the minority-sign seeds lie inside the range of the other seeds (mean code, `swd`: sliced Wasserstein 0.0206 for seed 4 against 0.0175 to 0.0212; energy distance 0.248 against 0.202 to 0.259).

Observations:

- The first principal axis of the audio code is mapped to the first principal axis of the art features in every seed, with absolute correlation about 0.9 for the free head.
- The orientation of the first axis has the same sign in 8 to 9 of 10 seeds. Seed 4 is in the minority under every configuration and both codes; the initial weights, which are shared across configurations, decide it. A larger weight on the distribution term does not remove the minority outcomes.
- The second and third axes show no preferred orientation (about 5 of 10 each) and correlations between -0.85 and +0.80.
- The validation distribution terms do not separate the two orientations. Selecting the seed with the lowest loss does not select an orientation.
- The art features are close to symmetric along their leading axes (skewness -0.06 and -0.03). A symmetric target distribution gives the distribution term nothing to prefer one orientation over its mirror image.
- Two seeds assign paintings to the same track with cosine similarity 0.56 to 0.59 against 0.49 for shuffled tracks. Beyond the first axis the mapping is not reproducible between seeds.
- The soft retrieval head has the highest r of the painting output and the lowest agreement between seeds.

## 2026-11-14 — F1: what limits the painting output (local, exploratory)

Script not in the repository. Configuration `swd`, seed 0, 789 test tracks. r, kNN overlap and FD as in the entries above.

A. Interpolation between the direct output (t = 0) and its nearest training painting (t = 1):

| t | Mean code: r | kNN | FD | Standardised code: r | kNN | FD |
|---|---|---|---|---|---|---|
| 0 | 0.945 | 0.696 | 854 | 0.988 | 0.817 | 897 |
| 0.25 | 0.929 | 0.621 | 651 | 0.949 | 0.703 | 663 |
| 0.5 | 0.868 | 0.369 | 484 | 0.837 | 0.424 | 471 |
| 0.75 | 0.765 | 0.221 | 396 | 0.696 | 0.223 | 367 |
| 1 | 0.645 | 0.132 | 384 | 0.573 | 0.136 | 344 |

B. Gallery size, with galleries sampled from a Gaussian fitted to the training-art features (FD of 4666 such samples to test art: 123; training art: 113):

| Gallery | Mean code: displacement | r | FD | Standardised code: displacement | r | FD |
|---|---|---|---|---|---|---|
| 4666 real paintings | 0.360 | 0.645 | 384 | 0.375 | 0.573 | 344 |
| 4666 Gaussian samples | 0.383 | 0.643 | 390 | 0.395 | 0.564 | 358 |
| 46660 Gaussian samples | 0.358 | 0.697 | 330 | 0.369 | 0.623 | 310 |
| 466600 Gaussian samples | 0.338 | 0.724 | 333 | 0.349 | 0.665 | 315 |

C. Mean of the k nearest paintings (not a real painting for k > 1): mean code r 0.686 (k = 2), 0.700 (4), 0.706 (8), 0.704 (16) with FD 427 to 663; standardised code 0.643, 0.676, 0.689, 0.693 with FD 402 to 642.

D. Displacement. Length of the displacement relative to the length of the output: 0.84 (mean code), 0.86 (standardised). Cosine between displacement and output: -0.44 for both. Replacing the displacement by a random vector of the same length drawn with the art covariance gives r 0.707 (mean code) and 0.555 (standardised).

E. Density of the collection (cosine distance to the nearest other painting): test painting to nearest training painting 0.250 (quartiles 0.195, 0.251, 0.313); training painting to nearest other training painting 0.255; mean distance between two training paintings 0.687. The outputs of the mapping are at 0.36 to 0.38 from their nearest painting.

Observations:

- Structure and distance to the art distribution trade off along a smooth curve between the direct output and the nearest painting. At FD about 425 the curve gives r about 0.80 (mean code); Q1 on the mean code reached r 0.825 at FD 425 with a trained generator. The generated-image result and the retrieval result lie on one trade-off curve.
- A Gaussian model of the art features reproduces the retrieval result at the real gallery size (r 0.643 against 0.645). In this model ten times more paintings add 0.05 in r and a hundred times more add 0.08. The limit of the painting output is the sparsity of a finite collection in a feature space with about 45 effective dimensions; it falls slowly with collection size.
- The collection is sparse for real paintings as well: a held-out painting is at cosine distance 0.25 from its nearest neighbour, more than a third of the mean distance between two paintings. The outputs of the mapping are further out (0.36 to 0.38) and their displacement points partly back toward the origin.
- A displacement of this length accounts for the loss in r whether it points to the nearest painting or in a random direction: the loss comes from the size of the move, not from a systematic distortion.

## 2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat

| Field | Value |
|---|---|
| Jobs | `scripts/kaggle/jobs/q1_mean_seeds12.yaml`, `scripts/kaggle/jobs/q1_mean_controls.yaml` |
| Kernels | `arunavganguly/abstraction-q1-mean-seeds12`, `arunavganguly/abstraction-q1-mean-controls`, version 1 each |
| Commit | `648394a` (`q1-structure`) |
| Hardware | 2 x Tesla T4 per job, one run per GPU |
| Result | completed; durations 10395 s and 10338 s |

Mean code, 20000 steps, VAE with `kl_weight` 0.02 (gate overridden), 789 test tracks. Seed 0 of `q1` is from the code-swap entry of 2026-11-12.

| Variant | Seed | Mantel r | Mantel rho | kNN overlap | Trustworthiness | FD | Participation ratio | Mean pairwise distance |
|---|---|---|---|---|---|---|---|---|
| `q1` | 0 | 0.825 | 0.840 | 0.282 | 0.915 | 425.4 | 24.3 | 0.601 |
| `q1` | 1 | 0.815 | 0.834 | 0.285 | 0.920 | 409.0 | 24.9 | 0.600 |
| `q1` | 2 | 0.818 | 0.836 | 0.290 | 0.922 | 409.7 | 23.7 | 0.598 |
| `shuffled` | 0 | 0.209 | 0.208 | 0.163 | 0.823 | 405.5 | 30.1 | 0.657 |
| `no_feat` | 0 | 0.869 | 0.873 | 0.392 | 0.965 | 675.2 | 21.6 | 0.305 |

Agreement between the three `q1` seeds (local analysis of the `features.npz` files): Pearson correlation between the image distance matrices 0.847 to 0.854; cosine similarity between the image features of the same track 0.415 to 0.528 against 0.400 to 0.410 for shuffled tracks. Correlation between the test tracks' scores on principal axis 1 of the audio code and of their images on principal axis 1 of the art features: -0.43, -0.85, -0.91 (same sign in the three seeds); `shuffled` -0.04; `no_feat` 0.00.

Observations:

- `q1` on the mean code is stable over three seeds: Mantel r 0.815 to 0.825, FD 409 to 425.
- `shuffled` falls to r 0.209: the structure of `q1` depends on the true audio distances.
- H2, first clause: FD of `q1` (409 to 425) is below FD of `no_feat` (675). The feature-space distribution term costs 0.05 in Mantel r and 0.11 in kNN overlap.
- The three seeds agree as geometries (0.85) and in the orientation of the first axis, and assign different images to the same track.

## 2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)

| Field | Value |
|---|---|
| Script | `methods/f1/phase1_batch.py` |
| Commit | `4b57977` (`f1-feature`) |
| Hardware | GTX 1650, 4 GB |
| Result | completed; 34 units, no failure, 7 h 2 min |

A. Orientation over random initialisations (`orientation.py`, configuration `swd` unless stated). Sign of the correlation between audio and painting scores on principal axis k; counts of the negative sign. Principal axes are the same within a code, so signs are comparable across network sizes.

| Code | Mapping network | Seeds | Axis 1 negative | Axis 2 negative | Axis 3 negative | Mean abs. correlation, axes 1, 2, 3 | r painting, mean (sd) |
|---|---|---|---|---|---|---|---|
| mean | 1024 x 3 | 50 | 40 | 26 | 7 | 0.89, 0.52, 0.38 | 0.646 (0.015) |
| mean | 256 x 3 | 20 | 11 | 10 | 2 | 0.84, 0.32, 0.34 | 0.639 (0.015) |
| mean | 1024 x 1 | 20 | 14 | 11 | 0 | 0.85, 0.42, 0.32 | 0.636 (0.015) |
| mean | 2048 x 3 | 20 | 13 | 12 | 2 | 0.88, 0.53, 0.40 | 0.642 (0.017) |
| mean, `soft_0.02_swd` | 1024 x 3 | 20 | 14 | 11 | 0 | 0.65, 0.43, 0.35 | 0.703 (0.014) |
| standardised | 1024 x 3 | 50 | 16 | 26 | 24 | 0.89, 0.52, 0.35 | 0.556 (0.010) |

Pooled over the five mean-code settings: axis 1 negative in 92 of 130 runs (71 %), axis 2 in 70 of 130 (54 %), axis 3 positive in 119 of 130 (92 %). For the standardised code the majority sign of axis 1 holds in 34 of 50 runs (68 %). The same seed gives the majority or the minority sign under both codes in 44 of 50 cases; between network sizes the same seed agrees in 9 to 12 of 20 cases.

Majority against minority orientation of axis 1 (mean code, 1024 x 3, 40 and 10 seeds):

| Quantity | Majority | Minority |
|---|---|---|
| r painting | 0.647 | 0.643 |
| Validation sliced Wasserstein | 0.0190 | 0.0188 |
| Validation energy distance | 0.2265 | 0.2249 |
| Validation structure loss | 0.0523 | 0.0525 |
| Batch distance correlation at step 100 / 300 / 1000 / 3900 | 0.935 / 0.958 / 0.979 / 0.987 | 0.934 / 0.955 / 0.979 / 0.988 |
| Batch sliced Wasserstein at step 1000 / 3900 | 0.0374 / 0.0270 | 0.0364 / 0.0268 |

The same comparison for the standardised code and for the other network sizes shows differences of the same size or smaller.

Observations:

- D-vote: one orientation of axis 1 is reached more often than the other in every setting (55 % to 80 %; pooled 71 % for the mean code, 68 % for the standardised code). The preference is present and is not decisive. It weakens for the smaller and the larger network.
- D-ease and D-loss: the two orientations are learned at the same speed and reach the same losses. Neither test separates them.
- Axis 3 of the mean code has a stable orientation (92 %). Its audio scores are the most skewed of the three axes (0.57) and the art axis is the most skewed of the three (0.16). Axis 2, where the art axis has skewness -0.03, shows no preference. For the standardised code axis 3 shows no preference (24 of 50).
- The orientation is set by the initial weights: the same seed lands on the same side for both codes (44 of 50) and not across network sizes.
- r of the painting output has a standard deviation of 0.010 to 0.017 over seeds.

B. Combination sweep (`realism.py --grid sweep`, seed 0): 100 combinations per code of output head (free, pca64, soft05, soft02, st05), distribution term (swd1, swd5, energy1, swd1_energy1, swd1_chamfer01) and structure term (global, global_local, global_views, global3). Full rows are in the run directories `f1_realism_sweep_*`.

Means over the 20 or 25 combinations sharing a component (r, kNN overlap and FD of the painting output; distinct paintings; mean r under pixel-based features):

| Component | Mean code: r | kNN | FD | Distinct | Pixel r | Standardised: r | kNN | FD | Distinct | Pixel r |
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

Collapsed combinations (fewer than 100 distinct paintings): 7 (mean code), 5 (standardised). Combinations meeting the selection rule (at least 300 distinct paintings, FD at most 450): 58 and 54.

C. Reruns of the selected combinations, seeds 0 to 4 (mean and standard deviation over five seeds). Selection rule as fixed in `phase1_batch.py`; the baseline is `free__swd1__global`.

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

Other extremes of the sweep at seed 0: lowest FD of the painting output 234 with 555 distinct paintings (`pca64__swd1_energy1__global`, mean code, r 0.611); highest kNN overlap 0.191 (`soft05__swd1_chamfer01__global3`, standardised code, r 0.596, FD 323).

Observations:

- Combinations raise r of the painting output by about 0.10 over the baseline for the mean code (0.641 to 0.737 to 0.747) and by 0.06 to 0.085 for the standardised code (0.556 to 0.614 to 0.641). The gains are several standard deviations over seeds.
- The gains come from retrieval inside training (soft or straight-through head) together with a distribution term; a small chamfer weight added to sliced Wasserstein gives the highest r, the energy distance gives the lowest FD at a similar r.
- `soft02__swd1_chamfer01__global3` improves every reported quantity over the baseline for both codes (mean code: r +0.096, kNN +0.023, FD about equal; standardised: r +0.058, kNN +0.044, FD -64, 61 more distinct paintings).
- The straight-through head with chamfer has the highest r and the lowest kNN overlap for the mean code (0.079): global structure improves while neighbourhoods degrade.
- The multi-layer structure term lowers r for the mean code and leaves it unchanged for the standardised code. The local term and the larger structure weight change r by 0.01 or less on average.
- No combination exceeds r 0.76 with several hundred distinct paintings; the limit set by the sparsity of the collection (entry of 2026-11-14 above) remains.

## 2026-11-14 — corrections and reference values

Earlier entries are left as written. This entry resolves inconsistencies between them.

- S1 `s0`, Mantel r: 0.083 in "S1 seed 0, all variants, test split" and 0.082 in "comparison of S1 and Q1 under a common audio code". Both are roundings of the same value, 0.0825. Reference: 0.083 (the value written by `evaluate.py`).
- Commit of the F1 variant runs: "F1 variants and controls, seed 0 (local)" gives `aa6783c` for all runs. The `f1` and `f0` runs were made before that commit, on a working tree with the F1 files uncommitted; their `meta.json` records `4234cad`, which `docs/results.md` reports. The `shuffled` and random M runs were made at `aa6783c`. The code of the four runs is identical.
- F1 baseline (`free` head, sliced Wasserstein, global structure term), r of the painting output, seed 0. Mean code: 0.644 (`diagnose.py`), 0.645 (`realism.py`, `orientation.py`), 0.637 (exploratory precursor with a different batch composition). Standardised code: 0.572 (`diagnose.py`), 0.573 (`realism.py`), 0.578 (`train_mapper.py` and `evaluate.py`, which draw batches in a different order). Reference values: the five-seed means of the batch of 2026-11-14, 0.641 (0.013) for the mean code and 0.556 (0.011) for the standardised code; single-seed reference 0.645 and 0.573 (`realism.py`).
- Variance fraction of the first principal axis of the mean code: 0.117 in "exploratory analysis of the seed-0 outputs", where the axes are fitted on individual training windows, and 0.130 in "audio-code axes against FMA metadata", where they are fitted on window-averaged training tracks. Both are correct for their definition. The orientation analyses use window-averaged training tracks.
- Selection of combinations in "F1 batch at the end of phase 1": the selection rule used scores on the test tracks. The sweep is repeated with selection on the validation tracks (`methods/f1/phase1_validation.py`); its result is logged separately.
- The scripts of the exploratory entries are now in `analysis/exploratory/`, with a table that maps each script to its entry.

## 2026-11-15 — F1 combination sweep with selection on the validation tracks (local)

- Code: `f1-feature` 219bbd2, `methods/f1/phase1_validation.py`; aggregation `analysis/exploratory/valsel.py`.
- Runs: `f1_realism_valsweep_{mean,standardised}_{0..3}_seed0_*` (100 combinations per code, seed 0, validation and test scores recorded), `f1_realism_valrerun_{mean,standardised}_seed{1..4}_*`. State: `runs/f1_validation_phase1/`. Wall time 4.6 h on the GTX 1650.
- Selection rule, validation tracks only: at least 300 distinct paintings, Frechet distance of the painting output to validation art at most 450, ranked by r of the painting output; five best per code plus the baseline, rerun on seeds 1 to 4.
- Eligible combinations: 59 of 100 (mean code), 57 of 100 (standardised code).
- Agreement of validation and test r of the painting output over the 100 combinations (seed 0): Pearson 0.975 (mean code), 0.988 (standardised code).

Five seeds, mean (standard deviation). Test columns are free of selection on test data.

Mean code:

| Combination | rank val / test | test r | test kNN | test FD | test distinct | val r | val FD |
|---|---|---|---|---|---|---|---|
| `free__swd1__global` (baseline) | 59 / 42 | 0.641 (0.014) | 0.122 (0.007) | 346 (22) | 371 (24) | 0.603 (0.018) | 381 (19) |
| `st05__swd1_chamfer01__global_local` | 1 / 2 | 0.743 (0.026) | 0.090 (0.005) | 364 (49) | 338 (39) | 0.731 (0.024) | 347 (40) |
| `st05__swd1_chamfer01__global` | 2 / 1 | 0.747 (0.008) | 0.079 (0.003) | 375 (31) | 330 (19) | 0.731 (0.013) | 361 (29) |
| `soft02__swd1_chamfer01__global3` | 3 / 3 | 0.737 (0.014) | 0.145 (0.005) | 340 (21) | 359 (16) | 0.716 (0.014) | 359 (8) |
| `st05__swd1_energy1__global_local` | 4 / 7 | 0.712 (0.013) | 0.126 (0.005) | 353 (18) | 346 (18) | 0.701 (0.017) | 335 (13) |
| `st05__energy1__global_local` | 5 / 4 | 0.704 (0.020) | 0.128 (0.005) | 371 (12) | 323 (13) | 0.692 (0.019) | 354 (21) |

Standardised code:

| Combination | rank val / test | test r | test kNN | test FD | test distinct | val r | val FD |
|---|---|---|---|---|---|---|---|
| `free__swd1__global` (baseline) | 62 / 67 | 0.556 (0.012) | 0.137 (0.007) | 338 (13) | 394 (20) | 0.559 (0.015) | 367 (15) |
| `st05__swd1_chamfer01__global3` | 1 / 1 | 0.641 (0.007) | 0.132 (0.007) | 393 (9) | 389 (15) | 0.643 (0.004) | 423 (15) |
| `st05__energy1__global_views` | 5 / 10 | 0.615 (0.016) | 0.162 (0.004) | 298 (7) | 385 (5) | 0.623 (0.016) | 304 (9) |
| `soft02__swd1__global_views` | 9 / 22 | 0.602 (0.012) | 0.167 (0.008) | 446 (30) | 307 (10) | 0.613 (0.017) | 439 (30) |
| `st05__swd5__global3` | 10 / 5 | 0.624 (0.005) | 0.152 (0.003) | 423 (14) | 302 (7) | 0.627 (0.003) | 432 (12) |
| `st05__swd5__global_views` | 12 / 27 | 0.610 (0.005) | 0.159 (0.006) | 345 (7) | 348 (12) | 0.615 (0.008) | 353 (10) |

Ranks are by r of the painting output at seed 0 over all 100 combinations, without the eligibility filter.

Result:

- Selection on validation tracks returns the same leading combinations as the earlier selection on test tracks: `st05__swd1_chamfer01__global` and `soft02__swd1_chamfer01__global3` for the mean code, `st05__swd1_chamfer01__global3` for the standardised code. Their five-seed test values equal those of the batch of 2026-11-14 (0.747, 0.737, 0.641) to the third decimal or within 0.001.
- The gain over the baseline is 0.10 (mean code) and 0.085 (standardised code) in test r; it is not an effect of selecting on test data.
- The baseline reproduces: 0.641 (0.014) and 0.556 (0.012).
- The straight-through head with the chamfer term gives the highest r and the lowest kNN overlap (0.08 to 0.09, mean code); `soft02__swd1_chamfer01__global3` keeps kNN overlap at 0.145 with r 0.737.
- The selection note in the corrections entry of 2026-11-14 is resolved by this entry.

## 2026-11-15 — structure under two other image encoders (local, evaluation only)

- Script: `analysis/exploratory/second_encoder.py` (stages `gen_s1q1`, `gen_f1`, `eval`). Run by a subagent; numbers checked against its `results.json`.
- Question: is the audio-to-image structure of the S1, Q1 and F1 outputs specific to DINOv2, the encoder used in the training losses?
- Encoders: DINOv2 ViT-S/14 (reference, CLS), MAE ViT-B/16 (`facebook/vit-mae-base`, mask ratio 0, mean of patch tokens), DINO v1 ViT-B/16 (`facebook/dino-vitb16`, loaded from the safetensors conversion at `refs/pr/1`; not compared with the original weights). Neither MAE nor DINO v1 is used in any training loss.
- Measure: Mantel r and kNN overlap (k = 10) between audio-code distances and image-feature distances, 789 test tracks. F1 mappings retrained locally with `realism.fit`, mean code, seed 0, 4000 steps. Frechet distance not computed.
- Check: the DINOv2 column reproduces the logged values for every set (largest difference 0.005, for the retrained F1 combination).

| Image set | DINOv2 r | DINOv2 kNN | DINO v1 r | DINO v1 kNN | MAE r | MAE kNN |
|---|---|---|---|---|---|---|
| S1 `s1` | 0.861 | 0.268 | 0.508 | 0.220 | 0.232 | 0.096 |
| S1 `s0` (control) | 0.082 | 0.085 | 0.251 | 0.111 | 0.249 | 0.077 |
| S1 `shuffled` (control) | 0.140 | 0.049 | 0.175 | 0.107 | 0.149 | 0.074 |
| Q1 mean code, seed 0 | 0.825 | 0.282 | 0.453 | 0.244 | 0.233 | 0.139 |
| Q1 mean code, seed 1 | 0.815 | 0.284 | 0.479 | 0.255 | 0.219 | 0.125 |
| Q1 mean code, seed 2 | 0.818 | 0.290 | 0.448 | 0.257 | 0.172 | 0.133 |
| Q1 `shuffled` (control) | 0.209 | 0.163 | 0.146 | 0.170 | 0.102 | 0.108 |
| Q1 `no_feat` | 0.869 | 0.392 | 0.621 | 0.307 | 0.408 | 0.130 |
| Q1 standardised code | 0.797 | 0.351 | 0.625 | 0.316 | 0.313 | 0.163 |
| F1 baseline, nearest painting | 0.645 | 0.132 | 0.404 | 0.108 | 0.084 | 0.097 |
| F1 `soft02__swd1_chamfer01__global3` | 0.732 | 0.142 | 0.479 | 0.116 | 0.065 | 0.101 |
| F1 untrained mapper, standardised code | 0.066 | 0.038 | 0.067 | 0.037 | 0.016 | 0.036 |

Participation ratio of the features of the 595 test paintings: DINOv2 44.8, DINO v1 68.6, MAE 1.6.

Result:

- DINO v1: the structure is present in every trained set, at r 0.40 to 0.63 against 0.07 to 0.25 for the controls. It is lower than under DINOv2 (0.65 to 0.87). kNN overlap is close to the DINOv2 values for S1 and Q1 (0.22 to 0.32 against 0.27 to 0.39).
- The structure is therefore not only a property of the DINOv2 feature space; a second self-supervised encoder that was not in the loss sees about 55 to 80 % of the r value.
- Under DINO v1 the order of the sets differs: Q1 standardised code (0.625) and Q1 `no_feat` (0.621) are above S1 `s1` (0.508) and Q1 mean code (0.45 to 0.48).
- S1 `s0` reads 0.251 under DINO v1 and 0.249 under MAE, against 0.082 under DINOv2. The margin of S1 `s1` over its control under DINO v1 is 0.26.
- MAE: the mean-patch features are close to one-dimensional on this collection (participation ratio 1.6). Controls score as high as trained sets. The MAE column does not discriminate and gives no evidence in either direction.
- F1 untrained mapper on the mean code gave 0.036 in this rerun; no logged value exists for that setting (the logged 0.066 is for the standardised code), and the other encoders were not run on it.

Limitations: one seed for S1 and F1; MAE feature pooling was not varied (CLS token or a normalised layer could behave differently).

## 2026-11-15 — G-GAN, first run (Kaggle): failed, discriminator did not train

- Code: `g-gan` e6b3d5a, `methods/gan/train_gan.py`; job `abstraction-gan-art`; runs `gan_ch64_seed0_20261114T112652Z`, `gan_ch96_seed0_20261114T112652Z`. Two T4 GPUs, 9 h each.
- Setting: plain convolutional generator and discriminator (no residual connections), spectral normalisation on every discriminator layer, hinge loss, differentiable augmentation, batch 64, learning rate 2e-4.
- Reference Frechet distances (DINOv2 features, 595 test paintings): training art 113.3, Q1 VAE reconstructions 208.5, Q1 VAE prior samples 974.7.

| Width | Steps reached | Step time | FD by evaluation step | Last FD | Last kernel distance |
|---|---|---|---|---|---|
| 64 | 30575 | 1.06 s | 1468, 1334, 1226, 1123, 1049, 979 (steps 5000 to 30000) | 1094 | 43.3 |
| 96 | 13128 | 2.47 s | 1338, 973 (steps 5000, 10000) | 1387 | 51.3 |

- Discriminator loss stayed at 2.00 for the whole run at width 64 and until step 11000 at width 96; mean discriminator output on real and on generated images was equal and between 0.003 and 0.05. Under the hinge loss this means the discriminator did not separate the two sets on average, at an output scale far below the margin of 1.
- Samples: few distinct patterns (dark blobs, grid textures), near-greyscale; mode collapse.
- The last evaluation of each run is worse than the one before it (979 to 1094; 973 to 1387).
- Step time was set by image decoding in the data loader (one process, no workers), not by the GPU.

Conclusion: the run measures the first settings, not the adversarial generator as a method.

## 2026-11-15 — G-GAN, second version: local check, second Kaggle run launched

- Code: `g-gan` 0c558e9. Changes: residual blocks in generator and discriminator (sum pooling in the discriminator); spectral normalisation or an R1 penalty (every 16 steps) as alternatives; training images held on the device, flip on the device.
- Local check (GTX 1650, width 32, batch 32, 3000 steps, 2000 samples; runs `gan_localA_sn_*`, `gan_localB_r1_*`; step time 0.5 s with a second process on the machine):

| Regulariser | FD at 1500 | FD at 3000 | Kernel distance at 3000 |
|---|---|---|---|
| spectral normalisation | 1972 | 1506 | 77.8 |
| R1 penalty 10, no spectral normalisation | 1863 | 1583 | 96.1 |

- Samples of both at step 3000 are varied in colour and layout, with checkerboard artefacts; no collapse to a few patterns. FD falls by about 300 to 450 per 1500 steps at this stage. The first version needed 5000 steps at width 64 to reach 1468.
- With spectral normalisation the mean discriminator outputs on real and generated images are again equal (loss 2.00) while FD falls; equal means are therefore not by themselves a sign of failure. In the first version the output scale was below 0.05; here it is 0.07 to 0.28.
- Launched: `abstraction-gan-art` version 2, width 64, batch 64, GPU 0 spectral normalisation (`--tag sn`), GPU 1 R1 penalty 10 (`--tag r1`), 150000 steps or 8 h. Results to follow.

## 2026-11-16 — G-GAN, second version (Kaggle): R1 penalty trains, spectral normalisation does not

- Code: `g-gan` 0c558e9, `methods/gan/train_gan.py`; job `abstraction-gan-art` version 2; runs `gan_sn_seed0_20261115T061651Z` (GPU 0, spectral normalisation), `gan_r1_seed0_20261115T061651Z` (GPU 1, R1 penalty 10 every 16 steps, no spectral normalisation). Width 64, batch 64, seed 0, limit 150000 steps or 8 h. Output: `runs/kaggle/abstraction-gan-art/runs/`.
- Both runs stopped at the time limit. Step time 2.11 s on a T4 for both; 9 % of the configured steps were reached. The cause of the step time is not established (the local check at width 32, batch 32 ran at 0.5 s per step on a GTX 1650).
- Reference Frechet distances (DINOv2 features, 595 test paintings): training art 113.3, Q1 VAE reconstructions 208.5, Q1 VAE prior samples 974.7 (lowest prior value of the KL sweep 685.2), first G-GAN run 1094 and 1387.

| Regulariser | Steps reached | FD at 5000 | FD at 10000 | Last FD (step) | Kernel distance at 5000 / 10000 / last |
|---|---|---|---|---|---|
| R1 penalty 10 | 13651 | 940.7 | 787.0 | 682.0 (13651) | 28.5 / 21.5 / 16.3 |
| spectral normalisation | 13714 | 1527.5 | 1575.2 | 1472.3 (13714) | 112.6 / 121.2 / 96.8 |

- R1: FD falls at every evaluation and had not levelled off when the run stopped (about 150 per 5000 steps at the end). Discriminator outputs separate real from generated images from step 5000 (`d_real` - `d_fake` 0.07 at step 5000, 0.53 at step 10000, 0.21 at step 13600). Samples at step 13651: dark shapes on light ground, scribble textures, vertical stripes, yellow and orange fields; a share of near-uniform light tiles; residual grid artefacts; muted colour range.
- Spectral normalisation: discriminator loss 2.00 and equal mean outputs on real and generated images (0.01 to 0.06) for the whole run; FD stays between 1472 and 1575. Samples: blurred colour patches with checkerboard blocks. The fall of FD seen in the local check at width 32 (1972 to 1506 in 3000 steps) did not continue at width 64.
- The R1 generator at step 13651 is below the prior samples of every Q1 VAE setting (682.0 against 685.2 to 1261.4) and above the Q1 VAE reconstructions (208.5) and the Q1 mapper outputs (402 to 425).

Conclusion: the R1 variant is the working G-GAN configuration. It is under-trained at 13651 steps; the step time has to be reduced or the run continued before the generator is used in a bridge. One seed.

## 2026-11-16 — G-GAN R1 job launched; CAV-MAE checkpoint on Kaggle

- Launched: `abstraction-gan-r1` version 1, code `pipeline` d9801ba, job file `scripts/kaggle/jobs/gan_r1.yaml`. Width 64, batch 64, seed 0, R1 penalty 10, no spectral normalisation, 150000 steps or 6 h per GPU. GPU 0: mixed precision (`--amp --tag r1amp`). GPU 1: float32 (`--tag r1`). Both write `checkpoint.pt` at every evaluation for `--resume`. The job reports the return code of GPU 1 only; the mixed-precision run exits at the first non-finite loss.
- Mixed precision was not validated before the launch: on the GTX 1650 it gives non-finite losses (half precision on that card), so this job is its first test. The run of 2026-11-15 cannot be resumed (only the averaged generator was saved); both runs start from step 0.
- Private Kaggle dataset `arunavganguly/abstraction-cav-mae` created for S4: `audio_model.21.pth` (CAV-MAE-Scale++, unmodified, from the checkpoint links of github.com/YuanGongND/cav-mae, BSD 2-Clause), 764386034 bytes, sha256 `15c8a9de088536341156d5f2a3d8f78cd6c82269fb03951d100963fa12bd89a7`. Local copy: `models/cav_mae/audio_model.21.pth` (not in git).
- Results of the job to follow.

## 2026-11-16 — S4: probe code and Kaggle job launched

- Code: `pipeline` 1c56ab2. `abstraction/encoders/cav_mae.py` (encoder of CAV-MAE-Scale++, parameter names of the published checkpoint; all 164.5 M encoder parameters load from `audio_model.21.pth`, decoder entries ignored), `methods/s4/extract.py`, `methods/s4/probe.py`, job `scripts/kaggle/jobs/s4_probe.yaml`.
- Inputs to the model: audio as in the MERT extraction (three 10 s windows from the centre of the track), 16 kHz, 128-bin filterbank, 1024 frames, normalised with -5.081 and 4.4849; paintings with the shorter side scaled to 224 and a centre crop. Embedding: token mean after the joint layers; a track is the mean of its windows.
- The probe trains nothing. It reports: agreement of CAV-MAE audio embeddings with MERT codes and of CAV-MAE image embeddings with DINOv2 features (test split); statistics of the track-painting cosine similarity (variance shares of track, painting and pair; agreement of two windows of one track); S4 as a retrieval bridge (most similar training painting per test track, scored like the F1 painting block, raw and with per-painting standardisation); and, for painting assignments of other methods given with `--assignments`, the mean similarity of the assigned pairs against random re-assignments (D-S4).
- Checked locally: smoke pipeline (`scripts/smoke.py --methods s4`) passes; the real checkpoint loads and embeds paintings on the CPU. Not checked locally: the filterbank path (`torchaudio` is not installed on the laptop).
- Launched: `abstraction-s4-probe` version 1 (extraction of all tracks and paintings, then the probe without `--assignments`). D-S4 on the F1 assignments is run locally from the fetched embeddings.
- Results to follow.

## 2026-11-18 — G-GAN R1 with mixed precision (Kaggle): three times faster, FD 492 at 29264 steps

- Code: `pipeline` d9801ba; job `abstraction-gan-r1` version 1 (`scripts/kaggle/jobs/gan_r1.yaml`); runs `gan_r1amp_seed0_20261115T203214Z` (GPU 0, `--amp`), `gan_r1_seed0_20261115T203214Z` (GPU 1, float32). R1 penalty 10, no spectral normalisation, width 64, batch 64, seed 0, 6 h limit. Output: `runs/kaggle/abstraction-gan-r1/runs/`. Both runs stopped at the time limit; return code 0.
- Mixed precision: no non-finite loss in 29264 steps; 0.74 s per step against 2.19 s in float32 (factor 2.96). FD at equal steps is within the spread of the float32 runs (one seed each).

| Run | Steps | FD at 5000 | FD at 10000 | FD at 15000 | FD at 20000 | FD at 25000 | Last FD (step) | Kernel distance, last |
|---|---|---|---|---|---|---|---|---|
| R1, mixed precision | 29264 | 860.1 | 807.5 | 727.2 | 654.9 | 515.6 | 492.3 (29264) | 10.4 |
| R1, float32 (this job) | 9862 | 911.4 | | | | | 817.9 (9862) | 21.7 |
| R1, float32 (2026-11-16 entry) | 13651 | 940.7 | 787.0 | | | | 682.0 (13651) | 16.3 |

- FD falls at every evaluation; the fall between 25000 and 29264 is smaller (515.6 to 492.3) than between 20000 and 25000 (654.9 to 515.6). Discriminator loss rises from about 2.0 to 3.4 (5.3 in the last logged step) with `d_real` above `d_fake` throughout the second half.
- Samples at step 29264: larger coloured shapes on light or orange grounds, scribble and lattice textures, stripes, framed rectangles; a share of near-uniform tiles (light grey, orange) remains; fewer grid artefacts than at 13651.
- Reference Frechet distances: training art 113.3, Q1 VAE reconstructions 208.5, Q1 mapper outputs 402 to 425, Q1 VAE prior samples 685.2 to 1261.4. The generator at 29264 is below every VAE prior and above the Q1 mapper outputs.
- `checkpoint.pt` of both runs holds generator, discriminator, averaged generator, optimisers, scalers and step (29264 for the mixed-precision run), so either run can be continued with `--resume`.

Conclusion: mixed precision is usable for G-GAN and is the setting for further runs. The R1 generator is still improving at 29264 steps. One seed.

## 2026-11-18 — S4 probe (Kaggle): CAV-MAE embeddings of all tracks and paintings

- Code: `pipeline` 1c56ab2; job `abstraction-s4-probe` version 1; runs `s4_extract_seed42_20261115T204546Z` (15 min on a T4), `s4_eval_probe_seed42_20261115T210034Z` (15 min). Embeddings: `runs/kaggle/abstraction-s4-probe/cav_mae_embeddings/` (7994 tracks, 3 windows each; 5832 paintings; 768 dimensions). Six tracks skipped (three too short, three not decodable), the same count as the MERT extraction.
- Agreement with the encoders of phase 1 (test split; Mantel 9999 permutations, all p = 0.0001):

| CAV-MAE against | n | Mantel r (Pearson) | Mantel rho | kNN overlap (k 10; chance) | Trustworthiness |
|---|---|---|---|---|---|
| MERT, mean code | 789 tracks | 0.324 | 0.328 | 0.154 (0.013) | 0.818 |
| MERT, standardised code | 789 tracks | 0.364 | 0.343 | 0.206 (0.013) | 0.871 |
| DINOv2 | 595 paintings | 0.562 | 0.556 | 0.228 (0.017) | 0.871 |

- Track-painting cosine similarity S (all 7994 x 5832 pairs): mean 0.187, standard deviation 0.088, range -0.291 to 0.628. Mean similarity within tracks 0.418, within paintings 0.520; cosine of the two centroids 0.400. Variance shares of S: track 0.473, painting 0.244, track-painting interaction 0.282. Interaction rows of S computed from window 1 and window 2 of the same test track correlate at 0.809; of different tracks at 0.005.
- S4 as a retrieval bridge without training (each test track receives the training painting with the highest S; scored with the standardised MERT code against DINOv2 features of the assigned paintings):

| Assignment | Mantel r | kNN overlap | Trustworthiness | FD | Distinct paintings | Largest share |
|---|---|---|---|---|---|---|
| raw S | 0.080 | 0.066 | 0.677 | 633.0 | 222 | 0.075 |
| S standardised per painting over training tracks | 0.060 | 0.047 | 0.619 | 402.9 | 444 | 0.032 |

- For comparison, on the same measure (standardised code, seed 0, 2026-11-13 entry): F1 `f1` painting output 0.578, `f0` 0.303, `shuffled` 0.279, untrained random mapping 0.066.

Conclusion: CAV-MAE orders tracks partly as MERT does (r 0.32 to 0.36) and paintings substantially as DINOv2 does (0.56). Its track-painting similarity has a pair-specific part (28 % of the variance) that is a stable property of the track (window agreement 0.81 against 0.005). Used directly as a bridge, that similarity carries little of the MERT distance structure into the paintings (r 0.06 to 0.08). Whether it prefers one orientation of the F1 mappings (D-S4) is tested in a separate entry.
## 2026-11-18 — MERT model revision of the feature extraction

- The extraction job `abstraction-features` loaded `m-a-p/MERT-v1-95M` at revision `main` (`mert.revision: null` in `configs/base.yaml`) on 2026-11-09. The commit history of the model repository on huggingface.co (checked 2026-11-18) has no commit after 2025-05-25; `main` was therefore commit `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` ("Update configuration_MERT.py", 2025-05-25) at the time of the extraction. Hidden states were read through forward hooks (`hidden-state route: hooks` in `logs/07.log` of the job).
- Resolves the item "MERT model revision (commit hash) used for extraction" in `docs/paper1/missing.md`, section 3.

## 2026-11-19 — D-S4: CAV-MAE scores the orientations of 100 F1 mappings (local)

- Code: `pipeline` 57ecf3e (`methods/f1/orientation.py --save-assignments`, `methods/s4/d_s4.py`); the orientation runs started with the same code uncommitted (`git_dirty` true in their meta.json), committed unchanged as 57ecf3e. GTX 1650. Runs `f1_orientation_dS4mean_seed0_20261118T170617Z`, `f1_orientation_dS4std_seed0_20261118T180444Z` (configuration `swd`, seeds 0 to 49, 4000 steps; the same configuration as the D-vote runs of 2026-11-14), `s4_d_s4_seed42_20261118T191133Z`, `s4_eval_probe_seed42_20261118T191146Z`. CAV-MAE embeddings from `abstraction-s4-probe`.
- Score per seed: mean CAV-MAE similarity of the 789 assigned test track-painting pairs, as z against 999 random re-assignments of the same paintings to the same tracks (track and painting main effects cancel). Seeds are split by the sign of the axis correlation (orientation); p from 9999 permutations of the seed labels.

| Code | Axis | Seeds positive / negative | Mean z, positive | Mean z, negative | p | Spearman (corr, z) |
|---|---|---|---|---|---|---|
| mean | 1 | 10 / 40 | -1.40 | 1.32 | 0.001 | -0.45 |
| mean | 2 | 24 / 26 | 0.28 | 1.23 | 0.16 | -0.34 |
| mean | 3 | 43 / 7 | 0.66 | 1.50 | 0.39 | -0.07 |
| standardised | 1 | 34 / 16 | 1.34 | -1.26 | 0.002 | 0.45 |
| standardised | 2 | 24 / 26 | 0.04 | 0.93 | 0.27 | -0.22 |
| standardised | 3 | 26 / 24 | 0.09 | 0.95 | 0.28 | -0.11 |

- Seeds with p < 0.05 against re-assignment: 21 of 50 (mean code), 16 of 50 (standardised). Mean z 0.77 (sd 2.32) and 0.50 (sd 2.79).
- z does not follow mapping quality: Spearman with painting r -0.04 (mean) and 0.27 (standardised), with the number of distinct paintings -0.14 and -0.02; painting r of the two orientation groups 0.643 / 0.647 (mean) and 0.558 / 0.550 (standardised). After regressing z on painting r and distinct paintings, the axis-1 difference remains (-2.78, p 0.0004; 1.87, p 0.020).
- Single F1 evaluations of 2026-11-12 (standardised code, seed 0), same score (raw S; S standardised per painting): `f1` z 4.66 (p 0.001); 5.55; `f0` 1.74 (0.043); 2.42; `shuffled` 1.13 (0.128); 1.51; untrained random mapping 1.38 (0.082); 1.07.

Conclusion: on both codes CAV-MAE scores the axis-1 orientation chosen by the majority of seeds (D-vote: 40/50 negative on the mean code, 34/50 positive on the standardised code) above the minority orientation, and the difference is not explained by mapping quality. D-vote and D-S4 agree on axis 1 for both codes; neither test separates the orientations of axes 2 and 3. The trained F1 assignment scores above its controls. The axis signs are defined per code (principal axes of each code), so the two codes are not compared by sign.

## 2026-11-19 — S2: blind matching by Gromov-Wasserstein (local)

- Code: `pipeline` 21f22ac (`methods/s2/match.py`; entropic GW in PyTorch, square loss, cosine distances divided by their mean, regularisation annealed from 0.05 to 0.0005 by a factor 0.95 per step, 300 steps, 200 Sinkhorn iterations; 10 restarts from random couplings, the lowest objective selected). Orientations recomputed with the axis convention of `methods/f1/orientation.py` by `analysis/exploratory/s2_orientation.py` (5a1d829; the run itself used an SVD convention, fixed for later runs in a677747). GTX 1650. Run `s2_eval_match_seed42_20261118T172954Z`, 50 min per code. D-S4 on the restarts: `s4_d_s4_seed42_20261118T191603Z`.
- Problem: the 789 test tracks (audio code, mean over windows) against the 4666 training paintings (DINOv2), no pairs, nothing trained. Solved on the test tracks themselves: the assignment uses the distances among the test tracks, unlike F1, whose mapping is trained on the training tracks.
- Settings chosen before the run on the mean code with single solves: without annealing at 0.01 the coupling was nearly uniform (mean share of a track's mass on its largest painting 0.003); at 0.0005 with annealing it is 0.156, close to the limit set by the uniform marginals, with the lowest objective (0.0570 against 0.0742).

| Code | GW (selected) | Gaussian reference, mean (sd) | Same-space reference | Mantel r | rho | kNN | Trustworthiness | FD | Kernel distance | Distinct paintings |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | 0.0570 | 0.0692 (0.0013) | 0.0269 | 0.809 | 0.832 | 0.158 | 0.874 | 166.4 | -0.02 | 789 |
| standardised | 0.0329 | 0.0253 (0.0003) | 0.0153 | 0.532 | 0.519 | 0.148 | 0.876 | 163.6 | -0.06 | 789 |

- References (5 draws, best of 2 restarts each, against the best of the first 2 restarts of the real match): `gaussian`, as many samples of a Gaussian with the mean and covariance of the painting features as there are paintings; same-space, as many random training tracks. The mean code fits the art collection better than the Gaussian (z -9.2); the standardised code fits the Gaussian better than the art (z +28.4).
- Restarts: all 10 objectives equal to four decimals; 91 % (mean) and 99 % (standardised) of tracks receive the same painting in two restarts; mean cosine of the two paintings 0.956 and 0.996 (random pairs 0.31). Coupling concentration 0.156 and 0.145.
- Orientation (F1 axis convention), 10 restarts: mean code axis 1 negative in 10 (mean corr -0.885), axis 2 positive in 10 (0.774), axis 3 positive in 10 (0.395); standardised code axes 1, 2 and 3 positive in 10 (0.94, 0.65, 0.63). F1 D-vote for comparison (50 seeds): mean code 10/24/43 positive on axes 1/2/3; standardised 34/24/26.
- D-S4 on the S2 assignments: mean z 0.00 (sd 0.25) on the mean code and 0.18 (sd 0.06) on the standardised code; no restart with p < 0.05.
- For comparison (2026-11-13 and 2026-11-14 entries; F1 is trained on training tracks and evaluated on test tracks): F1 `f1` painting output, standardised code, seed 0: r 0.578, FD 342.8, 385 paintings; best mean-code recipes 0.737 to 0.747 with fewer than 400 paintings.

Conclusion: matching shapes alone, without training, assigns each test track a different painting, keeps the audio distances at r 0.81 (mean code) and 0.53 (standardised code), and stays close to the art distribution (FD 164 to 166; training art 113). The solution does not depend on the initial coupling. Its axis-1 orientation is the one chosen by the majority of F1 seeds and preferred by CAV-MAE, on both codes; on the mean code its axis 3 also matches the F1 majority. CAV-MAE does not score the S2 pairs above random re-assignment, unlike the trained F1 pairs. Whether the mean-code result holds out of sample (matching training tracks, then placing test tracks) is not tested. One run per code.

## 2026-11-19 — G-GAN R1 continued to 57568 steps, and a second seed (Kaggle): FD 388

- Code: `pipeline` c5a7670; job `abstraction-gan-r1-resume` version 1 (`scripts/kaggle/jobs/gan_r1_resume.yaml`); runs `gan_r1amp_seed0_20261118T171246Z` (GPU 0, continued with `--resume` from `gan_r1amp_seed0_20261115T203214Z/checkpoint.pt` at step 29264) and `gan_r1amp_seed1_20261118T171246Z` (GPU 1, seed 1, from scratch). Both mixed precision, R1 penalty 10, width 64, batch 64, 5.5 h limit; both stopped at the limit. `git_dirty` true in meta.json (the Kaggle copy of the repository). Output: `runs/kaggle/abstraction-gan-r1-resume/runs/`.
- Speed with both GPUs busy: 0.70 s per step (seed 0, 28304 steps) and 0.75 s per step (seed 1, 26364 steps).

| Run | FD by step | Last FD (step) | Kernel distance, last |
|---|---|---|---|
| seed 0, continued | 522.9 (30000), 424.3 (35000), 489.9 (40000), 527.8 (45000), 406.8 (50000), 406.2 (55000) | 388.2 (57568) | 7.17 |
| seed 1, new | 958.2 (5000), 740.9 (10000), 620.3 (15000), 566.9 (20000), 515.8 (25000) | 546.2 (26364) | 12.75 |

- Seed 0 no longer falls at every evaluation: FD rises from 424 to 528 between 35000 and 45000, then falls to 388. The first evaluation after resuming (522.9 at 30000) is above the last one before it (492.3 at 29264).
- Seed 1 follows seed 0 of the first job closely (860, 808, 727, 655, 516 at 5000 to 25000).
- Samples of seed 0 at 57568: colour fields, blobs on light or orange grounds, scribble and crumpled textures, framed rectangles; about one tile in eight is still near uniform (light grey or off-white).
- Reference Frechet distances: training art 113.3, Q1 VAE reconstructions 208.5, Q1 mapper outputs 402 to 425, Q1 VAE prior samples 685.2 to 1261.4. Seed 0 at 57568 is below the Q1 mapper outputs for the first time.
- `checkpoint.pt` of both runs can be continued with `--resume`.

Conclusion: the R1 generator keeps improving with more steps but no longer at every evaluation; FD 388 at 57568 steps (seed 0). A second seed reproduces the first 25000 steps of seed 0. `generator.pt` of seed 0 at 57568 is the current candidate for the GAN bridge.

## 2026-11-19 — S2 out of sample: match training tracks, place test tracks (local)

- Code: `pipeline` 30ec60c (`methods/s2/oos.py`, section `oos` of `methods/s2/config.yaml`); the run started with the same code uncommitted. GTX 1650. Run `s2_eval_oos_seed42_20261119T102209Z`, 17 to 19 min per code.
- Method: entropic GW (settings of `match.py`) between 3000 random training tracks (audio code, mean over windows) and the 4666 training paintings; 3 restarts, the lowest objective selected. Each matched track gets a barycentric target (coupling-weighted mean of painting features). A test track is placed from its audio code alone, by `knn` (mean target of its 10 nearest matched tracks, cosine) or `linear` (ridge from audio code to target, strength 1.0 relative to the mean variance), and receives the nearest training painting (cosine). The test tracks take no part in the match. Control `shuffled`: the same placements from a random one-to-one pairing of matched tracks and paintings.

| Code | Placement | Mantel r | rho | kNN | FD | Distinct paintings | Largest share |
|---|---|---|---|---|---|---|---|
| mean | gw, knn | 0.611 | 0.611 | 0.118 | 489.0 | 247 | 0.049 |
| mean | gw, linear | 0.718 | 0.711 | 0.131 | 447.7 | 254 | 0.047 |
| mean | shuffled, knn | -0.011 | -0.013 | 0.023 | 734.1 | 165 | 0.238 |
| mean | shuffled, linear | 0.202 | 0.213 | 0.013 | 1701.8 | 26 | 0.929 |
| standardised | gw, knn | 0.531 | 0.522 | 0.165 | 398.5 | 306 | 0.032 |
| standardised | gw, linear | 0.528 | 0.529 | 0.148 | 406.1 | 276 | 0.029 |
| standardised | shuffled, knn | 0.017 | 0.011 | 0.025 | 746.1 | 169 | 0.231 |
| standardised | shuffled, linear | 0.025 | 0.022 | 0.019 | 1544.5 | 38 | 0.835 |

- Restarts: objectives equal (0.0498 mean code, 0.0317 standardised); 99 % of test tracks receive the same painting in two restarts.
- Orientation of the knn placement (F1 axis convention), 3 restarts: mean code axes 1/2/3 negative/positive/positive in 3 of 3 (corr -0.90, 0.82, 0.64), as in the transductive S2 run and the D-vote majority on axis 1; standardised code positive/positive/negative (0.88, 0.28, -0.14): axes 1 and 2 as in the transductive run, axis 3 reversed.
- Comparison: transductive S2 (2026-11-19 entry) mean code r 0.809, FD 166.4, 789 distinct; standardised 0.532, FD 163.6, 789. F1 five-seed best mean-code recipes 0.737 to 0.747 with fewer than 400 paintings; F1 `f1` standardised seed 0: 0.578, FD 342.8, 385.

Conclusion: the S2 structure holds out of sample. On the mean code, matching training tracks by shape and placing test tracks with a linear map gives Mantel r 0.718 (transductive 0.809; shuffled control 0.20); on the standardised code 0.53, equal to the transductive value. What does not carry over is the spread over paintings: placing tracks one at a time and taking the nearest painting gives 250 to 300 distinct paintings and FD 400 to 490, against 789 and FD 164 to 166 when all test tracks are matched jointly. One run per code; subset of 3000 training tracks.

## 2026-11-19 — Human judging pilot: first 4 judges; Arunav's marks on the image atlas

- Judging page (https://claude.ai/artifact/FqBet4gUoTsJdZGyM7YK5o, `scripts/judging/`): 4 sessions, 124 answers (one session stored directly, three imported from answer codes). Median 8.9 s per answer. Systems: S1 (`s1_eval_s1_judging`), Q1 (`q1_eval_q1_judging`, mean code), F1 (`f1_eval_f1`, standardised code, painting output); 24 test tracks.
- Match trials (hear one track, pick its image from two; chance 0.5; one-sided binomial p): S1 8/20 (0.40, p 0.87), Q1 12/20 (0.60, p 0.25), F1 10/20 (0.50, p 0.59); all 30/60 (p 0.55).
- Triplet trials (pair chosen as most alike from the audio, against the pair chosen from the three images of the same tracks; chance 1/3): S1 6/16 (p 0.45), Q1 7/16 (p 0.26), F1 5/16 (p 0.66).
- With 20 match trials per system only accuracies of about 0.70 or more reach p < 0.05; about 150 to 200 answers per system are needed to detect 0.58.
- Image atlas (https://claude.ai/artifact/SatUpkFo4tkVeL5NURDZ7a): every non-smoke image grid of the runs, 38 cards; Arunav marked 31 (art / partly / blobs and greys). Art: G-GAN R1 mixed precision seed 0 (to 57568 steps) and seed 1. Partly: G-GAN R1 float32 runs (v2, v3), Q1 standardised code. Blobs and greys: every S1 grid and control, Q1 mean code seeds 0 to 2, Q1 whitened code, Q1 controls, all VAE prior and reconstruction grids, Q1 PCA-alignment baselines (his note: the retrieval view is art), G-GAN v1 and spectral-norm runs. F1 and S2 (real paintings) not marked.

Conclusion: in this pilot, judges do not match tracks to their images above chance for any system; the pilot can only exclude large effects. By eye, every image generated through the VAE is judged blobs and greys, and the R1 GAN is the only generator judged art.

## 2026-11-19 — S2 out of sample with joint assignment of the placed test tracks (local)

- Code: `pipeline` (commit after 30ec60c: `joint()` in `methods/s2/oos.py`). Run `s2_eval_oos_seed42_20261119T110951Z`, one GW restart per code (the three restarts of the previous run had equal objectives and gave 99 % the same paintings); otherwise the settings of the previous entry. The match itself is identical (GW 0.0498 and 0.0317; non-joint values reproduce the previous run within 0.005).
- `_joint`: all 789 placed test tracks are assigned to the 4666 training paintings together, each painting at most once, maximising the summed cosine similarity between placed feature and painting (Hungarian algorithm). The step uses the placed features only, not the distances among the test tracks; it does use the test tracks as one batch.

| Code | Placement | Mantel r | kNN | FD | Distinct | Orientation corr (axes 1/2/3) |
|---|---|---|---|---|---|---|
| mean | gw, linear, joint | 0.710 | 0.134 | 224.7 | 789 | -0.91 / 0.86 / 0.73 |
| mean | gw, knn, joint | 0.632 | 0.122 | 243.5 | 789 | -0.91 / 0.83 / 0.71 |
| mean | shuffled, linear, joint | 0.132 | 0.030 | 295.4 | 789 | |
| mean | shuffled, knn, joint | 0.033 | 0.021 | 260.8 | 789 | |
| standardised | gw, linear, joint | 0.548 | 0.141 | 218.2 | 789 | 0.90 / 0.28 / -0.17 |
| standardised | gw, knn, joint | 0.520 | 0.147 | 220.7 | 789 | 0.89 / 0.27 / -0.14 |
| standardised | shuffled, linear, joint | 0.078 | 0.026 | 296.6 | 789 | |
| standardised | shuffled, knn, joint | 0.013 | 0.023 | 274.3 | 789 | |

- Without the joint step (same run): mean code linear 0.717, FD 445.5, 256 distinct; standardised linear 0.527, FD 406.5, 275 distinct.
- Comparison: F1 five-seed best mean-code recipes 0.737 to 0.747 (FD about 340, fewer than 400 paintings); F1 lowest-FD recipe (pca64 + energy) r 0.61, FD 234, 555 paintings; transductive S2 0.809, FD 166.4, 789.

Conclusion: assigning the placed test tracks jointly removes the crowding without losing structure: out of sample, S2 on the mean code gives every test track a different painting with Mantel r 0.710 and FD 225, against 0.717, FD 446 and 256 paintings with nearest-painting retrieval. Shuffled controls stay at or below 0.13. This is the best combination of structure, realism and variety among the out-of-sample methods so far; it needs no pairs and no trained network beyond a ridge map. One GW restart per code, one subset of 3000 training tracks.

## 2026-11-19 — S2 out of sample against a pool of G-GAN images (local)

- Code: `pipeline` 33ba3c8 (`methods/s2/gan_pool.py`; `--gallery` in `methods/s2/oos.py`). GTX 1650.
- Pool `runs/gan_pool_r1amp_seed0_57568/`: 6000 images from `gan_r1amp_seed0_20261118T171246Z/generator.pt` (step 57568, averaged generator), latents N(0, I), seed 0. Near-blank images dropped: grey standard deviation below 0.06, or below 0.10 with mean saturation below 0.08 (thresholds set by inspecting 64 samples sorted by grey standard deviation; plain saturated colour fields are kept). Kept 5372 (89.5 %). The same rule would drop 5.5 % of the training paintings (near-empty minimal canvases). DINOv2 features of the kept images.
- Run `s2_eval_oos_pool_seed42_20261119T123740Z`: as the joint-assignment entry above (3000 training tracks, one GW restart, knn and linear placement, nearest or joint assignment), with the GAN pool as the gallery in place of the 4666 training paintings. FD against the held-out paintings. Orientation corr here uses the principal axes of the pool, so axes 2 and 3 are not comparable with the painting runs.

| Code | Placement | Mantel r | FD | Distinct images |
|---|---|---|---|---|
| mean | gw, linear, joint | 0.672 | 487.2 | 789 |
| mean | gw, knn, joint | 0.617 | 497.8 | 789 |
| mean | gw, linear (nearest) | 0.665 | 706.7 | 251 |
| mean | shuffled, linear, joint | 0.106 | 528.6 | 789 |
| mean | shuffled, knn, joint | -0.007 | 506.0 | 789 |
| standardised | gw, linear, joint | 0.583 | 479.9 | 789 |
| standardised | gw, knn, joint | 0.570 | 487.5 | 789 |
| standardised | shuffled, linear, joint | 0.089 | 528.0 | 789 |

- GW objective 0.0496 (mean code; paintings 0.0498) and 0.0312 (standardised; paintings 0.0317): the music matches the shape of the GAN pool as well as that of the paintings. Axis-1 corr -0.92 (mean code) and 0.88 (standardised), the same signs as with the paintings.
- Reference FDs: generator at step 57568, random samples 388.2 (2026-11-19 GAN entry); training art 113.3; Q1 mapper outputs 402 to 425; S2 joint with real paintings 218 to 225.

Conclusion: matched against generated images instead of real paintings, out-of-sample S2 with joint assignment gives every test track its own new image and keeps Mantel r 0.672 (mean code) and 0.583 (standardised, above the 0.548 with paintings); shuffled controls stay at or below 0.11. FD 480 to 490 is above the generator's own random samples (388), so the selected subset is less art-like than a random draw of the pool. First result in the project where generated images judged art-like by Arunav are assigned to music by a structure-preserving method. One run per code; no human evaluation yet.


## 2026-11-21 — B-cluster and B-anchor on S2 out of sample (local)

- Code: `pipeline` 5ad1a6e (`methods/s2/cluster.py`, `methods/s2/anchor.py`). GTX 1650. 3000 matched training tracks, test 789, gallery 4666 training paintings, joint assignment.
- B-cluster: k-means groups of the matched tracks and of the gallery, groups paired by GW between centroids (weighted by size, best of 50 restarts) and the Hungarian algorithm, GW inside each paired group, then placement and joint assignment. Control: random pairing of groups. Mantel r, linear placement:

| Code | K | Paired by GW | Random pairing |
|---|---|---|---|
| mean | 8 | 0.486 | 0.427 |
| mean | 16 | 0.530 | 0.352 |
| mean | 32 | 0.567 | 0.294 |
| standardised | 8 | 0.449 | 0.339 |
| standardised | 16 | 0.455 | 0.230 |
| standardised | 32 | 0.473 | 0.243 |

- Reference: flat S2 out of sample 0.710 (mean), 0.548 (standardised). Shuffled 0.16 and 0.08. FD 231 to 245.
- B-anchor: for each of the 8 sign patterns of the top-3 axes, GW started from the transport plan of the oriented linear cost. All 8 starts end at the same coupling (GW 0.0498 mean, 0.0320 standardised; same train corr, test r 0.709 to 0.711 and 0.548 to 0.551). The start at epsilon 0.05 erases the initial orientation, so there are no distinct candidates for the judge. Judge z differences between candidates are noise (train/test z corr -0.90 on mean).

Conclusion: B-cluster lowers structure at every K on both codes; group pairing by shape carries information (above random pairing, gap growing with K) but less than the flat match. B-anchor as built does not produce alternatives: a smaller starting epsilon or a fused orientation term is needed before the judge can choose.

## 2026-11-21 — B-anchor and B-cluster on F1 (local)

- Code: `pipeline` 5ad1a6e (`methods/f1/b_tasks.py`). 20 seeds of F1 config `swd` per code. Runs `f1_b_tasks_seed0_20261121T110306Z` (mean), `f1_b_tasks_seed0_20261121T113719Z` (standardised).
- B-anchor: CAV-MAE judge z of each seed on the validation tracks; anchored seed = highest validation z.

| Code | val/test judge z corr | anchored: test r, z | lowest val SWD: r, z | mean of seeds: r, z | axis-1 majority |
|---|---|---|---|---|---|
| mean | 0.88 | 0.624, 3.67 | 0.664, 1.64 | 0.646, 1.02 | 16/20 negative; anchored negative |
| standardised | 0.88 | 0.556, 4.85 | 0.573, 5.00 | 0.556, 1.36 | 14/20 positive; anchored positive |

- B-cluster: test tracks restricted to the paintings of the GW-paired group. Mean of 20 seeds, Mantel r: mean code K 8 0.515 (random groups 0.407), K 16 0.472 (0.440); standardised K 8 0.428 (0.406), K 16 0.399 (0.300). Judge z in this arm varies from -0.9 to 2.8 without pattern.

Conclusion: the machine judge's choice of seed carries over to held-out tracks (corr 0.88) and picks the axis-1 majority orientation on both codes, at about the mean structure; axes 2 and 3 stay unfixed. B-cluster lowers structure on F1 as on S2.

## 2026-11-21 — S1 with the G-GAN bridge (Kaggle)

- Job `abstraction-gan-bridge-s1`, `pipeline` 8bd0ce9. Generator `gan_r1amp_seed0_20261118T171246Z`, prior N(0, I) as latent reference. 20000 steps, seed 0, T4. Final val l_dist 0.115, l_struct 0.245.

| Variant | Mantel r | FD (held-out art) | Participation ratio |
|---|---|---|---|
| s1 | 0.755 | 1210.3 | 11.6 |
| shuffled | 0.108 | 866.6 | 21.4 |
| random M | 0.046 | 645.2 | 26.2 |

- Reference: training art FD 113.3; generator random samples 388.2; held-out art participation ratio 44.8.
- Samples: most test tracks map to one family of red-orange horizontal stripe images; a few outliers (white with black shapes, yellow bars, textured fields).

Conclusion: S1 through the GAN preserves structure strongly (r 0.755) but collapses most tracks onto one image family; FD 1210 is far above random generator samples. The latent distribution term does not keep the mapper inside the varied part of the GAN's input space. Needs a stronger latent term or a feature distribution term before it is usable.

## 2026-11-21 — Q1 with the G-GAN bridge (Kaggle)

- Job `abstraction-gan-bridge-q1`, `pipeline` 8bd0ce9. Generator `gan_r1amp_seed0_20261118T171246Z`, L_lat against N(0, I), mean code, seed 0, 20000 steps, T4.

| Variant | Mantel r | kNN overlap (chance 0.013) | FD (held-out art) |
|---|---|---|---|
| q1 | 0.537 | 0.108 | 458.2 |
| shuffled | 0.040 | 0.057 | 427.5 |
| random M | 0.046 | 0.025 | 645.2 |

- Reference: generator random samples 388.2; Q1 through the VAE 402 to 425; S1 through the GAN 0.755 / 1210.
- Samples: varied (stripes, colour fields, black shapes on white, textures, red and teal compositions); a red-teal family recurs but there is no collapse.

Conclusion: Q1 through the GAN keeps structure (r 0.54, shuffled 0.04) with varied, art-like images at FD 458, near the generator's own samples. With the feature-space realism term the GAN bridge does not collapse, unlike S1 (which has only the latent term). Best generated-image bridge so far by eye; one seed.

## 2026-11-21 — S2 B-anchor, fixed (local)

- Code: `pipeline` 1030cc4 (`methods/s2/anchor.py`: start epsilon 0.005 instead of 0.05; optional fused GW with the oriented linear cost, weight alpha; `methods/s2/common.py`: `feature_cost`, `alpha`). GTX 1650. 3000 matched training tracks, gallery 4666, test 789, ridge placement, joint assignment. 8 sign patterns of the top-3 axes for alpha 0 and alpha 0.1. Log `b_anchor2.log`.
- The starts now survive: alpha 0 ends in distinct couplings (axis-1 train corr +0.85 or -0.91 on the mean code); alpha 0.1 holds every pattern (|corr| 0.88 to 0.98 on all three axes).
- Judge z on the training tracks carries over to the test tracks: corr over the 8 candidates 0.94 (mean, alpha 0), 0.86 (mean, 0.1), 0.85 (standardised, both).

| Code | alpha | Anchored (highest train z) | train z | test r | test z | Lowest GW | Range of test r |
|---|---|---|---|---|---|---|---|
| mean | 0 | (-1, -1, -1) | 8.74 | 0.694 | 1.89 | same | 0.679 to 0.710 |
| mean | 0.1 | (-1, -1, +1) | 8.01 | 0.668 | 4.07 | same | 0.490 to 0.671 |
| standardised | 0 | (+1, -1, -1) | 10.27 | 0.552 | 5.42 | (+1, -1, +1) | 0.538 to 0.559 |
| standardised | 0.1 | (+1, -1, -1) | 7.19 | 0.542 | 6.47 | (+1, +1, +1) | 0.542 to 0.558 |

- Signs are those of the start; the train corr of the result has the same signs except where alpha 0 lets axis 3 drift.
- Axis 1: mean train z of the 8 candidates with axis 1 negative against the 8 with axis 1 positive: 4.46 against -2.75 (mean code), -0.96 against 5.28 (standardised). The judge's axis-1 choice is the same as the shape-only (lowest GW) choice, the F1 seed majority and D-S4 for both codes.
- Axis 2: the anchored candidate has axis 2 negative in all four settings.
- Cost of anchoring in structure against flat S2 out of sample (0.710 mean, 0.548 standardised): 0.016 at alpha 0 and 0.042 at alpha 0.1 on the mean code; none on the standardised code. With alpha 0.1, forcing axis 3 negative on the mean code lowers r to 0.49 to 0.55.

Conclusion: with a lower start epsilon the oriented starts give real alternatives, and the CAV-MAE judge chooses among them consistently: its choice on training tracks predicts the test tracks (corr 0.85 to 0.94), agrees with shape matching and the earlier machine judges on axis 1, and adds a consistent choice for axis 2 (negative on both codes). At alpha 0 the anchored S2 match keeps the structure of flat S2 within 0.02. One subset of training tracks, one run per setting.

## 2026-11-21/22 — S3 through the G-GAN (Kaggle) and second-reader check (local)

- Job `abstraction-s3-gan`, `pipeline` 1030cc4 (`methods/s3/train.py`, `identify.py`; decisions of 2026-11-21: reader trained with the mapper, no flip, light noise and blur, G-GAN first). Generator `gan_r1amp_seed0_20261118T171246Z`. Mean code, seed 0, 15000 steps per run (none hit the 2.75 h limit), T4. Augmentation: crop 85 to 100 % of the area, noise std 0.02, blur sigma 0.6 on 30 % of images. Loss InfoNCE (temperature 0.1) + 1.0 latent SWD + 1.0 DINOv2-feature SWD.
- Identification on the 789 test tracks (chance top1 0.0013, maximum 9.62 bits). "track": image from the mean code; "window": image from one random window, ranked against the track codes. clean / augmented = reader input without / with one draw of the training augmentation.

| Variant | track top1 clean | track top1 aug | top10 clean | bits clean | window top1 clean / aug | Mantel r | FD |
|---|---|---|---|---|---|---|---|
| s3 | 0.777 | 0.859 | 0.990 | 7.42 | 0.397 / 0.460 | 0.179 | 425.8 |
| s3_noaug | 0.863 | 0.649 | 0.999 | 7.76 | 0.437 / 0.321 | 0.190 | 424.1 |
| shuffled | 0.000 | 0.003 | 0.015 | 0.00 | 0.001 / 0.003 | 0.078 | 390.2 |
| random_m | 0.138 | 0.144 | 0.460 | 3.17 | 0.077 / 0.087 | — | — |

- Second reader (`methods/s3/second_reader.py`, pipeline 8847d85, GTX 1650): mapper frozen, one image per track for all splits, a fresh reader of the same architecture trained by InfoNCE on the 6408 training tracks only (3000 steps, batch 256, best validation loss), on frozen features of DINO ViT-B/16 (v1, CLS; not used in S3 training) or of DINOv2 (as in training). Test tracks:

| Run | DINO v1 top1 | top10 | bits | DINOv2 (fresh reader) top1 | top10 | bits |
|---|---|---|---|---|---|---|
| s3 | 0.525 | 0.914 | 6.10 | 0.389 | 0.850 | 5.48 |
| s3_noaug | 0.447 | 0.888 | 5.94 | 0.361 | 0.817 | 5.37 |
| random_m | 0.056 | 0.272 | 2.11 | 0.049 | 0.228 | 1.79 |

- Samples (s3): varied colour fields, black shapes on white, stripes, textures, gestural marks; a few near-blank white images.

Conclusion: the jointly trained mapper and reader invent a code in which the test track is named from its image alone in 78 % of cases (chance 0.13 %; untrained mapper 14 %), at FD 426, close to the generator's own samples (388). The code survives the training augmentation (86 %) and a different 10 s window of the track (40 to 46 %). Without augmentation the code is read better from clean images but breaks under the augmentation (86 to 65 %), so the augmentation makes it robust without removing it. A fresh reader on DINO v1, an encoder the game never saw, recovers 52 % (untrained mapper 6 %), so the code is carried by visual content that another self-supervised encoder also sees, not only by DINOv2-specific patterns; the augmented run transfers better (52 % against 45 %). Distance structure is weak (r 0.18): S3 makes tracks distinguishable, not similar tracks similar. One seed.

## 2026-11-22 — S1 through the G-GAN with a weaker structure term (Kaggle)

- Job `abstraction-gan-bridge-s1-lambda`, `pipeline` 8847d85. S1 mapper into the G-GAN (`gan_r1amp_seed0_20261118T171246Z`), mean code, seed 0, 20000 steps, T4; only `lambda_struct` changed from 1.0. Evaluation on the 789 test tracks.

| lambda_struct | Mantel r | Spearman | kNN overlap (chance 0.013) | trustworthiness | FD | participation ratio | mean pairwise dist (test art 0.672) |
|---|---|---|---|---|---|---|---|
| 1.0 (earlier run, s1gan) | 0.755 | — | — | — | 1210 | — | — |
| 0.3 | 0.662 | 0.688 | 0.088 | 0.744 | 1091 | 15.6 | 0.439 |
| 0.1 | 0.668 | 0.694 | 0.085 | 0.738 | 1061 | 13.9 | 0.485 |

Conclusion: a weaker structure term costs about 0.09 of r and barely moves FD (1210 to 1061-1091; Q1 through the same GAN reaches 458). The collapse of S1 through the GAN is therefore not caused by the weight of the structure term. One seed.

## 2026-11-22 — S4 bridge through the G-GAN (Kaggle)

- Job `abstraction-s4-bridge-gan`, `pipeline` 8847d85 (`methods/s4/bridge.py`). Generator `gan_r1amp_seed0_20261118T171246Z`. Mapper trained so that the frozen CAV-MAE pairs each generated image with its own sound among the batch (symmetric InfoNCE, temperature 0.1, centred embeddings) + 1.0 latent SWD + 1.0 DINOv2-feature SWD. Mean code, seed 0, batch 48, time limit 4.5 h (s4 stopped at step 9570, shuffled at 10953), T4.
- Identification by CAV-MAE on the 789 test tracks (image of each track ranked among the 789 CAV-MAE audio embeddings; chance top1 0.0013, chance median rank about 395), and S1 evaluation.

| Variant | top1 | top10 | median rank | mean own sim | mean other sim | Mantel r | kNN overlap | FD | participation ratio |
|---|---|---|---|---|---|---|---|---|---|
| s4 | 0.004 | 0.070 | 136 | 0.097 | 0.001 | 0.063 | 0.073 | 472.1 | 23.8 |
| shuffled | 0.005 | 0.015 | 373 | 0.004 | 0.000 | 0.147 | 0.070 | 470.9 | 21.7 |

Conclusion: CAV-MAE can be steered. Trained images sit closer to their own sound than to other sounds (median rank 136 against 373 for shuffled, top10 7 % against 1.5 %), at the same realism as the control (FD 472). The effect is weak, and the distance structure is lost (r 0.06, below the shuffled control's 0.15). S4 alone is not a candidate system; its term is tested as a weak anchor on top of Q1 (job `abstraction-q1-anchor-gan`). One seed.

## 2026-11-22 — G-GAN run 5: resume of R1 with amplitude (Kaggle)

- Job `abstraction-gan-r1-resume2`, `pipeline` 5ad1a6e. Seeds 0 and 1 resumed from the 2026-11-18 checkpoints, time limit 9 h, target 150000 steps. Seed 0 reached step 101996, seed 1 step 70338. FD of samples to real art (same evaluation as earlier GAN runs; the generator in use, seed 0 step 57568, is at 388):

| Seed | FD by step |
|---|---|
| 0 | 60k 369.8, 70k 349.8, 80k 354.5, 85k 345.8, 90k 332.9, 95k 329.8, 100k 366.4, 101996 345.7 |
| 1 | 50k 398.7, 60k 377.4, 70k 386.5, 70338 411.0 |

Conclusion: seed 0 keeps improving (best 329.8 at 95k, saved checkpoint 345.7, against 388 for the generator in use), with noise of about 30 between evaluations. Seed 1 is behind. The new checkpoint is not used yet: switching generators changes every GAN result, so it is kept for the reruns after exploration.

## 2026-11-22 — B-anchor on S2 against a pool of G-GAN images (local)

- `methods/s2/anchor.py --gallery`, `pipeline` 09ce1bf, GTX 1650. Gallery: the 5372 kept images of `gan_pool_r1amp_seed0_57568` (G-GAN seed 0, step 57568) in place of the 4666 training paintings; their CAV-MAE image embeddings from `methods/s4/extract.py --images` (same preprocessing as the paintings). Otherwise as the anchor run of 2026-11-21: eps start 0.005, alpha 0 and 0.1, 8 orientations of the three leading axes, judge z on training tracks, scored on the 789 test tracks (FD against held-out art).
- Since 2026-11-22 picking systems are references, not contenders (project owner: the system has to draw the image).

| Code | alpha | anchored start | test r | FD | test judge z | train/test judge z corr | lowest-GW start: test r, z |
|---|---|---|---|---|---|---|---|
| mean | 0 | (-1, -1, 1) | 0.685 | 485.7 | 5.67 | 0.99 | (1, -1, 1): 0.696, -0.59 |
| mean | 0.1 | (-1, -1, -1) | 0.575 | 496.5 | 5.55 | 0.98 | (-1, 1, -1): 0.607, 3.02 |
| standardised | 0 | (1, -1, 1) | 0.590 | 481.5 | 3.94 | 1.00 | (-1, -1, 1): 0.590, -3.16 |
| standardised | 0.1 | (1, 1, 1) | 0.575 | 498.9 | 3.41 | 0.96 | same |

- Range over the 8 orientations at alpha 0, mean code: test r 0.667 to 0.696, test judge z -4.63 to 6.09, train judge z -8.85 to 12.69.

Conclusion: against GAN images the judge separates orientations more sharply than against paintings (train z up to ±12), and its choice on training tracks predicts the test tracks almost perfectly (0.96 to 1.00). The orientation of axis 1 decides the judge's sign. Anchoring costs 0.011 of r (0.685 against 0.696) at alpha 0. Structure stays close to S2 on paintings (0.71), at FD 486 (GAN images) instead of real art.

## 2026-11-22 — S3 window consistency (local)

- `methods/s3/windows.py`, `pipeline` 1bcd7cd, GTX 1650. The four S3 runs of `abstraction-s3-gan` (G-GAN seed 0, step 57568). One image per window, 3 windows per test track (2367 images). Each image compared with all other images by cosine similarity, on DINOv2 (training encoder) and DINO v1 (ViT-B/16 CLS, not used in training). window_top1: most similar other image is from the same track (chance 0.0008). same_rank: median rank of the closest same-track image. ratio: mean same-track distance / mean different-track distance. Audio: the same measures on the MERT window codes.

| Source | DINOv2 window_top1 | same_rank | ratio | DINO v1 window_top1 | same_rank | ratio |
|---|---|---|---|---|---|---|
| audio codes (MERT) | 0.935 | 1 | 0.238 | | | |
| s3 | 0.217 | 16 | 0.574 | 0.239 | 11 | 0.555 |
| s3_noaug | 0.231 | 16 | 0.572 | 0.249 | 11 | 0.549 |
| shuffled | 0.157 | 39 | 0.662 | 0.178 | 31 | 0.658 |
| random_m | 0.046 | 152 | 0.798 | 0.058 | 130 | 0.808 |

Conclusion: the windows of a track are close in the audio code (93 % nearest-neighbour), and every trained mapper passes part of that on: the shuffled control, trained without any audio information in the loss, still reaches 16 %, because a smooth mapper sends close codes to close images. S3 adds a modest step over that (22 to 25 %), and the outside encoder agrees (DINO v1 slightly higher than DINOv2). The images of one track are similar, but far less similar than the windows themselves: most of the audio's window consistency is lost in the image. Augmentation does not change it. One seed.

## 2026-11-22 — Reach of the G-GAN towards DINOv2 descriptions (local)

- `methods/gan/reach.py`, `pipeline` d090fb1, GTX 1650. Step 1 of passing a picker's description (F1, S2) to a generator (project owner, 2026-11-22: contenders must draw the image; pickers become teachers). Targets: DINOv2 features of 192 random test-split paintings (never seen by the GAN). Per target: best of 512 prior samples by DINOv2 cosine, then 300 Adam steps on z (lr 0.01, loss -cos + 0.1 x prior-radius penalty, best z kept), plain and with the S3 augmentation at every step (aug). Scores are cosine similarities: DINOv2 to the target (searched), DINO v1 ViT-B/16 CLS of the drawn image to DINO v1 of the target painting (never used in the search), novelty = DINOv2 cosine to the nearest training painting. Two generators: in use (seed 0, step 57568, FD 388) and the latest (seed 0, step 101996, FD 346).

| Image | DINOv2, 57568 | DINO v1, 57568 | DINOv2, 101996 | DINO v1, 101996 | novelty (101996) |
|---|---|---|---|---|---|
| random prior sample | 0.333 | 0.197 | 0.352 | 0.209 | 0.709 |
| best of 512 samples | 0.636 | 0.377 | 0.640 | 0.392 | 0.760 |
| search, plain | 0.766 | 0.394 | 0.779 | 0.409 | 0.773 |
| search, aug | 0.669 | 0.403 | 0.683 | 0.419 | 0.764 |
| nearest training painting | 0.753 | 0.561 | 0.753 | 0.561 | — |

- Share of targets where the drawn image is closer (DINOv2) than the nearest training painting: plain 0.60 / 0.69, aug 0.21 / 0.30 (57568 / 101996). Grids: `runs/session_files/reach_grid_s57568.png`, `reach_grid_s101996.png` (rows: target, nearest training painting, best of 512, plain, aug; first 16 targets).

Conclusion: in DINOv2 terms the GAN can draw as close to a held-out painting as the closest real painting (0.78 against 0.75), but an encoder the search did not use sees little of that gain: DINO v1 rises from 0.39 (best of 512) to 0.41 only, against 0.56 for the nearest real painting. Most of what the search adds is read by DINOv2 alone. The augmentation lowers the DINOv2 score without raising DINO v1 much (0.42). In the grids the drawn images keep palette, texture family and density of the target (stripes, colour fields, splashes, line drawings), not its composition. The drawn images are not closer to a training painting than random samples are relative to their targets (novelty 0.77 against similarity 0.78). The later checkpoint is slightly better on every measure. For a picker-to-drawer transfer through this GAN, the description is carried at the level of palette and texture; a training target read by more than one encoder (DINOv2 and DINO v1), or a decoder built for DINOv2 descriptions (G-Diff, RAE), is needed to carry more.

## 2026-11-22 — S3 with a structure term through the G-GAN (Kaggle) and checks (local)

- Job `abstraction-s3-struct-gan`, `pipeline` 19eade2 (`methods/s3/train.py --w-struct`). As `abstraction-s3-gan` (generator `gan_r1amp_seed0_20261118T171246Z`, mean code, seed 0, 15000 steps, same augmentation), plus the Q1 structure term on the DINOv2 features with weight 1.0 (ganw10) or 0.3 (ganw03). Identification and S1 evaluation on the 789 test tracks; window consistency (`methods/s3/windows.py`) and second reader (`methods/s3/second_reader.py`) on the GTX 1650.

| Variant | track top1 clean / aug | top10 clean | bits clean | window top1 clean / aug | Mantel r | kNN overlap | FD | participation ratio |
|---|---|---|---|---|---|---|---|---|
| s3 (no structure term, earlier run) | 0.777 / 0.859 | 0.990 | 7.42 | 0.397 / 0.460 | 0.179 | — | 425.8 | — |
| w_struct 0.3 | 0.761 / 0.853 | 0.991 | 7.39 | 0.380 / 0.435 | 0.222 | 0.102 | 435.1 | 24.1 |
| w_struct 1.0 | 0.750 / 0.833 | 0.984 | 7.42 | 0.388 / 0.407 | 0.513 | 0.111 | 486.1 | 23.4 |
| Q1 + G-GAN (reference) | — | — | — | — | 0.537 | 0.108 | 458.2 | — |

| Variant | second reader DINO v1 top1 / bits | DINOv2 (fresh reader) top1 / bits | window_top1 DINOv2 / DINO v1 | ratio DINOv2 / DINO v1 |
|---|---|---|---|---|
| s3 (earlier) | 0.525 / 6.10 | 0.389 / 5.48 | 0.217 / 0.239 | 0.574 / 0.555 |
| w_struct 0.3 | 0.497 / 6.00 | 0.371 / 5.53 | 0.227 / 0.257 | 0.567 / 0.547 |
| w_struct 1.0 | 0.563 / 6.32 | 0.435 / 5.85 | 0.240 / 0.269 | 0.529 / 0.509 |

Conclusion: with the structure term at weight 1.0, S3 keeps its code (track named from its image in 75 % of cases, against 78 % without the term) and gains the distance structure of Q1 (Mantel r 0.51 against 0.18; Q1 through the same GAN 0.54). It is the first system that does both. The code is read at least as well by a fresh reader on DINO v1 (56 % against 53 %), and window consistency rises slightly (24 to 27 %). The cost is realism: FD 486 against 426. Weight 0.3 changes little. One seed.

## 2026-11-22 — Q1 through the G-GAN with a weak CAV-MAE anchor (Kaggle)

- Job `abstraction-q1-anchor-gan`, `pipeline` 19eade2 (`methods/s4/bridge.py --variant q1_anchor`). Q1 losses on the DINOv2 features (w_feat 1.0, w_lat 0.25, w_global 1.0, w_local 1.0) plus the S4 term (frozen CAV-MAE pairs each image with its own sound, InfoNCE) with weight 0.1 (a01) or 0 (a00). Generator `gan_r1amp_seed0_20261118T171246Z`, mean code, seed 0, batch 48, time limit 4.5 h (a00 stopped at step 10609, a01 at 10157), T4.

| Variant | CAV-MAE top10 | median rank (chance about 395) | mean own / other sim | Mantel r | Spearman | kNN overlap | FD | participation ratio |
|---|---|---|---|---|---|---|---|---|
| w_av 0 | 0.014 | 389 | 0.003 / 0.001 | 0.462 | 0.493 | 0.095 | 503.3 | 21.6 |
| w_av 0.1 | 0.015 | 312 | 0.027 / 0.001 | 0.483 | 0.506 | 0.103 | 538.1 | 20.1 |
| S4 alone (earlier) | 0.070 | 136 | 0.097 / 0.001 | 0.063 | — | 0.073 | 472.1 | 23.8 |

Conclusion: the weak anchor moves CAV-MAE's ranking only a little (median rank 312 against 389) and costs realism (FD 538 against 503), without lowering structure. The w_av 0 run is below the earlier Q1 through the GAN (r 0.46 against 0.54, FD 503 against 458), so the bridge setup (batch 48, 4.5 h) is weaker than the Q1 training it replaces. The anchor is not pursued further for now. One seed.

## 2026-11-23 — Second reader on Q1 through the G-GAN (local)

- `methods/s3/second_reader.py` (`pipeline` ad57358; now reads the code variant of Q1 checkpoints), GTX 1650. Mappers of `abstraction-gan-bridge-q1` (q1 and shuffled, generator `gan_r1amp_seed0_20261118T171246Z`). As for S3: mapper frozen, one image per track, a fresh reader trained by InfoNCE on the 6408 training tracks, scored on the 789 test tracks (chance top1 0.0013).

| Run | DINO v1 top1 | top10 | bits | DINOv2 top1 | top10 | bits |
|---|---|---|---|---|---|---|
| q1 | 0.430 | 0.867 | 5.66 | 0.286 | 0.762 | 4.97 |
| q1 shuffled | 0.231 | 0.688 | 4.53 | 0.162 | 0.610 | 4.07 |
| s3_struct w 1.0 (2026-11-22) | 0.563 | 0.925 | 6.32 | 0.435 | 0.891 | 5.85 |
| s3 random_m (2026-11-22) | 0.056 | 0.272 | 2.11 | 0.049 | 0.228 | 1.79 |

Conclusion: a fresh reader names Q1 tracks from their images in 43 % of cases (DINO v1), against 23 % for Q1's shuffled control and 6 % for an untrained S3 mapper. A trained mapper makes tracks distinguishable even without audio information in its loss, so the shuffled control, not chance, is the baseline. Q1 training adds about 20 points over it; S3 with the structure term reaches 56 % at similar structure (r 0.51 against 0.54). The fresh reader on the S3 shuffled control was not run. One seed.

## 2026-11-23 — Picker-to-drawer (teach) on the laptop, and a repetition measure

- `methods/gan/teacher_targets.py` and `methods/gan/teach.py` (`pipeline`), GTX 1650. Teacher: F1 mapper `f1_mapper_f1_seed0_20261112T213100Z`. Generator `gan_r1amp_seed0_20261121T142711Z` (FD 346). Variant teach, batch 16, 6000 steps, seed 0. Run `teach_teach_laptop_seed0_20261122T195845Z`, evaluation `s1_eval_teach_teachlap_seed0_20261122T214441Z`. The shuffled control was not run.
- Test cosine of the drawn image to the teacher's pick: DINOv2 0.407, DINO v1 0.424 (30-step smoke test 0.335 / 0.373; the teacher's own pick 0.622 / 0.820).
- S1 evaluation: Mantel r 0.152, kNN overlap 0.069 (chance 0.013), trustworthiness 0.687, FD 431, participation ratio 20.4 (test art 44.8).
- Repetition measure (new, `twins`): share of the 789 test images whose nearest other test image has DINOv2 cosine above 0.9.

| System | twin > 0.9 | twin > 0.8 |
|---|---|---|
| G-GAN, random codes | 0.20 | 0.55 |
| S1 + GAN | 0.76 | 0.92 |
| Q1 + CAV-MAE pull, a 0.1 / a 0 | 0.33 / 0.31 | 0.79 / 0.77 |
| S3 + GAN / shuffled | 0.13 / 0.13 | 0.62 / 0.58 |
| S3 + structure, w 0.3 / w 1.0 | 0.16 / 0.30 | 0.67 / 0.74 |
| S4 / shuffled | 0.16 / 0.16 | 0.71 / 0.68 |
| Teach (drawer) | 0.15 | 0.60 |

Conclusion: the twin share follows Arunav's Atlas marks. The systems he called "blobs" or repetitive (S1 + GAN, Q1 + pull, S3 + structure w 1.0) have the highest shares. The drawer repeats little and its samples look varied, but it keeps little structure (r 0.15 against 0.51 for S3 + structure w 1.0). One seed; no shuffled control yet.

## 2026-11-23 — Laptop queue batch 1: drawer control, fresh readers, structure + batch repetition penalty

- GTX 1650, queue `runs/queue/batch1.sh` (`pipeline` 666a07c). Improved G-GAN (`gan_r1amp_seed0_20261121T142711Z`) except the S3 shuffled control (generator in use). Seed 0 for training; the fresh reader uses its default seed 42. Twin share (S1 evaluation, new): share of the 789 test images whose nearest other test image has DINOv2 cosine above 0.8 / 0.9. Test art: 0.10 / 0.007.

| Run | cos to teacher pick (v2 / v1) | Mantel r | FD | PR | twin 0.8 / 0.9 | fresh reader top1 (DINO v1 / DINOv2) |
|---|---|---|---|---|---|---|
| teach | 0.407 / 0.424 | 0.152 | 431 | 20.4 | 0.60 / 0.15 | 0.354 / 0.237 |
| teach shuffled | 0.345 / 0.341 | 0.057 | 439 | 24.5 | 0.62 / 0.12 | 0.292 / 0.208 |
| teach + struct 1.0 + batch penalty 1.0 | 0.377 / 0.366 | 0.254 | 500 | 24.9 | 0.63 / 0.20 | 0.330 / 0.210 |
| S3 shuffled control (2026-11-21 run) | — | — | — | — | 0.58 / 0.13 | 0.293 / 0.169 |

Conclusions:
- S3's shuffled control is named by a fresh reader in 29 % of cases (DINO v1). S3 + structure w 1.0 reaches 56 %, so its training adds about 27 points; Q1 + GAN (43 %) adds about 20 over its own control (23 %).
- The drawer adds only 6 points over its shuffled control (35 % vs 29 %) and r 0.15 vs 0.06. Teaching from F1's single pick carries little of the song; most of what it learns is art-likeness.
- The batch-only repetition penalty stayed at zero after a few steps: within 16 random tracks no pair exceeds cosine 0.8. Replaced by a penalty against a bank of the last 1024 images (`methods/s3/common.py` TwinPenalty, 2fdf8e5), which fires on about 70 % of images at the start of training.
- Every G-GAN system, random codes included (0.55), has a twin share at 0.8 far above real art (0.10). Part of the repetition is the generator's own range.

## 2026-11-23 — Laptop queue batch 2: S3 + structure with the repetition penalty

- GTX 1650, `runs/queue/batch2.sh` (`pipeline` 2fdf8e5). Improved G-GAN (`gan_r1amp_seed0_20261121T142711Z`). S3 variant s3_struct, w_struct 1.0, batch 32 (Kaggle uses 64), 12000 steps, seed 0. Repetition penalty `TwinPenalty`: squared hinge on each image's nearest DINOv2 cosine among the batch and the last 1024 images, margin 0.8. Fresh reader as before (seed 42).

| Run | w_div | Mantel r | FD | PR | twin 0.8 / 0.9 | fresh reader top1 (DINO v1 / DINOv2) |
|---|---|---|---|---|---|---|
| lapd0 | 0 | 0.399 | 468 | 23.6 | 0.65 / 0.12 | 0.447 / 0.381 |
| lapd10 | 10 | 0.351 | 461 | 24.7 | 0.68 / 0.14 | 0.526 / 0.378 |
| lapd30 | 30 | 0.362 | 426 | 29.1 | 0.55 / 0.04 | 0.537 / 0.387 |
| teach sd10 (drawer, w_struct 1.0) | 10 | 0.339 | 472 | 21.6 | 0.65 / 0.12 | not run |

Test art: twin 0.10 / 0.007, PR 44.8.

Conclusion: at w_div 30 near-copies (0.9) fall from 12 % to 4 % of test images, FD falls 468 → 426 and PR rises 23.6 → 29.1, while Mantel r drops 0.04 and the fresh reader names more tracks (45 → 54 %). w_div 10 is too weak to change the twin share. The samples at w_div 30 still show a recurring family (vertical stripes in orange, teal and navy). One seed; batch 32 lowers r against the Kaggle batch-64 run (0.51).

## 2026-11-24 — Laptop queue batch 3: penalty strength, margin, lighter structure

Setup as batch 2 (`runs/queue/batch3.sh`, seed 0, batch 32, 12000 steps, improved G-GAN).

| Run | w_div | margin | w_struct | Mantel r | FD | PR | twin 0.8 / 0.9 | fresh reader top1 (DINO v1 / DINOv2) |
|---|---|---|---|---|---|---|---|---|
| lapd30 (batch 2) | 30 | 0.8 | 1.0 | 0.362 | 426 | 29.1 | 0.55 / 0.04 | 0.537 / 0.387 |
| lapd100 | 100 | 0.8 | 1.0 | 0.310 | 434 | 29.3 | 0.48 / 0.035 | 0.461 / 0.338 |
| lapd30m7 | 30 | 0.7 | 1.0 | 0.321 | 424 | 32.9 | 0.41 / 0.037 | 0.458 / 0.332 |
| lapd30s03 | 30 | 0.8 | 0.3 | 0.242 | 401 | 30.4 | 0.47 / 0.066 | 0.485 / 0.378 |

Conclusion: a stronger penalty or a lower margin removes more look-alikes (twin 0.8 down to 0.41) but costs structure (r 0.31 to 0.32) and naming (46 %). A lighter structure term gives the best FD (401) at r 0.24. w_div 30 at margin 0.8 with w_struct 1.0 stays the best balance. One seed each; batch 4 repeats the key pair on seed 1 and adds the shuffled control with the penalty.

## 2026-11-24 — Are image twins audio neighbours?

- Script: scratchpad `twin_audio.py`, now `twin_audio_backing` in `methods/s1/evaluate.py` (`pipeline` 38f1756; key `image_distribution.twin_audio_backing`). DINOv2 image features and audio codes of the 789 test tracks, each centred; cosine. A twin is the nearest other image with cosine above 0.9 (centred, so shares are lower than `twin_share`). For each twin, the rank of the twin's track among all 788 other tracks by audio cosine (chance median about 394).

| Run | twin share | median audio rank of twin | twin in audio top 10 | audio cos of twin pairs |
|---|---|---|---|---|
| S1 + GAN | 0.12 | 12 | 0.46 | 0.46 |
| Q1 + CAV-MAE pull, a 0.1 | 0.14 | 36 | 0.27 | 0.35 |
| S3 + structure w 1.0 (Kaggle) | 0.09 | 26 | 0.29 | 0.35 |
| S3 + structure, laptop, w_div 0 | 0.04 | 28 | 0.32 | 0.38 |
| S3 + structure, laptop, w_div 30 | 0.01 | 14 | 0.45 | 0.56 |
| Drawer, shuffled control | 0.05 | 52 | 0.10 | 0.26 |

Mean audio cosine of all pairs: 0.00.

Conclusion: image twins come from similar-sounding tracks far more often than chance (median rank 12 to 36 against 394), but in most systems 55 to 70 % of twins are not among each other's ten nearest tracks: the mapper merges moderately similar tracks. The repetition penalty removes mainly these; its remaining twins are the most audio-backed. Even the drawer's shuffled control gives rank 52, because a smooth mapper sends nearby codes to nearby images.

- Generation is deterministic: mapper and generator add no noise at inference (augmentation is used only in training), so the same audio input always gives the same image. Different 10 s windows of one track give different images (window identification 24 to 37 %).
- Image Atlas v7: 10 cards in the new groups "Drawer" and "S3 no-repeat", with near-copy and fresh-reader numbers.

## 2026-11-25 — Evaluation-only diagnostics for the combination analysis

Script: session scratchpad `diagnostics.py` (no training; DINOv2 CLS cache `datasets/dino_art_features/cls_112.npz`, existing evaluation features). Assumptions of `docs/combinations.md` checked:

- **Q1 + G-GAN repeats as much as S3 + structure.** Twin share (DINOv2 cosine of nearest other image) 0.8 / 0.9: Q1 + GAN 0.71 / 0.27, its shuffled control 0.54 / 0.08, random codes 0.55 / 0.20. The no-repeat penalty has a clear target in Q1 + GAN.
- **Real art repeats far less than any GAN system, in every style.** Twin share 0.8 / 0.9 at 789 images: all styles 0.10 / 0.02 (test split 0.10 / 0.007); Abstract Expressionism 0.09 / 0.00, Minimalism 0.13 / 0.02, Colour Field 0.23 / 0.04, Action painting (98 images) 0.15 / 0.00. Colour Field is the most repetitive style, but still far below the GAN's 0.55. The repetition comes mainly from the generator, not from the data.
- **Skew of leading axes.** Mapper input (mean code, standardised per dimension), axes 1-5: 0.25, -0.64, -0.21, -1.00, -0.15. DINOv2 art features, axes 1-5: -0.06, -0.03, 0.16, -0.10, -0.49. The art cloud stays near-symmetric on its leading axes; the code is skewed on axes 2 and 4, so a reflection along axis 2 moves inputs into regions the mapper rarely saw.
- **The art has little cluster structure.** k-means silhouette, art vs a Gaussian with the same covariance: K 4 0.048 vs 0.027, K 8 0.030 vs 0.022, K 16 0.028 vs 0.018. Clusters follow the four styles only partly (majority-style share 0.65 to 0.71). Hard B-cluster has little to work with on this art.
- **S3 does not overfit within the steps used.** Validation in-batch top1 rises to the end (Kaggle w 1.0: 0.983 at 15000; laptop lapd30: 0.970 at 12000) and the validation structure loss keeps falling (Kaggle w 1.0: 0.81 at 500 to 0.48 at 15000). Longer training is not ruled out; more data is not shown to be needed for S3 at this size.
- WikiArt class counts were not checked (needs the WikiArt index online).

## 2026-11-25 — Laptop batches 4 and 5: second seed of the no-repeat penalty, drawer taught by S2 over a GAN pool, B-anchor by reflection

All on the improved G-GAN (FD 346), laptop GTX 1650. Fresh reader: DINO v1 second reader top1 among 789 test tracks (chance 0.13 %).

**Batch 4: S3 + structure w 1.0, batch 32, 12000 steps, seed 1, and a shuffled control.**

| Run | r | kNN overlap | FD | Twins 0.8 / 0.9 | Fresh reader |
|---|---|---|---|---|---|
| w_div 30, seed 0 | 0.36 | 0.10 | 426 | 0.55 / 0.04 | 54 % |
| w_div 0, seed 0 | 0.40 | — | 468 | — / 0.12 | 45 % |
| w_div 30, seed 1 | 0.39 | 0.10 | 432 | 0.59 / 0.09 | 48.8 % |
| w_div 0, seed 1 | 0.40 | 0.12 | 466 | 0.67 / 0.16 | 48.2 % |
| w_div 30, shuffled codes | 0.06 | 0.05 | 370 | 0.42 / 0.04 | 29 % |

Conclusion: on both seeds the penalty roughly halves the share of near-copies at 0.9 and lowers FD by about 35, with structure unchanged. The fresh-reader gain of seed 0 (54 against 45 %) does not repeat on seed 1 (48.8 against 48.2 %); the penalty makes images more varied, not reliably more identifiable. The shuffled control with the penalty gives 29 %, the same as without it.

**Batch 5a: drawer taught by S2 over a G-GAN pool (`methods/gan/pool_teach.py`).** Pool: 12000 latents, 10861 kept after the blank rule. Assignment of the 6408 training tracks, each to a different pool image: GW 0.048; structure of the assignment on training tracks r 0.68 (shuffled 0.01). Mapper regressed to the assigned latents: training track-mean cosine to the target 0.95 in both variants.

| Variant | Test r | kNN overlap | FD | Twins 0.8 / 0.9 | Fresh reader |
|---|---|---|---|---|---|
| teach | 0.14 | 0.02 | 383 | 0.36 / 0.01 | 1.9 % |
| shuffled | 0.00 | 0.01 | 382 | 0.31 / 0.02 | 0.5 % |

Conclusion: the assignment itself carries strong structure (r 0.68), but a mapper that regresses generator latents memorises the training tracks and generalises poorly to test tracks (r 0.14, fresh reader 1.9 %). Nearby GAN images are not nearby latents, so latent targets are a poor regression target. Any further version should pull the generated image's features towards the assigned image's features through the generator (S3 with fixed targets), not regress latents.

**Batch 5b: B-anchor by code reflection (`methods/s4/reflect_anchor.py`, 3 axes, 8 sign patterns, CAV-MAE judge).**

| Mapper | Chosen | Test gap, original | Best / worst test gap of the flips | Val-test gap correlation | Test median rank, original |
|---|---|---|---|---|---|
| S3 + structure, lapd30 | original | 0.024 | 0.023 / −0.008 | 0.93 | 301 of 789 |
| Drawer (F1-taught) | original | 0.014 | 0.014 / −0.010 | 0.89 | 340 of 789 |

Conclusion: in both mappers the trained orientation is already the one CAV-MAE prefers; every flip is equal or worse, and validation and test agree on the ordering. Reflection gives no gain. Caveat: the code is skewed on axis 2 (−0.64), so flips also move inputs off the training distribution, which by itself lowers the scores. The CAV-MAE signal remains weak (median rank 301 against chance 395).

## 2026-11-25 — Laptop batch 6: S3 + structure + penalty 30 trained twice as long

Same as `lapd30` (S3 + structure w 1.0, w_div 30, batch 32, improved G-GAN, seed 0) but 24000 steps instead of 12000.

| Steps | r | kNN overlap | FD | Twins 0.8 / 0.9 | Fresh reader (DINO v1 / DINOv2) | Window identification |
|---|---|---|---|---|---|---|
| 12000 | 0.36 | 0.10 | 426 | 0.55 / 0.04 | 54 % / — | — |
| 24000 | 0.48 | 0.10 | 400 | 0.55 / 0.06 | 52 % / 42 % | 44 % |

Conclusion: longer training raises structure clearly (0.36 to 0.48, the highest of any laptop S3 run) and lowers FD (426 to 400), with near-copies and the fresh reader about the same. This agrees with the validation curves, which were still improving at 12000 steps. Batch 7 repeats 24000 steps on seed 1 and runs 48000 steps on seed 0.
