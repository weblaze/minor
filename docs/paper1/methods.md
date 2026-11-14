# Paper 1: methods

Source of method definitions: `methods/s1/README.md` (branch `s1-structure`), `methods/q1/README.md` (`q1-structure`), `methods/f1/README.md` (`f1-feature`), the `config.yaml` files on those branches, the Kaggle job files `scripts/kaggle/jobs/*.yaml` on the method branches, and shared code on `base`. Source of all numbers: `docs/run_log.md`. Configuration values below were checked against the commits named in the run log: the S1 configuration at `e040aba` and `8feff77` is identical; the Q1 configuration is identical at `7c5da53`, `eaffe89`, `76b055e`, `6bb84a7` and `648394a`; the F1 configuration is identical at `aa6783c` and `4b57977`. Per-run overrides (audio code, KL weight, lambda) are given on the command line of the job files and are stated below.

## 1. Constraints

- No text-supervised model at any stage (no CLAP, CLIP, ImageBind, captions or mood words).
- No paired data: no track is associated with any image.
- No human labels as a training signal. FMA genre labels, FMA audio descriptors, Echo Nest attributes and WikiArt style names are used only in post hoc interpretation and enter no training step or model selection.
- Hand-written correspondences between sound and image properties are excluded.
- Human judgement is reserved for evaluation; none has been made in phase 1.

## 2. Data

### 2.1 Music

- Source: FMA-small, obtained from the Kaggle dataset `imsparsh/fma-free-music-archive-small-medium`, directory `fma_small`. 8000 mp3 files found.
- Embedded: 7994 tracks. Skipped: 6 (`098565`, `098567`, `098569` shorter than 30 s; `099134`, `108925`, `133297` not decodable; these are the six files listed as defective in the FMA documentation for `fma_small`).
- Per track: the middle 30 s, cut into three windows of 10 s.
- FMA-small has 8 genres; the run log uses a genre chance level of 0.125 ("8 balanced genres").

### 2.2 Art

- Source: Kaggle dataset `steubk/wikiart`, built with `scripts/build_art_set.py`. Images are EXIF-transposed, converted to RGB and resized to 128 x 128 px (bilinear; no crop, aspect ratio not preserved).
- Four styles: Abstract_Expressionism 2782, Color_Field_Painting 1615, Minimalism 1337, Action_painting 98; total 5832. Unreadable files: 0. No de-duplication was performed.

### 2.3 Splits

- Every track and image is assigned by the MD5 hash of its identifier (track identifier or file name), modulo 100: buckets 0 to 9 test, 10 to 19 validation, 20 to 99 train (80 / 10 / 10).
- Tracks: 6408 train, 797 validation, 789 test. Images: 4666 train, 571 validation, 595 test.
- The official FMA split is not used (stated as a deviation from the original S1 plan in `methods/s1/README.md`).
- All code transforms and normalisation statistics are fitted on the training split.

## 3. Encoders

Both encoders are frozen and were trained without text.

### 3.1 Audio: MERT-v1-95M

- Hugging Face model `m-a-p/MERT-v1-95M`, 24 kHz input, revision not pinned (`mert.revision: null`, main branch at extraction time).
- All 13 hidden states of the model are read with forward hooks (under transformers 5.20.2 the model output carried `hidden_states = None`), averaged over time within each window.
- Stored array per track: (3 windows, 13 layers, 768), float16.
- Extraction: Tesla T4, torch 2.11.0+cu130, transformers 5.20.2, commit `7f190ac`.

### 3.2 Image: DINOv2 ViT-S/14

- Hugging Face model `facebook/dinov2-small`. Input images in [-1, 1] are resized to 112 px (8 x 8 patches) and normalised with ImageNet statistics. Output: CLS token, 384-d.
- In S1 and Q1 DINOv2 is applied, differentiably, to every decoded image during training. In F1 it is applied once to the art collection and the features are cached.
- The same DINOv2 feature space is used for the training losses and for evaluation (see `deviations_and_limitations.md`).

## 4. Audio codes

A code transform maps the per-track array (3, 13, 768) to one code per window. Statistics are fitted on training windows.

| Code | Definition | Dimension |
|---|---|---|
| mean | unweighted mean of the 13 layer vectors | 768 |
| standardised | each layer z-scored per dimension with training statistics, then averaged over layers | 768 |
| whitened | standardised, centred, projected on the leading 128 principal axes and scaled to unit variance per axis | 128 |
| single layer (F1 diagnosis only) | standardised, restricted to one layer (0, 3, 6, 9 or 12) | 768 |

Training draws one random window per track per batch. Evaluation uses one code per track, the mean of its three window codes. Mapper inputs are additionally standardised per dimension with training statistics.

Properties measured before training (diagnostic D1, 789 test tracks): see results table (b1).

## 5. Bridges

All bridges share the mapping network `M`: an MLP with `depth` blocks of Linear, LayerNorm and GELU (residual connection where input and output widths agree), followed by a linear output layer. Hidden width 1024, depth 3, deterministic. Only `M` (and, in S1 cycle variants, the inverse network) receives gradient updates.

Shared structure term:

```
L_struct = 1 - Pearson(D_a, D_v)  over the B(B-1)/2 off-diagonal pairs of a batch
D_a[i,j] = cosine distance between audio codes a_i and a_j
D_v[i,j] = cosine distance between image features v_i and v_j
```

Sliced Wasserstein distance (SWD): squared sliced 2-Wasserstein distance with 256 random unit projections; samples of unequal size are compared on the quantile grid of the smaller one.

### 5.1 S1: mapping into a spatial VAE latent

```
a -> M -> z in R^(4 x 16 x 16) -> VAE decoder (frozen) -> image (128 px) -> DINOv2 -> v in R^384
L = L_dist + lambda_struct * L_struct (+ mu_cycle * L_cycle)
L_dist  = SWD between {z_i} and VAE latents of real training images (latents in units of the
          standard deviation of training latents)
L_cycle = mean squared error between I(z_i) and the standardised code a_i; I is an inverse MLP of the
          same architecture
```

VAE (stage A, `train_vae.py`, seed 0): convolutional, spatial latent 4 x 16 x 16, base width 64, 5.84 M parameters. Objective: L1 reconstruction + `kl_weight` x KL (linear warm-up over 5000 steps) + `dino_weight` x mean squared error between DINOv2 tokens of input and reconstruction. Batch 32, AdamW, learning rate 2e-4 with 500 warm-up steps, 80 epochs (11600 steps), mixed precision. `kl_weight` 0.001, `dino_weight` 0.1. The checkpoint with the lowest validation L1 is kept (0.0763, epoch 78). Standard deviation of training latents: 1.093. FD of reconstructions against test art: 163.0.

Mapper: input dimension 768 (mean or standardised code) or 128 (whitened), output 1024. Batch 64 tracks (one window each), 256 real latents per step. AdamW, learning rate 1e-4, cosine decay to zero, gradient-norm clipping at 1.0, float32 (float16 gave non-finite gradients through decoder and DINOv2 in the smoke run), 20000 steps, validation every 500 steps. `lambda_struct` 1.0 (selected from {0.1, 0.3, 1, 3}; see results (a1)). `mu_cycle` 1.0. The checkpoint of the final step is evaluated.

S1 variants and controls:

| Name | Objective | Role |
|---|---|---|
| `s1` | L_dist + lambda L_struct | method |
| `s0` | L_dist + mu L_cycle | objective of the 2025 cycle-consistency pipeline (bridge S0); no structure term |
| `s1_cycle` | L_dist + lambda L_struct + mu L_cycle | whether a cycle term adds to S1 |
| `shuffled` | as `s1`, D_a computed on a permutation of the batch | same terms, uninformative structure target |
| random M | untrained mapping | distance preservation of a random network |

Audio codes run: mean (all variants, seed 0, commit `8feff77`); whitened (`s1` only, seed 0, second clone at `7a17f65`); standardised (`s1` only, seed 0, second clone at `d3b327b`). All S1 mapper runs use the VAE of run `s1_vae_seed0_20261110T155619Z`.

### 5.2 Q1: vector VAE with feature-space distribution term

```
a -> M -> z in R^128 -> VAE decoder (frozen) -> image (128 px) -> DINOv2 -> v in R^384
L = w_feat L_feat + w_lat L_lat + w_global L_global + w_local L_local
L_feat   = SWD({v_i}, DINOv2 features of real training art)
L_lat    = SWD({z_i}, samples mu + sigma * eps of the posteriors of real training art)
L_global = L_struct as above
L_local  = mean_i KL(P_i || Q_i),  P_i = softmax_j(-D_a[i,j] / (mean(D_a) T)),  Q_i likewise from D_v,  j != i
```

Weights: `w_feat` 1.0, `w_lat` 0.25, `w_global` 1.0, `w_local` 1.0, temperature T 0.5. Mapper as in S1 with output 128; batch 64 tracks, 256 real samples per step; AdamW, learning rate 1e-4, cosine decay, gradient-norm clipping at 1.0, float32, 20000 steps. Final-step checkpoint evaluated.

VAE (`train_vae.py`, seed 0): convolutional, 128-d vector latent, base width 64, 15.66 M parameters; same objective form as S1 (L1 + `kl_weight` x KL averaged over latent dimensions with 5000-step linear warm-up + 0.1 x DINOv2 token MSE); batch 32, learning rate 2e-4, 80 epochs, mixed precision. `kl_weight` swept over {0.02, 0.1, 0.3, 1.0, 3.0}.

Pre-training diagnostics (`diagnostics.py`, `gate.py`, `baselines.py`):

| Id | Measures |
|---|---|
| D1 | same-track window retrieval, coefficient of variation of pairwise distances and participation ratio of each audio code (789 test tracks); selects the code without labels |
| D2 | augmented-view retrieval, distance spread and participation ratio of the DINOv2 art features (595 test images) |
| D3 | Wasserstein-1 distance between mean-normalised pairwise-distance distributions of audio codes and image features |
| D4 (gate) | FD and kernel distance to test art of reconstructions, prior samples, interpolations and samples of a 10-component Gaussian mixture (2000 samples each); active latent dimensions. Rule fixed before any run: pass if FD(prior) <= 2.0 x FD(reconstructions) |
| D5 | non-learned baselines (below) |

Gate outcome: `kl_weight` 0.3 is the lowest weight that passes; 0.02 fails (ratio 4.68). Both VAEs were carried to a seed-0 mapper run; all later Q1 runs use the 0.02 VAE (run `q1_vae_seed0_20261110T155658Z`) with `--ignore-gate` (protocol deviation). The 0.3 VAE is run `q1_vae_kl0.3_seed0_20261110T192251Z`.

Q1 variants and controls:

| Name | Objective | Run |
|---|---|---|
| `q1` | all four terms | whitened code seed 0 (VAE 0.3 and 0.02); mean code seeds 0, 1, 2 (VAE 0.02); standardised code seed 0 (VAE 0.02) |
| `shuffled` | as `q1`, D_a on a permutation of the batch | mean code, seed 0 |
| `no_feat` | `w_feat` = 0 | mean code, seed 0 |
| `no_local`, `no_global` | `w_local` = 0, `w_global` = 0 | not run |
| random M | untrained mapping | not run for Q1 |

### 5.3 F1: mapping into image-feature space with retrieval

```
a -> M -> f in R^384                                   (direct output)
f -> nearest training painting by cosine similarity of DINOv2 CLS features   (painting output)
L = L_dist + lambda_struct * L_struct
L_dist: SWD (or energy distance) between outputs and training-art features, both divided by the
        standard deviation of the training-art features
```

No decoder and no feature extractor in the training loop. Gallery: the 4666 training paintings. Test art is used only as the reference of the distribution scores. Mapper: input 768 (or 128), output 384; batch 256 tracks, 256 real features per step; AdamW, learning rate 1e-4; 4000 steps; `lambda_struct` 1.0; default distribution term SWD. Final-step mapping evaluated. One mapping run takes about one minute on a 4 GB GPU.

Variants and controls (`train_mapper.py`, standardised code, seed 0):

| Name | Definition |
|---|---|
| `f1` | L_dist + lambda_struct L_struct |
| `f0` | L_dist only |
| `shuffled` | as `f1`, audio distance matrix of each batch permuted |
| random M | untrained mapping |

Staged diagnosis (`diagnose.py`, seed 0, codes mean, standardised, whitened and single layers 0, 3, 6, 9, 12):

| Stage | Definition |
|---|---|
| `train_direct` | r on outputs for a subset of training tracks |
| `test_direct` | r on outputs for the test tracks |
| `test_painting` | r after replacing each test output by its nearest training painting |
| `gallery_<n>` | as above with n randomly chosen paintings (500, 1000, 2000 configured) |
| `oracle_painting` | mapping trained on the test tracks themselves, then nearest painting (reference, not a bound) |

Direct assignment (exploratory, script not in the repository): soft assignment of the 789 test tracks to the 4666 paintings, optimised by gradient descent on 1 - r with temperature annealing, then hardened.

Experiment grid (`realism.py`, seed 0, mean and standardised codes; 27 configurations). Configuration components:

| Component | Options |
|---|---|
| Output head | `free` (output is the feature vector); `pca` (M outputs k coordinates in the principal axes of training-art features, k = 32 or 64); `soft` (output is the softmax(cos(q, g) / tau)-weighted mean of gallery features g, tau 0.05, 0.02 or 0.01); `st` (forward pass uses the nearest gallery feature, backward pass as `soft`; tau 0.02, 0.05 or 0.1) |
| Distribution term | SWD; critic (hinge loss against a spectral-norm MLP 384-512-512-1, Adam 2e-4, betas 0.5, 0.9), weight 0.3, 1 or 3; chamfer (mean distance of each output to its nearest of 2048 gallery features plus of each real feature to its nearest output), weight 1, 3 or 10; none |
| Structure term | global (L_struct); local (row-wise neighbour KL, temperature 0.5); views (global term averaged over single-layer codes 0, 3, 6, 9, 12) |
| Other | 16 standard-normal noise inputs; batch 1024 |

Default: head `free`, SWD weight 1, global structure weight 1, no noise, batch 256.

Combination sweep (`realism.py --grid sweep`, via `phase1_batch.py`, commit `4b57977`, seed 0): 5 heads (`free`, `pca64`, `soft05`, `soft02`, `st05`) x 5 distribution terms (`swd1`, `swd5`, `energy1`, `swd1_energy1`, `swd1_chamfer01`) x 4 structure terms (`global`, `global_local`, `global_views` = 0.5 global + 0.5 views, `global3` = global weight 3) = 100 combinations per code. Left out because the grid had shown them to fail: no distribution term, the critic, chamfer as the only distribution term, straight-through retrieval without a distribution term. Reruns: the five highest-r combinations meeting the rule (at least 300 distinct paintings, FD of the painting output at most 450) and the baseline `free__swd1__global`, seeds 1 to 4. The rule is fixed in `phase1_batch.py` before the sweep ran; selection uses scores on the test tracks.

Orientation runs (`orientation.py`): one mapping per seed; recorded per seed: sign and value of the correlation between the test tracks' scores on principal axis k (k = 1, 2, 3) of the audio code and their paintings' scores on principal axis k of the art features; r of the painting output; SWD and energy distance between outputs for validation tracks and validation-art features; validation structure loss; training history every 100 steps (batch distance correlation, batch SWD). Settings: configuration `swd` (weight 1, 5, 20), `soft_0.02_swd`; network sizes 1024 x 3, 256 x 3, 1024 x 1, 2048 x 3 (hidden width x depth); 10, 20 or 50 seeds.

## 6. Non-learned baselines (Q1 D5)

Both use one linear map built from second-order statistics of the training sets: the i-th principal axis of training audio codes (track means) is sent to the i-th principal axis of the VAE posterior means of training art, i = 1 to 32, with each coordinate rescaled from the audio standard deviation to the latent standard deviation on that axis. Axes are matched by variance rank; signs are those returned by the eigendecomposition (arbitrary).

- PCA alignment: the mapped latent is decoded by the VAE.
- Retrieval: the real training artwork whose posterior mean is nearest (Euclidean, in the 32 standardised coordinates) to the mapped latent.

Baselines were computed for the standardised code (VAE 0.02) and the whitened code (VAE 0.3 and 0.02). They were not computed for the mean code.

## 7. Evaluation

All evaluation is on the 789 test tracks, one output per track (mean of window codes). D_a: cosine distances of the audio code the mapping was trained on. D_v: cosine distances of DINOv2 CLS features of the output (generated image, direct F1 output or retrieved painting).

| Metric | Definition |
|---|---|
| Mantel r, rho | Pearson and Spearman correlation between the upper triangles of D_a and D_v. One-sided permutation p-value from 9999 joint row-column permutations of D_v: p = (1 + #{r_perm >= r}) / (1 + 9999); minimum 0.0001 |
| kNN overlap | mean fraction of shared 10 nearest neighbours under D_a and D_v; chance k / (n - 1) = 0.0127 |
| Trustworthiness | Venna and Kaski (2001), scikit-learn implementation, k = 10, audio distances precomputed |
| Frechet distance (FD) | Frechet distance between Gaussians fitted to DINOv2 features of outputs and of the 595 test images. References: training art 113.3; S1 VAE reconstructions 163.0; Q1 VAE reconstructions 208.5 (`kl_weight` 0.02) and 576.8 (0.3). The estimate is biased upward for small samples |
| Kernel distance | unbiased MMD^2 with kernel (x.y / d + 1)^3 over 100 subsets of up to 1000; reported only in the gate |
| Participation ratio (PR) | (sum of covariance eigenvalues)^2 / sum of squared eigenvalues; test art 44.8 |
| Mean pairwise distance | mean cosine distance between output features; test art 0.672 |
| Distinct paintings | number of distinct paintings assigned to the 789 test tracks (F1) |
| Displacement | mean cosine distance between an F1 output (or query) and its nearest painting |
| r views | mean r of the painting output against the five single-layer codes |
| Pixel-feature structure | Pearson r between D_a and Euclidean distances of feature sets computed from pixels, independent of DINOv2: colour mean and standard deviation per channel (6-d); 64-bin colour histogram (4 levels per channel); 8 x 8 average-pooled layout (192-d); radial log power spectrum of the grey image (63-d) |
| Orientation statistics | per principal axis k = 1, 2, 3: correlation between audio-code scores and output scores on the art-feature axis, its sign, counts of each sign over seeds; skewness of the axes; mean cosine similarity between the paintings assigned to the same track by two seeds, against shuffled tracks |
| Seed agreement (Q1) | Pearson correlation between image distance matrices of two seeds; cosine similarity of image features of the same track against shuffled tracks |

Post hoc interpretation (evaluation only, no training): FMA genre and artist neighbour rates (k = 10), genre eta^2 per principal axis, 5-fold ridge R^2 from 100 low-level descriptors (`features.csv`), Spearman correlation with Echo Nest attributes (1294 tracks), contingency of genre against WikiArt style of the assigned painting with Cramer V.

## 8. Hardware and run bookkeeping

- S1 and Q1: Kaggle kernels, 2 x Tesla T4 (15360 MiB), one chain per GPU. Mapper step time 0.46 to 0.49 s; about 2.6 to 2.7 h per 20000-step run.
- F1 and local analyses: GTX 1650, 4 GB.
- Every run writes a directory with configuration, git commit, uncommitted-changes flag, seed, software versions, device and metrics. `docs/results.md` is generated from these directories.
