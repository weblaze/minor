# Paper 1: protocol deviations and limitations

Every item names its run-log entry by heading. Items marked "from code" are read from the scripts on the method branches, not from the run log.

## A. Protocol deviations

### A1. Q1 gate overridden

- The gate rule (fixed before any run: pass if FD(prior) <= 2.0 x FD(reconstructions)) selects `kl_weight` 0.3. The VAE with `kl_weight` 0.02 failed the gate (ratio 4.68) and was used with `--ignore-gate`. "2026-11-10 — Q1 KL-weight sweep"; "2026-11-12 — Q1 seed 0 on two VAEs, test split".
- Every Q1 run after the VAE-pair run uses the gate-failing VAE: code swap (mean code), standardised code, mean-code seeds 1 and 2, `shuffled`, `no_feat`. "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — S1 and Q1 on the standardised audio code"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".
- Reason recorded: the ratio criterion can be met by degrading reconstructions. "2026-11-10 — Q1 KL-weight sweep".

### A2. Q1 audio code changed after D1, and later overridden

- D1 selected the whitened code. The first Q1 job was configured with the standardised code; its baselines (D5) use the standardised code, the VAE and gate do not depend on the code. "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".
- Q1 was later run on the mean and standardised codes, against the D1 selection, after the comparison with S1 showed that the code determines the structure score. These runs are post hoc. "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — S1 and Q1 on the standardised audio code".
- D1 to D3 and the gate read the test split for measurement; the choice of code and of `kl_weight` therefore depends on test data (stated in `methods/q1/README.md`, section 7).

### A3. Selection rules stated after seeing results

- S1 lambda: "the largest lambda whose validation L_dist is within 10 % of the lowest validation L_dist in the sweep" was stated after seeing the sweep values. It selects the configured value 1.0. "2026-11-10 — S1 lambda sweep".
- F1 grid observations ("best `soft_0.02_swd` ... ") are post hoc readings of one seed. "2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)".

### A4. Model selection on the test split (F1)

- The F1 grid and the combination sweep score every configuration on the 789 test tracks. The rerun selection (top five by r of the painting output, at least 300 distinct paintings, FD at most 450) uses these test scores; the reruns (seeds 1 to 4) are evaluated on the same test tracks. The rule was fixed in `phase1_batch.py` before the sweep ran, but it selects on test data. "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)"; from code: `methods/f1/realism.py`, `methods/f1/phase1_batch.py`.
- The orientation runs report validation-split distribution terms; r of the painting output is on test tracks. "2026-11-13 — F1 orientation across ten seeds (local)".

### A5. Runs from uncommitted or later-committed code

- F1 diagnosis: run before commit `aa6783c`, working tree not clean. "2026-11-13 — F1 diagnosis (local)".
- F1 grid: run on a working tree with `realism.py` uncommitted; the grid was committed after the run. "2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)".
- F1 ten-seed orientation: `orientation.py` committed after the run. "2026-11-13 — F1 orientation across ten seeds (local)".
- `docs/results.md` lists commit `4234cad` for the F1 `f1` and `f0` evaluation runs; the run log names `aa6783c` for the same entry. "2026-11-13 — F1 variants and controls, seed 0 (local)". To be reconciled (see `missing.md`).

### A6. Exploratory analyses from scripts not in the repository

| Entry | Content |
|---|---|
| "2026-11-12 — exploratory analysis of the seed-0 outputs (local)" | A: pixel-feature structure of S1 and Q1; B: principal-axis removal; C: direct mapping to DINOv2 features with nearest-painting replacement |
| "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)" | genre and artist neighbour rates, axis interpretation |
| "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code" | pixel-feature structure of the Q1 outputs (local, exploratory) |
| "2026-11-13 — F1 diagnosis (local)" | direct assignment |
| "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)" | A, B, C |
| "2026-11-14 — F1: what limits the painting output (local, exploratory)" | A to E |
| "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat" | seed agreement and axis-1 correlations ("local analysis of the `features.npz` files"; script not named) |
| "2026-11-12 — comparison of S1 and Q1 under a common audio code" | post hoc, "not part of either protocol"; script not named |

### A7. Single-seed results

- S1: every evaluated S1 run is seed 0. Seeds 1 and 2 required by H1 are not run. "2026-11-12 — S1 seed 0, all variants, test split".
- Q1: only `q1` on the mean code has three seeds. `q1` on the whitened and standardised codes, `shuffled` and `no_feat` are seed 0 only. "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".
- F1: diagnosis, variants and controls, the 27-configuration grid, the 100-combination sweep and all exploratory analyses are seed 0. Multi-seed: reruns (5 seeds), orientation (10 to 50 seeds), genre-style table (5 seeds).
- The code comparison across S1 and Q1 (results (c1), (c2)) is one seed per cell.

### A8. Incomplete protocols

- S1 H1 requires three seeds; one is run. S1 H2 ("not substantially larger") has no numeric threshold; `s1` FD 1172.8 against `s0` 1108.7. "2026-11-12 — S1 seed 0, all variants, test split".
- Q1 H1 requires `q1` to exceed `shuffled`, random M and `pca_alignment` for three seeds; on the mean code only `shuffled` (seed 0) is run. Q1 H3 (`no_local`) is not tested. Q1 H4 is addressed only by the seed-agreement analysis. "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".
- Q1 H2, second clause, is tested on the whitened code only; first clause on the mean code only. "2026-11-12 — Q1 seed 0 on two VAEs, test split"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".
- The S1 genre-leakage probe, DEAM analysis, descriptor probes and human study listed in the S1 and Q1 plans are not run (from `methods/s1/README.md` section 7 and `methods/q1/README.md` section 9).

### A9. Other recorded departures

- Hash-based splits replace the official FMA split (`methods/s1/README.md`, section 7).
- Mixed precision disabled for the mapping stage (`methods/s1/README.md`, section 7).
- The protocol evaluates the final step although the S1 structure term overfits after step 5500; this is not changed. "2026-11-10 — S1 lambda sweep".
- S1 runs on the whitened and standardised codes were made from second clones at `7a17f65` and `d3b327b`, inside Q1 jobs. "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — S1 and Q1 on the standardised audio code".

## B. Limitations

### B1. Measurement

- One image encoder (DINOv2 ViT-S/14) is used both in the training losses and in evaluation. Agreement in this space is not independent evidence. The pixel-based check shows that most of the structure is specific to it. "2026-11-12 — exploratory analysis of the seed-0 outputs (local)"; results (i1), (i2).
- One audio encoder (MERT-v1-95M); its revision is not pinned (from `configs/base.yaml`).
- Structure is measured against the code each mapping was trained on; scores under different codes are not directly comparable. "2026-11-12 — comparison of S1 and Q1 under a common audio code".
- FD is computed against 595 test images in 384 dimensions and is biased upward for small samples (from `abstraction/metrics/distribution.py`). Kernel distances are reported for the gate only.
- The Mantel p-value floor is 0.0001 (9999 permutations); the p-values do not separate trained variants. "2026-11-12 — S1 seed 0, all variants, test split".
- No human evaluation has been made. No claim about perceptual or affective correspondence is supported.

### B2. Data

- 7994 tracks (FMA-small, 8 genres, 30 s clips; middle 30 s used) and 5832 paintings from four WikiArt styles; one style (Action_painting) has 98 images. "2026-11-10 — feature extraction".
- Images at 128 px, resized without preserving aspect ratio; DINOv2 applied at 112 px.
- No de-duplication of the art collection; near-duplicates between training and test art are possible and would affect retrieval and density measurements. "2026-11-10 — feature extraction"; "2026-11-14 — F1: what limits the painting output (local, exploratory)".
- The gallery for retrieval is the 4666 training paintings; F1 outputs cannot leave this collection.

### B3. Method

- The objective depends on distances only and is invariant to reflections of the mapping; the orientation of the correspondence is not identified. "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)".
- The art features are close to symmetric along their leading axes (skewness -0.06, -0.03), which gives a distribution term no preference between an orientation and its mirror image. "2026-11-13 — F1 orientation across ten seeds (local)".
- Assignments of individual images are not reproducible between seeds. "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".
- The VAE generators produce texture without spatial composition (S1 sample grids) and no tested KL weight gives prior samples near the art distribution. "2026-11-12 — S1 seed 0, all variants, test split"; "2026-11-10 — Q1 KL-weight sweep".
- Mapping networks are deterministic: one image per track.
- One code per track (mean of three windows); temporal structure within a track is not used.
- The PCA-alignment baseline matches axes by variance rank with arbitrary signs; it is one of 2^32 sign choices (`methods/q1/README.md`, section 10).
- The F1 oracle stage is a reference, not a bound; the direct-assignment optimiser is not a proof of an upper bound. "2026-11-13 — F1 diagnosis (local)".
- Loss weights other than S1 lambda and the Q1 KL weight were not tuned; the Q1 weights are initial values (`methods/q1/README.md`, section 10).

### B4. Compute

- S1 and Q1 runs on free Kaggle T4 sessions (about 2.6 to 2.7 h per 20000-step run); F1 and local analyses on a 4 GB GTX 1650. This limited the number of seeds for S1 and Q1. "2026-11-10 — S1 lambda sweep"; "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".
