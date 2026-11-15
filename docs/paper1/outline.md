# Paper 1: outline

Authors: Arunav Ganguly, Aman Dhingra, Sourabh Singh Verma.

Scope: phase 1 bridges S1, Q1 and F1 with a VAE generator or retrieval of real paintings. S0 (the 2025 cycle-consistency idea) is tested as the `s0` and `s1_cycle` variants of S1. S2 (blind matching) is addressed by F1 and its direct-assignment experiment. Claims are limited to the transfer of distance structure between representations; no claim about perceived or felt correspondence is made.

Evidence pointers name the run-log entry by its heading in `docs/run_log.md`. Tables are in `results.md`; findings are numbered P (positive), N (negative), O (open) as in the findings list.

## Candidate titles

1. Transferring distance structure from music to abstract images without text or paired data
2. What distance matching alone determines: music-to-abstract-image mapping with frozen self-supervised encoders
3. Structure transfer from music to abstract art: generated images, retrieved paintings and an unidentified orientation

## Abstract draft (188 words)

Music-to-image mapping is studied under three constraints: no text-supervised model, no paired data and no human labels in training. Tracks of FMA-small and 5832 abstract paintings from WikiArt are encoded by frozen self-supervised encoders (MERT-v1-95M, DINOv2 ViT-S/14). A mapping network is trained to make output distances correlate with audio-code distances, while a distribution term matches the outputs to the art collection. Three bridges are compared: a spatial VAE latent (S1), a vector VAE with a feature-space distribution term (Q1), and image-feature space with nearest-painting retrieval (F1). On 789 held-out tracks, Q1 on the mean MERT code reaches Mantel r 0.815 to 0.825 over three seeds; permuting the audio distances lowers it to 0.209. The audio code changes the structure score more than the method does, and most of the structure is specific to the DINOv2 feature space. Structure and closeness to the art distribution trade off along one curve; retrieval is limited by the sparsity of a finite collection. The leading audio axis, an acoustic-to-electronic contrast, is mapped to the leading art axis in every seed, but its orientation is not identified. No human evaluation has been made.

Supporting entries: "2026-11-10 — feature extraction"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; "2026-11-13 — S1 and Q1 on the standardised audio code"; "2026-11-12 — exploratory analysis of the seed-0 outputs (local)"; "2026-11-14 — F1: what limits the painting output (local, exploratory)"; "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)"; "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)".

Note on the abstract: the shuffled control (0.209) is one seed; the DINOv2-specificity and retrieval-limit statements rest on exploratory analyses (one seed, scripts not in the repository) and must be qualified or rerun before submission (see `missing.md`).

## 1. Introduction

Claims:

1. A mapping from music to images can be trained without text, pairs or labels if the only constraint is that relative distances between tracks are reproduced between images. (Problem statement; `methods/s1/README.md`, section 1.)
2. The structure term transfers the distance structure of the audio code to outputs on held-out tracks, and the controls without true audio distances do not. Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; "2026-11-13 — F1 variants and controls, seed 0 (local)".
3. The audio code determines the structure score; the bridge determines the distance to the art distribution. Evidence: "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — S1 and Q1 on the standardised audio code".
4. Removing the generator (F1) locates the loss of structure at the step that places outputs on real paintings; this loss follows from the sparsity of a finite collection. Evidence: "2026-11-13 — F1 diagnosis (local)"; "2026-11-14 — F1: what limits the painting output (local, exploratory)".
5. Distance matching identifies the coarse correspondence of leading axes but not its orientation or finer detail. Evidence: "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)".
6. Negative results are reported in full: cycle consistency, latent distribution matching, a generator gate and a label-free code selection criterion each fail in a specific, documented way. Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; "2026-11-10 — Q1 KL-weight sweep"; "2026-11-12 — Q1 seed 0 on two VAEs, test split".

## 2. Related work

All citations: to verify (identifier, title, authors, venue, and that the stated role matches the content).

| Citation | Role in this paper |
|---|---|
| DistanceGAN, arXiv 1706.00826 (to verify) | unpaired one-sided domain mapping that preserves pairwise distances; closest precedent for the structure term |
| GW GAN, "Learning Generative Models across Incomparable Spaces", arXiv 1905.05461 (to verify) | generative modelling across spaces with no common metric via Gromov-Wasserstein; relational objective between incomparable spaces |
| Neural Gromov-Monge, arXiv 2407.14957 (to verify) | neural maps that preserve intra-space distances between incomparable spaces; theory of distance-preserving maps and their non-uniqueness |
| vec2vec, "Harnessing the Universal Geometry of Embeddings", arXiv 2505.12540 (to verify) | unsupervised translation between embedding spaces without paired data; evidence that geometry alone can align spaces in some settings |
| "It's a (Blind) Match!", arXiv 2503.24129 (to verify) | matching vision and language embeddings without parallel data; precedent for bridge S2 and the direct-assignment experiment |
| "Back into Plato's Cave", arXiv 2604.18572 (to verify) | limits of cross-modal representation convergence; context for whether audio and image representations share alignable structure |
| SynVAE, "Translating Visual Art into Music", arXiv 1909.01218 (to verify) | art-music translation without paired data, in the reverse direction; closest application precedent |
| SonicDiffusion, arXiv 2405.00878 (to verify) | audio-conditioned image generation with a pretrained diffusion model; contrast: relies on paired or text-aligned components |
| Seeing Sound, arXiv 2501.05413 (to verify) | audio-to-image generation; contrast in supervision |
| SeeingSounds, arXiv 2510.11738 (to verify) | audio-to-image generation; contrast in supervision |
| RCG, arXiv 2312.03701 (to verify) | image generation conditioned on self-supervised representations; motivates a generator conditioned on F1 feature vectors (future work) |
| RAE, arXiv 2510.11690 (to verify) | generation from representation-encoder latents; same motivation as RCG |
| StyleGAN2-ADA, arXiv 2006.06676 (to verify) | GAN training on small datasets; planned generator for phase 2; context for the weak VAE generator |
| MERT (to verify) | frozen self-supervised music encoder used for all audio codes |
| DINOv2 (to verify) | frozen self-supervised image encoder used in the losses and for evaluation |
| FMA dataset (to verify) | music data (FMA-small) and metadata used for interpretation only |
| WikiArt (to verify) | source of the abstract paintings (four styles) |

Claims:

1. Distance-preserving unpaired mappings exist for image-to-image and embedding-to-embedding settings; the music-to-abstract-image setting with frozen self-supervised encoders and no text has not, to the authors' knowledge, been tested with controls of this kind. (Literature claim; requires the citation check.)
2. Audio-to-image generators in the cited work use paired audio-visual data or text-aligned models; this work excludes both. (Literature claim; requires the citation check.)

## 3. Problem setting and constraints

Claims:

1. Constraints: no text-supervised encoder or pipeline, no pairs, no human labels in training, no hand-written sound-image correspondences; metadata used only for interpretation. Source: `README.md`; `docs/roadmap.md` (Excluded).
2. Formal setting: two unrelated collections, encoders fixed, a mapping M trained from a structure term and a distribution term. Source: method READMEs.
3. The objective depends only on distances and is invariant to isometries of the output space, so it can at most identify the mapping up to such symmetries. Evidence for the practical consequence: "2026-11-13 — F1 orientation across ten seeds (local)".

## 4. Data and encoders

Claims:

1. 7994 FMA-small tracks embedded (6 defective files skipped), middle 30 s in three 10 s windows; 5832 paintings from four WikiArt styles at 128 px; no de-duplication. Evidence: "2026-11-10 — feature extraction".
2. Hash-based 80 / 10 / 10 splits: 6408 / 797 / 789 tracks, 4666 / 571 / 595 images. Evidence: "2026-11-10 — S1 VAE and mapper timing".
3. Three audio codes with different geometry (participation ratio 25.7, 38.8, 104.7; CV of distances 0.336, 0.152, 0.090); all three retrieve the same track's other windows at 0.914 to 0.957 (chance 0.0013). Evidence: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".
4. DINOv2 features of the art carry usable structure (augmented-view retrieval 0.908, chance 0.0017; participation ratio 44.8). Evidence: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing".

## 5. Methods: S1, Q1, F1

Claims (descriptive; full text in `methods.md`):

1. S1: MLP into a 4 x 16 x 16 VAE latent; latent SWD plus structure term; cycle variants test S0. Source: `methods/s1/README.md`; lambda selection: "2026-11-10 — S1 lambda sweep".
2. Q1: MLP into a 128-d VAE latent; feature-space and latent SWD, global and local structure terms; sampleability gate. Source: `methods/q1/README.md`; gate: "2026-11-10 — Q1 KL-weight sweep".
3. F1: MLP into DINOv2 feature space, nearest-painting retrieval; staged diagnosis; grid of output heads, distribution terms and structure terms. Source: `methods/f1/README.md`; "2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)"; "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)".
4. Controls in every bridge: no structure term, shuffled audio distances, untrained mapping; non-learned baselines PCA alignment and retrieval. Source: method READMEs.

## 6. Evaluation protocol

Claims:

1. Metrics: Mantel r and rho with 9999 permutations, kNN overlap (k = 10, chance 0.0127), trustworthiness, FD on DINOv2 features against 595 test images (training art 113.3), participation ratio, distinct paintings, pixel-based structure, orientation statistics. Source: method READMEs; `methods.md`, section 7.
2. Final-step checkpoints; no checkpoint selection; test data used only for evaluation in S1, Q1 and F1 variants, with the exceptions listed in `deviations_and_limitations.md` (Q1 diagnostics on the test split; F1 sweep selection on test tracks).
3. FD reference values: S1 VAE reconstructions 163.0; Q1 VAE reconstructions 208.5 (0.02) and 576.8 (0.3). Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; "2026-11-12 — Q1 seed 0 on two VAEs, test split".

## 7. Results

### 7.1 Structure transfer against controls

1. S1 (mean code, one seed): Mantel r 0.861 against `s0` 0.083, `shuffled` 0.140, random M 0.055. Evidence: "2026-11-12 — S1 seed 0, all variants, test split". Table (a2).
2. Q1 (mean code, three seeds): 0.815 to 0.825; `shuffled` 0.209 (one seed). Evidence: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat". Table (b5).
3. F1 (standardised code, one seed): painting r 0.578 against `f0` 0.303, `shuffled` 0.279, random M 0.066; random M already gives 0.734 on the direct output. Evidence: "2026-11-13 — F1 variants and controls, seed 0 (local)". Table (e).

### 7.2 Distance to the art distribution

1. S1 outputs are far from the art distribution under all variants (FD 1108.7 to 1877.5 against 163.0 for reconstructions). Evidence: "2026-11-12 — S1 seed 0, all variants, test split". Table (a2).
2. The Q1 gate can be passed by degrading reconstructions; the gate-failing VAE gives the better mapper. Evidence: "2026-11-10 — Q1 KL-weight sweep"; "2026-11-12 — Q1 seed 0 on two VAEs, test split". Tables (b2), (b4).
3. The Q1 feature-space term lowers FD (409.0 to 425.4 against 675.2 for `no_feat`) at a cost of 0.05 in Mantel r. Evidence: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat". Table (b5).

### 7.3 The audio code

1. Code, not method, sets the structure score; method, not code, sets FD (one seed per cell). Evidence: "2026-11-13 — S1 and Q1 on the standardised audio code"; "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code". Table (c1).
2. Each method preserves the structure of its training code and little of the other. Evidence: "2026-11-12 — comparison of S1 and Q1 under a common audio code". Table (c2).
3. The label-free D1 criterion selects the code with the least transferable geometry (whitened). Evidence: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing"; "2026-11-13 — F1 diagnosis (local)". Tables (b1), (d2).

### 7.4 Where structure is lost (F1)

1. The mapping generalises (test r within 0.02 of training r). Evidence: "2026-11-13 — F1 diagnosis (local)". Table (d2).
2. Structure is lost at retrieval (test r 0.945 to 0.995 direct, 0.516 to 0.644 painting); a mapping trained on the test tracks does no better. Evidence: "2026-11-13 — F1 diagnosis (local)". Table (d2).
3. Combinations of retrieval inside training and a distribution term raise painting r to 0.747 (0.007) (mean) and 0.641 (0.006) (standardised) over five seeds; no combination exceeds 0.76 with several hundred paintings. Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)". Tables (f2), (f3).

## 8. Analysis

1. One trade-off curve: interpolation from the direct output to the nearest painting passes near the Q1 result (r about 0.80 at FD about 425, mean code). Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)". Table (h1).
2. Collection sparsity: a Gaussian model reproduces the real-gallery result; 10x and 100x more samples add 0.05 and 0.08. The loss comes from the size of the move, not its direction. Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)". Tables (h2), (h3).
3. Most of the structure lies in about ten directions of the mean code. Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)". Table (c3).
4. The leading audio axis is an acoustic-to-electronic timbre and production contrast. Evidence: "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)". Table (j2).
5. It is mapped to the leading art axis in every seed; genres at the electronic end receive mostly Abstract Expressionism in the majority orientation. Evidence: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)"; "2026-11-13 — F1 orientation across ten seeds (local)". Tables (g1), (k1), (k2).
6. Orientation is not identified: majority 71 % (mean code, 130 runs) and 68 % (standardised, 50 runs); D-ease and D-loss do not separate the orientations; it is set by the initial weights. Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)". Tables (g2), (g3).
7. Beyond the first axis, assignments are not reproducible across seeds in F1 or Q1. Evidence: "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat". Tables (g1), (b5).
8. Structure measured in DINOv2 is mostly absent in pixel-based features. Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)"; "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — F1 variants and controls, seed 0 (local)". Tables (i1), (i2).

## 9. Limitations

1. One image encoder for training and evaluation; one audio encoder. Evidence: method READMEs; table (i1).
2. No human evaluation.
3. Many results are single-seed or exploratory; selection on test tracks in the F1 sweep; Q1 gate overridden. See `deviations_and_limitations.md`.
4. Small, stylistically narrow art collection without de-duplication; 30 s music clips; one code per track. Evidence: "2026-11-10 — feature extraction".
5. Orientation and finer detail not identified by the objective. Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)".

## 10. Conclusion

1. Distance matching between frozen self-supervised representations transfers audio structure to images on held-out tracks, against permuted and untrained controls.
2. What is transferred is decided mainly by the audio code and the image feature space; what is realistic is decided by the bridge; the two trade off along one curve.
3. Distance matching fixes the coarse correspondence of leading axes, not its orientation or finer detail. A signal beyond distances is needed (roadmap phases 2 and 2.5).
