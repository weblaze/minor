# Paper 1: findings

One sentence per finding, then the evidence: run-log entry heading and the table in `results.md`. Seed counts and audio codes are stated in each finding. "Exploratory" marks results from scripts not in the repository.

## Positive findings

P1. With the structure term, S1 transfers the distance structure of the mean code to generated images on held-out tracks (Mantel r 0.861, kNN overlap 0.268; one seed), while `s0` (0.083), `shuffled` (0.140) and random M (0.055) do not.
Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; results (a2).

P2. Q1 on the mean code transfers structure stably over three seeds (Mantel r 0.815 to 0.825, kNN overlap 0.282 to 0.290, FD 409.0 to 425.4), and permuting the audio distances removes most of it (`shuffled`, seed 0: 0.209).
Evidence: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; results (b5).

P3. In F1 the structure term adds to what the distribution term and network smoothness give on the painting output (standardised code, one seed: `f1` 0.578 against `f0` 0.303, `shuffled` 0.279, random M 0.066).
Evidence: "2026-11-13 — F1 variants and controls, seed 0 (local)"; results (e).

P4. The feature-space distribution term of Q1 brings generated images closer to the art distribution (mean code: FD 409.0 to 425.4 over three `q1` seeds against 675.2 for `no_feat`, one seed) at a cost of 0.05 in Mantel r and 0.11 in kNN overlap.
Evidence: "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; results (b5).

P5. The distance to the art distribution is set by the method, not the audio code: Q1 gives FD 402.1 to 425.4 and S1 gives 1172.8 to 1364.0 under each of the three codes (one seed per cell).
Evidence: "2026-11-13 — S1 and Q1 on the standardised audio code"; results (c1).

P6. The standardised code gives the highest kNN overlap and trustworthiness of the three codes for both S1 (0.410, 0.969) and Q1 (0.351, 0.955), with Mantel r within 0.03 of the mean code (one seed each).
Evidence: "2026-11-13 — S1 and Q1 on the standardised audio code"; results (c1).

P7. An F1 mapping trained without decoder or feature extractor in the loop generalises: r on test tracks is within 0.02 of r on training tracks for every code tested (one seed).
Evidence: "2026-11-13 — F1 diagnosis (local)"; results (d2).

P8. The first principal axis of the audio code is mapped to the first principal axis of the art features in every seed tested (absolute correlation 0.87 to 0.92 over 10 seeds, mean code, `swd`; mean 0.89 over 50 seeds for each of the mean and standardised codes).
Evidence: "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)"; results (g1), (g2).

P9. The first axis of the mean code is a timbre and production contrast between acoustic and electronic, high-energy material (acousticness rho +0.68, energy -0.51; R^2 0.82 from low-level spectral descriptors; genre eta^2 0.31), not loudness (exploratory, no mapping involved).
Evidence: "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)"; results (j2).

P10. Through F1 retrieval, genres at the electronic end are assigned mostly Abstract Expressionism (Electronic 0.85, Hip-Hop 0.82; mean code, seed 0) and Folk mostly Colour Field and Minimalism (combined share 0.83), with Cramer V 0.285 (mean) and 0.326 (standardised) (exploratory, one seed; orientation varies by seed, see N13).
Evidence: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)"; results (k1), (k2).

P11. Combining retrieval inside training (soft or straight-through head) with a distribution term raises r of the painting output over the baseline by about 0.10 for the mean code (0.641 (0.013) to 0.747 (0.007)) and by 0.06 to 0.085 for the standardised code (0.556 (0.011) to 0.641 (0.006)), over five seeds.
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part C; results (f3).

P12. `soft02__swd1_chamfer01__global3` improves every reported quantity over the baseline for both codes over five seeds (mean code: r 0.737 against 0.641, kNN 0.145 against 0.122, FD 340 against 346; standardised: r 0.614 against 0.556, kNN 0.181 against 0.137, FD 274 against 338, 455 against 394 distinct paintings).
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part C; results (f3).

P13. The generated-image result and the retrieval result lie on one structure-realism trade-off curve: interpolating from the F1 direct output to its nearest painting gives r about 0.80 at FD about 425 (mean code, one seed, as read from the curve in the log), and Q1 on the mean code reached r 0.825 at FD 425.4.
Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)"; results (h1), (b5).

P14. r of the F1 painting output varies little across random initialisations (standard deviation 0.010 to 0.017 over 20 to 50 seeds per setting).
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part A; results (g2).

## Negative findings

N1. Matching latent marginals does not place S1 outputs on the art distribution: FD is 1108.7 to 1877.5 for all S1 variants and controls against 163.0 for VAE reconstructions, and outputs have about half the feature spread of test art (mean code, one seed).
Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; results (a2).

N2. The cycle-consistency term (the 2025 S0 idea) adds no structure: `s0` gives Mantel r 0.083 and `s1_cycle` 0.839 against 0.861 for `s1` (mean code, one seed).
Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; results (a2).

N3. The S1 structure term overfits the 6408 training tracks: for every lambda, validation L_struct is lower at step 5500 than at step 20000 while training L_struct keeps falling (mean code, one seed).
Evidence: "2026-11-10 — S1 lambda sweep"; results (a1).

N4. The Q1 sampleability gate can be passed by degrading reconstructions, and no KL weight brings prior samples within a factor of six of training art (lowest FD(prior) 685.2 against 113.3).
Evidence: "2026-11-10 — Q1 KL-weight sweep"; results (b2).

N5. On the gate-selected VAE (`kl_weight` 0.3) the Q1 mapper does not exceed PCA alignment (Mantel r 0.179 against 0.180; kNN 0.101 against 0.130; whitened code, one seed), and the gate-failing VAE (0.02) gives the better result on every measure.
Evidence: "2026-11-12 — Q1 seed 0 on two VAEs, test split"; results (b4).

N6. The label-free code selection of D1 (same-track window retrieval) selects the whitened code, which gives the lowest structure scores of the three codes (S1 0.272, Q1 0.259, one seed each; F1 painting output 0.066, one seed).
Evidence: "2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing"; "2026-11-13 — S1 and Q1 on the standardised audio code"; "2026-11-13 — F1 diagnosis (local)"; results (b1), (c1), (d2).

N7. The audio code determines the structure score more than the method: both S1 and Q1 reach Mantel r 0.83 to 0.86 on the mean code and 0.26 to 0.27 on the whitened code, and each preserves little of the other code's structure (S1 against whitened distances 0.155; Q1 VAE 0.02 against mean distances 0.160; one seed each).
Evidence: "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-12 — comparison of S1 and Q1 under a common audio code"; results (c1), (c2).

N8. The structure S1 preserves lies in about ten directions of the mean code: removing the first principal axis lowers r with S1 image distances from 0.861 to 0.369, and removing ten lowers it to 0.027 (exploratory, one seed).
Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part B; results (c3).

N9. Most of the structure measured in DINOv2 features is absent in pixel-based features: S1 0.861 against 0.144 to 0.280, Q1 on the mean code 0.825 against 0.023 to 0.307, F1 painting output 0.578 against 0.153 to 0.193 (one seed each; exploratory for S1 and Q1).
Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part A; "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code"; "2026-11-13 — F1 variants and controls, seed 0 (local)"; results (i1), (i2).

N10. An untrained mapping preserves distances on the direct F1 output (random M: Mantel r 0.734, standardised code, one seed), so the direct output is not evidence of learning.
Evidence: "2026-11-13 — F1 variants and controls, seed 0 (local)"; results (e).

N11. No batch-level distribution term tested places the direct F1 outputs on the art distribution (FD 575 to 1205 in the exploratory precursor; 854 to 1197 in the diagnosis; one seed).
Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part C; "2026-11-13 — F1 diagnosis (local)"; results (d1), (d2).

N12. Restricting outputs to real paintings loses most of the structure: r falls from 0.945 to 0.995 (direct, test) to 0.516 to 0.644 (painting) and kNN overlap to 0.115 to 0.154, for the mean, standardised and single-layer codes (one seed).
Evidence: "2026-11-13 — F1 diagnosis (local)"; results (d2).

N13. The orientation of the audio-art correspondence is not identified by the data: the majority sign of axis 1 holds in 71 % of 130 mean-code runs and 68 % of 50 standardised-code runs, the two orientations reach equal losses and are learned at equal speed (D-ease, D-loss), and seed 4 gives the mirror-image genre-style assignment under both codes.
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part A; "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)"; results (g1) to (g3), (k2).

N14. Beyond the first axis the mapping is not reproducible across seeds: axes 2 and 3 show no preferred orientation in most settings, two F1 seeds assign paintings to the same track with mean cosine similarity 0.558 against 0.490 for shuffled tracks (mean code, 10 seeds), and three Q1 seeds give same-track image similarity 0.415 to 0.528 against 0.400 to 0.410 for shuffled tracks while their distance matrices agree at 0.847 to 0.854.
Evidence: "2026-11-13 — F1 orientation across ten seeds (local)"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat"; results (g1), (b5).

N15. The retrieval limit is set by the sparsity of a finite collection and falls slowly with its size: a Gaussian model of the art features reproduces the real-gallery result (r 0.643 against 0.645, mean code) and ten or a hundred times more samples add only 0.05 and 0.08 in r (exploratory, one seed).
Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)"; results (h2).

N16. The loss of structure at retrieval comes from the size of the move to the nearest painting, not its direction: a random displacement of the same length gives r 0.707 (mean code) and 0.555 (standardised) (exploratory, one seed).
Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)"; results (h3).

N17. Neither a direct assignment of tracks to paintings (r 0.545 mean, 0.547 standardised) nor a mapping trained on the test tracks themselves (oracle 0.650 mean, 0.571 standardised) exceeds map-then-retrieve by a meaningful margin (one seed; the direct assignment is exploratory and not an upper bound).
Evidence: "2026-11-13 — F1 diagnosis (local)"; results (d2), (d3).

N18. Restricting the measurement to the leading axes of the art features does not raise r of the painting output above about 0.65; uncontrolled variation in the remaining axes is not the ceiling (exploratory, one seed).
Evidence: "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)", part A; results (h4).

N19. Without a distribution term F1 retrieval loses the structure (painting r 0.063 to 0.354), the adversarial critic lowers r without lowering FD in 4000 steps, and chamfer terms or straight-through retrieval without a distribution term collapse to 1 to 63 paintings (one seed).
Evidence: "2026-11-13 — F1 experiment grid: placing outputs on the art collection (local)"; results (f1).

N20. No F1 combination exceeds painting r 0.76 with several hundred distinct paintings, and the highest-r combination for the mean code has the lowest kNN overlap (0.079 (0.003), five seeds): global structure improves while neighbourhoods degrade.
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", parts B and C; results (f3).

## Open questions

O1. What sets the majority orientation of axis 1? The log shows it is set by the initial weights (same seed, same side under both codes in 44 of 50 cases) and that axis 3 of the mean code, the most skewed audio and art axis, has a stable orientation (92 %); a causal account is not established.
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)", part A.

O2. Does the structure transfer hold under an image encoder not used in training? Only DINOv2 and four pixel-based feature sets have been used.
Evidence: results (i1), (i2); `deviations_and_limitations.md`.

O3. Do human observers perceive any correspondence between a track and its image? No human evaluation has been made.

O4. Do the S1 results hold over three seeds, and does Q1 on the mean code exceed PCA alignment and random M on that code (H1 of both protocols)? S1 seeds 1 and 2 and the Q1 mean-code baselines and random M are not run.
Evidence: "2026-11-12 — S1 seed 0, all variants, test split"; "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat".

O5. What does the Q1 local structure term contribute (H3)? `no_local` and `no_global` are not run.

O6. Which label-free criterion predicts whether a code's geometry can be represented in the image feature space? Same-track retrieval (D1) does not; the whitened code has near-uniform distances (CV 0.090) and a participation ratio (104.7) above that of art features (44.8).
Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part C observations.

O7. What do the leading directions of the mean code beyond axis 1 encode? Axes 2 to 6 have genre eta^2 0.09 to 0.19 and descriptor R^2 0.45 to 0.65; the log states "What these directions encode is not established".
Evidence: "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", part B; "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)".

O8. Does a larger or denser real art collection raise painting r as the Gaussian model predicts? Only Gaussian galleries have been enlarged.
Evidence: "2026-11-14 — F1: what limits the painting output (local, exploratory)".

O9. Do the F1 combination gains survive selection on held-out data? The sweep and the rerun selection scored test tracks.
Evidence: "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)"; `deviations_and_limitations.md`.

O10. Would an image generator conditioned on F1 feature vectors reach a better point on the structure-realism curve than Q1? Not implemented.
Evidence: `methods/f1/README.md` (section 1).
