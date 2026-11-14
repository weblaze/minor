# Paper 1: missing before submission

## 1. Runs not made

| Item | Needed for | Source of the requirement |
|---|---|---|
| S1 `s1`, `s0`, `s1_cycle`, `shuffled`, random M at seeds 1 and 2 (mean code) | S1 H1 ("for each of three seeds") | `methods/s1/README.md`, section 4; "2026-11-12 — S1 seed 0, all variants, test split" |
| Q1 random M on the mean code | Q1 H1 | `methods/q1/README.md`, section 6 |
| Q1 `pca_alignment` and `retrieval` baselines on the mean code | Q1 H1; comparison with the mean-code `q1` runs | baselines exist only for the standardised and whitened codes |
| Q1 `shuffled` and `no_feat` at seeds 1 and 2 | three-seed controls | Q1 H1 |
| Q1 `no_local`, `no_global` | Q1 H3; contribution of each structure term | "2026-11-12 — Q1 seed 0 on two VAEs, test split" |
| Q1 on the gate-passing VAE (0.3) with the mean code | whether the gate override changes the mean-code conclusion | A1 in `deviations_and_limitations.md` |
| S1 and Q1 on the standardised code at more than one seed | code comparison (results (c1)) | one seed per cell |
| F1 variants and controls at more than one seed; F1 controls on the mean code | P3 in the findings list | "2026-11-13 — F1 variants and controls, seed 0 (local)" |
| F1 combination selection on the validation split, then evaluation on test | removes selection on test data | A4 in `deviations_and_limitations.md` |
| Controls (`f0`, `shuffled`) for the best combinations | whether the combination gains need the true audio distances | "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)" |
| Evaluation with a second image encoder (text-free) | independence of the structure measurement | roadmap: E-2nd-image, scheduled after all methods |
| Human evaluation | any perceptual claim | roadmap, phase 1 step 5 |

## 2. Analyses to move into the repository

The following were run from scripts not in the repository. Their scripts must be committed (or rewritten and rerun) so that the numbers are reproducible:

- "2026-11-12 — exploratory analysis of the seed-0 outputs (local)", parts A, B, C.
- "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)".
- Pixel-feature structure of the Q1 outputs in "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code".
- Direct assignment in "2026-11-13 — F1 diagnosis (local)".
- "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)".
- "2026-11-14 — F1: what limits the painting output (local, exploratory)".
- Seed agreement and axis-1 correlations in "2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat" (possibly `methods/q1/compare_seeds.py` or `scripts/results.py agreement`; not stated).
- "2026-11-12 — comparison of S1 and Q1 under a common audio code" (script not named).

## 3. Numbers not available in the log

- Kernel distances of the evaluated S1, Q1 and F1 runs (computed by the evaluation scripts according to the method READMEs; not in the log).
- Mantel rho of the Q1 baselines on the whitened code (blank in the log).
- Mantel p-values of the Q1 runs and of the F1 grid and sweep (the log gives p only for S1 seed 0, the D5 baselines and the F1 variants).
- F1 `gallery_1000` stage (configured; not in the log).
- Per-seed values of the five-seed reruns and of the orientation votes (only means, standard deviations and counts are logged); available in the run directories.
- The mapping from the three Q1 axis-1 correlations (-0.43, -0.85, -0.91) to seeds is implied by order only.
- Full rows of the 100-combination sweep (in `f1_realism_sweep_*` directories, not in the log).
- Standardised-code and other-network-size values of the majority/minority comparison (the log states only that differences are "of the same size or smaller").
- Confidence intervals or seed standard deviations for every single-seed result.
- Number of near-duplicate paintings between training and test art (no de-duplication was performed).
- MERT model revision (commit hash) used for extraction.
- Wall-clock and GPU-hour totals for the paper's compute statement (partly available from the per-entry durations).

## 4. Inconsistencies to resolve

- S1 `s0` Mantel r: 0.083 in "2026-11-12 — S1 seed 0, all variants, test split"; 0.082 in "2026-11-12 — comparison of S1 and Q1 under a common audio code" and in `docs/results.md`. One of the two is a rounding or transcription difference; check `results.json` of `s1_eval_s0_seed0_20261112T062319Z`.
- F1 commit: `docs/results.md` gives `4234cad` for `f1_eval_f1_seed0_20261112T213235Z` and `f1_eval_f0_seed0_20261112T213913Z`; the run log gives `aa6783c`.
- The same nominal F1 configuration (standardised code, SWD, `lambda_struct` 1, seed 0) gives painting r 0.572 (diagnosis), 0.573 (grid `swd`) and 0.578 (`train_mapper.py` variant `f1`); the mean code gives 0.644 (diagnosis), 0.645 (grid) and 0.637 (precursor, 2026-11-12). The differences come from separate runs of different scripts; the paper must state which number is the reference.
- kNN overlap of the same mean-code `swd` painting output: 0.131 (grid) and 0.132 (interpolation table, t = 1); direct output 0.698 (precursor) and 0.696 (interpolation, t = 0).
- Variance fraction of axis 1 of the mean code: 0.117 (axes fitted on training windows, "2026-11-12 — exploratory analysis of the seed-0 outputs (local)") and 0.130 (axes fitted on window-averaged training tracks, "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)"). The paper must use one definition.
- Dates: run directory timestamps (UTC) and run-log dates differ by one day for some F1 runs (e.g. `f1_diagnose_seed0_20261112T211340Z` under the entry dated 2026-11-13).
- `README.md` on `base` states "All results are from a single seed"; this is no longer true after the 2026-11-14 entries. Not edited here (outside scope).

## 5. Citations to verify

Every citation in `outline.md` is marked "to verify". For each: confirm the arXiv identifier, title, authors, year and published venue, and that the role assigned in the outline matches the paper's content. In particular:

- "Back into Plato's Cave" (arXiv 2604.18572): confirm identifier and content.
- vec2vec (arXiv 2505.12540), "It's a (Blind) Match!" (arXiv 2503.24129), RAE (arXiv 2510.11690), SeeingSounds (arXiv 2510.11738): confirm titles and venues.
- MERT, DINOv2, FMA, WikiArt: identifiers are not given in the sources; obtain the canonical references (MERT: Li et al.; DINOv2: Oquab et al.; FMA: Defferrard et al.; WikiArt: dataset as distributed on Kaggle as `steubk/wikiart` plus the original WikiArt source). Confirm licences of FMA-small and of the WikiArt images.
- Trustworthiness: Venna and Kaski (2001), as cited in `methods/s1/README.md`.
- Mantel test, sliced Wasserstein distance, energy distance, Frechet distance on DINOv2 features, participation ratio, Cramer V: standard references to be chosen.

## 6. Text and presentation

- A statement of the hypotheses as pre-registered in the method READMEs, and which were tested, partly tested or not tested.
- A single reference table of FD reference values (training art 113.3; reconstructions 163.0, 208.5, 576.8).
- Captions and data for the figures listed in `figures.md` that are marked "log only".
- Licence of the code (none granted yet, per `README.md`) and a data availability statement.
