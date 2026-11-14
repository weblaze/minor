# Reproducibility index

Generated from docs/run_log.md. One row per log entry.

| Entry | Job or script | Kernel | Commit | Hardware | Result |
|---|---|---|---|---|---|
| 2026-11-10 — feature extraction | `scripts/kaggle/jobs/features.yaml` | `arunavganguly/abstraction-features`, version 2 | `7f190ac` (`base`) | Tesla T4, 15360 MiB | completed |
| 2026-11-10 — S1 VAE and mapper timing | `scripts/kaggle/jobs/s1_vae.yaml` | `arunavganguly/abstraction-s1-vae`, version 1 | `1c4709e` (`s1-structure`) | 2 x Tesla T4, 15360 MiB; one used | completed |
| 2026-11-10 — Q1 diagnostics, VAE, gate, baselines, mapper timing | `scripts/kaggle/jobs/q1_vae.yaml` | `arunavganguly/abstraction-q1-vae`, version 1 | `43a2389` (`q1-structure`) | 2 x Tesla T4, 15360 MiB; one used | completed; gate failed |
| 2026-11-10 — Q1 KL-weight sweep | `scripts/kaggle/jobs/q1_kl_sweep.yaml` | `arunavganguly/abstraction-q1-kl-sweep`, version 1 | `7c5da53` (`q1-structure`) | 2 x Tesla T4, one chain per GPU | completed; duration 7605 s |
| 2026-11-10 — S1 lambda sweep | `scripts/kaggle/jobs/s1_lambda.yaml` | `arunavganguly/abstraction-s1-lambda`, version 1 | `e040aba` (`s1-structure`) | 2 x Tesla T4, one chain per GPU | completed; duration 18953 s |
| 2026-11-12 — S1 seed 0, all variants, test split | `scripts/kaggle/jobs/s1_seed0.yaml` | `arunavganguly/abstraction-s1-seed0`, version 1 | `8feff77` (`s1-structure`) | 2 x Tesla T4, one chain per GPU | completed; duration 12163 s |
| 2026-11-12 — Q1 seed 0 on two VAEs, test split | `scripts/kaggle/jobs/q1_seed0_vae_pair.yaml` | `arunavganguly/abstraction-q1-seed0-vae-pair`, version 1 | `eaffe89` (`q1-structure`) | 2 x Tesla T4, one chain per GPU | completed; duration 10910 s |
| 2026-11-12 — comparison of S1 and Q1 under a common audio code | n/a | local | n/a | n/a | n/a |
| 2026-11-12 — exploratory analysis of the seed-0 outputs (local) | local analysis, script not in the repository | local | n/a | n/a | n/a |
| 2026-11-12 — audio-code axes against FMA metadata (local, evaluation only) | local analysis, script not in the repository | local | n/a | n/a | n/a |
| 2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code | `scripts/kaggle/jobs/code_swap.yaml` | `arunavganguly/abstraction-code-swap`, version 1 | `76b055e` (`q1-structure`); S1 from a second clone at `7a17f65` (`s1-structure`) | 2 x Tesla T4, one method per GPU | completed; duration 9407 s |
| 2026-11-13 — F1 diagnosis (local) | `methods/f1/diagnose.py --seed 0` | local | `aa6783c` (`f1-feature`); run before the commit, working tree not clean | GTX 1650, 4 GB | completed |
| 2026-11-13 — F1 variants and controls, seed 0 (local) | `methods/f1/train_mapper.py`, `methods/f1/evaluate.py` | local | `aa6783c` (`f1-feature`) | GTX 1650, 4 GB | completed |
| 2026-11-13 — F1 experiment grid: placing outputs on the art collection (local) | `methods/f1/realism.py --seed 0 --code <mean, standardised>` | local | grid as committed in `f1-feature` after the run; run made on a working tree with `realism.py` uncommitted | GTX 1650, 4 GB | completed; 27 configurations per code |
| 2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory) | local analysis, script not in the repository | local | n/a | n/a | n/a |
| 2026-11-13 — S1 and Q1 on the standardised audio code | `scripts/kaggle/jobs/std_code.yaml` | `arunavganguly/abstraction-std-code`, version 1 | `6bb84a7` (`q1-structure`); S1 from a second clone at `d3b327b` (`s1-structure`) | 2 x Tesla T4, one method per GPU | completed; duration 11089 s |
| 2026-11-13 — F1 orientation across ten seeds (local) | `methods/f1/orientation.py --seeds 10` | local | `f1-feature`, script committed after the run | GTX 1650, 4 GB | completed; 50 mappings |
| 2026-11-14 — F1: what limits the painting output (local, exploratory) | local analysis, script not in the repository | local | n/a | n/a | n/a |
| 2026-11-14 — Q1 on the mean code: seeds 1 and 2, variants shuffled and no_feat | `scripts/kaggle/jobs/q1_mean_seeds12.yaml`, `scripts/kaggle/jobs/q1_mean_controls.yaml` | `arunavganguly/abstraction-q1-mean-seeds12`, `arunavganguly/abstraction-q1-mean-controls`, version 1 each | `648394a` (`q1-structure`) | 2 x Tesla T4 per job, one run per GPU | completed; durations 10395 s and 10338 s |
| 2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local) | `methods/f1/phase1_batch.py` | local | `4b57977` (`f1-feature`) | GTX 1650, 4 GB | completed; 34 units, no failure, 7 h 2 min |

## Entries without a commit

- 2026-11-12 — comparison of S1 and Q1 under a common audio code
- 2026-11-12 — exploratory analysis of the seed-0 outputs (local)
- 2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)
- 2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)
- 2026-11-14 — F1: what limits the painting output (local, exploratory)
