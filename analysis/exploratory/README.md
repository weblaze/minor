# Exploratory analysis scripts

Scripts that produced the entries of `docs/run_log.md` marked "exploratory" or "script not in the repository" at the time of writing. They are kept as run, for traceability. They are not part of any method protocol.

Conditions of use:

- Paths are those of the machine they were run on (repository checkout, fetched Kaggle output under `runs/kaggle/`, scratch directories). They must be edited before the scripts run elsewhere.
- Several scripts import method code that lives on the method branches (`methods/q1` on `q1-structure`, `methods/f1` on `f1-feature`) or copies of it; `spatial_vae.py` is the S1 model file from `s1-structure`.
- FMA metadata (`fma_metadata.zip`: `tracks.csv`, `features.csv`, `echonest.csv`) is used by `meta.py`, `subspace.py` and `seeds_style.py` for interpretation only. No metadata enters training.

| Script | Run-log entry |
|---|---|
| `prep.py` | data for "2026-11-12 — exploratory analysis of the seed-0 outputs (local)": regenerates S1 and Q1 test images, DINOv2 features of the art collection |
| `analyse.py` | same entry, parts A (pixel-based features), B (principal axes of the mean code), C (mapping into feature space, sliced Wasserstein) |
| `feature_space_terms.py` | same entry, part C: energy distance, minibatch optimal transport, standardised code |
| `meta.py` | "2026-11-12 — audio-code axes against FMA metadata (local, evaluation only)" |
| `swapcheck.py` | pixel-feature structure of the Q1 outputs in "2026-11-12 — code swap: S1 on the whitened code, Q1 on the mean code" |
| `assign.py` | direct assignment in "2026-11-13 — F1 diagnosis (local)" |
| `subspace.py` | "2026-11-13 — F1 follow-up: art subspace, genre against painting style, mapper seeds (local, exploratory)", parts A and B |
| `seeds_style.py` | same entry, part C (five mapper seeds) |
| `ceiling.py` | "2026-11-14 — F1: what limits the painting output (local, exploratory)", parts A to D; part E was computed inline from the cached art features |
| `night.py` | aggregation for "2026-11-14 — F1 batch at the end of phase 1: orientation votes, combination sweep, reruns (local)" |
| `check_white.py`, `check_std.py` | checks that the S1 whitened and standardised codes equal the shared implementation |

Not preserved as files (computed inline in the session): the comparison of S1 and Q1 under a common audio code (2026-11-12; now reproduced by `scripts/results.py agreement`), part E of the retrieval-limit entry, and the seed agreement and axis-1 correlations of the three Q1 seeds (2026-11-14).
