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
