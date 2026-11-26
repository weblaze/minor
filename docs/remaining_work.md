# Abstraction: everything left to do (as of 2026-11-26)

Collected from `docs/roadmap.md` (order of 2026-11-24), `docs/triage.md`, `docs/combinations.md`, `docs/generator_options.md`, the paper 1 gap list, `docs/learnability_test.md`, the run log, the project owner's decisions and the vault task list. Items marked *(proposal)* were suggested by Claude and are not yet decided by the owner.

## Where things stand

- Best drawing system: S3 + structure (w 1.0) + no-repeat penalty (w_div 30), improved G-GAN (FD 346), 24000 to 48000 steps at batch 32. Two-seed mean at 48000 steps: r 0.54, fresh reader 53 % (shuffled control 29 %). 96000 steps overfits the training reader.
- Same song in, same image out (deterministic). Repetition is still far above real art (twins at 0.8: 0.55 against 0.10) and comes mainly from the generator.
- Owner marks: no drawing system is "art" yet (best are "partly"); pickers (F1, S2) are "art" but are not contenders, since contenders must generate.

## Decisions waiting for the owner

1. Kaggle plan for the reset (by 2026-11-27): S3 + structure + penalty at 48000 steps and batch 64; R3GAN generator (script not written yet).
2. Whether the Stable Diffusion VAE counts as text-free (it may be indirectly CLIP-filtered); matters for latent-diffusion options.
3. Human learnability test (`docs/learnability_test.md`): designed, not approved.
4. Paper 1: waits for the owner's word.
5. `gh auth login` on the laptop (owner does it later), then GitHub About section and Releases.

## Research, in the owner's order of 2026-11-24

### Step 1: other generator techniques (now)

- R3GAN loss on the G-GAN, then a StyleGAN2-style mapping network (ranked first in `generator_options.md`). Write and debug at small width on the laptop, then train on Kaggle.
- Conditional flow-matching or diffusion decoder on the project's own non-DINOv2 representation (Diff-AE or RCG style, DINO v1 or MAE), mapper trained in that space, then distilled or shortcut for through-generator training. Likely needs more data.
- One-step flow from scratch (shortcut model first, then MeanFlow / iMF), in an own-autoencoder latent or pixels at 64 to 128 px. Likely needs more data.
- Side bet: Projected GAN / FastGAN with DINO v1 or MAE as the feature network.
- If a technique needs more and better data, the data step (step 4) moves up here. *(proposal)* Prepare the varied-art set during this step anyway.
- On each new generator: S3 + structure + penalty (24000 to 48000 steps) and Q1 + GAN, compared at the same batch and steps.
- Triage items still open in this step:
  - Q1 + GAN with the no-repeat penalty (cheap, laptop-sized; rank 3).
  - S3 + structure + penalty at Kaggle batch 64 (rank 4; separates the batch effect).
  - Drawer through a DINOv2-decodable generator (RAE or DINOv2-conditioned diffusion), taught by S2 or F1 (rank 5).
  - *(proposal)* Pool-taught drawer retried with a feature-space loss through the generator instead of latent regression (latent regression failed, 2026-11-25).

### Step 2: B-anchor and B-cluster on all contenders

- B-anchor: reflection of the code axes gave no gain on S3 + structure and on the drawer (2026-11-25). Still open: another anchor form (for example seed selection by the CAV-MAE judge), and the application to S1, Q1, S3, S4 and the G-GAN bridges on the final generator.
- B-cluster: one soft-form run on the best contender. Hard B-cluster lowered structure on both pickers; the art has little cluster structure (silhouette close to a Gaussian's).
- Throughout: D-vote, D-ease, D-loss and D-S4 wherever they give usable data.

### Step 3: B-time on the best contenders

- Sequence encoder over the track's windows (3-window version with window dropout first). The full test needs finer-window re-embedding of the music, which fits step 4.

### Step 4: larger and more varied data (after phase 3 is wrapped)

- Music: FMA-medium, re-embedded with finer windows (also serves B-time). Art: a varied set; check the WikiArt class counts first.
- Retrain the contenders on it.
- Retry the methods that may have been held back by data: D-ease / D-loss orientation check (the one partly revivable dead end), soft B-cluster, S3 light-structure and margin variants, drawer with an S2 painting teacher through a conditional decoder, Q1 + CAV-MAE pull (parked).
- Data-size curve: owner said "not yet".

### Step 5: last stand

- S5 (human-made co-occurrence: album covers, film scores), S6 (human preference as a training signal), D-human, D-complexity.

### Winners, at the end of the research phase

- Higher image resolution, winners only.
- Multi-seed runs of the winners.
- Human judging round once the images are worth judging (judging page is not refreshed without the owner's OK).
- DELAUNAY added to the art data at the end of the research.

## After all methods

- Other text-free encoders: E-other-music, E-2nd-image (second image encoder on a few methods, to see whether the outcome changes).
- Validation against human knowledge: DEAM emotion ratings, psychophysics probes, colour psychology; genre leakage.

## Last

- Creative extensions: sampling without audio, several interpretations of one track, self-directed learning from its own outputs, temporal sequences, image content beyond abstract art, an abstraction control.
- A single `generate(audio) -> image` interface with bridge and generator chosen by configuration.

## Writing and upkeep

- Paper 1 (draft parked, kept outside the public repository since 2026-11-27): the gaps in its list of missing items (analyses to move into the repository, numbers not in the log); organise the phase 1 data, logs, findings and method; funny title when the owner reopens it. A comparison paper later if the material supports it.
- Regenerate `docs/results.md` / `docs/results.csv` with `scripts/results.py`; the laptop S3 runs of 2026-11-23 to 2026-11-26 are not in it yet.
- Main README: add the training-length finding.
- Vault task list: the phase items are stale against the order of 2026-11-24.

## Dead ends (no further treatment)

S0 / cycle, S1 standalone, whitened code, G-VAE, S4 as a standalone bridge, latent-regression drawers (taught by F1, or by S2 over a GAN pool), B-anchor by code reflection. Excluded by rule: text-supervised encoders (CLAP, CLIP, ImageBind, captions), hand-written signal correspondences, CycleGAN-style translation.
