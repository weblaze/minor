# Changelog

Versioning scheme: `vX.Y.Z` — `X` = approach being delivered (v1 = approach 01, v2 = approach 02, …),
`Y` = substantial addition or fix within that approach, `Z` = minor fixes and small additions.
`v0.x` = pre-approach foundation work. Each release gets an annotated git tag.

## [Unreleased]

### Added (branch `s1-structure`)
- S1: spatial VAE (GroupNorm, residual blocks), mapping trained with sliced Wasserstein and distance-correlation losses, variants s0, s1_cycle and shuffled, evaluation script, method description in `methods/s1/README.md`

### Added (branch `base`)
- Frozen encoders: MERT-v1-95M (per-layer, time-averaged) and DINOv2 ViT-S/14 (differentiable with respect to the image)
- Audio windowing, hash-based train/val/test splits, embedding shards, extraction scripts
- Losses: distance-correlation, sliced Wasserstein
- Metrics: Mantel test with permutation p-value, kNN overlap, retrieval accuracy, Frechet distance, kernel distance, participation ratio
- Run log: one directory per run with config, git commit, seed and metrics
- `scripts/build_art_set.py`: art collection from WikiArt style folders with a manifest
- `scripts/kaggle/`: job launcher, kernel runner and the feature-extraction job for Kaggle
- Run tag in run directory names (`system.tag`)
- `abstraction/data/codes.py`: mean, standardised and whitened audio-code transforms, optionally over a subset of MERT layers
- `docs/run_log.md`: record of every run on real data

### Added (branch `s1-structure`)
- S1: spatial VAE, mapping with latent sliced Wasserstein and distance-correlation terms, variants `s1`, `s0`, `s1_cycle`, `shuffled`, evaluation
- `--code` option: mean, standardised or whitened audio code

### Added (branch `q1-structure`)
- Q1: vector VAE, sampleability gate, diagnostics D1 to D3, baselines, mapping with feature-space and latent distribution terms and global and local structure terms, evaluation, seed comparison
- `--code` option; Kaggle jobs for the KL-weight sweep, the code swap and the standardised code

### Added (branch `f1-feature`)
- F1: mapping from audio codes to DINOv2 features, nearest-painting retrieval, staged diagnosis, evaluation with pixel-based feature sets

## [0.4.1] - 2026-11-13

### Added
- `CITATION.cff`
- README: status of the experiments (qualitative), versioning scheme, authors, licence statement

### Tags
- Method-branch snapshots are tagged `<method>-vX.Y.Z` and published as pre-releases: `base-v0.1.0`, `s1-v0.1.0`, `q1-v0.1.0` (first runs on real data, seed 0).

## [0.4.0] - 2026-11-10

Cleanup after the 2026-11 audit. Removed code remains in the git history (last commit before the cleanup: `ff46080`).

### Removed: text-supervised or label-supervised models (violate the training constraints)
- `abstraction/audio/clap.py`, `preprocessing/extract_clap_features.py`
- `approaches/01_conditioned_autoencoder/` (CLAP-conditioned decoder; VGG16 style loss)
- `approaches/02_latent_diffusion_clap/` (CLAP conditioning; `make_pairs.py` built pseudo-pairs from 16 mood words through CLAP and OpenCLIP text encoders)
- `approaches/04_temporal_video/` (depends on approach 02)
- `abstraction/pipelines/latent_diffusion.py`, `app.py`, `run_overnight.ps1`, `NEXT_STEPS.md`
- `evaluation/clip_alignment.py`, `evaluation/retrieval.py` (CLAP and CLIP), `evaluation/fid.py` (Inception), `evaluation/lpips_diversity.py` (LPIPS)
- `abstraction/models/vae_codec.py` (wrapper for the Stable Diffusion VAE, which is trained with LPIPS on a text-image corpus)
- `ConditionedImageVAE`, `ClapImageDataset`, `LatentDataset`
- `visualization/visualize_latent_space.py` (CLAP embeddings)

### Removed: defects
- `abstraction/models/scheduler.py`: `DDPMScheduler.step` applied the one-step posterior while sampling on a strided subset of the timesteps
- `approaches/03_byol_a_encoder/eval_encoder.py`: genre labels read from folder names; FMA-small folders are numbered
- `abstraction/audio/features.py`, `preprocessing/extract_audio_features.py`, `AudioFeatureDataset`: six 5 s segments averaged frame by frame; per-file normalisation
- `evaluation/reconstruction.py`, `scratch/`: depended on removed code or on files that are not in the repository
- `legacy/` (v0.1): the mapping network was trained with MSE between audio and image latents paired by dataset index (random pairing); the inference scripts require an `ImageVAE` definition that is no longer in the code

### Removed: simulated or stale artefacts
- `legacy/visualization/plot_loss_curves.py`, `plot_cycle_consistency.py`, `plot_image_evolution.py`: the plotted values were generated in the script, not logged from training
- Figures in `visualization/` produced by those scripts (`fid_scores.png`, `cycle_consistency.png`, `image_evolution.png`, three loss-curve figures) and `latent_space_distribution.png`
- `tmodels/` (v0.1 checkpoints that cannot be loaded with the current model definitions; reconstruction images)
- `evaluation/results/vae_reconstruction_test.png`

### Changed
- `README.md` rewritten to state the actual status
- Dependencies reduced to those used by the remaining code
- Placeholder author email removed from `pyproject.toml`

## [0.3.0] — 2026-06-11

Foundation restructure: modular core package + one subdirectory per approach.

### Added
- `abstraction/` shared core package (models, audio encoders, datasets, pipelines, utils) — `pip install -e .`
- `approaches/01_conditioned_autoencoder/` — stamp the VAE approach complete with CLAP conditioning
- `approaches/02_latent_diffusion_clap/` — latent diffusion + CLAP + classifier-free guidance (primary)
- `approaches/03_byol_a_encoder/` — self-supervised BYOL-A encoder ablation vs CLAP
- `approaches/04_temporal_video/` — per-segment generation + latent interpolation music video
- Evaluation suite: FID, CLAP↔CLIP alignment, LPIPS diversity, retrieval metrics
- Two-layer config (`configs/base.yaml` + per-approach `config.yaml`)
- `CHANGELOG.md`, `NEXT_STEPS.md`, `datasets/README.md`, per-approach READMEs

### Changed
- Dependencies completed (`diffusers`, `transformers`, `wandb`, `laion-clap`, etc. now declared)
- wandb runs grouped per approach with timestamped names
- All `torch.load` calls use `weights_only=True`

### Fixed
- `ImageVAE(latent_dim=...)` constructor mismatch crashing 5 scripts
- Inference flattened spatial latents before the Conv2d decoder (crash)
- `ConditionalUNet.forward` computed a dead path twice per step
- Hardcoded `D:\musicc\` paths and stale comments

### Archived
- v0.1 mapping-network pipeline and legacy CLI scripts moved to `legacy/` (see `legacy/README.md`)

## [0.2.0] — earlier

- Codebase foundation, wandb tracking, latent diffusion pipeline with CLAP conditioning, parallel training scripts.
