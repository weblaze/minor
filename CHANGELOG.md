# Changelog

Versioning scheme: `vX.Y.Z` — `X` = approach being delivered (v1 = approach 01, v2 = approach 02, …),
`Y` = substantial addition or fix within that approach, `Z` = minor fixes and small additions.
`v0.x` = pre-approach foundation work. Each release gets an annotated git tag.

## [Unreleased]

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
