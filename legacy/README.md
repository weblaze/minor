# Legacy v0.1 — VAE + Mapping Network Pipeline (archived)

This folder preserves the original approach exactly as it was when development paused:
**AudioVAE → MappingNetwork → ImageVAE decoder**, with cycle-consistency, MMD, and
diversity losses bridging two *unpaired* latent spaces. It is kept for the project
report and as a historical baseline. **It is not maintained.**

## What's here

| File | Role |
|---|---|
| `models/mapping_network.py` | MappingNetwork + InverseMappingNetwork MLPs |
| `training/train_audio_vae.py` | AudioVAE training (completed 100 epochs historically) |
| `training/train_mapping_network.py` | Multi-objective mapping training |
| `training/master_train.py` | Parallel process orchestrator |
| `training/test_model.py` | AudioVAE MSE smoke test (hardcoded old paths) |
| `scripts/generate_image.py` | CLI: audio features → mapping → image (no diffusion) |
| `scripts/generate_image_from_mp3.py` | CLI: MP3 → librosa features → image |
| `scripts/extract_audio_features.py` | Duplicate of the preprocessing extractor |
| `visualization/plot_*.py` | Stub/simulated-data plot scripts |

## Why these scripts no longer run as-is

1. **Checkpoint incompatibility**: `tmodels/mapping_network.pth` and
   `inverse_mapping_network.pth` were trained against an older ImageVAE with a
   **flat Linear 512-d latent**. The current `abstraction.models.image_vae.ImageVAE`
   uses a spatial 8×16×16 latent — the mapping checkpoints cannot bridge to it.
   The old VAE class only exists in git history (pre-v0.2 commits).
2. **Constructor mismatch**: several scripts call `ImageVAE(latent_dim=512)`,
   which the current class does not accept.
3. **Moved imports/config**: these files still import from the old `models/` and
   `training/` layout and read `configs/config.yaml`, both of which were
   restructured in v0.3 (`abstraction/` package, `configs/base.yaml`).

To resurrect the pipeline for a side-by-side comparison, check out a pre-v0.3
commit (e.g. `3a8542f`) rather than patching these copies.

## What replaced it

- The latent-space bridging idea lives on as **approach 02** (latent diffusion
  conditioned on CLAP embeddings) — see `approaches/02_latent_diffusion_clap/`.
- The trained `tmodels/audio_vae.pth` still loads against
  `abstraction.models.audio_vae.AudioVAE` if you need audio latents.
