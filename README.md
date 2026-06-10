# Abstraction — Music-to-Image AI (v0.3)

**Generates abstract visual artwork from how music *feels* — timbre, mood, rhythm —
with no text prompts anywhere in the pipeline.** Text descriptions of music are
lossy: they drop micro-rhythm, harmonic tension, and emotional nuance that only
exists in the waveform. This system conditions image generation on the waveform
directly, and can also *dream*: generate images from its internalized musical
imagination with no audio input at all.

## The approaches

This repo is a comparative study. Each approach lives in its own subdirectory
with a config, runnable scripts, and a README that says exactly what to run.
Work through them in order — see [NEXT_STEPS.md](NEXT_STEPS.md) for the live checklist.

| # | Approach | Status | Entry point |
|---|---|---|---|
| 01 | [Conditioned autoencoder](approaches/01_conditioned_autoencoder/README.md) — CLAP-conditioned VAE decoder; stamps the original approach complete | code ready, training pending | `train.py --stage pretrain/finetune` |
| 02 | [Latent diffusion + CLAP + CFG](approaches/02_latent_diffusion_clap/README.md) — **primary**; classifier-free guidance gives dreaming for free | code ready, training pending | `make_pairs.py` → `precompute_latents.py` → `train.py` |
| 03 | [BYOL-A encoder ablation](approaches/03_byol_a_encoder/README.md) — self-supervised encoder vs CLAP: what does supervision buy? | code ready, training pending | `train_byol_a.py` → `eval_encoder.py` |
| 04 | [Temporal music video](approaches/04_temporal_video/README.md) — per-segment generation + latent slerp, dream segments | code ready, needs 02 trained | `generate_video.py` |

The v0.1 mapping-network pipeline is archived untouched in [legacy/](legacy/README.md).
Approaches scored ≤5/10 in the 2026 research audit (paired-supervised, CycleGAN,
VQGAN+CLIP, VQ-VAE transformer) are intentionally not scaffolded.

## Repository layout

```
abstraction/          shared core package (pip install -e .)
  models/             AudioVAE · ImageVAE (+conditioned) · ConditionalUNet · DDPM scheduler
                      vae_codec (own VAE | pretrained SD VAE via diffusers)
  audio/              ClapEncoder wrapper · librosa feature extraction
  data/               ImageDataset · AudioFeatureDataset · ClapImageDataset · LatentDataset
  pipelines/          latent_diffusion: load_models + generate_diffusion (CFG, dream mode)
  utils/              config loader · checkpoint helpers · wandb grouping
approaches/           one directory per approach (see table above)
preprocessing/        feature extraction CLIs (MFCC, CLAP)
evaluation/           FID · CLAP↔CLIP alignment · LPIPS diversity · retrieval (README inside)
visualization/        latent-space t-SNE · real wandb loss-curve export
configs/base.yaml     shared config; each approach merges its own config.yaml on top
legacy/               archived v0.1 pipeline
app.py                Streamlit demo (upload/YouTube → image, dreaming mode, CFG slider)
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .                        # makes `abstraction` importable everywhere
# optional extras:
pip install -e .[eval]                  # clean-fid, lpips, open_clip_torch
pip install -e .[video]                 # imageio[ffmpeg] for approach 04
```

Then place datasets and the CLAP checkpoint per [datasets/README.md](datasets/README.md),
and `wandb login` (or `set WANDB_MODE=offline`).

## Hardware

Two tiers (per the 2026 research stress test):
- **Dev / validation:** GTX 1650 (4 GB). Everything runs at 128×128 with
  precomputed latents; diffusion training is an overnight job.
- **Final runs:** rented cloud GPU (A100/L40S) for the BYOL-A encoder and the
  SD-VAE variant of approach 02 — a few hours, cheap.

## Demo

```bash
streamlit run app.py
```

Upload an MP3/WAV or paste a YouTube link; adjust the guidance scale in the
sidebar; hit **Start Dreaming** for null-conditioned generation.

## Versioning

`vX.Y.Z` — `X` = approach being delivered (v1 = approach 01, …), `Y` =
substantial addition/fix, `Z` = minor. `v0.x` = foundation. History in
[CHANGELOG.md](CHANGELOG.md); each release is an annotated git tag.

Research context (approach audits, methodology, stress test) lives in the
local `reserach/` folder (untracked).
