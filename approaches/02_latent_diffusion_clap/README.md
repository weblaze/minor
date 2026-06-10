# Approach 02 — Latent Diffusion Conditioned on CLAP Embeddings

**Priority 1 (scored 9/10 in the 2026 audit).** A small UNet denoiser operates in
VAE latent space, conditioned on 512-d LAION CLAP audio embeddings. Classifier-free
guidance is trained in from the start, which gives the **dreaming mode**
(null-conditioned generation) for free — the distinguishing feature of this project.

## How it works

```
audio.mp3 ──CLAP──► 512-d embedding ─┐
                                     ▼
noise [C,16,16] ──UNet (T steps, CFG)──► denoised latent ──VAE decoder──► 128×128 image
```

- **VAE codec** (`vae.kind` in config.yaml): `own` = your ImageVAE (8ch latents),
  `sd` = pretrained Stable Diffusion VAE via diffusers (4ch latents, higher quality
  ceiling, fp16). Latents are precomputed so training VRAM stays tiny either way.
- **Unpaired data**: `make_pairs.py` bridges the unpaired audio/image sets through
  a shared mood vocabulary (CLAP text scores for audio, CLIP text scores for images).
  `--strategy random` gives the baseline where conditioning carries no signal.
- **CFG**: during training, conditioning is zeroed with probability `cfg_dropout`.
  At inference the UNet runs twice per step and blends:
  `eps = eps_null + s · (eps_cond − eps_null)`.

## Your part — do this next

Run from the repo root (after `pip install -e .` and dataset setup — see NEXT_STEPS.md):

```bash
# 1. Build audio↔image pairs (needs clap_features extracted; pseudo also needs pip install open_clip_torch)
python approaches/02_latent_diffusion_clap/make_pairs.py --strategy pseudo

# 2. Encode all paired images to latents (one VAE pass, ~minutes)
python approaches/02_latent_diffusion_clap/precompute_latents.py

# 3. Smoke test the trainer (seconds)
python approaches/02_latent_diffusion_clap/train.py --max-steps 5 --no-wandb

# 4. Real training
python approaches/02_latent_diffusion_clap/train.py
#    GTX 1650: overnight at 128px. Rented A100: a few hours.

# 5. Sample + dream
python approaches/02_latent_diffusion_clap/sample.py --audio path/to/song.mp3 --n 4
python approaches/02_latent_diffusion_clap/sample.py --dream --n 4
```

**Go/no-go gate before approach 03:** generations for different genres must be
visibly different, and `evaluation/clip_alignment.py` must score pseudo-paired
training above random-paired training. If conditioning shows no effect, revisit
`pairing.top_k` and `cfg_dropout` before moving on.

### Experiments to run (record results in wandb)

| Experiment | How |
|---|---|
| random vs pseudo pairing | `make_pairs.py --strategy random`, retrain, compare clip_alignment |
| guidance scale sweep | `sample.py --guidance 1.5 / 3 / 5 / 7.5` on the same song |
| own VAE vs SD VAE | set `vae.kind: sd`, change `paths.unet_checkpoint` to `unet_sd.pth` and `paths.latents_dir` to `datasets/latents_02/sd`, re-run precompute + train (cloud GPU recommended) |

## Prerequisites

- `datasets/clap_features/` populated (`python preprocessing/extract_clap_features.py`)
- For `vae.kind: own`: a trained ImageVAE at `tmodels/01_conditioned_autoencoder/image_vae.pth`
  (approach 01 pretrain stage, or copy your previously trained image_vae.pth there)
- For `vae.kind: sd`: nothing — weights download automatically
- CLAP checkpoint at repo root (see datasets/README.md)
