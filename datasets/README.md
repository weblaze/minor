# Datasets

This directory is gitignored — place data here locally. Expected layout:

```
datasets/
├── fma_small/                  # FMA small subset: genre folders containing .mp3/.wav
│   ├── Electronic/
│   ├── Rock/
│   └── ...
├── abstract_art/               # ~7000 abstract art images (.png/.jpg), 512x512 (trained at 128)
├── audio_features/             # generated: preprocessing/extract_audio_features.py (.npy MFCC dicts)
├── clap_features/              # generated: preprocessing/extract_clap_features.py (*_clap.npy, 512-d)
├── byol_features/              # generated: approaches/03_byol_a_encoder/extract_byol_features.py
├── pairs_02.json               # generated: approaches/02_latent_diffusion_clap/make_pairs.py
└── latents_02/                 # generated: approaches/02_latent_diffusion_clap/precompute_latents.py
    ├── own/                    # latents from your ImageVAE (8ch x 16x16)
    └── sd/                     # latents from pretrained SD VAE (4ch x 16x16)
```

## External checkpoints

- **LAION CLAP music checkpoint** (required for CLAP embeddings and inference):
  download `music_audioset_epoch_15_esc_90.14.pt` from
  https://huggingface.co/lukewys/laion_clap/blob/music_audioset_epoch_15_esc_90.14.pt
  and place it at the repo root (path configurable in `configs/base.yaml` → `paths.clap_checkpoint`).
- **SD VAE** (`stabilityai/sd-vae-ft-mse`) downloads automatically via `diffusers` on first use.

## Sources

- FMA: https://github.com/mdeff/fma (use the `fma_small` 8k-track subset)
- Abstract art: any collection of abstract artwork images works; WikiArt abstract category is a good source
