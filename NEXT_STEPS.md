# NEXT STEPS — your running checklist

Everything here is **your part**: long-running training, data placement, and decisions that need results.
Development never blocks on these — items get added as phases complete. Work top to bottom.

## Setup (one-time, ~30 min)

- [ ] Place datasets per [datasets/README.md](datasets/README.md) (`fma_small/`, `abstract_art/`, and your existing `audio_features/`, `clap_features/` if you still have them)
- [ ] Download the CLAP checkpoint `music_audioset_epoch_15_esc_90.14.pt` to the repo root (link in datasets/README.md)
- [ ] `pip install -r requirements.txt && pip install -e .`
- [ ] `wandb login` (or set `WANDB_MODE=offline`)

## Training run order (all commands from repo root)

0. [ ] Extract features if not already on disk:
       `python preprocessing/extract_audio_features.py` and `python preprocessing/extract_clap_features.py`
1. [ ] Pairs (needs `pip install open_clip_torch` for pseudo):
       `python approaches/02_latent_diffusion_clap/make_pairs.py --strategy pseudo`
2. [ ] Approach 01 pretrain (skip if you copy an existing trained `image_vae.pth`
       to `tmodels/01_conditioned_autoencoder/image_vae.pth`):
       `python approaches/01_conditioned_autoencoder/train.py --stage pretrain`
3. [ ] Approach 01 finetune (~1–2 h on GTX 1650):
       `python approaches/01_conditioned_autoencoder/train.py --stage finetune`
4. [ ] **Gate** — genre gallery with 5 contrasting songs; rows must differ visibly:
       `python approaches/01_conditioned_autoencoder/sample_gallery.py --audio a.mp3 b.mp3 ...`
       → if it passes, tag it: `git tag -a v1.0.0 -m "approach 01 complete"`
5. [ ] Precompute latents: `python approaches/02_latent_diffusion_clap/precompute_latents.py`
6. [ ] Approach 02 smoke then train (overnight on 1650 / hours on A100):
       `python approaches/02_latent_diffusion_clap/train.py --max-steps 5 --no-wandb`
       `python approaches/02_latent_diffusion_clap/train.py`
7. [ ] Sample + dream, then run the evaluation protocol in [evaluation/README.md](evaluation/README.md)
       → tag `v2.0.0` when alignment beats the shuffled baseline
8. [ ] Approach 03 BYOL-A (start `--epochs 10` locally, full run on A100):
       `python approaches/03_byol_a_encoder/train_byol_a.py`
       then `extract_byol_features.py` + `eval_encoder.py` → tag `v3.0.0`
9. [ ] Approach 04 video (`pip install "imageio[ffmpeg]"`):
       `python approaches/04_temporal_video/generate_video.py --audio song.mp3` → tag `v4.0.0`
10. [ ] Human study (design in evaluation/README.md + research docs): ≥20 raters,
        conditioned vs dream vs random, Likert 1–5

## Decisions to make once results exist

- [ ] Own ImageVAE vs pretrained SD VAE for approach 02 (train own first on 1650; compare with SD VAE on cloud)
- [ ] Pairing strategy: `random` vs `pseudo` (compare clip_alignment scores after short runs of each)
- [ ] Guidance scale sweep (`sample.py --guidance ...`)
- [ ] Whether BYOL-A replaces CLAP (driven by `eval_encoder.py` retrieval numbers)

## Flagged for your confirmation (nothing deleted without your OK)

- [ ] `tmodels/` contains ~40 epoch visualization PNGs from old training runs — proposal: move them to `tmodels/_archive/` to declutter
- [ ] `generated_images/` holds 18 old inference outputs — keep, archive, or clear?
- [ ] `evaluation/results/` and `visualization/latent_space_plots/` PNGs are tracked in git from old runs — untrack them? (now gitignored going forward)
