# NEXT STEPS — your running checklist

Everything here is **your part**: long-running training, data placement, and decisions that need results.
Development never blocks on these — items get added as phases complete. Work top to bottom.

## Setup (one-time, ~30 min)

- [ ] Place datasets per [datasets/README.md](datasets/README.md) (`fma_small/`, `abstract_art/`, and your existing `audio_features/`, `clap_features/` if you still have them)
- [ ] Download the CLAP checkpoint `music_audioset_epoch_15_esc_90.14.pt` to the repo root (link in datasets/README.md)
- [ ] `pip install -r requirements.txt && pip install -e .`
- [ ] `wandb login` (or set `WANDB_MODE=offline`)

## Training run order

(Items will be filled in with exact commands as each approach phase lands.)

1. [ ] Approach 02: `make_pairs.py` — generate audio↔image pseudo-pairs
2. [ ] Approach 02: `precompute_latents.py`
3. [ ] Approach 01: pretrain stage (or reuse an existing `image_vae.pth` if you have one)
4. [ ] Approach 01: finetune stage with CLAP conditioning (~1–2 h on GTX 1650)
5. [ ] Approach 01 gate: check `sample_gallery.py` output — images must differ visibly by genre before moving on
6. [ ] Approach 02: train diffusion (overnight on 1650 at 128px, or a few hours on a rented A100)
7. [ ] Evaluation suite over approach 01 + 02 outputs
8. [ ] Approach 03: BYOL-A training (A100 recommended, several days on 1650)
9. [ ] Approach 04: generate music videos from the approach 02 model

## Decisions to make once results exist

- [ ] Own ImageVAE vs pretrained SD VAE for approach 02 (train own first on 1650; compare with SD VAE on cloud)
- [ ] Pairing strategy: `random` vs `pseudo` (compare clip_alignment scores after short runs of each)
- [ ] Guidance scale sweep (`sample.py --guidance ...`)
- [ ] Whether BYOL-A replaces CLAP (driven by `eval_encoder.py` retrieval numbers)

## Flagged for your confirmation (nothing deleted without your OK)

- [ ] `tmodels/` contains ~40 epoch visualization PNGs from old training runs — proposal: move them to `tmodels/_archive/` to declutter
- [ ] `generated_images/` holds 18 old inference outputs — keep, archive, or clear?
- [ ] `evaluation/results/` and `visualization/latent_space_plots/` PNGs are tracked in git from old runs — untrack them? (now gitignored going forward)
