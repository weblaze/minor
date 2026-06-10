# Approach 03 — Self-Supervised BYOL-A Encoder (ablation vs CLAP)

**Priority 2 (scored 8/10 in the 2026 audit).** Trains an audio encoder from
scratch with zero labels and zero text supervision — the purest version of the
project's "no human bias" philosophy. The research contribution is the
**ablation**: does CLAP's semantic supervision actually buy better music→image
alignment than pure acoustics, and by how much?

## How it works

```
log-mel ──augment×2 (mixup + random-resize-crop)──► online / target encoders
        ──BYOL loss (predict target projection, EMA target, no negatives)
```

The trained encoder produces 512-d embeddings that drop into approach 02's
diffusion pipeline **unchanged** — only the conditioning source swaps.

## Your part — do this next

```bash
# 1. Train the encoder self-supervised on FMA (no labels needed)
python approaches/03_byol_a_encoder/train_byol_a.py
#    GTX 1650: several days for 50 epochs. Rented A100: hours. Start with
#    --epochs 10 locally to sanity-check loss decreases, then go cloud.

# 2. Extract embeddings for the whole dataset
python approaches/03_byol_a_encoder/extract_byol_features.py

# 3. The encoder ablation (needs clap_features extracted too)
python approaches/03_byol_a_encoder/eval_encoder.py
#    -> evaluation/results/encoder_ablation.json + t-SNE comparison plot

# 4. Retrain the diffusion model with BYOL-A conditioning:
#    in approaches/02_latent_diffusion_clap/config.yaml set
#      cond.source: byol_a
#      paths.unet_checkpoint: tmodels/02_latent_diffusion_clap/unet_byola.pth
#    then rebuild pairs/latents pointing clap_features -> byol_features and re-run
#    make_pairs / precompute_latents / train.

# 5. Compare: clip_alignment + retrieval for CLAP-conditioned vs BYOL-A-conditioned
#    generations. Document the numbers — this comparison IS the contribution.
```

**What to expect** (from the research docs): BYOL-A captures timbre/rhythm
precisely but may miss semantic/emotional categories; CLAP clusters by
genre/mood. If CLAP wins alignment but BYOL-A wins acoustic retrieval, that
split result is itself a publishable observation.
