# Approach 01 — Conditioned Autoencoder (stamping the original approach complete)

The original v0.1/v0.2 autoencoder produced color blobs because nothing connected
audio to the decoder. Per the 2026 stress-test doc, the minimal fix that makes
this approach *demonstrably conditioned* is: project the 512-d CLAP embedding
into the decoder and finetune briefly. That is exactly what this stage does —
no diffusion, no new architecture, just the missing conditioning layer.

## How it works

```
stage pretrain:  image ──ImageVAE──► recon          (MSE + VGG16 style + KL warm-up)
stage finetune:  image + CLAP emb ──ConditionedImageVAE──► recon
                 (cond_proj adds the projected embedding to the decoder input;
                  warm-started from pretrain, only cond_proj starts untrained)
```

Conditioning is dropped 10% of the time during finetune so the decoder still
works without audio.

Two genuine fixes over the old trainer are baked in: the VGG style loss now
actually backpropagates (the v0.2 version detached the reconstruction, silently
disabling it), and checkpoints load with `weights_only=True`.

## Your part — do this next

```bash
# 0. If you still have your previously trained image_vae.pth, copy it to
#    tmodels/01_conditioned_autoencoder/image_vae.pth and skip step 1.

# 1. Pretrain the plain VAE (GTX 1650: a few hours for 40 epochs)
python approaches/01_conditioned_autoencoder/train.py --stage pretrain

# 2. Build pairs if you haven't (shared with approach 02)
python approaches/02_latent_diffusion_clap/make_pairs.py

# 3. Finetune with CLAP conditioning (GTX 1650: ~1-2 h for 20 epochs)
python approaches/01_conditioned_autoencoder/train.py --stage finetune

# 4. The gate: generate the genre gallery with 5 contrasting songs
python approaches/01_conditioned_autoencoder/sample_gallery.py \
    --audio jazz.mp3 metal.mp3 piano.mp3 edm.mp3 folk.mp3
```

**Stamp-complete checklist** (from the stress-test doc):
- [ ] Gallery rows differ visibly by song (palette/texture vary with genre/mood)
- [ ] `evaluation/retrieval.py` beats random retrieval
- [ ] Gallery image saved for the report

When all three pass, this approach is done — tag it `v1.0.0` and move fully to
approach 02 (which reuses the pretrained VAE from step 1 as its codec).
