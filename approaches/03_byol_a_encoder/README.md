# Approach 03: self-supervised BYOL-A audio encoder

An audio encoder trained from scratch on log-mel spectrograms with the BYOL objective (Niizumi et al., 2021, simplified). No labels and no text are used.

Status: untrained. No result exists for this encoder.

```
log-mel -> two augmented views (mixup, random resize crop) -> online / target encoders
        -> BYOL loss (online predicts the target projection; target is an EMA of the online weights)
```

## Usage

```bash
python approaches/03_byol_a_encoder/train_byol_a.py
python approaches/03_byol_a_encoder/extract_byol_features.py
```

`extract_byol_features.py` writes one 512-d embedding per track to `datasets/byol_features/`.

The former evaluation script (`eval_encoder.py`) was removed in v0.4.0. It read genre labels from folder names; FMA-small folders are numbered, so its linear probe did not measure genre separability.
