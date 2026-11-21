# S3: self-invented visual language

Status 2026-11-21: design by Claude; the three open points were decided by Arunav on 2026-11-21 (below). Code: `train.py`, `identify.py`, `config.yaml`; Kaggle job `scripts/kaggle/experiments/s3_gan.yaml`. No run on real data yet.

## Idea

S1, Q1, F1 and S2 transfer the geometry of the music: similar tracks get similar images. They do not ask whether an image carries enough of its track that the track can be recovered from it. S3 asks only that. A mapping M turns an audio code into an image, and a reader R has to tell from the image alone which track it was made for. M and R are trained together, so the correspondence between sound and image is invented by the two networks rather than taken from distances, labels, text or human rules. This is a referential game (speaker M, listener R) with the image as the message, trained by information maximisation.

## Model

```
audio code a --M--> latent z --G (frozen G-GAN)--> image x --aug--> DINOv2 (frozen) --R--> e
```

- G: the G-GAN generator (gan_r1amp seed 0, the one marked "art"), frozen. Only images G can make are possible messages, which keeps them in the abstract-art manifold.
- M: the mapper of S1 (`abstraction/models/mapper.py`), audio code to z.
- R: a small MLP from DINOv2 features to an embedding of the audio-code dimension.
- aug: light random resized crop (85 to 100 % of the area, aspect kept), small noise and, on 30 % of images, a light blur, applied to x before DINOv2. The message has to survive changes a viewer would not notice. This should stop it from hiding in pixel patterns that only DINOv2 sees. No flip, because it changes the composition; no colour jitter, because colour is expected to carry information.
- The audio side of the game is a second small MLP from the audio code to the same embedding space; both MLPs together are R.

## Loss

```
L = L_info + w_lat * L_lat + w_feat * L_feat
```

- L_info: InfoNCE over the batch (temperature tau). Each reader output R(DINOv2(aug(x_i))) has to pick its own audio code a_i among the batch's codes (and a_i its image, symmetric). This is a lower bound on the mutual information between track and image.
- L_lat: sliced Wasserstein distance between M(a) and samples of N(0, I), as in the GAN bridge of S1, so that M stays where G was trained.
- L_feat: sliced Wasserstein distance between DINOv2 features of the generated images and of real training art, as in Q1. This keeps the outputs art-like as a set.

There is no structure term: S3 does not ask that similar tracks look similar. Whether that emerges on its own is one of the measurements.

## Variants and controls

| Name | Change | Purpose |
|---|---|---|
| s3 | as above | main |
| s3_noaug | no augmentation | how much of the message is invisible without it |
| s3_struct | plus the S1 structure term | combination with S1 (for the later combination round) |
| shuffled | InfoNCE targets permuted within the batch | the loss carries no information about the input |
| random M | untrained M, R trained | how much an arbitrary mapping already transmits |

## Evaluation (test tracks only)

- Information: the reader identifies the track of each test image among the 789 test tracks; top-1 and top-10 accuracy, and bits = log2(789) minus the conditional entropy estimate from the reader's softmax.
- Second-reader check: a fresh reader on frozen DINO v1 features (the second image encoder already used) is trained on training tracks and tested on test tracks. If it also decodes the message, the language is not specific to DINOv2.
- Window consistency: images of two windows of the same track compared with images of two different tracks.
- Structure (side measure): Mantel r between audio distances and image-feature distances. It is not optimised.
- Realism: FD against held-out art, as for the other bridges, and the Image Atlas for a human look.
- Human judging later: whether people can match images to tracks better than chance once they see a few examples (learnability of the language).

## Cost

One S3 step is about one S1 step plus a small reader: about 2.2 s per step on the laptop GPU, about 1 s on a Kaggle T4. 20000 steps take about 6 h per run. Two runs fit in one Kaggle session (s3 and shuffled on two GPUs).

## Decisions (Arunav, 2026-11-21)

1. The reader is trained together with the mapper.
2. Augmentation: flip and noise might ruin the composition, and blur might collapse detail; use them only if they do not. Implemented: no flip; noise and blur kept light (config `s3.aug`). The `s3_noaug` variant and the clean versus augmented scores of `identify.py` measure what the augmentation costs.
3. G-GAN first; other generators later.

Not yet implemented from the evaluation list: the second reader on DINO v1 and the window-consistency comparison of two windows (identify.py tests single-window images against the track's code instead).
