# S3: self-invented visual language (design, not implemented)

Status 2026-11-21: design proposal by Claude, not yet approved by the project owner. No code exists.

## Idea

S1, Q1, F1 and S2 transfer the geometry of the music: similar tracks get similar images. They do not ask whether an image carries enough of its track that the track can be recovered from it. S3 asks only that. A mapping M turns an audio code into an image, and a reader R has to tell from the image alone which track it was made for. M and R are trained together, so the correspondence between sound and image is invented by the two networks rather than taken from distances, labels, text or human rules. This is a referential game (speaker M, listener R) with the image as the message, trained by information maximisation.

## Model

```
audio code a --M--> latent z --G (frozen G-GAN)--> image x --aug--> DINOv2 (frozen) --R--> e
```

- G: the G-GAN generator (gan_r1amp seed 0, the one marked "art"), frozen. Only images G can make are possible messages, which keeps them in the abstract-art manifold.
- M: the mapper of S1 (`abstraction/models/mapper.py`), audio code to z.
- R: a small MLP from DINOv2 features to an embedding of the audio-code dimension.
- aug: random resized crop, flip, small blur and noise applied to x before DINOv2. The message has to survive changes a viewer would not notice. This should stop it from hiding in pixel patterns that only DINOv2 sees. There is no colour jitter, because colour is expected to carry information.

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

## Open points for the project owner

1. Is a jointly trained reader acceptable? The alternative is a fixed linear readout, which is close to S1/F1 and invents less.
2. The augmentation set: crop, flip, blur and noise, without colour jitter.
3. Should the G-GAN be the only generator for S3 at first? The roadmap says GAN first, diffusion second.
