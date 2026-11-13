# Roadmap

Set by the project owner on 2026-11-14 (second revision of that day). A bridge connects music to images; a generator produces the image; an encoder turns audio or an image into a vector.

## Identifiers

| Kind | Identifier | Meaning |
|---|---|---|
| Bridge | S0 | cycle consistency (2025 pipeline) |
| Bridge | S1 | structure-preserving mapping into a VAE latent |
| Bridge | Q1 | S1 with a vector VAE and a feature-space distribution term |
| Bridge | F1 | mapping into image-feature space with nearest-painting retrieval |
| Bridge | S2 | blind matching of the two spaces by shape |
| Bridge | S3 | self-invented visual language (information maximisation) |
| Bridge | S4 | natural audio-visual statistics: how sound and sight co-occur in real-world video |
| Bridge | S5 | human-made co-occurrence (album covers, film scores) |
| Bridge | S6 | human preference as a training signal |
| Method | B-anchor | settling the orientation of the mapping, judged by the machine |
| Method | B-cluster | cluster both collections, then align groups to groups |
| Method | B-time | use the sequence of windows of a track instead of one averaged code |
| Orientation test | D-vote | majority orientation over many random initialisations |
| Orientation test | D-ease | which orientation is learned faster or with less capacity |
| Orientation test | D-loss | comparison of the training and validation losses of the two orientations |
| Orientation test | D-S4 | a model trained on natural audio-visual co-occurrence scores the two orientations |
| Orientation test | D-complexity | simple-to-simple as an imposed rule |
| Orientation test | D-human | orientation chosen by human judges |
| Generator | G-VAE, G-GAN, G-Diff | variational autoencoder, adversarial network, diffusion or flow model |
| Encoder | E-MERT, E-DINOv2 | in use |
| Encoder | E-other-music, E-2nd-image | other text-free encoders, not yet used |

## Phase 1: S0, S1, Q1, F1, S2 with G-VAE

Status: experiments conclude when the Q1 seed and control runs of 2026-11-14 are fetched and logged.

Remaining steps:

1. Fetch and log `abstraction-q1-mean-seeds12` and `abstraction-q1-mean-controls`.
2. Combination sweep of the mapper techniques (structure terms, distribution terms, output heads, audio codes) with each approach. A combination is excluded only when the existing results make its failure certain.
3. Assemble data, protocol, deviations, results and negative results from `docs/run_log.md`.
4. First paper. A comparison paper follows if the material supports it.
5. Human judging of the phase 1 systems.

## Phase 2: S4, then S3

For each bridge: G-GAN first, then G-Diff and other diffusion or flow methods. The phase ends with a comparison of G-VAE, G-GAN and G-Diff, and with human judging.

## Phase 2.5: B-anchor and B-cluster

Applied to all methods available at that point.

## Phase 3: S5, then S6, then B-time

B-time is applied to all methods available at that point. D-human and D-complexity are carried out in this phase first and may then be applied to earlier methods once all methods exist. The phase ends with human judging.

## Throughout

D-vote, D-ease, D-loss and D-S4 are applied wherever they are expected to give usable data. D-S4 requires an S4 model and is available from phase 2.

Every evaluated run is collected by `scripts/results.py` into `docs/results.md` and `docs/results.csv`; `scripts/smoke.py` runs the smoke pipeline of every method on a branch.

## After all methods

- E-other-music and E-2nd-image. The second image encoder is tested on a selected few methods to measure whether a change of encoder changes the outcome.
- Validation against human knowledge, at the very end: emotion ratings (DEAM), psychophysics probes and colour psychology, to establish which known relations the machine reached without being given them and which other relations it found.

## Last

Creative extensions: sampling without audio, several interpretations of one track, self-directed learning from its own outputs, temporal sequences, image content beyond abstract art, an abstraction control, and a single `generate(audio) -> image` interface with bridge and generator selected by configuration.

## Excluded

- Text-supervised encoders and pipelines (CLAP, CLIP, ImageBind, captions, mood words).
- Hand-written correspondences between signal properties of sound and image.
- CycleGAN-style unpaired translation.
