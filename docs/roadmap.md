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

Status 2026-11-22: steps 1 to 3 are done (run log entries of 2026-11-14 and 2026-11-15). A draft of the first paper exists; since 2026-11-27 it is kept outside this public repository until publication, and it is parked until the project owner reopens it. S2 has been run transductively, out of sample with joint assignment, and against a pool of G-GAN images. Step 5: a human pilot ran on 2026-11-19 (4 judges, 124 answers, plus 31 imported answers on 2026-11-21); the full judgement waits for a shortlist of winning systems.

Remaining steps:

1. Fetch and log `abstraction-q1-mean-seeds12` and `abstraction-q1-mean-controls`.
2. Combination sweep of the mapper techniques (structure terms, distribution terms, output heads, audio codes) with each approach. A combination is excluded only when the existing results make its failure certain.
3. Assemble data, protocol, deviations, results and negative results from `docs/run_log.md`.
4. First paper. A comparison paper follows if the material supports it.
5. Human judging of the phase 1 systems.

## Phase 2: S4, then S3

For each bridge: G-GAN first, then G-Diff and other diffusion or flow methods. The phase ends with a comparison of G-VAE, G-GAN and G-Diff, and with human judging.

Status 2026-11-22: G-GAN trained in five runs (best FD 388). S4: probe, D-S4 and B-anchor use CAV-MAE as a judge; the S4 bridge through the G-GAN (`methods/s4/bridge.py`) is built and queued on Kaggle. S3 through the G-GAN (`methods/s3/`) has run on Kaggle. S1 and Q1 run through the G-GAN (Q1 varied and art-like; S1 collapses). G-Diff not started.

## Phase 2.5: B-anchor and B-cluster

Applied to all methods available at that point.

Status 2026-11-22: both tried on S2 (out of sample) and F1. B-cluster lowers structure on both. B-anchor: CAV-MAE's choice on training tracks carries over to test tracks (F1 seeds, corr 0.88; S2 orientations, 0.85 to 0.94). Not yet applied to S1, Q1, S3, S4 or the G-GAN bridges.

## Phase 3: S5, then S6, then B-time

B-time is applied to all methods available at that point. D-human and D-complexity are carried out in this phase first and may then be applied to earlier methods once all methods exist. The phase ends with human judging.

## Throughout

D-vote, D-ease, D-loss and D-S4 are applied wherever they are expected to give usable data. D-S4 requires an S4 model and is available from phase 2.

From 2026-11-16 the working branch is `pipeline`: all methods (S1, Q1, F1, G-GAN) in one tree with the shared front matter of `abstraction/pipeline/` (see `docs/pipeline.md` on that branch). The method branches are kept as they are. Code replaced by the shared pipeline is kept under `methods/<method>/legacy/`.

Every evaluated run is collected by `scripts/results.py` into `docs/results.md` and `docs/results.csv`; `scripts/smoke.py` runs the smoke pipeline of every method on a branch.

## Order from 2026-11-24 (project owner's decision; supersedes the order of phases 2.5 and 3 above)

1. Other generator techniques (G-Diff and other diffusion or flow methods, other GANs) for the contending bridges. If a technique needs more and better data to work, step 4 moves up to this point.
2. B-anchor and B-cluster on all contending techniques.
3. B-time on the best contenders.
4. Larger and more varied music and art data, after phase 3 is wrapped: the contenders are retrained on it, and methods that did poorly earlier are retried in case data was their limit.
5. Last stand: S5, S6, D-human and D-complexity.

- Higher image resolution is applied to the winning systems only.
- A human judging round is held once there are proper questions to put (systems whose images read as art).
- Multi-seed runs are made for the winning candidates at the end of the research phase.
- Images in the art set that are photos of objects or rooms stay in the set.

## After all methods

- E-other-music and E-2nd-image. The second image encoder is tested on a selected few methods to measure whether a change of encoder changes the outcome.
- Validation against human knowledge, at the very end: emotion ratings (DEAM), psychophysics probes and colour psychology, to establish which known relations the machine reached without being given them and which other relations it found.

## Last

Creative extensions: sampling without audio, several interpretations of one track, self-directed learning from its own outputs, temporal sequences, image content beyond abstract art, an abstraction control, and a single `generate(audio) -> image` interface with bridge and generator selected by configuration.

## Excluded

- Text-supervised encoders and pipelines (CLAP, CLIP, ImageBind, captions, mood words).
- Hand-written correspondences between signal properties of sound and image.
- CycleGAN-style unpaired translation.
