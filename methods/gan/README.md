# G-GAN: unconditional generator of abstract art

Status: implemented and smoke-tested on synthetic data. First run on real data launched on 2026-11-14; results are recorded in `docs/run_log.md`. This document contains no results.

## 1. Purpose

Phase 2 replaces the VAE generator of phase 1 by an adversarial generator and later by diffusion generators, under the same bridges. G-GAN is the generator only: it is trained on the art collection without any audio input. A bridge then maps audio codes to its latent vectors.

## 2. Method

- Data: training split of the art collection (hash-based split, as in all methods), 128 x 128 px, horizontal flip.
- Generator: latent vector (128-d, standard normal) -> linear layer to a 4 x 4 map -> five blocks of nearest-neighbour upsampling, two 3 x 3 convolutions, batch normalisation and LeakyReLU -> 3 x 3 convolution, tanh.
- Discriminator: mirrored, strided convolutions, spectral normalisation, no batch normalisation.
- Objective: hinge loss. Optional R1 penalty (`gan.r1_weight`).
- Differentiable augmentation (colour, translation, cutout) of real and generated images before the discriminator, for training on a few thousand images.
- An exponential moving average of the generator weights is saved and evaluated.

The latent has the same form as the Q1 VAE latent (128-d vector), so the Q1 bridge can address it without a change of its output dimension.

## 3. Evaluation

Every `gan.eval_every` steps: Frechet and kernel distance (DINOv2 ViT-S/14 CLS features) between `gan.n_eval_samples` samples and the test split of the art collection, and a grid of 64 samples from fixed latent vectors. Reference values measured with the same features and reference set in phase 1: training art 113.3; Q1 VAE reconstructions 208.5; Q1 VAE prior samples 974.7 (`docs/run_log.md`).

## 4. Protocol

`python methods/gan/train_gan.py --seed 0 [--channels 64] [--hours 9]`

The run stops at `gan.steps` or after `--hours`. The checkpoint `generator.pt` holds the averaged generator of the last evaluation step.

## 5. Limitations

- Unconditional; no bridge uses it yet.
- No mode-coverage measure beyond the Frechet and kernel distances.
- Architecture and hyperparameters are first settings.
