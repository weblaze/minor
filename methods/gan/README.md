# G-GAN: unconditional generator of abstract art

Status: second version. The first version (plain convolutional networks, data loaded from disk) failed on real data; the networks are now residual and the training images are held in memory. Results are recorded in `docs/run_log.md`. This document contains no results.

## 1. Purpose

Phase 2 replaces the VAE generator of phase 1 by an adversarial generator and later by diffusion generators, under the same bridges. G-GAN is the generator only: it is trained on the art collection without any audio input. A bridge then maps audio codes to its latent vectors.

## 2. Method

- Data: training split of the art collection (hash-based split, as in all methods), 128 x 128 px, horizontal flip.
- Generator: latent vector (128-d, standard normal) -> linear layer to a 4 x 4 map -> five residual blocks (batch normalisation, ReLU, nearest-neighbour upsampling, two 3 x 3 convolutions, 1 x 1 skip) -> batch normalisation, ReLU, 3 x 3 convolution, tanh.
- Discriminator: five residual blocks with average pooling and one without, sum pooling, linear output; no batch normalisation. Two regularisers are available: spectral normalisation (`gan.spectral_norm`) or an R1 penalty on real images (`gan.r1_weight`, applied every `gan.r1_every` steps).
- Objective: hinge loss.
- Differentiable augmentation (colour, translation, cutout) of real and generated images before the discriminator, for training on a few thousand images.
- An exponential moving average of the generator weights is saved and evaluated.
- `--amp` runs the forward passes in mixed precision; steps with the R1 penalty stay in float32. Not validated by a full run; the run exits at the first non-finite loss. A GTX 1650 returns non-finite values in float16, so the option cannot be checked on that card.
- Every evaluation writes `checkpoint.pt` (generator, discriminator, averaged generator, both optimisers, step); `--resume <checkpoint.pt>` continues from it in a new run directory.

The latent has the same form as the Q1 VAE latent (128-d vector), so the Q1 bridge can address it without a change of its output dimension.

## 3. Evaluation

Every `gan.eval_every` steps: Frechet and kernel distance (DINOv2 ViT-S/14 CLS features) between `gan.n_eval_samples` samples and the test split of the art collection, and a grid of 64 samples from fixed latent vectors. Reference values measured with the same features and reference set in phase 1: training art 113.3; Q1 VAE reconstructions 208.5; Q1 VAE prior samples 974.7 (`docs/run_log.md`).

## 4. Protocol

`python methods/gan/train_gan.py --seed 0 [--channels 64] [--hours 9] [--no-spectral-norm --r1 10]`

The run stops at `gan.steps` or after `--hours`. The checkpoint `generator.pt` holds the averaged generator of the last evaluation step.

## 5. Limitations

- Unconditional; no bridge uses it yet.
- No mode-coverage measure beyond the Frechet and kernel distances.
- Architecture and hyperparameters are first settings. Results so far (`docs/run_log.md`, 2026-11-16): the R1 variant reaches FD 682.0 at 13651 steps and is under-trained; the spectral-normalisation variant does not train (FD about 1500). Step time on a T4 in float32: 2.1 s at base_channels 64, batch 64.
