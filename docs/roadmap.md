# Roadmap

Set by the project owner on 2026-11-14. Names of bridges, generators and encoders follow the scheme used in the planning notes: a bridge connects music to images, a generator produces the image.

## Phase 1: structure transfer (S1, Q1, F1) with a VAE generator

Status: experiments complete once the Q1 seed and control runs of 2026-11-14 are fetched and logged.

Content:

- S1: mapping into a spatial VAE latent, latent distribution term, distance-correlation term.
- Q1: vector VAE, feature-space distribution term, global and local structure terms, diagnostics and baselines.
- F1: mapping into DINOv2 feature space with nearest-painting retrieval; diagnosis, experiment grid, orientation across seeds, retrieval limit.
- Audio codes: mean, standardised, whitened; relation of the leading axes to FMA metadata.

Closing steps:

1. Fetch and log `abstraction-q1-mean-seeds12` and `abstraction-q1-mean-controls`.
2. Assemble methods, data, protocol deviations, results and negative results from `docs/run_log.md` into a paper outline.
3. Draft the paper.

Not done in phase 1 and stated as limitations: evaluation with a second image encoder, human evaluation, DEAM valence and arousal analysis, psychophysics probes, seeds beyond those listed in the run log.

## Phase 2: bridges without human involvement, GAN then diffusion

Order:

1. S4: bridge from natural audio-visual co-occurrence (video of the physical world; no text, no human pairing of music and images).
2. S3: bridge by information maximisation (the generator produces images from which the audio code can be recovered).

Generators, for each bridge in this order:

1. G-GAN.
2. G-Diff and other diffusion or flow-based generators.

Comparison at the end of the phase: G-VAE (phase 1), G-GAN and G-Diff under the same bridges, on structure, distance to the art distribution and reproducibility across seeds.

## Phase 3: human involvement

All work that uses human judgement or human-made pairings is collected in this phase:

- S5: bridge from human-made co-occurrence (album covers, film scores).
- S6: human preference as a training signal.
- Human evaluation of the phase 1 and phase 2 systems (two-alternative forced choice; includes the orientation question raised by the F1 seed study).

## Later

Creative extensions (unconditional sampling, several images per track, temporal sequences), image content beyond abstract art, and a single `generate(audio) -> image` interface with bridge and generator selected by configuration.
