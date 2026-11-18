# Pipeline

How the code is laid out, how a stage is run, how a new part is added, how an experiment becomes a Kaggle job, and how to check that a change leaves the numbers unchanged.

## Layout

```
abstraction/                 shared package (pip install -e .)
  data/codes.py              audio-code transforms over MERT layers: mean, standardised, whitened, layer subsets
  encoders/                  frozen MERT, DINOv2 and CAV-MAE (S4)
  generators/                image generators behind one interface: spatial_vae (S1), vector_vae (Q1), gan (G-GAN)
  pipeline/                  front matter shared by the methods
    cli.py                   parser (--config --seed --tag --code --smoke), config setup, run names
    data.py                  MERT layers and splits, codes_of, art datasets, synthetic smoke data, window sampling
    features.py              DINOv2 builder, features of a dataset
    scores.py                structure and distribution scores, genre leakage
  losses/, metrics/, models/, utils/
methods/<method>/            one folder per bridge: common.py (method-specific helpers), stage scripts, config.yaml, README.md
scripts/run.py               front door
scripts/smoke.py             smoke pipeline of every method
scripts/regression.py        numerical regression check over the smoke pipelines
scripts/results.py           table of all evaluation runs
scripts/kaggle/              launch.py, kernel.py, make_job.py, jobs/, experiments/
```

A method folder keeps what is specific to the method: its losses and variants, its VAE training, its own code transform where it differs from the shared one (S1, see below), and its evaluation output format. Everything that two or more methods used in identical form lives in `abstraction/pipeline/`.

S1 keeps its own code transform (`methods/s1/data.py`). Its `standardised` and `whitened` statistics are computed with `numpy.mean`/`numpy.std` in float64, the shared transform with chunked sums of x and x²; the results differ in the last bits, so replacing it would change S1 runs with those codes. Q1 uses the shared transform, which gives bit-identical codes for Q1.

Code replaced by the shared pipeline is not deleted. It is moved unchanged to `methods/<method>/legacy/` with a README naming its replacement; nothing imports it. At present: `methods/q1/legacy/data.py`, `methods/gan/legacy/train_gan.py`.

## Front door

```bash
python scripts/run.py --list                              # methods and stages, with the first line of each docstring
python scripts/run.py <method> <stage> [args...]          # = python methods/<method>/<stage>.py [args...]
python scripts/run.py q1 train_mapper --vae runs/<vae run>/vae.pt --variant q1 --seed 0
```

A stage is any script in `methods/<method>/` other than `common.py` and `data.py`. The stage runs in the same process with the same arguments; starting the script directly works as before.

Options common to every stage: `--config`, `--seed`, `--tag`, `--code {mean,standardised,whitened}`, `--smoke`. F1 stages also take `--layers`. S1 and Q1 stages that load a generator take `--generator` (default: the VAE of the method) and `--vae` (the checkpoint).

## Adding an audio code

1. Add the variant to `CODE_VARIANTS` and to `fit_code_transform` / `apply_code_transform` in `abstraction/data/codes.py`. The transform is fitted on training tracks only and must be storable with `transform_to_state` (a dict of strings and arrays).
2. Add the name to the `--code` choices in `abstraction/pipeline/cli.py`.
3. Q1 and F1 pick it up through the shared transform. S1 needs the variant in `methods/s1/data.py:load_codes` and a code dimension in `methods/s1/common.py:code_dim`.
4. Add it to the list of codes in `methods/f1/config.yaml` (`diagnose.codes`) if the F1 diagnosis should cover it.

## Adding a bridge

1. Create `methods/<name>/` with `config.yaml` (merged over `configs/base.yaml`), `common.py` and one script per stage.
2. In `common.py`, define `CONFIG_PATH` and wrap the shared parser and run names:
   ```python
   from abstraction.pipeline import cli
   CONFIG_PATH = Path(__file__).with_name("config.yaml")
   def base_parser(description):
       return cli.base_parser(description, CONFIG_PATH)
   def run_name(stem, smoke):
       return cli.run_name("<name>", stem, smoke)
   ```
3. In each stage: `config, device = setup(args)`, `set_seed(seed)`, a `RunLog` named by `run_name(...)`, data from `abstraction.pipeline.data` (`load_layers`, `codes_of`, `art_dataset`), DINOv2 from `abstraction.pipeline.features.build_dino`, and every stage must support `--smoke` (synthetic data, few steps).
4. Add the smoke pipeline to `PIPELINES` in `scripts/smoke.py` (it is reused by `scripts/regression.py`).
5. Write `results.json` from the evaluation stage in a form `scripts/results.py` reads (see below).

## Adding a generator

A generator implements `abstraction.generators.base.Generator`:

| Member | Meaning |
|---|---|
| `load(path, config, device)` | class method; returns the frozen generator from a checkpoint |
| `decode(z)` | latents `(b, *latent_shape)` to images `(b, 3, h, w)` in [-1, 1] |
| `latent_shape` | shape of one latent without the batch dimension; the mapping outputs `prod(latent_shape)` values |
| `latent_scale` | S1 maps into units of the latent standard deviation and multiplies by this before decoding; Q1 decodes the mapping output as it is |
| `encode(x)` | (posterior mean, log-variance); only generators with an encoder |

Register the class in `GENERATORS` in `abstraction/generators/__init__.py`; it then appears as a `--generator` choice of the S1 and Q1 stages. The bridges call only these members.

What `encode` is used for: S1 `L_dist` compares mapped latents with latents of real art, Q1 `L_lat` with samples of their posteriors, the evaluation scripts report VAE reconstructions as a reference, and the Q1 gate decodes posterior means.

A generator without an encoder sets `has_encoder = False` (G-GAN; most diffusion decoders). By the decision of 2026-11-18, S1 and Q1 then use samples of the prior N(0, I) (`prior_sample`) where they would use latents of real art: S1 `L_dist` compares mapped latents with prior samples (as many as there are training and validation paintings, fixed by the seed), Q1 `L_lat` with fresh prior samples in each batch (posterior mean 0, log-variance 0). The evaluation scripts omit the reconstruction reference, and Q1 does not require a gate result. `gate.py` and `baselines.py` of Q1 still require an encoder.

```bash
python scripts/run.py q1 train_mapper --generator gan --vae runs/<gan run>/generator.pt --seed 0
python scripts/run.py s1 train_mapper --generator gan --vae runs/<gan run>/generator.pt --seed 0
```

## Adding an evaluation

An evaluation stage writes `results.json` into its run directory, whose name contains `_eval_`, plus `features.npz` with `track_ids`, `audio_codes` and `image_features` (or `painting_features`) for the agreement analysis. `scripts/results.py table` reads:

- a block with `structure` (`mantel_pearson`, `mantel_spearman`, `knn_overlap`, `trustworthiness`) and `image_distribution` (`frechet_distance`, `participation_ratio`; or the S1 form with `generated.frechet_distance` and `participation_ratio_generated`); the whole file is one block, or the blocks are under `direct` and `painting` (F1);
- `variant`, `code_variant`, `mapper_seed` at the top level, if present;
- the seed, tag and code from `config.yaml` (`system.seed`, `system.tag`, `run.code` or `run.code_variant` or `audio_code.variant`) and the commit from `meta.json`.

`abstraction.pipeline.scores.full_scores` returns the `structure` and `image_distribution` blocks in the readable form. Smoke runs (`_smoke` in the name) are skipped.

## Experiments and Kaggle jobs

An experiment file (`scripts/kaggle/experiments/<name>.yaml`) lists the job metadata, setup commands and chains of stage commands:

```yaml
description: Q1 variant q1 on the mean code, seeds 1 and 2. One seed per GPU.
name: abstraction-q1-mean-seeds12
branch: pipeline
kernels: ["{user}/abstraction-features", "{user}/abstraction-q1-vae"]
inputs: {MERT: mert_embeddings, ART_ZIP: abstract_art.zip, Q1_VAE: q1_vae_seed0_20261110T155658Z}
setup:
  - python -m zipfile -e {ART_ZIP} datasets/
  - ln -s {MERT} datasets/mert_embeddings
chains:
  - - q1 train_mapper --seed 1 --code mean --vae {Q1_VAE}/vae.pt --ignore-gate
    - q1 evaluate --seed 1 --vae {Q1_VAE}/vae.pt --mapper runs/q1_mapper_q1_seed1_*/mapper.pt
  - - q1 train_mapper --seed 2 --code mean --vae {Q1_VAE}/vae.pt --ignore-gate
    - q1 evaluate --seed 2 --vae {Q1_VAE}/vae.pt --mapper runs/q1_mapper_q1_seed2_*/mapper.pt
```

```bash
python scripts/kaggle/make_job.py scripts/kaggle/experiments/<name>.yaml          # writes scripts/kaggle/jobs/<name>.yaml
python scripts/kaggle/make_job.py <experiment> --print                              # prints the job
python scripts/kaggle/make_job.py <experiment> --compare scripts/kaggle/jobs/<job>.yaml
python scripts/kaggle/launch.py push scripts/kaggle/jobs/<name>.yaml
```

Chain commands `<method> <stage> args` become `python methods/<method>/<stage>.py args`; other commands are copied. Chain i runs on GPU i mod 2; all chains of a GPU run in one background shell with `set -e`, logging to `runs/gpu<i>.log`; the job waits for both GPUs and fails if either failed. `{NAME}` is an input; other braces are escaped for the kernel's `str.format`, so shell variables are written normally. The job runs the head of `branch` on `origin`, so the branch has to be pushed. Since all methods live on one branch, a job can mix methods without a second clone.

`scripts/kaggle/experiments/q1_mean_seeds12.yaml` reproduces `jobs/q1_mean_seeds12.yaml`; `--compare` reports no difference in metadata, setup or per-GPU commands. The only textual difference is the exit line: `exit $((r0 != 0 || r1 != 0))` instead of `exit $((r0 + r1))`, which would report success if the two return codes summed to a multiple of 256.

## Regression check

```bash
python scripts/regression.py --save runs/regression_baseline.json      # on the code before the change
python scripts/regression.py --compare runs/regression_baseline.json   # after the change
python scripts/regression.py --compare <file> --methods q1             # one method only
```

The script runs the smoke pipeline of every method (seed 0, the command lists of `scripts/smoke.py`), forces the CPU (`CUDA_VISIBLE_DEVICES=-1`), and collects every number from `results.json`, `diagnosis.json`, `gate.json`, `diagnostics.json`, `result.json` and the last line of `metrics.jsonl` of the run directories created by each step (`elapsed_s` is ignored). `--compare` prints every value that changed beyond a relative tolerance of 1e-5, is missing or is new, and exits non-zero if there is any. A refactoring is accepted only with zero differences. The full run takes 20 to 25 minutes on a laptop CPU; `--methods` restricts it.

The check covers the smoke configurations only: default code variant of each method, synthetic data, randomly initialised DINOv2. Code paths that the smoke pipelines do not reach (real data loading, `--genres`, non-default code variants of S1) are not covered and have to be checked by reading the change.
