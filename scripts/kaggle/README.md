# Running jobs on Kaggle

Training and feature extraction run in Kaggle script kernels. The laptop is used to write code, launch jobs and read results.

## Requirements

- `kaggle` command-line client (`pip install kaggle`).
- API token at `~/.kaggle/kaggle.json` (Kaggle account settings, section API, "Create New Token").
- The Kaggle account has to be phone-verified; internet access inside kernels depends on it.
- The branch named in the job is pushed to `origin`. The kernel clones the public repository at that commit.

## Use

```bash
python scripts/kaggle/launch.py push   scripts/kaggle/jobs/<job>.yaml
python scripts/kaggle/launch.py status scripts/kaggle/jobs/<job>.yaml
python scripts/kaggle/launch.py fetch  scripts/kaggle/jobs/<job>.yaml
```

`push` creates or replaces the private kernel `<user>/<slug of name>` and starts it. `fetch` downloads the kernel output to `runs/kaggle/<slug>/`. `push --dry-run DIR` writes the kernel files to `DIR` without contacting Kaggle.

## Writing a job from an experiment

Jobs that run stages on both GPUs of a session are generated from a short experiment file instead of being written by hand:

```bash
python scripts/kaggle/make_job.py scripts/kaggle/experiments/<name>.yaml           # writes scripts/kaggle/jobs/<name>.yaml
python scripts/kaggle/make_job.py scripts/kaggle/experiments/<name>.yaml --print   # prints it instead
python scripts/kaggle/make_job.py <experiment> --compare scripts/kaggle/jobs/<job>.yaml
```

The experiment file has the job keys `name`, `branch`, `gpu` (default true), `datasets`, `kernels` and `inputs`, a `description` (written as the header comment), `setup` (commands run first) and `chains`. A chain is a list of commands run one after another on one GPU; a command is written as for `scripts/run.py` (`q1 train_mapper --seed 1 ...`) and becomes `python methods/q1/train_mapper.py --seed 1 ...`. Chain i runs on GPU i mod `--gpus` (default 2); each GPU logs to `runs/gpu<i>.log`; the job waits for every GPU and fails if any failed. `{NAME}` refers to an input; every other brace is escaped for `str.format` in the kernel, so `${var}` is written as in a shell. An existing job file is not overwritten without `--force`.

`experiments/q1_mean_seeds12.yaml` reproduces `jobs/q1_mean_seeds12.yaml` (check with `--compare`). The generated exit line is `exit $((r0 != 0 || r1 != 0))`; the hand-written `exit $((r0 + r1))` would succeed if the return codes summed to 256. Because all methods are on one branch, a generated job can run S1 and Q1 stages side by side without the second clone used in `jobs/code_swap.yaml`. See `docs/pipeline.md`.

## What a kernel does

`kernel.py` clones the repository at the commit of the job, installs it, links `runs/` to `/kaggle/working/runs`, resolves the job inputs under `/kaggle/input`, and runs the commands in order, stopping at the first failure. It writes `job.json` (commit, resolved inputs, return code and duration per command) and `logs/NN.log` to the kernel output.

Output of one kernel is passed to another by listing the first under `kernels` and naming the file or directory under `inputs`.

## Data sources

| Data | Kaggle source | Notes |
|---|---|---|
| Music | dataset `imsparsh/fma-free-music-archive-small-medium`, directory `fma_small` | mirror of FMA-small (Defferrard et al. 2017): 8,000 clips of 30 s, 8 genres. The job prints the number of mp3 files found. |
| Art | dataset `steubk/wikiart`, style directories `Abstract_Expressionism`, `Action_painting`, `Color_Field_Painting`, `Minimalism` | mirror of the WikiArt collection organised by style. `build_art_set.py` writes the 128 px images and `manifest.csv`. |

Neither mirror has been checked against the original distribution. The counts printed by the first run are recorded with the results. DELAUNAY is not included.

## Jobs

| Job | Branch | Output |
|---|---|---|
| `features.yaml` | `base` | `abstract_art.zip`, `art_manifest.csv`, `mert_embeddings/` (float16, 3 windows x 13 layers x 768 per track) |

Limits of the free tier (not confirmed on an official page): about 30 GPU hours per week, sessions of at most 12 hours.
