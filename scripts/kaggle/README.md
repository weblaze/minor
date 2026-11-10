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
