# Evaluation Suite

Shared metrics for comparing approaches. All scripts write JSON to
`evaluation/results/` so runs are comparable across approaches and configs.

| Script | Measures | When to run |
|---|---|---|
| `reconstruction.py` | VAE reconstruction quality (visual, side-by-side) | After approach 01 pretrain |
| `retrieval.py` | Recall@K: audio→generated-image retrieval vs random | Approach 01 gate + every trained model |
| `clip_alignment.py` | Mood-profile cosine between audio and its generation, vs shuffled baseline — **the headline "conditioning works" number** | Every trained model; pseudo vs random pairing comparison |
| `fid.py` | Distribution distance to the abstract-art training set (lower = better) | Approach 02 onwards |
| `lpips_diversity.py` | Pairwise perceptual diversity (mode-collapse detector) | Same-song batch + across-song batch |

Optional dependencies: `pip install clean-fid lpips open_clip_torch`
(or `pip install -e .[eval]`).

## Standard protocol per trained model

```bash
# generate a test set: one image per held-out song (e.g. 50 songs across genres)
python approaches/02_latent_diffusion_clap/sample.py --audio <song> ...

python evaluation/clip_alignment.py --audio_dir <songs> --image_dir <generations>
python evaluation/retrieval.py      --audio_dir <songs> --image_dir <generations>
python evaluation/fid.py            --generated <generations>
python evaluation/lpips_diversity.py --image_dir <generations>
```

The human study (Likert "does this image match the mood of the audio?",
conditioned vs dream vs random baseline, ≥20 raters) is designed in the
research docs — run it once approach 02 passes the automatic metrics.
