# F1: mapping audio codes to image features

Status: implemented; run on real data for seed 0 only. Results are in `docs/run_log.md`. This document contains no results.

## 1. Purpose

S1 and Q1 train a mapping through an image decoder and a feature extractor. Their first runs (2026-11-12) left three questions that those methods cannot separate:

1. how much structure a mapping can carry for a given audio code,
2. how much of it survives when outputs must be real images,
3. how much is lost in the image generator.

F1 removes the generator. The mapping outputs a DINOv2 feature vector, and a real painting is assigned to it by nearest-neighbour search. Question 1 is measured on the feature vectors, question 2 on the retrieved paintings. Question 3 is the difference to S1 and Q1 under the same audio code.

F1 is used first as a diagnostic. It selects existing paintings and generates no image. A generator conditioned on the feature vector is a possible second stage and is not implemented.

## 2. Method

```
track -> 3 windows of 10 s -> MERT-v1-95M (frozen) -> 13 x 768 per window
      -> code transform (fitted on training tracks) -> a in R^d
a -> M (MLP) -> f in R^384                         (direct output)
f -> nearest training painting by cosine similarity of DINOv2 CLS features   (painting output)
```

Audio code (`audio_code`, shared implementation in `abstraction/data/codes.py`): `mean`, `standardised` or `whitened`, over all 13 MERT layers or over a subset (`audio_code.layers`).

Objective, per batch of tracks (one window per track) and a batch of training-art features:

```
L = L_dist + lambda_struct * L_struct
L_dist    sliced Wasserstein distance (or energy distance) between outputs and art features,
          both divided by the standard deviation of the training-art features
L_struct  1 - Pearson(D_a, D_f); D_a, D_f: cosine distances between audio codes and between outputs
```

No decoder and no feature extractor is in the training loop; DINOv2 is applied once to the art collection and cached.

## 3. Variants and controls

| Name | Definition | Tests |
|---|---|---|
| `f1` | L_dist + lambda_struct * L_struct | full method |
| `f0` | L_dist only | structure given by the distribution term and the smoothness of M alone |
| `shuffled` | as `f1`, audio distance matrix of each batch permuted | whether the structure term needs the true audio distances |
| random M | untrained mapping (`evaluate.py` without `--mapper`) | distance preservation of a random network |

## 4. Diagnosis (`diagnose.py`)

For every code in `diagnose.codes` a mapping is trained and scored at these stages (r: Pearson correlation between cosine distances of audio codes and of image features):

| Stage | Definition | Difference to the previous stage |
|---|---|---|
| `train_direct` | outputs for a subset of training tracks | |
| `test_direct` | outputs for the test tracks | generalisation gap of M |
| `test_painting` | nearest training painting for each test output | loss from restricting outputs to real paintings |
| `gallery_<n>` | as above with n randomly chosen paintings | dependence of that loss on the number of paintings |
| `oracle_painting` | mapping trained on the test tracks, then nearest painting | reference: what map-then-retrieve reaches when M has seen the tracks |

Each code is also described by its participation ratio and the coefficient of variation of its pairwise distances.

## 5. Evaluation (`evaluate.py`)

Test tracks, window-averaged codes. For the direct output and the painting output: Mantel correlation (Pearson and Spearman, permutation p-value), k-nearest-neighbour overlap, trustworthiness, Frechet and kernel distance to test-art features, mean pairwise distance, participation ratio. For the painting output additionally: number of distinct paintings, and the Pearson correlation between audio distances and distances of feature sets computed from the pixels of the paintings (colour mean and standard deviation, colour histogram, 8 x 8 layout, radial power spectrum). These do not depend on DINOv2.

## 6. Protocol

1. `python scripts/extract_mert.py --source <fma zip or directory>`
2. `python methods/f1/diagnose.py --seed 0`
3. `python methods/f1/train_mapper.py --variant <name> --seed <n>` for every variant
4. `python methods/f1/evaluate.py --mapper <mapper.pt>`; `python methods/f1/evaluate.py --seed <n>` for random M

Splits as in S1 and Q1 (MD5 hash of the track identifier or file name; 80 / 10 / 10). The retrieval gallery is the training split of the art collection; test art is used as the reference of the distribution scores only. The checkpoint of the final step is evaluated.

One mapping run takes about one minute on a 4 GB GPU.

## 7. Files

| File | Output |
|---|---|
| `train_mapper.py` | `mapper.pt`, `metrics.jsonl`, `data.json` |
| `evaluate.py` | `results.json`, `features.npz`, `samples.png` (retrieved paintings of the first 64 test tracks) |
| `diagnose.py` | `diagnosis.json` |

`--smoke` replaces codes and art features with synthetic data. It verifies that the code runs; its numbers have no meaning.

## 8. Limitations

- The painting output cannot leave the training collection (4666 paintings in the current data); several tracks can receive the same painting.
- The distribution term does not place the direct outputs on the art distribution (exploratory entry of 2026-11-12 in `docs/run_log.md`). The direct outputs are intermediate quantities, not results.
- Structure is measured against the audio code the mapping was trained on. Codes differ in what distances they assign, so scores under different codes are not comparable without a common reference.
- The oracle stage is a reference, not a bound: a different retrieval rule or a direct assignment method could exceed it.
