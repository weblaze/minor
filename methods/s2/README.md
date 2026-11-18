# S2: blind matching by shape

S2 asks whether the audio space and the art space have compatible shapes, without any pair and without training: the cosine distance matrix of the test tracks (MERT audio codes) is matched to that of the training paintings (DINOv2 features) by entropic Gromov-Wasserstein (GW), which looks for a coupling under which pairs of tracks are as far apart as the pairs of paintings they are coupled with. Planned in the audit of 2026-11-10 as a measurement first ("is there any shared shape to find?"), then as a bridge: the coupling assigns a painting to each track.

Prior work: Gromov-Wasserstein alignment of embedding spaces (Alvarez-Melis and Jaakkola, 2018), unpaired vision-language matching by quadratic assignment ("It's a (Blind) Match!", Schnaus et al., CVPR 2025), Gromov-Wasserstein GANs (Bunne et al., ICML 2019).

## Stage

| Stage | Output |
|---|---|
| `match.py` | `results.json` per audio code: GW objective of each restart and of the selected match, GW of the same tracks against two reference spaces, the assignment scored as the F1 painting block, orientation of the leading axes per restart, agreement between restarts; `assignments_<code>.npz` (all restarts) and `features_<code>.npz` (selected match) |

```bash
python scripts/run.py s2 match
python scripts/run.py s4 d_s4 --runs runs/s2_eval_match_*          # CAV-MAE score of each restart (D-S4)
python scripts/run.py s4 probe --assignments "runs/s2_eval_match_*/features_*.npz"
```

## Solver

Entropic GW with square loss (Peyre, Cuturi and Solomon, 2016), implemented in `common.py` with PyTorch (log-domain Sinkhorn). Distances are divided by their mean off-diagonal value. The regularisation starts at `epsilon_start` and decays geometrically to `epsilon`. Settings chosen on the mean code before the run (one restart each, test tracks against training paintings): at `epsilon` 0.01 without decay the coupling is close to uniform (mean share of a track's mass on its largest painting 0.003), at 0.0005 with decay from 0.05 it is 0.156, close to the limit set by the uniform marginals (789 tracks, 4666 paintings), and the objective is lowest (0.0570 against 0.0742). With the decay, restarts from random couplings end at nearly the same solution (81 to 97 % of tracks receive the same painting).

## References for the objective

GW values depend on the distribution of distances in both spaces, so the objective is reported against two references, each matched to the same test tracks:

- `gaussian`: samples of a Gaussian with the mean and covariance of the painting features, as many as there are paintings. The same second-order statistics, no further structure. A real art objective below this reference means the tracks fit the shape of the art collection better than a structureless cloud of the same spread.
- `audio`: random training tracks, as many as there are paintings. The same space as the test tracks; the objective a perfect shape match approaches with this sample size.

## Reading the result

- The match is selected by its objective only (no label, no audio-image pair).
- `painting` is directly comparable with the F1 painting block and with the S4 retrieval bridge (Mantel r of the audio codes against DINOv2 features of the assigned paintings, FD, distinct paintings).
- GW is invariant to isometries of either space, so a mirrored audio space gives the same objective. The orientation of the leading axes in the selected match (`painting.corr`, `orientation_counts`) is therefore decided by asymmetries of the two shapes alone, not by the initialisation of a network as in F1.
