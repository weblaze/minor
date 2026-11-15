# Q1 legacy code

Code that the shared pipeline replaced. It is kept unchanged for reference and is not imported by any stage.

| File | State | Replaced by |
|---|---|---|
| `data.py` | `methods/q1/data.py` as of commit `7cbaa49` (code transforms, window dataset, art datasets of Q1) | `abstraction/data/codes.py` (code transforms, bit-identical codes for Q1) and `abstraction/pipeline/data.py` |

The runs of phase 1 logged in `docs/run_log.md` were made with this file. The complete code of those runs is at the tags `s1-v0.1.0`, `q1-v0.1.0` and `f1-v0.1.0` and at the commits named in the run log.
