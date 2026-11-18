# G-GAN legacy code

Code that the shared package replaced. It is kept unchanged for reference and is not imported by any stage.

| File | State | Replaced by |
|---|---|---|
| `train_gan.py` | `methods/gan/train_gan.py` as of commit `c5a7670` (generator classes `UpBlock` and `Generator` defined in the script) | the same script importing `Generator` from `abstraction/models/gan.py`, where the two classes were moved unchanged so that bridges can load a trained generator (`abstraction/generators/gan.py`) |

The runs `gan_sn_*`, `gan_r1_*` and `gan_r1amp_*` logged in `docs/run_log.md` up to 2026-11-18 were made with this file. Their `generator.pt` and `checkpoint.pt` load into the moved classes, since the parameter names are unchanged.
