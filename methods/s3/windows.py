"""S3 window consistency: do different 10 s windows of the same track give images that look alike?

For every test track one image is generated from each of its windows (3 per track). Each image is embedded
by DINOv2 (the encoder used in training) and by DINO v1 (ViT-B/16, CLS; not used in training), and compared
with the images of all other windows by cosine similarity.

  window_top1   share of images whose most similar other image comes from the same track
                (chance (w - 1) / (n w - 1))
  same_rank     median rank of the closest same-track image among all other images (1 = best)
  ratio         mean cosine distance between same-track images / mean distance between images of different tracks
                (1 = no consistency, 0 = identical images per track)

The same measures on the MERT window codes themselves (cosine) give the consistency the audio offers.

Usage:
  python methods/s3/windows.py --vae <gan run>/generator.pt --run runs/<s3 run> [--run ...]
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.generators import load_generator
from abstraction.pipeline.cli import setup
from abstraction.pipeline.features import build_dino
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_mapper, generate, load_codes, run_name
from second_reader import DinoV1


def consistency(x, n_windows):
    x = F.normalize(x.float(), dim=1)
    sims = x @ x.T
    n = len(x)
    track = torch.arange(n, device=x.device) // n_windows
    same = track[:, None] == track[None, :]
    eye = torch.eye(n, dtype=torch.bool, device=x.device)
    sims_off = sims.masked_fill(eye, -2.0)
    nearest = sims_off.argmax(dim=1)
    order = sims_off.argsort(dim=1, descending=True)
    ranks = (same & ~eye).gather(1, order).float().argmax(dim=1) + 1
    distance = 1 - sims
    return {"window_top1": float((track[nearest] == track).float().mean()),
            "chance_top1": (n_windows - 1) / (n - 1),
            "same_rank": float(ranks.float().median()),
            "ratio": float(distance[same & ~eye].mean() / distance[~same].mean())}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--run", required=True, nargs="+", help="S3 run directories with mapper.pt")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    config["run"] = {"s3_runs": args.run, "vae": str(args.vae)}
    log = RunLog(config["paths"]["runs_dir"], run_name("windows", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    encoders = {"dinov2": build_dino(config, device, args.smoke)}
    if not args.smoke:
        encoders["dino_v1"] = DinoV1(device)
    codes = {torch.load(Path(r) / "mapper.pt", map_location="cpu", weights_only=True)["code"] for r in args.run}
    if len(codes) != 1:
        raise SystemExit(f"runs use different codes: {codes}")
    config["audio_code"]["variant"] = codes.pop()
    _, test = load_codes(config, "test", args.smoke)
    n_tracks, n_windows = test.shape[:2]
    windows = test.reshape(n_tracks * n_windows, -1)
    results = {"n_test_tracks": int(n_tracks), "n_windows": int(n_windows),
               "audio": consistency(torch.from_numpy(windows).to(device), n_windows), "runs": {}}
    print("audio", results["audio"], flush=True)

    for run in map(Path, args.run):
        state = torch.load(run / "mapper.pt", map_location=device, weights_only=True)
        mapper = build_mapper(config, generator).to(device).eval()
        mapper.load_state_dict(state["model"])
        code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
        features = {name: [] for name in encoders}
        with torch.no_grad():
            for start in range(0, len(windows), 32):
                images, _ = generate(mapper, generator, torch.from_numpy(windows[start:start + 32]).to(device),
                                     code_mean, code_std)
                for name, encoder in encoders.items():
                    features[name].append(encoder(images).float())
        block = {name: consistency(torch.cat(f), n_windows) for name, f in features.items()}
        results["runs"][run.name] = {"variant": state["variant"], **block}
        print(run.name, block, flush=True)
        log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
