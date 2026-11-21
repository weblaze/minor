"""S4 bridge evaluation: does CAV-MAE pair each test image with its own track among all test tracks?

One image per test track from the mean of its window codes. CAV-MAE embeds the image; each image ranks all
test tracks by the similarity of their CAV-MAE audio embedding (mean over windows); both centred with the
centres stored in mapper.pt.

  top1, top10, median_rank   as methods/s3/identify.py (chance 1/n and 10/n)
  mean_own, mean_other       mean similarity of an image to its own track and to the other tracks

Structure, realism and the images come from methods/s1/evaluate.py on the same mapper.pt.

Usage:
  python methods/s4/bridge_identify.py --vae <gan run>/generator.pt --run runs/<s4 bridge run>
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from abstraction.encoders.cav_mae import IMAGE_MEAN, IMAGE_STD, load_cav_mae, smoke_cav_mae
from abstraction.generators import add_generator_argument, load_generator
from abstraction.models.mapper import Mapper
from abstraction.pipeline.cli import setup
from abstraction.utils.runlog import RunLog, set_seed

from bridge import to_cav
from common import aligned_codes, base_parser, cav_inputs, run_name


def main():
    parser = add_generator_argument(base_parser(__doc__), "gan")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    state = torch.load(Path(args.run) / "mapper.pt", map_location=device, weights_only=True)
    config["audio_code"]["variant"] = state["code"]
    embeddings = args.embeddings or config["cav_mae"]["embeddings"]
    config["run"] = {"bridge_run": args.run, "variant": state["variant"], "vae": str(args.vae)}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"bridge_identify_{state['variant']}", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    cav = smoke_cav_mae(device) if args.smoke else load_cav_mae(args.checkpoint or config["cav_mae"]["checkpoint"], device)
    cav_audio, _, _ = cav_inputs(embeddings, args.smoke, cav)
    codes, audio = aligned_codes(config, "test", cav_audio, args.smoke)
    hp = config["mapper"]
    mapper = Mapper(codes.shape[-1], math.prod(generator.latent_shape), hp["hidden"], hp["depth"]).to(device).eval()
    mapper.load_state_dict(state["model"])
    code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
    centre_a, centre_v = state["centre_audio"].to(device), state["centre_image"].to(device)
    mean = torch.tensor(IMAGE_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGE_STD, device=device)[None, :, None, None]

    with torch.no_grad():
        u = F.normalize(torch.from_numpy(audio.mean(axis=1)).to(device) - centre_a, dim=1)
        track_codes = torch.from_numpy(codes.mean(axis=1)).to(device)
        v = []
        for start in range(0, len(track_codes), 32):
            z = mapper((track_codes[start:start + 32] - code_mean) / code_std)
            images = generator.decode(z.view(-1, *generator.latent_shape) * generator.latent_scale)
            v.append(F.normalize(cav.embed_image(to_cav(images, mean, std)).float() - centre_v, dim=1))
        sims = torch.cat(v) @ u.T
    n = len(sims)
    own = sims.diagonal()
    rank = (sims > own[:, None]).sum(1) + 1
    results = {"variant": state["variant"], "steps": state["steps"], "n_test_tracks": n,
               "top1": float((rank == 1).float().mean()), "top10": float((rank <= 10).float().mean()),
               "median_rank": float(rank.float().median()), "chance_top1": 1 / n,
               "mean_own": float(own.mean()), "mean_other": float((sims.sum() - own.sum()) / (n * n - n))}
    log.save_json("results.json", results)
    print(results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
