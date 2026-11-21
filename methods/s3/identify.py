"""S3 evaluation: can the reader tell, from an image alone, which of the test tracks it was made for?

For every test track one image is generated from the mean of its window codes ("track") and one from a
single random window ("window"). The reader embeds each image (clean, and after one fixed draw of the
training augmentation) and ranks all test tracks by the similarity of their audio embedding.

  top1, top10   share of images whose own track is ranked first / among the first ten (chance 1/n, 10/n)
  median_rank   median rank of the own track (1 = best)
  bits          log2(n) minus the cross-entropy (bits) of the reader's softmax at the own track; a lower
                bound on the information about the track carried by the image, clipped at 0

Structure, realism and the images themselves come from methods/s1/evaluate.py on the same mapper.pt.

Usage:
  python methods/s3/identify.py --vae <gan run>/generator.pt --run runs/<s3 run>
"""
import math
from pathlib import Path

import numpy as np
import torch

from abstraction.generators import load_generator
from abstraction.pipeline.cli import setup
from abstraction.pipeline.features import build_dino
from abstraction.utils.runlog import RunLog, set_seed

from common import Reader, augment, base_parser, build_mapper, code_dim, generate, load_codes, run_name


def scores(image_embed, audio_embed, temperature):
    sims = image_embed @ audio_embed.T
    n = len(sims)
    own = sims.diagonal()
    rank = (sims > own[:, None]).sum(1) + 1
    log_p = torch.log_softmax(sims / temperature, dim=1).diagonal()
    return {"top1": float((rank == 1).float().mean()), "top10": float((rank <= 10).float().mean()),
            "median_rank": float(rank.float().median()), "chance_top1": 1 / n,
            "bits": max(0.0, math.log2(n) + float(log_p.mean()) / math.log(2)), "max_bits": math.log2(n)}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--run", required=True, help="S3 run directory with mapper.pt and reader.pt")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    run = Path(args.run)
    state = torch.load(run / "mapper.pt", map_location=device, weights_only=True)
    config["audio_code"]["variant"] = state["code"]
    reader_state = torch.load(run / "reader.pt", map_location=device, weights_only=True)
    config["run"] = {"s3_run": str(run), "variant": state["variant"], "vae": str(args.vae)}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"identify_{state['variant']}", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    mapper = build_mapper(config, generator).to(device).eval()
    mapper.load_state_dict(state["model"])
    reader = Reader(code_dim(config), reader_state["embed_dim"], reader_state["hidden"]).to(device).eval()
    reader.load_state_dict(reader_state["model"])
    code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
    _, test = load_codes(config, "test", args.smoke)
    rng = np.random.default_rng(seed)
    inputs = {"track": test.mean(axis=1), "window": test[np.arange(len(test)), rng.integers(test.shape[1], size=len(test))]}
    g = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        audio_embed = reader.embed_audio((torch.from_numpy(test.mean(axis=1)).to(device) - code_mean) / code_std)
        results = {"n_test_tracks": len(test), "variant": state["variant"], "steps": state["steps"]}
        for name, codes in inputs.items():
            clean, seen = [], []
            for start in range(0, len(codes), 64):
                images, _ = generate(mapper, generator, torch.from_numpy(codes[start:start + 64]).to(device),
                                     code_mean, code_std)
                clean.append(reader.embed_images(dino(images).float()))
                seen.append(reader.embed_images(dino(augment(images, config["s3"]["aug"], g)).float()))
            for view, embed in (("clean", clean), ("augmented", seen)):
                results[f"{name}_{view}"] = scores(torch.cat(embed), audio_embed, config["s3"]["temperature"])
                print(name, view, results[f"{name}_{view}"], flush=True)
    log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
