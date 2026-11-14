"""F1: train the mapping M from audio codes to DINOv2 image features.

Variants:
  f1        L_dist + lambda_struct * L_struct
  f0        L_dist only
  shuffled  as f1, with the audio distance matrix of each batch permuted

Usage:
  python methods/f1/train_mapper.py --variant f1 --seed 0
"""
import torch

from abstraction.data.codes import transform_to_state
from abstraction.utils.runlog import RunLog, set_seed

from common import VARIANTS, base_parser, codes_of, load_art, load_layers, run_name, setup, train_mapper


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="f1")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    steps = args.steps or (60 if args.smoke else config["mapper"]["steps"])
    set_seed(seed)
    config["run"] = {"variant": args.variant, "steps": steps}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"mapper_{args.variant}", args.smoke), config, seed)

    _, layers, splits = load_layers(config, args.smoke)
    code = config["audio_code"]
    codes, transform = codes_of(layers, splits, code["variant"], code["whiten_dim"], code["layers"])
    _, art, art_splits = load_art(config, device, args.smoke)
    log.save_json("data.json", {"n_train_tracks": int((splits == "train").sum()),
                                "n_val_tracks": int((splits == "val").sum()),
                                "n_train_art": int((art_splits == "train").sum()),
                                "code_dim": int(codes.shape[-1])})

    mapper, code_mean, code_std = train_mapper(
        config, codes[splits == "train"], codes[splits == "val"],
        art[art_splits == "train"], art[art_splits == "val"], device, args.variant, steps, seed, log)
    torch.save({"model": mapper.state_dict(), "code_mean": code_mean.cpu(), "code_std": code_std.cpu(),
                "code_transform": transform_to_state(transform), "variant": args.variant, "seed": seed,
                "steps": steps}, log.dir / "mapper.pt")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
