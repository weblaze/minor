import argparse

import torch

from abstraction.utils.config import load_config

SMOKE_HELP = "synthetic data, randomly initialised DINOv2, few steps; checks the code path only"


def base_parser(description, config_path, smoke_help=SMOKE_HELP):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(config_path))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="appended to the run directory name")
    parser.add_argument("--code", choices=("mean", "standardised", "whitened"), default=None,
                        help="audio code variant; overrides audio_code.variant")
    parser.add_argument("--smoke", action="store_true", help=smoke_help)
    return parser


def setup(args):
    """Config with the command-line overrides applied, and the device."""
    config = load_config(args.config)
    if args.seed is not None:
        config["system"]["seed"] = args.seed
    if args.tag:
        config["system"]["tag"] = args.tag
    if args.code:
        config["audio_code"]["variant"] = args.code
    if getattr(args, "layers", None):
        config["audio_code"]["layers"] = args.layers
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    return config, device


def run_name(method, stem, smoke):
    return f"{method}_{stem}" + ("_smoke" if smoke else "")
