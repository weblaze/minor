"""Image generators behind a common interface (see base.Generator).

A bridge loads its generator with `load_generator(args.generator, args.vae, config, device)`;
`--generator` selects the implementation and `--vae` names the checkpoint.
"""
from .base import Generator
from .vae import SpatialVAEGenerator, VectorVAEGenerator

GENERATORS = {
    "spatial_vae": SpatialVAEGenerator,
    "vector_vae": VectorVAEGenerator,
}


def add_generator_argument(parser, default):
    parser.add_argument("--generator", choices=sorted(GENERATORS), default=default,
                        help=f"image generator loaded from --vae (default: {default})")
    return parser


def load_generator(name, path, config, device):
    return GENERATORS[name].load(path, config, device)
