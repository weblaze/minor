import sys
import numpy as np
sys.path.insert(0, "methods/s1")
from abstraction.utils.config import load_config
from abstraction.data.codes import apply_code_transform, fit_code_transform
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of
from data import load_codes
config = load_config("methods/s1/config.yaml")
config["audio_code"]["variant"] = "standardised"
ids, codes = load_codes(config, "test")
all_ids, layers = load_shards(config["paths"]["mert_embeddings"])
splits = np.array([split_of(i, 10, 10) for i in all_ids])
ref = apply_code_transform(layers[splits == "test"], fit_code_transform(layers[splits == "train"], "standardised"))
print(codes.shape, "max abs diff", np.abs(codes - ref).max())
