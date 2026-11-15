import sys, glob
import numpy as np
sys.path.insert(0, "methods/s1")
from abstraction.utils.config import load_config
from data import load_codes
config = load_config("methods/s1/config.yaml")
config["paths"]["mert_embeddings"] = "runs/kaggle/abstraction-features/mert_embeddings"
config["audio_code"]["variant"] = "whitened"
ids, codes = load_codes(config, "test")
z = np.load(glob.glob("runs/kaggle/abstraction-q1-seed0-vae-pair/runs/q1_eval_q1_kl0.02_*/features.npz")[0])
print(len(ids), codes.shape, (np.array(ids) == z["track_ids"]).all())
a, b = codes.mean(axis=1), z["audio_codes"]
print("max abs diff", np.abs(a - b).max(), "scale", np.abs(b).mean())
