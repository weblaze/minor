"""S3 second-reader check: is the invented language readable by an image encoder the game never saw?

The mapper of an S3 run is frozen. One image is generated per track (mean of its window codes) for the
training, validation and test tracks. A fresh reader (same architecture as in training, new weights) is
trained by InfoNCE on the training tracks only, on frozen features of one image encoder:

  dino_v1   DINO ViT-B/16 (facebook/dino-vitb16, CLS token): self-supervised without text, not used in S3
  dinov2    DINOv2 ViT-S/14 as in training (reference: what a fresh reader recovers in the original space)

The reader with the lowest validation loss is scored on the test tracks as methods/s3/identify.py
(top1, top10, median rank, bits; chance 1/789).

Usage:
  python methods/s3/second_reader.py --vae <gan run>/generator.pt --run runs/<s3 run> [--encoders dino_v1 dinov2]
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.generators import load_generator
from abstraction.pipeline.cli import setup
from abstraction.pipeline.features import build_dino
from abstraction.utils.runlog import RunLog, set_seed

from common import Reader, base_parser, build_mapper, code_dim, generate, info_nce, load_codes, run_name
from identify import scores

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class DinoV1:
    dim = 768

    def __init__(self, device):
        from transformers import ViTModel
        self.model = ViTModel.from_pretrained("facebook/dino-vitb16", add_pooling_layer=False,
                                              revision="refs/pr/1").to(device).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, images):
        x = F.interpolate(images * 0.5 + 0.5, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state[:, 0]


def train_reader(features, codes, config, device, seed, steps):
    s3 = config["s3"]
    torch.manual_seed(seed)
    reader = Reader(codes["train"].shape[1], s3["embed_dim"], s3["reader_hidden"], features["train"].shape[1]).to(device)
    optimiser = torch.optim.AdamW(reader.parameters(), lr=s3["reader_lr"], weight_decay=0.05)
    g = torch.Generator().manual_seed(seed)
    n, batch = len(codes["train"]), 256
    best, best_state = float("inf"), None
    for step in range(steps):
        index = torch.randperm(n, generator=g)[:batch].to(device)
        loss = info_nce(reader.embed_images(features["train"][index]), reader.embed_audio(codes["train"][index]),
                        s3["temperature"])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if (step + 1) % 100 == 0:
            reader.eval()
            with torch.no_grad():
                val = float(info_nce(reader.embed_images(features["val"]), reader.embed_audio(codes["val"]),
                                     s3["temperature"]))
            reader.train()
            if val < best:
                best, best_state = val, {k: v.clone() for k, v in reader.state_dict().items()}
    reader.load_state_dict(best_state)
    return reader.eval(), best


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--encoders", nargs="+", default=["dino_v1", "dinov2"])
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    run = Path(args.run)
    state = torch.load(run / "mapper.pt", map_location=device, weights_only=True)
    config["audio_code"]["variant"] = state["code"]
    config["run"] = {"s3_run": str(run), "variant": state["variant"], "encoders": args.encoders}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"second_reader_{state['variant']}", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    mapper = build_mapper(config, generator).to(device).eval()
    mapper.load_state_dict(state["model"])
    code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
    tracks = {split: load_codes(config, split, args.smoke)[1].mean(axis=1) for split in ("train", "val", "test")}
    codes = {split: (torch.from_numpy(x).to(device) - code_mean) / code_std for split, x in tracks.items()}

    results = {"variant": state["variant"], "steps": state["steps"], "n_test_tracks": len(tracks["test"])}
    for name in args.encoders:
        encoder = DinoV1(device) if name == "dino_v1" and not args.smoke else build_dino(config, device, args.smoke)
        features = {}
        with torch.no_grad():
            for split, x in tracks.items():
                out = []
                for start in range(0, len(x), 32):
                    images, _ = generate(mapper, generator, torch.from_numpy(x[start:start + 32]).to(device),
                                         code_mean, code_std)
                    out.append(encoder(images).float())
                features[split] = torch.cat(out)
        mean, std = features["train"].mean(0), features["train"].std(0) + 1e-6
        features = {k: (v - mean) / std for k, v in features.items()}
        reader, val_loss = train_reader(features, codes, config, device, seed, 200 if args.smoke else args.steps)
        with torch.no_grad():
            results[name] = {**scores(reader.embed_images(features["test"]), reader.embed_audio(codes["test"]),
                                      config["s3"]["temperature"]), "val_loss": val_loss}
        print(name, results[name], flush=True)
        del encoder
        torch.cuda.empty_cache()
    log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
