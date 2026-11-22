"""Teacher descriptions for the drawing mapper (`teach.py`): what a picker says each window should look like.

Teacher: a trained F1 mapper (audio code -> DINOv2 feature). For every window of every track (all splits):

  dinov2   the F1 output itself (384), the picker's description
  dino_v1  the same description read by DINO v1 (ViT-B/16 CLS, 768): the mean DINO v1 feature of the `--k`
           training paintings nearest to the description (DINOv2 cosine), weighted by softmax(cos / `--tau`).
           A soft description, so the drawer is not pushed towards a single training painting.
  pick     index of the nearest training painting (what F1 itself shows), with its DINOv2 cosine

Writes <out>/targets.npz: ids (n_tracks), splits, dinov2 (n, w, 384), dino_v1 (n, w, 768) as float16,
pick (n, w), pick_cos (n, w), train_names.

Usage:
  python methods/gan/teacher_targets.py --teacher runs/f1_mapper_f1_seed0_<stamp>/mapper.pt --out <dir>
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from abstraction.data.codes import apply_code_transform, transform_from_state
from abstraction.data.datasets import ImageDataset
from abstraction.data.splits import split_of
from abstraction.models.mapper import Mapper
from abstraction.pipeline import cli
from abstraction.pipeline.data import load_layers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s3"))
from second_reader import DinoV1  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def main():
    parser = cli.base_parser(__doc__, CONFIG_PATH)
    parser.add_argument("--teacher", required=True, help="F1 mapper.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.02)
    args = parser.parse_args()
    config, device = cli.setup(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.teacher, map_location=device, weights_only=True)
    weights = state["model"]
    teacher = Mapper(weights["blocks.0.net.0.weight"].shape[1], 384, weights["blocks.0.net.0.weight"].shape[0],
                     sum(1 for k in weights if k.endswith("net.0.weight") and k.startswith("blocks."))).to(device)
    teacher.load_state_dict(weights)
    teacher.eval()

    ids, layers, splits = load_layers(config)
    codes = apply_code_transform(layers, transform_from_state(state["code_transform"]))
    n, w, d = codes.shape
    with torch.no_grad():
        x = (torch.from_numpy(codes.reshape(n * w, d)).to(device) - state["code_mean"].to(device)) / state["code_std"].to(device)
        desc = torch.cat([teacher(x[i:i + 4096]) for i in range(0, len(x), 4096)]).float()

    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
    with np.load(cache) as data:
        names, art = data["names"], torch.from_numpy(data["features"]).float().to(device)
    train = np.array([split_of(m, val, test) for m in names]) == "train"
    train_names, train_art = names[train], F.normalize(art[train], dim=1)

    dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    row_of = {m: i for i, m in enumerate(dataset.image_files)}
    dino_v1 = DinoV1(device)
    v1 = []
    with torch.no_grad():
        for start in range(0, len(train_names), 32):
            images = torch.stack([dataset[row_of[str(m)]] for m in train_names[start:start + 32]]).to(device)
            v1.append(dino_v1(images).float())
        v1 = torch.cat(v1)
        sims = F.normalize(desc, dim=1) @ train_art.T
        top, index = sims.topk(args.k, dim=1)
        weight = torch.softmax(top / args.tau, dim=1)
        target_v1 = (weight[:, :, None] * v1[index]).sum(1)

    np.savez(out / "targets.npz", ids=ids, splits=splits,
             dinov2=desc.view(n, w, -1).cpu().numpy().astype(np.float16),
             dino_v1=target_v1.view(n, w, -1).cpu().numpy().astype(np.float16),
             pick=index[:, 0].view(n, w).cpu().numpy(), pick_cos=top[:, 0].view(n, w).cpu().numpy(),
             train_names=train_names)
    print(f"{n} tracks x {w} windows, top-1 painting cosine {float(top[:, 0].mean()):.3f}; wrote {out / 'targets.npz'}")


if __name__ == "__main__":
    main()
