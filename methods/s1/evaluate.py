"""Evaluate a mapping on the held-out test split. Nothing is trained here except the
genre probes, which are fitted on test-split features only.

One image is generated per test track from the mean of its window codes.

Structure (audio codes vs DINOv2 features of the generated images, cosine distance):
  Mantel correlation (Pearson and Spearman) with a permutation p-value,
  k-nearest-neighbour overlap with its chance level, trustworthiness.
Image distribution (DINOv2 features, against test-split art):
  Frechet distance, kernel distance, mean pairwise distance of the outputs.
  Reference values: training art against test art, and VAE reconstructions of
  training art against test art.
Genre leakage (optional, --genres CSV with columns track_id,genre):
  5-fold cross-validated accuracy of logistic regression on the image features
  and on the audio codes, with the majority-class rate.

Without --mapper, an untrained mapping initialised from --seed is evaluated
(the random-M baseline).

Usage:
  python methods/s1/evaluate.py --vae runs/<vae run>/vae.pt --mapper runs/<run>/mapper.pt
  python methods/s1/evaluate.py --vae runs/<vae run>/vae.pt --seed 0
"""
import numpy as np
import torch
from torchvision.utils import save_image

from abstraction.generators import load_generator
from abstraction.metrics.geometry import participation_ratio
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import art_dataset
from abstraction.pipeline.features import build_dino, features_of_dataset
from abstraction.pipeline.scores import (distribution_scores, genre_leakage, mean_pairwise_distance,
                                         structure_scores)
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_mapper, generate, run_name
from data import load_codes


def main():
    parser = base_parser(__doc__, generator=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--mapper", default=None)
    parser.add_argument("--genres", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--save-images", action="store_true",
                        help="write the image of every test track to images/<track id>.png in the run directory")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    hp = config["eval"]
    n_perm = args.permutations or (199 if args.smoke else hp["n_permutations"])

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    state = torch.load(args.mapper, map_location=device, weights_only=True) if args.mapper else None
    if state is not None:
        config["audio_code"]["variant"] = state.get("code", "mean")
    mapper = build_mapper(config, generator).to(device).eval()
    track_ids, test_codes = load_codes(config, "test", args.smoke)
    if state is not None:
        mapper.load_state_dict(state["model"])
        code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
        variant = state["variant"]
    else:
        train = torch.from_numpy(load_codes(config, "train", args.smoke)[1]).to(device)
        code_mean, code_std = train.mean(dim=(0, 1)), train.std(dim=(0, 1)) + 1e-6
        variant = "random"
    config["run"] = {"variant": variant, "vae": str(args.vae), "mapper": args.mapper,
                     "code": config["audio_code"]["variant"], "generator": args.generator}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"eval_{variant}", args.smoke), config, seed)

    codes = test_codes.mean(axis=1)
    features, first_images = [], None
    with torch.no_grad():
        for start in range(0, len(codes), 64):
            batch = torch.from_numpy(codes[start:start + 64]).to(device)
            images, _ = generate(mapper, generator, batch, code_mean, code_std)
            features.append(dino(images).float().cpu())
            if args.save_images:
                (log.dir / "images").mkdir(exist_ok=True)
                for offset, image in enumerate(images.cpu()):
                    save_image(image * 0.5 + 0.5, log.dir / "images" / f"{track_ids[start + offset]}.png")
            if first_images is None:
                first_images = images.cpu()
    features = torch.cat(features).numpy()
    save_image(first_images * 0.5 + 0.5, log.dir / "samples.png", nrow=8)

    structure = structure_scores(codes, features, hp["knn_k"], n_perm, seed)
    results = {
        "variant": variant, "n_test_tracks": len(codes), "n_permutations": n_perm,
        "structure": {k: v for k, v in structure.items() if k not in ("n_tracks", "n_permutations")},
    }

    test_art = features_of_dataset(dino, art_dataset(config, "test", False, args.smoke), device).numpy()
    train_set = art_dataset(config, "train", False, args.smoke)
    train_art = features_of_dataset(dino, train_set, device).numpy()
    recon_art = (features_of_dataset(dino, train_set, device,
                                     transform=lambda x: generator.decode(generator.encode(x)[0])).numpy()
                 if generator.has_encoder else None)

    def against_test_art(x):
        scores = distribution_scores(x, test_art, seed)
        return {k: v for k, v in scores.items() if k != "n_reference"}

    results["image_distribution"] = {
        "n_test_art": len(test_art),
        "generated": against_test_art(features),
        "reference_train_art": against_test_art(train_art),
        "mean_pairwise_distance_generated": mean_pairwise_distance(features),
        "mean_pairwise_distance_test_art": mean_pairwise_distance(test_art),
        "participation_ratio_generated": participation_ratio(features),
        "participation_ratio_test_art": participation_ratio(test_art),
        "participation_ratio_audio": participation_ratio(codes),
    }
    if recon_art is not None:
        results["image_distribution"]["reference_vae_reconstructions"] = against_test_art(recon_art)
    if args.genres:
        results["genre_leakage"] = genre_leakage(track_ids, features, codes, args.genres)

    log.save_json("results.json", results)
    np.savez(log.dir / "features.npz", track_ids=np.asarray(track_ids), image_features=features, audio_codes=codes)
    print(f"run directory: {log.dir}")
    print(results["structure"])


if __name__ == "__main__":
    main()
