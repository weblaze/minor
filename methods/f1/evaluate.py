"""F1: evaluate a mapping on the test tracks.

Two outputs are scored for every track:
  direct    the feature vector produced by M
  painting  the training painting whose DINOv2 feature is nearest to it (cosine)

Structure: Mantel correlation, k-nearest-neighbour overlap and trustworthiness between cosine
distances of audio codes and of image features. For `painting`, the Pearson correlation is also
reported for feature sets computed from the pixels of the paintings (colour, layout, power spectrum).
Distribution: Frechet and kernel distance to the features of test art.

Without --mapper an untrained mapping is evaluated (random M).

Usage:
  python methods/f1/evaluate.py --mapper runs/<mapper run>/mapper.pt
"""
import numpy as np
import torch
from torchvision.utils import save_image

from abstraction.data.codes import apply_code_transform, transform_from_state
from abstraction.metrics.structure import distance_matrix
from abstraction.models.mapper import Mapper
from abstraction.utils.runlog import RunLog, set_seed

from common import (FEATURE_DIM, base_parser, codes_of, correlation, full_scores, image_statistics, load_art,
                    load_images, load_layers, map_codes, nearest_paintings, run_name, setup)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--mapper", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    hp = config["eval"]
    n_perm = args.permutations or (199 if args.smoke else hp["n_permutations"])

    ids, layers, splits = load_layers(config, args.smoke)
    if args.mapper:
        state = torch.load(args.mapper, map_location=device, weights_only=True)
        transform = transform_from_state(state["code_transform"])
        codes = apply_code_transform(layers, transform)
        code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
        variant = state["variant"]
    else:
        code = config["audio_code"]
        codes, transform = codes_of(layers, splits, code["variant"], code["whiten_dim"], code["layers"])
        train = torch.from_numpy(codes[splits == "train"]).to(device)
        code_mean, code_std = train.mean(dim=(0, 1)), train.std(dim=(0, 1)) + 1e-6
        variant = "random"
    mapper = Mapper(codes.shape[-1], FEATURE_DIM, config["mapper"]["hidden"], config["mapper"]["depth"]).to(device)
    if args.mapper:
        mapper.load_state_dict(state["model"])
    mapper.eval()
    config["run"] = {"variant": variant, "mapper": args.mapper, "code_variant": transform["variant"],
                     "code_layers": transform["layers"].tolist()}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"eval_{variant}", args.smoke), config, seed)

    names, art, art_splits = load_art(config, device, args.smoke)
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    test_art = art[art_splits == "test"]
    test_codes = codes[splits == "test"].mean(axis=1)
    direct = map_codes(mapper, test_codes, code_mean, code_std, device)
    nearest = nearest_paintings(direct, gallery)
    painting = gallery[nearest]

    results = {"variant": variant, "code_variant": transform["variant"],
               "code_layers": transform["layers"].tolist(),
               "direct": full_scores(test_codes, direct, test_art, hp["knn_k"], n_perm, seed),
               "painting": full_scores(test_codes, painting, test_art, hp["knn_k"], n_perm, seed)}
    results["painting"]["n_distinct_paintings"] = int(len(np.unique(nearest)))
    if not args.smoke:
        unique, inverse = np.unique(nearest, return_inverse=True)
        images = load_images(config, gallery_names[unique])
        d_audio = distance_matrix(test_codes)
        results["painting"]["structure_other_features"] = {
            name: correlation(d_audio, distance_matrix(stats[inverse], metric="euclidean"))
            for name, stats in image_statistics(images).items()}
        save_image(images[inverse[:64]] * 0.5 + 0.5, log.dir / "samples.png", nrow=8)
    log.save_json("results.json", results)
    np.savez(log.dir / "features.npz", track_ids=ids[splits == "test"], audio_codes=test_codes,
             direct_features=direct, painting_features=painting, painting_names=gallery_names[nearest])
    for key in ("direct", "painting"):
        print(key, results[key]["structure"], "FD", results[key]["image_distribution"]["frechet_distance"])
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
