"""Evaluate a Q1 mapping on the held-out test split. Nothing is trained here except the
genre probes, which are fitted on test-split features only.

One image is generated per test track from the mean of its window codes.

Structure (audio codes vs DINOv2 features of the generated images, cosine distance):
  Mantel correlation (Pearson and Spearman) with a permutation p-value,
  k-nearest-neighbour overlap with its chance level, trustworthiness.
Image distribution (DINOv2 features, against test-split art):
  Frechet distance, kernel distance, mean pairwise distance, participation ratio.
  Reference values: training art against test art, and VAE reconstructions of
  training art against test art.
Genre leakage (optional, --genres CSV with columns track_id,genre):
  5-fold cross-validated accuracy of logistic regression on the image features
  and on the audio codes, with the majority-class rate.

Without --mapper, an untrained mapping initialised from --seed is evaluated
(the random-M baseline), with the audio-code variant of the config.

Usage:
  python methods/q1/evaluate.py --vae runs/<vae run>/vae.pt --mapper runs/<run>/mapper.pt
  python methods/q1/evaluate.py --vae runs/<vae run>/vae.pt --seed 0
"""
import numpy as np
import torch
from torchvision.utils import save_image

from abstraction.data.codes import apply_code_transform, fit_code_transform, transform_from_state
from abstraction.generators import load_generator
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import art_dataset, load_layers
from abstraction.pipeline.features import build_dino, features_of_dataset
from abstraction.pipeline.scores import distribution_scores, full_scores, genre_leakage
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_mapper, code_statistics, generate, run_name


def main():
    parser = base_parser(__doc__, generator=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--mapper", default=None)
    parser.add_argument("--genres", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    ids, layers, splits = load_layers(config, args.smoke)
    if args.mapper:
        state = torch.load(args.mapper, map_location=device, weights_only=True)
        transform = transform_from_state(state["code_transform"])
        code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
        variant, trained_seed = state["variant"], state["seed"]
    else:
        transform = fit_code_transform(layers[splits == "train"], config["audio_code"]["variant"],
                                       config["audio_code"]["whiten_dim"])
        train_codes = torch.from_numpy(apply_code_transform(layers[splits == "train"], transform)).to(device)
        code_mean, code_std = code_statistics(train_codes)
        variant, trained_seed = "random", seed
    codes = apply_code_transform(layers[splits == "test"], transform).mean(axis=1)
    track_ids = [t for t, s in zip(ids, splits) if s == "test"]
    del layers

    mapper = build_mapper(config, codes.shape[-1], generator).to(device).eval()
    if args.mapper:
        mapper.load_state_dict(state["model"])
    config["run"] = {"variant": variant, "vae": str(args.vae), "mapper": args.mapper,
                     "mapper_seed": trained_seed, "code_variant": transform["variant"],
                     "generator": args.generator}
    log = RunLog(config["paths"]["runs_dir"], run_name(f"eval_{variant}", args.smoke), config, seed)

    features, first_images = [], None
    with torch.no_grad():
        for start in range(0, len(codes), 64):
            batch = torch.from_numpy(codes[start:start + 64]).to(device)
            images, _ = generate(mapper, generator, batch, code_mean, code_std)
            features.append(dino(images).float().cpu())
            if first_images is None:
                first_images = images.cpu()
    features = torch.cat(features).numpy()
    save_image(first_images * 0.5 + 0.5, log.dir / "samples.png", nrow=8)

    test_art = features_of_dataset(dino, art_dataset(config, "test", False, args.smoke), device).numpy()
    train_set = art_dataset(config, "train", False, args.smoke)
    train_art = features_of_dataset(dino, train_set, device).numpy()
    recon_art = (features_of_dataset(dino, train_set, device,
                                     transform=lambda x: generator.decode(generator.encode(x)[0])).numpy()
                 if generator.has_encoder else None)

    results = {"variant": variant, "mapper_seed": trained_seed, "code_variant": transform["variant"],
               **full_scores(codes, features, test_art, config["eval"]["knn_k"], n_perm, seed)}
    results["image_distribution"]["reference_train_art"] = distribution_scores(train_art, test_art, seed)
    if recon_art is not None:
        results["image_distribution"]["reference_vae_reconstructions"] = distribution_scores(recon_art, test_art, seed)
    if args.genres:
        results["genre_leakage"] = genre_leakage(track_ids, features, codes, args.genres)

    log.save_json("results.json", results)
    np.savez(log.dir / "features.npz", track_ids=np.asarray(track_ids), image_features=features, audio_codes=codes)
    print(f"run directory: {log.dir}")
    print(results["structure"])


if __name__ == "__main__":
    main()
