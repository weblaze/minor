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
import csv

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.utils import save_image

from abstraction.metrics.distribution import frechet_distance, kernel_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix, knn_overlap, mantel
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_dino, build_mapper, features_of_dataset, generate, load_vae, setup
from data import art_dataset, load_codes


def genre_leakage(track_ids, image_features, codes, genres_csv):
    with open(genres_csv, newline="", encoding="utf-8") as f:
        genre_of = {row["track_id"].lstrip("0"): row["genre"] for row in csv.DictReader(f)}
    keep = [i for i, t in enumerate(track_ids) if t.lstrip("0") in genre_of]
    labels = np.array([genre_of[track_ids[i].lstrip("0")] for i in keep])
    _, counts = np.unique(labels, return_counts=True)

    def probe(x):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return float(cross_val_score(model, x[keep], labels, cv=5).mean())

    return {"n_tracks": len(keep), "n_genres": len(counts), "majority_rate": float(counts.max() / counts.sum()),
            "accuracy_from_images": probe(image_features), "accuracy_from_audio": probe(codes)}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--mapper", default=None)
    parser.add_argument("--genres", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    hp = config["eval"]
    n_perm = args.permutations or (199 if args.smoke else hp["n_permutations"])

    vae, latent_std = load_vae(args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    mapper = build_mapper(config).to(device).eval()
    track_ids, test_codes = load_codes(config, "test", args.smoke)
    if args.mapper:
        state = torch.load(args.mapper, map_location=device, weights_only=True)
        mapper.load_state_dict(state["model"])
        code_mean, code_std = state["code_mean"].to(device), state["code_std"].to(device)
        variant = state["variant"]
    else:
        train = torch.from_numpy(load_codes(config, "train", args.smoke)[1]).to(device)
        code_mean, code_std = train.mean(dim=(0, 1)), train.std(dim=(0, 1)) + 1e-6
        variant = "random"
    config["run"] = {"variant": variant, "vae": str(args.vae), "mapper": args.mapper}
    log = RunLog(config["paths"]["runs_dir"], f"s1_eval_{variant}" + ("_smoke" if args.smoke else ""), config, seed)

    codes = test_codes.mean(axis=1)
    features, first_images = [], None
    with torch.no_grad():
        for start in range(0, len(codes), 64):
            batch = torch.from_numpy(codes[start:start + 64]).to(device)
            images, _ = generate(mapper, vae, latent_std, batch, code_mean, code_std, config)
            features.append(dino(images).float().cpu())
            if first_images is None:
                first_images = images.cpu()
    features = torch.cat(features).numpy()
    save_image(first_images * 0.5 + 0.5, log.dir / "samples.png", nrow=8)

    d_audio = distance_matrix(codes)
    d_image = distance_matrix(features)
    r_pearson, p_pearson = mantel(d_audio, d_image, "pearson", n_perm, seed)
    r_spearman, p_spearman = mantel(d_audio, d_image, "spearman", n_perm, seed)
    k = hp["knn_k"]
    overlap, chance = knn_overlap(d_audio, d_image, k)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    results = {
        "variant": variant, "n_test_tracks": len(codes), "n_permutations": n_perm,
        "structure": {
            "mantel_pearson": r_pearson, "mantel_pearson_p": p_pearson,
            "mantel_spearman": r_spearman, "mantel_spearman_p": p_spearman,
            "knn_k": k, "knn_overlap": overlap, "knn_overlap_chance": chance,
            "trustworthiness": float(trustworthiness(d_audio, unit, n_neighbors=k, metric="precomputed")),
        },
    }

    test_art = features_of_dataset(dino, art_dataset(config, "test", False, args.smoke), device).numpy()
    train_set = art_dataset(config, "train", False, args.smoke)
    train_art = features_of_dataset(dino, train_set, device).numpy()
    recon_art = features_of_dataset(dino, train_set, device,
                                    transform=lambda x: vae.decode(vae.encode(x)[0])).numpy()
    iu = np.triu_indices(len(features), k=1)
    it = np.triu_indices(len(test_art), k=1)

    def against_test_art(x):
        kid_mean, kid_std = kernel_distance(x, test_art, seed=seed)
        return {"n": len(x), "frechet_distance": frechet_distance(x, test_art),
                "kernel_distance": kid_mean, "kernel_distance_std": kid_std}

    results["image_distribution"] = {
        "n_test_art": len(test_art),
        "generated": against_test_art(features),
        "reference_train_art": against_test_art(train_art),
        "reference_vae_reconstructions": against_test_art(recon_art),
        "mean_pairwise_distance_generated": float(d_image[iu].mean()),
        "mean_pairwise_distance_test_art": float(distance_matrix(test_art)[it].mean()),
        "participation_ratio_generated": participation_ratio(features),
        "participation_ratio_test_art": participation_ratio(test_art),
        "participation_ratio_audio": participation_ratio(codes),
    }
    if args.genres:
        results["genre_leakage"] = genre_leakage(track_ids, features, codes, args.genres)

    log.save_json("results.json", results)
    np.savez(log.dir / "features.npz", track_ids=np.asarray(track_ids), image_features=features, audio_codes=codes)
    print(f"run directory: {log.dir}")
    print(results["structure"])


if __name__ == "__main__":
    main()
