"""Scores of one set of outputs against audio codes of the same tracks and against reference art.

All distances are cosine distances unless stated otherwise.
"""
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from abstraction.metrics.distribution import frechet_distance, kernel_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix, knn_overlap, mantel


def correlation(d_a, d_b):
    """Pearson correlation of the upper triangles of two distance matrices."""
    iu = np.triu_indices(len(d_a), k=1)
    return float(np.corrcoef(d_a[iu], d_b[iu])[0, 1])


def mean_pairwise_distance(features):
    d = distance_matrix(features)
    return float(d[np.triu_indices(len(d), k=1)].mean())


def structure_scores(codes, features, k, n_permutations, seed):
    """Agreement between cosine distances of audio codes and of image features over the same tracks."""
    d_audio, d_image = distance_matrix(codes), distance_matrix(features)
    r_pearson, p_pearson = mantel(d_audio, d_image, "pearson", n_permutations, seed)
    r_spearman, p_spearman = mantel(d_audio, d_image, "spearman", n_permutations, seed)
    overlap, chance = knn_overlap(d_audio, d_image, k)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return {
        "n_tracks": len(codes), "n_permutations": n_permutations,
        "mantel_pearson": r_pearson, "mantel_pearson_p": p_pearson,
        "mantel_spearman": r_spearman, "mantel_spearman_p": p_spearman,
        "knn_k": k, "knn_overlap": overlap, "knn_overlap_chance": chance,
        "trustworthiness": float(trustworthiness(d_audio, unit, n_neighbors=k, metric="precomputed")),
    }


def distribution_scores(features, reference, seed=0):
    kid_mean, kid_std = kernel_distance(features, reference, seed=seed)
    return {"n": len(features), "n_reference": len(reference),
            "frechet_distance": frechet_distance(features, reference),
            "kernel_distance": kid_mean, "kernel_distance_std": kid_std}


def full_scores(codes, features, test_art, k, n_permutations, seed):
    """Structure, image distribution and collapse indicators of one set of outputs."""
    return {
        "structure": structure_scores(codes, features, k, n_permutations, seed),
        "image_distribution": {
            **distribution_scores(features, test_art, seed),
            "mean_pairwise_distance": mean_pairwise_distance(features),
            "mean_pairwise_distance_test_art": mean_pairwise_distance(test_art),
            "participation_ratio": participation_ratio(features),
            "participation_ratio_test_art": participation_ratio(test_art),
            "participation_ratio_audio": participation_ratio(codes),
        },
    }


def quick_scores(codes, features, test_art, k):
    """Structure and distribution scores without permutation tests."""
    d_audio, d_image = distance_matrix(codes), distance_matrix(features)
    return {"r": correlation(d_audio, d_image), "knn_overlap": knn_overlap(d_audio, d_image, k)[0],
            "frechet_distance": frechet_distance(features, test_art),
            "participation_ratio": participation_ratio(features)}


def genre_leakage(track_ids, image_features, codes, genres_csv):
    """5-fold accuracy of a linear genre probe on image features and on audio codes (test tracks only)."""
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
