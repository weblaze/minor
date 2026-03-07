import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import open_clip

import laion_clap
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_recall_at_k(similarity_matrix, k=1):
    """
    Computes Recall@K for a square similarity matrix.
    similarity_matrix: [N, N] tensor where element (i, j) is the similarity
                       between audio i and image j.
    """
    N = similarity_matrix.shape[0]
    
    # Ranks is [N, N].
    # argsort(-sim) sorts descending, then for each row, 
    # argsort gives the rank of that element.
    sorted_indices = torch.argsort(similarity_matrix, descending=True, dim=1)
    
    # We want to know if the correct index `i` is within the top K.
    # The correct index for row i is exactly i (audio i paired with image i).
    correct_in_top_k = 0
    for i in range(N):
        if i in sorted_indices[i, :k]:
            correct_in_top_k += 1
            
    return correct_in_top_k / N

def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio-Image Semantic Alignment")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing the conditioning source audio files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing generated images to evaluate")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}")

    # 1. Load CLIP for Images
    print("Loading OpenCLIP Model for Image Embeddings...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model = clip_model.to(device)
    clip_model.eval()

    # 2. Load LAION-CLAP for Audio
    print("Loading LAION-CLAP Model for Audio Embeddings...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap_model.load_ckpt('music_audioset_epoch_15_esc_90.14.pt')
    clap_model.to(device)
    clap_model.eval()

    # 3. Discover Pairs
    audio_files = sorted(list(Path(args.audio_dir).glob("*.mp3")) + list(Path(args.audio_dir).glob("*.wav")))
    image_files = sorted(list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg")))
    
    # For a perfect paired test, we evaluate matching N
    N = min(len(audio_files), len(image_files))
    if N == 0:
        print("ERROR: Could not find paired audio/image files in the provided directories.")
        return
        
    audio_files = audio_files[:N]
    image_files = image_files[:N]
    
    print(f"Found {N} paired audio-image samples for evaluation.")

    audio_embeddings = []
    image_embeddings = []

    # 4. Extract Embeddings
    print("Extracting Audio Embeddings...")
    with torch.no_grad():
        for audio_path in tqdm(audio_files):
            # Get audio embedding
            embed = clap_model.get_audio_embedding_from_filelist(x=[str(audio_path)], use_tensor=True)
            audio_embeddings.append(embed)
            
    print("Extracting Image Embeddings...")
    with torch.no_grad():
        for img_path in tqdm(image_files):
            # Get image embedding
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            embed = clip_model.encode_image(img_tensor)
            
            # Normalize the image embedding (cosine similarity requires normalized vectors)
            embed = embed / embed.norm(dim=-1, keepdim=True)
            image_embeddings.append(embed)

    # 5. Compute Similarity Matrix
    # [N, 512]
    audio_tensor = torch.cat(audio_embeddings, dim=0)
    image_tensor = torch.cat(image_embeddings, dim=0)
    
    # Compute dot product similarity (both spaces are normalized text spaces)
    # Returns [N, N] matrix where [i, j] is similarity of audio i with image j
    similarity_matrix = torch.matmul(audio_tensor, image_tensor.t())

    # 6. Evaluate
    recall_1 = compute_recall_at_k(similarity_matrix, k=1)
    recall_5 = compute_recall_at_k(similarity_matrix, k=5)

    print("\n" + "="*40)
    print("Retrieval Evaluation Results")
    print("="*40)
    print(f"Random Baseline (R@1): {1/N:.4f}")
    if N >= 5:
        print(f"Random Baseline (R@5): {5/N:.4f}")
    print("-" * 40)
    print(f"Model Recall@1: {recall_1:.4f} ({(recall_1/(1/N)):.2f}x Baseline)")
    if N >= 5:
         print(f"Model Recall@5: {recall_5:.4f} ({(recall_5/(5/N)):.2f}x Baseline)")
    print("="*40)

if __name__ == "__main__":
    main()
