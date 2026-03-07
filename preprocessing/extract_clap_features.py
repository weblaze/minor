import os
import torch
import numpy as np
import laion_clap
from tqdm import tqdm
import yaml

# Adjust path based on execution location to find configs/config.yaml
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(base_dir, "configs", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Hardcode dataset path for FMA small
DATASET_PATH = os.path.join(base_dir, "datasets", "fma_small")
OUTPUT_DIR = os.path.join(base_dir, "datasets", "clap_features")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device Configuration
device_str = config["system"]["device"]
if device_str == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but not available. Falling back to CPU for feature extraction.")
    device = torch.device("cpu")
else:
    device = torch.device(device_str)
print(f"Using device: {device}")

# Initialize LAION CLAP (music_audioset checkpoint)
print("Loading LAION CLAP (music_audioset)...")
# Using the amodel= 'HTSAT-base' as per default laion-clap
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

# Load the pretrained checkpoint
model.load_ckpt('music_audioset_epoch_15_esc_90.14.pt')
model = model.to(device)
model.eval()

def extract_clap_embedding(audio_path):
    try:
        # laion_clap has a built in audio retrieval feature
        # get_audio_embedding_from_filelist takes a list of paths
        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
        return audio_embed[0] # Return the 1D numpy array (512,)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Audio dataset path does not exist: {dataset_path}")
    
    # Track genres processing
    for genre_folder in tqdm(os.listdir(dataset_path), desc="Processing Audio Genres"):
        genre_path = os.path.join(dataset_path, genre_folder)
        if os.path.isdir(genre_path):
            file_list = [f for f in os.listdir(genre_path) if f.endswith((".mp3", ".wav"))]
            
            for file in tqdm(file_list, desc=f"Processing {genre_folder}", leave=False):
                file_path = os.path.join(genre_path, file)
                
                # We save with the _clap.npy suffix to distinguish from old features
                feature_filename = file.rsplit(".", 1)[0] + "_clap.npy"
                out_path = os.path.join(OUTPUT_DIR, feature_filename)
                
                # Skip if already processed
                if os.path.exists(out_path):
                    continue
                    
                features = extract_clap_embedding(file_path)
                if features is not None:
                    # CLAP embeddings are (512,) arrays.
                    np.save(out_path, features)

if __name__ == "__main__":
    print("Extracting LAION CLAP audio features...")
    # Will download weights if not present
    process_audio_dataset(DATASET_PATH)
    print("\n✅ CLAP audio feature extraction complete!")
