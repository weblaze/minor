# Music Feature Analysis Project

This project processes and analyzes music features from the FMA (Free Music Archive) dataset, performing feature extraction, normalization, and dimensionality reduction.

## Project Structure

### Directories

- `data/`: Contains processed data files and intermediate results
- `datasets/`: Contains the FMA dataset and metadata
- `dscripts/`: Contains all processing and analysis scripts

### Scripts in `dscripts/`

- `feature_extraction.py`: Extracts audio features from MP3 files using librosa
- `normalize_features.py`: Standardizes extracted features using sklearn
- `reduce_dimensionality.py`: Performs PCA dimensionality reduction on normalized features
- `data_loader.py`: Utility functions for loading various data files
- `process_music.py`: Main pipeline script that orchestrates the entire processing workflow
- `compare_pca.py`: Script for analyzing and visualizing PCA results

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place the FMA dataset in the `datasets/` directory:
   - `fma_small/`: Contains MP3 files
   - `fma_metadata/`: Contains metadata CSV files

3. Run the processing pipeline:
```bash
python dscripts/process_music.py
```

## Data Flow

1. Raw audio files → Feature extraction → `audio_features.json`
2. Features → Normalization → `normalized_features.npy` + `track_ids.json`
3. Normalized features → PCA reduction → `features_reduced.json` + `pca_model.pkl`

## Notes

- The project uses the FMA small dataset (8,000 tracks of 30s each)
- Feature extraction includes spectral, rhythm, mel, harmonic, and energy features
- PCA reduction maintains 95% of the variance while reducing dimensionality to 64 components 