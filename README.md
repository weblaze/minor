Let’s create a `README.md` file for your audio-to-image generation project. The README will provide an overview of the project, describe its components, explain how to set up and run the code, and outline the current status and future improvements. Since this is a machine learning project involving VAEs and a mapping network to generate abstract art images from audio inputs, I’ll tailor the README to reflect that context, incorporating details from the scripts and discussions we’ve had (e.g., the use of `AudioVAE`, `ImageVAE`, `MappingNetwork`, dataset paths, and the current issue with diversity in generated images).

Below is the `README.md` file for your project:

---

# Audio-to-Image Generation: Abstract Art from Music

## Overview

This project implements an audio-to-image generation pipeline that transforms audio inputs into abstract art images using a combination of Variational Autoencoders (VAEs) and a mapping network. The pipeline consists of three main components:

1. **AudioVAE**: Encodes audio features into a latent space.
2. **ImageVAE**: Encodes and decodes abstract art images into/from a latent space.
3. **MappingNetwork**: Maps the audio latent space to the image latent space, enabling the generation of images from audio inputs.

The project uses PyTorch for model implementation and training, and the generated images are abstract art pieces inspired by the audio inputs. The audio dataset contains diverse audio features, and the image dataset includes a variety of unique abstract art images.

### Current Status
- The pipeline successfully generates images from audio inputs.
- However, the generated images currently lack diversity, producing nearly identical outputs with only slight shape variations, despite diverse audio inputs and variability in the latent spaces (`z_audio` and `z_image`).
- The primary issue has been identified as a limitation in the `ImageVAE`, likely due to a constrained latent space (KLD of 5.8492, on the lower side of the target range 3.0–6.0) and/or limited decoder capacity.
- Efforts are underway to retrain the `ImageVAE` with a higher KLD (`MAX_BETA=4.0`, `KLD_WEIGHT=1.5`) to improve diversity.

## Project Structure

```
audio-to-image-generation/
│
├── datasets/
│   ├── audio_features/         # Directory containing preprocessed audio features
│   └── abstract_art/           # Directory containing abstract art images
│
├── audio_vae.py                # AudioVAE model implementation
├── image_vae.py                # ImageVAE model implementation
├── mapping_network.py          # MappingNetwork model implementation
├── datasets.py                 # Dataset classes (AudioFeatureDataset, ImageDataset)
├── train_audio_vae.py          # Script to train the AudioVAE
├── train_image_vae.py          # Script to train the ImageVAE
├── train_mapping_network.py    # Script to train the MappingNetwork
├── generate_image.py           # Script to generate images from audio inputs
│
├── audio_vae.pth               # Pre-trained AudioVAE model weights
├── image_vae.pth               # Pre-trained ImageVAE model weights
├── mapping_network.pth         # Pre-trained MappingNetwork model weights
│
├── generated_images/           # Directory where generated images are saved
│
└── README.md                   # Project documentation (this file)
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Torchvision
- NumPy
- (Optional) Librosa (for audio feature extraction, if not preprocessed)

Install the required dependencies using:

```bash
pip install torch torchvision numpy
```

If you need to preprocess audio files, install Librosa:

```bash
pip install librosa
```

## Setup

1. **Prepare the Datasets**:
   - **Audio Features**: Place preprocessed audio features (e.g., MFCCs, spectrograms) in the `datasets/audio_features/` directory. The `AudioFeatureDataset` expects normalized features with shape `(channels, time_steps)`, where `channels=20` and `time_steps=216`.
   - **Abstract Art Images**: Place abstract art images in the `datasets/abstract_art/` directory. The `ImageDataset` expects images to be in a format compatible with Torchvision (e.g., PNG, JPEG).

2. **Pre-trained Models**:
   - The pre-trained model weights (`audio_vae.pth`, `image_vae.pth`, `mapping_network.pth`) should be in the project root directory. If you need to train the models from scratch, follow the training steps below.

3. **Directory for Generated Images**:
   - The `generate_image.py` script saves generated images to the `generated_images/` directory. This directory will be created automatically if it doesn’t exist.

## Usage

### 1. Train the Models (Optional)
If you don’t have the pre-trained model weights, you can train the models from scratch.

#### Train the AudioVAE
The `AudioVAE` encodes audio features into a latent space.

```bash
python train_audio_vae.py
```

- **Hyperparameters**:
  - `BATCH_SIZE`: 16
  - `EPOCHS`: 100
  - `LEARNING_RATE`: 1e-4
  - `LATENT_DIM`: 64
  - `MAX_BETA`: 3.0 (for KLD regularization)
- **Output**: Saves the trained model to `audio_vae.pth`.

#### Train the ImageVAE
The `ImageVAE` encodes and decodes abstract art images into/from a latent space.

```bash
python train_image_vae.py
```

- **Hyperparameters**:
  - `BATCH_SIZE`: 16
  - `EPOCHS`: 75
  - `LEARNING_RATE`: 1e-4
  - `LATENT_DIM`: 64
  - `MAX_BETA`: 4.0 (increased from 3.0 to improve diversity)
  - `KLD_WEIGHT`: 1.5 (added to emphasize KLD)
- **Output**: Saves the trained model to `image_vae.pth`.
- **Note**: The current `ImageVAE` model has a KLD of 5.8492, which is being increased to improve diversity in generated images.

#### Train the MappingNetwork
The `MappingNetwork` maps the audio latent space to the image latent space.

```bash
python train_mapping_network.py
```

- **Hyperparameters**:
  - `BATCH_SIZE`: 16
  - `EPOCHS`: 100
  - `LEARNING_RATE`: 1e-4
  - `LATENT_DIM`: 64
- **Output**: Saves the trained model to `mapping_network.pth`.
- **Training Progress**:
  - Epoch 20: Mapping Loss: 0.6079
  - Estimated losses (based on trend):
    - Epoch 25: Mapping Loss: 0.6014
    - Epoch 50: Mapping Loss: 0.5689
    - Epoch 100: Mapping Loss: 0.5439
  - Actual loss values for epochs 25, 50, and 100 should be confirmed after training completion.

### 2. Generate Images from Audio
Use the `generate_image.py` script to generate abstract art images from audio inputs.

```bash
python generate_image.py
```

- **What It Does**:
  - Selects 10 random audio samples from the `AudioFeatureDataset`.
  - Encodes each audio sample into a latent vector (`z_audio`) using the `AudioVAE`.
  - Maps `z_audio` to an image latent vector (`z_image`) using the `MappingNetwork`.
  - Decodes `z_image` into an image using the `ImageVAE`.
  - Saves the generated images as `generated_image_1.png` to `generated_image_10.png` in the `generated_images/` directory.
- **Hyperparameters**:
  - `LATENT_DIM`: 64
  - `LOGVAR_SCALE`: 2.0 (to increase variability in `z_image`)
  - `NOISE_STD`: 0.1 (to add random noise to `z_image` for more diversity)
- **Current Issue**: The generated images lack diversity, producing nearly identical outputs. This is being addressed by retraining the `ImageVAE` with a higher KLD.

## Results

- **Generated Images**: The pipeline successfully generates abstract art images from audio inputs, but the current outputs lack diversity (e.g., similar colors, textures, and compositions with only slight shape variations).
- **Logs from Latest Generation**:
  - Random audio indices: [2364, 1311, 6367, 6902, 1649, 1511, 7814, 1229, 2127, 553]
  - Audio features show variability (e.g., mean range: 0.3979 to 0.8440, std range: 0.0914 to 0.1403).
  - `z_audio` range: -3.4546 to 2.5047
  - `z_image` range: -3.2771 to 2.8307
  - Despite variability in the latent spaces, the generated images are nearly identical, indicating a bottleneck in the `ImageVAE`.

## Future Improvements

1. **Improve Diversity in Generated Images**:
   - **Retrain `ImageVAE`**: Currently in progress with `MAX_BETA=4.0` and `KLD_WEIGHT=1.5` to increase the KLD (target: 6.0–7.0) and encourage a more diverse latent space.
   - **Increase `ImageVAE` Decoder Capacity**: Add more layers or filters to the `ImageVAE` decoder to better capture variations in the latent space.
   - **Retrain `MappingNetwork`**: Extend training to 200 epochs with a reduced scheduler `patience` of 10 to learn a more nuanced mapping.

2. **Enhance Audio-to-Image Mapping**:
   - Log more detailed audio features (e.g., tempo, pitch, energy) using Librosa to better understand the audio-to-image mapping.
   - Fine-tune the `MappingNetwork` to ensure that variations in `z_audio` lead to more distinct `z_image` vectors.

3. **Experiment with Generation Parameters**:
   - Further increase `LOGVAR_SCALE` (e.g., to 3.0) and `NOISE_STD` (e.g., to 0.2) to introduce more variability in `z_image`, though this is a temporary workaround.
   - Explore conditional generation techniques (e.g., conditioning the `ImageVAE` decoder on audio features directly).

## Contributing

Contributions are welcome! If you’d like to contribute, please:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Built with PyTorch and Torchvision.
- Inspired by research on cross-modal generation and VAEs.
- Thanks to the open-source community for providing tools and datasets.

---
