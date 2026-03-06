<h1 align="center">Synesthesia: Audio-to-Image Generation (v0.1.0) 🎶➡️🖼️</h1>

<p align="center">
  <b>Transforming music and audio features into unique abstract art using Variational Autoencoders (VAEs) and Latent Space Mapping.</b>
</p>

## 📖 Overview

Synesthesia is an experimental deep learning pipeline that bridges the gap between sound and vision. By extracting acoustical features from audio files (MFCCs, spectral centroids, RMS, and tempo), the system maps the audio landscape to a visual latent space, which is then decoded into generative abstract art.

The core architecture consists of three interconnected PyTorch models:
1. **AudioVAE**: Encodes raw audio features (22 channels across time steps) into a compressed latent representation.
2. **ImageVAE**: Decodes latent vectors into 128x128 abstract art images (trained on a specialized abstract art dataset).
3. **Mapping Network**: A sophisticated translation network connecting the Audio latent space to the Image latent space, trained using Cycle Consistency, Maximum Mean Discrepancy (MMD), and target Diversity constraints.

---

## 🚀 Features

- **Offline ML Pipeline**: Generate art entirely offline using local PyTorch model weights.
- **Web Interface**: A sleek, interactive UI powered by **Streamlit**.
- **Drag & Drop Upload**: Instantly process `.mp3` or `.wav` files.
- **YouTube Support**: Paste any YouTube link to fetch the audio automatically via `yt-dlp` and generate corresponding art.

---

## 🛠️ Installation & Setup

### 1. Requirements
Ensure you have **Python 3.11** (or 3.12) installed on your system. 

Clone the repository and install the dependencies:
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the environment (Windows)
.\venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

### 2. Pre-trained Weights
The application requires the pre-trained weights for the autoencoders. Ensure the following files exist in the `tmodels/` directory:
- `audio_vae.pth`
- `image_vae.pth`
- `mapping_network.pth`

---

## 💻 Usage

### Launching the Web Application
The easiest and recommended way to interact with the models is through the Streamlit UI.

1. Activate your virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`. From here, you can upload audio files or paste YouTube links to see the generative process in action.

---

## 🧠 Deep-Dive: System Architecture & Methodology

Synesthesia is driven by three primary neural network components, each serving a specific role in translating audio features into the visual domain.

### 1. Audio Latent Encoding (`AudioVAE`)
The `AudioVAE` is a 1D Convolutional Variational Autoencoder responsible for understanding musical characteristics.
*   **Input Features**: The network ingests a fused tensor of shape `[Batch, 22, 216]`. The 22 channels comprise 20 Mel-Frequency Cepstral Coefficients (MFCCs), 1 Spectral Centroid channel, and 1 Root Mean Square (RMS) energy channel over 216 time steps.
*   **Encoder Structure**: A 3-layer `Conv1d` network (kernel size 3, stride 2) compresses the temporal dimension, squeezing the audio features through `BatchNorm1d` and `ReLU` activations down to a flattened representation.
*   **Latent Space**: The flattened tensor is mapped via fully connected layers to a highly compressed multivariate Gaussian distribution (represented by $\mu$ and $\log\sigma^2$, producing a robust latent dimensionality of `Z=512`).

### 2. Image Latent Encoding (`ImageVAE`)
The `ImageVAE` is a symmetric 2D Convolutional Autoencoder functioning as the "visual brain" of the application.
*   **Input**: RGB Abstract Art images resized and normalized to `128x128`.
*   **Encoder**: A deep 5-layer `Conv2d` stack progressively downsamples the spatial resolution (`128 -> 64 -> 32 -> 16 -> 8 -> 4`) while expanding channel capacity (up to 1024 filters).
*   **Latent Space**: Like the Audio VAE, it projects its spatial embeddings into a shared latent dimensionality (`Z=512`), parameterizing $\mu$ and $\log\sigma^2$.
*   **Decoder**: The generator. Using `ConvTranspose2d` layers, it upsamples the `Z=512` vector back into a `128x128x3` image. A final `Tanh` activation ensures the image bounds are dynamically constrained to `[-1, 1]` before denormalization.

### 3. The Bridge: Latent Space Translation (`Mapping Network`)
This is the core of the cross-modal generation. The `MappingNetwork` is tasked with taking a sequence of audio features ($Z_{audio}$) and accurately predicting what the corresponding visual ($Z_{image}$) should look like.

*   **Conditioning**: The raw audio latent vector is concatenated with a condensed 3-dimensional deterministic audio "condition" vector (mean spectral centroid, mean RMS, and global track tempo).
*   **Network Structure**: A deep Multi-Layer Perceptron (MLP) with consecutive 512-neuron `Linear` layers and `ReLU` activations maps the concatenated input to the target $Z_{image}$ distribution ($\mu_{mapped}, \log\sigma^2_{mapped}$).

### 🧪 Advanced Training Constraints & Loss Functions

Training the bridge between two entirely different modalities requires a highly constrained optimization landscape. `train_mapping_network.py` relies on a multi-objective loss function to guarantee stability:

1.  **Inverse Mapping & Cycle Consistency ($L_{cycle}$)**: 
    *   Alongside the forward `MappingNetwork`, an `InverseMappingNetwork` is trained simultaneously to map $Z_{image} \rightarrow Z_{audio}$. 
    *   Cycle Consistency Loss enforces that if we map an audio track to an image, and then map that image *back* to an audio track, the result should equal the original audio track ($A \rightarrow I \rightarrow A' \approx A$). This prevents the mapping network from collapsing into outputting the exact same generic image for every song.
2.  **Maximum Mean Discrepancy (MMD)**:
    *   A kernel-based statistical test (using Gaussian RBF kernels) ensures that the *distribution* of mapped audio vectors ($Z_{mapped}$) closely matches the *distribution* of true image vectors ($Z_{image}$).
3.  **Targeted Diversity Loss ($L_{div}$)**: 
    *   To further combat mode-collapse (a common VAE issue where all generated images look identical), a diversity penalty repels batch samples from one another: $-E[||z_1 - z_2||]$. This forces the network to utilize the full extent of the 512-dimension space.
4.  **KL Divergence ($L_{kl}$)**: 
    *   Standard VAE regularization is applied with a warm-up schedule ($\beta$-annealing) to keep the mapped latent space continuous and uniformly distributed without overwhelming the Cycle Consistency and MMD losses early in training.

---

## 📁 Repository Map

```text
audio-to-image-generation/
│
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Dependency manifest
│
├── datasets/                   # Training data directory
│   ├── audio_features/         
│   └── abstract_art/           
│
├── scripts/
│   ├── inference.py            # Headless inference logic used by the UI
│   ├── extract_audio_features.py 
│   ├── generate_image.py       # Legacy CLI generation script
│   └── generate_image_from_mp3.py
│
├── models/
│   └── autoencoder/
│       ├── audio_vae.py        # 1D Conv architecture (Audio)
│       ├── image_vae.py        # 2D Conv architecture (Images)
│       ├── mapping_network.py  # Forward/Inverse MLP Translation
│       └── datasets.py         # PyTorch DataLoader classes
│
├── training/                   # Model training scripts
│   ├── train_audio_vae.py      
│   ├── train_image_vae.py      
│   └── train_mapping_network.py
│
├── tmodels/                    # Directory for PyTorch `.pth` weights
└── generated_images/           # Output directory for locally saved CLI runs
```

---

## 🔬 Training from Scratch

If you are a researcher or developer looking to train the models from scratch:

1. **Train Audio Latent Space**: `python training/train_audio_vae.py`
2. **Train Image Latent Space**: `python training/train_image_vae.py`
   *Note: Ensure your `datasets/abstract_art/` folder is populated with images before running this.*
3. **Train the Translation Bridge**: `python training/train_mapping_network.py`
   *This trains both the forward and inverse mapping networks using the multi-objective loss architecture (KL, Cycle, MMD) described above.*

---

## 🤝 Contributing

Contributions are welcome! If you’d like to contribute:
1. Fork the repository.
2. Create a semantic feature branch.
3. Submit a pull request with a detailed description of your architectural changes.

## 📄 License

This project is open-source and available under the MIT License.
