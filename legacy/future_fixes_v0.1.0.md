# Future Fixes & Roadmap (v0.1.0)

With the successful deployment of **v0.1.0**—which introduced the Streamlit web application, YouTube integration, and inference modularity—the following architectural fixes and enhancements are scheduled for the next major release (v0.2.0).

---

## 1. Model Replacements for Output Diversity
**Issue:** The current `ImageVAE` (standard continuous VAE) and `MappingNetwork` combination suffers from posterior collapse, resulting in somewhat homogenized, similar-looking abstract art regardless of the audio input variation.

**Proposed Fixes:**
- **Vector Quantized VAE (VQ-VAE):** Replace the standard continuous Image VAE with a VQ-VAE constraint. This eliminates posterior collapse by forcing the latent space into discrete, learned codebook vectors, yielding drastically sharper textural outputs.
- **Diffusion Model Transition:** Longer-term, replace the decoder pipeline entirely with a lightweight unconditional-to-conditional Diffusion Model or a latent consistency model (LCM).
- **Contrastive Learning (CLIP-Style):** Replace the current MSE/Cycle-Consistency mapping architecture with a pure Contrastive Loss mechanism to strictly enforce separation of representations for distinct audio tracks.

## 2. Audio Feature Granularity
**Issue:** The pre-processing pipeline standardizes audio arrays to fixed lengths, which might crop important thematic segments of a song or add silence padding, reducing inference accuracy on long/complex tracks.

**Proposed Fixes:**
- **Sliding Window Inference:** Implement a rolling context window for audio tracks, allowing the model to generate a sequence of continuous Latent Frames.
- **Mel-Spectrogram Embeddings:** Ditch the raw handcrafted `librosa` features (Tempo, RMS) in favor of deep embeddings directly from a pre-trained Audio backbone (e.g., `CLAP` or `AudioMAE`).

## 3. Codebase Architecture & MLOps
**Issue:** The project still relies on manually tweaking local variables in `training/` scripts.

**Proposed Fixes:**
- **Centralized Config System:** Implement `Hydra` or `PyYAML` to control `LATENT_DIM`, `BATCH_SIZE`, and `LEARNING_RATE` externally. This ensures parity across training and inference.
- **Experiment Tracking:** Integrate `Weights & Biases (wandb)` or `TensorBoard` to visually track cycle-consistency loss curves and observe live output image variance over epochs.
- **Package Refactor:** Establish a `pyproject.toml` so the `models` and `scripts` directories can be installed seamlessly via `pip install -e .`.

---
*Documented upon the release of v0.1.0 Phase 1.*
