import numpy as np
import torch

CLAP_DIM = 512


class ClapEncoder:
    """Wrapper around LAION CLAP (music_audioset checkpoint, HTSAT-base).

    Isolates the version-sensitive laion_clap loading boilerplate in one place.
    Produces 512-d audio embeddings.
    """

    def __init__(self, checkpoint_path, device="cuda"):
        import laion_clap

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(str(checkpoint_path))
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_file(self, audio_path):
        """Returns a [1, 512] tensor on the encoder's device."""
        with torch.no_grad():
            embed = self.model.get_audio_embedding_from_filelist(
                x=[str(audio_path)], use_tensor=True
            )
        return embed.to(self.device)

    def embed_files(self, audio_paths):
        """Returns a [N, 512] numpy array (use for batch feature extraction)."""
        with torch.no_grad():
            embeds = self.model.get_audio_embedding_from_filelist(
                x=[str(p) for p in audio_paths], use_tensor=False
            )
        return np.asarray(embeds)

    def embed_array(self, y, sr):
        """Embed a raw waveform segment. Returns a [1, 512] tensor.

        CLAP expects 48 kHz input; resamples if needed.
        """
        import librosa

        if sr != 48000:
            y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        y = torch.from_numpy(y).float().unsqueeze(0)
        with torch.no_grad():
            embed = self.model.get_audio_embedding_from_data(x=y, use_tensor=True)
        return embed.to(self.device)

    def embed_texts(self, texts):
        """Returns a [N, 512] numpy array of CLAP text embeddings."""
        with torch.no_grad():
            embeds = self.model.get_text_embedding(texts, use_tensor=False)
        return np.asarray(embeds)
