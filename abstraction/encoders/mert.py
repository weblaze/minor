import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_ID = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000
N_LAYERS = 13
DIM = 768


class MertEncoder:
    """Frozen MERT-v1-95M. Returns one time-averaged vector per hidden layer.

    `revision` pins a specific commit of the model repository. The official
    weights are stored as a pickle file, which transformers >= 5 refuses to
    load on torch < 2.6; on such machines a revision that carries
    `model.safetensors` has to be given.
    """

    def __init__(self, device, revision=None):
        self.device = device
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_ID, revision=revision, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MERT_ID, revision=revision, trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def embed(self, windows):
        """windows: array (n, samples) at 24 kHz -> tensor (n, 13, 768)."""
        inputs = self.processor(list(windows), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self.model(**inputs, output_hidden_states=True).hidden_states
        return torch.stack(hidden, dim=1).mean(dim=2).float().cpu()
