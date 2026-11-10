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

    The 13 hidden states are the input of the first transformer layer and the
    output of each of the 12 layers. With transformers 5 the model code of the
    repository returns `hidden_states = None`; in that case the same tensors
    are read with forward hooks on the transformer layers. `self.route` records
    which path was used ("output" or "hooks").
    """

    def __init__(self, device, revision=None):
        self.device = device
        self.route = None
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            MERT_ID, revision=revision, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            MERT_ID, revision=revision, trust_remote_code=True).to(device).eval()

    def _hidden_states(self, inputs):
        if self.route != "hooks":
            hidden = getattr(self.model(**inputs, output_hidden_states=True), "hidden_states", None)
            if hidden is not None:
                self.route = "output"
                return hidden
            self.route = "hooks"
        captured = []

        def before_first(module, args, kwargs):
            captured.append(args[0] if args else kwargs["hidden_states"])

        def after_layer(module, args, output):
            captured.append(output[0] if isinstance(output, (tuple, list)) else output)

        layers = self.model.encoder.layers
        handles = [layers[0].register_forward_pre_hook(before_first, with_kwargs=True)]
        handles += [layer.register_forward_hook(after_layer) for layer in layers]
        try:
            self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @torch.no_grad()
    def embed(self, windows):
        """windows: array (n, samples) at 24 kHz -> tensor (n, 13, 768)."""
        inputs = self.processor(list(windows), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        hidden = self._hidden_states(inputs)
        if len(hidden) != N_LAYERS:
            raise RuntimeError(f"expected {N_LAYERS} hidden states, got {len(hidden)}")
        return torch.stack(list(hidden), dim=1).mean(dim=2).float().cpu()
