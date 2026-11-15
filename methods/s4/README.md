# S4: natural audio-visual statistics

S4 takes the relation between sound and image from a model that was pretrained on video (audio track and frames of the same clip) without text or labels. The model is CAV-MAE-Scale++ (Gong et al., ICLR 2023), checkpoint `audio_model.21.pth`, used frozen. Encoder code: `abstraction/encoders/cav_mae.py`.

State: probe only. Nothing is trained and no generator is driven yet.

## Stages

| Stage | Output |
|---|---|
| `extract.py` | CAV-MAE embeddings of every track (one per window) and of every painting, in `cav_mae.embeddings` |
| `probe.py` | `results.json`: agreement of CAV-MAE with MERT and with DINOv2, statistics of the track-painting similarity, S4 as a retrieval bridge, and scores of painting assignments made by other methods (D-S4) |

```bash
python scripts/run.py s4 extract --source datasets/fma_small --checkpoint models/cav_mae/audio_model.21.pth
python scripts/run.py s4 probe --assignments "runs/f1_eval_*/features.npz"
python scripts/kaggle/launch.py push scripts/kaggle/jobs/s4_probe.yaml
```

`extract.py` needs `torchaudio` (filterbank) and the audio files; both are available on Kaggle. `probe.py` needs only the embeddings, the MERT embeddings and the DINOv2 features of the art set, and runs on a CPU.

## Inputs to the model

- Audio: the windows of the MERT extraction (three contiguous 10 s windows from the centre of the track), 16 kHz, 128-bin filterbank with the settings of the CAV-MAE repository, padded from 998 to 1024 frames, normalised with mean -5.081 and standard deviation 4.4849.
- Images: shorter side scaled to 224, centre crop, ImageNet normalisation. Paintings that are not square lose their margins; DINOv2 features of the other methods are computed on the whole painting scaled to a square.
- Embedding: mean over the tokens after the joint layers, as in the retrieval evaluation of the CAV-MAE repository. A track is the mean of its window embeddings.

## Reading the probe

- `encoder_agreement`: whether CAV-MAE orders the test tracks as MERT does and the test paintings as DINOv2 does.
- `cross_modal.variance_shares`: the share of the variance of the similarity matrix that belongs to the track alone, to the painting alone, and to the pair. Only the pair share can relate particular tracks to particular paintings.
- `cross_modal.window_consistency`: whether the pair part of the similarity is a property of the track (two windows of one track agree) or noise.
- `painting`, `painting_centred`: each test track receives its most similar training painting; scored like the `painting` block of F1.
- `d_s4`: for an assignment of paintings to tracks by another method, the mean similarity of the assigned pairs against random re-assignments of the same paintings (`z`, `p_greater`). Two assignments of opposite orientation can be compared by this score.

The model was pretrained on general video (AudioSet and VGGSound clips). Abstract paintings and studio music are far from that material; the probe measures whether its similarity carries any usable signal for them before a bridge is built on it.
