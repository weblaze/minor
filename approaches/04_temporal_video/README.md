# Approach 04 — Temporal Conditioning / Music Video Generation

**Priority 3.** The portfolio piece: a full song becomes a sequence of
AI-generated images synchronized to the music. No new training — this reuses
the approach 02 diffusion model, so it only works after approach 02 is trained.

## How it works

```
song ──4s segments──► CLAP embedding per segment
                         │ (same initial noise for every segment ⇒ coherence)
                         ▼
              diffusion keyframe latent per segment
                         │ slerp interpolation between neighbors
                         ▼
              decoded frames ──ffmpeg──► .mp4 with original audio muxed in
```

- **Temporal coherence**: all segments share one initial noise tensor, so
  adjacent keyframes differ only through their conditioning — visuals evolve
  with the music rather than jumping randomly.
- **Dream segments** (`--dream-every N`): every Nth segment is null-conditioned;
  the model wanders between audio cues — the dreaming feature woven into video.

## Your part — do this next

```bash
pip install "imageio[ffmpeg]"   # one-time (or pip install -e .[video])

# requires approach 02's trained UNet (and VAE for kind: own)
python approaches/04_temporal_video/generate_video.py --audio path/to/song.mp3

# with dream segments every 4th window
python approaches/04_temporal_video/generate_video.py --audio song.mp3 --dream-every 4
```

GTX 1650 is fine here: per-keyframe diffusion at 16×16 latents takes seconds;
a 3-minute song is ~45 keyframes plus CPU-side decoding/assembly.

**Tuning for aesthetics** (config.yaml): `fps` and `frames_per_transition`
control pacing (12/12 = one keyframe per second of video); `segment_seconds`
controls how fast visuals react to the music; `guidance_scale` controls how
literal the conditioning is.
