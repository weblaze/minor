"""Material for the phase 1 human judging page (the human pilot of the S1 plan, section 4.4).

Selects `--n` test tracks (seeded), copies the image each system gives them and writes `trials.json`:

  match     two-alternative choice: one clip, the image a system gives that track and the image the same
            system gives another selected track; the judge picks the image that fits the clip. Chance 1/2.
  triplet   three selected tracks. Audio form: three clips, the judge picks the two that sound most alike.
            Image form, per system: the three images, the judge picks the two that look most alike.
            Chance agreement between the two forms 1/3.

Systems: S1 and Q1 (generated images, from `evaluate.py --save-images`), F1 (assigned training painting,
from `features.npz` of an F1 evaluation). The clips are cut on Kaggle by `clips.py` from `tracks.txt`.

Usage:
  python scripts/judging/prepare.py --s1 <s1 eval run> --q1 <q1 eval run> --f1 <f1 eval run> --out <dir>
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--q1", required=True)
    parser.add_argument("--f1", required=True)
    parser.add_argument("--art", default="datasets/abstract_art")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--triplets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)

    with np.load(Path(args.f1) / "features.npz") as data:
        painting_of = {str(t): str(n) for t, n in zip(data["track_ids"], data["painting_names"])}
    have = {name: {p.stem for p in (Path(run) / "images").glob("*.png")} for name, run in (("s1", args.s1), ("q1", args.q1))}
    candidates = sorted(set(painting_of) & have["s1"] & have["q1"])
    chosen = sorted(rng.choice(candidates, size=args.n, replace=False).tolist())
    (HERE / "tracks.txt").write_text("\n".join(chosen) + "\n")

    tracks = []
    for track in chosen:
        images = {}
        for name, run in (("s1", args.s1), ("q1", args.q1)):
            target = out / "img" / name / f"{track}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(run) / "images" / f"{track}.png", target)
            images[name] = target.relative_to(out).as_posix()
        target = out / "img" / "f1" / f"{track}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        painting = ImageOps.exif_transpose(Image.open(Path(args.art) / painting_of[track])).convert("RGB")
        painting.thumbnail((320, 320))
        painting.save(target, quality=85)
        images["f1"] = target.relative_to(out).as_posix()
        tracks.append({"id": track, "clip": f"clips/{track}.mp3", "images": images, "painting": painting_of[track]})

    match = []
    for system in ("s1", "q1", "f1"):
        for i, track in enumerate(chosen):
            other = chosen[(i + int(rng.integers(1, len(chosen)))) % len(chosen)]
            match.append({"kind": "match", "system": system, "track": track, "foil": other})
    triplets = []
    for _ in range(args.triplets):
        triplets.append(sorted(rng.choice(chosen, size=3, replace=False).tolist()))
    trials = {"systems": {"s1": "S1 (generated, VAE)", "q1": "Q1 (generated, VAE)", "f1": "F1 (assigned painting)"},
              "sources": {"s1": str(args.s1), "q1": str(args.q1), "f1": str(args.f1)},
              "seed": args.seed, "tracks": tracks, "match": match, "triplets": triplets}
    (out / "trials.json").write_text(json.dumps(trials, indent=1))
    print(f"{len(tracks)} tracks, {len(match)} match trials, {len(triplets)} triplets -> {out}")


if __name__ == "__main__":
    main()
