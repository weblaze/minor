"""Cut a 10 s clip from the middle of each track in tracks.txt (the middle window of the MERT extraction).

Writes <out>/<track id>.mp3 (mono, 22.05 kHz, 96 kbit/s) with ffmpeg.

Usage:
  python scripts/judging/clips.py --source <FMA directory> --out <dir>
"""
import argparse
import subprocess
from pathlib import Path

import librosa

HERE = Path(__file__).resolve().parent
LENGTH_S = 10.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracks", default=str(HERE / "tracks.txt"))
    args = parser.parse_args()
    wanted = set(Path(args.tracks).read_text().split())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {p.stem: p for p in Path(args.source).rglob("*.mp3") if p.stem in wanted}
    missing = sorted(wanted - set(paths))
    for track, path in sorted(paths.items()):
        duration = librosa.get_duration(path=str(path))
        start = max(0.0, (duration - LENGTH_S) / 2)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-t", str(LENGTH_S), "-i", str(path),
                        "-ac", "1", "-ar", "22050", "-b:a", "96k", str(out / f"{track}.mp3")], check=True)
        print(track, f"{duration:.2f} s, clip from {start:.2f} s")
    print(f"{len(paths)} clips written, missing: {missing}")


if __name__ == "__main__":
    main()
