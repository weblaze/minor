import io
import zipfile
from pathlib import Path

import librosa
import numpy as np

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg")


def cut_windows(y, sr, n_windows=3, window_s=10.0):
    """Cut `n_windows` contiguous windows from the centre of a signal.

    Returns an array (n_windows, window_s * sr), or None if the signal is shorter
    than the requested total length (up to 2 % missing samples are zero-padded,
    which covers FMA clips that decode a few frames short of 30 s).
    """
    win = int(round(window_s * sr))
    total = n_windows * win
    if len(y) < total:
        if len(y) < 0.98 * total:
            return None
        y = np.pad(y, (0, total - len(y)))
    start = (len(y) - total) // 2
    return y[start:start + total].reshape(n_windows, win).astype(np.float32)


def load_windows(source, sr, n_windows=3, window_s=10.0):
    y, _ = librosa.load(source, sr=sr, mono=True)
    return cut_windows(y, sr, n_windows, window_s)


def iter_audio(source):
    """Yield (track_id, loadable) for every audio file in a directory tree or a zip archive.

    Tracks are yielded in sorted order of their path, so a run can be resumed.
    A zip archive is read member by member and is never extracted to disk.
    """
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(n for n in archive.namelist() if n.lower().endswith(AUDIO_EXTENSIONS))
            for name in names:
                yield Path(name).stem, io.BytesIO(archive.read(name))
    else:
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
        for path in paths:
            yield path.stem, str(path)
