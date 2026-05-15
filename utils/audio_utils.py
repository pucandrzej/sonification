from pathlib import Path
import numpy as np
from scipy.io import wavfile

from .constants import SEMITONES_NO, FS, C3, NOTE_OFFSETS, SCALES, SWING_CLIP


def base_frequency(note="C", octave_shift=0):
    """Return the base frequency in Hz for a note and octave shift relative to C3."""
    semitones = NOTE_OFFSETS[note] + SEMITONES_NO * octave_shift
    return C3 * (2 ** (semitones / SEMITONES_NO))


def nearest_allowed_semitone(x, scale_name):
    """Snap a semitone value to the nearest degree allowed by the given scale."""
    scale = SCALES[scale_name]
    if scale is None:
        return x
    if scale_name == "chromatyczna":
        return round(x)
    allowed = sorted(set(scale + [SEMITONES_NO]))
    return min(allowed, key=lambda s: abs(s - x))


def value_to_frequency(value, note, global_octave, track_octave, scale_name):
    """Map a normalised value [0, 1] to a frequency in Hz using the given scale and octave settings."""
    value = float(np.clip(value, 0.0, 1.0))
    semitone = value * SEMITONES_NO
    semitone = nearest_allowed_semitone(semitone, scale_name)
    base = base_frequency(note, global_octave + track_octave)
    return base * (2 ** (semitone / SEMITONES_NO))


def sixteenth_duration(bpm, swing_percent, step_index):
    """Return the duration in seconds of a sixteenth note, applying swing if non-zero."""
    bpm = max(1.0, float(bpm))
    base = 60.0 / bpm / 4.0
    swing_percent = float(np.clip(swing_percent, SWING_CLIP[0], SWING_CLIP[1]))
    if swing_percent == SWING_CLIP[0]:
        return base
    pair_total = 2.0 * base
    long_part = pair_total * swing_percent / 100.0
    short_part = pair_total - long_part
    return long_part if step_index % 2 == 0 else short_part


def soft_limiter(y, drive=1.2, ceiling=0.92):
    """Soft-clip audio via tanh saturation, scaling output to the given ceiling amplitude."""
    y = np.asarray(y, dtype=np.float32)
    return (np.tanh(drive * y) / np.tanh(drive) * ceiling).astype(np.float32)


def normalize_audio(y, peak=0.95):
    """Normalise audio so its peak absolute amplitude equals the given target level."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y
    m = float(np.max(np.abs(y)))
    if m > 0:
        y = y / m * peak
    return y.astype(np.float32)


def next_numbered_wav(prefix="loop", folder="EXPORTS"):
    """Return the next available numbered WAV path (e.g. EXPORTS/loop_003.wav) without overwriting existing files."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    i = 1
    while True:
        path = folder / f"{prefix}_{i:03d}.wav"
        if not path.exists():
            return path
        i += 1


def save_wav(path, y, fs=FS):
    """Apply a final soft limiter and write audio to a 32-bit float WAV file."""
    y = soft_limiter(y, drive=1.1, ceiling=0.92)
    wavfile.write(str(path), fs, y.astype(np.float32))


def semitone_to_frequency(semitone, note, octave_shift=0):
    """
    Convert an absolute semitone offset to frequency.

    semitone:
        0 = tonic
        12 = octave above tonic
    """
    base = base_frequency(note, octave_shift)
    return base * (2 ** (semitone / SEMITONES_NO))
