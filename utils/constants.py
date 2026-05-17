import os

#  AUDIO CONSTANTS

PREFILL = 8  # number of steps to buffer ahead in producer-consumer logic

FS = 44100
C3 = 130.8

SEMITONES_NO = 12  # 1 octave = 12 semitones

NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "H": 11,
}

SCALES = {
    "dowolny ton": None,
    "chromatyczna": list(range(SEMITONES_NO)),
    "dur": [0, 2, 4, 5, 7, 9, 11],
    "moll": [0, 2, 3, 5, 7, 8, 10],
    "pentatoniczna": [0, 2, 4, 7, 9],
    "cygańska": [0, 2, 3, 6, 7, 8, 11],
}

SWING_CLIP = (50.0, 75.0)  # 50 = straight, 75 = maximum swing

# DATA LOADER CONSTANTS

VS_CURRENCY = "usd"
DAYS = 1
INTERVAL = None
TOP_N = 10

API_TIMEOUT = 25
API_SLEEP_TIME_BETWEEN_CALLS = 1

# Paths

MARKET_DATA_FILE = os.path.join("MARKET_SNAPSHOT", "market_data.json")
os.makedirs("MARKET_SNAPSHOT", exist_ok=True)
SECRETS_FILE = "secrets.json"


# Audio engine defaults

STREAM_BLOCKSIZE = (
    2058  # number of frames (frame is one sample per channel) per audio callback block.
)

AMP_TOTAL = 0.55  # peak amplitude budget shared across all active tracks (0.0 – 1.0).

SOFT_LIMITER_DRIVE = (
    1.15  # input gain applied inside the soft limiter before the ceiling clamp.
)

SOFT_LIMITER_CEILING = 0.86  # output ceiling of the soft limiter (0.0 – 1.0).

# UI defaults

DEFAULT_NOTE = "C"

DEFAULT_GLOBAL_OCTAVE = 0

DEFAULT_SCALE = "chromatyczna"

DEFAULT_WAVE = "sinusoida"

DEFAULT_DATA_MODE = "kurs"

DEFAULT_BPM = 120.0

WAV_EXPORT_PREFIX = "notowania_loop"  # filename prefix for exported WAV loops

PLOT_YLIM = (
    -0.05,
    1.05,
)  # Y-axis limits for the normalized data plot, with a small margin on each side

TRACK_OCTAVE_VALUES = [
    -1,
    0,
    1,
]  # allowed per-track octave offsets shown in each track's combo-box

GLOBAL_OCTAVE_VALUES = [
    -2,
    -1,
    0,
    1,
    2,
]  # allowed global octave offsets shown in the global-octave combo-box

BPM_RANGE = (80, 180)  # BPM values for the tempo slider

# ============================================================
# IMAGE SONIFICATION CONSTANTS
# ============================================================

THUMBNAIL_SIZE = (420, 260)

IMAGE_QUANTIZATION_LEVELS = 16

STEPS_OPTIONS = [16, 32, 64]

OCTAVE_RANGE_OPTIONS = [1, 2, 3]

GREYSCALE_NORMALIZATION_CONST = 255.0
