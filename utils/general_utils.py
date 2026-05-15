import numpy as np
from PIL import Image
from datetime import datetime


def normalize_minmax(values):
    """Normalize data to [0,1] range"""
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def ms_to_datetime(ms):
    """Converts milliseconds timestamps to uman-readable datetime"""
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
