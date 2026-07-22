"""Spectrogram computation and on-disk persistence.

Not a scalar-feature engine like groove/phrasing/mood/curation — this saves the
full log-magnitude STFT array to disk (.npy) for later reload/inspection, keyed
by a stable identifier (track_id when available, filename stem otherwise).
"""

import logging
from pathlib import Path

import numpy as np
import librosa

logger = logging.getLogger(__name__)

DEFAULT_SPECTROGRAM_DIR = "data/spectrograms"


def compute_spectrogram(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute a log-magnitude (dB) STFT spectrogram.

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: STFT hop length (keep in sync with DSPConfig.hop_length)

    Returns:
        2D array (freq_bins, frames) in dB
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    return librosa.amplitude_to_db(S, ref=np.max)


def spectrogram_path(key, base_dir: str = DEFAULT_SPECTROGRAM_DIR) -> Path:
    """Deterministic .npy path for a given track_id or filename-stem key."""
    return Path(base_dir) / f"{key}.npy"


def save_spectrogram(S: np.ndarray, key, base_dir: str = DEFAULT_SPECTROGRAM_DIR) -> Path:
    """Save a computed spectrogram array to base_dir/{key}.npy, creating the dir if needed."""
    out_path = spectrogram_path(key, base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, S)
    return out_path


def compute_and_save_spectrogram(
    y: np.ndarray,
    sr: int,
    key,
    hop_length: int = 512,
    base_dir: str = DEFAULT_SPECTROGRAM_DIR,
) -> Path:
    """Compute a spectrogram and persist it in one call. Returns the saved path."""
    S = compute_spectrogram(y, sr, hop_length=hop_length)
    return save_spectrogram(S, key, base_dir=base_dir)
