"""Model-free stem-proxy features.

Demucs stem separation is optional and network-gated (the weights download from a
host the analysis environment may block). These features approximate the same
"what's in the low end / how busy is the percussion / is there a vocal" questions
using only librosa/numpy — an STFT band-energy split plus per-band onset rates and
an HPSS harmonic-energy ratio. They run on every track, offline, and feed the
similarity/clustering feature vector (`matching.similarity.SIMILARITY_FEATURES`).

Everything here is best-effort: a missing/silent/corrupt signal yields ``None`` for
each feature (imputed to the corpus mean downstream), never an exception.
"""

import logging
from typing import Dict, Optional

import numpy as np
import librosa

logger = logging.getLogger(__name__)

# Spectral energy-ratio bands (Hz): share of total STFT power in each band.
SUB_BAND = (20.0, 60.0)
BASS_BAND = (60.0, 250.0)
# Harmonic energy in this band (post-HPSS) is a weak "vocal / tonal lead" proxy.
VOCAL_BAND = (200.0, 3500.0)

# Transient onset-rate bands (Hz) — the finer percussion split.
KICK_BAND = (30.0, 150.0)     # kick drum
PERC_BAND = (2000.0, 6000.0)  # snare / clap / mid percussion
HAT_BAND = (8000.0, 16000.0)  # hats / cymbals (clamped to Nyquist by the bin mask)

_N_FFT = 4096
_HOP = 1024

# The feature keys this module produces, in a stable order. Kept here so callers
# (worker, tests) can reason about the contract without importing internals.
STEM_PROFILE_KEYS = (
    "sub_ratio",
    "bass_ratio",
    "kick_rate",
    "perc_rate",
    "hat_rate",
    "vocal_presence",
)


def _empty_profile() -> Dict[str, Optional[float]]:
    return {k: None for k in STEM_PROFILE_KEYS}


def compute_stem_profile(y, sr: int) -> Dict[str, Optional[float]]:
    """Compute model-free stem-proxy features for one track.

    Args:
        y: mono (or multi-channel, mixed down here) audio samples.
        sr: sample rate in Hz.

    Returns:
        Dict with keys ``STEM_PROFILE_KEYS``:
            - ``sub_ratio`` / ``bass_ratio``: fraction of total STFT power in the
              20-60 Hz / 60-250 Hz bands (0-1).
            - ``kick_rate`` / ``perc_rate`` / ``hat_rate``: detected onsets per
              second within the low / mid / high transient bands.
            - ``vocal_presence``: fraction of *harmonic* (post-HPSS) power in the
              200-3500 Hz vocal band (0-1).
        Any feature that cannot be computed is ``None`` rather than raising.
    """
    profile = _empty_profile()

    if y is None or sr is None or sr <= 0:
        return profile

    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    if y.size == 0 or not np.any(np.isfinite(y)) or np.allclose(y, 0.0):
        return profile

    duration = y.size / float(sr)

    try:
        stft = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
        total = float(stft.sum())

        def band_ratio(lo: float, hi: float) -> Optional[float]:
            if total <= 0:
                return None
            mask = (freqs >= lo) & (freqs < hi)
            return float(stft[mask].sum()) / total

        def onset_rate(lo: float, hi: float) -> Optional[float]:
            if duration <= 0:
                return None
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return 0.0
            env = stft[mask].sum(axis=0)
            peak = float(env.max())
            if peak <= 0:
                return 0.0
            env = env / peak
            onsets = librosa.onset.onset_detect(
                onset_envelope=env, sr=sr, hop_length=_HOP
            )
            return len(onsets) / duration

        profile["sub_ratio"] = band_ratio(*SUB_BAND)
        profile["bass_ratio"] = band_ratio(*BASS_BAND)
        profile["kick_rate"] = onset_rate(*KICK_BAND)
        profile["perc_rate"] = onset_rate(*PERC_BAND)
        profile["hat_rate"] = onset_rate(*HAT_BAND)

        # Vocal presence: harmonic energy share in the vocal band, on the
        # HPSS-harmonic component so the wideband transient wash of drums doesn't
        # masquerade as vocal content.
        y_harmonic = librosa.effects.hpss(y)[0]
        stft_h = np.abs(librosa.stft(y_harmonic, n_fft=_N_FFT, hop_length=_HOP)) ** 2
        total_h = float(stft_h.sum())
        if total_h > 0:
            mask = (freqs >= VOCAL_BAND[0]) & (freqs < VOCAL_BAND[1])
            profile["vocal_presence"] = float(stft_h[mask].sum()) / total_h
    except Exception as e:  # noqa: BLE001 - best-effort, mirrors the DSP engines
        logger.warning(f"stem profile computation failed: {e}")
        return _empty_profile()

    return profile
