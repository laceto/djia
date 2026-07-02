"""Curation Engine (Step 4) — Extract danceability: energy, tags, complexity."""

import numpy as np
import librosa
from typing import List
from ..features.schema import CurationResult


def compute_danceability(
    rms_energy: np.ndarray,
    bpm: float,
    swing_score: float
) -> float:
    """
    Compute danceability score (0=not danceable, 1=highly danceable).

    Danceability combines:
    - BPM in sweet spot (110-135 BPM is most danceable for techno)
    - Consistent/predictable groove (low swing for industrial, high for house)
    - Moderate energy variation (not too flat, not chaotic)

    Args:
        rms_energy: RMS energy array
        bpm: Track BPM
        swing_score: Swing score (0-1)

    Returns:
        danceability: 0.0-1.0 score
    """
    # BPM component: peak at 120 BPM, falls off outside 110-135
    bpm_component = 1.0 - np.clip(np.abs(bpm - 120) / 30, 0, 1)

    # Energy consistency component: regular variations are good
    # Too flat (CV < 0.1) = hypnotic but maybe boring
    # Too chaotic (CV > 0.5) = too unpredictable
    energy_mean = np.mean(rms_energy) + 1e-8
    energy_cv = np.std(rms_energy) / energy_mean
    energy_component = 1.0 - np.clip(np.abs(energy_cv - 0.3) / 0.4, 0, 1)

    # Groove component: both swing and no-swing can be danceable
    # Midpoint (0.5) is optimal
    groove_component = 1.0 - np.abs(swing_score - 0.5)

    # Combine (weighted average)
    danceability = (bpm_component * 0.4 + energy_component * 0.3 + groove_component * 0.3)

    return float(np.clip(danceability, 0, 1))


def classify_energy_profile(rms_energy: np.ndarray) -> str:
    """
    Classify energy profile into type: "flat", "dynamic", or "gradual".

    Args:
        rms_energy: RMS energy array

    Returns:
        energy_type: "flat", "dynamic", or "gradual"
    """
    # Normalize
    rms_norm = (rms_energy - np.mean(rms_energy)) / (np.std(rms_energy) + 1e-8)

    # Compute energy statistics
    energy_cv = np.std(rms_energy) / (np.mean(rms_energy) + 1e-8)

    # Compute trend (linear regression slope)
    x = np.arange(len(rms_energy))
    z = np.polyfit(x, rms_norm, 1)
    trend_slope = z[0]

    # Compute autocorrelation for periodicity
    autocorr = np.correlate(rms_norm, rms_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Check for periodic peaks (indicator of dynamic, beat-aligned energy)
    max_lag = min(len(autocorr) // 2, 200)
    periodic_peaks = len(np.where(autocorr[1:max_lag] > 0.3)[0])

    # Classify
    if energy_cv < 0.15:
        return "flat"  # Low variation = hypnotic/steady-groove
    elif abs(trend_slope) > 0.01:
        return "gradual"  # Clear upward/downward trend
    elif periodic_peaks > 5:
        return "dynamic"  # Regular peak patterns
    else:
        # Default to dynamic if energy variation is moderate
        return "dynamic" if energy_cv > 0.25 else "flat"


def compute_spectral_flux(y: np.ndarray, sr: int, hop_length: int = 512) -> float:
    """
    Compute spectral flux (measure of spectral change/complexity).

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length

    Returns:
        flux_mean: Mean spectral flux (normalized)
    """
    S = librosa.stft(y, hop_length=hop_length)
    magnitude = np.abs(S)

    # Normalize
    magnitude_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-8)

    # Spectral flux (requires at least 2 frames)
    if magnitude_norm.shape[1] < 2:
        # If too few frames, return 0 (no change)
        return 0.0

    flux = np.sqrt(np.sum(np.diff(magnitude_norm, axis=1)**2, axis=0))

    # Normalize
    flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-8)

    return float(np.mean(flux_norm))


def generate_semantic_tags(
    danceability: float,
    energy_type: str,
    bpm: float,
    swing_score: float,
    brightness: float,
    spectral_flux: float
) -> List[str]:
    """
    Generate semantic tags for the track.

    Args:
        danceability: Danceability score (0-1)
        energy_type: "flat", "dynamic", "gradual"
        bpm: Track BPM
        swing_score: Swing score (0-1)
        brightness: Brightness (0-1)
        spectral_flux: Spectral complexity (0-1)

    Returns:
        tags: List of semantic labels
    """
    tags = []

    # Energy tags
    if danceability > 0.7:
        tags.append("high-energy")
    elif danceability < 0.4:
        tags.append("low-energy")
    else:
        tags.append("moderate-energy")

    # Groove tags
    if energy_type == "flat":
        tags.append("steady-groove")
    elif energy_type == "dynamic":
        tags.append("peak-heavy")
    else:
        tags.append("builds")

    # Swing tags
    if swing_score > 0.6:
        tags.append("groovy")
    elif swing_score < 0.3:
        tags.append("industrial")
    else:
        tags.append("tight")

    # Brightness tags
    if brightness > 0.7:
        tags.append("bright")
    elif brightness < 0.3:
        tags.append("dark")

    # BPM tags
    if bpm < 100:
        tags.append("slow")
    elif bpm < 110:
        tags.append("deep-house")
    elif bpm < 130:
        tags.append("techno")
    elif bpm < 150:
        tags.append("hard-techno")
    else:
        tags.append("rave")

    # Complexity tags
    if spectral_flux > 0.6:
        tags.append("complex")
    elif spectral_flux < 0.2:
        tags.append("minimalist")

    return tags


def analyze_curation(
    y: np.ndarray,
    sr: int,
    bpm: float,
    swing_score: float,
    brightness: float,
    hop_length: int = 512
) -> CurationResult:
    """
    Complete curation analysis (danceability, energy, tags).

    Args:
        y: Audio waveform
        sr: Sample rate
        bpm: Track BPM (from groove engine)
        swing_score: Swing score (from groove engine)
        brightness: Brightness (from mood engine)
        hop_length: Hop length for analysis

    Returns:
        CurationResult with danceability, energy curve, type, and tags
    """
    # Compute energy curve
    energy_curve = librosa.feature.rms(y=y)[0]

    # Compute danceability
    danceability = compute_danceability(energy_curve, bpm, swing_score)

    # Classify energy profile
    energy_type = classify_energy_profile(energy_curve)

    # Compute spectral flux
    spectral_flux = compute_spectral_flux(y, sr, hop_length)

    # Compute complexity score (combination of flux and energy variation)
    energy_cv = np.std(energy_curve) / (np.mean(energy_curve) + 1e-8)
    complexity_score = (spectral_flux + np.clip(energy_cv / 0.5, 0, 1)) / 2

    # Generate tags
    tags = generate_semantic_tags(
        danceability,
        energy_type,
        bpm,
        swing_score,
        brightness,
        spectral_flux
    )

    return CurationResult(
        danceability=danceability,
        energy_curve=energy_curve,
        energy_type=energy_type,
        semantic_tags=tags,
        complexity_score=complexity_score
    )
