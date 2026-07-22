"""Mood Engine (Step 3) — Extract tonality: key, camelot, brightness, confidence."""

import numpy as np
import librosa
from typing import Tuple
from ..features.schema import MoodResult


# Note names for key detection
NOTE_NAMES = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
MINOR_OFFSETS = [0, 9, 2, 11, 4, 1, 10, 5, 2, 9, 4, 11]  # Relative minor offsets
MAJOR_OFFSETS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # Relative major offsets


def detect_key_from_chroma(chroma: np.ndarray) -> Tuple[str, str, float]:
    """
    Detect musical key from chromagram.

    Uses template matching with major/minor profiles.

    Args:
        chroma: Chromagram (12 x frames)

    Returns:
        (key_name, key_type): e.g., ("A", "minor")
        confidence: 0-1 confidence score
    """
    # Average chroma across time
    chroma_mean = np.mean(chroma, axis=1)

    # Normalize
    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-8)

    # Major key profile (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]) / 100

    # Minor key profile
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]) / 100

    # Test all 12 keys
    best_key = 0
    best_score = 0
    best_type = "major"

    # Test major keys
    for shift in range(12):
        rotated = np.roll(chroma_norm, shift)
        correlation = np.dot(rotated, major_profile)
        if correlation > best_score:
            best_score = correlation
            best_key = shift
            best_type = "major"

    # Test minor keys
    for shift in range(12):
        rotated = np.roll(chroma_norm, shift)
        correlation = np.dot(rotated, minor_profile)
        if correlation > best_score:
            best_score = correlation
            best_key = shift
            best_type = "minor"

    # Confidence: normalized correlation score
    confidence = min(1.0, best_score * 2)  # Scale up for perceptual range

    key_name = NOTE_NAMES[best_key]
    return key_name, best_type, confidence


def convert_to_camelot(note: str, key_type: str) -> str:
    """
    Convert musical key to Camelot notation for DJ mixing.

    Camelot Wheel: 1A-12A (minor), 1B-12B (major)
    e.g., "A minor" -> "8A", "C major" -> "8B"

    Args:
        note: Note name (e.g., "A", "C#/Db")
        key_type: "major" or "minor"

    Returns:
        camelot_key: String in format "XA" or "XB" (e.g., "7A")
    """
    # Map note to number (0-11)
    note_to_num = {
        "C": 0, "C#/Db": 1, "D": 2, "D#/Eb": 3, "E": 4, "F": 5,
        "F#/Gb": 6, "G": 7, "G#/Ab": 8, "A": 9, "A#/Bb": 10, "B": 11
    }

    note_num = note_to_num.get(note, 0)

    # Convert to Camelot position (1-12) by walking the circle of fifths.
    # Both wheels are anchored at position 8 — C major = 8B, A minor = 8A — so the
    # circle-of-fifths index (0 for the anchor) is offset by +8, then wrapped to 1-12.
    # Major: C=8B, G=9B, D=10B, A=11B, E=12B, B=1B, F#/Gb=2B, C#/Db=3B, ...
    # Minor: A=8A, E=9A, B=10A, F#/Gb=11A, C#/Db=12A, G#/Ab=1A, D#/Eb=2A, A#/Bb=3A, F=4A, ...

    if key_type == "major":
        # Major key Camelot: anchored at C=8B
        camelot_pos = ((note_num * 7) % 12) + 8  # Chromatic circle of 5ths
        camelot_type = "B"
    else:
        # Minor key Camelot: anchored at A=8A (A is note_num 9)
        camelot_pos = (((note_num - 9) * 7) % 12) + 8  # Offset from A
        camelot_type = "A"

    # Clamp to 1-12
    camelot_pos = ((camelot_pos - 1) % 12) + 1

    return f"{int(camelot_pos)}{camelot_type}"


def compute_brightness(y: np.ndarray, sr: int) -> float:
    """
    Compute brightness (0=dark/subby, 1=bright/crisp).

    Brightness is measured as normalized spectral centroid.

    Args:
        y: Audio waveform
        sr: Sample rate

    Returns:
        brightness: 0.0-1.0 score
    """
    # Compute spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Mean spectral centroid
    mean_centroid = np.mean(spectral_centroid)

    # Nyquist frequency
    nyquist = sr / 2

    # Normalize to 0-1
    # Darkness: 0-2000 Hz -> brightness 0
    # Brightness: 8000-20000 Hz -> brightness 1
    brightness = np.clip((mean_centroid - 1000) / (nyquist - 1000), 0, 1)

    return float(brightness)


def compute_zero_crossing_rate(y: np.ndarray) -> float:
    """
    Compute zero-crossing rate: how rapidly the waveform crosses zero amplitude.

    Args:
        y: Audio waveform

    Returns:
        zcr_mean: 0.0-1.0ish (typically < 0.3). High-frequency noise and jagged,
        distorted "acid" sounds have a much higher ZCR than clean bass/pads.
    """
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(zcr))


def _sethares_dissonance(f1: float, f2: float) -> float:
    """Plomp-Levelt sensory dissonance between two pure tones (Sethares 1993 formula)."""
    b1, b2 = 3.5, 5.75
    xstar = 0.24
    s1, s2 = 0.0207, 18.96
    fmin = min(f1, f2)
    fdiff = abs(f2 - f1)
    s = xstar / (s1 * fmin + s2)
    return float(np.exp(-b1 * s * fdiff) - np.exp(-b2 * s * fdiff))


def compute_roughness(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_peaks: int = 6,
    max_frames: int = 200,
) -> float:
    """
    Compute timbral roughness (0=smooth/round, 1=rough/harsh) via a pragmatic
    approximation of the Plomp-Levelt psychoacoustic roughness model (Sethares 1993):
    for each analyzed frame, pick the top spectral peaks, sum the pairwise sensory
    dissonance between them (weighted by their relative amplitude), then average
    across frames and squash to 0-1.

    This favors severe modulation/distortion/close-packed harmonics (acid basslines,
    distorted industrial kicks, metallic synths) over clean sine-like tones (deep
    house bass, jazzy chords).

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length for the STFT
        n_fft: FFT window size
        n_peaks: Number of loudest spectral peaks to compare per frame
        max_frames: Cap on frames analyzed (subsampled evenly) — keeps this tractable
            on long tracks since it's O(n_peaks^2) per frame

    Returns:
        roughness: 0.0 (smooth/consonant) to 1.0 (rough/dissonant), tanh-squashed
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S.shape[1]
    if n_frames == 0:
        return 0.0

    stride = max(1, n_frames // max_frames)
    frame_roughness = []

    for i in range(0, n_frames, stride):
        spectrum = S[:, i]
        if spectrum.max() <= 0:
            continue
        k = min(n_peaks, len(spectrum))
        peak_idx = np.argpartition(spectrum, -k)[-k:]
        peak_freqs = freqs[peak_idx]
        peak_amps = spectrum[peak_idx]
        peak_amps = peak_amps / (peak_amps.max() + 1e-8)

        r = 0.0
        for a in range(len(peak_idx)):
            for b in range(a + 1, len(peak_idx)):
                r += peak_amps[a] * peak_amps[b] * _sethares_dissonance(peak_freqs[a], peak_freqs[b])
        frame_roughness.append(r)

    if not frame_roughness:
        return 0.0
    return float(np.tanh(np.mean(frame_roughness)))


def analyze_mood(y: np.ndarray, sr: int) -> MoodResult:
    """
    Complete mood analysis (key, camelot, brightness, confidence, ZCR, roughness).

    Args:
        y: Audio waveform
        sr: Sample rate

    Returns:
        MoodResult with key, camelot, brightness, confidence, zero-crossing rate,
        and timbral roughness
    """
    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Detect key
    key_note, key_type, key_confidence = detect_key_from_chroma(chroma)

    # Convert to Camelot
    camelot_key = convert_to_camelot(key_note, key_type)

    # Compute brightness
    brightness = compute_brightness(y, sr)

    # Create full key name
    key_name = f"{key_note} {key_type}"

    # Timbral character
    zero_crossing_rate = compute_zero_crossing_rate(y)
    roughness = compute_roughness(y, sr)

    return MoodResult(
        key=key_name,
        camelot_key=camelot_key,
        brightness=brightness,
        key_confidence=key_confidence,
        zero_crossing_rate=zero_crossing_rate,
        roughness=roughness,
    )
