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

    # Convert to Camelot position (1-12)
    # Major: C=8B, G=9B, D=10B, A=11B, E=12B, B=1B, etc.
    # Minor: A=8A, E=9A, B=10A, F#/Gb=11A, C#/Db=12A, G#/Ab=1A, etc.

    if key_type == "major":
        # Major key Camelot: starts at C=8B
        camelot_pos = ((note_num * 7) % 12) + 1  # Chromatic circle of 5ths
        camelot_type = "B"
    else:
        # Minor key Camelot: starts at A=8A
        camelot_pos = (((note_num - 9) * 7) % 12) + 1  # Offset from A
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


def analyze_mood(y: np.ndarray, sr: int) -> MoodResult:
    """
    Complete mood analysis (key, camelot, brightness, confidence).

    Args:
        y: Audio waveform
        sr: Sample rate

    Returns:
        MoodResult with key, camelot, brightness, and confidence
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

    return MoodResult(
        key=key_name,
        camelot_key=camelot_key,
        brightness=brightness,
        key_confidence=key_confidence
    )
