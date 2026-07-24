"""Mood Engine (Step 3) — Extract tonality: key, camelot, brightness, confidence."""

import logging
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


def camelot_to_open_key(camelot: str) -> str:
    """Convert a Camelot code to Open Key Notation (used by DJUCED, Rekordbox "Open Key").

    Open Key numbers the same wheel rotated -7 from Camelot, with letters
    'm' (minor) / 'd' (major, "dur") instead of 'A' / 'B':
      A minor  = 8A  -> 1m
      C major  = 8B  -> 1d
      C#/Db minor = 12A -> 5m   (this is DJUCED's "5m" for Pak Pak)

    Args:
        camelot: Camelot code like "12A" or "8B" (case-insensitive letter).

    Returns:
        Open Key code like "5m" or "1d".
    """
    camelot = camelot.strip()
    num, letter = int(camelot[:-1]), camelot[-1].upper()
    if letter not in ("A", "B") or not 1 <= num <= 12:
        raise ValueError(f"Invalid Camelot code: {camelot!r}")
    open_num = ((num - 8) % 12) + 1
    return f"{open_num}{'m' if letter == 'A' else 'd'}"


def open_key_to_camelot(open_key: str) -> str:
    """Convert Open Key Notation (DJUCED, e.g. "5m") to a Camelot code (e.g. "12A").

    Inverse of camelot_to_open_key: Camelot number = Open Key number + 7 (wheel-wrapped),
    with 'm' -> 'A' (minor) and 'd' -> 'B' (major).

    Args:
        open_key: Open Key code like "5m" or "1d" (case-insensitive letter).

    Returns:
        Camelot code like "12A" or "8B".
    """
    open_key = open_key.strip()
    num, letter = int(open_key[:-1]), open_key[-1].lower()
    if letter not in ("m", "d") or not 1 <= num <= 12:
        raise ValueError(f"Invalid Open Key code: {open_key!r}")
    camelot_num = ((num + 6) % 12) + 1
    return f"{camelot_num}{'A' if letter == 'm' else 'B'}"


# --- Optional S-KEY (deep-learning) key backend ------------------------------
# S-KEY (Kong et al., ICASSP 2025 — https://github.com/deezer/skey) is a trained
# ChromaNet model that far outperforms the Krumhansl chroma template above on
# electronic music. It is an OPTIONAL dependency (needs `skey` + torch, which are
# NOT in requirements.txt): when the package is absent, analyze_mood() silently
# falls back to the chroma method, so nothing here is load-bearing.

# S-KEY spells pitch classes as sharps (plus "Bb"); map them to this module's
# NOTE_NAMES entries so convert_to_camelot() and the stored key string stay consistent.
_SKEY_NOTE_TO_REPO = {
    "C": "C", "C#": "C#/Db", "D": "D", "D#": "D#/Eb", "E": "E", "F": "F",
    "F#": "F#/Gb", "G": "G", "G#": "G#/Ab", "A": "A", "Bb": "A#/Bb", "B": "B",
}

# Cache the loaded model components per (checkpoint, device) so a library analysis
# doesn't rebuild the network for every track.
_skey_cache: dict = {}


def skey_label_to_key_camelot(label: str) -> Tuple[str, str]:
    """Convert an S-KEY label (e.g. "C# minor", "Bb Major") to this repo's
    (key_name, camelot_key) conventions — e.g. ("C#/Db minor", "12A").

    Pure string/lookup logic with no torch dependency, so it is unit-testable
    without the optional S-KEY package installed.
    """
    note_tok, mode_tok = label.rsplit(" ", 1)
    mode = mode_tok.lower()  # "Major"/"minor" -> "major"/"minor"
    if mode not in ("major", "minor"):
        raise ValueError(f"Unrecognized S-KEY mode in label {label!r}")
    repo_note = _SKEY_NOTE_TO_REPO[note_tok]
    return f"{repo_note} {mode}", convert_to_camelot(repo_note, mode)


def detect_key_skey(file_path: str, device: str = "cpu") -> Tuple[str, str, float]:
    """Detect key with the S-KEY model, returning (key_name, camelot_key, confidence)
    in this repo's conventions. `confidence` is the softmax probability of the
    predicted class (a real 0-1 value, unlike the chroma method's near-constant score).

    Raises on any failure (missing package, load error, inference error) — callers
    are expected to catch and fall back to the chroma method.
    """
    import torch
    from skey.key_detection import (
        DEFAULT_CHECKPOINT_PATH,
        key_map,
        load_audio,
        load_checkpoint,
        load_model_components,
    )

    cache_key = (str(DEFAULT_CHECKPOINT_PATH), device)
    if cache_key not in _skey_cache:
        ckpt = load_checkpoint(DEFAULT_CHECKPOINT_PATH)
        d = torch.device(device)
        hcqt, chromanet, crop_fn = load_model_components(ckpt, d)
        _skey_cache[cache_key] = (hcqt, chromanet, crop_fn, ckpt["audio"]["sr"], d)
    hcqt, chromanet, crop_fn, sr, d = _skey_cache[cache_key]

    batch = load_audio(str(file_path), sr).to(d)
    with torch.no_grad():
        cropped = crop_fn(hcqt(batch.unsqueeze(0).to(d)), torch.zeros(1).to(d))
        # ChromaNet already outputs a probability distribution over the 24 keys
        # (softmax internally; the row sums to 1) — averaged over time crops here.
        # Use it directly; do NOT re-softmax, which would flatten it toward uniform.
        probs = torch.mean(chromanet(cropped), dim=0)
        idx = int(probs.argmax())
        confidence = float(probs[idx])

    key_name, camelot_key = skey_label_to_key_camelot(key_map[idx])
    return key_name, camelot_key, confidence


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


def analyze_mood(
    y: np.ndarray,
    sr: int,
    file_path: str = None,
    prefer_skey: bool = True,
) -> MoodResult:
    """
    Complete mood analysis (key, camelot, brightness, confidence, ZCR, roughness).

    Key detection uses the S-KEY deep-learning model when it is available and a
    `file_path` is given (S-KEY reads and resamples the original file itself);
    otherwise it falls back to the chroma template method on `y`. The returned
    `MoodResult.key_source` records which path was taken, and `key_confidence`
    must be interpreted accordingly (S-KEY: softmax probability; chroma: the
    legacy near-constant correlation score).

    Args:
        y: Audio waveform
        sr: Sample rate
        file_path: Original audio path; required to use the S-KEY backend.
        prefer_skey: Try S-KEY first when available (default True). Set False to
            force the chroma method (e.g. for reproducibility or offline runs).

    Returns:
        MoodResult with key, camelot, brightness, confidence, zero-crossing rate,
        timbral roughness, and key_source.
    """
    key_source = "chroma"
    skey_result = None
    if prefer_skey and file_path is not None:
        try:
            skey_result = detect_key_skey(file_path)
        except Exception as e:
            logging.getLogger(__name__).info(
                f"S-KEY unavailable/failed for {file_path} ({e}); using chroma key detection"
            )

    if skey_result is not None:
        key_name, camelot_key, key_confidence = skey_result
        key_source = "skey"
    else:
        # Chroma template fallback
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_note, key_type, key_confidence = detect_key_from_chroma(chroma)
        camelot_key = convert_to_camelot(key_note, key_type)
        key_name = f"{key_note} {key_type}"

    # Brightness and timbral character are always measured from the waveform
    brightness = compute_brightness(y, sr)
    zero_crossing_rate = compute_zero_crossing_rate(y)
    roughness = compute_roughness(y, sr)

    return MoodResult(
        key=key_name,
        camelot_key=camelot_key,
        brightness=brightness,
        key_confidence=key_confidence,
        zero_crossing_rate=zero_crossing_rate,
        roughness=roughness,
        key_source=key_source,
    )
