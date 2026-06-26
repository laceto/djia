"""Groove Engine (Step 2) — Extract rhythm: BPM, beat grid, swing, tempo stability."""

import numpy as np
import librosa
from typing import Tuple, List
from ..features.schema import GrooveResult


def compute_swing_score(beat_frames: np.ndarray, onset_frames: np.ndarray) -> float:
    """
    Compute swing score (0=stiff industrial, 1=groovy bouncy).

    Swing is measured by deviation from perfect grid. Onsets that fall off the
    beat grid create a "shuffled" or swung feel.

    Args:
        beat_frames: Detected beat frame positions
        onset_frames: Detected onset frame positions

    Returns:
        swing_score: 0.0 (stiff) to 1.0 (groovy)
    """
    if len(beat_frames) < 2 or len(onset_frames) == 0:
        return 0.5  # Default neutral swing

    # Compute beat grid spacing
    beat_intervals = np.diff(beat_frames)
    mean_beat_interval = np.mean(beat_intervals)

    # For each onset, find distance to nearest beat
    onset_deviations = []
    for onset in onset_frames:
        # Distance to nearest beat
        distances = np.abs(onset - beat_frames)
        min_dist = np.min(distances)
        onset_deviations.append(min_dist)

    # Calculate off-grid percentage
    # Onsets significantly off the beat indicate swing
    threshold = mean_beat_interval * 0.1  # 10% of beat interval
    off_grid_ratio = np.mean([d > threshold for d in onset_deviations])

    # Scale swing from 0-1:
    # - High precision (low off_grid_ratio) = low swing = 0
    # - More variation = higher swing = 1
    swing_score = min(1.0, off_grid_ratio * 2.0)

    return float(swing_score)


def check_tempo_stability(beat_frames: np.ndarray, sr: int, hop_length: int = 512) -> Tuple[bool, float]:
    """
    Check if tempo drifts significantly over track.

    Args:
        beat_frames: Detected beat frame positions
        sr: Sample rate
        hop_length: Hop length used in beat tracking

    Returns:
        (is_stable, variance): Stability flag and tempo variance
    """
    if len(beat_frames) < 4:
        return True, 0.0

    # Compute beat intervals
    beat_intervals = np.diff(beat_frames)

    # Convert to tempo changes
    tempos = (sr / hop_length) * 60 / beat_intervals

    # Calculate variance
    tempo_variance = np.var(tempos) / (np.mean(tempos) ** 2 + 1e-8)  # Normalized

    # Stability threshold: if variance is low, tempo is stable
    # Threshold: 0.02 (2% normalized variance)
    is_stable = tempo_variance < 0.02

    return bool(is_stable), float(tempo_variance)


def analyze_groove(y: np.ndarray, sr: int, hop_length: int = 512) -> GrooveResult:
    """
    Complete groove analysis (BPM, beat grid, swing, tempo stability).

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length for feature extraction

    Returns:
        GrooveResult with BPM, beat grid, swing, and stability
    """
    # Extract onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Beat tracking
    bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
    )

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length
    )

    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    # Calculate swing score
    swing_score = compute_swing_score(beat_frames, onset_frames)

    # Check tempo stability
    is_stable, tempo_variance = check_tempo_stability(beat_frames, sr, hop_length)

    # Ensure bpm is scalar
    if isinstance(bpm, np.ndarray):
        bpm = float(bpm[0] if len(bpm) > 0 else bpm.item())
    else:
        bpm = float(bpm)

    return GrooveResult(
        bpm=bpm,
        beat_grid=beat_frames,
        beat_times=beat_times,
        swing_score=swing_score,
        tempo_stability=is_stable,
        stability_variance=tempo_variance
    )
