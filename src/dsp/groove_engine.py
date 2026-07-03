"""Groove Engine (Step 2) — Extract rhythm: BPM, beat grid, swing, tempo stability."""

import numpy as np
import librosa
from typing import Tuple
from ..features.schema import GrooveResult


def compute_swing_score(beat_frames: np.ndarray, onset_frames: np.ndarray) -> float:
    """
    Compute swing score (0=straight/machine grid, 1=fully swung/triplet shuffle).

    Swing is measured from WHERE offbeat onsets land within the beat, not from
    how many onsets are off the grid: a straight groove places its offbeat
    events (e.g. hats) exactly halfway between beats (beat phase 0.5), while a
    swung/shuffled groove pushes them late toward the triplet position
    (phase ~0.67). Onsets on the grid itself (kicks) carry no swing
    information and are ignored — otherwise any track with offbeat hats
    saturates the measure.

    Args:
        beat_frames: Detected beat frame positions
        onset_frames: Detected onset frame positions

    Returns:
        swing_score: 0.0 (straight) to 1.0 (full triplet swing)
    """
    if len(beat_frames) < 2 or len(onset_frames) == 0:
        return 0.5  # Default neutral swing

    # Beat phase (0-1) of each onset that falls between two detected beats
    offbeat_phases = []
    for onset in onset_frames:
        idx = np.searchsorted(beat_frames, onset)
        if idx == 0 or idx >= len(beat_frames):
            continue
        prev_beat, next_beat = beat_frames[idx - 1], beat_frames[idx]
        interval = next_beat - prev_beat
        if interval <= 0:
            continue
        phase = (onset - prev_beat) / interval
        # Keep only clear offbeat events; near-grid onsets say nothing about swing
        if 0.25 <= phase <= 0.85:
            offbeat_phases.append(phase)

    if len(offbeat_phases) < 4:
        return 0.0  # no offbeat events to swing — rigid on-grid groove

    # 0.5 = perfectly straight 8ths; 2/3 = full triplet swing
    median_phase = float(np.median(offbeat_phases))
    swing_score = abs(median_phase - 0.5) / (2.0 / 3.0 - 0.5)

    return float(np.clip(swing_score, 0.0, 1.0))


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
