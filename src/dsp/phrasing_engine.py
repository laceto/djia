"""Phrasing Engine (Step 1) — Extract structure: segments, boundaries, cue points."""

import numpy as np
import librosa
from typing import List, Tuple
from scipy import signal
from ..features.schema import Segment, CuePoint, PhrasingResult


def compute_novelty_curve(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute spectral novelty curve (measure of change in spectral content).

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length for STFT

    Returns:
        novelty: Array of novelty values (0-1 range)
    """
    # Compute STFT
    S = librosa.stft(y, hop_length=hop_length)
    magnitude = np.abs(S)

    # Normalize each frame
    magnitude_norm = magnitude / (np.sum(magnitude, axis=0, keepdims=True) + 1e-8)

    # Compute spectral flux (L2 norm of diff)
    flux = np.sqrt(np.sum(np.diff(magnitude_norm, axis=1)**2, axis=0))

    # Normalize to 0-1
    novelty = (flux - np.min(flux)) / (np.max(flux) - np.min(flux) + 1e-8)

    return novelty


def detect_segment_boundaries(
    novelty: np.ndarray,
    sr: int,
    hop_length: int = 512,
    threshold: float = 0.3,
    min_segment_duration: float = 4.0
) -> List[float]:
    """
    Detect section boundaries from novelty curve peaks.

    Args:
        novelty: Novelty curve array
        sr: Sample rate
        hop_length: Hop length used in novelty calculation
        threshold: Peak detection threshold (0-1)
        min_segment_duration: Minimum duration between segments (seconds)

    Returns:
        segment_boundaries: List of boundary times in seconds
    """
    # Find peaks in novelty curve
    peaks, _ = signal.find_peaks(novelty, height=threshold, distance=int(sr / hop_length * min_segment_duration))

    # Convert frame indices to time
    segment_boundaries = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length).tolist()

    return segment_boundaries


def label_segment(start_time: float, end_time: float, position: int, total: int, bpm: float) -> str:
    """
    Auto-label a segment based on position and duration.

    Args:
        start_time: Segment start time
        end_time: Segment end time
        position: Which segment this is (0-indexed)
        total: Total number of segments
        bpm: Track BPM for beat calculation

    Returns:
        label: String label for segment
    """
    duration = end_time - start_time

    # If it's the first segment and relatively short, it's probably an intro
    if position == 0 and duration < 32:
        return "intro"

    # If it's the last segment and short, it's probably an outro
    if position == total - 1 and duration < 32:
        return "outro"

    # Mid-track shorter segments are often breakdowns
    if duration < 16:
        return "breakdown"

    # Longer segments are builds or main sections
    if position < total // 2:
        return "build"
    else:
        return "drop"


def create_segments(
    boundaries: List[float],
    duration: float,
    bpm: float
) -> List[Segment]:
    """
    Create labeled segments from boundaries.

    Args:
        boundaries: Segment boundary times
        duration: Total track duration
        bpm: Track BPM

    Returns:
        segments: List of Segment objects
    """
    segments = []

    # Add boundaries at start and end if not present
    all_times = [0.0] + boundaries + [duration]
    all_times = sorted(set(all_times))  # Remove duplicates and sort

    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        label = label_segment(start, end, i, len(all_times) - 2, bpm)

        segments.append(Segment(
            start_time=start,
            end_time=end,
            label=label,
            confidence=0.8
        ))

    return segments


def map_segments_to_hotcues(segments: List[Segment]) -> List[CuePoint]:
    """
    Map structural segments to hot-cue positions (Pad 1, 2, 4).

    Target: Pad 1 at intro/first drop, Pad 2 at breakdown, Pad 4 at main drop.

    Args:
        segments: List of Segment objects

    Returns:
        cues: List of CuePoint objects
    """
    cues = []

    for seg in segments:
        cue_time = (seg.start_time + seg.end_time) / 2  # Midpoint of segment

        if seg.label == "intro":
            cues.append(CuePoint(time=cue_time, label="Pad 1", type="intro"))
        elif seg.label == "breakdown":
            cues.append(CuePoint(time=cue_time, label="Pad 2", type="breakdown"))
        elif seg.label == "drop" and len([c for c in cues if "Pad 4" in c.label]) == 0:
            # First major drop gets Pad 4
            cues.append(CuePoint(time=cue_time, label="Pad 4", type="drop"))

    return cues


def analyze_structure(
    y: np.ndarray,
    sr: int,
    bpm: float,
    hop_length: int = 512
) -> PhrasingResult:
    """
    Complete phrasing analysis.

    Args:
        y: Audio waveform
        sr: Sample rate
        bpm: Track BPM (from groove engine)
        hop_length: Hop length for STFT

    Returns:
        PhrasingResult with segments, boundaries, and cue points
    """
    # Compute novelty curve
    novelty = compute_novelty_curve(y, sr, hop_length)

    # Detect boundaries
    boundaries = detect_segment_boundaries(novelty, sr, hop_length)

    # Get track duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Create segments
    segments = create_segments(boundaries, duration, bpm)

    # Map to cue points
    cue_points = map_segments_to_hotcues(segments)

    # Calculate structure confidence (average segment confidence)
    avg_confidence = np.mean([s.confidence for s in segments]) if segments else 0.0

    return PhrasingResult(
        segment_boundaries=boundaries,
        segments=segments,
        cue_points=cue_points,
        structure_confidence=avg_confidence
    )
