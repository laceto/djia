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
    threshold: float = 0.5,
    min_segment_duration: float = 8.0
) -> List[float]:
    """
    Detect section boundaries from novelty curve peaks.

    Args:
        novelty: Novelty curve array
        sr: Sample rate
        hop_length: Hop length used in novelty calculation
        threshold: Peak detection threshold (0-1). Higher = fewer detections.
                   0.3=aggressive, 0.5=balanced, 0.7=conservative
        min_segment_duration: Minimum duration between segments (seconds).
                              Higher = longer segments, fewer detections.
                              4.0=aggressive, 8.0=balanced, 16.0=conservative

    Returns:
        segment_boundaries: List of boundary times in seconds
    """
    # Find peaks in novelty curve
    peaks, _ = signal.find_peaks(novelty, height=threshold, distance=int(sr / hop_length * min_segment_duration))

    # Convert frame indices to time
    segment_boundaries = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length).tolist()

    return segment_boundaries


def time_to_beat(time_seconds: float, bpm: float) -> int:
    """
    Convert time (seconds) to beat number (0-indexed).

    Args:
        time_seconds: Time in seconds
        bpm: Beats per minute

    Returns:
        beat_number: Beat index (integer)
    """
    return int(round(time_seconds * bpm / 60))


def beat_to_bar_group(beat: int, beats_per_group: int = 4) -> int:
    """
    Convert beat number to start of its group (aligned to groups of N beats).

    Args:
        beat: Beat number (0-indexed)
        beats_per_group: Group size (default 4)

    Returns:
        group_start: Start beat of this group
    """
    return (beat // beats_per_group) * beats_per_group


def time_to_bar(time_seconds: float, bpm: float) -> float:
    """
    Convert time (seconds) to bar number (floating point).

    Args:
        time_seconds: Time in seconds
        bpm: Beats per minute

    Returns:
        bar_number: Bar position (float, where 1 bar = 4 beats)
    """
    beats = time_seconds * bpm / 60
    return beats / 4


def bar_to_time(bar_number: float, bpm: float) -> float:
    """
    Convert bar number to time (seconds).

    Args:
        bar_number: Bar position (float)
        bpm: Beats per minute

    Returns:
        time_seconds: Time in seconds
    """
    beats = bar_number * 4
    return beats * 60 / bpm


def snap_to_bar_boundary(time_seconds: float, bpm: float, bars_per_phrase: int = 16) -> float:
    """
    Snap a time position to the nearest bar or phrase boundary.

    Args:
        time_seconds: Time to snap (seconds)
        bpm: Beats per minute
        bars_per_phrase: Size of phrase (e.g., 16 bars = 1 phrase)

    Returns:
        snapped_time: Time snapped to nearest phrase boundary
    """
    bar = time_to_bar(time_seconds, bpm)
    snapped_bar = round(bar / bars_per_phrase) * bars_per_phrase
    return bar_to_time(snapped_bar, bpm)


def label_segment(
    start_time: float,
    end_time: float,
    position: int,
    total: int,
    bpm: float,
    breakdown_threshold: float = 24.0,
    include_beats: bool = True
) -> str:
    """
    Auto-label a segment based on position and duration, with beat range.

    Args:
        start_time: Segment start time (seconds)
        end_time: Segment end time (seconds)
        position: Which segment this is (0-indexed)
        total: Total number of segments
        bpm: Track BPM for beat calculation
        breakdown_threshold: Duration threshold for "breakdown" label (seconds).
                            Segments shorter than this are labeled breakdown.
                            16.0=aggressive, 24.0=balanced, 32.0=conservative
        include_beats: Add beat range to label (e.g., "drop (beats 32-64)")

    Returns:
        label: String label for segment (e.g., "intro", "drop (beats 32-64)")
    """
    duration = end_time - start_time

    # Determine segment type
    if position == 0 and duration < 32:
        seg_type = "intro"
    elif position == total - 1 and duration < 32:
        seg_type = "outro"
    elif duration < breakdown_threshold:
        seg_type = "breakdown"
    elif position < total // 2:
        seg_type = "build"
    else:
        seg_type = "drop"

    # Add beat range if requested
    if include_beats:
        start_beat = beat_to_bar_group(time_to_beat(start_time, bpm))
        end_beat = beat_to_bar_group(time_to_beat(end_time, bpm))
        return f"{seg_type} (beats {start_beat}-{end_beat})"
    else:
        return seg_type


def create_phrase_locked_segments(
    duration: float,
    bpm: float,
    bars_per_phrase: int = 16,
    breakdown_threshold: float = 24.0,
    include_beats: bool = True
) -> List[Segment]:
    """
    Create segments locked to phrase boundaries (fixed bar count).

    Instead of detecting arbitrary boundaries, this creates segments
    that are exactly N bars long (standard DJ structure).

    Args:
        duration: Total track duration (seconds)
        bpm: Track BPM
        bars_per_phrase: Number of bars per phrase (default 16).
                        Common values: 8, 16, 32
        breakdown_threshold: Duration threshold for "breakdown" label (seconds)
        include_beats: Add beat range to labels

    Returns:
        segments: List of Segment objects, each N bars long

    Example:
        For a 6-minute track at 126 BPM with 16 bars/phrase:
        - Total bars: ~380 bars
        - Phrases: ~24 phrases of exactly 16 bars each
        - Each phrase: ~7.6 seconds
    """
    segments = []
    phrase_duration_sec = bar_to_time(bars_per_phrase, bpm)

    # Generate segments at phrase boundaries
    current_time = 0.0
    phrase_index = 0

    while current_time < duration:
        segment_start = current_time
        segment_end = min(current_time + phrase_duration_sec, duration)

        # Label based on position (intro/build/drop/outro)
        total_phrases = int(np.ceil(duration / phrase_duration_sec))

        if phrase_index == 0:
            seg_type = "intro"
        elif phrase_index == total_phrases - 1:
            seg_type = "outro"
        elif phrase_index < total_phrases // 2:
            seg_type = "build"
        else:
            seg_type = "drop"

        # Add beat range if requested
        if include_beats:
            start_bar = int(time_to_bar(segment_start, bpm))
            end_bar = int(time_to_bar(segment_end, bpm))
            start_beat = start_bar * 4
            end_beat = end_bar * 4
            label = f"{seg_type} (bars {start_bar}-{end_bar} | beats {start_beat}-{end_beat})"
        else:
            label = seg_type

        segments.append(Segment(
            start_time=segment_start,
            end_time=segment_end,
            label=label,
            confidence=0.95  # High confidence for locked phrases
        ))

        current_time += phrase_duration_sec
        phrase_index += 1

    return segments


def create_segments(
    boundaries: List[float],
    duration: float,
    bpm: float,
    breakdown_threshold: float = 24.0,
    include_beats: bool = True
) -> List[Segment]:
    """
    Create labeled segments from boundaries with beat ranges.

    Args:
        boundaries: Segment boundary times
        duration: Total track duration
        bpm: Track BPM
        breakdown_threshold: Duration threshold for "breakdown" label (seconds)
        include_beats: Add beat range to segment labels (default True)

    Returns:
        segments: List of Segment objects with labels like "drop (beats 32-64)"
    """
    segments = []

    # Add boundaries at start and end if not present
    all_times = [0.0] + boundaries + [duration]
    all_times = sorted(set(all_times))  # Remove duplicates and sort

    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        label = label_segment(
            start, end, i, len(all_times) - 2, bpm,
            breakdown_threshold,
            include_beats=include_beats
        )

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
    hop_length: int = 512,
    novelty_threshold: float = 0.5,
    min_segment_duration: float = 8.0,
    breakdown_threshold: float = 24.0,
    include_beats: bool = True
) -> PhrasingResult:
    """
    Complete phrasing analysis with beat-based labels.

    Args:
        y: Audio waveform
        sr: Sample rate
        bpm: Track BPM (from groove engine)
        hop_length: Hop length for STFT
        novelty_threshold: Peak detection threshold (0-1).
                          0.3=aggressive, 0.5=balanced, 0.7=conservative
        min_segment_duration: Minimum duration between segments (seconds).
                             4.0=aggressive, 8.0=balanced, 16.0=conservative
        breakdown_threshold: Duration threshold for "breakdown" label (seconds).
                            16.0=aggressive, 24.0=balanced, 32.0=conservative
        include_beats: Add beat range to segment labels (default True).
                      Labels will be like "drop (beats 32-64)"

    Returns:
        PhrasingResult with segments, boundaries, and cue points
    """
    # Compute novelty curve
    novelty = compute_novelty_curve(y, sr, hop_length)

    # Detect boundaries
    boundaries = detect_segment_boundaries(
        novelty, sr, hop_length,
        threshold=novelty_threshold,
        min_segment_duration=min_segment_duration
    )

    # Get track duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Create segments with beat ranges
    segments = create_segments(
        boundaries, duration, bpm, breakdown_threshold,
        include_beats=include_beats
    )

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
