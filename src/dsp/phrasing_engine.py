"""Phrasing Engine (Step 1) — Extract structure: segments, boundaries, cue points."""

import numpy as np
import librosa
from typing import List, Optional, Tuple
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


def compute_lowband_energy(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    fmin: float = 20.0,
    fmax: float = 150.0,
) -> np.ndarray:
    """
    Mean magnitude in the kick+bass band per STFT frame.

    For four-on-the-floor techno the low end carries the structure: it is present
    during drops and falls away during breakdowns. This is far more reliable than
    generic spectral novelty for detecting drops/breakdowns.

    Args:
        y: Audio waveform
        sr: Sample rate
        hop_length: Hop length for STFT
        fmin: Low bound of the band (Hz)
        fmax: High bound of the band (Hz)

    Returns:
        energy: 1D array of mean low-band magnitude per frame
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(band) == 0:
        return np.zeros(S.shape[1])
    return S[band, :].mean(axis=0)


def detect_energy_sections(
    energy: np.ndarray,
    sr: int,
    bpm: float,
    hop_length: int = 512,
    min_bars: int = 4,
    thresh_frac: float = 0.4,
) -> List[Tuple[bool, float, float]]:
    """
    Split the track into DROP (kick on) / BREAKDOWN (kick off) sections.

    The low-band energy is smoothed over ~1 bar and thresholded at a fraction of
    its peak. Sections shorter than ``min_bars`` are merged into the previous one
    so short blips do not fragment the structure. Sections stay contiguous and
    cover the whole track.

    Args:
        energy: Low-band energy envelope (from compute_lowband_energy)
        sr: Sample rate
        bpm: Track BPM (used to derive bar length and the smoothing window)
        hop_length: Hop length used for the energy envelope
        min_bars: Minimum section length in bars
        thresh_frac: Kick-on threshold as a fraction of peak low-band energy

    Returns:
        List of (is_drop, start_time, end_time) tuples, time-ordered & contiguous
    """
    n = len(energy)
    if n == 0:
        return []

    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_length)
    end_time = float(times[-1]) if n > 1 else 0.0
    spb = (60.0 / bpm) * 4.0 if bpm > 0 else 2.0

    # Smooth over ~1 bar
    win = max(1, int(spb / (hop_length / sr)))
    energy_s = np.convolve(energy, np.ones(win) / win, mode="same")

    peak = float(energy_s.max()) if energy_s.size else 0.0
    if peak <= 0:
        return [(True, 0.0, end_time)]  # degenerate: treat as one drop

    on = energy_s > (thresh_frac * peak)

    # Build raw contiguous sections
    raw: List[List] = []
    state = bool(on[0])
    start_i = 0
    for i in range(1, n):
        if bool(on[i]) != state:
            raw.append([state, float(times[start_i]), float(times[i])])
            state = bool(on[i])
            start_i = i
    raw.append([state, float(times[start_i]), end_time])

    # Merge sections shorter than min_bars into the previous section (keep contiguity)
    min_len = min_bars * spb
    merged: List[List] = []
    for sec in raw:
        if merged and (sec[2] - sec[1]) < min_len:
            merged[-1][2] = sec[2]  # extend previous section's end
        else:
            merged.append(sec)

    return [(bool(s[0]), s[1], s[2]) for s in merged]


def label_energy_sections(sections: List[Tuple[bool, float, float]]) -> List[Segment]:
    """
    Label DROP/BREAKDOWN sections as intro/drop/breakdown/outro.

    - kick-on section          -> "drop"
    - kick-off first section    -> "intro"
    - kick-off last section     -> "outro"
    - kick-off in the middle    -> "breakdown"

    Args:
        sections: (is_drop, start, end) tuples from detect_energy_sections

    Returns:
        List of labeled Segment objects
    """
    segments = []
    n = len(sections)
    for i, (is_drop, start, end) in enumerate(sections):
        if is_drop:
            label = "drop"
        elif i == 0:
            label = "intro"
        elif i == n - 1:
            label = "outro"
        else:
            label = "breakdown"
        segments.append(Segment(start_time=start, end_time=end, label=label, confidence=0.85))
    return segments


def map_segments_to_hotcues(
    segments: List[Segment],
    max_pads: Optional[int] = None,
) -> List[CuePoint]:
    """
    Map structural segments to hot-cue positions at their START (the cue point a
    DJ actually wants to jump to).

    Two modes:

    * ``max_pads is None`` (default) — one cue per structural section, labelled by
      type: intro -> Pad 1, first (main) drop -> Pad 4, later drops -> Pad 3,
      breakdown -> Pad 2, outro -> Pad 1. Useful when the target (e.g. Traktor)
      has unlimited cues.

    * ``max_pads = N`` — keep only the N most important sections and re-label them
      ``Pad 1..N`` in chronological order, for controllers with a fixed number of
      performance pads (e.g. 4). Importance: the intro, the first (main) drop and
      the outro (mix-out point) are always kept; remaining slots go to the longest
      sections (drops and breakdowns). Because pads are numbered by time, the
      outro naturally takes the last pad. The original section type is preserved
      on ``CuePoint.type`` for colour-coding.

    Args:
        segments: List of Segment objects
        max_pads: Max number of hot cues / physical pads (None = unlimited)

    Returns:
        cues: List of CuePoint objects at segment start times, time-ordered
    """
    # --- default unlimited mode: label by type ---
    if max_pads is None:
        cues = []
        drop_seen = False
        for seg in segments:
            if seg.label == "intro":
                cues.append(CuePoint(time=seg.start_time, label="Pad 1", type="intro"))
            elif seg.label == "breakdown":
                cues.append(CuePoint(time=seg.start_time, label="Pad 2", type="breakdown"))
            elif seg.label == "drop":
                if not drop_seen:
                    cues.append(CuePoint(time=seg.start_time, label="Pad 4", type="drop"))
                    drop_seen = True
                else:
                    cues.append(CuePoint(time=seg.start_time, label="Pad 3", type="drop"))
            elif seg.label == "outro":
                cues.append(CuePoint(time=seg.start_time, label="Pad 1", type="outro"))
        return cues

    # --- limited mode: pick the N most important sections ---
    if max_pads < 1:
        return []

    # Guaranteed sections rank above any real section length (seconds):
    #   intro (mix-in) > main drop > outro (mix-out) > longest other sections
    candidates = []  # (start_time, type, importance)
    drop_seen = False
    for seg in segments:
        duration = seg.end_time - seg.start_time
        if seg.label == "intro":
            importance = 1e12  # always keep the mix-in point
        elif seg.label == "outro":
            importance = 1e10  # always keep the mix-out point
        elif seg.label == "drop":
            importance = duration
            if not drop_seen:
                importance += 1e11  # main drop is guaranteed a pad
                drop_seen = True
        else:  # breakdown (and any other) ranked by length
            importance = duration
        candidates.append((seg.start_time, seg.label, importance))

    # Keep the most important, then restore chronological order
    keep = sorted(candidates, key=lambda c: c[2], reverse=True)[:max_pads]
    keep = sorted(keep, key=lambda c: c[0])

    return [
        CuePoint(time=t, label=f"Pad {i}", type=typ)
        for i, (t, typ, _) in enumerate(keep, start=1)
    ]


def analyze_structure(
    y: np.ndarray,
    sr: int,
    bpm: float,
    hop_length: int = 512,
    min_bars: int = 4,
    thresh_frac: float = 0.4,
    max_pads: Optional[int] = None,
) -> PhrasingResult:
    """
    Complete phrasing analysis driven by kick+bass (low-band) energy.

    Drops (kick present) and breakdowns (kick absent) are detected from the
    20-150 Hz energy envelope — the reliable structural signal for techno —
    rather than generic spectral novelty, which over-segments four-on-the-floor
    material. Boundaries land on real drop/breakdown transitions and hot cues are
    placed at each section start.

    Args:
        y: Audio waveform
        sr: Sample rate
        bpm: Track BPM (from groove engine)
        hop_length: Hop length for STFT
        min_bars: Ignore/merge sections shorter than this many bars
        thresh_frac: Kick-on threshold as a fraction of peak low-band energy
        max_pads: Limit hot cues to this many pads (None = one per section)

    Returns:
        PhrasingResult with segments, boundaries, and cue points
    """
    # Low-band energy -> DROP/BREAKDOWN sections
    energy = compute_lowband_energy(y, sr, hop_length)
    sections = detect_energy_sections(
        energy, sr, bpm, hop_length=hop_length,
        min_bars=min_bars, thresh_frac=thresh_frac,
    )

    # Label sections and place cue points at their starts
    segments = label_energy_sections(sections)
    cue_points = map_segments_to_hotcues(segments, max_pads=max_pads)

    # Boundaries = the interior section starts (skip the 0.0 start of the track)
    boundaries = [seg.start_time for seg in segments if seg.start_time > 0.0]

    # Structure confidence (average segment confidence)
    avg_confidence = float(np.mean([s.confidence for s in segments])) if segments else 0.0

    return PhrasingResult(
        segment_boundaries=boundaries,
        segments=segments,
        cue_points=cue_points,
        structure_confidence=avg_confidence,
    )
