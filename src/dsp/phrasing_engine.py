"""Phrasing Engine (Step 1) — Extract structure: segments, boundaries, cue points."""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
from scipy import signal
from ..features.schema import Segment, CuePoint, PhrasingResult, ElementOnset


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


def _band_name(freq_low: float, freq_high: float) -> str:
    """Human hint for a band, in DJ-EQ language, from its geometric-center freq."""
    center = np.sqrt(max(freq_low, 1.0) * max(freq_high, 1.0))
    if center < 120:
        return "sub"
    if center < 300:
        return "low"
    if center < 800:
        return "low-mid"
    if center < 2500:
        return "mid"
    if center < 6000:
        return "high-mid"
    return "high"


def _band_log_energy(
    y: np.ndarray, sr: int, n_bands: int = 8, hop_length: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the spectrogram into log-spaced frequency bands and return their
    log-energy over time.

    Returns:
        log_energy: (n_bands, n_frames) array, np.log1p of summed magnitude per band
        edge_hz: (n_bands + 1,) array of band edge frequencies in Hz
    """
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    n_freq = S.shape[0]
    n_fft = 2 * (n_freq - 1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Log-spaced band edges from ~20 Hz to Nyquist, mapped to STFT bins
    edge_hz = np.logspace(np.log10(20.0), np.log10(sr / 2), n_bands + 1)
    edge_bin = np.clip(np.searchsorted(freqs, edge_hz), 0, n_freq)

    band_energy = np.zeros((n_bands, S.shape[1]))
    for b in range(n_bands):
        lo = edge_bin[b]
        hi = max(lo + 1, edge_bin[b + 1])  # ensure ≥1 bin per band
        band_energy[b] = S[lo:hi].sum(axis=0)

    return np.log1p(band_energy), edge_hz


def _rectified_band_novelty(log_energy: np.ndarray) -> np.ndarray:
    """
    Half-wave-rectified temporal flux per band, each band min-max normalized to 0-1.

    Keeping only energy *increases* means an element being added lights up its
    band, while an element dropping out (a negative flux) is clipped to zero.
    """
    flux = np.diff(log_energy, axis=1)
    flux = np.maximum(flux, 0.0)  # additive events only
    flux = np.concatenate([np.zeros((log_energy.shape[0], 1)), flux], axis=1)  # keep frame alignment

    lo = flux.min(axis=1, keepdims=True)
    hi = flux.max(axis=1, keepdims=True)
    return (flux - lo) / (hi - lo + 1e-8)


def compute_additive_novelty(
    y: np.ndarray, sr: int, n_bands: int = 8, hop_length: int = 512
) -> np.ndarray:
    """
    Per-band "a new element entered" curve.

    Unlike `compute_novelty_curve` (symmetric spectral flux over the whole
    spectrum, which fires on *any* change and buries a quiet element under a
    loud one), this splits the spectrum into log-spaced bands and keeps only
    per-band energy increases. A new beep/hat/synth line lights up its own
    band; unchanged bands stay flat; elements dropping out are ignored.

    Args:
        y: Audio waveform
        sr: Sample rate
        n_bands: Number of log-spaced frequency bands
        hop_length: Hop length for STFT

    Returns:
        novelty: (n_bands, n_frames) array in 0-1, one additive-novelty curve
                 per band (lowest band first).
    """
    log_energy, _ = _band_log_energy(y, sr, n_bands=n_bands, hop_length=hop_length)
    return _rectified_band_novelty(log_energy)


def detect_element_onsets(
    y: np.ndarray,
    sr: int,
    bpm: float,
    n_bands: int = 8,
    hop_length: int = 512,
    threshold: float = 0.4,
    min_sustain_bars: float = 2.0,
    snap_to_bar: bool = True,
) -> List[ElementOnset]:
    """
    Detect the moments new sound elements enter, per frequency band.

    Pipeline: additive-novelty per band → peak-pick above `threshold` →
    keep only peaks whose band energy stays elevated for `min_sustain_bars`
    (rejects one-shot FX) → snap each onset to the nearest bar.

    Args:
        y: Audio waveform
        sr: Sample rate
        bpm: Track BPM (needed for bar math / sustain window)
        n_bands: Number of frequency bands to watch
        hop_length: Hop length for STFT
        threshold: Peak height (0-1) on the additive-novelty curve
        min_sustain_bars: Bars the new element must persist to count
        snap_to_bar: Snap onset times to the nearest bar

    Returns:
        List[ElementOnset] sorted by (time, band).
    """
    log_energy, edge_hz = _band_log_energy(y, sr, n_bands=n_bands, hop_length=hop_length)
    novelty = _rectified_band_novelty(log_energy)
    n_frames = novelty.shape[1]

    sustain_frames = max(1, int(round(bar_to_time(min_sustain_bars, bpm) * sr / hop_length)))

    # Gate out near-dead bands so their normalized noise can't fire false onsets
    band_mean = log_energy.mean(axis=1)
    energy_gate = band_mean.max() * 0.15 if band_mean.max() > 0 else 0.0

    onsets: List[ElementOnset] = []
    for b in range(n_bands):
        if band_mean[b] < energy_gate:
            continue

        peaks, props = signal.find_peaks(
            novelty[b], height=threshold, distance=sustain_frames
        )
        for pk, height in zip(peaks, props["peak_heights"]):
            pre = log_energy[b, max(0, pk - sustain_frames):pk]
            post = log_energy[b, pk:min(n_frames, pk + sustain_frames)]
            if pre.size == 0 or post.size == 0:
                continue
            # Sustained (not a one-shot): post-onset energy holds above the
            # pre-onset baseline for most of the window (20th percentile guard).
            if np.percentile(post, 20) <= pre.mean():
                continue

            t = librosa.frames_to_time(pk, sr=sr, hop_length=hop_length)
            if snap_to_bar:
                t = bar_to_time(round(time_to_bar(t, bpm)), bpm)

            onsets.append(ElementOnset(
                time=float(t),
                band=int(b),
                freq_low=float(edge_hz[b]),
                freq_high=float(edge_hz[b + 1]),
                confidence=float(round(height, 3)),
                label=_band_name(edge_hz[b], edge_hz[b + 1]),
            ))

    onsets.sort(key=lambda o: (o.time, o.band))
    return onsets


def derive_mix_points(
    onsets: List[ElementOnset],
    bpm: float,
    duration: float,
    mix_out_bars: int = 32,
) -> Dict[str, Optional[float]]:
    """
    Turn a track's element onsets into named, deck-actionable mix points.

    Args:
        onsets: Output of `detect_element_onsets` (may be empty)
        bpm: Track BPM
        duration: Track duration in seconds
        mix_out_bars: Bars before the end to place the mix-out point

    Returns:
        Dict with (each None when underivable):
        - "mix_in": first element entry — where to start the incoming deck
        - "bass_in": first sub/low-band entry — where to swap the lows
        - "full_on": all detected bands have entered — blend should be done
        - "mix_out": `mix_out_bars` before the end, bar-snapped — start leaving
    """
    points: Dict[str, Optional[float]] = {
        "mix_in": None, "bass_in": None, "full_on": None, "mix_out": None,
    }

    if onsets:
        points["mix_in"] = onsets[0].time
        points["bass_in"] = next(
            (o.time for o in onsets if o.label in ("sub", "low")), None
        )
        all_bands = {o.band for o in onsets}
        entered = set()
        for o in onsets:
            entered.add(o.band)
            if entered == all_bands:
                points["full_on"] = o.time
                break

    if bpm > 0 and duration > 0:
        out_bar = int(time_to_bar(duration, bpm)) - mix_out_bars
        if out_bar > 0:
            points["mix_out"] = bar_to_time(out_bar, bpm)

    return points


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
    include_beats: bool = True,
    detect_elements: bool = False,
    element_n_bands: int = 8,
    element_onset_threshold: float = 0.4,
    element_min_sustain_bars: float = 2.0,
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

    # Optionally detect new-element entries (beep/hat/synth coming in)
    element_onsets = []
    if detect_elements:
        element_onsets = detect_element_onsets(
            y, sr, bpm,
            n_bands=element_n_bands,
            hop_length=hop_length,
            threshold=element_onset_threshold,
            min_sustain_bars=element_min_sustain_bars,
        )

    # Calculate structure confidence (average segment confidence)
    avg_confidence = np.mean([s.confidence for s in segments]) if segments else 0.0

    return PhrasingResult(
        segment_boundaries=boundaries,
        segments=segments,
        cue_points=cue_points,
        structure_confidence=avg_confidence,
        element_onsets=element_onsets,
    )
