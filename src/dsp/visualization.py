"""Visualization — the 8 diagnostic plots derived from the DSP+AI pipeline output.

Single source of truth for plotting: waveform, beat grid, novelty curve, chromagram,
spectrogram, energy curve, mood radar, and low-band structure bars. Consumes the same
`Track` dataclass (`src.features.schema.Track`) and `classify_mood()` output already
used everywhere else in the package — no separate re-implementation of the DSP math.

The standalone root scripts `demo_capabilities.py` and `detect_structure.py` predate
this module and keep their own inline plotting code (they're documented in CLAUDE.md
as intentionally outside `src/`); new callers — the CLI `plot` command, notebooks,
tests — should use this module instead.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")  # headless-safe default; callers that want windows can switch
                        # backends before importing this module and pass show=True.
import matplotlib.pyplot as plt

from ..features.schema import Track
from .phrasing_engine import (
    compute_novelty_curve,
    compute_lowband_energy,
    smooth_lowband_energy,
    detect_energy_sections,
)
from .spectrogram import compute_spectrogram

logger = logging.getLogger(__name__)

DEFAULT_PLOTS_DIR = "results/plots"

PLOT_FILENAMES = {
    "waveform": "01_waveform.png",
    "beat_grid": "02_beat_grid.png",
    "novelty": "03_novelty.png",
    "chromagram": "04_chromagram.png",
    "spectrogram": "05_spectrogram.png",
    "energy": "06_energy.png",
    "mood_radar": "07_mood_radar.png",
    "structure_bars": "structure_bars.png",
}


def to_bar(t: float, bpm: float) -> int:
    """1-indexed bar number at time `t` (4/4)."""
    sec_per_bar = (60.0 / bpm) * 4
    return int(round(t / sec_per_bar)) + 1


def transition_beats(
    segments, beat_times, labels: Tuple[str, ...] = ("drop", "breakdown"), t_max: Optional[float] = None
) -> List[Tuple[float, int]]:
    """(time, cumulative beat count) at each segment transition matching `labels`."""
    beat_times = np.asarray(beat_times)
    points = []
    for seg in segments:
        if seg.label not in labels:
            continue
        if t_max is not None and seg.start_time > t_max:
            continue
        points.append((seg.start_time, int(np.searchsorted(beat_times, seg.start_time))))
    return points


def annotate_beat_counts(ax, points, color: str = "black", draw_vlines: bool = True, bpm: Optional[float] = None) -> None:
    """Add a secondary top x-axis whose ticks land on each (time, beat_number) point.

    When `bpm` is given, each tick also shows its bar number (e.g. "#63\\nb16")."""
    if not points:
        return
    times, beat_nums = zip(*points)
    if draw_vlines:
        for t in times:
            ax.axvline(t, color=color, alpha=0.15, linewidth=0.6, linestyle=":")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(times)
    if bpm:
        labels = [f"#{n}\nb{to_bar(t, bpm)}" for t, n in zip(times, beat_nums)]
        ax2.set_xlabel("Beat count / Bar")
    else:
        labels = [f"#{n}" for n in beat_nums]
        ax2.set_xlabel("Beat count")
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)


def _savefig(out_dir: Path, filename: str, show: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    logger.info(f"Saved plot -> {path}")
    if show:
        plt.show()
    plt.close()
    return path


def plot_waveform(y: np.ndarray, sr: int, track: Track, out_dir: Path, show: bool = False) -> Path:
    """Full waveform with drop/breakdown transition markers (beat count + bar)."""
    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(14, 4))
    plt.plot(t, y, linewidth=0.4, color="#1f77b4")
    plt.title("Waveform (22.05 kHz mono)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    pts = transition_beats(track.phrasing.segments, track.groove.beat_times)
    annotate_beat_counts(plt.gca(), pts, color="#d62728", bpm=track.groove.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["waveform"], show)


def plot_beat_grid(y: np.ndarray, sr: int, track: Track, out_dir: Path, show: bool = False) -> Path:
    """Full-duration waveform with every detected beat marked; sparse beat/bar ticks."""
    g = track.groove
    beat_times_arr = np.array(g.beat_times)
    duration_s = len(y) / sr
    plt.figure(figsize=(14, 4))
    plt.plot(np.linspace(0, duration_s, len(y)), y, linewidth=0.3, color="#999")
    ax = plt.gca()
    for bt in beat_times_arr:
        ax.axvline(bt, color="#d62728", alpha=0.15, linewidth=0.5)
    plt.title(f"Beat grid (full {duration_s:.1f}s) — {g.bpm:.2f} BPM — {len(beat_times_arr)} beats shown")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    # sparse (every Nth beat) so labels stay legible over the full track
    step = max(1, len(beat_times_arr) // 40)
    pts = list(zip(beat_times_arr[::step].tolist(), range(1, len(beat_times_arr) + 1, step)))
    annotate_beat_counts(ax, pts, color="#d62728", draw_vlines=False, bpm=g.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["beat_grid"], show)


def plot_novelty(y: np.ndarray, sr: int, track: Track, out_dir: Path, show: bool = False) -> Path:
    """Spectral novelty curve with detected phrasing section boundaries."""
    p = track.phrasing
    g = track.groove
    nov = compute_novelty_curve(y, sr)
    nov_t = librosa.frames_to_time(np.arange(len(nov)), sr=sr, hop_length=512)
    plt.figure(figsize=(12, 4))
    plt.plot(nov_t, nov, color="#2ca02c", linewidth=0.7, label="Spectral novelty")
    ax = plt.gca()
    for b in p.segment_boundaries:
        ax.axvline(b, color="#ff7f0e", alpha=0.35, linewidth=0.8)
    plt.title("Phrasing engine — novelty curve & detected section boundaries")
    plt.xlabel("Time (s)")
    plt.ylabel("Novelty (0-1)")
    plt.legend(loc="upper right")
    beat_times_arr = np.array(g.beat_times)
    pts = [(b, int(np.searchsorted(beat_times_arr, b))) for b in p.segment_boundaries]
    annotate_beat_counts(ax, pts, color="#ff7f0e", draw_vlines=False, bpm=g.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["novelty"], show)


def plot_chromagram(y: np.ndarray, sr: int, track: Track, out_dir: Path, show: bool = False) -> Path:
    """Full-duration chromagram with the detected key and drop/breakdown transitions."""
    mo = track.mood
    p = track.phrasing
    g = track.groove
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sr, cmap="magma")
    plt.colorbar(label="Intensity")
    plt.title(f"Chromagram (full {len(y) / sr:.1f}s) -> detected {mo.key} ({mo.camelot_key})")
    pts = transition_beats(p.segments, g.beat_times, t_max=None)
    annotate_beat_counts(plt.gca(), pts, color="white", bpm=g.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["chromagram"], show)


def plot_spectrogram(
    y: np.ndarray, sr: int, track: Track, out_dir: Path, hop_length: int = 512, show: bool = False
) -> Path:
    """Full-duration log-frequency STFT spectrogram (dB) with drop/breakdown transitions.

    Reuses `spectrogram.compute_spectrogram` — the same STFT-to-dB computation persisted
    to `data/spectrograms/{track_id}.npy` at analyze time — so this plot matches that
    array rather than a separate re-implementation.
    """
    p = track.phrasing
    g = track.groove
    S_db = compute_spectrogram(y, sr, hop_length=hop_length)
    plt.figure(figsize=(14, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(f"Spectrogram (full {len(y) / sr:.1f}s) — log-frequency STFT")
    pts = transition_beats(p.segments, g.beat_times, t_max=None)
    annotate_beat_counts(plt.gca(), pts, color="white", bpm=g.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["spectrogram"], show)


def plot_energy(y: np.ndarray, sr: int, track: Track, out_dir: Path, show: bool = False) -> Path:
    """RMS energy curve with its classified energy profile and drop/breakdown transitions."""
    cu = track.curation
    p = track.phrasing
    g = track.groove
    ec = cu.energy_curve
    ec_t = librosa.frames_to_time(np.arange(len(ec)), sr=sr, hop_length=512)
    plt.figure(figsize=(12, 4))
    plt.plot(ec_t, ec, color="#9467bd", linewidth=0.6)
    plt.fill_between(ec_t, ec, color="#9467bd", alpha=0.25)
    plt.title(f"Energy (RMS) curve — profile classified as '{cu.energy_type}'")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS energy")
    pts = transition_beats(p.segments, g.beat_times)
    annotate_beat_counts(plt.gca(), pts, color="black", bpm=g.bpm)
    return _savefig(out_dir, PLOT_FILENAMES["energy"], show)


def plot_mood_radar(moods: Dict[str, float], out_dir: Path, show: bool = False) -> Path:
    """Radar chart of the rule-based mood classifier's category scores."""
    labels = list(moods.keys())
    vals = list(moods.values())
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals_c = vals + vals[:1]
    ang_c = ang + ang[:1]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(ang_c, vals_c, color="#d62728", linewidth=2)
    ax.fill(ang_c, vals_c, color="#d62728", alpha=0.25)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels)
    ax.set_title("Mood profile (rule-based classifier)", pad=20)
    return _savefig(out_dir, PLOT_FILENAMES["mood_radar"], show)


def plot_structure_bars(
    y: np.ndarray,
    sr: int,
    bpm: float,
    beat_times,
    out_dir: Path,
    min_bars: int = 4,
    thresh_frac: float = 0.4,
    hop_length: int = 512,
    show: bool = False,
) -> Path:
    """Low-band (20-150 Hz kick+bass) energy with DROP/BREAKDOWN sections in bars.

    Same detector as `detect_structure.py` (`phrasing_engine.detect_energy_sections`),
    reusing the track's already-detected `beat_times`/`bpm` instead of re-running beat
    tracking, so this stays consistent with the rest of the plots.
    """
    beat_times = np.asarray(beat_times)
    energy = compute_lowband_energy(y, sr, hop_length=hop_length)
    sections = detect_energy_sections(
        energy, sr, bpm, hop_length=hop_length, min_bars=min_bars, thresh_frac=thresh_frac
    )
    energy_s, spb = smooth_lowband_energy(energy, sr, bpm, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)

    plt.figure(figsize=(13, 4))
    ax = plt.gca()
    plt.plot(times, energy_s, color="#333", linewidth=0.8, label="Kick+bass energy (20-150 Hz)")
    plt.axhline(thresh_frac * energy_s.max(), color="grey", ls="--", lw=0.8, label="kick-on threshold")

    rows = []
    for is_drop, t0, t1 in sections:
        bar = int(round(t0 / spb)) + 1
        label = "DROP" if is_drop else "BREAKDOWN"
        color = "#2ca02c" if is_drop else "#d62728"
        plt.axvspan(t0, t1, color=color, alpha=0.12)
        plt.axvline(t0, color=color, lw=1.2)
        plt.text(t0, energy_s.max() * 0.95, f"b{bar}", rotation=90, va="top", ha="right", fontsize=8, color=color)
        rows.append((bar, t0, label))

    plt.title(f"Structure from low-band energy — green=DROP, red=BREAKDOWN  ({bpm:.1f} BPM)")
    plt.xlabel("Time (s)")
    plt.ylabel("Low-band energy")
    plt.legend(loc="upper right", fontsize=8)
    if rows:
        ax2 = ax.twiny()  # must come after title/xlabel/ylabel/legend — twiny() makes the top axis current
        ax2.set_xlim(ax.get_xlim())
        xt = [t0 for _, t0, _ in rows]
        beat_nums = [int(np.searchsorted(beat_times, t0)) for t0 in xt]
        ax2.set_xticks(xt)
        ax2.set_xticklabels([f"#{n}" for n in beat_nums], rotation=90, fontsize=7)
        ax2.set_xlabel("Beat count")
    return _savefig(out_dir, PLOT_FILENAMES["structure_bars"], show)


def generate_all_plots(
    file_path: str,
    out_dir: Optional[str] = None,
    key: Optional[str] = None,
    show: bool = False,
) -> List[Path]:
    """Load a track, run the full DSP+AI pipeline, and render all 8 diagnostic plots.

    Args:
        file_path: Path to the audio file.
        out_dir: Base plots directory (default `DEFAULT_PLOTS_DIR`). Plots are written
            under `out_dir/<key>/`.
        key: Subdirectory name (default: the file's stem). Pass a DB track_id (as str)
            to namespace plots the same way `spectrogram_key` does for spectrograms.
        show: Also open interactive windows (requires a non-Agg backend to be set
            *before* this module is imported).

    Returns:
        List of the 8 saved plot paths, in the order:
        waveform, beat_grid, novelty, chromagram, spectrogram, energy, mood_radar,
        structure_bars.
    """
    from .extractor import extract_track_features
    from ..ai.classifier import classify_mood
    from ..ingestion.loader import AudioLoader

    path = Path(file_path)
    loader = AudioLoader()
    audio_data = loader.load_audio(path)
    if not audio_data:
        raise ValueError(f"Failed to load audio: {file_path}")
    y, sr = audio_data["audio_array"], audio_data["sample_rate"]

    track = extract_track_features(str(path))
    mood_result = classify_mood(y, sr)
    moods = mood_result.get("moods", {})

    target_dir = Path(out_dir or DEFAULT_PLOTS_DIR) / str(key or path.stem)

    return [
        plot_waveform(y, sr, track, target_dir, show),
        plot_beat_grid(y, sr, track, target_dir, show),
        plot_novelty(y, sr, track, target_dir, show),
        plot_chromagram(y, sr, track, target_dir, show),
        plot_spectrogram(y, sr, track, target_dir, show=show),
        plot_energy(y, sr, track, target_dir, show),
        plot_mood_radar(moods, target_dir, show),
        plot_structure_bars(y, sr, track.groove.bpm, track.groove.beat_times, target_dir, show=show),
    ]
