"""Traktor Pro-style frequency-coloured waveform renderer.

Traktor draws a waveform whose *height* follows the signal amplitude and whose
*colour* follows the spectral balance at each point in time: low frequencies
(kick / bass) read warm (red / orange), mids read orange / amber and highs
(hats / cymbals / transients) tint cool (blue / violet). This module reproduces
that look by:

  1. computing a short-time magnitude spectrogram,
  2. summing it into low / mid / high bands per time column,
  3. mixing those band energies into an RGB colour per column, and
  4. rasterising a centre-mirrored, amplitude-scaled waveform directly into an
     RGB image (fast and crisp, no matplotlib line overdraw).

Optionally an overlaid beat grid and a playhead are drawn, matching Traktor's
deck view. The public entry points are :func:`render_waveform` (works on an
in-memory signal) and :func:`render_waveform_file` (loads a file first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RGB = Tuple[int, int, int]


@dataclass
class WaveformStyle:
    """Look-and-feel knobs for the rendered waveform.

    The three ``*_color`` values are the base colours that the low / mid / high
    frequency bands contribute; the per-column colour is an energy-weighted mix
    of them. Defaults reproduce Traktor's stock warm/cool scheme seen on the
    coloured deck waveform.
    """

    width: int = 2000
    height: int = 340
    background: RGB = (11, 11, 13)

    # Base band colours (low, mid, high). Highs push the mix toward blue/violet.
    low_color: RGB = (255, 40, 24)  # kick / sub — red
    mid_color: RGB = (255, 138, 30)  # body — orange / amber
    high_color: RGB = (70, 120, 255)  # hats / air — blue
    high_accent: RGB = (150, 70, 255)  # violet tint for high-heavy transients

    # Band split points in Hz (edges of low, mid, high).
    low_hz: float = 200.0
    high_hz: float = 3000.0

    # Colour balance. Raw band energy is bass-heavy in dance music, so bands are
    # compressed (``band_gamma`` < 1 flattens the low dominance) and re-gained
    # before the colour mix — this is what lets hats/transients read blue/violet
    # instead of being swamped by the kick's red.
    band_gamma: float = 0.5
    low_gain: float = 1.15
    mid_gain: float = 1.2
    high_gain: float = 1.9

    # Vertical fraction of the panel the waveform is allowed to fill (0-1).
    amp_headroom: float = 0.94
    # Perceptual lift so quiet detail stays visible (gamma < 1 brightens).
    amp_gamma: float = 0.62

    # Beat grid.
    show_beatgrid: bool = True
    beat_color: RGB = (150, 150, 155)
    downbeat_color: RGB = (225, 225, 230)
    beats_per_bar: int = 4
    grid_alpha: float = 0.16

    # Playhead (drawn at ``playhead`` fraction of the width if not None).
    playhead: Optional[float] = None
    playhead_color: RGB = (255, 60, 40)


# Frequencies below this contribute nothing (DC / rumble guard).
_MIN_HZ = 20.0


def _bandsplit_energy(
    S: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum spectrogram magnitude into low / mid / high energy per time column.

    ``S`` is (n_freq, n_frames) magnitude, ``freqs`` the bin centre frequencies.
    Returns three (n_frames,) arrays.
    """
    low_mask = (freqs >= _MIN_HZ) & (freqs < low_hz)
    mid_mask = (freqs >= low_hz) & (freqs < high_hz)
    high_mask = freqs >= high_hz

    low = S[low_mask].sum(axis=0)
    mid = S[mid_mask].sum(axis=0)
    high = S[high_mask].sum(axis=0)
    return low, mid, high


def _column_colors(
    low: np.ndarray, mid: np.ndarray, high: np.ndarray, style: WaveformStyle
) -> np.ndarray:
    """Map per-column band energies to an (n, 3) float RGB array in 0-255.

    Each column's colour is the energy-weighted blend of the band base colours.
    Because ``high_color`` (and its violet accent for high-dominant columns) is
    cool, columns rich in transients/air drift blue/violet against the warm
    body — the signature Traktor contrast.
    """
    total = low + mid + high
    total = np.where(total <= 0, 1.0, total)

    wl = low / total
    wm = mid / total
    wh = high / total

    lc = np.array(style.low_color, dtype=np.float64)
    mc = np.array(style.mid_color, dtype=np.float64)
    hc = np.array(style.high_color, dtype=np.float64)
    ac = np.array(style.high_accent, dtype=np.float64)

    # High band leans from plain blue toward violet as it becomes dominant, so
    # bright hat/transient hits pop violet rather than flat blue.
    hi_dom = np.clip((wh - 0.34) / 0.4, 0.0, 1.0)[:, None]
    high_rgb = hc[None, :] * (1 - hi_dom) + ac[None, :] * hi_dom

    colors = wl[:, None] * lc[None, :] + wm[:, None] * mc[None, :] + wh[:, None] * high_rgb

    # Renormalise toward full saturation so the mix stays vivid, not muddy.
    peak = colors.max(axis=1, keepdims=True)
    peak = np.where(peak <= 0, 1.0, peak)
    colors = colors / peak * (0.55 + 0.45 * (peak / 255.0)) * 255.0
    return np.clip(colors, 0, 255)


def _resample_to_columns(values: np.ndarray, width: int, reducer=np.max) -> np.ndarray:
    """Reduce a 1-D array to exactly ``width`` columns via bucketed reduction."""
    n = len(values)
    if n == width:
        return values.astype(np.float64)
    edges = np.linspace(0, n, width + 1).astype(int)
    out = np.empty(width, dtype=np.float64)
    for i in range(width):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            b = a + 1
        seg = values[a : min(b, n)]
        out[i] = reducer(seg) if len(seg) else 0.0
    return out


def render_waveform(
    y: np.ndarray,
    sr: int,
    style: Optional[WaveformStyle] = None,
    beat_frames_sec: Optional[Sequence[float]] = None,
    downbeat_offset: int = 0,
) -> np.ndarray:
    """Render ``y`` into a Traktor-style RGB image array (H, W, 3) uint8.

    Args:
        y: mono signal (float).
        sr: sample rate of ``y``.
        style: :class:`WaveformStyle` overrides (defaults if None).
        beat_frames_sec: beat times in seconds for the grid. If None and the
            style requests a grid, beats are detected with librosa.
        downbeat_offset: index into the beat list that is a downbeat, so every
            ``beats_per_bar`` beats from it is drawn brighter (bar line).

    Returns:
        (height, width, 3) uint8 image, ready to save with PIL/matplotlib.
    """
    style = style or WaveformStyle()
    W, H = style.width, style.height

    if y.ndim > 1:
        y = y.mean(axis=0)
    y = np.asarray(y, dtype=np.float64)
    duration = len(y) / sr if sr else 0.0

    # --- spectral colour ----------------------------------------------------
    import librosa  # local import: keeps module importable without audio deps

    n_fft = 2048
    hop = max(1, len(y) // (W * 2)) if len(y) > W * 2 else 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low, mid, high = _bandsplit_energy(S, freqs, style.low_hz, style.high_hz)

    low_c = _resample_to_columns(low, W, reducer=np.mean)
    mid_c = _resample_to_columns(mid, W, reducer=np.mean)
    high_c = _resample_to_columns(high, W, reducer=np.mean)

    # Compress + re-gain so colour reflects spectral *balance*, not raw (bass-
    # dominated) energy — otherwise everything reads red.
    g = style.band_gamma
    low_c = np.power(low_c, g) * style.low_gain
    mid_c = np.power(mid_c, g) * style.mid_gain
    high_c = np.power(high_c, g) * style.high_gain
    colors = _column_colors(low_c, mid_c, high_c, style)  # (W, 3)

    # --- amplitude silhouette (peak envelope per column) --------------------
    amp = _resample_to_columns(np.abs(y), W, reducer=np.max)
    if amp.max() > 0:
        amp = amp / amp.max()
    amp = np.power(amp, style.amp_gamma)  # perceptual lift
    half = int(H * style.amp_headroom / 2)
    col_h = np.clip((amp * half).astype(int), 1, half)

    # --- rasterise ----------------------------------------------------------
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = np.array(style.background, dtype=np.uint8)
    center = H // 2

    rows = np.arange(H)[:, None]  # (H, 1)
    top = center - col_h  # (W,)
    bot = center + col_h
    mask = (rows >= top[None, :]) & (rows < bot[None, :])  # (H, W)
    color_plane = colors.T[:, None, :]  # (3, 1, W)
    color_full = np.broadcast_to(color_plane, (3, H, W)).transpose(1, 2, 0)
    img = np.where(mask[:, :, None], color_full.astype(np.uint8), img)

    # --- beat grid ----------------------------------------------------------
    if style.show_beatgrid and duration > 0:
        beats = beat_frames_sec
        if beats is None:
            try:
                _, bframes = librosa.beat.beat_track(y=y, sr=sr, units="time")
                beats = np.atleast_1d(bframes)
            except Exception as exc:  # pragma: no cover - detection is best-effort
                logger.warning("beat detection failed: %s", exc)
                beats = []
        _draw_beatgrid(img, beats, duration, style, downbeat_offset)

    # --- playhead -----------------------------------------------------------
    if style.playhead is not None:
        x = int(np.clip(style.playhead, 0, 1) * (W - 1))
        img[:, x] = np.array(style.playhead_color, dtype=np.uint8)

    return img


def _draw_beatgrid(
    img: np.ndarray,
    beats_sec: Sequence[float],
    duration: float,
    style: WaveformStyle,
    downbeat_offset: int,
) -> None:
    """Blend vertical beat lines into ``img`` in place."""
    H, W, _ = img.shape
    beat_rgb = np.array(style.beat_color, dtype=np.float64)
    down_rgb = np.array(style.downbeat_color, dtype=np.float64)
    a = style.grid_alpha
    for i, t in enumerate(beats_sec):
        if t < 0 or t > duration:
            continue
        x = int(t / duration * (W - 1))
        is_down = ((i - downbeat_offset) % max(1, style.beats_per_bar)) == 0
        rgb = down_rgb if is_down else beat_rgb
        alpha = min(1.0, a * (2.1 if is_down else 1.0))
        col = img[:, x].astype(np.float64)
        img[:, x] = (col * (1 - alpha) + rgb * alpha).astype(np.uint8)


def render_waveform_file(
    audio_path: str | Path,
    out_path: str | Path,
    style: Optional[WaveformStyle] = None,
    sr: int = 44100,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Path:
    """Load an audio file and write a Traktor-style waveform PNG.

    Args:
        audio_path: input audio (any librosa-readable format).
        out_path: destination PNG.
        style: optional :class:`WaveformStyle`.
        sr: analysis sample rate (44.1 kHz keeps highs crisp for colouring).
        duration: seconds to render (None = whole track).
        offset: start offset in seconds.

    Returns:
        The written ``out_path`` as a :class:`~pathlib.Path`.
    """
    import librosa
    from PIL import Image

    audio_path = Path(audio_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True, duration=duration, offset=offset)
    img = render_waveform(y, sr, style=style)
    Image.fromarray(img, "RGB").save(out_path)
    logger.info("wrote waveform %s (%dx%d)", out_path, img.shape[1], img.shape[0])
    return out_path
