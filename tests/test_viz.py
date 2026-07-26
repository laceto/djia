"""Tests for the Traktor-style waveform renderer (src/viz/waveform.py)."""

import numpy as np
import pytest

from src.viz.waveform import (
    WaveformStyle,
    _bandsplit_energy,
    _column_colors,
    _resample_to_columns,
    render_waveform,
)


def _tone(freq: float, sr: int = 22050, dur: float = 2.0) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def test_render_shape_and_dtype():
    y = _tone(220.0)
    style = WaveformStyle(width=400, height=120, show_beatgrid=False)
    img = render_waveform(y, 22050, style=style)
    assert img.shape == (120, 400, 3)
    assert img.dtype == np.uint8


def test_background_present_on_silence():
    y = np.zeros(22050, dtype=np.float64)
    style = WaveformStyle(width=200, height=80, show_beatgrid=False)
    img = render_waveform(y, 22050, style=style)
    # Silence still draws a 1px centre line, but the top row must be background.
    assert tuple(img[0, 0]) == style.background


def test_resample_to_columns_exact_width():
    out = _resample_to_columns(np.arange(1000.0), 250)
    assert out.shape == (250,)


def test_low_tone_reads_warmer_than_high_tone():
    """A bass tone should mix redder (more R than B); a treble tone the reverse."""
    style = WaveformStyle()
    sr, n_fft = 22050, 2048
    import librosa

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def color_of(freq):
        S = np.abs(librosa.stft(_tone(freq, sr=sr), n_fft=n_fft))
        low, mid, high = _bandsplit_energy(S, freqs, style.low_hz, style.high_hz)
        g = style.band_gamma
        cols = _column_colors(
            np.power(low.mean(keepdims=True), g) * style.low_gain,
            np.power(mid.mean(keepdims=True), g) * style.mid_gain,
            np.power(high.mean(keepdims=True), g) * style.high_gain,
            style,
        )
        return cols[0]  # (R, G, B)

    bass = color_of(80.0)
    treble = color_of(8000.0)
    assert bass[0] > bass[2]  # bass: red beats blue
    assert treble[2] > treble[0]  # treble: blue beats red


def test_playhead_draws_a_column():
    y = _tone(440.0)
    style = WaveformStyle(width=300, height=100, show_beatgrid=False, playhead=0.5)
    img = render_waveform(y, 22050, style=style)
    x = int(0.5 * (300 - 1))
    assert tuple(img[0, x]) == style.playhead_color


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
