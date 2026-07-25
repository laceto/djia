"""Tests for the model-free stem-proxy feature extractor."""

import numpy as np
import pytest

from src.dsp.stem_profile import (
    STEM_PROFILE_KEYS,
    compute_stem_profile,
)

SR = 22050
DUR = 4.0


def _t():
    return np.arange(int(SR * DUR)) / SR


def _sub_heavy():
    """Sustained 40 Hz sine — almost all energy in the sub band."""
    return 0.8 * np.sin(2 * np.pi * 40 * _t())


def _bright_hats():
    """Repeated short high-frequency noise bursts — busy top-end transients."""
    t = _t()
    y = np.zeros_like(t)
    rng = np.random.default_rng(0)
    burst = int(0.02 * SR)
    step = int(0.1 * SR)  # ~10 hats/sec
    for start in range(0, len(t) - burst, step):
        seg = rng.standard_normal(burst) * np.exp(-np.linspace(0, 6, burst))
        y[start:start + burst] += 0.5 * seg
    # High-pass-ish: emphasize highs by adding a 10 kHz tone
    y += 0.2 * np.sin(2 * np.pi * 10000 * t)
    return y


def _vocal_band_tone():
    """Sustained harmonic tone in the vocal band (800 Hz + overtones)."""
    t = _t()
    return (
        0.5 * np.sin(2 * np.pi * 800 * t)
        + 0.3 * np.sin(2 * np.pi * 1600 * t)
        + 0.2 * np.sin(2 * np.pi * 2400 * t)
    )


class TestStemProfileContract:
    def test_returns_all_keys(self):
        profile = compute_stem_profile(_sub_heavy(), SR)
        assert set(profile.keys()) == set(STEM_PROFILE_KEYS)

    def test_empty_signal_all_none(self):
        profile = compute_stem_profile(np.array([]), SR)
        assert all(profile[k] is None for k in STEM_PROFILE_KEYS)

    def test_silence_all_none(self):
        profile = compute_stem_profile(np.zeros(int(SR * DUR)), SR)
        assert all(profile[k] is None for k in STEM_PROFILE_KEYS)

    def test_none_signal(self):
        assert compute_stem_profile(None, SR) == {k: None for k in STEM_PROFILE_KEYS}

    def test_bad_sr(self):
        assert compute_stem_profile(_sub_heavy(), 0) == {k: None for k in STEM_PROFILE_KEYS}

    def test_stereo_is_mixed_down(self):
        mono = _sub_heavy()
        stereo = np.stack([mono, mono])  # (2, N)
        profile = compute_stem_profile(stereo, SR)
        assert profile["sub_ratio"] is not None


class TestStemProfileSemantics:
    def test_sub_heavy_has_high_sub_ratio(self):
        profile = compute_stem_profile(_sub_heavy(), SR)
        # A 40 Hz sine should put the vast majority of energy in the sub band.
        assert profile["sub_ratio"] > 0.5

    def test_bright_has_lower_sub_than_sub_heavy(self):
        sub = compute_stem_profile(_sub_heavy(), SR)
        bright = compute_stem_profile(_bright_hats(), SR)
        assert bright["sub_ratio"] < sub["sub_ratio"]

    def test_bright_has_more_hat_activity(self):
        sub = compute_stem_profile(_sub_heavy(), SR)
        bright = compute_stem_profile(_bright_hats(), SR)
        # Busy high-frequency bursts -> more onsets in the hat band than a
        # steady sub sine (which has ~none).
        assert bright["hat_rate"] >= sub["hat_rate"]

    def test_vocal_tone_has_more_vocal_presence_than_sub(self):
        sub = compute_stem_profile(_sub_heavy(), SR)
        vocal = compute_stem_profile(_vocal_band_tone(), SR)
        assert vocal["vocal_presence"] > sub["vocal_presence"]

    def test_ratios_in_unit_range(self):
        for signal in (_sub_heavy(), _bright_hats(), _vocal_band_tone()):
            p = compute_stem_profile(signal, SR)
            for key in ("sub_ratio", "bass_ratio", "vocal_presence"):
                assert 0.0 <= p[key] <= 1.0
            for key in ("kick_rate", "perc_rate", "hat_rate"):
                assert p[key] >= 0.0
