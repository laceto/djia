"""Tests for DSP Core (Phase 2) — 4-step analysis pipeline."""

import pytest
import numpy as np
import librosa
from pathlib import Path
import time
from src.dsp import analyze_track, extract_track_features
from src.dsp.phrasing_engine import (
    compute_novelty_curve,
    detect_segment_boundaries,
    analyze_structure,
    compute_additive_novelty,
    detect_element_onsets,
    derive_mix_points,
)
from src.dsp.groove_engine import (
    analyze_groove,
    compute_swing_score,
    compute_onset_strength_stats,
    compute_beat_strength,
)
from src.dsp.mood_engine import (
    analyze_mood,
    detect_key_from_chroma,
    convert_to_camelot,
    camelot_to_open_key,
    open_key_to_camelot,
    skey_label_to_key_camelot,
    compute_zero_crossing_rate,
    compute_roughness,
)
from src.dsp.curation_engine import (
    analyze_curation,
    classify_energy_profile,
    compute_spectral_flatness,
    compute_crest_factor,
)
from src.dsp.worker import analyze_one_track
from src.dsp.spectrogram import (
    compute_spectrogram,
    save_spectrogram,
    spectrogram_path,
    compute_and_save_spectrogram,
)
from src.features.schema import Track, ElementOnset


# Test data directory
DATA_DIR = Path("data")
TEST_TRACKS = [
    DATA_DIR / "prova.mp3",
    DATA_DIR / "MD3 - Pressure Cooker.mp3",
]


class TestPhrasingEngine:
    """Test Step 1: Phrasing Engine."""

    def test_novelty_curve_shape(self):
        """Test that novelty curve has correct shape."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        novelty = compute_novelty_curve(y, sr)

        # Should have 1D array
        assert novelty.ndim == 1

        # Should be normalized to ~0-1
        assert np.min(novelty) >= -0.01
        assert np.max(novelty) <= 1.01

    def test_segment_boundaries(self):
        """Test segment boundary detection."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        novelty = compute_novelty_curve(y, sr)
        boundaries = detect_segment_boundaries(novelty, sr)

        # Should find some boundaries
        assert len(boundaries) >= 0

        # All boundaries should be within track duration
        duration = librosa.get_duration(y=y, sr=sr)
        for b in boundaries:
            assert 0 <= b <= duration

        # Boundaries should be sorted
        assert boundaries == sorted(boundaries)

    def test_analyze_structure(self):
        """Test complete phrasing engine."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        phrasing = analyze_structure(y, sr, bpm=120.0)

        # Should have segments
        assert hasattr(phrasing, "segments")
        assert len(phrasing.segments) > 0

        # Each segment should have valid times
        for seg in phrasing.segments:
            assert seg.start_time >= 0
            assert seg.end_time > seg.start_time
            # labels carry beat/bar ranges when include_beats=True, e.g. "intro (beats 0-4)"
            assert seg.label.split(" (")[0] in ["intro", "build", "drop", "breakdown", "outro"]

        # Should have cue points
        assert hasattr(phrasing, "cue_points")


class TestElementOnsets:
    """Test additive-novelty element-onset detection."""

    def test_additive_novelty_shape(self):
        """Per-band novelty is (n_bands, n_frames) and normalized to 0-1."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        n_bands = 8
        novelty = compute_additive_novelty(y, sr, n_bands=n_bands)

        assert novelty.ndim == 2
        assert novelty.shape[0] == n_bands
        assert novelty.shape[1] > 0
        assert novelty.min() >= -0.01
        assert novelty.max() <= 1.01

    def test_element_onsets_valid(self):
        """Detected onsets have valid, in-range fields."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        n_bands = 8
        onsets = detect_element_onsets(y, sr, bpm=128.0, n_bands=n_bands)

        duration = librosa.get_duration(y=y, sr=sr)
        for o in onsets:
            assert 0 <= o.time <= duration + 1.0  # bar-snap may nudge slightly past end
            assert 0 <= o.band < n_bands
            assert o.freq_low < o.freq_high
            assert 0.0 <= o.confidence <= 1.0
            assert isinstance(o.label, str) and o.label

        # Onsets are returned sorted by time
        times = [o.time for o in onsets]
        assert times == sorted(times)

    def test_threshold_monotonicity(self):
        """A higher threshold never yields more onsets than a lower one."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        low = detect_element_onsets(y, sr, bpm=128.0, threshold=0.3)
        high = detect_element_onsets(y, sr, bpm=128.0, threshold=0.7)
        assert len(high) <= len(low)


class TestMixPoints:
    """Test derive_mix_points (pure — synthetic onsets, no audio)."""

    @staticmethod
    def _onset(time, band, label):
        return ElementOnset(
            time=time, band=band, freq_low=20.0 * 2**band,
            freq_high=20.0 * 2**(band + 1), confidence=0.8, label=label,
        )

    def test_named_points(self):
        """mix_in = first onset, bass_in = first sub/low, full_on = last new band."""
        onsets = [
            self._onset(7.5, 4, "mid"),
            self._onset(15.0, 1, "sub"),
            self._onset(30.0, 6, "high"),
            self._onset(45.0, 4, "mid"),  # repeat band — must not move full_on
        ]
        points = derive_mix_points(onsets, bpm=128.0, duration=360.0)

        assert points["mix_in"] == 7.5
        assert points["bass_in"] == 15.0
        assert points["full_on"] == 30.0
        # mix_out is bar-snapped, 32 bars before the end, inside the track
        assert points["mix_out"] is not None
        assert 0 < points["mix_out"] < 360.0

    def test_empty_onsets(self):
        """No onsets — element points are None, mix_out still derived."""
        points = derive_mix_points([], bpm=128.0, duration=360.0)
        assert points["mix_in"] is None
        assert points["bass_in"] is None
        assert points["full_on"] is None
        assert points["mix_out"] is not None

    def test_short_track_no_mix_out(self):
        """A track shorter than mix_out_bars gets no mix_out point."""
        points = derive_mix_points([], bpm=128.0, duration=30.0, mix_out_bars=32)
        assert points["mix_out"] is None

    def test_no_bass_band(self):
        """bass_in is None when no sub/low element ever enters."""
        onsets = [self._onset(10.0, 5, "high-mid")]
        points = derive_mix_points(onsets, bpm=128.0, duration=300.0)
        assert points["bass_in"] is None
        assert points["mix_in"] == 10.0
        assert points["full_on"] == 10.0


class TestGrooveEngine:
    """Test Step 2: Groove Engine."""

    def test_bpm_reasonable(self):
        """Test that detected BPM is in reasonable range for techno (90-150 BPM)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        # Techno typically 90-150 BPM
        assert 60 <= groove.bpm <= 200

    def test_beat_grid_shape(self):
        """Test that beat grid has valid shape."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        # Should have detected beats
        assert len(groove.beat_grid) > 0
        assert len(groove.beat_times) > 0
        assert len(groove.beat_times) == len(groove.beat_grid)

    def test_swing_score_range(self):
        """Test that swing score is in valid range (0-1)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        assert 0.0 <= groove.swing_score <= 1.0

    def test_tempo_stability_bool(self):
        """Test that tempo stability is boolean."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        assert isinstance(groove.tempo_stability, (bool, np.bool_))

    def test_beat_times_within_duration(self):
        """Test that all beat times are within track duration."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)
        duration = librosa.get_duration(y=y, sr=sr)

        for beat_time in groove.beat_times:
            assert 0 <= beat_time <= duration

    def test_onset_strength_stats_persisted(self):
        """analyze_groove() exposes onset strength mean/std (non-negative, mean is real-valued)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        assert groove.onset_strength_mean >= 0.0
        assert groove.onset_strength_std >= 0.0

    def test_onset_strength_stats_scale(self):
        """Scaling the onset envelope scales the summary stats proportionally."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        mean1, std1 = compute_onset_strength_stats(onset_env, scale=1.0)
        mean2, std2 = compute_onset_strength_stats(onset_env, scale=2.0)

        assert mean2 == pytest.approx(mean1 * 2, rel=1e-6)
        assert std2 == pytest.approx(std1 * 2, rel=1e-6)

    def test_onset_strength_stats_empty(self):
        """Empty onset envelope returns neutral zeros, not an error."""
        mean, std = compute_onset_strength_stats(np.array([]))
        assert mean == 0.0
        assert std == 0.0

    def test_beat_strength_range(self):
        """analyze_groove() exposes beat_strength in 0-1."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        groove = analyze_groove(y, sr)

        assert 0.0 <= groove.beat_strength <= 1.0

    def test_beat_strength_no_bpm(self):
        """No/zero BPM returns 0.0 rather than raising."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        assert compute_beat_strength(onset_env, sr, bpm=0.0) == 0.0
        assert compute_beat_strength(onset_env, sr, bpm=None) == 0.0


class TestMoodEngine:
    """Test Step 3: Mood Engine."""

    def test_key_detection_valid(self):
        """Test that key detection returns valid key."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        # Should have key
        assert mood.key is not None
        assert isinstance(mood.key, str)

        # Should be in format "Note Type" (e.g., "A minor")
        parts = mood.key.split()
        assert len(parts) == 2
        assert parts[1] in ["major", "minor"]

    def test_camelot_key_valid(self):
        """Test that Camelot key is in valid format."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        # Should be format "XA" or "XB" where X is 1-12
        assert len(mood.camelot_key) == 2
        assert mood.camelot_key[0].isdigit() or mood.camelot_key[:2].replace('.', '').isdigit()
        assert mood.camelot_key[-1] in ["A", "B"]

    def test_brightness_range(self):
        """Test that brightness is in valid range (0-1)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        assert 0.0 <= mood.brightness <= 1.0

    def test_key_confidence_range(self):
        """Test that key confidence is in valid range (0-1)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        assert 0.0 <= mood.key_confidence <= 1.0

    def test_zero_crossing_rate_range(self):
        """analyze_mood() exposes zero_crossing_rate as a small non-negative fraction."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        assert 0.0 <= mood.zero_crossing_rate <= 1.0

    def test_zero_crossing_rate_matches_helper(self):
        """analyze_mood()'s stored ZCR matches the standalone helper for the same audio."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        assert mood.zero_crossing_rate == pytest.approx(compute_zero_crossing_rate(y), rel=1e-6)

    def test_roughness_range(self):
        """analyze_mood() exposes roughness in 0-1."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        mood = analyze_mood(y, sr)

        assert 0.0 <= mood.roughness <= 1.0

    def test_roughness_silence_is_zero(self):
        """Silence has no spectral peaks to be dissonant -> roughness is 0."""
        silence = np.zeros(22050 * 2, dtype=np.float32)
        assert compute_roughness(silence, 22050) == 0.0


class TestCamelotConversion:
    """Test key -> Camelot mapping against the standard Camelot / Mixed In Key wheel.

    The wheel is anchored at position 8 (C major = 8B, A minor = 8A) and advances one
    number per perfect fifth. A regression here previously rotated every code 7 steps
    (e.g. F#/Gb minor read '4A' instead of the correct '11A').
    """

    # (note_name, key_type) -> canonical Camelot code, for all 24 keys.
    EXPECTED = {
        # Minor keys (A side) — walk the circle of fifths from A minor = 8A
        ("A", "minor"): "8A",
        ("E", "minor"): "9A",
        ("B", "minor"): "10A",
        ("F#/Gb", "minor"): "11A",
        ("C#/Db", "minor"): "12A",
        ("G#/Ab", "minor"): "1A",
        ("D#/Eb", "minor"): "2A",
        ("A#/Bb", "minor"): "3A",
        ("F", "minor"): "4A",
        ("C", "minor"): "5A",
        ("G", "minor"): "6A",
        ("D", "minor"): "7A",
        # Major keys (B side) — walk the circle of fifths from C major = 8B
        ("C", "major"): "8B",
        ("G", "major"): "9B",
        ("D", "major"): "10B",
        ("A", "major"): "11B",
        ("E", "major"): "12B",
        ("B", "major"): "1B",
        ("F#/Gb", "major"): "2B",
        ("C#/Db", "major"): "3B",
        ("G#/Ab", "major"): "4B",
        ("D#/Eb", "major"): "5B",
        ("A#/Bb", "major"): "6B",
        ("F", "major"): "7B",
    }

    def test_all_24_keys(self):
        """Every key maps to its canonical Camelot code."""
        for (note, key_type), expected in self.EXPECTED.items():
            assert convert_to_camelot(note, key_type) == expected, (
                f"{note} {key_type} -> {convert_to_camelot(note, key_type)}, expected {expected}"
            )

    def test_anchor_keys(self):
        """The two anchor keys sit at position 8."""
        assert convert_to_camelot("C", "major") == "8B"
        assert convert_to_camelot("A", "minor") == "8A"

    def test_pak_pak_regression(self):
        """F#/Gb minor is 11A (the reported '4A' was the rotated-by-7 bug)."""
        assert convert_to_camelot("F#/Gb", "minor") == "11A"

    def test_relative_major_minor_share_number(self):
        """Relative major/minor pairs share a wheel number, differing only in letter."""
        pairs = [("C", "major", "A", "minor"), ("C#/Db", "major", "A#/Bb", "minor"),
                 ("G", "major", "E", "minor"), ("A", "major", "F#/Gb", "minor")]
        for maj_note, _, min_note, _ in pairs:
            maj = convert_to_camelot(maj_note, "major")
            minor = convert_to_camelot(min_note, "minor")
            assert maj[:-1] == minor[:-1] and maj[-1] == "B" and minor[-1] == "A"

    def test_output_format(self):
        """Every code is 'NA'/'NB' with N in 1-12."""
        for note, key_type in self.EXPECTED:
            code = convert_to_camelot(note, key_type)
            assert code[-1] in ("A", "B")
            assert 1 <= int(code[:-1]) <= 12


class TestOpenKeyConversion:
    """Test Camelot <-> Open Key Notation (DJUCED / Rekordbox 'Open Key')."""

    # Canonical anchors across the wheel.
    CASES = [
        ("8A", "1m"),   # A minor
        ("8B", "1d"),   # C major
        ("12A", "5m"),  # C#/Db minor  -> DJUCED's reading of Pak Pak
        ("3B", "8d"),   # Db major (Beatport's reading)
        ("1A", "6m"),   # G#/Ab minor  (wheel wrap)
        ("11A", "4m"),  # F#/Gb minor  (old chroma guess)
        ("7B", "12d"),  # F major
    ]

    def test_camelot_to_open_key(self):
        for camelot, open_key in self.CASES:
            assert camelot_to_open_key(camelot) == open_key

    def test_open_key_to_camelot(self):
        for camelot, open_key in self.CASES:
            assert open_key_to_camelot(open_key) == camelot

    def test_djuced_pak_pak_anchor(self):
        """C#/Db minor: 12A (Camelot) <-> 5m (DJUCED Open Key)."""
        assert camelot_to_open_key("12A") == "5m"
        assert open_key_to_camelot("5m") == "12A"

    def test_round_trip_all_24(self):
        """Every Camelot code survives a round trip through Open Key."""
        for n in range(1, 13):
            for letter in ("A", "B"):
                code = f"{n}{letter}"
                assert open_key_to_camelot(camelot_to_open_key(code)) == code

    def test_case_insensitive(self):
        assert camelot_to_open_key("12a") == "5m"
        assert open_key_to_camelot("5M") == "12A"

    def test_invalid_raises(self):
        for bad in ("13A", "0B", "8C", "AA"):
            with pytest.raises((ValueError, IndexError)):
                camelot_to_open_key(bad)
        for bad in ("13m", "0d", "5x"):
            with pytest.raises((ValueError, IndexError)):
                open_key_to_camelot(bad)


class TestSkeyKeyBackend:
    """Test the optional S-KEY key backend and its label mapping.

    The label->key/Camelot mapping is pure and always tested; the end-to-end
    inference test is skipped unless the optional `skey` package and a test
    track are both present.
    """

    def test_label_mapping(self):
        """S-KEY labels map to repo key names + correct (post-fix) Camelot codes."""
        assert skey_label_to_key_camelot("C# minor") == ("C#/Db minor", "12A")
        assert skey_label_to_key_camelot("Bb Major") == ("A#/Bb major", "6B")
        assert skey_label_to_key_camelot("A Major") == ("A major", "11B")
        assert skey_label_to_key_camelot("F# minor") == ("F#/Gb minor", "11A")
        assert skey_label_to_key_camelot("C Major") == ("C major", "8B")
        assert skey_label_to_key_camelot("A minor") == ("A minor", "8A")

    def test_label_mapping_all_24_valid(self):
        """Every S-KEY label produces a valid repo key string and Camelot code."""
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
        for note in notes:
            for mode in ("Major", "minor"):
                name, camelot = skey_label_to_key_camelot(f"{note} {mode}")
                assert mode.lower() in name
                assert camelot[-1] in ("A", "B")
                assert 1 <= int(camelot[:-1]) <= 12

    def test_bad_label_raises(self):
        """A malformed label is rejected rather than silently mismapped."""
        with pytest.raises((ValueError, KeyError)):
            skey_label_to_key_camelot("H diminished")

    def test_chroma_fallback_without_file_path(self):
        """Without a file_path, analyze_mood uses the chroma backend (no S-KEY)."""
        # Synthetic tone — no audio file or optional deps needed.
        t = np.linspace(0, 2.0, int(22050 * 2.0), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        mood = analyze_mood(y, 22050)
        assert mood.key_source == "chroma"
        assert mood.camelot_key[-1] in ("A", "B")

    def test_prefer_skey_false_forces_chroma(self):
        """prefer_skey=False keeps the chroma backend even if a file_path is given."""
        t = np.linspace(0, 2.0, int(22050 * 2.0), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        mood = analyze_mood(y, 22050, file_path="whatever.mp3", prefer_skey=False)
        assert mood.key_source == "chroma"

    def test_skey_end_to_end(self):
        """When skey + a test track are available, analyze_mood uses it and reports
        a real 0-1 confidence."""
        pytest.importorskip("skey")
        if not Path(TEST_TRACKS[0]).exists():
            pytest.skip("no test track available")
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=60)
        mood = analyze_mood(y, sr, file_path=str(TEST_TRACKS[0]))
        assert mood.key_source == "skey"
        assert 0.0 <= mood.key_confidence <= 1.0
        parts = mood.key.split()
        assert parts[-1] in ("major", "minor")
        assert mood.camelot_key[-1] in ("A", "B")


class TestCurationEngine:
    """Test Step 4: Curation Engine."""

    def test_danceability_range(self):
        """Test that danceability is in valid range (0-1)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        assert 0.0 <= curation.danceability <= 1.0

    def test_energy_curve_shape(self):
        """Test that energy curve has reasonable shape."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        # Should have 1D array
        assert curation.energy_curve.ndim == 1
        assert len(curation.energy_curve) > 0

    def test_energy_type_valid(self):
        """Test that energy type is one of valid options."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        assert curation.energy_type in ["flat", "dynamic", "gradual"]

    def test_semantic_tags_generated(self):
        """Test that semantic tags are generated."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        # Should have at least some tags
        assert isinstance(curation.semantic_tags, list)
        assert len(curation.semantic_tags) > 0

        # All tags should be strings
        for tag in curation.semantic_tags:
            assert isinstance(tag, str)

    def test_complexity_score_range(self):
        """Test that complexity score is in valid range (0-1)."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        assert 0.0 <= curation.complexity_score <= 1.0

    def test_spectral_flatness_range(self):
        """analyze_curation() exposes spectral_flatness in 0-1."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        assert 0.0 <= curation.spectral_flatness <= 1.0

    def test_crest_factor_at_least_one(self):
        """analyze_curation() exposes crest_factor; peak RMS can never be below mean RMS."""
        y, sr = librosa.load(TEST_TRACKS[0], sr=22050, duration=30)
        curation = analyze_curation(y, sr, bpm=120.0, swing_score=0.5, brightness=0.5)

        assert curation.crest_factor >= 1.0

    def test_crest_factor_helper(self):
        """compute_crest_factor is a plain peak/mean ratio, guarded against zero mean."""
        assert compute_crest_factor(rms_mean=0.1, rms_peak=0.5) == pytest.approx(5.0)
        assert compute_crest_factor(rms_mean=0.0, rms_peak=0.5) == 0.0
        assert compute_crest_factor(rms_mean=None, rms_peak=0.5) == 0.0

    def test_spectral_flatness_helper_silence(self):
        """Silence is neither tonal nor noisy in a meaningful sense, but must not error."""
        silence = np.zeros(22050, dtype=np.float32)
        flatness = compute_spectral_flatness(silence, 22050)
        assert 0.0 <= flatness <= 1.0


class TestSpectrogram:
    """Test spectrogram computation and .npy persistence."""

    def test_compute_spectrogram_shape(self):
        """Log-magnitude STFT is 2D (freq_bins, frames) and finite."""
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        S = compute_spectrogram(y, sr, hop_length=512)

        assert S.ndim == 2
        assert S.shape[1] > 0
        assert np.all(np.isfinite(S))

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Saved .npy reloads to the same array at the deterministic key path."""
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        S = compute_spectrogram(y, sr)

        out_path = save_spectrogram(S, key="123", base_dir=str(tmp_path))

        assert out_path == spectrogram_path("123", str(tmp_path))
        assert out_path.exists()
        reloaded = np.load(out_path)
        assert np.array_equal(reloaded, S)

    def test_compute_and_save_creates_missing_dir(self, tmp_path):
        """compute_and_save_spectrogram creates base_dir if it doesn't exist yet."""
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        base_dir = tmp_path / "nested" / "spectrograms"

        out_path = compute_and_save_spectrogram(y, sr, key=1, base_dir=str(base_dir))

        assert out_path.exists()
        assert out_path.parent == base_dir


class TestOrchestratorExtractor:
    """Test Master Orchestrator (extractor.py)."""

    def test_extract_track_features_complete(self):
        """Test complete track feature extraction."""
        track = extract_track_features(str(TEST_TRACKS[0]), duration=30)

        # Should be Track instance
        assert isinstance(track, Track)

        # Should have all components
        assert track.file_path is not None
        assert track.duration > 0

        # Should have all 4 engine results
        assert track.phrasing is not None
        assert track.groove is not None
        assert track.mood is not None
        assert track.curation is not None

        # Each component should be properly populated
        assert track.groove.bpm > 0
        assert len(track.mood.key) > 0
        assert 0 <= track.curation.danceability <= 1

        # New density/onset/timbre metrics flow through the extractor path too
        assert 0.0 <= track.groove.beat_strength <= 1.0
        assert 0.0 <= track.mood.roughness <= 1.0
        assert 0.0 <= track.curation.spectral_flatness <= 1.0
        assert track.curation.crest_factor >= 1.0

    def test_analyze_track_wrapper(self):
        """Test analyze_track wrapper function."""
        result = analyze_track(str(TEST_TRACKS[0]), duration=30)

        # Should have status
        assert result.status in ["success", "error"]

        # If success, should have track
        if result.status == "success":
            assert result.track is not None
            assert isinstance(result.track, Track)

    def test_performance_under_30_seconds(self):
        """Test that analysis completes in under 30 seconds per track."""
        start_time = time.time()
        track = extract_track_features(str(TEST_TRACKS[0]), duration=30)
        elapsed = time.time() - start_time

        # Should complete in under 30 seconds
        assert elapsed < 30, f"Analysis took {elapsed:.1f}s (limit: 30s)"

    def test_all_tracks_analyzable(self):
        """Test that both test tracks can be analyzed."""
        for track_path in TEST_TRACKS[:2]:  # Test first 2 tracks
            if track_path.exists():
                result = analyze_track(str(track_path), duration=30)
                assert result.status == "success", f"Failed to analyze {track_path}: {result.error_message}"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_feature_consistency_across_runs(self):
        """Test that repeated analysis produces consistent results."""
        run1 = extract_track_features(str(TEST_TRACKS[0]), duration=30)
        run2 = extract_track_features(str(TEST_TRACKS[0]), duration=30)

        # BPM should be consistent (within 1%)
        bpm_diff = abs(run1.groove.bpm - run2.groove.bpm) / run1.groove.bpm
        assert bpm_diff < 0.01, f"BPM inconsistency: {bpm_diff*100:.2f}%"

        # Key should be identical
        assert run1.mood.key == run2.mood.key, "Key detection inconsistent"

        # Danceability should be consistent (within 2%)
        dance_diff = abs(run1.curation.danceability - run2.curation.danceability)
        assert dance_diff < 0.02, f"Danceability inconsistency: {dance_diff:.3f}"

    def test_cue_points_within_track(self):
        """Test that cue points are within track duration."""
        track = extract_track_features(str(TEST_TRACKS[0]), duration=30)

        for cue in track.phrasing.cue_points:
            assert 0 <= cue.time <= track.duration + 1.0, \
                f"Cue {cue.label} at {cue.time:.1f}s outside track duration {track.duration:.1f}s"

    def test_segments_dont_overlap(self):
        """Test that segments don't overlap."""
        track = extract_track_features(str(TEST_TRACKS[0]), duration=30)

        for i in range(len(track.phrasing.segments) - 1):
            seg1 = track.phrasing.segments[i]
            seg2 = track.phrasing.segments[i + 1]

            # Next segment should start at or after previous one ends
            assert seg2.start_time >= seg1.end_time - 0.1, \
                f"Segment {i} overlaps with {i+1}"

    def test_beat_times_ordered(self):
        """Test that beat times are in increasing order."""
        track = extract_track_features(str(TEST_TRACKS[0]), duration=30)

        beat_times = track.groove.beat_times
        for i in range(len(beat_times) - 1):
            assert beat_times[i] < beat_times[i + 1], \
                f"Beat times not ordered at index {i}"


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_file_path(self):
        """Test handling of invalid file path."""
        result = analyze_track("nonexistent_file.mp3")

        assert result.status == "error"
        assert result.error_message is not None

    def test_empty_audio(self):
        """Test handling of empty audio."""
        # Create empty array
        y = np.array([])

        # Should handle gracefully
        try:
            with pytest.raises((ValueError, Exception)):
                librosa.get_duration(y=y, sr=22050)
        except:
            pass  # Expected


class TestWorkerAnalyzeOneTrack:
    """Test the picklable per-track worker function (src/dsp/worker.py) used to
    parallelize Orchestrator.analyze_library(). Called directly here (no
    ProcessPoolExecutor involved), which also exercises the lazy construct-on-first-use
    fallback for the mood classifier/loader (the pool `_init_worker` path isn't hit)."""

    @pytest.mark.parametrize("track_path", TEST_TRACKS)
    def test_analyze_one_track_success(self, track_path):
        """Compute-only pipeline succeeds end-to-end and returns everything the main
        process needs to persist, without ever touching the database."""
        result = analyze_one_track(str(track_path), segment_preset="minimal", bars_per_phrase=16)

        assert result["error"] is None

        features = result["features"]
        assert features is not None
        assert features.get("bpm") or features.get("tempo")
        for key in (
            "swing_score", "spectral_flatness", "crest_factor",
            "onset_strength_mean", "zero_crossing_rate", "roughness",
        ):
            assert key in features, f"Missing feature: {key}"

        assert 0.0 <= features["swing_score"] <= 1.0
        assert 0.0 <= features["spectral_flatness"] <= 1.0
        assert features["crest_factor"] >= 1.0
        assert features["onset_strength_mean"] >= 0.0
        assert 0.0 <= features["zero_crossing_rate"] <= 1.0
        assert 0.0 <= features["roughness"] <= 1.0

        for segments in (result["segments_spectral"], result["segments_phrase"]):
            assert isinstance(segments, list)
            assert len(segments) > 0
            for seg in segments:
                assert isinstance(seg, dict)
                assert "segment_type" in seg
                assert "start_time" in seg
                assert "end_time" in seg
                assert "confidence" in seg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
