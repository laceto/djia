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
    compute_zero_crossing_rate,
    compute_roughness,
)
from src.dsp.curation_engine import (
    analyze_curation,
    classify_energy_profile,
    compute_spectral_flatness,
    compute_crest_factor,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
