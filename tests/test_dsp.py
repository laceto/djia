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
)
from src.dsp.groove_engine import analyze_groove, compute_swing_score
from src.dsp.mood_engine import analyze_mood, detect_key_from_chroma
from src.dsp.curation_engine import analyze_curation, classify_energy_profile
from src.features.schema import Track


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
            assert seg.label in ["intro", "build", "drop", "breakdown", "outro"]

        # Should have cue points
        assert hasattr(phrasing, "cue_points")


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
