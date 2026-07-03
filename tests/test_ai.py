"""Tests for Phase 3 AI features - stem separation, mood classification, structure detection."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
import librosa

from src.ai.stem_separator import StemSeparator, separate_stems
from src.ai.classifier import MoodClassifier, classify_mood
from src.ai.segmentation import StructureSegmenter, detect_structure, StructurePoint
from src.ai.processor import AIProcessor, process_with_stems


# Fixtures
@pytest.fixture
def sample_audio():
    """Generate a sample audio track for testing."""
    sr = 22050
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, sr * duration)

    # Create a synthetic techno beat
    # Bass drum at 120 BPM (every 0.5 seconds)
    drums = np.sin(2 * np.pi * 60 * t) * (np.abs(np.sin(2 * np.pi * 2 * t)) > 0.5)

    # Bass line
    bass = 0.5 * np.sin(2 * np.pi * 55 * t)

    # Synth melody (pad)
    melody = 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 5)

    # Mix
    y = drums + bass + melody
    y = y / (np.max(np.abs(y)) + 1e-8) * 0.9  # Normalize

    return y, sr


@pytest.fixture
def temp_audio_file(sample_audio):
    """Create a temporary audio file."""
    y, sr = sample_audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, y, sr)
        temp_path = f.name

    yield temp_path

    # Cleanup - handle file locking on Windows
    try:
        Path(temp_path).unlink(missing_ok=True)
    except PermissionError:
        # File still in use, try again with a delay
        import time
        time.sleep(0.1)
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass  # Best effort cleanup


# Tests for MoodClassifier
class TestMoodClassifier:
    """Tests for mood classification."""

    def test_mood_classifier_initialization(self):
        """Test classifier initialization."""
        classifier = MoodClassifier(use_essentia=False)
        assert classifier is not None
        assert not classifier.use_essentia

    def test_classify_mood_returns_dict(self, sample_audio):
        """Test that classify_mood returns proper structure."""
        y, sr = sample_audio
        classifier = MoodClassifier(use_essentia=False)
        result = classifier.classify_mood(y, sr)

        assert isinstance(result, dict)
        assert 'moods' in result
        assert 'energy' in result
        assert 'danceability' in result

    def test_mood_scores_are_valid(self, sample_audio):
        """Test that mood scores are valid probabilities."""
        y, sr = sample_audio
        result = classify_mood(y, sr)

        moods = result['moods']
        for mood, score in moods.items():
            assert 0 <= score <= 1.0
            assert isinstance(mood, str)

    def test_mood_sum_approximately_one(self, sample_audio):
        """Test that mood scores sum to approximately 1.0."""
        y, sr = sample_audio
        result = classify_mood(y, sr)

        total_score = sum(result['moods'].values())
        assert 0.95 <= total_score <= 1.05

    def test_energy_level_valid(self, sample_audio):
        """Test that energy level is valid."""
        y, sr = sample_audio
        result = classify_mood(y, sr)

        energy = result['energy']
        assert energy in MoodClassifier.ENERGY_LEVELS

    def test_danceability_in_range(self, sample_audio):
        """Test danceability is in valid range."""
        y, sr = sample_audio
        result = classify_mood(y, sr)

        danceability = result['danceability']
        assert 0 <= danceability <= 1.0


# Tests for StructureSegmenter
class TestStructureSegmenter:
    """Tests for structural segmentation."""

    def test_segmenter_initialization(self):
        """Test segmenter initialization."""
        segmenter = StructureSegmenter()
        assert segmenter is not None

    def test_structure_point_creation(self):
        """Test StructurePoint object."""
        point = StructurePoint(time=10.5, structure_type='drop', confidence=0.85)

        assert point.time == 10.5
        assert point.structure_type == 'drop'
        assert point.confidence == 0.85

    def test_structure_point_to_dict(self):
        """Test StructurePoint serialization."""
        point = StructurePoint(time=10.5, structure_type='drop', confidence=0.85)
        d = point.to_dict()

        assert d['time'] == 10.5
        assert d['type'] == 'drop'
        assert d['confidence'] == 0.85

    def test_detect_structure_returns_list(self, sample_audio):
        """Test that detect_structure returns list of StructurePoint."""
        y, sr = sample_audio
        segmenter = StructureSegmenter()
        points = segmenter.detect_structure(y, sr)

        assert isinstance(points, list)
        for point in points:
            assert isinstance(point, StructurePoint)

    def test_structure_points_sorted_by_time(self, sample_audio):
        """Test that structure points are sorted by time."""
        y, sr = sample_audio
        points = detect_structure(y, sr)

        if len(points) > 1:
            for i in range(len(points) - 1):
                assert points[i].time <= points[i + 1].time

    def test_structure_confidence_valid(self, sample_audio):
        """Test that structure confidence scores are valid."""
        y, sr = sample_audio
        points = detect_structure(y, sr)

        for point in points:
            assert 0 <= point.confidence <= 1.0

    def test_structure_type_valid(self, sample_audio):
        """Test that structure types are recognized."""
        y, sr = sample_audio
        points = detect_structure(y, sr)

        for point in points:
            assert point.structure_type in StructureSegmenter.STRUCTURE_TYPES


# Tests for StemSeparator
class TestStemSeparator:
    """Tests for stem separation."""

    def test_stem_separator_initialization(self):
        """Test stem separator initialization."""
        separator = StemSeparator()
        assert separator is not None
        assert separator.model == StemSeparator.DEFAULT_MODEL

    def test_get_track_hash_consistency(self):
        """Test that track hash is consistent."""
        separator = StemSeparator()
        hash1 = separator._get_track_hash("test_audio.wav")
        hash2 = separator._get_track_hash("test_audio.wav")

        assert hash1 == hash2

    def test_get_track_hash_different(self):
        """Test that different paths produce different hashes."""
        separator = StemSeparator()
        hash1 = separator._get_track_hash("audio1.wav")
        hash2 = separator._get_track_hash("audio2.wav")

        assert hash1 != hash2

    def test_cache_path_generation(self):
        """Test cache path generation."""
        separator = StemSeparator()
        cache_path = separator._get_cache_path("test.wav")

        assert isinstance(cache_path, Path)
        assert 'stems' in str(cache_path)

    def test_normalize_stem_loudness_valid_output(self):
        """Test stem loudness normalization."""
        separator = StemSeparator()

        # Create test stems
        stems = {
            'drums': np.random.randn(2, 44100) * 0.5,
            'bass': np.random.randn(2, 44100) * 0.1,
        }

        normalized = separator._normalize_stem_loudness(stems)

        assert len(normalized) == len(stems)
        for stem_name, audio in normalized.items():
            assert audio.shape == stems[stem_name].shape
            # Check that it's been normalized (not zero)
            assert np.max(np.abs(audio)) > 0


# Tests for AIProcessor
class TestAIProcessor:
    """Tests for AI orchestration."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = AIProcessor()
        assert processor is not None
        assert processor.stem_separator is not None
        assert processor.mood_classifier is not None
        assert processor.segmenter is not None

    def test_load_audio(self, temp_audio_file):
        """Test audio loading."""
        processor = AIProcessor()
        y, sr = processor._load_audio(temp_audio_file)

        assert isinstance(y, np.ndarray)
        assert sr > 0
        assert len(y) > 0

    def test_analyze_stem_valid_output(self, sample_audio):
        """Test stem analysis."""
        processor = AIProcessor()
        y, sr = sample_audio

        analysis = processor._analyze_stem(y, sr, 'drums')

        assert isinstance(analysis, dict)
        assert 'stem_name' in analysis
        assert 'duration' in analysis
        assert 'spectral_centroid_mean' in analysis
        assert 'rms_mean' in analysis
        assert analysis['stem_name'] == 'drums'

    def test_process_with_stems_structure(self, temp_audio_file):
        """Test that process_with_stems returns proper structure."""
        processor = AIProcessor()
        result = processor.process_with_stems(temp_audio_file)

        assert isinstance(result, dict)
        assert 'audio_path' in result
        assert 'full_audio' in result
        assert 'stems_data' in result
        assert 'mood_classification' in result
        assert 'structural_landmarks' in result
        assert 'enhanced_features' in result

    def test_process_with_stems_with_phase2_features(self, temp_audio_file):
        """Test processor with existing Phase 2 features."""
        processor = AIProcessor()
        phase2_features = {
            'bpm': 120.0,
            'key': 'A',
        }

        result = processor.process_with_stems(
            temp_audio_file,
            features_dict=phase2_features
        )

        enhanced = result['enhanced_features']
        assert 'bpm' in enhanced  # Original feature preserved
        assert 'key' in enhanced
        # Should have new Phase 3 features too
        assert any(key in enhanced for key in ['stems', 'mood', 'structure'])


# Integration Tests
class TestIntegration:
    """Integration tests for Phase 3."""

    def test_full_pipeline_with_sample_audio(self, temp_audio_file):
        """Test complete Phase 3 pipeline."""
        # This is a full end-to-end test
        result = process_with_stems(
            temp_audio_file,
            features_dict={'test': 'data'}
        )

        # Check all components executed
        assert result['audio_path'] == temp_audio_file
        assert result['full_audio']['duration'] > 0

        # Mood should be present
        if result['mood_classification']:
            assert 'moods' in result['mood_classification']
            assert 'energy' in result['mood_classification']
            assert 'danceability' in result['mood_classification']

        # Structure should be detected
        assert isinstance(result['structural_landmarks'], list)

    def test_processor_performance(self, temp_audio_file):
        """Test that processor completes in reasonable time."""
        import time

        processor = AIProcessor()

        start = time.time()
        result = processor.process_with_stems(temp_audio_file)
        elapsed = time.time() - start

        # Should complete within 2 minutes for a test audio
        assert elapsed < 120
        print(f"Processed audio in {elapsed:.2f}s")


class TestSetlistGenerator:
    """Test the 5-phase setlist generator (pure functions, no audio/DB)."""

    @staticmethod
    def _library(n=40):
        """Synthetic library spanning calm-hypnotic to aggressive-euphoric."""
        tracks = []
        for i in range(n):
            x = i / (n - 1)  # 0 = calmest, 1 = hardest
            tracks.append({
                'id': i + 1,
                'file_name': f'track_{i + 1:02d}.mp3',
                'file_path': None,
                'title': f'Track {i + 1}',
                'artist': 'Test',
                'duration': 360.0,
                'bpm': 122.0 + 6.0 * x,
                'camelot_key': f'{(i % 12) + 1}A',
                'rms_mean': 0.10 + 0.20 * x,
                'brightness': 1000.0 + 2000.0 * x,
                'dark': 0.5,
                'hypnotic': 1.0 - x,
                'euphoric': x,
                'aggressive': max(0.0, x - 0.3),
                'industrial': 0.3,
                'minimal': 1.0 - x,
            })
        return tracks

    def test_phase_quotas_sum_and_breakdown(self):
        from src.ai.setlist_generator import phase_quotas

        for n in (15, 25, 28, 30):
            quotas = phase_quotas(n)
            assert sum(quotas.values()) == n
            assert quotas['breakdown'] == (1 if n < 26 else 2)
            assert all(q >= 1 for q in quotas.values())
            # Peak is the biggest phase
            assert quotas['peak'] == max(quotas.values())

    def test_phase_quotas_too_small(self):
        from src.ai.setlist_generator import phase_quotas

        with pytest.raises(ValueError):
            phase_quotas(4)

    def test_camelot_score(self):
        from src.ai.setlist_generator import camelot_score

        assert camelot_score('7A', '7A') == 1.0
        assert camelot_score('7A', '7B') == 0.9   # relative
        assert camelot_score('6A', '7A') == 0.9   # adjacent same mode
        assert camelot_score('12A', '1A') == 0.9  # wheel wraps around
        assert camelot_score('7A', '12A') == 0.3  # clash-prone
        assert camelot_score(None, '7A') == 0.5
        assert camelot_score('garbage', '7A') == 0.5

    def test_build_setlist_structure(self):
        from src.ai.setlist_generator import PHASE_ORDER, build_setlist

        setlist = build_setlist(self._library(), n_tracks=28)
        order = setlist['order']

        assert len(order) == 28
        assert len({t['id'] for t in order}) == 28, 'no duplicates'

        # Phases appear in canonical order (each track tagged with its phase)
        phase_seq = [t['phase'] for t in order]
        seen = [p for i, p in enumerate(phase_seq) if i == 0 or p != phase_seq[i - 1]]
        assert seen == [p for p in PHASE_ORDER if p in seen]

        # The energy arc: peak tracks are hotter than warm-up tracks on average
        by_phase = {}
        for t in order:
            by_phase.setdefault(t['phase'], []).append(t['rms_mean'])
        assert (sum(by_phase['peak']) / len(by_phase['peak'])
                > sum(by_phase['warmup']) / len(by_phase['warmup']))

    def test_setlist_needs_enough_tracks(self):
        from src.ai.setlist_generator import build_setlist

        with pytest.raises(ValueError):
            build_setlist(self._library(10), n_tracks=28)

    def test_render_report_with_stub_mix_points(self):
        from src.ai.setlist_generator import build_setlist, render_report

        setlist = build_setlist(self._library(), n_tracks=25)
        stub = lambda t: {'mix_in': 0.0, 'bass_in': 30.0, 'full_on': 90.0,
                          'mix_out': 300.0, 'n_onsets': 10}
        report = render_report(setlist, mix_points_fn=stub)

        for header in ('WARMUP', 'BUILD', 'PEAK', 'BREAKDOWN', 'COMEBACK'):
            assert header in report
        # One mix sheet per consecutive pair
        assert report.count('Beatmatch') == 24
        assert 'Bass swap' in report
        assert 'Clean break' in report  # the breakdown transition


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
