"""End-to-end integration tests for DJIA full pipeline."""

import pytest
from pathlib import Path

from src.orchestrator import Orchestrator
from src.database.store import TrackStore
from src.ai import (
    score_transition,
    build_transition_graph,
    generate_playlist,
)


class TestFullPipeline:
    """End-to-end pipeline tests."""

    def test_complete_analysis_pipeline(self, temp_audio_dir, temp_db):
        """Test complete analysis pipeline from audio to database."""
        orchestrator = Orchestrator(temp_db)

        # Analyze library
        result = orchestrator.analyze_library(str(temp_audio_dir))

        # Verify results
        assert result['analyzed'] > 0
        assert result['errors'] == 0

        # Verify database has tracks
        store = TrackStore(temp_db)
        all_tracks = store.get_all_tracks()
        assert len(all_tracks) > 0

        # Verify features were stored
        for track in all_tracks:
            features = store.get_track_features(track['id'])
            assert features is not None
            assert features.get('bpm') is not None

    def test_single_track_analysis(self, temp_audio_dir, temp_db):
        """Test analyzing a single track."""
        orchestrator = Orchestrator(temp_db)

        # Get first audio file
        audio_files = list(Path(temp_audio_dir).glob("*.wav"))
        assert len(audio_files) > 0

        # Analyze single track
        result = orchestrator.analyze_single_track(str(audio_files[0]))

        assert result is not None
        assert result.get('duration') > 0
        assert result.get('tempo') is not None
        assert result.get('file_name') is not None

    def test_database_storage(self, temp_db, sample_features, sample_mood):
        """Test storing and retrieving data from database."""
        store = TrackStore(temp_db)

        # Insert track
        track_id = store.insert_track(
            file_path="/test/track.wav",
            file_name="track.wav",
            format=".wav",
            duration=300.0,
            artist="Test Artist",
            title="Test Track",
        )

        # Insert features
        store.insert_features(track_id, sample_features)

        # Insert mood
        store.insert_mood(track_id, sample_mood)

        # Retrieve and verify
        track = store.get_track(track_id)
        assert track is not None
        assert track['file_name'] == "track.wav"

        features = store.get_track_features(track_id)
        assert features is not None
        assert abs(features.get('bpm') - 128.5) < 0.01

        mood = store.get_track_mood(track_id)
        assert mood is not None
        assert abs(mood['dark'] - 0.2) < 0.01

    def test_transition_scoring(self, sample_tracks_data):
        """Test transition scoring between tracks."""
        track_1 = sample_tracks_data[1]
        track_2 = sample_tracks_data[2]

        # Score transition
        score = score_transition(track_1, track_2)

        # Verify score components
        assert 0.0 <= score.bpm_score <= 1.0
        assert 0.0 <= score.key_score <= 1.0
        assert 0.0 <= score.mood_score <= 1.0
        assert 0.0 <= score.energy_score <= 1.0
        assert 0.0 <= score.overall_score <= 1.0

    def test_transition_graph(self, sample_tracks_data):
        """Test building transition graph."""
        graph = build_transition_graph(sample_tracks_data)

        # Verify graph structure
        assert len(graph) == len(sample_tracks_data)

        for track_id, edges in graph.items():
            # Should have edges to all other tracks
            assert len(edges) == len(sample_tracks_data) - 1

            # Edges should be sorted by score
            scores = [score for _, score in edges]
            assert scores == sorted(scores, reverse=True)

    def test_playlist_generation(self, sample_tracks_data):
        """Test generating a playlist."""
        start_id = 1
        end_id = 3
        num_steps = 3

        playlist = generate_playlist(
            sample_tracks_data,
            start_id,
            end_id,
            num_steps=num_steps,
        )

        # Verify playlist
        assert playlist is not None
        assert len(playlist) <= num_steps
        assert playlist[0] == start_id
        assert playlist[-1] == end_id

    def test_orchestrator_get_tracks_dict(self, temp_audio_dir, temp_db):
        """Test orchestrator retrieving all tracks as dict."""
        orchestrator = Orchestrator(temp_db)

        # Analyze library
        orchestrator.analyze_library(str(temp_audio_dir))

        # Get tracks dict
        tracks_dict = orchestrator.get_all_tracks_dict()

        # Verify structure
        assert len(tracks_dict) > 0

        for track_id, track_data in tracks_dict.items():
            assert isinstance(track_id, int)
            assert 'file_name' in track_data
            assert 'duration' in track_data
            assert 'tempo' in track_data

    def test_track_count(self, temp_audio_dir, temp_db):
        """Test getting track count."""
        orchestrator = Orchestrator(temp_db)

        # Initially empty
        assert orchestrator.get_track_count() == 0

        # Analyze library
        orchestrator.analyze_library(str(temp_audio_dir))

        # Should have tracks
        count = orchestrator.get_track_count()
        assert count > 0


class TestTransitionScoring:
    """Tests for transition scoring logic."""

    def test_identical_bpm_scores_high(self):
        """Identical BPM should score high."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}
        track_b = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}

        score = score_transition(track_a, track_b)
        assert score.bpm_score > 0.9
        assert score.overall_score > 0.85

    def test_similar_bpm_scores_well(self):
        """Similar BPM should score well."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}
        track_b = {'tempo': 130.0, 'key': 'C', 'rms_mean': 0.1}

        score = score_transition(track_a, track_b)
        assert score.bpm_score > 0.7

    def test_different_bpm_scores_lower(self):
        """Very different BPM should score lower."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}
        track_b = {'tempo': 180.0, 'key': 'C', 'rms_mean': 0.1}

        score = score_transition(track_a, track_b)
        assert score.bpm_score < 0.5

    def test_same_key_scores_high(self):
        """Same key should score high."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}
        track_b = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.1}

        score = score_transition(track_a, track_b)
        assert score.key_score == 1.0

    def test_missing_key_defaults_neutral(self):
        """Missing key should default to neutral score."""
        track_a = {'tempo': 128.0, 'rms_mean': 0.1}
        track_b = {'tempo': 128.0, 'rms_mean': 0.1}

        score = score_transition(track_a, track_b)
        assert 0.4 <= score.key_score <= 0.6

    def test_smooth_energy_arc(self):
        """Similar RMS should have smooth energy arc."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.10}
        track_b = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.11}

        score = score_transition(track_a, track_b)
        assert score.energy_score > 0.8

    def test_jarring_energy_arc(self):
        """Very different RMS should have jarring energy arc."""
        track_a = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.05}
        track_b = {'tempo': 128.0, 'key': 'C', 'rms_mean': 0.25}

        score = score_transition(track_a, track_b)
        assert score.energy_score < 0.5


class TestPlaylistGeneration:
    """Tests for playlist generation."""

    def test_playlist_respects_track_count(self, sample_tracks_data):
        """Generated playlist should respect requested step count."""
        playlist = generate_playlist(
            sample_tracks_data, 1, 3, num_steps=3
        )

        assert len(playlist) == 3

    def test_playlist_starts_and_ends_correctly(self, sample_tracks_data):
        """Playlist should start and end with specified tracks."""
        start = 1
        end = 3
        playlist = generate_playlist(sample_tracks_data, start, end, num_steps=3)

        assert playlist[0] == start
        assert playlist[-1] == end

    def test_invalid_track_ids_return_none(self, sample_tracks_data):
        """Invalid track IDs should return None."""
        playlist = generate_playlist(
            sample_tracks_data, 999, 888, num_steps=3
        )

        assert playlist is None

    def test_no_self_loops_in_playlist(self, sample_tracks_data):
        """Playlist should not have duplicate tracks."""
        playlist = generate_playlist(
            sample_tracks_data, 1, 3, num_steps=3
        )

        # Check for duplicates
        assert len(playlist) == len(set(playlist))
