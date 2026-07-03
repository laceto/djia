"""Tests for the database module."""

import pytest
import sqlite3
import tempfile
import logging
import os
from pathlib import Path
from src.database import init_db, get_connection, TrackStore

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    import gc
    # Create a temporary file in a location we can control
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)  # Close the file descriptor

    try:
        yield db_path
    finally:
        # Force garbage collection to close any open connections
        gc.collect()
        # Give Windows time to release the lock
        import time
        time.sleep(0.1)
        # Clean up the database file
        try:
            if Path(db_path).exists():
                Path(db_path).unlink()
        except Exception as e:
            logging.warning(f"Could not delete temp db {db_path}: {e}")


class TestDatabaseSchema:
    """Test cases for database schema."""

    def test_init_db_creates_file(self, temp_db):
        """Test that init_db creates database file."""
        conn = init_db(temp_db)
        conn.close()
        assert Path(temp_db).exists(), "Database file not created"

    def test_init_db_creates_tables(self, temp_db):
        """Test that all tables are created."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table'
        """)
        tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {'tracks', 'features', 'mood', 'segments'}
        assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"
        conn.close()

    def test_tracks_table_structure(self, temp_db):
        """Test tracks table has correct columns."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(tracks)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'file_path', 'file_name', 'format', 'artist',
            'title', 'album', 'duration', 'analysis_date', 'created_at', 'updated_at'
        }
        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"
        conn.close()

    def test_features_table_structure(self, temp_db):
        """Test features table has correct columns."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(features)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'track_id', 'bpm', 'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_flux_mean', 'harmonic_ratio',
            'percussive_ratio', 'mfcc_mean', 'mfcc_std', 'mfcc_delta_mean',
            'chroma_variance', 'chroma_entropy', 'rms_mean', 'rms_std', 'rms_peak',
            'mfcc_vector', 'created_at', 'updated_at'
        }
        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"
        conn.close()

    def test_mood_table_structure(self, temp_db):
        """Test mood table has correct columns."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(mood)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'track_id', 'dark', 'hypnotic', 'euphoric',
            'aggressive', 'industrial', 'minimal', 'created_at', 'updated_at'
        }
        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"
        conn.close()

    def test_segments_table_structure(self, temp_db):
        """Test segments table has correct columns."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(segments)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            'id', 'track_id', 'segment_type', 'start_time', 'end_time', 'confidence', 'created_at'
        }
        assert expected_columns.issubset(columns), f"Missing columns: {expected_columns - columns}"
        conn.close()

    def test_foreign_key_constraints_enabled(self, temp_db):
        """Test foreign key constraints are enabled."""
        conn = init_db(temp_db)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys")
        result = cursor.fetchone()
        assert result[0] == 1, "Foreign keys not enabled"
        conn.close()


class TestTrackStore:
    """Test cases for TrackStore operations."""

    def test_store_initialization(self, temp_db):
        """Test store initializes correctly."""
        store = TrackStore(temp_db)
        assert Path(temp_db).exists(), "Database not created"

    def test_insert_track(self, temp_db):
        """Test inserting a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
            artist="Artist Name",
            title="Track Title",
            album="Album Name",
        )
        assert isinstance(track_id, int), "Track ID should be integer"
        assert track_id > 0, "Track ID should be positive"

    def test_get_track(self, temp_db):
        """Test retrieving a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
            artist="Artist Name",
            title="Track Title",
        )

        track = store.get_track(track_id)
        assert track is not None, "Track not found"
        assert track['artist'] == "Artist Name"
        assert track['title'] == "Track Title"
        assert track['duration'] == 120.5

    def test_get_track_id(self, temp_db):
        """Test getting track ID by file path."""
        store = TrackStore(temp_db)
        file_path = "/path/to/track.mp3"
        track_id = store.insert_track(
            file_path=file_path,
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        retrieved_id = store.get_track_id(file_path)
        assert retrieved_id == track_id, "Track ID mismatch"

    def test_get_all_tracks(self, temp_db):
        """Test retrieving all tracks."""
        store = TrackStore(temp_db)
        store.insert_track("/path/1.mp3", "1.mp3", ".mp3", 100.0)
        store.insert_track("/path/2.mp3", "2.mp3", ".mp3", 120.0)
        store.insert_track("/path/3.mp3", "3.mp3", ".mp3", 140.0)

        tracks = store.get_all_tracks()
        assert len(tracks) == 3, "Should retrieve 3 tracks"

    def test_insert_features(self, temp_db):
        """Test inserting features for a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        features = {
            'tempo': 125.0,
            'spectral_centroid_mean': 2500.0,
            'harmonic_ratio': 1.5,
            'rms_peak': 0.8,
        }
        feature_id = store.insert_features(track_id, features)
        assert isinstance(feature_id, int), "Feature ID should be integer"

    def test_get_track_features(self, temp_db):
        """Test retrieving features for a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        features = {
            'tempo': 125.0,
            'spectral_centroid_mean': 2500.0,
            'harmonic_ratio': 1.5,
            'swing_score': 0.35,
        }
        store.insert_features(track_id, features)

        retrieved_features = store.get_track_features(track_id)
        assert retrieved_features is not None, "Features not found"
        assert retrieved_features['bpm'] == 125.0
        assert retrieved_features['spectral_centroid_mean'] == 2500.0
        assert retrieved_features['swing_score'] == 0.35

    def test_insert_mood(self, temp_db):
        """Test inserting mood classification."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        mood_scores = {
            'dark': 0.8,
            'hypnotic': 0.6,
            'euphoric': 0.3,
        }
        mood_id = store.insert_mood(track_id, mood_scores)
        assert isinstance(mood_id, int), "Mood ID should be integer"

    def test_get_track_mood(self, temp_db):
        """Test retrieving mood for a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        mood_scores = {'dark': 0.8, 'hypnotic': 0.6}
        store.insert_mood(track_id, mood_scores)

        mood = store.get_track_mood(track_id)
        assert mood is not None, "Mood not found"
        assert mood['dark'] == 0.8
        assert mood['hypnotic'] == 0.6

    def test_insert_segment(self, temp_db):
        """Test inserting a segment."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        segment_id = store.insert_segment(
            track_id=track_id,
            segment_type="intro",
            start_time=0.0,
            end_time=10.0,
            confidence=0.95,
        )
        assert isinstance(segment_id, int), "Segment ID should be integer"

    def test_get_track_segments(self, temp_db):
        """Test retrieving segments for a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        store.insert_segment(track_id, "intro", 0.0, 10.0, 0.95)
        store.insert_segment(track_id, "build", 10.0, 40.0, 0.90)
        store.insert_segment(track_id, "drop", 40.0, 100.0, 0.98)

        segments = store.get_track_segments(track_id)
        assert len(segments) == 3, "Should retrieve 3 segments"
        assert segments[0]['segment_type'] == 'intro'
        assert segments[1]['segment_type'] == 'build'
        assert segments[2]['segment_type'] == 'drop'

    def test_replace_segments_per_method(self, temp_db):
        """Test that replace_segments only replaces its own method's rows."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        spectral = [
            {'segment_type': 'intro', 'start_time': 0.0, 'end_time': 20.0, 'confidence': 0.8},
            {'segment_type': 'outro', 'start_time': 100.0, 'end_time': 120.5, 'confidence': 0.8},
        ]
        grid = [
            {'segment_type': 'intro', 'start_time': 0.0, 'end_time': 30.0, 'confidence': 0.95},
            {'segment_type': 'build', 'start_time': 30.0, 'end_time': 60.0, 'confidence': 0.95},
            {'segment_type': 'drop', 'start_time': 60.0, 'end_time': 90.0, 'confidence': 0.95},
            {'segment_type': 'outro', 'start_time': 90.0, 'end_time': 120.5, 'confidence': 0.95},
        ]
        store.replace_segments(track_id, spectral, method="spectral")
        store.replace_segments(track_id, grid, method="phrase16")

        all_segments = store.get_track_segments(track_id)
        assert len(all_segments) == 6, "Both methods should be stored"

        # Re-running one method replaces only that method's rows
        store.replace_segments(track_id, spectral[:1], method="spectral")
        all_segments = store.get_track_segments(track_id)
        assert len(all_segments) == 5, "Spectral rows replaced, phrase16 untouched"
        methods = {s['method'] for s in all_segments}
        assert methods == {'spectral', 'phrase16'}

    def test_get_tracks_count(self, temp_db):
        """Test getting total track count."""
        store = TrackStore(temp_db)
        store.insert_track("/path/1.mp3", "1.mp3", ".mp3", 100.0)
        store.insert_track("/path/2.mp3", "2.mp3", ".mp3", 120.0)

        count = store.get_tracks_count()
        assert count == 2, "Track count should be 2"

    def test_delete_track(self, temp_db):
        """Test deleting a track."""
        store = TrackStore(temp_db)
        track_id = store.insert_track(
            file_path="/path/to/track.mp3",
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )

        success = store.delete_track(track_id)
        assert success, "Delete should succeed"

        track = store.get_track(track_id)
        assert track is None, "Track should be deleted"

    def test_search_tracks(self, temp_db):
        """Test searching tracks."""
        store = TrackStore(temp_db)
        store.insert_track(
            "/path/1.mp3", "1.mp3", ".mp3", 100.0,
            artist="Artist A", title="Song A"
        )
        store.insert_track(
            "/path/2.mp3", "2.mp3", ".mp3", 120.0,
            artist="Artist B", title="Song B"
        )
        store.insert_track(
            "/path/3.mp3", "3.mp3", ".mp3", 140.0,
            artist="Artist A", title="Song C"
        )

        results = store.search_tracks("Artist A")
        assert len(results) == 2, "Should find 2 tracks by Artist A"

        results = store.search_tracks("Song B")
        assert len(results) == 1, "Should find 1 track with Song B"

    def test_duplicate_track_handling(self, temp_db):
        """Test handling of duplicate file paths."""
        store = TrackStore(temp_db)
        file_path = "/path/to/track.mp3"

        track_id_1 = store.insert_track(
            file_path=file_path,
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )
        # Insert same file path again
        track_id_2 = store.insert_track(
            file_path=file_path,
            file_name="track.mp3",
            format=".mp3",
            duration=120.5,
        )
        # Should return existing ID, not create duplicate
        assert track_id_1 == track_id_2, "Should return existing track ID"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
