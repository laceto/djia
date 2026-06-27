"""Tests for Traktor NML export and similarity engine."""

import pytest
import xml.etree.ElementTree as ET
import tempfile
import sqlite3
from pathlib import Path
import numpy as np

from src.traktor.exporter import (
    parse_traktor_nml,
    add_track_analysis,
    export_nml,
    export_all_tracks,
    _ms_to_traktor_offset,
)
from src.matching.similarity import (
    normalize_features,
    compute_similarity,
    find_similar_tracks,
)
from src.database.schema import init_db


class TestTraktorParsing:
    """Tests for NML parsing."""

    @pytest.fixture
    def sample_nml(self, tmp_path):
        """Create a sample Traktor NML file for testing."""
        nml_content = """<?xml version="1.0" encoding="UTF-8"?>
<COLLECTION ENTRIES="2">
    <ENTRY AUDIO_ID="123456" TITLE="Test Track 1" ARTIST="Artist A">
        <TITLE>Test Track 1</TITLE>
        <ARTIST>Artist A</ARTIST>
        <ALBUM>Album X</ALBUM>
        <TEMPO BPM="128" />
        <INFO KEY="8A" />
    </ENTRY>
    <ENTRY AUDIO_ID="654321" TITLE="Test Track 2" ARTIST="Artist B">
        <TITLE>Test Track 2</TITLE>
        <ARTIST>Artist B</ARTIST>
        <ALBUM>Album Y</ALBUM>
        <TEMPO BPM="130" />
        <INFO KEY="9A" />
    </ENTRY>
</COLLECTION>
"""
        nml_file = tmp_path / "Collection.nml"
        nml_file.write_text(nml_content)
        return str(nml_file)

    def test_parse_valid_nml(self, sample_nml):
        """Test parsing valid NML file."""
        root = parse_traktor_nml(sample_nml)
        assert root.tag == "COLLECTION"
        assert root.get("ENTRIES") == "2"

    def test_parse_nonexistent_nml(self):
        """Test parsing nonexistent NML raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_traktor_nml("/nonexistent/Collection.nml")

    def test_parse_malformed_nml(self, tmp_path):
        """Test parsing malformed NML raises ParseError."""
        bad_nml = tmp_path / "bad.nml"
        bad_nml.write_text("<COLLECTION>incomplete")

        with pytest.raises(ET.ParseError):
            parse_traktor_nml(str(bad_nml))


class TestTraktorExport:
    """Tests for NML export and modification."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create test database with sample data."""
        db_path = tmp_path / "test.db"
        conn = init_db(str(db_path))

        # Insert sample track
        conn.execute(
            """
            INSERT INTO tracks (file_path, file_name, format, duration, title, artist)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("path/to/track1.wav", "track1.wav", "wav", 300.0, "Test Track 1", "Artist A")
        )

        # Insert features
        conn.execute(
            """
            INSERT INTO features
            (track_id, bpm, spectral_centroid_mean, spectral_centroid_std, spectral_rolloff_mean,
             spectral_flux_mean, harmonic_ratio, percussive_ratio, mfcc_mean, mfcc_std,
             mfcc_delta_mean, chroma_variance, chroma_entropy, rms_mean, rms_std, rms_peak)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, 128.5, 2500.0, 100.0, 8000.0, 0.05, 1.5, 0.4, 50.0, 5.0, 1.0, 0.5, 2.5, 0.1, 0.02, 0.3)
        )

        conn.commit()
        conn.close()

        return str(db_path)

    @pytest.fixture
    def sample_nml_for_export(self, tmp_path):
        """Create sample NML for export testing."""
        nml_content = """<?xml version="1.0" encoding="UTF-8"?>
<COLLECTION ENTRIES="1">
    <ENTRY AUDIO_ID="123" TITLE="Track 1" ARTIST="Artist A">
        <TITLE>Test Track 1</TITLE>
        <ARTIST>Artist A</ARTIST>
        <TEMPO BPM="128" />
    </ENTRY>
</COLLECTION>
"""
        nml_file = tmp_path / "Collection.nml"
        nml_file.write_text(nml_content)
        return str(nml_file)

    def test_add_track_analysis(self, sample_nml_for_export, test_db):
        """Test adding analysis to track entry."""
        root = parse_traktor_nml(sample_nml_for_export)

        analysis = {
            'bpm': 130.0,
            'brightness': 75,
            'danceability': 85,
            'cue_points': [
                {'time': 45.2, 'type': 'drop'},
                {'time': 120.5, 'type': 'breakdown'},
            ]
        }

        result = add_track_analysis(root, 1, analysis, test_db)
        assert result is not None

        # Check BPM was updated
        tempo = result.find('TEMPO')
        assert tempo is not None
        assert tempo.get('BPM') == '130.0'

        # Check hot cues were added
        cues = result.findall('CUE_V2')
        assert len(cues) >= 2

    def test_export_nml_creates_valid_file(self, sample_nml_for_export, tmp_path):
        """Test NML export creates valid XML file."""
        root = parse_traktor_nml(sample_nml_for_export)
        output = tmp_path / "output.nml"

        success = export_nml(root, str(output))
        assert success
        assert output.exists()

        # Verify exported NML is parseable
        exported_root = parse_traktor_nml(str(output))
        assert exported_root.tag == "COLLECTION"

    def test_export_all_tracks(self, sample_nml_for_export, test_db, tmp_path):
        """Test batch export of all tracks."""
        output = tmp_path / "collection_analyzed.nml"

        result = export_all_tracks(
            sample_nml_for_export,
            test_db,
            str(output)
        )

        assert result == str(output)
        assert Path(result).exists()


class TestSimilarityEngine:
    """Tests for similarity computation."""

    def test_normalize_features_basic(self):
        """Test feature normalization."""
        features = {
            'bpm': 128.0,
            'spectral_centroid_mean': 2500.0,
            'spectral_centroid_std': 100.0,
            'spectral_rolloff_mean': 8000.0,
            'spectral_flux_mean': 0.05,
            'harmonic_ratio': 1.5,
            'percussive_ratio': 0.4,
            'mfcc_mean': 50.0,
            'mfcc_std': 5.0,
            'mfcc_delta_mean': 1.0,
            'chroma_variance': 0.5,
            'chroma_entropy': 2.5,
            'rms_mean': 0.1,
            'rms_std': 0.02,
            'rms_peak': 0.3,
        }

        normalized = normalize_features(features)

        # Should be 1D array
        assert normalized.ndim == 1
        # Should have 15 features
        assert len(normalized) == 15
        # Should be normalized (roughly zero mean)
        assert abs(np.mean(normalized)) < 0.1

    def test_normalize_features_with_missing_values(self):
        """Test normalization handles missing values."""
        features = {
            'bpm': 128.0,
            'spectral_centroid_mean': 2500.0,
            # Other features missing
        }

        normalized = normalize_features(features)
        assert normalized.ndim == 1
        assert len(normalized) == 15

    def test_compute_similarity_identical(self):
        """Test similarity of identical vectors."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        similarity = compute_similarity(v1, v2)
        # Should be close to 1.0
        assert 0.99 < similarity <= 1.0

    def test_compute_similarity_orthogonal(self):
        """Test similarity of orthogonal vectors."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        similarity = compute_similarity(v1, v2)
        # Should be close to 0
        assert -0.1 < similarity < 0.1

    def test_compute_similarity_bounds(self):
        """Test similarity is in [0, 1] range."""
        v1 = np.random.randn(10)
        v2 = np.random.randn(10)

        similarity = compute_similarity(v1, v2)
        assert 0.0 <= similarity <= 1.0


class TestSimilarityMatching:
    """Tests for similar track finding."""

    @pytest.fixture
    def similarity_test_db(self, tmp_path):
        """Create test database with multiple tracks for similarity testing."""
        db_path = tmp_path / "similarity_test.db"
        conn = init_db(str(db_path))

        # Insert 5 test tracks
        tracks = [
            ("track1.wav", "track1.wav", "Track 1", "Artist A", 120.0),
            ("track2.wav", "track2.wav", "Track 2", "Artist B", 122.0),
            ("track3.wav", "track3.wav", "Track 3", "Artist C", 130.0),
            ("track4.wav", "track4.wav", "Track 4", "Artist D", 121.0),
            ("track5.wav", "track5.wav", "Track 5", "Artist E", 115.0),
        ]

        for file_path, file_name, title, artist, bpm in tracks:
            conn.execute(
                """
                INSERT INTO tracks (file_path, file_name, format, duration, title, artist)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_path, file_name, "wav", 300.0, title, artist)
            )

            # Insert similar features (same BPM group)
            conn.execute(
                """
                INSERT INTO features
                (track_id, bpm, spectral_centroid_mean, spectral_centroid_std,
                 spectral_rolloff_mean, spectral_flux_mean, harmonic_ratio,
                 percussive_ratio, mfcc_mean, mfcc_std, mfcc_delta_mean,
                 chroma_variance, chroma_entropy, rms_mean, rms_std, rms_peak)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conn.execute("SELECT last_insert_rowid()").fetchone()[0],
                    bpm, 2500.0 + np.random.randn() * 100,
                    100.0, 8000.0, 0.05, 1.5, 0.4, 50.0, 5.0, 1.0,
                    0.5, 2.5, 0.1, 0.02, 0.3
                )
            )

        conn.commit()
        conn.close()

        return str(db_path)

    def test_find_similar_tracks_basic(self, similarity_test_db):
        """Test finding similar tracks."""
        matches = find_similar_tracks(1, top_k=3, db_path=similarity_test_db)

        # Should return up to 3 results
        assert len(matches) <= 3
        # Results should be (track_dict, score) tuples
        assert all(isinstance(m, tuple) and len(m) == 2 for m in matches)
        # Scores should be in [0, 1]
        assert all(0.0 <= m[1] <= 1.0 for m in matches)

    def test_find_similar_tracks_sorted(self, similarity_test_db):
        """Test similar tracks are sorted by score descending."""
        matches = find_similar_tracks(1, top_k=5, db_path=similarity_test_db)

        scores = [m[1] for m in matches]
        # Should be sorted descending
        assert scores == sorted(scores, reverse=True)

    def test_find_similar_tracks_bpm_filter(self, similarity_test_db):
        """Test BPM filter works."""
        # Track 1 has BPM ~120, filter ±2 should exclude 130 (track 3)
        matches = find_similar_tracks(
            1,
            top_k=5,
            bpm_tolerance=2.0,
            db_path=similarity_test_db
        )

        for track, score in matches:
            bpm = track.get('bpm', 120.0)
            # All matches should be within ±2 BPM
            assert abs(bpm - 120.0) <= 2.0

    def test_find_similar_tracks_nonexistent(self, similarity_test_db):
        """Test finding similar to nonexistent track raises error."""
        with pytest.raises(ValueError):
            find_similar_tracks(9999, db_path=similarity_test_db)


class TestUtilities:
    """Tests for utility functions."""

    def test_ms_to_traktor_offset(self):
        """Test millisecond to Traktor offset conversion."""
        # 1000 ms = 1 second = 48000 samples at 48kHz
        offset = _ms_to_traktor_offset(1000)
        assert offset == 48000

        # 500 ms = 24000 samples
        offset = _ms_to_traktor_offset(500)
        assert offset == 24000

    def test_ms_to_traktor_offset_zero(self):
        """Test zero milliseconds."""
        offset = _ms_to_traktor_offset(0)
        assert offset == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
