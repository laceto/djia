"""Pytest configuration and fixtures for DJIA tests."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
import numpy as np
import soundfile as sf

from src.database.schema import init_db
from src.ingestion.scanner import AudioScanner
from src.ingestion.loader import AudioLoader


@pytest.fixture
def temp_audio_dir():
    """Create a temporary directory with test audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create two test WAV files with different characteristics
        sr = 22050

        # Track 1: 128 BPM, simple kick pattern
        duration = 10  # 10 seconds
        t = np.arange(int(sr * duration)) / sr
        # 128 BPM = 2.13 Hz
        freq = 128 / 60  # BPM to Hz
        audio1 = 0.3 * np.sin(2 * np.pi * freq * t)  # Kick drum
        audio1 += 0.1 * np.random.normal(size=len(audio1))  # Noise
        sf.write(tmpdir_path / "test_track_1.wav", audio1, sr)

        # Track 2: 132 BPM, different tone
        freq2 = 132 / 60
        audio2 = 0.2 * np.sin(2 * np.pi * freq2 * t)
        audio2 += 0.05 * np.sin(2 * np.pi * freq2 * 2 * t)  # Harmonic
        audio2 += 0.1 * np.random.normal(size=len(audio2))
        sf.write(tmpdir_path / "test_track_2.wav", audio2, sr)

        # Track 3: Different BPM
        freq3 = 125 / 60
        audio3 = 0.15 * np.sin(2 * np.pi * freq3 * t)
        audio3 += 0.2 * np.sin(2 * np.pi * freq3 * 0.5 * t)  # Sub frequency
        sf.write(tmpdir_path / "test_track_3.wav", audio3, sr)

        yield tmpdir_path


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    # Initialize database
    init_db(db_path)

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def audio_scanner(temp_audio_dir):
    """Create an AudioScanner for temporary directory."""
    return AudioScanner(str(temp_audio_dir))


@pytest.fixture
def audio_loader():
    """Create an AudioLoader instance."""
    return AudioLoader()


@pytest.fixture
def sample_tracks_data():
    """Create sample track data for testing."""
    return {
        1: {
            'id': 1,
            'file_name': 'track_1.wav',
            'duration': 300,
            'tempo': 128.0,
            'key': 'D',
            'rms_mean': 0.1,
            'spectral_centroid_mean': 3500,
            'harmonic_ratio': 1.2,
            'mood': {
                'dark': 0.2,
                'hypnotic': 0.4,
                'euphoric': 0.2,
                'aggressive': 0.1,
                'industrial': 0.05,
                'minimal': 0.05,
            }
        },
        2: {
            'id': 2,
            'file_name': 'track_2.wav',
            'duration': 320,
            'tempo': 130.0,
            'key': 'A',
            'rms_mean': 0.12,
            'spectral_centroid_mean': 3800,
            'harmonic_ratio': 1.4,
            'mood': {
                'dark': 0.15,
                'hypnotic': 0.3,
                'euphoric': 0.35,
                'aggressive': 0.15,
                'industrial': 0.04,
                'minimal': 0.01,
            }
        },
        3: {
            'id': 3,
            'file_name': 'track_3.wav',
            'duration': 280,
            'tempo': 125.0,
            'key': 'E',
            'rms_mean': 0.08,
            'spectral_centroid_mean': 3200,
            'harmonic_ratio': 1.0,
            'mood': {
                'dark': 0.3,
                'hypnotic': 0.5,
                'euphoric': 0.1,
                'aggressive': 0.08,
                'industrial': 0.02,
                'minimal': 0.0,
            }
        },
    }


@pytest.fixture
def sample_features():
    """Create sample audio features for testing."""
    return {
        'tempo': 128.5,
        'spectral_centroid_mean': 3400,
        'spectral_centroid_std': 800,
        'spectral_rolloff_mean': 12000,
        'spectral_flux_mean': 0.02,
        'harmonic_ratio': 1.3,
        'percussive_ratio': 0.6,
        'mfcc_mean': 50.0,
        'mfcc_std': 12.0,
        'mfcc_delta_mean': 0.8,
        'chroma_variance': 0.4,
        'chroma_entropy': 3.2,
        'rms_mean': 0.11,
        'rms_std': 0.02,
        'rms_peak': 0.5,
    }


@pytest.fixture
def sample_mood():
    """Create sample mood classification."""
    return {
        'dark': 0.2,
        'hypnotic': 0.4,
        'euphoric': 0.2,
        'aggressive': 0.1,
        'industrial': 0.05,
        'minimal': 0.05,
    }
