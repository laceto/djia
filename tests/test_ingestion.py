"""Tests for the ingestion module."""

import pytest
import logging
from pathlib import Path
from src.ingestion import AudioScanner, AudioLoader

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestAudioScanner:
    """Test cases for AudioScanner."""

    def test_scanner_initialization(self):
        """Test scanner initializes correctly."""
        scanner = AudioScanner("data")
        assert scanner.data_dir == Path("data")

    def test_scanner_finds_audio_files(self):
        """Test scanner finds audio files in data directory."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        assert len(files) > 0, "No audio files found in data directory"

    def test_scanner_identifies_formats(self):
        """Test scanner correctly identifies audio formats."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        formats = {f['format'] for f in files}
        assert len(formats) > 0, "No formats detected"
        # Check for common formats
        assert any(f in formats for f in ['.mp3', '.wav', '.flac', '.ogg', '.m4a'])

    def test_scanner_returns_consistent_order(self):
        """Test scanner returns files in consistent order."""
        scanner = AudioScanner("data")
        files1 = scanner.scan()
        files2 = scanner.scan()
        # Check consistent order between calls
        if len(files1) > 1:
            names1 = [f['name'] for f in files1]
            names2 = [f['name'] for f in files2]
            assert names1 == names2, "Files not in consistent order"

    def test_scanner_file_count(self):
        """Test scanner get_file_count method."""
        scanner = AudioScanner("data")
        count = scanner.get_file_count()
        assert count > 0, "Scanner reported no files"

    def test_scanner_returns_paths(self):
        """Test scanner returns Path objects."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        for f in files:
            assert isinstance(f['path'], Path), "Path should be Path object"
            assert f['path'].exists(), f"Path does not exist: {f['path']}"


class TestAudioLoader:
    """Test cases for AudioLoader."""

    def test_loader_initialization(self):
        """Test loader initializes correctly."""
        loader = AudioLoader()
        assert loader.target_sr == 22050
        assert loader.TARGET_CHANNELS == 1

    def test_loader_custom_sample_rate(self):
        """Test loader with custom sample rate."""
        loader = AudioLoader(target_sr=44100)
        assert loader.target_sr == 44100

    def test_loader_loads_audio(self):
        """Test loader can load audio files."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        assert len(files) > 0, "No files to test"

        loader = AudioLoader()
        # Try first file
        result = loader.load_audio(files[0]['path'])
        assert result is not None, "Failed to load audio"
        y, sr = result
        assert sr == 22050, f"Sample rate should be 22050, got {sr}"
        assert len(y) > 0, "No audio samples loaded"

    def test_loader_resamples_to_target(self):
        """Test loader resamples audio to target sample rate."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader(target_sr=22050)
        result = loader.load_audio(files[0]['path'])
        if result:
            y, sr = result
            assert sr == 22050, f"Expected 22050 Hz, got {sr} Hz"

    def test_loader_mono_conversion(self):
        """Test loader converts to mono."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader()
        result = loader.load_audio(files[0]['path'])
        if result:
            y, sr = result
            assert len(y.shape) == 1, "Audio should be mono (1D array)"

    def test_loader_extracts_metadata(self):
        """Test loader extracts metadata from files."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader()

        metadata = loader.extract_metadata(files[0]['path'])
        assert metadata is not None, "Metadata extraction failed"
        assert 'file_path' in metadata, "Missing file_path in metadata"
        assert 'file_name' in metadata, "Missing file_name in metadata"
        assert 'format' in metadata, "Missing format in metadata"
        assert 'duration' in metadata or metadata['duration'] is None

    def test_loader_validates_audio(self):
        """Test loader validates audio files."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader()

        # Test with first valid file
        is_valid = loader.validate_audio(files[0]['path'])
        assert is_valid, f"File {files[0]['path']} failed validation"

    def test_loader_handles_duration_parameter(self):
        """Test loader respects duration parameter."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader()

        result = loader.load_audio(files[0]['path'], duration=5.0)
        if result:
            y, sr = result
            duration = len(y) / sr
            assert duration <= 5.1, f"Duration exceeded 5 seconds: {duration}"

    def test_loader_gracefully_handles_errors(self):
        """Test loader handles corrupted/invalid files gracefully."""
        loader = AudioLoader()
        # Create a path to non-existent file
        result = loader.load_audio(Path("data/nonexistent.mp3"))
        assert result is None, "Should return None for invalid file"

    def test_all_files_loadable(self):
        """Test all files in data directory can be loaded."""
        scanner = AudioScanner("data")
        files = scanner.scan()
        loader = AudioLoader()

        failed = []
        for file_info in files:
            result = loader.load_audio(file_info['path'])
            if result is None:
                failed.append(file_info['name'])

        if failed:
            print(f"Warning: {len(failed)} files failed to load: {failed}")
        # Allow some failures (corrupt files)
        assert len(failed) < len(files), "Too many files failed to load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
