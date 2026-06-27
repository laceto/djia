"""Audio loader with metadata extraction and resampling."""

import logging
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.mp4 import MP4
from mutagen import MutagenError

logger = logging.getLogger(__name__)


class AudioLoader:
    """Loads audio files with metadata extraction and resampling."""

    TARGET_SR = 22050  # Standard sample rate for analysis
    TARGET_CHANNELS = 1  # Mono for consistency

    def __init__(self, target_sr: int = TARGET_SR):
        """Initialize loader with target sample rate."""
        self.target_sr = target_sr

    def load_audio(
        self,
        file_path: Path,
        duration: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load audio file and resample to target sample rate.

        Args:
            file_path: Path to audio file
            duration: Duration in seconds to load (None for full track)

        Returns:
            Dict with audio_array, sample_rate, duration, channels or None if error
        """
        try:
            y, sr = librosa.load(
                str(file_path),
                sr=self.target_sr,
                mono=True,
                duration=duration
            )
            audio_duration = len(y) / sr
            logger.debug(f"Loaded {file_path.name}: {sr}Hz, {len(y)} samples")
            return {
                'audio_array': y,
                'sample_rate': sr,
                'duration': audio_duration,
                'channels': 1,
                'num_samples': len(y)
            }
        except Exception as e:
            logger.error(f"Error loading audio from {file_path}: {e}")
            return None

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with metadata (artist, title, album, duration, etc.)
        """
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'format': file_path.suffix.lower(),
            'artist': None,
            'title': None,
            'album': None,
            'duration': None,
        }

        suffix = file_path.suffix.lower()

        try:
            if suffix == '.mp3':
                metadata.update(self._extract_id3(file_path))
            elif suffix == '.flac':
                metadata.update(self._extract_flac(file_path))
            elif suffix == '.ogg':
                metadata.update(self._extract_ogg(file_path))
            elif suffix in ['.m4a', '.aac']:
                metadata.update(self._extract_mp4(file_path))
            else:
                logger.warning(f"Metadata extraction not supported for {suffix}")
        except Exception as e:
            logger.warning(f"Could not extract metadata from {file_path}: {e}")

        # Extract duration from librosa if not found
        if metadata['duration'] is None:
            try:
                audio_data = librosa.get_duration(filename=str(file_path))
                metadata['duration'] = float(audio_data)
            except Exception as e:
                logger.warning(f"Could not extract duration from {file_path}: {e}")

        return metadata

    def _extract_id3(self, file_path: Path) -> Dict[str, Any]:
        """Extract ID3 metadata from MP3 file."""
        result = {}
        try:
            audio = EasyID3(str(file_path))
            result['artist'] = audio.get('artist', [None])[0]
            result['title'] = audio.get('title', [None])[0]
            result['album'] = audio.get('album', [None])[0]
            logger.debug(f"Extracted ID3 from {file_path.name}")
        except MutagenError:
            logger.debug(f"No ID3 tags found in {file_path.name}")
        except Exception as e:
            logger.debug(f"Error extracting ID3 from {file_path.name}: {e}")
        return result

    def _extract_flac(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from FLAC file."""
        result = {}
        try:
            audio = FLAC(str(file_path))
            result['artist'] = audio.get('artist', [None])[0]
            result['title'] = audio.get('title', [None])[0]
            result['album'] = audio.get('album', [None])[0]
            if audio.info:
                result['duration'] = float(audio.info.length)
            logger.debug(f"Extracted FLAC metadata from {file_path.name}")
        except Exception as e:
            logger.debug(f"Error extracting FLAC metadata from {file_path.name}: {e}")
        return result

    def _extract_ogg(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from OGG Vorbis file."""
        result = {}
        try:
            audio = OggVorbis(str(file_path))
            result['artist'] = audio.get('artist', [None])[0]
            result['title'] = audio.get('title', [None])[0]
            result['album'] = audio.get('album', [None])[0]
            if audio.info:
                result['duration'] = float(audio.info.length)
            logger.debug(f"Extracted OGG metadata from {file_path.name}")
        except Exception as e:
            logger.debug(f"Error extracting OGG metadata from {file_path.name}: {e}")
        return result

    def _extract_mp4(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from MP4/M4A file."""
        result = {}
        try:
            audio = MP4(str(file_path))
            result['artist'] = audio.get('\xa9ART', [None])[0]
            result['title'] = audio.get('\xa9nam', [None])[0]
            result['album'] = audio.get('\xa9alb', [None])[0]
            if audio.info:
                result['duration'] = float(audio.info.length)
            logger.debug(f"Extracted MP4 metadata from {file_path.name}")
        except Exception as e:
            logger.debug(f"Error extracting MP4 metadata from {file_path.name}: {e}")
        return result

    def validate_audio(self, file_path: Path) -> bool:
        """
        Validate that audio file can be loaded.

        Args:
            file_path: Path to audio file

        Returns:
            True if file can be loaded, False otherwise
        """
        try:
            y, sr = librosa.load(str(file_path), sr=None, mono=True, duration=0.1)
            return len(y) > 0
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return False
