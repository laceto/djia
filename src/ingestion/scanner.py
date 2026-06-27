"""Audio file scanner for the data directory."""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class AudioScanner:
    """Scans a directory recursively for audio files."""

    # Supported audio formats
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

    def __init__(self, data_dir: str = "data"):
        """Initialize scanner with data directory path."""
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            logger.warning(f"Data directory '{data_dir}' does not exist")

    def scan(self) -> List[Dict[str, Any]]:
        """
        Scan data directory recursively for audio files.

        Returns:
            List of dictionaries with 'path' and 'format' keys.
        """
        audio_files = []

        if not self.data_dir.exists():
            logger.error(f"Data directory '{self.data_dir}' not found")
            return audio_files

        try:
            for audio_file in sorted(self.data_dir.rglob('*')):
                if audio_file.is_file() and audio_file.suffix.lower() in self.SUPPORTED_FORMATS:
                    audio_files.append({
                        'path': audio_file,
                        'format': audio_file.suffix.lower(),
                        'name': audio_file.name,
                    })
                    logger.debug(f"Found audio file: {audio_file}")
        except Exception as e:
            logger.error(f"Error scanning directory: {e}")

        logger.info(f"Scan complete: found {len(audio_files)} audio file(s)")
        return audio_files

    def get_file_count(self) -> int:
        """Get count of audio files in directory."""
        return len(self.scan())
