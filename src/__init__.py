"""DJIA - DJ Mixing Analytics package.

Main modules:
- orchestrator: Master orchestrator for full pipeline
- cli: Command-line interface
- audio_analysis: DSP feature extraction
- ai: AI layer (mood classification, transitions, playlists)
- database: SQLite storage and queries
"""

from .orchestrator import Orchestrator
from .audio_analysis import analyze_track
from .database.store import TrackStore

__all__ = [
    'Orchestrator',
    'analyze_track',
    'TrackStore',
]

__version__ = '1.0.0'
