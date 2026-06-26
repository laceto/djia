"""Data structures for DJIA track analysis and features."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from datetime import datetime


@dataclass
class Segment:
    """A labeled section of a track."""
    start_time: float  # seconds
    end_time: float    # seconds
    label: str         # e.g., "intro", "build", "drop", "breakdown", "outro"
    confidence: float = 1.0  # 0-1


@dataclass
class CuePoint:
    """A predicted hot-cue position for mixing."""
    time: float        # seconds
    label: str         # e.g., "Pad 1", "Pad 2", "Pad 4"
    type: str = "hotcue"  # cue type


@dataclass
class PhrasingResult:
    """Output from phrasing engine (Step 1)."""
    segment_boundaries: List[float]  # timestamps of section changes
    segments: List[Segment]           # labeled sections with times
    cue_points: List[CuePoint]        # predicted hot-cue positions
    structure_confidence: float = 0.0  # overall confidence in detection


@dataclass
class GrooveResult:
    """Output from groove engine (Step 2)."""
    bpm: float                     # decimal BPM (e.g., 126.04)
    beat_grid: np.ndarray         # beat frame positions
    beat_times: List[float]        # beat times in seconds
    swing_score: float             # 0.0 (stiff) to 1.0 (groovy)
    tempo_stability: bool          # whether tempo drifts
    stability_variance: float = 0.0  # tempo variance


@dataclass
class MoodResult:
    """Output from mood engine (Step 3)."""
    key: str                       # musical key (e.g., "A minor")
    camelot_key: str               # Camelot notation (e.g., "7A")
    brightness: float              # 0.0 (dark) to 1.0 (bright)
    key_confidence: float           # 0.0-1.0 confidence in key detection


@dataclass
class CurationResult:
    """Output from curation engine (Step 4)."""
    danceability: float            # 0.0-1.0 score
    energy_curve: np.ndarray       # RMS energy over time
    energy_type: str               # "flat", "dynamic", or "gradual"
    semantic_tags: List[str]       # auto-generated labels
    complexity_score: float = 0.0  # 0-1 complexity metric


@dataclass
class Track:
    """Complete feature vector for a single track."""
    file_path: str
    duration: float                # seconds

    # Phase 2 DSP Analysis
    phrasing: PhrasingResult
    groove: GrooveResult
    mood: MoodResult
    curation: CurationResult

    # Metadata
    sample_rate: int = 22050
    analysis_timestamp: Optional[str] = None  # ISO format

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()


@dataclass
class AnalysisResult:
    """Wrapper for a completed track analysis."""
    track: Track
    status: str = "success"  # "success" or "error"
    error_message: Optional[str] = None


# Helper function to create a minimal Track for testing
def create_test_track(file_path: str = "test.wav", duration: float = 60.0) -> Track:
    """Create a test track with default values."""
    return Track(
        file_path=file_path,
        duration=duration,
        phrasing=PhrasingResult(
            segment_boundaries=[],
            segments=[],
            cue_points=[],
        ),
        groove=GrooveResult(
            bpm=120.0,
            beat_grid=np.array([]),
            beat_times=[],
            swing_score=0.5,
            tempo_stability=True,
        ),
        mood=MoodResult(
            key="A minor",
            camelot_key="7A",
            brightness=0.5,
            key_confidence=0.8,
        ),
        curation=CurationResult(
            danceability=0.7,
            energy_curve=np.array([]),
            energy_type="dynamic",
            semantic_tags=[],
        ),
    )
