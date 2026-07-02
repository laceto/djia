"""State definitions for the Track Tuner agent."""

from __future__ import annotations
import operator
from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SegmentQuality(TypedDict):
    """Quality assessment of segmentation."""
    num_segments: int
    avg_bars_per_segment: float
    regularity_std: float
    has_false_breakdowns: bool
    quality_score: float  # 0-1


class TuneConfig(TypedDict):
    """Tuning configuration."""
    novelty_threshold: float
    min_segment_duration: float
    breakdown_duration_threshold: float
    iteration: int


class TrackAnalysisResult(TypedDict):
    """Result of analyzing a single track."""
    track_path: str
    track_name: str
    bpm: float
    duration: float
    num_segments: int
    avg_bars: float
    config: TuneConfig
    quality_score: float
    satisfied: bool
    iterations_used: int


class TrackTunerState(TypedDict):
    """State for track tuning agent."""
    # Track info
    track_path: str
    track_name: Optional[str]
    bpm: Optional[float]
    duration: Optional[float]

    # Current analysis
    current_segments: Optional[List[dict]]
    current_quality: Optional[SegmentQuality]

    # Configuration
    current_config: Optional[TuneConfig]
    initial_config: Optional[TuneConfig]

    # Tuning state
    iterations_completed: int
    max_iterations: int
    satisfied: bool
    reason: Optional[str]

    # Results
    analysis_history: Annotated[List[dict], operator.add]
    recommendations: Annotated[List[str], operator.add]

    # Messages for traceability
    messages: Annotated[List[BaseMessage], add_messages]


class BatchTrackTunerState(TypedDict):
    """State for batch processing multiple tracks."""
    track_paths: List[str]
    current_track_index: int
    results: Annotated[List[TrackAnalysisResult], operator.add]
    config_preset: str  # "minimal", "default", "house", etc.
    messages: Annotated[List[BaseMessage], add_messages]


# Default tuning configurations
DEFAULT_CONFIGS = {
    "default": {
        "novelty_threshold": 0.5,
        "min_segment_duration": 8.0,
        "breakdown_duration_threshold": 24.0,
    },
    "minimal": {
        "novelty_threshold": 0.65,
        "min_segment_duration": 12.0,
        "breakdown_duration_threshold": 32.0,
    },
    "house": {
        "novelty_threshold": 0.55,
        "min_segment_duration": 10.0,
        "breakdown_duration_threshold": 28.0,
    },
    "techno": {
        "novelty_threshold": 0.45,
        "min_segment_duration": 6.0,
        "breakdown_duration_threshold": 20.0,
    },
    "aggressive": {
        "novelty_threshold": 0.3,
        "min_segment_duration": 4.0,
        "breakdown_duration_threshold": 16.0,
    },
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "excellent": 0.85,  # Well-balanced segmentation
    "good": 0.70,       # Acceptable segmentation
    "fair": 0.50,       # Needs improvement
    "poor": 0.0,        # Poor segmentation
}
