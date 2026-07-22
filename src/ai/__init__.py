"""AI layer for DJIA - advanced audio analysis and playlist generation."""

# Phase 3 components
from .stem_separator import StemSeparator, separate_stems
from .classifier import MoodClassifier, classify_mood
from .segmentation import StructureSegmenter, detect_structure
from .processor import AIProcessor, process_with_stems

# Phase 5 components (advanced AI)
from .transition_mapper import score_transition, build_transition_graph, TransitionScore
from .playlist_generator import generate_playlist, playlist_summary
from .setlist_generator import build_setlist, generate_setlist

__all__ = [
    # Phase 3
    'StemSeparator',
    'separate_stems',
    'MoodClassifier',
    'classify_mood',
    'StructureSegmenter',
    'detect_structure',
    'AIProcessor',
    'process_with_stems',
    # Phase 5
    'score_transition',
    'build_transition_graph',
    'TransitionScore',
    'generate_playlist',
    'playlist_summary',
    'build_setlist',
    'generate_setlist',
]
