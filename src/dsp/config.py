"""
DSP Engine Configuration — Tunable parameters for feature extraction.

Adjust these presets based on track genre and characteristics.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class PhrasingConfig:
    """Phrasing Engine parameters for structure detection."""

    # Peak detection threshold (0-1): higher = fewer segments detected
    # 0.3 = aggressive (many segments)
    # 0.5 = balanced
    # 0.7 = conservative (only major changes)
    novelty_threshold: float = 0.5

    # Minimum gap between segments in seconds: higher = fewer, longer segments
    # 4.0 = aggressive (allows short segments)
    # 8.0 = balanced
    # 16.0 = conservative (forces long segments)
    min_segment_duration: float = 8.0

    # Duration threshold for labeling segments as "breakdown" (seconds)
    # 16 = aggressive (any short segment is breakdown)
    # 24 = balanced
    # 32 = conservative (must be very short to be breakdown)
    breakdown_duration_threshold: float = 24.0

    # --- Element-onset detection (when a new sound element enters) ---
    # Number of log-spaced frequency bands to watch independently.
    # More bands = finer frequency localization, more potential onsets.
    element_n_bands: int = 8

    # Peak height (0-1) on the per-band additive-novelty curve to call an onset.
    # Lower = more sensitive (catches quieter elements, more false positives).
    element_onset_threshold: float = 0.4

    # Minimum bars a new element must persist to count (rejects one-shot FX).
    element_min_sustain_bars: float = 2.0


@dataclass
class GrooveConfig:
    """Groove Engine parameters for BPM and rhythm detection."""

    # Onset strength sensitivity (higher = more onsets detected)
    onset_strength_scale: float = 1.0

    # Tempogram smoothing window in seconds
    tempogram_window: float = 2.0


@dataclass
class MoodConfig:
    """Mood Engine parameters for key and brightness detection."""

    # Spectral centroid percentile for brightness (0-100)
    brightness_percentile: float = 75.0


@dataclass
class CurationConfig:
    """Curation Engine parameters for danceability and tags."""

    # Minimum spectral flux for danceability threshold
    danceability_threshold: float = 0.5


@dataclass
class DSPConfig:
    """Master DSP configuration — all engines in one place."""

    phrasing: PhrasingConfig = None
    groove: GrooveConfig = None
    mood: MoodConfig = None
    curation: CurationConfig = None
    hop_length: int = 512  # STFT hop length

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.phrasing is None:
            self.phrasing = PhrasingConfig()
        if self.groove is None:
            self.groove = GrooveConfig()
        if self.mood is None:
            self.mood = MoodConfig()
        if self.curation is None:
            self.curation = CurationConfig()


# Presets for different track types
PRESETS: Dict[str, DSPConfig] = {
    "default": DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=0.5,
            min_segment_duration=8.0,
            breakdown_duration_threshold=24.0,
        ),
    ),
    "minimal": DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=0.65,  # ← Higher threshold = fewer false detections
            min_segment_duration=12.0,  # ← Larger gap = longer segments
            breakdown_duration_threshold=32.0,  # ← Less aggressive labeling
        ),
    ),
    "house": DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=0.55,
            min_segment_duration=10.0,
            breakdown_duration_threshold=28.0,
        ),
    ),
    "techno": DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=0.45,
            min_segment_duration=6.0,
            breakdown_duration_threshold=20.0,
        ),
    ),
    "aggressive": DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=0.3,  # ← Low threshold = many segments
            min_segment_duration=4.0,  # ← Small gap = short segments
            breakdown_duration_threshold=16.0,  # ← Aggressive labeling
        ),
    ),
}


def get_config(preset: str = "default") -> DSPConfig:
    """
    Get DSP configuration by preset name.

    Args:
        preset: One of "default", "minimal", "house", "techno", "aggressive"

    Returns:
        DSPConfig: Configuration object with all parameters
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")
    return PRESETS[preset]


def custom_config(
    novelty_threshold: float = 0.5,
    min_segment_duration: float = 8.0,
    breakdown_duration_threshold: float = 24.0,
) -> DSPConfig:
    """
    Create a custom DSP configuration.

    Args:
        novelty_threshold: Peak detection threshold (0-1)
        min_segment_duration: Minimum gap between segments (seconds)
        breakdown_duration_threshold: Duration threshold for breakdown label (seconds)

    Returns:
        DSPConfig: Custom configuration object
    """
    return DSPConfig(
        phrasing=PhrasingConfig(
            novelty_threshold=novelty_threshold,
            min_segment_duration=min_segment_duration,
            breakdown_duration_threshold=breakdown_duration_threshold,
        ),
    )
