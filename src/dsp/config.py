"""
DSP Engine Configuration — Tunable parameters for feature extraction.

Adjust these presets based on track genre and characteristics.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PhrasingConfig:
    """Phrasing Engine parameters for low-band-energy structure detection."""

    # Minimum section length in bars; shorter sections get merged into the
    # previous one so short blips don't fragment the structure.
    # 2 = aggressive (allows short sections)
    # 4 = balanced
    # 8 = conservative (forces long sections)
    min_bars: int = 4

    # Kick-on threshold as a fraction of peak low-band energy (0-1): lower =
    # more of the track reads as "kick on" (drop), so more/shorter breakdowns.
    # 0.3 = aggressive (many drop/breakdown transitions)
    # 0.4 = balanced
    # 0.5 = conservative (only clear drops register)
    thresh_frac: float = 0.4

    # Max hot cues / physical pads (None = one cue per structural section)
    max_pads: Optional[int] = None


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
            min_bars=4,
            thresh_frac=0.4,
        ),
    ),
    "minimal": DSPConfig(
        phrasing=PhrasingConfig(
            min_bars=8,  # ← Larger minimum = fewer, longer sections
            thresh_frac=0.5,  # ← Higher threshold = fewer false detections
        ),
    ),
    "house": DSPConfig(
        phrasing=PhrasingConfig(
            min_bars=6,
            thresh_frac=0.45,
        ),
    ),
    "techno": DSPConfig(
        phrasing=PhrasingConfig(
            min_bars=4,
            thresh_frac=0.35,
        ),
    ),
    "aggressive": DSPConfig(
        phrasing=PhrasingConfig(
            min_bars=2,  # ← Small minimum = short sections kept
            thresh_frac=0.3,  # ← Low threshold = more drop/breakdown transitions
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
    min_bars: int = 4,
    thresh_frac: float = 0.4,
    max_pads: Optional[int] = None,
) -> DSPConfig:
    """
    Create a custom DSP configuration.

    Args:
        min_bars: Minimum section length in bars
        thresh_frac: Kick-on threshold as a fraction of peak low-band energy
        max_pads: Max hot cues / physical pads (None = one per section)

    Returns:
        DSPConfig: Custom configuration object
    """
    return DSPConfig(
        phrasing=PhrasingConfig(
            min_bars=min_bars,
            thresh_frac=thresh_frac,
            max_pads=max_pads,
        ),
    )
