"""Master Orchestrator — Load audio and run all 4 DSP engines."""

import librosa
import numpy as np
from typing import Optional
from ..features.schema import Track, AnalysisResult
from .phrasing_engine import analyze_structure
from .groove_engine import analyze_groove
from .mood_engine import analyze_mood
from .curation_engine import analyze_curation
from .config import DSPConfig, get_config, custom_config


def load_audio(file_path: str, sr: int = 22050, duration: Optional[float] = None) -> tuple[np.ndarray, int]:
    """
    Load audio file using librosa.

    Args:
        file_path: Path to audio file
        sr: Sample rate (default 22050)
        duration: Maximum duration to load in seconds (optional)

    Returns:
        (y, sr): Audio waveform and sample rate
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    return y, sr


def extract_track_features(
    file_path: str,
    sr: int = 22050,
    duration: Optional[float] = None,
    hop_length: int = 512,
    config: Optional[DSPConfig] = None
) -> Track:
    """
    Extract complete track DNA using all 4 DSP engines.

    Pipeline:
    1. Load audio
    2. Run Groove Engine -> BPM, beat grid, swing
    3. Run Phrasing Engine -> segments, boundaries, cues (uses BPM)
    4. Run Mood Engine -> key, camelot, brightness
    5. Run Curation Engine -> danceability, energy, tags (uses BPM, swing, brightness)
    6. Combine results into Track

    Args:
        file_path: Path to audio file
        sr: Sample rate (default 22050)
        duration: Maximum duration to load (optional)
        hop_length: Hop length for DSP analysis
        config: DSPConfig with tunable parameters.
                If None, uses "default" preset.
                Use get_config("minimal"/"house"/"techno") or custom_config() for presets.

    Returns:
        Track: Complete feature vector with all DSP results
    """
    # Use default config if none provided
    if config is None:
        config = get_config("default")

    # Step 0: Load audio
    y, sr = load_audio(file_path, sr=sr, duration=duration)

    if len(y) == 0:
        raise ValueError(f"Failed to load audio from {file_path}")

    track_duration = librosa.get_duration(y=y, sr=sr)

    # Step 1: Groove Engine (must be first to get BPM)
    groove = analyze_groove(y, sr, hop_length=hop_length)

    # Step 2: Phrasing Engine (uses BPM from groove + config parameters)
    phrasing = analyze_structure(
        y, sr,
        bpm=groove.bpm,
        hop_length=hop_length,
        novelty_threshold=config.phrasing.novelty_threshold,
        min_segment_duration=config.phrasing.min_segment_duration,
        breakdown_threshold=config.phrasing.breakdown_duration_threshold,
        include_beats=True  # Add beat ranges to segment labels
    )

    # Step 3: Mood Engine (independent)
    mood = analyze_mood(y, sr)

    # Step 4: Curation Engine (uses BPM, swing, brightness)
    curation = analyze_curation(
        y, sr,
        bpm=groove.bpm,
        swing_score=groove.swing_score,
        brightness=mood.brightness,
        hop_length=hop_length
    )

    # Combine into Track
    track = Track(
        file_path=file_path,
        duration=track_duration,
        phrasing=phrasing,
        groove=groove,
        mood=mood,
        curation=curation,
        sample_rate=sr
    )

    return track


def analyze_track(
    file_path: str,
    sr: int = 22050,
    duration: Optional[float] = None,
    hop_length: int = 512,
    config: Optional[DSPConfig] = None
) -> AnalysisResult:
    """
    Analyze a single track and return wrapped result.

    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Maximum duration to load
        hop_length: Hop length for DSP analysis
        config: DSPConfig with tunable parameters

    Returns:
        AnalysisResult: Wrapped track with status
    """
    try:
        track = extract_track_features(
            file_path=file_path,
            sr=sr,
            duration=duration,
            hop_length=hop_length,
            config=config
        )

        return AnalysisResult(
            track=track,
            status="success"
        )

    except Exception as e:
        # Return error result
        from ..features.schema import create_test_track
        test_track = create_test_track(file_path)

        return AnalysisResult(
            track=test_track,
            status="error",
            error_message=str(e)
        )


def extract_feature_vector(track: Track) -> dict:
    """
    Convert Track to flat feature vector for ML/similarity matching.

    Args:
        track: Track with all features

    Returns:
        feature_dict: Flat dictionary of normalized features
    """
    features = {
        # Groove features
        "bpm": track.groove.bpm,
        "swing_score": track.groove.swing_score,
        "tempo_stability": float(track.groove.tempo_stability),

        # Mood features
        "brightness": track.mood.brightness,
        "key_confidence": track.mood.key_confidence,

        # Curation features
        "danceability": track.curation.danceability,
        "energy_type_flat": 1.0 if track.curation.energy_type == "flat" else 0.0,
        "energy_type_dynamic": 1.0 if track.curation.energy_type == "dynamic" else 0.0,
        "energy_type_gradual": 1.0 if track.curation.energy_type == "gradual" else 0.0,
        "complexity_score": track.curation.complexity_score,

        # Phrasing features
        "num_segments": len(track.phrasing.segments),
        "num_cue_points": len(track.phrasing.cue_points),
        "structure_confidence": track.phrasing.structure_confidence,

        # Metadata
        "duration": track.duration,
    }

    return features


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = analyze_track(file_path)

        if result.status == "success":
            track = result.track
            print(f"File: {track.file_path}")
            print(f"Duration: {track.duration:.1f}s")
            print(f"\nGroove:")
            print(f"  BPM: {track.groove.bpm:.2f}")
            print(f"  Swing: {track.groove.swing_score:.2f}")
            print(f"  Stable: {track.groove.tempo_stability}")
            print(f"\nMood:")
            print(f"  Key: {track.mood.key}")
            print(f"  Camelot: {track.mood.camelot_key}")
            print(f"  Brightness: {track.mood.brightness:.2f}")
            print(f"\nPhrasing:")
            print(f"  Segments: {len(track.phrasing.segments)}")
            print(f"  Cues: {len(track.phrasing.cue_points)}")
            print(f"\nCuration:")
            print(f"  Danceability: {track.curation.danceability:.2f}")
            print(f"  Energy: {track.curation.energy_type}")
            print(f"  Tags: {', '.join(track.curation.semantic_tags)}")
        else:
            print(f"Error: {result.error_message}")
    else:
        print("Usage: python extractor.py <audio_file>")
