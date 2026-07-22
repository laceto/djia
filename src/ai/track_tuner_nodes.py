"""Node functions for the Track Tuner agent."""

import numpy as np
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from ..dsp.extractor import extract_track_features
from ..dsp.config import custom_config
from ..dsp.phrasing_engine import time_to_bar
from .track_tuner_state import (
    TrackTunerState,
    SegmentQuality,
    TuneConfig,
    DEFAULT_CONFIGS,
    QUALITY_THRESHOLDS,
)


def load_track(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Load and extract basic info from track."""
    try:
        import librosa
        from ..dsp.groove_engine import analyze_groove

        track_path = state["track_path"]
        track_name = Path(track_path).stem

        # Load audio
        y, sr = librosa.load(track_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)

        # Get BPM
        groove = analyze_groove(y, sr)
        bpm = groove.bpm

        msg = f"[LoadTrack] Loaded {track_name} | BPM: {bpm:.1f} | Duration: {duration:.1f}s"

        return {
            "track_name": track_name,
            "bpm": bpm,
            "duration": duration,
            "messages": [AIMessage(content=msg)],
        }

    except Exception as e:
        msg = f"[LoadTrack] ERROR: {str(e)}"
        return {
            "messages": [AIMessage(content=msg)],
        }


def initialize_config(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Initialize tuning configuration."""
    # Get preset from config or default to "minimal"
    preset = (config or {}).get("configurable", {}).get("preset", "minimal")

    if preset not in DEFAULT_CONFIGS:
        preset = "minimal"

    cfg = DEFAULT_CONFIGS[preset]
    tune_cfg: TuneConfig = {
        "novelty_threshold": cfg["novelty_threshold"],
        "min_segment_duration": cfg["min_segment_duration"],
        "breakdown_duration_threshold": cfg["breakdown_duration_threshold"],
        "iteration": 0,
    }

    msg = f"[InitializeConfig] Using preset: {preset}"

    return {
        "current_config": tune_cfg,
        "initial_config": tune_cfg,
        "iterations_completed": 0,
        "messages": [AIMessage(content=msg)],
    }


def analyze_track(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Analyze track with current config."""
    try:
        track_path = state["track_path"]
        cfg = state["current_config"]

        # Create config
        dsp_config = custom_config(
            novelty_threshold=cfg["novelty_threshold"],
            min_segment_duration=cfg["min_segment_duration"],
            breakdown_duration_threshold=cfg["breakdown_duration_threshold"],
        )

        # Extract features
        track = extract_track_features(track_path, config=dsp_config)

        # Get segment info
        segments = track.phrasing.segments
        bpm = state["bpm"]

        # Calculate bar distribution
        bars_per_seg = []
        for seg in segments:
            start_bar = time_to_bar(seg.start_time, bpm)
            end_bar = time_to_bar(seg.end_time, bpm)
            bars = end_bar - start_bar
            bars_per_seg.append(bars)

        avg_bars = np.mean(bars_per_seg) if bars_per_seg else 0
        std_bars = np.std(bars_per_seg) if bars_per_seg else 0

        # Count false breakdowns (too many breakdown labels = poor)
        breakdown_count = sum(1 for seg in segments if "breakdown" in seg.label.lower())
        has_false_breakdowns = breakdown_count > 3  # More than 3 is suspicious

        msg = f"[AnalyzeTrack] {len(segments)} segments | Avg bars: {avg_bars:.1f} | Std: {std_bars:.1f}"

        return {
            "current_segments": [
                {
                    "label": seg.label,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "start_bar": time_to_bar(seg.start_time, bpm),
                    "end_bar": time_to_bar(seg.end_time, bpm),
                }
                for seg in segments
            ],
            "current_quality": {
                "num_segments": len(segments),
                "avg_bars_per_segment": avg_bars,
                "regularity_std": std_bars,
                "has_false_breakdowns": has_false_breakdowns,
                "quality_score": 0.0,  # Will be set by evaluator
            },
            "messages": [AIMessage(content=msg)],
        }

    except Exception as e:
        msg = f"[AnalyzeTrack] ERROR: {str(e)}"
        return {
            "messages": [AIMessage(content=msg)],
        }


def evaluate_quality(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Evaluate segmentation quality."""
    quality = state["current_quality"]
    bpm = state["bpm"]
    duration = state["duration"]

    # Calculate quality score (0-1)
    # Factors:
    # 1. Segment count (target: 2-12 for typical track)
    # 2. Regularity (low std = high score)
    # 3. No false breakdowns
    # 4. Reasonable bar distribution

    num_segs = quality["num_segments"]
    avg_bars = quality["avg_bars_per_segment"]
    std_bars = quality["regularity_std"]
    has_false = quality["has_false_breakdowns"]

    # Ideal: 2-10 segments
    segment_score = 1.0 if 2 <= num_segs <= 10 else max(0, 1 - abs(num_segs - 6) / 10)

    # Ideal: 16-256 bars per segment (for typical DJ tracks)
    bars_score = 1.0 if 16 <= avg_bars <= 256 else max(0, 1 - abs(avg_bars - 100) / 200)

    # Regularity: lower std is better
    regularity_score = 1.0 / (1.0 + std_bars / 50)  # Normalize std deviation

    # False breakdowns penalty
    false_penalty = 0.3 if has_false else 0.0

    # Composite score
    quality_score = (segment_score * 0.3 + bars_score * 0.3 + regularity_score * 0.4) - false_penalty
    quality_score = max(0, min(1, quality_score))

    # Determine satisfaction
    satisfied = quality_score >= QUALITY_THRESHOLDS["good"]
    reason = ""

    if satisfied:
        reason = "Quality threshold met"
    elif num_segs > 10:
        reason = "Too many segments detected"
    elif has_false:
        reason = "Too many false breakdown labels"
    else:
        reason = "Irregular segment distribution"

    msg = f"[EvaluateQuality] Score: {quality_score:.2f} | {reason}"

    return {
        "current_quality": {**quality, "quality_score": quality_score},
        "satisfied": satisfied,
        "reason": reason,
        "analysis_history": [
            {
                "iteration": state["iterations_completed"],
                "config": state["current_config"],
                "quality_score": quality_score,
                "reason": reason,
            }
        ],
        "messages": [AIMessage(content=msg)],
    }


def suggest_tuning(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Suggest parameter adjustments."""
    quality = state["current_quality"]
    current_cfg = state["current_config"]
    reason = state["reason"]

    new_cfg: TuneConfig = {
        "novelty_threshold": current_cfg["novelty_threshold"],
        "min_segment_duration": current_cfg["min_segment_duration"],
        "breakdown_duration_threshold": current_cfg["breakdown_duration_threshold"],
        "iteration": current_cfg["iteration"] + 1,
    }

    recommendations = []

    if quality["num_segments"] > 10 and reason == "Too many segments detected":
        # Increase threshold to reduce detections
        new_cfg["novelty_threshold"] = min(0.8, current_cfg["novelty_threshold"] + 0.1)
        new_cfg["min_segment_duration"] = min(20, current_cfg["min_segment_duration"] + 2)
        recommendations.append("Reduced sensitivity to filter noise")

    elif quality["has_false_breakdowns"]:
        # Increase breakdown threshold
        new_cfg["breakdown_duration_threshold"] = current_cfg["breakdown_duration_threshold"] + 8
        recommendations.append("Increased breakdown duration threshold")

    elif quality["regularity_std"] > 50:
        # Better structure detection
        new_cfg["novelty_threshold"] = max(0.3, current_cfg["novelty_threshold"] - 0.05)
        recommendations.append("Increased sensitivity to detect better structure")

    msg = f"[SuggestTuning] {'; '.join(recommendations) if recommendations else 'No changes needed'}"

    return {
        "current_config": new_cfg,
        "recommendations": recommendations,
        "messages": [AIMessage(content=msg)],
    }


def finalize(state: TrackTunerState, config: RunnableConfig) -> dict:
    """Finalize analysis and prepare result."""
    quality = state["current_quality"]
    cfg = state["current_config"]

    msg = f"[Finalize] Analysis complete | Quality: {quality['quality_score']:.2f} | Iterations: {state['iterations_completed']}"

    return {
        "messages": [AIMessage(content=msg)],
    }
