"""Detect breakdowns, drops, and track structure elements."""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import List


@dataclass
class StructurePoint:
    """A significant structural point in a track."""
    time: float  # seconds
    beat: float  # beat number (at detected tempo)
    type: str  # 'drop', 'breakdown', 'transition'
    confidence: float  # 0-1, confidence in detection
    description: str


def detect_drops(y: np.ndarray, sr: int, tempo: float, threshold: float = 0.6) -> List[StructurePoint]:
    """Detect drops (sudden return of energy/percussion after silence/reduction)."""
    # RMS energy over time
    rms = librosa.feature.rms(y=y)[0]
    frames = np.arange(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)

    # Normalize RMS
    rms_norm = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)

    # Find sudden increases in energy (drops)
    rms_diff = np.diff(rms_norm)
    drop_frames = np.where(rms_diff > threshold)[0]

    drops = []
    last_drop_time = -5  # Don't detect drops closer than 5 seconds apart

    for frame in drop_frames:
        time = times[frame]
        if time - last_drop_time >= 5:
            beat = (time * tempo) / 60.0
            confidence = min(1.0, rms_diff[frame] / (threshold * 2))
            drops.append(StructurePoint(
                time=time,
                beat=beat,
                type='drop',
                confidence=confidence,
                description=f"Energy drop at {time:.1f}s (beat {beat:.0f})"
            ))
            last_drop_time = time

    return drops


def detect_breakdowns(y: np.ndarray, sr: int, tempo: float, threshold: float = -0.5) -> List[StructurePoint]:
    """Detect breakdowns (reduction in percussive content)."""
    # Separate harmonic and percussive
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Frame-by-frame percussive energy
    perc_rms = librosa.feature.rms(y=y_percussive)[0]
    frames = np.arange(len(perc_rms))
    times = librosa.frames_to_time(frames, sr=sr)

    # Normalize
    perc_norm = (perc_rms - np.mean(perc_rms)) / (np.std(perc_rms) + 1e-8)

    # Find sudden decreases (breakdowns start here)
    perc_diff = np.diff(perc_norm)
    breakdown_frames = np.where(perc_diff < threshold)[0]

    breakdowns = []
    last_breakdown_time = -8

    for frame in breakdown_frames:
        time = times[frame]
        if time - last_breakdown_time >= 8:  # Min 8 seconds between breakdowns
            beat = (time * tempo) / 60.0
            confidence = min(1.0, abs(perc_diff[frame]) / (abs(threshold) * 2))
            breakdowns.append(StructurePoint(
                time=time,
                beat=beat,
                type='breakdown',
                confidence=confidence,
                description=f"Breakdown at {time:.1f}s (beat {beat:.0f})"
            ))
            last_breakdown_time = time

    return breakdowns


def detect_transitions(y: np.ndarray, sr: int, tempo: float) -> List[StructurePoint]:
    """Detect major beat/spectral transitions."""
    # Spectral flux (change in spectral content)
    S = librosa.stft(y)
    magnitude = np.abs(S)
    flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))

    # Normalize
    flux_norm = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

    # Find peaks in spectral flux
    threshold = np.percentile(flux_norm, 85)
    transition_frames = np.where(flux_norm > threshold)[0]

    transitions = []
    last_transition = -4
    times = librosa.frames_to_time(transition_frames, sr=sr)

    for i, frame in enumerate(transition_frames):
        time = times[i]
        if time - last_transition >= 4:  # Min 4 seconds apart
            beat = (time * tempo) / 60.0
            confidence = min(1.0, flux_norm[frame] / (threshold * 2))
            transitions.append(StructurePoint(
                time=time,
                beat=beat,
                type='transition',
                confidence=confidence,
                description=f"Transition at {time:.1f}s (beat {beat:.0f})"
            ))
            last_transition = time

    return transitions


def analyze_structure(file_path: str, tempo: float, sr: int = 22050) -> dict:
    """Complete structural analysis of a track."""
    y, sr = librosa.load(file_path, sr=sr)

    drops = detect_drops(y, sr, tempo)
    breakdowns = detect_breakdowns(y, sr, tempo)
    transitions = detect_transitions(y, sr, tempo)

    # Combine and sort by time
    all_points = drops + breakdowns + transitions
    all_points.sort(key=lambda p: p.time)

    return {
        'drops': drops,
        'breakdowns': breakdowns,
        'transitions': transitions,
        'all_points': all_points,
        'structure_summary': {
            'num_drops': len(drops),
            'num_breakdowns': len(breakdowns),
            'num_transitions': len(transitions),
        }
    }
