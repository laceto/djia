"""Transition mapper for scoring and building track transition graphs."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class TransitionScore:
    """Score breakdown for a track transition."""
    bpm_score: float
    key_score: float
    mood_score: float
    energy_score: float
    overall_score: float


# Camelot wheel for harmonic mixing
CAMELOT_WHEEL = {
    'C': 8, 'C#': 3, 'D': 10, 'D#': 5, 'E': 12,
    'F': 7, 'F#': 2, 'G': 9, 'G#': 4, 'A': 11,
    'A#': 6, 'B': 1,
}


def _calculate_bpm_compatibility(bpm_a: float, bpm_b: float) -> float:
    """Calculate BPM compatibility score (0.0-1.0)."""
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0

    # Identical or very close
    if abs(bpm_a - bpm_b) < 0.5:
        return 1.0

    # Within 2%
    if abs(bpm_a - bpm_b) / max(bpm_a, bpm_b) < 0.02:
        return 0.95

    # Within 5%
    if abs(bpm_a - bpm_b) / max(bpm_a, bpm_b) < 0.05:
        return 0.85

    # Check harmonic relationships (halving/doubling)
    ratio = bpm_a / bpm_b
    if 0.48 < ratio < 0.52 or 1.98 < ratio < 2.02:  # 1:2 or 2:1
        return 0.75
    if 0.73 < ratio < 0.77 or 1.28 < ratio < 1.32:  # 3:4 or 4:3
        return 0.70

    # Large difference
    if abs(bpm_a - bpm_b) > 30:
        return 0.2

    # Linear falloff
    percent_diff = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)
    return max(0.1, 1.0 - percent_diff)


def _calculate_key_compatibility(key_a: Optional[str], key_b: Optional[str]) -> float:
    """Calculate key compatibility using Camelot wheel (0.0-1.0)."""
    if key_a is None or key_b is None:
        return 0.5  # Neutral score when key data unavailable

    # Normalize keys
    key_a = str(key_a).strip()
    key_b = str(key_b).strip()

    if key_a == key_b:
        return 1.0

    # Get Camelot positions
    pos_a = CAMELOT_WHEEL.get(key_a)
    pos_b = CAMELOT_WHEEL.get(key_b)

    if pos_a is None or pos_b is None:
        return 0.5  # Unknown keys

    # Calculate distance on wheel (12-key circle)
    distance = abs(pos_a - pos_b)
    distance = min(distance, 12 - distance)

    if distance < 0.5:  # Same key
        return 1.0
    elif distance < 1.1:  # Adjacent (±1)
        return 0.9
    elif distance < 2.0:  # ±2
        return 0.85
    elif distance < 3.0:  # ±3 (perfect 5th/4th)
        return 0.80
    elif distance < 5.0:
        return 0.65
    else:
        return 0.4


def _calculate_mood_continuity(mood_a: Optional[Dict[str, float]],
                               mood_b: Optional[Dict[str, float]]) -> float:
    """Calculate mood continuity score using cosine similarity (0.0-1.0)."""
    if mood_a is None or mood_b is None:
        return 0.5

    if not mood_a or not mood_b:
        return 0.5

    # Get common mood dimensions
    mood_dims = set(mood_a.keys()) & set(mood_b.keys())
    if not mood_dims:
        return 0.5

    # Calculate cosine similarity
    try:
        vec_a = np.array([mood_a[m] for m in mood_dims])
        vec_b = np.array([mood_b[m] for m in mood_dims])

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.5

        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float(max(0.0, similarity))
    except Exception:
        return 0.5


def _calculate_energy_arc(rms_a: Optional[float], rms_b: Optional[float]) -> float:
    """Calculate energy arc compatibility (0.0-1.0)."""
    if rms_a is None or rms_b is None:
        return 0.7

    # Small changes are best
    if abs(rms_a - rms_b) < 0.01:
        return 1.0

    # Moderate changes
    if abs(rms_a - rms_b) < 0.05:
        return 0.85

    # Larger changes
    if abs(rms_a - rms_b) < 0.1:
        return 0.7

    # Very large changes (jarring)
    return max(0.3, 1.0 - abs(rms_a - rms_b) / 0.2)


def score_transition(track_a: Dict, track_b: Dict,
                    weights: Optional[Dict[str, float]] = None) -> TransitionScore:
    """
    Score the quality of a transition from track_a to track_b.

    Considers:
    - BPM compatibility (40%)
    - Key harmonic distance (30%)
    - Mood continuity (20%)
    - Energy arc smoothness (10%)

    Args:
        track_a: Source track features dict
        track_b: Target track features dict
        weights: Custom scoring weights

    Returns:
        TransitionScore with breakdown and overall score
    """
    if weights is None:
        weights = {
            'bpm': 0.40,
            'key': 0.30,
            'mood': 0.20,
            'energy': 0.10,
        }

    # Extract features
    bpm_a = track_a.get('tempo') or track_a.get('bpm', 120)
    bpm_b = track_b.get('tempo') or track_b.get('bpm', 120)

    key_a = track_a.get('key')
    key_b = track_b.get('key')

    mood_a = track_a.get('mood', {})
    mood_b = track_b.get('mood', {})

    rms_a = track_a.get('rms_mean')
    rms_b = track_b.get('rms_mean')

    # Calculate component scores
    bpm_score = _calculate_bpm_compatibility(bpm_a, bpm_b)
    key_score = _calculate_key_compatibility(key_a, key_b)
    mood_score = _calculate_mood_continuity(mood_a, mood_b)
    energy_score = _calculate_energy_arc(rms_a, rms_b)

    # Weighted overall
    overall = (
        bpm_score * weights['bpm'] +
        key_score * weights['key'] +
        mood_score * weights['mood'] +
        energy_score * weights['energy']
    )

    return TransitionScore(
        bpm_score=round(bpm_score, 3),
        key_score=round(key_score, 3),
        mood_score=round(mood_score, 3),
        energy_score=round(energy_score, 3),
        overall_score=round(overall, 3),
    )


def build_transition_graph(all_tracks: Dict[int, Dict]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Build a directed graph of all tracks with transition scores as edge weights.

    Creates a complete transition matrix where each track can transition to any other.
    Useful for playlist generation algorithms.

    Args:
        all_tracks: Dictionary mapping track_id -> track_features_dict

    Returns:
        Dictionary mapping track_id -> list of (target_id, score) tuples
        sorted by score (highest first)
    """
    graph: Dict[int, List[Tuple[int, float]]] = {}

    track_ids = list(all_tracks.keys())

    for source_id in track_ids:
        edges: List[Tuple[int, float]] = []

        for target_id in track_ids:
            if source_id == target_id:
                continue  # No self-loops

            score = score_transition(all_tracks[source_id], all_tracks[target_id])
            edges.append((target_id, score.overall_score))

        # Sort by score (highest first)
        edges.sort(key=lambda x: x[1], reverse=True)
        graph[source_id] = edges

    return graph
