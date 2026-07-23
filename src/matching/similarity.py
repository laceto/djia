"""Track similarity engine using cosine similarity."""

import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


# Features to include in similarity computation
SIMILARITY_FEATURES = [
    'bpm',
    'spectral_centroid_mean',
    'spectral_centroid_std',
    'spectral_rolloff_mean',
    'spectral_flux_mean',
    'harmonic_ratio',
    'percussive_ratio',
    'mfcc_mean',
    'mfcc_std',
    'mfcc_delta_mean',
    'chroma_variance',
    'chroma_entropy',
    'rms_mean',
    'rms_std',
    'rms_peak',
]


def normalize_features(track_features: Dict) -> np.ndarray:
    """
    Normalize audio features using Z-score normalization.

    Standardizes all features to mean=0, std=1 so no feature dominates.

    Args:
        track_features: Dict of track features (from features table)

    Returns:
        Normalized feature vector as 1D numpy array

    Example:
        >>> features = {'bpm': 128, 'spectral_centroid_mean': 2500, ...}
        >>> normalized = normalize_features(features)
    """
    feature_values = []

    for feature_name in SIMILARITY_FEATURES:
        value = track_features.get(feature_name, 0.0)
        # Handle None values
        if value is None:
            value = 0.0
        feature_values.append(float(value))

    # Create feature vector
    feature_vector = np.array(feature_values, dtype=np.float32).reshape(1, -1)

    # Normalize using StandardScaler
    scaler = StandardScaler()
    normalized = scaler.fit_transform(feature_vector)

    return normalized.flatten()


def compute_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    Returns score in range [0.0, 1.0] where 1.0 is identical.

    Args:
        vector_a: First normalized feature vector
        vector_b: Second normalized feature vector

    Returns:
        Cosine similarity score (0.0-1.0)

    Example:
        >>> v1 = np.array([...])
        >>> v2 = np.array([...])
        >>> score = compute_similarity(v1, v2)
        >>> print(f"Similarity: {score:.2f}")  # 0.87
    """
    # Ensure vectors are 2D for cosine_similarity
    v1 = vector_a.reshape(1, -1) if vector_a.ndim == 1 else vector_a
    v2 = vector_b.reshape(1, -1) if vector_b.ndim == 1 else vector_b

    similarity = cosine_similarity(v1, v2)[0, 0]

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, float(similarity)))


def _apply_bpm_filter(track_bpm: float, bpm_tolerance: Optional[float]) -> bool:
    """Check if track BPM is within tolerance."""
    if bpm_tolerance is None:
        return True
    return True  # Filter applied in find_similar_tracks


def _apply_mood_filter(track_id: int, mood_filter: Optional[str], conn: sqlite3.Connection) -> bool:
    """Check if track mood matches filter."""
    if mood_filter is None:
        return True

    cursor = conn.execute(
        "SELECT * FROM mood WHERE track_id = ?",
        (track_id,)
    )
    mood_row = cursor.fetchone()

    if mood_row is None:
        return False

    mood_dict = dict(mood_row)
    mood_score = mood_dict.get(mood_filter, 0.0)

    return mood_score > 0.3  # Threshold for mood match


def find_similar_tracks(
    track_id: int,
    top_k: int = 5,
    bpm_tolerance: Optional[float] = None,
    mood_filter: Optional[str] = None,
    db_path: str = "db/djia.db"
) -> List[Tuple]:
    """
    Find similar tracks to a given track using cosine similarity.

    Queries database, computes similarity to query track, applies optional filters,
    and returns top-K results sorted by score.

    Args:
        track_id: ID of query track
        top_k: Number of results to return (default 5)
        bpm_tolerance: Optional BPM range tolerance (e.g., 2 for ±2 BPM)
        mood_filter: Optional mood to filter by (e.g., 'hypnotic', 'aggressive')
        db_path: Path to SQLite database

    Returns:
        List of tuples: [(track_dict, similarity_score), ...]
        Sorted by similarity_score descending

    Example:
        >>> matches = find_similar_tracks(
        ...     track_id=42,
        ...     top_k=5,
        ...     bpm_tolerance=2,
        ...     mood_filter="hypnotic"
        ... )
        >>> for track, score in matches:
        ...     print(f"{track['title']}: {score:.2f}")
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        # Get query track features
        cursor = conn.execute(
            """
            SELECT t.id, t.file_name, t.artist, t.title, f.*
            FROM tracks t
            LEFT JOIN features f ON t.id = f.track_id
            WHERE t.id = ?
            """,
            (track_id,)
        )
        query_row = cursor.fetchone()

        if query_row is None:
            raise ValueError(f"Track not found: {track_id}")

        query_dict = dict(query_row)
        query_vector = normalize_features(query_dict)
        query_bpm = query_dict.get('bpm', 128.0)

        # Get all other tracks
        cursor = conn.execute(
            """
            SELECT t.id, t.file_name, t.artist, t.title, f.*
            FROM tracks t
            LEFT JOIN features f ON t.id = f.track_id
            WHERE t.id != ?
            """,
            (track_id,)
        )

        candidates = []

        for row in cursor.fetchall():
            track_dict = dict(row)

            # Apply BPM filter
            if bpm_tolerance is not None:
                track_bpm = track_dict.get('bpm', 128.0)
                if abs(track_bpm - query_bpm) > bpm_tolerance:
                    continue

            # Apply mood filter
            if mood_filter is not None:
                if not _apply_mood_filter(track_dict['id'], mood_filter, conn):
                    continue

            # Compute similarity
            track_vector = normalize_features(track_dict)
            similarity_score = compute_similarity(query_vector, track_vector)

            candidates.append((track_dict, similarity_score))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return candidates[:top_k]

    finally:
        conn.close()


def batch_normalize_all_tracks(db_path: str = "db/djia.db") -> Dict[int, np.ndarray]:
    """
    Pre-compute and cache normalized feature vectors for all tracks.

    Useful for speeding up multiple similarity queries.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary mapping track_id to normalized feature vector
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute(
            """
            SELECT t.id, f.*
            FROM tracks t
            LEFT JOIN features f ON t.id = f.track_id
            """
        )

        vectors = {}
        for row in cursor.fetchall():
            track_dict = dict(row)
            track_id = track_dict['id']
            vector = normalize_features(track_dict)
            vectors[track_id] = vector

        return vectors

    finally:
        conn.close()
