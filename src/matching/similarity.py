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
    # Model-free stem-proxy features (src/dsp/stem_profile.py): low-end weight,
    # the kick/perc/hat transient split, and vocal presence. Older DBs predating
    # these columns store NULL -> imputed to the corpus mean by _raw_row.
    'sub_ratio',
    'bass_ratio',
    'kick_rate',
    'perc_rate',
    'hat_rate',
    'vocal_presence',
]


def _raw_row(track_features: Dict, corpus_means: Optional[List[float]] = None) -> List[float]:
    """
    Build one raw (unscaled) SIMILARITY_FEATURES row, imputing None as the
    corpus mean for that column (or 0.0 if no corpus_means given yet, e.g.
    while the means themselves are being computed).
    """
    row = []
    for i, feature_name in enumerate(SIMILARITY_FEATURES):
        value = track_features.get(feature_name)
        if value is None:
            value = corpus_means[i] if corpus_means is not None else 0.0
        row.append(float(value))
    return row


def fit_scaler(all_track_features: List[Dict]) -> Tuple[StandardScaler, List[float]]:
    """
    Fit a StandardScaler across an entire corpus of tracks (one row per
    track, one column per SIMILARITY_FEATURES entry).

    Normalization must be computed across the population, not per-track:
    fitting a scaler on a single track's row (the previous behavior) leaves
    each column with a single value, i.e. zero variance, so sklearn returns
    an all-zero vector for every track regardless of its actual features.

    None/missing feature values are imputed as that column's corpus mean
    (computed over the tracks that do have a value), not 0.0 — imputing 0.0
    would treat "unknown" as a specific, usually-extreme raw value instead of
    "average", distorting every other track's similarity ranking.

    Args:
        all_track_features: dicts of raw feature values, keyed by
            SIMILARITY_FEATURES names (e.g. rows from the `features` table)

    Returns:
        (fitted StandardScaler, corpus_means) — corpus_means is the
        per-column mean used for imputation, needed again by transform_one
        for rows not included in the fit (e.g. a query track fit alongside
        its candidates already covers this, but callers that transform a
        fresh row later need the same imputation values).
    """
    n_features = len(SIMILARITY_FEATURES)
    sums = [0.0] * n_features
    counts = [0] * n_features

    for track_features in all_track_features:
        for i, feature_name in enumerate(SIMILARITY_FEATURES):
            value = track_features.get(feature_name)
            if value is not None:
                sums[i] += float(value)
                counts[i] += 1

    corpus_means = [sums[i] / counts[i] if counts[i] > 0 else 0.0 for i in range(n_features)]

    matrix = np.array(
        [_raw_row(tf, corpus_means) for tf in all_track_features],
        dtype=np.float64,
    )

    scaler = StandardScaler()
    scaler.fit(matrix)

    return scaler, corpus_means


def transform_one(
    scaler: StandardScaler, track_features: Dict, corpus_means: List[float]
) -> np.ndarray:
    """Transform a single track's features into the corpus-normalized vector space."""
    row = np.array([_raw_row(track_features, corpus_means)], dtype=np.float64)
    return scaler.transform(row).flatten()


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


def camelot_penalty(key_a: Optional[str], key_b: Optional[str]) -> float:
    """
    Camelot-wheel key-compatibility penalty (0-1; 0.5 when either code is
    missing/unparseable), applied multiplicatively on top of the audio
    cosine similarity. Adapted from setlist_generator.camelot_score's tiers.
    """
    def parse(code):
        if not code:
            return None
        s = str(code).strip().upper()
        if len(s) < 2 or s[-1] not in "AB" or not s[:-1].isdigit():
            return None
        n = int(s[:-1])
        return (n, s[-1]) if 1 <= n <= 12 else None

    pa, pb = parse(key_a), parse(key_b)
    if pa is None or pb is None:
        return 0.5
    (na, la), (nb, lb) = pa, pb
    dist = min(abs(na - nb), 12 - abs(na - nb))
    if na == nb and la == lb:
        return 1.00
    if na == nb:
        return 0.90  # relative major/minor
    if dist == 1 and la == lb:
        return 0.90  # adjacent, same mode
    if dist == 2 and la == lb:
        return 0.70
    if dist == 1:
        return 0.60  # diagonal neighbour
    return 0.30


def bpm_penalty(bpm_a: Optional[float], bpm_b: Optional[float]) -> float:
    """
    BPM-closeness penalty (0-1), applied multiplicatively on top of the
    audio cosine similarity. Adapted from
    transition_mapper._calculate_bpm_compatibility's falloff tiers.
    """
    if not bpm_a or not bpm_b:
        return 0.5

    diff = abs(bpm_a - bpm_b)
    if diff < 0.5:
        return 1.0
    pct = diff / max(bpm_a, bpm_b)
    if pct < 0.02:
        return 0.95
    if pct < 0.05:
        return 0.85
    if pct < 0.10:
        return 0.65
    if pct < 0.20:
        return 0.40
    return 0.15


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
    Find similar tracks to a given track using cosine similarity over
    corpus-normalized audio features, softly penalized (not hard-filtered)
    by Camelot key distance and BPM difference.

    Queries database, computes similarity to query track, applies optional
    pre-filters, and returns top-K results sorted by score.

    Args:
        track_id: ID of query track
        top_k: Number of results to return (default 5)
        bpm_tolerance: Optional BPM range pre-filter (e.g., 2 for ±2 BPM) —
            candidates outside this range are excluded before scoring, in
            addition to (not instead of) the BPM penalty applied to survivors
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
        query_bpm = query_dict.get('bpm')

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

        rows = [dict(row) for row in cursor.fetchall()]

        filtered = []
        for track_dict in rows:
            if bpm_tolerance is not None:
                track_bpm = track_dict.get('bpm')
                if track_bpm is None or query_bpm is None or abs(track_bpm - query_bpm) > bpm_tolerance:
                    continue

            if mood_filter is not None:
                if not _apply_mood_filter(track_dict['id'], mood_filter, conn):
                    continue

            filtered.append(track_dict)

        if not filtered:
            return []

        # Fit normalization across the query track + surviving candidates —
        # cosine similarity is meaningless without a population to scale against.
        scaler, corpus_means = fit_scaler([query_dict] + filtered)
        query_vector = transform_one(scaler, query_dict, corpus_means)

        candidates = []
        for track_dict in filtered:
            track_vector = transform_one(scaler, track_dict, corpus_means)
            base_score = compute_similarity(query_vector, track_vector)

            score = (
                base_score
                * camelot_penalty(query_dict.get('camelot_key'), track_dict.get('camelot_key'))
                * bpm_penalty(query_bpm, track_dict.get('bpm'))
            )

            candidates.append((track_dict, score))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        return candidates[:top_k]

    finally:
        conn.close()


def batch_normalize_all_tracks(db_path: str = "db/djia.db") -> Dict[int, np.ndarray]:
    """
    Pre-compute and cache normalized feature vectors for all tracks, fit
    across the whole corpus in one pass.

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

        rows = [dict(row) for row in cursor.fetchall()]
        if not rows:
            return {}

        scaler, corpus_means = fit_scaler(rows)

        return {
            row['id']: transform_one(scaler, row, corpus_means)
            for row in rows
        }

    finally:
        conn.close()
