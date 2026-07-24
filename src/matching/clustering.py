"""Hierarchical (agglomerative) clustering of the track library.

Groups similar tracks by cutting a dendrogram built over the SAME corpus-normalized
feature vectors the similarity search uses (`similarity.SIMILARITY_FEATURES`, which
now includes the model-free stem-proxy features). Average linkage over cosine
distance is the natural partner for the cosine similarity the rest of the system is
built on, and needs no ``k`` chosen up front — the tree is cut either into a fixed
number of clusters or at a cosine-distance threshold.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .similarity import batch_normalize_all_tracks

logger = logging.getLogger(__name__)

MOOD_DIMENSIONS = (
    "dark",
    "hypnotic",
    "euphoric",
    "aggressive",
    "industrial",
    "minimal",
)

DEFAULT_DISTANCE_THRESHOLD = 0.25


def cluster_library(
    db_path: str = "db/djia.db",
    method: str = "average",
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> Dict[int, int]:
    """Hierarchically cluster every track by its normalized feature vector.

    Args:
        db_path: Path to the SQLite database.
        method: scipy linkage method ('average' pairs best with cosine distance;
            'ward' is intentionally not supported here since it requires Euclidean
            raw observations, not a cosine distance matrix).
        n_clusters: Cut the tree into exactly this many clusters (maxclust).
        distance_threshold: Cut the tree wherever the cosine distance exceeds this
            (distance criterion). Mutually exclusive with ``n_clusters``; when both
            are omitted, defaults to ``DEFAULT_DISTANCE_THRESHOLD``.

    Returns:
        Dict mapping ``track_id -> cluster_label`` (labels are 1-based ints).
        Empty dict if the library has no tracks.

    Raises:
        FileNotFoundError: if the database does not exist.
        ValueError: if both ``n_clusters`` and ``distance_threshold`` are given.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if n_clusters is not None and distance_threshold is not None:
        raise ValueError("Pass either n_clusters or distance_threshold, not both")
    if n_clusters is None and distance_threshold is None:
        distance_threshold = DEFAULT_DISTANCE_THRESHOLD

    vectors = batch_normalize_all_tracks(db_path)
    if not vectors:
        return {}

    track_ids = sorted(vectors)
    n = len(track_ids)
    if n == 1:
        return {track_ids[0]: 1}

    matrix = np.array([vectors[tid] for tid in track_ids], dtype=np.float64)

    # Cosine distance is undefined for all-zero rows (yields nan) and can produce
    # tiny negatives from float error — sanitize before linkage so scipy doesn't
    # choke or emit a degenerate tree.
    dists = pdist(matrix, metric="cosine")
    dists = np.nan_to_num(dists, nan=1.0, posinf=1.0, neginf=0.0)
    dists = np.clip(dists, 0.0, None)

    linkage_matrix = linkage(dists, method=method)

    if n_clusters is not None:
        k = max(1, min(int(n_clusters), n))
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    else:
        labels = fcluster(linkage_matrix, t=float(distance_threshold), criterion="distance")

    return {tid: int(label) for tid, label in zip(track_ids, labels)}


def _mean(values) -> Optional[float]:
    """Mean of the non-None values, or None if there are none."""
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _std(values) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return float(np.std(nums)) if nums else None


def _modal_key(values) -> Optional[str]:
    keys = [v for v in values if v]
    if not keys:
        return None
    counts: Dict[str, int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get)


def _dominant_mood(members: List[Dict]) -> Optional[str]:
    best_dim, best_val = None, None
    for dim in MOOD_DIMENSIONS:
        avg = _mean([m.get(dim) for m in members])
        if avg is not None and (best_val is None or avg > best_val):
            best_dim, best_val = dim, avg
    return best_dim


def describe_clusters(
    labels: Dict[int, int], db_path: str = "db/djia.db"
) -> List[Dict]:
    """Summarize each cluster for human display.

    Args:
        labels: ``track_id -> cluster_label`` mapping from ``cluster_library``.
        db_path: Path to the SQLite database.

    Returns:
        List of per-cluster summary dicts (size, BPM mean/spread, modal Camelot key,
        dominant mood, mean sub_ratio/vocal_presence, example track names), sorted by
        cluster size descending.
    """
    if not labels:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = {
            row["id"]: dict(row)
            for row in conn.execute(
                """
                SELECT t.id, t.file_name, t.artist, t.title,
                       f.bpm, f.camelot_key, f.sub_ratio, f.bass_ratio,
                       f.kick_rate, f.perc_rate, f.hat_rate, f.vocal_presence,
                       m.dark, m.hypnotic, m.euphoric, m.aggressive,
                       m.industrial, m.minimal
                FROM tracks t
                LEFT JOIN features f ON t.id = f.track_id
                LEFT JOIN mood m ON t.id = m.track_id
                """
            )
        }
    finally:
        conn.close()

    grouped: Dict[int, List[int]] = {}
    for track_id, label in labels.items():
        grouped.setdefault(label, []).append(track_id)

    summaries = []
    for label, ids in grouped.items():
        members = [rows[i] for i in ids if i in rows]
        example_names = [
            (m.get("title") or m.get("file_name") or f"track {m['id']}")
            for m in members[:3]
        ]
        summaries.append(
            {
                "cluster": label,
                "size": len(ids),
                "bpm_mean": _mean([m.get("bpm") for m in members]),
                "bpm_std": _std([m.get("bpm") for m in members]),
                "camelot_key": _modal_key([m.get("camelot_key") for m in members]),
                "dominant_mood": _dominant_mood(members),
                "sub_ratio_mean": _mean([m.get("sub_ratio") for m in members]),
                "vocal_presence_mean": _mean([m.get("vocal_presence") for m in members]),
                "examples": example_names,
            }
        )

    summaries.sort(key=lambda s: s["size"], reverse=True)
    return summaries
