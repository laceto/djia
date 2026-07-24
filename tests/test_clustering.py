"""Tests for hierarchical clustering of the track library."""

import numpy as np
import pytest

from src.database.schema import init_db
from src.matching.clustering import cluster_library, describe_clusters
from src.matching.similarity import SIMILARITY_FEATURES


# Two well-separated archetypes across the full feature vector so average-linkage
# cosine clustering has an unambiguous 2-way cut.
_GROUP_A = {  # bright, vocal, minimal low-end
    "bpm": 128.0, "spectral_centroid_mean": 4000.0, "spectral_centroid_std": 200.0,
    "spectral_rolloff_mean": 9000.0, "spectral_flux_mean": 0.08, "harmonic_ratio": 0.7,
    "percussive_ratio": 0.3, "mfcc_mean": 60.0, "mfcc_std": 6.0, "mfcc_delta_mean": 1.5,
    "chroma_variance": 0.6, "chroma_entropy": 2.6, "rms_mean": 0.12, "rms_std": 0.03,
    "rms_peak": 0.4, "sub_ratio": 0.10, "bass_ratio": 0.25, "kick_rate": 2.0,
    "perc_rate": 5.0, "hat_rate": 8.0, "vocal_presence": 0.35,
}
_GROUP_B = {  # dark, sub-heavy, instrumental
    "bpm": 134.0, "spectral_centroid_mean": 1500.0, "spectral_centroid_std": 90.0,
    "spectral_rolloff_mean": 4000.0, "spectral_flux_mean": 0.03, "harmonic_ratio": 0.85,
    "percussive_ratio": 0.15, "mfcc_mean": 30.0, "mfcc_std": 3.0, "mfcc_delta_mean": 0.6,
    "chroma_variance": 0.3, "chroma_entropy": 2.0, "rms_mean": 0.2, "rms_std": 0.05,
    "rms_peak": 0.6, "sub_ratio": 0.45, "bass_ratio": 0.42, "kick_rate": 3.2,
    "perc_rate": 1.5, "hat_rate": 1.0, "vocal_presence": 0.02,
}

_COLUMNS = list(_GROUP_A.keys())


def _insert(conn, title, base, rng):
    conn.execute(
        "INSERT INTO tracks (file_path, file_name, format, duration, title) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"{title}.wav", f"{title}.wav", "wav", 300.0, title),
    )
    track_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    # Small per-track jitter so members aren't identical but stay in-group.
    values = [track_id] + [
        base[col] * (1.0 + rng.standard_normal() * 0.02) for col in _COLUMNS
    ]
    placeholders = ", ".join(["?"] * (len(_COLUMNS) + 1))
    conn.execute(
        f"INSERT INTO features (track_id, {', '.join(_COLUMNS)}) "
        f"VALUES ({placeholders})",
        values,
    )
    return track_id


@pytest.fixture
def two_group_db(tmp_path):
    """DB with two clearly-separated groups of 3 tracks each."""
    db_path = tmp_path / "cluster_test.db"
    conn = init_db(str(db_path))
    rng = np.random.default_rng(42)
    ids = {"A": [], "B": []}
    for i in range(3):
        ids["A"].append(_insert(conn, f"A{i}", _GROUP_A, rng))
    for i in range(3):
        ids["B"].append(_insert(conn, f"B{i}", _GROUP_B, rng))
    conn.commit()
    conn.close()
    return str(db_path), ids


class TestClusterLibrary:
    def test_similarity_features_include_stem_proxies(self):
        for key in ("sub_ratio", "bass_ratio", "kick_rate", "perc_rate",
                    "hat_rate", "vocal_presence"):
            assert key in SIMILARITY_FEATURES

    def test_two_clusters_separate_the_groups(self, two_group_db):
        db_path, ids = two_group_db
        labels = cluster_library(db_path=db_path, n_clusters=2)

        assert len(labels) == 6
        a_labels = {labels[i] for i in ids["A"]}
        b_labels = {labels[i] for i in ids["B"]}
        # Each group internally consistent...
        assert len(a_labels) == 1
        assert len(b_labels) == 1
        # ...and distinct from the other group.
        assert a_labels != b_labels

    def test_n_clusters_respected(self, two_group_db):
        db_path, _ = two_group_db
        labels = cluster_library(db_path=db_path, n_clusters=3)
        assert len(set(labels.values())) == 3

    def test_distance_threshold_path(self, two_group_db):
        db_path, _ = two_group_db
        labels = cluster_library(db_path=db_path, distance_threshold=0.5)
        assert len(labels) == 6
        assert all(isinstance(v, int) for v in labels.values())

    def test_default_cut_when_no_args(self, two_group_db):
        db_path, _ = two_group_db
        labels = cluster_library(db_path=db_path)
        assert len(labels) == 6

    def test_mutually_exclusive_args_raise(self, two_group_db):
        db_path, _ = two_group_db
        with pytest.raises(ValueError):
            cluster_library(db_path=db_path, n_clusters=2, distance_threshold=0.3)

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cluster_library(db_path=str(tmp_path / "nope.db"))

    def test_empty_db_returns_empty(self, tmp_path):
        db_path = tmp_path / "empty.db"
        init_db(str(db_path)).close()
        assert cluster_library(db_path=str(db_path)) == {}

    def test_single_track(self, tmp_path):
        db_path = tmp_path / "one.db"
        conn = init_db(str(db_path))
        rng = np.random.default_rng(1)
        tid = _insert(conn, "solo", _GROUP_A, rng)
        conn.commit()
        conn.close()
        labels = cluster_library(db_path=str(db_path), n_clusters=2)
        assert labels == {tid: 1}


class TestDescribeClusters:
    def test_describe_returns_summaries(self, two_group_db):
        db_path, _ = two_group_db
        labels = cluster_library(db_path=db_path, n_clusters=2)
        summaries = describe_clusters(labels, db_path=db_path)

        assert len(summaries) == 2
        assert {s["size"] for s in summaries} == {3}
        # Sorted by size descending; each carries the display fields.
        for s in summaries:
            assert s["bpm_mean"] is not None
            assert "dominant_mood" in s
            assert "examples" in s and len(s["examples"]) <= 3

    def test_describe_empty_labels(self, two_group_db):
        db_path, _ = two_group_db
        assert describe_clusters({}, db_path=db_path) == []

    def test_sub_heavy_group_has_higher_sub_ratio(self, two_group_db):
        db_path, ids = two_group_db
        labels = cluster_library(db_path=db_path, n_clusters=2)
        summaries = describe_clusters(labels, db_path=db_path)
        by_label = {s["cluster"]: s for s in summaries}
        b_label = labels[ids["B"][0]]  # sub-heavy group
        a_label = labels[ids["A"][0]]  # bright group
        assert by_label[b_label]["sub_ratio_mean"] > by_label[a_label]["sub_ratio_mean"]
