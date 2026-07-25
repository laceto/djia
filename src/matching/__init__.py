"""Track similarity matching module."""

from .similarity import (
    fit_scaler,
    transform_one,
    compute_similarity,
    camelot_penalty,
    bpm_penalty,
    find_similar_tracks,
)

__all__ = [
    'fit_scaler',
    'transform_one',
    'compute_similarity',
    'camelot_penalty',
    'bpm_penalty',
    'find_similar_tracks',
]
