"""Mixing quality metrics and scoring system."""

import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler


def normalize_features(features_list: list) -> Dict:
    """Normalize all audio features using StandardScaler."""
    feature_names = [k for k in features_list[0].keys() if k not in ['file_path', 'track_name']]
    
    data = np.array([[f[name] for name in feature_names] for f in features_list])
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data)
    
    normalized_features = []
    for i, f in enumerate(features_list):
        norm_dict = f.copy()
        for j, name in enumerate(feature_names):
            norm_dict[name + '_normalized'] = float(normalized[i, j])
        normalized_features.append(norm_dict)
    
    return normalized_features, feature_names


def calculate_mixing_quality_score(features: Dict) -> float:
    """Calculate overall mixing quality score (0-100)."""
    score = 0.0
    
    if features['rms_peak'] > 0.05:
        score += 15
    if features['rms_std'] > 0.01:
        score += 10
    
    if 120 < features['tempo'] < 135:
        score += 25
    elif 115 < features['tempo'] < 140:
        score += 20
    elif 110 < features['tempo'] < 145:
        score += 15
    
    spectral_balance = abs(features['spectral_centroid_mean'] - 2500)
    if spectral_balance < 1500:
        score += 20
    elif spectral_balance < 2500:
        score += 15
    
    if features['harmonic_ratio'] > 1.0:
        score += 15
    
    if 2 < features['chroma_entropy'] < 3.5:
        score += 15
    
    if features['percussive_ratio'] > 0.4:
        score += 10
    
    if features['spectral_flux_mean'] > 0.01:
        score += 10
    
    if features['mfcc_std'] > 10:
        score += 10
    
    if features['duration'] > 240:
        score += 5
    
    return min(100.0, max(0.0, score))


def calculate_danceability(features: Dict) -> float:
    """Score based on danceability characteristics (0-100)."""
    score = 50.0
    
    if 120 < features['tempo'] < 135:
        score += 30
    elif 110 < features['tempo'] < 140:
        score += 20
    
    if features['percussive_ratio'] > 0.5:
        score += 20
    elif features['percussive_ratio'] > 0.3:
        score += 10
    
    if features['rms_std'] > 0.015:
        score += 10
    
    if features['spectral_flux_mean'] > 0.015:
        score += 10
    
    return min(100.0, max(0.0, score))


def calculate_harmonic_richness(features: Dict) -> float:
    """Score for harmonic content and complexity (0-100)."""
    score = 50.0
    
    if features['harmonic_ratio'] > 1.5:
        score += 25
    elif features['harmonic_ratio'] > 1.0:
        score += 15
    
    if 2.5 < features['chroma_entropy'] < 3.5:
        score += 25
    
    if features['mfcc_std'] > 15:
        score += 15
    elif features['mfcc_std'] > 10:
        score += 10
    
    if features['chroma_variance'] > 0.5:
        score += 10
    
    return min(100.0, max(0.0, score))


def calculate_production_quality(features: Dict) -> float:
    """Score for production/mastering quality (0-100)."""
    score = 50.0
    
    if features['rms_peak'] > 0.08:
        score += 20
    elif features['rms_peak'] > 0.05:
        score += 10
    
    if features['spectral_centroid_std'] < 2000:
        score += 15
    
    if features['rms_std'] > 0.01:
        score += 10
    
    if features['spectral_rolloff_mean'] > 10000:
        score += 15
    
    if features['mfcc_delta_mean'] < 1.5:
        score += 15
    
    return min(100.0, max(0.0, score))


def score_track(features: Dict) -> Dict[str, float]:
    """Generate comprehensive scoring for a track."""
    return {
        'mixing_quality': round(calculate_mixing_quality_score(features), 2),
        'danceability': round(calculate_danceability(features), 2),
        'harmonic_richness': round(calculate_harmonic_richness(features), 2),
        'production_quality': round(calculate_production_quality(features), 2),
    }


def calculate_overall_score(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate weighted overall score."""
    if weights is None:
        weights = {
            'mixing_quality': 0.35,
            'danceability': 0.25,
            'harmonic_richness': 0.25,
            'production_quality': 0.15,
        }
    
    total = sum(scores[k] * weights[k] / 100.0 for k in weights.keys() if k in scores)
    return round(total * 100, 2)
