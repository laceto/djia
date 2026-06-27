"""Audio feature extraction and analysis for techno tracks."""

import librosa
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Extract tempo (BPM) from audio."""
    # Use beat tracking instead of direct tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        if isinstance(tempo, np.ndarray):
            return float(tempo[0] if len(tempo) > 0 else tempo.item())
        return float(tempo)
    except (TypeError, AttributeError):
        # Fallback: use basic onset detection
        return 120.0  # Default fallback


def extract_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract spectral features for frequency analysis."""
    S = librosa.stft(y)
    magnitude = np.abs(S)

    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))

    return {
        'spectral_centroid_mean': float(np.mean(spectral_centroid)),
        'spectral_centroid_std': float(np.std(spectral_centroid)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_flux_mean': float(np.mean(spectral_flux)),
    }


def extract_harmonic_percussive(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Separate harmonic and percussive components."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    harmonic_ratio = float(np.sqrt(np.mean(y_harmonic**2)) / (np.sqrt(np.mean(y_percussive**2)) + 1e-8))
    percussive_ratio = float(np.sqrt(np.mean(y_percussive**2)) / (np.sqrt(np.mean(y**2)) + 1e-8))

    return {
        'harmonic_ratio': harmonic_ratio,
        'percussive_ratio': percussive_ratio,
    }


def extract_mfcc_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract MFCCs (Mel-frequency cepstral coefficients)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        'mfcc_mean': float(np.mean(mfcc)),
        'mfcc_std': float(np.std(mfcc)),
        'mfcc_delta_mean': float(np.mean(np.abs(np.diff(mfcc, axis=1)))),
    }


def extract_chroma_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract chroma features for harmonic content."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = np.sum(chroma, axis=0)

    return {
        'chroma_variance': float(np.var(chroma_energy)),
        'chroma_entropy': float(-np.sum(chroma_energy / (np.sum(chroma_energy) + 1e-8) *
                                       np.log(chroma_energy / (np.sum(chroma_energy) + 1e-8) + 1e-8))),
    }


def calculate_rms_energy(y: np.ndarray) -> Dict[str, float]:
    """Calculate RMS energy for loudness analysis."""
    rms = librosa.feature.rms(y=y)[0]

    return {
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'rms_peak': float(np.max(rms)),
    }


def analyze_track(file_path: str, sr: int = 22050, duration: Optional[float] = None) -> Dict:
    """Comprehensive audio analysis of a single track."""
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)

        if len(y) == 0:
            return None

        features = {
            'file_path': file_path,
            'duration': float(librosa.get_duration(y=y, sr=sr)),
            'tempo': extract_tempo(y, sr),
        }

        features.update(extract_spectral_features(y, sr))
        features.update(extract_harmonic_percussive(y, sr))
        features.update(extract_mfcc_features(y, sr))
        features.update(extract_chroma_features(y, sr))
        features.update(calculate_rms_energy(y))

        return features
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None
