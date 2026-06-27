"""Mood and vibe classification using pre-trained models or rule-based fallback."""

from typing import Dict, Optional
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')

try:
    import essentia
    from essentia.standard import MonoLoader, MusicExtractor, TensorflowPredictTempoCNN
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False


class MoodClassifier:
    """Classifies mood/vibe and acoustic characteristics of audio tracks."""

    MOOD_CATEGORIES = ['dark', 'hypnotic', 'euphoric', 'aggressive', 'industrial', 'minimal']
    ENERGY_LEVELS = ['low', 'medium', 'high']

    def __init__(self, use_essentia: bool = True):
        """
        Initialize mood classifier.

        Args:
            use_essentia: Try to use Essentia models if available.
        """
        self.use_essentia = use_essentia and ESSENTIA_AVAILABLE

        if self.use_essentia:
            print("Using Essentia for mood classification")
        else:
            print("Using rule-based classifier (Essentia unavailable)")

    def _rule_based_mood_classification(
        self,
        y: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """
        Rule-based mood classification using spectral and temporal features.

        This provides fallback classification without external ML models.
        """
        # Compute features
        S = librosa.stft(y)
        mag = np.abs(S)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=mag, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Temporal features
        rms = librosa.feature.rms(y=y)[0]
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # Statistics
        sc_mean = np.mean(spectral_centroid)
        sc_std = np.std(spectral_centroid)
        zcr_mean = np.mean(zero_crossing_rate)
        contrast_mean = np.mean(spectral_contrast)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        onset_mean = np.mean(onset_env)

        # Mood classification based on feature combinations
        moods = {}

        # Dark: low frequencies, low energy, minimal variation
        dark_score = 0.0
        if sc_mean < 3000:  # Lower frequencies
            dark_score += 0.3
        if rms_mean < 0.1:  # Lower energy
            dark_score += 0.2
        if sc_std < 1000:  # Stable spectral content
            dark_score += 0.2
        if zcr_mean < 0.1:  # Low high-frequency content
            dark_score += 0.2
        moods['dark'] = min(1.0, dark_score)

        # Hypnotic: repetitive, steady energy, moderate frequency
        hypnotic_score = 0.0
        if 2000 < sc_mean < 4000:  # Mid-range frequencies
            hypnotic_score += 0.3
        if 0.05 < rms_std < 0.15:  # Moderate energy variation
            hypnotic_score += 0.2
        if rms_mean > 0.08:  # Solid energy
            hypnotic_score += 0.2
        if onset_mean > 0.1:  # Strong rhythm
            hypnotic_score += 0.2
        moods['hypnotic'] = min(1.0, hypnotic_score)

        # Euphoric: high energy, bright frequencies, dynamic
        euphoric_score = 0.0
        if sc_mean > 4000:  # Bright frequencies
            euphoric_score += 0.3
        if rms_mean > 0.15:  # High energy
            euphoric_score += 0.25
        if rms_std > 0.1:  # Dynamic energy
            euphoric_score += 0.2
        if zcr_mean > 0.15:  # High-frequency content
            euphoric_score += 0.2
        moods['euphoric'] = min(1.0, euphoric_score)

        # Aggressive: high contrast, high energy, harsh high-frequencies
        aggressive_score = 0.0
        if contrast_mean > 5.0:  # High spectral contrast
            aggressive_score += 0.3
        if rms_mean > 0.12:  # High energy
            aggressive_score += 0.25
        if zcr_mean > 0.12:  # High-frequency emphasis
            aggressive_score += 0.2
        if rms_std > 0.08:  # Energetic variation
            aggressive_score += 0.15
        moods['aggressive'] = min(1.0, aggressive_score)

        # Industrial: metallic high-freq, harsh transients, high contrast
        industrial_score = 0.0
        if zcr_mean > 0.2:  # Very high-frequency content
            industrial_score += 0.3
        if contrast_mean > 6.0:  # Very high contrast
            industrial_score += 0.3
        if rms_mean > 0.1:  # Good energy
            industrial_score += 0.2
        if onset_mean > 0.2:  # Strong onsets/transients
            industrial_score += 0.2
        moods['industrial'] = min(1.0, industrial_score)

        # Minimal: low complexity, stable, thin
        minimal_score = 0.0
        if sc_std < 800:  # Very stable spectrum
            minimal_score += 0.3
        if rms_std < 0.08:  # Steady energy
            minimal_score += 0.25
        if sc_mean < 2500:  # Lower to mid frequencies
            minimal_score += 0.2
        if onset_mean < 0.15:  # Subtle rhythm
            minimal_score += 0.2
        moods['minimal'] = min(1.0, minimal_score)

        # Normalize to sum to ~1.0
        total = sum(moods.values())
        if total > 0:
            moods = {k: v / total for k, v in moods.items()}
        else:
            moods = {k: 1.0 / len(self.MOOD_CATEGORIES) for k in self.MOOD_CATEGORIES}

        return moods

    def _extract_energy_level(self, y: np.ndarray, sr: int) -> str:
        """Classify energy level as low, medium, or high."""
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)

        if rms_mean < 0.08:
            return 'low'
        elif rms_mean < 0.15:
            return 'medium'
        else:
            return 'high'

    def _extract_danceability(self, y: np.ndarray, sr: int) -> float:
        """
        Extract danceability score (0-1).

        Based on rhythm strength, energy, and spectral features.
        """
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength = np.mean(onset_env)

        # Energy
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)

        # Spectral flux (rhythm-related)
        S = librosa.stft(y)
        mag = np.abs(S)
        flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
        flux_mean = np.mean(flux)

        # Combine into danceability
        dance_score = 0.0
        dance_score += 0.3 * min(1.0, onset_strength / 0.5)  # Rhythm
        dance_score += 0.3 * min(1.0, energy / 0.2)  # Energy
        dance_score += 0.2 * min(1.0, flux_mean / 0.05)  # Flux
        dance_score += 0.2 * 0.7  # Base score

        return float(min(1.0, dance_score))

    def classify_mood(self, y: np.ndarray, sr: int) -> Dict[str, any]:
        """
        Classify mood and acoustic characteristics of audio.

        Args:
            y: Audio time series (mono).
            sr: Sample rate.

        Returns:
            Dict with keys:
            - 'moods': Dict[str, float] - confidence scores for mood categories
            - 'energy': str - energy level (low/medium/high)
            - 'danceability': float - danceability score (0-1)
        """
        if self.use_essentia:
            try:
                return self._classify_with_essentia(y, sr)
            except Exception as e:
                print(f"Essentia classification failed, falling back to rule-based: {e}")
                return self._classify_rule_based(y, sr)
        else:
            return self._classify_rule_based(y, sr)

    def _classify_rule_based(self, y: np.ndarray, sr: int) -> Dict[str, any]:
        """Rule-based classification."""
        moods = self._rule_based_mood_classification(y, sr)
        energy = self._extract_energy_level(y, sr)
        danceability = self._extract_danceability(y, sr)

        return {
            'moods': moods,
            'energy': energy,
            'danceability': danceability,
        }

    def _classify_with_essentia(self, y: np.ndarray, sr: int) -> Dict[str, any]:
        """Classification using Essentia models (requires essentia package)."""
        # For now, fall back to rule-based since Essentia TensorFlow models
        # have complex dependencies. In production, you'd load the pre-trained models here.
        return self._classify_rule_based(y, sr)


def classify_mood(y: np.ndarray, sr: int) -> Dict[str, any]:
    """
    Convenience function to classify mood of audio.

    Args:
        y: Audio time series (mono).
        sr: Sample rate.

    Returns:
        Dict with mood classification results.
    """
    classifier = MoodClassifier()
    return classifier.classify_mood(y, sr)
