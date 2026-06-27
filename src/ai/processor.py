"""Orchestrator for Phase 3 AI features - coordinates all AI analysis."""

from typing import Dict, Optional, Any
from pathlib import Path
import numpy as np
import librosa
import warnings

from src.ai.stem_separator import StemSeparator
from src.ai.classifier import MoodClassifier
from src.ai.segmentation import StructureSegmenter, StructurePoint

warnings.filterwarnings('ignore')


class AIProcessor:
    """Orchestrates stem separation, mood classification, and structural analysis."""

    def __init__(
        self,
        stem_model: str = 'htdemucs',
        cache_stems: bool = True,
        normalize_stems: bool = True
    ):
        """
        Initialize AI processor.

        Args:
            stem_model: Demucs model for stem separation.
            cache_stems: Whether to cache separated stems.
            normalize_stems: Whether to normalize stem loudness.
        """
        self.stem_separator = StemSeparator(model=stem_model)
        self.mood_classifier = MoodClassifier()
        self.segmenter = StructureSegmenter()
        self.cache_stems = cache_stems
        self.normalize_stems = normalize_stems

    def _load_audio(self, audio_path: str, sr: int = 22050) -> tuple:
        """Load audio file."""
        y, sr_loaded = librosa.load(audio_path, sr=sr, mono=True)
        return y, sr_loaded

    def _analyze_stem(
        self,
        stem_audio: np.ndarray,
        sr: int,
        stem_name: str
    ) -> Dict[str, Any]:
        """Analyze individual stem for characteristics."""
        analysis = {
            'stem_name': stem_name,
            'sample_rate': sr,
            'duration': librosa.get_duration(y=stem_audio, sr=sr),
        }

        # Spectral features
        S = librosa.stft(stem_audio)
        mag = np.abs(S)

        spectral_centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr)[0]

        analysis['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        analysis['spectral_centroid_std'] = float(np.std(spectral_centroid))
        analysis['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

        # Energy
        rms = librosa.feature.rms(y=stem_audio)[0]
        analysis['rms_mean'] = float(np.mean(rms))
        analysis['rms_std'] = float(np.std(rms))
        analysis['rms_peak'] = float(np.max(rms))

        # Rhythm (for drums especially)
        onset_env = librosa.onset.onset_strength(y=stem_audio, sr=sr)
        analysis['onset_strength'] = float(np.mean(onset_env))

        return analysis

    def process_with_stems(
        self,
        audio_path: str,
        features_dict: Optional[Dict[str, Any]] = None,
        sr: int = 22050,
        analyze_drums_tempo: bool = True,
        analyze_bass_key: bool = True
    ) -> Dict[str, Any]:
        """
        Complete AI processing pipeline with stem analysis.

        Args:
            audio_path: Path to audio file.
            features_dict: Optional Phase 2 DSP features to enhance.
            sr: Sample rate for audio loading.
            analyze_drums_tempo: Extract BPM from drums stem if available.
            analyze_bass_key: Extract key from bass stem if available.

        Returns:
            Enhanced features dict with:
            - stems_data: Individual stem analyses
            - mood_classification: Mood scores and energy
            - structural_landmarks: Detected sections
            - enhanced_features: Original features merged with AI results
        """
        print(f"Processing {Path(audio_path).name} with Phase 3 AI...")
        result = {'audio_path': audio_path}

        # Initialize features dict
        if features_dict is None:
            features_dict = {}

        # 1. Separate stems
        print("  → Separating stems...")
        try:
            stems = self.stem_separator.separate_stems(
                audio_path,
                sr=sr,
                use_cache=self.cache_stems,
                normalize=self.normalize_stems
            )
            result['stems_separated'] = True
        except Exception as e:
            print(f"  Warning: Stem separation failed: {e}")
            stems = None
            result['stems_separated'] = False

        # 2. Load full audio for reference
        y_full, sr_loaded = self._load_audio(audio_path, sr=sr)
        result['full_audio'] = {
            'duration': librosa.get_duration(y=y_full, sr=sr_loaded),
            'sample_rate': sr_loaded,
        }

        # 3. Analyze individual stems
        stems_data = {}
        if stems is not None:
            print("  → Analyzing stems...")
            for stem_name, stem_audio in stems.items():
                try:
                    # Convert stereo to mono if needed
                    if stem_audio.ndim > 1:
                        stem_audio_mono = librosa.to_mono(stem_audio)
                    else:
                        stem_audio_mono = stem_audio

                    stem_analysis = self._analyze_stem(stem_audio_mono, sr_loaded, stem_name)
                    stems_data[stem_name] = stem_analysis

                    # Optional: Extract BPM from drums
                    if analyze_drums_tempo and stem_name == 'drums':
                        try:
                            onset_env = librosa.onset.onset_strength(y=stem_audio_mono, sr=sr_loaded)
                            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr_loaded)
                            if isinstance(tempo, np.ndarray):
                                tempo = float(tempo[0])
                            stems_data[stem_name]['tempo_bpm'] = float(tempo)
                            print(f"    Drums BPM: {tempo:.1f}")
                        except Exception as e:
                            print(f"    Warning: Could not extract drums tempo: {e}")

                    # Optional: Extract key from bass
                    if analyze_bass_key and stem_name == 'bass':
                        try:
                            chroma = librosa.feature.chroma_cqt(y=stem_audio_mono, sr=sr_loaded)
                            chroma_mean = np.mean(chroma, axis=1)
                            key_idx = np.argmax(chroma_mean)
                            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                            key = keys[key_idx]
                            stems_data[stem_name]['detected_key'] = key
                            print(f"    Bass Key: {key}")
                        except Exception as e:
                            print(f"    Warning: Could not extract bass key: {e}")

                except Exception as e:
                    print(f"  Warning: Error analyzing {stem_name} stem: {e}")

        result['stems_data'] = stems_data

        # 4. Mood classification
        print("  → Classifying mood...")
        try:
            mood_result = self.mood_classifier.classify_mood(y_full, sr_loaded)
            result['mood_classification'] = mood_result
            print(f"    Energy: {mood_result['energy']}, Danceability: {mood_result['danceability']:.2f}")
            print(f"    Top mood: {max(mood_result['moods'].items(), key=lambda x: x[1])[0]}")
        except Exception as e:
            print(f"  Warning: Mood classification failed: {e}")
            result['mood_classification'] = None

        # 5. Structural segmentation
        print("  → Detecting structure...")
        try:
            # Use drums stem if available for structure detection
            y_drums = None
            if stems is not None and 'drums' in stems:
                y_drums_raw = stems['drums']
                # Convert stereo to mono if needed
                if y_drums_raw.ndim > 1:
                    y_drums = librosa.to_mono(y_drums_raw)
                else:
                    y_drums = y_drums_raw

            structure_points = self.segmenter.detect_structure(
                y_full,
                sr_loaded,
                y_drums=y_drums
            )

            structure_dicts = [point.to_dict() for point in structure_points]
            result['structural_landmarks'] = structure_dicts
            print(f"    Detected {len(structure_points)} structural points")
            for point in structure_points[:3]:  # Show first 3
                print(f"      {point}")

        except Exception as e:
            print(f"  Warning: Structure detection failed: {e}")
            result['structural_landmarks'] = []

        # 6. Merge with Phase 2 features
        print("  → Merging with Phase 2 features...")
        enhanced_features = features_dict.copy()

        # Add stems analysis to features
        if stems_data:
            enhanced_features['stems'] = stems_data

        # Add mood to features
        if result['mood_classification']:
            enhanced_features['mood'] = result['mood_classification']

        # Add structure to features
        if result['structural_landmarks']:
            enhanced_features['structure'] = result['structural_landmarks']

        result['enhanced_features'] = enhanced_features

        print(f"✓ Phase 3 processing complete\n")
        return result


def process_with_stems(
    audio_path: str,
    features_dict: Optional[Dict[str, Any]] = None,
    sr: int = 22050,
    stem_model: str = 'htdemucs'
) -> Dict[str, Any]:
    """
    Convenience function for Phase 3 AI processing.

    Args:
        audio_path: Path to audio file.
        features_dict: Optional Phase 2 features dict.
        sr: Sample rate.
        stem_model: Demucs model.

    Returns:
        Enhanced features dict with AI results.
    """
    processor = AIProcessor(stem_model=stem_model)
    return processor.process_with_stems(audio_path, features_dict=features_dict, sr=sr)
