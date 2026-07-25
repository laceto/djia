"""Stem separation using Demucs with caching for efficiency."""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

try:
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


class StemSeparator:
    """Separates audio tracks into individual stems (Drums, Bass, Vocals, Melody)."""

    STEM_NAMES = ['drums', 'bass', 'vocals', 'melody']
    DEFAULT_MODEL = 'htdemucs'
    CACHE_DIR = Path('results/stems')

    # Demucs 4-stem models (htdemucs, mdx_extra, ...) emit sources named
    # drums/bass/other/vocals. We expose Demucs' "other" (everything that isn't
    # drums/bass/vocals — synths, pads, leads) as this project's "melody" stem.
    DEMUCS_SOURCE_MAP = {
        'drums': 'drums',
        'bass': 'bass',
        'vocals': 'vocals',
        'other': 'melody',
    }

    def __init__(self, cache_dir: Optional[Path] = None, model: str = DEFAULT_MODEL):
        """
        Initialize stem separator.

        Args:
            cache_dir: Directory to cache separated stems. Defaults to results/stems/.
            model: Demucs model to use. Options: 'htdemucs', 'mdx_extra', 'mdx', etc.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.demucs_model = None

        if DEMUCS_AVAILABLE:
            try:
                self.demucs_model = get_model(model)
            except Exception as e:
                print(f"Warning: Could not load Demucs model '{model}': {e}")
                print("Stem separation will be unavailable.")

    def _get_track_hash(self, audio_path: str) -> str:
        """Generate a unique hash for the audio file based on its path."""
        return hashlib.md5(audio_path.encode()).hexdigest()

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get the cache directory path for a track."""
        track_hash = self._get_track_hash(audio_path)
        return self.cache_dir / track_hash

    def _load_cached_stems(self, audio_path: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load stems from cache if available.

        Returns:
            Dict mapping stem names to audio arrays, or None if not cached.
        """
        cache_path = self._get_cache_path(audio_path)
        metadata_file = cache_path / 'metadata.json'

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Verify cache is valid
            if metadata.get('source_path') != audio_path:
                return None

            # Load all stems
            stems = {}
            sr = metadata.get('sr', 16000)

            for stem_name in self.STEM_NAMES:
                stem_file = cache_path / f'{stem_name}.wav'
                if stem_file.exists():
                    y, _ = librosa.load(stem_file, sr=sr, mono=False)
                    stems[stem_name] = y
                else:
                    return None  # Incomplete cache

            print(f"Loaded cached stems for {Path(audio_path).name}")
            return stems

        except Exception as e:
            print(f"Error loading cached stems: {e}")
            return None

    def _save_stems_to_cache(self, audio_path: str, stems: Dict[str, np.ndarray], sr: int) -> None:
        """Save separated stems to cache."""
        cache_path = self._get_cache_path(audio_path)
        cache_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save each stem
            for stem_name, audio_data in stems.items():
                stem_file = cache_path / f'{stem_name}.wav'
                sf.write(str(stem_file), audio_data.T if audio_data.ndim > 1 else audio_data, sr)

            # Save metadata
            metadata = {
                'source_path': audio_path,
                'sr': sr,
                'stems': self.STEM_NAMES,
                'model': self.model,
            }

            with open(cache_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"Error saving stems to cache: {e}")

    def _normalize_stem_loudness(self, stems: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize loudness of each stem to prevent clipping and balance.

        Uses RMS-based normalization to maintain relative loudness balance
        while preventing digital clipping.
        """
        normalized = {}

        for stem_name, audio in stems.items():
            # Calculate RMS
            rms = np.sqrt(np.mean(audio ** 2))

            # Prevent division by zero
            if rms < 1e-8:
                normalized[stem_name] = audio
                continue

            # Normalize to -6dB (approx 0.5 amplitude) to leave headroom
            target_rms = 0.3
            normalized_audio = audio * (target_rms / rms)

            # Soft clip to prevent harshness
            normalized_audio = np.tanh(normalized_audio * 0.9) / 0.9

            normalized[stem_name] = normalized_audio

        return normalized

    def separate_stems(
        self,
        audio_path: str,
        sr: int = 16000,
        use_cache: bool = True,
        normalize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio track into individual stems.

        Args:
            audio_path: Path to audio file.
            sr: Sample rate for processing.
            use_cache: Whether to use cached stems if available.
            normalize: Whether to normalize stem loudness.

        Returns:
            Dict with keys ['drums', 'bass', 'vocals', 'melody'],
            each mapping to audio array of shape (channels, samples).

        Raises:
            RuntimeError: If Demucs is not available and model loading fails.
        """
        # Check cache first
        if use_cache:
            cached_stems = self._load_cached_stems(audio_path)
            if cached_stems is not None:
                if normalize:
                    cached_stems = self._normalize_stem_loudness(cached_stems)
                return cached_stems

        if not DEMUCS_AVAILABLE or self.demucs_model is None:
            print(f"Warning: Demucs unavailable. Returning empty stems.")
            return {name: np.zeros((1, 1)) for name in self.STEM_NAMES}

        try:
            print(f"Separating stems for {Path(audio_path).name} using model '{self.model}'...")

            # Load audio
            waveform, sr_loaded = librosa.load(audio_path, sr=sr, mono=False)

            # Ensure stereo
            if waveform.ndim == 1:
                waveform = np.stack([waveform, waveform], axis=0)

            # Convert to torch tensor and apply model
            import torch
            waveform_tensor = torch.from_numpy(waveform).float()
            if not waveform_tensor.shape[0] == 2:
                # Ensure stereo
                if waveform_tensor.shape[0] == 1:
                    waveform_tensor = waveform_tensor.repeat(2, 1)
                else:
                    # Multiple channels - take first two or mix
                    if waveform_tensor.shape[0] > 2:
                        waveform_tensor = waveform_tensor[:2]
                    else:
                        waveform_tensor = waveform_tensor.repeat(2 // waveform_tensor.shape[0] + 1, 1)[:2]

            # Add batch dimension
            waveform_tensor = waveform_tensor.unsqueeze(0)

            # Apply Demucs model
            with torch.no_grad():
                stems_output = apply_model(self.demucs_model, waveform_tensor)

            # Extract stems (Demucs returns [batch, stems, channels, samples])
            stems_output = stems_output[0].cpu().numpy()  # Remove batch dim

            # Map Demucs output to stem names by the model's OWN source order,
            # not by positional index into STEM_NAMES. htdemucs emits sources as
            # [drums, bass, other, vocals]; indexing STEM_NAMES positionally put
            # Demucs' "other" into "vocals" and the real vocals into "melody".
            model_sources = list(getattr(self.demucs_model, 'sources', []))
            stems = {
                name: np.zeros((2, stems_output.shape[-1]))
                for name in self.STEM_NAMES
            }
            for i, source in enumerate(model_sources):
                if i >= stems_output.shape[0]:
                    break
                stem_name = self.DEMUCS_SOURCE_MAP.get(source, source)
                stems[stem_name] = stems_output[i]

            # Normalize loudness
            if normalize:
                stems = self._normalize_stem_loudness(stems)

            # Cache results
            try:
                self._save_stems_to_cache(audio_path, stems, sr)
            except Exception as e:
                print(f"Warning: Could not cache stems: {e}")

            print(f"Successfully separated {len(stems)} stems")
            return stems

        except Exception as e:
            print(f"Error during stem separation: {e}")
            raise RuntimeError(f"Stem separation failed: {e}")


def separate_stems(
    audio_path: str,
    model: str = 'htdemucs',
    sr: int = 16000,
    use_cache: bool = True,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Convenience function to separate stems from an audio file.

    Args:
        audio_path: Path to audio file.
        model: Demucs model name.
        sr: Sample rate.
        use_cache: Use cached stems if available.
        normalize: Normalize stem loudness.

    Returns:
        Dict with stem names as keys and audio arrays as values.
    """
    separator = StemSeparator(model=model)
    return separator.separate_stems(audio_path, sr=sr, use_cache=use_cache, normalize=normalize)
