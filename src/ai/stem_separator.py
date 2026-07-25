"""Stem separation with caching for efficiency.

Two backends are supported behind one contract:

- ``demucs`` (default) — Meta's Demucs models (``htdemucs``, ``mdx_extra``, ...).
- ``audio-separator`` — the ``audio-separator`` package, which hosts MelBand /
  BS Roformer and MDX-Net models. For techno, MelBand Roformer usually separates
  kick, bass, and synth layers more cleanly than HTDemucs, so it's a good first
  choice when the extra is installed (``pip install "djia[roformer]"``).

Both backends return the same thing: ``separate_stems()`` yields a dict keyed by
``STEM_NAMES`` (``drums``/``bass``/``vocals``/``melody``), each an audio array of
shape ``(channels, samples)``. Caching and loudness normalization are
backend-agnostic and shared.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
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

try:
    from audio_separator.separator import Separator as _AudioSeparator
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False


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

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model: str = DEFAULT_MODEL,
        backend: str = 'auto',
    ):
        """
        Initialize stem separator.

        Args:
            cache_dir: Directory to cache separated stems. Defaults to results/stems/.
            model: Model to use. For the ``demucs`` backend: 'htdemucs', 'mdx_extra',
                'mdx', etc. For the ``audio-separator`` backend: an audio-separator
                model filename (e.g. a MelBand Roformer ``*.ckpt``). Names containing
                'roformer' (or an explicit backend) select the audio-separator path.
            backend: 'auto' (infer from the model name), 'demucs', or
                'audio-separator'.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.backend = self._resolve_backend(model, backend)

        self.demucs_model = None
        self.audio_separator = None

        if self.backend == 'demucs':
            self._init_demucs()
        elif self.backend == 'audio-separator':
            self._init_audio_separator()

    # ------------------------------------------------------------------ backends

    @staticmethod
    def _resolve_backend(model: str, backend: str) -> str:
        """Pick a separation backend from an explicit choice or the model name."""
        if backend and backend != 'auto':
            if backend not in ('demucs', 'audio-separator'):
                raise ValueError(
                    f"Unknown stem backend '{backend}'; expected "
                    "'demucs' or 'audio-separator'."
                )
            return backend

        name = model.lower()
        if 'roformer' in name or name.endswith(('.ckpt', '.onnx', '.pth')):
            return 'audio-separator'
        return 'demucs'

    def _init_demucs(self) -> None:
        if not DEMUCS_AVAILABLE:
            print("Warning: Demucs not installed; stem separation will be unavailable.")
            return
        try:
            self.demucs_model = get_model(self.model)
        except Exception as e:
            print(f"Warning: Could not load Demucs model '{self.model}': {e}")
            print("Stem separation will be unavailable.")

    def _init_audio_separator(self) -> None:
        if not AUDIO_SEPARATOR_AVAILABLE:
            print(
                "Warning: audio-separator not installed; stem separation will be "
                "unavailable. Install with: pip install \"djia[roformer]\""
            )
            return
        try:
            self.audio_separator = _AudioSeparator(output_dir=str(self.cache_dir))
            # Downloads the model automatically the first time.
            self.audio_separator.load_model(self.model)
        except Exception as e:
            print(f"Warning: Could not load audio-separator model '{self.model}': {e}")
            print("Stem separation will be unavailable.")
            self.audio_separator = None

    def _backend_ready(self) -> bool:
        if self.backend == 'demucs':
            return DEMUCS_AVAILABLE and self.demucs_model is not None
        if self.backend == 'audio-separator':
            return AUDIO_SEPARATOR_AVAILABLE and self.audio_separator is not None
        return False

    # -------------------------------------------------------------------- caching

    def _get_track_hash(self, audio_path: str) -> str:
        """Generate a unique hash for the audio file based on its path and model."""
        # Include the model so switching backends/models doesn't return a stale
        # cache keyed only by path.
        key = f"{self.model}::{audio_path}"
        return hashlib.md5(key.encode()).hexdigest()

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

            # Verify cache is valid (same source and same model).
            if metadata.get('source_path') != audio_path:
                return None
            if metadata.get('model') != self.model:
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
                'backend': self.backend,
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

    # ------------------------------------------------------------- source mapping

    @classmethod
    def _map_source_name(cls, raw_name: str) -> str:
        """Map a backend's source label to one of ``STEM_NAMES``.

        Handles Demucs' ``other`` as well as audio-separator filename tokens like
        ``(Vocals)`` / ``(Instrumental)`` / ``(Other)``. Anything not recognized as
        drums/bass/vocals folds into ``melody`` (the "everything else / tonal"
        stem), matching how Demucs' ``other`` is exposed.
        """
        name = raw_name.lower()
        if 'drum' in name:
            return 'drums'
        if 'bass' in name:
            return 'bass'
        if 'vocal' in name:
            return 'vocals'
        return 'melody'

    def _assemble_stems(self, raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Build a complete ``STEM_NAMES`` dict, zero-filling absent stems.

        Stems shorter/longer than the longest one are padded/truncated so every
        array shares a common ``(2, samples)`` shape.
        """
        length = max((a.shape[-1] for a in raw.values()), default=1)
        stems: Dict[str, np.ndarray] = {
            name: np.zeros((2, length)) for name in self.STEM_NAMES
        }
        for name, audio in raw.items():
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] != 2:
                audio = audio[:2] if audio.shape[0] > 2 else np.repeat(audio, 2, axis=0)[:2]
            fitted = np.zeros((2, length))
            n = min(length, audio.shape[-1])
            fitted[:, :n] = audio[:, :n]
            stems[name] = fitted
        return stems

    # -------------------------------------------------------------- separation

    def _separate_demucs(self, audio_path: str, sr: int) -> Dict[str, np.ndarray]:
        """Separate stems using the Demucs backend."""
        import torch

        # Load audio
        waveform, _ = librosa.load(audio_path, sr=sr, mono=False)

        # Ensure stereo
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform], axis=0)

        waveform_tensor = torch.from_numpy(waveform).float()
        if not waveform_tensor.shape[0] == 2:
            if waveform_tensor.shape[0] == 1:
                waveform_tensor = waveform_tensor.repeat(2, 1)
            elif waveform_tensor.shape[0] > 2:
                waveform_tensor = waveform_tensor[:2]
            else:
                waveform_tensor = waveform_tensor.repeat(2 // waveform_tensor.shape[0] + 1, 1)[:2]

        # Add batch dimension
        waveform_tensor = waveform_tensor.unsqueeze(0)

        with torch.no_grad():
            stems_output = apply_model(self.demucs_model, waveform_tensor)

        # Demucs returns [batch, stems, channels, samples]
        stems_output = stems_output[0].cpu().numpy()

        # Map Demucs output by the model's OWN source order, not positionally.
        # htdemucs emits [drums, bass, other, vocals].
        model_sources = list(getattr(self.demucs_model, 'sources', []))
        raw = {}
        for i, source in enumerate(model_sources):
            if i >= stems_output.shape[0]:
                break
            stem_name = self.DEMUCS_SOURCE_MAP.get(source, self._map_source_name(source))
            raw[stem_name] = stems_output[i]

        return self._assemble_stems(raw)

    def _separate_audio_separator(self, audio_path: str, sr: int) -> Dict[str, np.ndarray]:
        """Separate stems using the audio-separator backend (MelBand Roformer etc.)."""
        # Isolate this track's raw model output so filenames from different tracks
        # don't collide in the shared cache dir.
        raw_dir = self._get_cache_path(audio_path) / '_raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.audio_separator.output_dir = str(raw_dir)

        output_files: List[str] = self.audio_separator.separate(str(audio_path))

        raw: Dict[str, np.ndarray] = {}
        for f in output_files:
            path = Path(f)
            if not path.is_absolute() or not path.exists():
                # audio-separator returns basenames relative to output_dir.
                path = raw_dir / path.name
            if not path.exists():
                print(f"Warning: expected stem file not found: {path}")
                continue
            y, _ = librosa.load(str(path), sr=sr, mono=False)
            stem_name = self._map_source_name(path.stem)
            # If two files map to the same stem (rare), keep the louder one.
            if stem_name in raw and np.mean(raw[stem_name] ** 2) >= np.mean(y ** 2):
                continue
            raw[stem_name] = y

        return self._assemble_stems(raw)

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
            RuntimeError: If separation fails.
        """
        # Check cache first
        if use_cache:
            cached_stems = self._load_cached_stems(audio_path)
            if cached_stems is not None:
                if normalize:
                    cached_stems = self._normalize_stem_loudness(cached_stems)
                return cached_stems

        if not self._backend_ready():
            print(
                f"Warning: stem backend '{self.backend}' unavailable. "
                "Returning empty stems."
            )
            return {name: np.zeros((1, 1)) for name in self.STEM_NAMES}

        try:
            print(
                f"Separating stems for {Path(audio_path).name} using "
                f"'{self.model}' ({self.backend})..."
            )

            if self.backend == 'audio-separator':
                stems = self._separate_audio_separator(audio_path, sr)
            else:
                stems = self._separate_demucs(audio_path, sr)

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
    normalize: bool = True,
    backend: str = 'auto',
) -> Dict[str, np.ndarray]:
    """
    Convenience function to separate stems from an audio file.

    Args:
        audio_path: Path to audio file.
        model: Model name (Demucs model, or an audio-separator model filename).
        sr: Sample rate.
        use_cache: Use cached stems if available.
        normalize: Normalize stem loudness.
        backend: 'auto', 'demucs', or 'audio-separator'.

    Returns:
        Dict with stem names as keys and audio arrays as values.
    """
    separator = StemSeparator(model=model, backend=backend)
    return separator.separate_stems(audio_path, sr=sr, use_cache=use_cache, normalize=normalize)
