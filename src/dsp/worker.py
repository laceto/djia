"""Picklable, worker-safe per-track analysis function for parallel library analysis.

`analyze_one_track()` runs the CPU-heavy audio+DSP compute (load audio through mood
classification) exactly as `Orchestrator.analyze_library()`'s per-file loop does today,
but it never touches the database — all SQLite writes stay in the main process. This
lets `Orchestrator.analyze_library(workers=N)` dispatch this function to a
`concurrent.futures.ProcessPoolExecutor` without any SQLite concurrent-writer problems.

This is a module-level function (not a bound `Orchestrator` method) because instance
methods are not reliably picklable across process boundaries.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from ..ai.classifier import MoodClassifier
from ..audio_analysis import analyze_track as analyze_audio
from ..ingestion.loader import AudioLoader
from .config import DSPConfig, get_config
from .curation_engine import compute_crest_factor, compute_spectral_flatness
from .groove_engine import analyze_groove
from .mood_engine import analyze_mood as analyze_tonality
from .phrasing_engine import analyze_structure, create_phrase_locked_segments
from .spectrogram import DEFAULT_SPECTROGRAM_DIR, compute_and_save_spectrogram

logger = logging.getLogger(__name__)

# Populated once per worker process by _init_worker() (the ProcessPoolExecutor
# initializer). analyze_one_track() also works when called directly/synchronously
# (no pool involved, e.g. in unit tests or the workers<=1 fallback path) via the lazy
# getters below, which construct-and-cache on first use.
_worker_state: Dict[str, Any] = {}


def _init_worker() -> None:
    """ProcessPoolExecutor initializer — build expensive, reusable objects once per
    worker process instead of once per track."""
    _worker_state['mood_classifier'] = MoodClassifier()
    _worker_state['loader'] = AudioLoader()


def _get_mood_classifier() -> MoodClassifier:
    if 'mood_classifier' not in _worker_state:
        _worker_state['mood_classifier'] = MoodClassifier()
    return _worker_state['mood_classifier']


def _get_loader() -> AudioLoader:
    if 'loader' not in _worker_state:
        _worker_state['loader'] = AudioLoader()
    return _worker_state['loader']


def _segment_dicts(segments) -> List[Dict[str, Any]]:
    """Convert phrasing Segment objects to store dicts, stripping beat ranges from labels.

    Duplicated from `Orchestrator._segment_dicts` (a staticmethod with no `self`
    dependency) rather than imported, to avoid a worker.py <-> orchestrator.py circular
    import: `orchestrator.py` imports `analyze_one_track` from this module.
    """
    return [
        {
            # "drop (beats 32-64)" -> "drop"
            'segment_type': seg.label.split('(')[0].strip(),
            'start_time': seg.start_time,
            'end_time': seg.end_time,
            'confidence': seg.confidence,
        }
        for seg in segments
    ]


def _empty_result(error: str) -> Dict[str, Any]:
    return {
        "error": error,
        "features": None,
        "segments_spectral": [],
        "segments_phrase": [],
        "mood_scores": None,
    }


def _add_tonality(features: Dict[str, Any], y, sr, file_path) -> None:
    """Detect musical key/Camelot/timbral roughness/ZCR and merge into features
    (best-effort)."""
    try:
        tonality = analyze_tonality(y, sr, file_path=str(file_path))
        features['key'] = tonality.key
        features['camelot_key'] = tonality.camelot_key
        features['key_confidence'] = tonality.key_confidence
        features['key_source'] = tonality.key_source
        features['zero_crossing_rate'] = tonality.zero_crossing_rate
        features['roughness'] = tonality.roughness
    except Exception as e:
        logger.warning(f"Failed to detect key for {file_path}: {e}")


def _add_swing(features: Dict[str, Any], y, sr, file_path) -> None:
    """Measure swing, onset strength, and beat strength; merge into features
    (best-effort)."""
    try:
        groove = analyze_groove(y, sr)
        features['swing_score'] = groove.swing_score
        features['onset_strength_mean'] = groove.onset_strength_mean
        features['onset_strength_std'] = groove.onset_strength_std
        features['beat_strength'] = groove.beat_strength
    except Exception as e:
        logger.warning(f"Failed to measure swing for {file_path}: {e}")


def _add_density(features: Dict[str, Any], y, sr, file_path) -> None:
    """Measure spectral density (flatness, crest factor); merge into features
    (best-effort). Crest factor reuses rms_mean/rms_peak already extracted by
    analyze_audio."""
    try:
        features['spectral_flatness'] = compute_spectral_flatness(y, sr)
        features['crest_factor'] = compute_crest_factor(
            rms_mean=features.get('rms_mean'),
            rms_peak=features.get('rms_peak'),
        )
    except Exception as e:
        logger.warning(f"Failed to measure spectral density for {file_path}: {e}")


def _compute_segments(y, sr, bpm, file_path, segment_config: DSPConfig, bars_per_phrase: int):
    """Detect structure segments — spectral + phrase-locked grid (best-effort).

    Returns (segments_spectral, segments_phrase) — empty lists on missing BPM or failure.
    """
    if not bpm:
        logger.warning(f"No BPM for {file_path}; skipping segment detection")
        return [], []
    try:
        phrasing_cfg = segment_config.phrasing
        result = analyze_structure(
            y, sr, bpm,
            hop_length=segment_config.hop_length,
            min_bars=phrasing_cfg.min_bars,
            thresh_frac=phrasing_cfg.thresh_frac,
            max_pads=phrasing_cfg.max_pads,
        )
        segments_spectral = _segment_dicts(result.segments)

        locked = create_phrase_locked_segments(
            duration=len(y) / sr,
            bpm=bpm,
            bars_per_phrase=bars_per_phrase,
        )
        segments_phrase = _segment_dicts(locked)
        return segments_spectral, segments_phrase
    except Exception as e:
        logger.warning(f"Failed to detect segments for {file_path}: {e}")
        return [], []


def _add_spectrogram(key, y, sr, file_path, hop_length: int, spectrogram_dir: str) -> None:
    """Compute and persist the log-magnitude STFT spectrogram (.npy), keyed by track_id
    when available or filename stem otherwise (best-effort). Mirrors
    `Orchestrator._add_spectrogram` — a pure filesystem write (not a DB write), so it's
    safe to run inside a worker process as long as `key` is unique per track."""
    try:
        out_path = compute_and_save_spectrogram(
            y, sr, key, hop_length=hop_length, base_dir=spectrogram_dir,
        )
        logger.debug(f"Saved spectrogram for {file_path} -> {out_path}")
    except Exception as e:
        logger.warning(f"Failed to save spectrogram for {file_path}: {e}")


def analyze_one_track(
    file_path: str,
    segment_preset: str,
    bars_per_phrase: int,
    spectrogram_dir: str = DEFAULT_SPECTROGRAM_DIR,
    spectrogram_key=None,
) -> Dict[str, Any]:
    """
    Run the full compute-only per-track analysis pipeline in a worker-safe way.

    Performs, in order, exactly what `Orchestrator.analyze_library()`'s per-file loop
    does today for the "compute" half: load audio, DSP feature extraction
    (`audio_analysis.analyze_track`), tonality/swing/density enrichment, structure
    segmentation, spectrogram persistence, and mood classification. Never touches the
    database and never raises — all failures are caught and reported via the "error"
    key so one bad track can't crash a `ProcessPoolExecutor` worker or take down the
    batch it's part of.

    Args:
        file_path: Path to the audio file to analyze.
        segment_preset: DSP config preset *name* (e.g. "minimal"), resolved to a
            `DSPConfig` via `get_config()` inside the worker — only a plain, trivially
            picklable string needs to cross the process boundary.
        bars_per_phrase: Phrase length (bars) for the phrase-locked segment grid.
        spectrogram_dir: Directory the .npy spectrogram is saved under (best-effort;
            a pure filesystem write, not a DB write, so it's safe from a worker).
        spectrogram_key: Stable key (typically the DB track_id) the spectrogram .npy
            is saved as. Defaults to the file's stem when not provided (e.g. when
            called standalone/without a known track_id, as in unit tests).

    Returns:
        Dict with keys:
            "error": None on success, else a string describing what went wrong.
            "features": merged feature dict (audio_analysis + tonality + swing +
                density), or None on failure.
            "segments_spectral": list of segment dicts (spectral method).
            "segments_phrase": list of segment dicts (phrase-locked grid method).
            "mood_scores": dict of mood confidence scores, or None.
    """
    try:
        loader = _get_loader()
        # AudioLoader.load_audio() expects a Path (it reads .name for logging).
        path = Path(file_path)
        audio_data = loader.load_audio(path)
        if not audio_data:
            logger.error(f"Failed to load audio: {file_path}")
            return _empty_result(f"Failed to load audio: {file_path}")

        y = audio_data['audio_array']
        sr = audio_data['sample_rate']

        features = analyze_audio(str(file_path), sr, None)
        if not features:
            logger.error(f"Failed to extract features: {file_path}")
            return _empty_result(f"Failed to extract features: {file_path}")

        # analyze_audio() only sets 'tempo', not 'bpm' — alias so downstream callers
        # that read features.get('bpm') (e.g. segment detection) see it too.
        features.setdefault('bpm', features.get('tempo'))

        # Detect musical key (Camelot), timbre, swing, and density; merge into features
        _add_tonality(features, y, sr, file_path)
        _add_swing(features, y, sr, file_path)
        _add_density(features, y, sr, file_path)

        segment_config = get_config(segment_preset)

        # Detect structure segments (spectral + phrase-locked grid)
        segments_spectral, segments_phrase = _compute_segments(
            y, sr, features.get('bpm'), file_path, segment_config, bars_per_phrase
        )

        # Persist spectrogram (.npy), keyed by track_id when known, else filename stem
        key = spectrogram_key if spectrogram_key is not None else path.stem
        _add_spectrogram(key, y, sr, file_path, segment_config.hop_length, spectrogram_dir)

        # Classify mood (best-effort)
        mood_scores = None
        try:
            mood_classifier = _get_mood_classifier()
            mood_result = mood_classifier.classify_mood(y, sr)
            if mood_result and 'moods' in mood_result:
                mood_scores = mood_result['moods']
        except Exception as e:
            logger.warning(f"Failed to classify mood for {file_path}: {e}")

        return {
            "error": None,
            "features": features,
            "segments_spectral": segments_spectral,
            "segments_phrase": segments_phrase,
            "mood_scores": mood_scores,
        }

    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return _empty_result(str(e))
