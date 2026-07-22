"""Master orchestrator that ties together all analysis phases."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from .ingestion.scanner import AudioScanner
from .ingestion.loader import AudioLoader
from .audio_analysis import analyze_track as analyze_audio
from .dsp.config import get_config
from .dsp.groove_engine import analyze_groove
from .dsp.mood_engine import analyze_mood as analyze_tonality
from .dsp.curation_engine import compute_spectral_flatness, compute_crest_factor
from .dsp.spectrogram import compute_and_save_spectrogram, DEFAULT_SPECTROGRAM_DIR
from .dsp.worker import analyze_one_track, _init_worker
from .ai.classifier import MoodClassifier
from .database.store import TrackStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Orchestrator:
    """Master orchestrator for complete DJIA pipeline."""

    def __init__(
        self,
        db_path: str = "data/djia.db",
        segment_preset: str = "minimal",
        bars_per_phrase: int = 16,
        spectrogram_dir: str = DEFAULT_SPECTROGRAM_DIR,
    ):
        """
        Initialize orchestrator with database.

        Args:
            db_path: Path to SQLite database
            segment_preset: DSP config preset for spectral segment detection
            bars_per_phrase: Phrase length (bars) for the phrase-locked segment grid
            spectrogram_dir: Directory .npy spectrograms are saved under
        """
        self.db_path = db_path
        self.store = TrackStore(db_path)
        self.loader = AudioLoader()
        self.mood_classifier = MoodClassifier()
        # Keep both: self.segment_config (resolved) for the methods below that already
        # run in-process, and the preset *name* for analyze_one_track() — workers get a
        # plain picklable string and resolve it themselves via get_config() rather than
        # crossing the process boundary with a DSPConfig object.
        self.segment_preset_name = segment_preset
        self.segment_config = get_config(segment_preset)
        self.bars_per_phrase = bars_per_phrase
        self.spectrogram_dir = spectrogram_dir

    @staticmethod
    def _segment_dicts(segments) -> List[Dict[str, Any]]:
        """Convert phrasing Segment objects to store dicts, stripping beat ranges from labels."""
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

    def _add_tonality(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Detect musical key/Camelot/timbral roughness/ZCR and merge into the features dict
        (best-effort)."""
        try:
            tonality = analyze_tonality(y, sr)
            features['key'] = tonality.key
            features['camelot_key'] = tonality.camelot_key
            features['key_confidence'] = tonality.key_confidence
            features['zero_crossing_rate'] = tonality.zero_crossing_rate
            features['roughness'] = tonality.roughness
        except Exception as e:
            logger.warning(f"Failed to detect key for {file_path}: {e}")

    def _add_swing(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Measure swing, onset strength, and beat strength; merge into the features dict
        (best-effort)."""
        try:
            groove = analyze_groove(y, sr)
            features['swing_score'] = groove.swing_score
            features['onset_strength_mean'] = groove.onset_strength_mean
            features['onset_strength_std'] = groove.onset_strength_std
            features['beat_strength'] = groove.beat_strength
        except Exception as e:
            logger.warning(f"Failed to measure swing for {file_path}: {e}")

    def _add_density(self, features: Dict[str, Any], y, sr, file_path) -> None:
        """Measure spectral density (flatness, crest factor) and merge into the features dict
        (best-effort). Crest factor reuses rms_mean/rms_peak already extracted by analyze_audio."""
        try:
            features['spectral_flatness'] = compute_spectral_flatness(y, sr)
            features['crest_factor'] = compute_crest_factor(
                rms_mean=features.get('rms_mean'),
                rms_peak=features.get('rms_peak'),
            )
        except Exception as e:
            logger.warning(f"Failed to measure spectral density for {file_path}: {e}")

    def _add_spectrogram(self, key, y, sr, file_path) -> None:
        """Compute and persist the log-magnitude STFT spectrogram (.npy), keyed by
        track_id when available or filename stem otherwise (best-effort)."""
        try:
            out_path = compute_and_save_spectrogram(
                y, sr, key,
                hop_length=self.segment_config.hop_length,
                base_dir=self.spectrogram_dir,
            )
            logger.debug(f"Saved spectrogram for {file_path} -> {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save spectrogram for {file_path}: {e}")

    def _persist_result(
        self, track_id: int, file_path, result: Dict[str, Any], analyzed: int, errors: int
    ) -> Tuple[int, int]:
        """Persist one track's worker result (features/segments/mood) to the DB.

        This — and only this — is where analyze_library() ever opens a write
        connection to SQLite, regardless of whether the compute stage ran
        sequentially or across a ProcessPoolExecutor.

        Returns the updated (analyzed, errors) counters.
        """
        if result.get("error"):
            logger.error(f"Error analyzing {file_path}: {result['error']}")
            return analyzed, errors + 1

        self.store.insert_features(track_id, result["features"])
        self.store.replace_segments(track_id, result["segments_spectral"], method="spectral")
        self.store.replace_segments(
            track_id, result["segments_phrase"], method=f"phrase{self.bars_per_phrase}"
        )
        if result.get("mood_scores"):
            self.store.insert_mood(track_id, result["mood_scores"])

        logger.info(f"✓ Analyzed: {Path(file_path).name}")
        return analyzed + 1, errors

    def analyze_library(
        self,
        data_dir: str = "data",
        skip_existing: bool = False,
        workers: int = 1,
    ) -> Dict[str, Any]:
        """
        Analyze entire audio library through all phases.

        Pipeline:
        1. Scan directory for audio files, then — sequentially, in the main process —
           register every remaining file in the DB to assign/get its track_id. This is
           cheap (no audio loading) and must happen before dispatch since features/
           segments/mood all need a track_id foreign key.
        2. Run the CPU-heavy compute (audio load through mood classification) for each
           track via `dsp.worker.analyze_one_track`. When `workers <= 1` this is a
           plain sequential loop in-process (identical order/behavior to the original
           implementation); when `workers > 1` it's fanned out across a
           `ProcessPoolExecutor`. Worker processes never touch the database.
        3. Persist each track's results (features/segments/mood) — sequentially, in
           the main process. Only one process ever opens a write connection to the
           SQLite DB, so there's no concurrent-writer problem at any `workers` value.

        Args:
            data_dir: Directory containing audio files
            skip_existing: Skip tracks already in database
            workers: Number of worker processes for the compute stage. `workers <= 1`
                skips `ProcessPoolExecutor` entirely and analyzes tracks one at a time
                in the main process — same execution order, same error handling, no
                multiprocessing overhead. `workers > 1` parallelizes compute across
                that many worker processes (completion order is not submission order).

        Returns:
            Summary dict with analyzed/skipped/error counts
        """
        # Phase 1: Scan
        logger.info(f"Scanning directory: {data_dir}")
        scanner = AudioScanner(data_dir)
        audio_files = scanner.scan()

        if not audio_files:
            logger.warning(f"No audio files found in {data_dir}")
            return {
                'analyzed': 0,
                'skipped': 0,
                'errors': 0,
                'db_path': self.db_path,
            }

        # Filter out existing tracks if skip_existing
        if skip_existing:
            existing_paths = set()
            for track in self.store.get_all_tracks():
                existing_paths.add(track['file_path'])

            audio_files = [
                f for f in audio_files
                if str(f['path']) not in existing_paths
            ]

        logger.info(f"Found {len(audio_files)} audio file(s)")

        # Sequential bookkeeping: register every remaining file and assign/get its
        # track_id. No audio loading here, so this stays fast and single-threaded.
        analyzed = 0
        skipped = 0
        errors = 0
        jobs: List[Tuple[int, Any]] = []  # (track_id, file_path)

        for file_info in audio_files:
            try:
                file_path = file_info['path']
                metadata = self.loader.extract_metadata(file_path)

                existing_id = self.store.get_track_id(str(file_path))
                if existing_id:
                    if skip_existing:
                        skipped += 1
                        continue
                    track_id = existing_id
                else:
                    track_id = self.store.insert_track(
                        file_path=str(file_path),
                        file_name=metadata['file_name'],
                        format=metadata['format'],
                        duration=metadata.get('duration', 0),
                        artist=metadata.get('artist'),
                        title=metadata.get('title'),
                        album=metadata.get('album'),
                    )
                jobs.append((track_id, file_path))
            except Exception as e:
                logger.error(f"Error registering {file_info.get('path')}: {e}")
                errors += 1

        # Compute + persist: heavy DSP work fanned out (or not) across processes;
        # every DB write happens back here, in the main process.
        if workers <= 1:
            for track_id, file_path in tqdm(jobs, desc="Analyzing tracks", unit="track"):
                result = analyze_one_track(
                    str(file_path), self.segment_preset_name, self.bars_per_phrase,
                    spectrogram_dir=self.spectrogram_dir, spectrogram_key=track_id,
                )
                analyzed, errors = self._persist_result(track_id, file_path, result, analyzed, errors)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
                futures = {
                    executor.submit(
                        analyze_one_track, str(file_path), self.segment_preset_name,
                        self.bars_per_phrase, spectrogram_dir=self.spectrogram_dir,
                        spectrogram_key=track_id,
                    ): (track_id, file_path)
                    for track_id, file_path in jobs
                }

                for future in tqdm(
                    as_completed(futures), total=len(jobs), desc="Analyzing tracks", unit="track"
                ):
                    track_id, file_path = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        # Pool-level failure (e.g. worker process crashed) — not the
                        # per-track "error" key, which analyze_one_track never raises.
                        logger.error(f"Error analyzing {file_path}: {e}")
                        errors += 1
                        continue
                    analyzed, errors = self._persist_result(track_id, file_path, result, analyzed, errors)

        return {
            'analyzed': analyzed,
            'skipped': skipped,
            'errors': errors,
            'db_path': self.db_path,
        }

    def analyze_single_track(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single track through complete pipeline.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with track features, or None if failed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            # Load metadata
            metadata = self.loader.extract_metadata(file_path)

            # Load audio
            audio_data = self.loader.load_audio(file_path)
            if not audio_data:
                logger.error(f"Failed to load audio: {file_path}")
                return None

            y = audio_data['audio_array']
            sr = audio_data['sample_rate']

            # Extract features
            features = analyze_audio(str(file_path), sr, None)
            if not features:
                logger.error(f"Failed to extract features: {file_path}")
                return None

            # analyze_audio() only sets 'tempo', not 'bpm' — alias for downstream callers.
            features.setdefault('bpm', features.get('tempo'))

            # Detect musical key (Camelot), timbre, swing, and density; merge into features
            self._add_tonality(features, y, sr, file_path)
            self._add_swing(features, y, sr, file_path)
            self._add_density(features, y, sr, file_path)
            # No DB track_id on this standalone path — key by filename stem instead.
            self._add_spectrogram(file_path.stem, y, sr, file_path)

            # Classify mood
            try:
                mood_result = self.mood_classifier.classify_mood(y, sr)
                if mood_result and 'moods' in mood_result:
                    features['mood'] = mood_result['moods']
            except Exception as e:
                logger.warning(f"Failed to classify mood: {e}")

            # Add metadata
            features.update(metadata)

            return features

        except Exception as e:
            logger.error(f"Error analyzing track: {e}")
            return None

    def get_all_tracks_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all tracks from database as a dictionary with features.

        Returns:
            Dict mapping track_id -> features_dict
        """
        tracks_dict = {}
        all_tracks = self.store.get_all_tracks()

        for track in all_tracks:
            track_id = track['id']

            # Get features
            features = self.store.get_track_features(track_id)

            if features:
                # Get mood
                mood = self.store.get_track_mood(track_id)

                # Build features dict
                track_features = {
                    'id': track_id,
                    'file_name': track['file_name'],
                    'duration': track.get('duration', 0),
                    'artist': track.get('artist'),
                    'title': track.get('title'),
                    'tempo': features.get('bpm'),
                    'rms_mean': features.get('rms_mean'),
                    'key': features.get('key'),
                    'mood': mood if mood else {},
                }

                # Add other features
                track_features.update(features)
                tracks_dict[track_id] = track_features

        return tracks_dict

    def get_track_count(self) -> int:
        """Get total number of analyzed tracks."""
        return self.store.get_tracks_count()
