"""Master orchestrator that ties together all analysis phases."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from .ingestion.scanner import AudioScanner
from .ingestion.loader import AudioLoader
from .dsp.spectrogram import DEFAULT_SPECTROGRAM_DIR
from .dsp.worker import analyze_one_track, _init_worker
from .database.store import TrackStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Orchestrator:
    """Master orchestrator for complete DJIA pipeline."""

    def __init__(
        self,
        db_path: str = "db/djia.db",
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
        # analyze_one_track() (workers) gets a plain picklable preset *name* and
        # resolves it itself via get_config(), rather than crossing the process
        # boundary with a DSPConfig object.
        self.segment_preset_name = segment_preset
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
        Analyze a single track and persist it to the database.

        Registers the track (or reuses its existing track_id if it's already in
        the DB), runs the same worker-safe compute pipeline analyze_library()
        uses per file (dsp.worker.analyze_one_track), and writes through the same
        _persist_result() path — so a single-track run is DB-identical to a
        library scan that happens to contain just this one file.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with track features (plus 'track_id'), or None if the
            file doesn't exist or analysis/persistence failed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        try:
            metadata = self.loader.extract_metadata(file_path)

            existing_id = self.store.get_track_id(str(file_path))
            track_id = existing_id or self.store.insert_track(
                file_path=str(file_path),
                file_name=metadata['file_name'],
                format=metadata['format'],
                duration=metadata.get('duration', 0),
                artist=metadata.get('artist'),
                title=metadata.get('title'),
                album=metadata.get('album'),
            )

            result = analyze_one_track(
                str(file_path), self.segment_preset_name, self.bars_per_phrase,
                spectrogram_dir=self.spectrogram_dir, spectrogram_key=track_id,
            )
            _, errors = self._persist_result(track_id, file_path, result, 0, 0)
            if errors:
                return None

            features = dict(result["features"])
            features.update(metadata)
            features['track_id'] = track_id
            if result.get('mood_scores'):
                features['mood'] = result['mood_scores']

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
