"""Backfill bassline-type scores onto tracks already in the database.

Re-running the full `analyze` pipeline just to add six columns would redo ~40 s of
unrelated DSP per track. This loads each audio file once, reuses the stored BPM and
sub-bass profile, and writes only the `bassline_*` columns.

Mirrors the `analyze` split: the compute step (`compute_one`) is worker-safe and never
touches the database; only the main process writes, so there is no concurrent-writer
problem at any ``workers`` value.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database.store import TrackStore
from ..ingestion.loader import AudioLoader
from .bassline_engine import BASSLINE_KEYS, compute_bassline_profile
from .groove_engine import analyze_groove

logger = logging.getLogger(__name__)


def compute_one(file_path: str, bpm: Optional[float], sub_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load one track and score its bassline. Returns None if the audio can't be loaded.

    Beat times come from the groove engine, exactly as `analyze_one_track` does, so a
    backfilled score matches what a fresh `analyze` run would store.
    """
    try:
        audio = AudioLoader().load_audio(Path(file_path))
        if not audio:
            return None
        y, sr = audio['audio_array'], audio['sample_rate']
        try:
            beat_times = analyze_groove(y, sr).beat_times
        except Exception as e:  # noqa: BLE001 - fall back to the BPM grid
            logger.warning(f"No beat times for {file_path} ({e}); using BPM grid")
            beat_times = None
        return compute_bassline_profile(
            y, sr, bpm=bpm, beat_times=beat_times, sub_profile=sub_profile,
        )
    except Exception as e:  # noqa: BLE001 - one bad file must not kill a pool worker
        logger.error(f"Bassline scoring failed for {file_path}: {e}")
        return None


def backfill_bassline(
    db_path: str = "db/djia.db",
    track_ids: Optional[List[int]] = None,
    only_missing: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """Score and store the bassline types for tracks already in `db_path`.

    Args:
        db_path: SQLite database.
        track_ids: restrict to these track ids (default: every track with features).
        only_missing: skip tracks that already have all six scores.
        workers: worker processes for the compute step (1 = in-process, sequential).

    Returns:
        {'scored': n, 'skipped': n, 'errors': n, 'results': {track_id: profile}}
    """
    store = TrackStore(db_path)
    rows = [r for r in store.get_all_tracks_with_features() if r.get('track_id') is not None]
    if track_ids is not None:
        wanted = set(track_ids)
        rows = [r for r in rows if r['track_id'] in wanted]

    todo = []
    skipped = 0
    for r in rows:
        if only_missing and all(r.get(k) is not None for k in BASSLINE_KEYS):
            skipped += 1
            continue
        todo.append(r)

    summary: Dict[str, Any] = {'scored': 0, 'skipped': skipped, 'errors': 0, 'results': {}}

    def _record(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> None:
        if profile is None or not store.update_bassline(row['track_id'], profile):
            summary['errors'] += 1
            return
        summary['scored'] += 1
        summary['results'][row['track_id']] = profile

    args = lambda r: (r['file_path'], r.get('bpm'), r)  # noqa: E731
    if workers <= 1:
        for i, r in enumerate(todo, 1):
            _record(r, compute_one(*args(r)))
            logger.info(f"[{i}/{len(todo)}] {r.get('file_name')}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(compute_one, *args(r)): r for r in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row = futures[fut]
                try:
                    _record(row, fut.result())
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Worker failed for {row.get('file_name')}: {e}")
                    summary['errors'] += 1
                logger.info(f"[{i}/{len(todo)}] {row.get('file_name')}")
    return summary
