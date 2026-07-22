"""DJUCED hot-cue exporter.

Writes DJIA mix points as hot cues directly into DJUCED's SQLite database
(`Documents/DJUCED/DJUCED.db`), so mix-in/out marks show on the Hercules pads.

DJUCED's cue model (observed from DJUCED 5.x):
- `trackCues` rows keyed by `trackId` = the file's absolute path with forward
  slashes (same string as `tracks.absolutepath`).
- `cuenumber` 1-8 are the hot-cue pads; 0 is the main cue point.
- `cuepos` is in plain seconds.

Safety: always call `backup_djuced_db` before writing, and close DJUCED first —
SQLite writes will fail or race while the app holds the database.
"""

import logging
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DJUCED_DB = str(Path.home() / "Documents" / "DJUCED" / "DJUCED.db")

# Cues written by DJIA carry this prefix so re-exports replace only our own cues
CUE_PREFIX = "DJIA "

# DJUCED's default cue color (matches user- and demo-created cues)
DEFAULT_CUE_COLOR = 4

MAX_HOTCUE_PADS = 8


def normalize_track_name(name: str) -> str:
    """
    Reduce a track filename to a comparable token.

    Drops the extension, parenthesized/bracketed junk (release-group tags,
    "(Original Mix)"), and all non-alphanumerics, then lowercases. This lets
    DJIA's cleaned copies match DJUCED's originals despite renames like
    '01 - ambivalent - nineteen.mp3' vs
    '01-ambivalent-nineteen (0daymusic.org).mp3'.
    """
    stem = Path(name).stem
    stem = re.sub(r"[(\[].*?[)\]]", " ", stem)
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def load_djuced_library(db_path: str = DEFAULT_DJUCED_DB) -> List[Dict[str, str]]:
    """
    Load (absolutepath, filename) for every track DJUCED knows about.

    Returns:
        List of dicts with keys: absolutepath, filename, norm (normalized name).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT absolutepath, filename FROM tracks WHERE absolutepath IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "absolutepath": ap,
            "filename": fn or Path(ap).name,
            "norm": normalize_track_name(fn or Path(ap).name),
        }
        for ap, fn in rows
    ]


def match_djuced_tracks(file_name: str, library: List[Dict[str, str]]) -> List[str]:
    """
    Find every DJUCED `absolutepath` for a DJIA track filename.

    Tries an exact normalized-name match first, then a one-sided containment
    (either name is a prefix-like subset of the other, e.g. with a release
    tag stripped only from one side). DJUCED libraries often hold duplicate
    copies of the same track in different folders — all copies are returned
    so cues land on whichever one gets loaded on a deck.

    Args:
        file_name: DJIA track filename (e.g. from tracks.file_name)
        library: Output of `load_djuced_library`

    Returns:
        List of DJUCED absolutepaths (empty if no confident match).
    """
    norm = normalize_track_name(file_name)
    if not norm:
        return []

    exact = [t["absolutepath"] for t in library if t["norm"] == norm]
    if exact:
        return exact

    partial = [
        t
        for t in library
        if len(t["norm"]) >= 8
        and (norm.startswith(t["norm"]) or t["norm"].startswith(norm))
    ]
    # Partial hits are only trusted when they all point at the same track name
    # (duplicate copies), never when different tracks share a prefix.
    if partial and len({t["norm"] for t in partial}) == 1:
        return [t["absolutepath"] for t in partial]
    if partial:
        logger.warning(
            f"Ambiguous DJUCED match for {file_name}: {len(partial)} partial hits"
        )
    return []


def backup_djuced_db(db_path: str = DEFAULT_DJUCED_DB) -> str:
    """
    Copy DJUCED.db to a timestamped sibling file before writing.

    Returns:
        Path of the backup file.
    """
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"DJUCED database not found: {db_path}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = src.with_name(f"{src.name}.djia-backup-{stamp}")
    shutil.copy2(src, backup)
    logger.info(f"Backed up DJUCED DB to {backup}")
    return str(backup)


def write_track_cues(
    db_path: str,
    track_abs_path: str,
    cues: List[Tuple[str, float]],
    color: int = DEFAULT_CUE_COLOR,
) -> int:
    """
    Write DJIA hot cues for one track into DJUCED's trackCues table.

    Replaces any previous DJIA-prefixed cues for the track, then assigns each
    new cue the lowest free pad (1-8) not taken by the user's own cues. Cues
    that don't fit (all pads taken) are skipped with a warning.

    Args:
        db_path: Path to DJUCED.db (write access; close DJUCED first)
        track_abs_path: DJUCED trackId (tracks.absolutepath)
        cues: (name, seconds) pairs, written in order
        color: DJUCED cue color code

    Returns:
        Number of cues written.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "DELETE FROM trackCues WHERE trackId = ? AND cuename LIKE ?",
            (track_abs_path, f"{CUE_PREFIX}%"),
        )
        taken = {
            row[0]
            for row in conn.execute(
                "SELECT cuenumber FROM trackCues WHERE trackId = ?",
                (track_abs_path,),
            )
        }
        free_pads = [n for n in range(1, MAX_HOTCUE_PADS + 1) if n not in taken]

        written = 0
        for (name, seconds), pad in zip(cues, free_pads):
            conn.execute(
                """
                INSERT INTO trackCues
                    (trackId, cuename, cuenumber, cuepos, loopLength, cueColor, isSavedLoop)
                VALUES (?, ?, ?, ?, 0, ?, 0)
                """,
                (track_abs_path, f"{CUE_PREFIX}{name}", pad, float(seconds), color),
            )
            written += 1

        if written < len(cues):
            logger.warning(
                f"Only {written}/{len(cues)} cues fit on free pads for {track_abs_path}"
            )

        conn.commit()
        return written
    finally:
        conn.close()


def export_mix_cues(
    track_cues: Dict[str, List[Tuple[str, float]]],
    db_path: str = DEFAULT_DJUCED_DB,
    dry_run: bool = True,
) -> Dict[str, Dict]:
    """
    Export mix cues for many tracks into DJUCED, matching by filename.

    Cues are written to every DJUCED copy of a matched track, so the marks
    show whichever copy gets loaded on a deck.

    Args:
        track_cues: DJIA filename -> list of (cue name, seconds)
        db_path: Path to DJUCED.db
        dry_run: When True (default), match and report but write nothing

    Returns:
        Report dict: filename -> {"matched": list of abspaths,
                                  "cues": n requested, "written": n written}.
        A backup is created automatically before the first real write.
    """
    library = load_djuced_library(db_path)
    report: Dict[str, Dict] = {}
    backed_up = False

    for file_name, cues in track_cues.items():
        abs_paths = match_djuced_tracks(file_name, library)
        entry = {"matched": abs_paths, "cues": len(cues), "written": 0}

        if abs_paths and cues and not dry_run:
            if not backed_up:
                backup_djuced_db(db_path)
                backed_up = True
            for abs_path in abs_paths:
                entry["written"] += write_track_cues(db_path, abs_path, cues)

        report[file_name] = entry

    matched = sum(1 for r in report.values() if r["matched"])
    written = sum(r["written"] for r in report.values())
    logger.info(
        f"DJUCED export: {matched}/{len(report)} tracks matched, "
        f"{written} cues written{' (dry run)' if dry_run else ''}"
    )
    return report
