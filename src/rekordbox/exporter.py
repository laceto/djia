"""Rekordbox XML exporter (DJ_PLAYLISTS format).

Writes a standalone XML library from the DJIA database. In Rekordbox, point
Preferences > Advanced > Database > "rekordbox xml" at the file, then drag
tracks from the "rekordbox xml" tree into your Collection to import
BPM, key and cues.
"""

import logging
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Same pad layout as the Traktor exporter so muscle memory carries over.
# Rekordbox hot cue Num: 0=A, 1=B, 2=C, 3=D ...
LABEL_TO_HOTCUE = {"drop": 0, "breakdown": 1, "outro": 3}

# RGB per label (Rekordbox accepts Red/Green/Blue on POSITION_MARK).
LABEL_COLORS = {
    "drop": (230, 40, 40),
    "breakdown": (40, 120, 230),
    "outro": (40, 200, 120),
}
DEFAULT_COLOR = (200, 200, 200)

_NOTE_SHARPS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def key_to_tonality(key: Optional[str]) -> str:
    """'A minor' -> 'Am', 'C# major' -> 'C#'. Unparseable input -> ''."""
    if not key:
        return ""
    parts = key.strip().split()
    if len(parts) != 2:
        return ""
    note, mode = parts[0], parts[1].lower()
    note = note[0].upper() + note[1:]
    note = _NOTE_SHARPS.get(note, note)
    if mode.startswith("min"):
        return f"{note}m"
    if mode.startswith("maj"):
        return note
    return ""


def path_to_location(file_path: str) -> str:
    """Absolute path -> Rekordbox 'file://localhost/...' URL."""
    posix = Path(file_path).resolve().as_posix()
    if not posix.startswith("/"):  # Windows drive path, e.g. C:/Music/x.mp3
        posix = "/" + posix
    return "file://localhost" + quote(posix, safe="/:")


def segments_to_marks(segments: List[Dict]) -> List[Dict]:
    """
    Convert segment rows to POSITION_MARK dicts.

    Hot cues: first segment of each label in LABEL_TO_HOTCUE.
    Memory cues (Num=-1): every segment, so the whole structure shows on the
    waveform.
    """
    marks: List[Dict] = []
    used = set()
    seen = set()
    for seg in sorted(segments, key=lambda s: s["start_time"]):
        label = (seg["segment_type"] or "").lower()
        start = float(seg["start_time"])
        if (label, round(start, 2)) in seen:  # DB can hold the same segment from several methods
            continue
        seen.add((label, round(start, 2)))
        rgb = LABEL_COLORS.get(label, DEFAULT_COLOR)
        base = {"Name": label.capitalize() or "Cue", "Type": "0",
                "Start": f"{start:.3f}", "Red": rgb[0], "Green": rgb[1], "Blue": rgb[2]}
        pad = LABEL_TO_HOTCUE.get(label)
        if pad is not None and pad not in used:
            used.add(pad)
            marks.append({**base, "Num": pad})
        marks.append({**base, "Num": -1})
    return marks


def export_rekordbox_xml(
    db_path: str = "db/djia.db",
    output_path: str = "results/djia_rekordbox.xml",
    playlist_name: str = "DJIA",
) -> str:
    """Export every analyzed track in the DB to a Rekordbox XML file."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT t.id, t.file_path, t.file_name, t.artist, t.title, t.album,
                   t.duration, f.bpm, f.key
            FROM tracks t LEFT JOIN features f ON f.track_id = t.id
            ORDER BY t.id
            """
        ).fetchall()

        root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
        ET.SubElement(root, "PRODUCT", Name="rekordbox", Version="6.0.0", Company="AlphaTheta")
        collection = ET.SubElement(root, "COLLECTION", Entries=str(len(rows)))

        for row in rows:
            attrs = {
                "TrackID": str(row["id"]),
                "Name": row["title"] or Path(row["file_name"]).stem,
                "Artist": row["artist"] or "",
                "Album": row["album"] or "",
                "TotalTime": str(int(round(row["duration"] or 0))),
                "Location": path_to_location(row["file_path"]),
            }
            tonality = key_to_tonality(row["key"])
            if tonality:
                attrs["Tonality"] = tonality
            if row["bpm"]:
                attrs["AverageBpm"] = f"{float(row['bpm']):.2f}"
            track = ET.SubElement(collection, "TRACK", attrs)

            if row["bpm"]:
                # Beat-grid anchor isn't stored in the DB; Rekordbox re-analyzes
                # the grid on import, Inizio 0 is a placeholder.
                ET.SubElement(track, "TEMPO", Inizio="0.000",
                              Bpm=f"{float(row['bpm']):.2f}", Metro="4/4", Battito="1")

            segs = [dict(s) for s in conn.execute(
                "SELECT segment_type, start_time FROM segments WHERE track_id = ?",
                (row["id"],))]
            for mark in segments_to_marks(segs):
                ET.SubElement(track, "POSITION_MARK", {k: str(v) for k, v in mark.items()})

        playlists = ET.SubElement(root, "PLAYLISTS")
        node_root = ET.SubElement(playlists, "NODE", Type="0", Name="ROOT", Count="1")
        node = ET.SubElement(node_root, "NODE", Name=playlist_name, Type="1",
                             KeyType="0", Entries=str(len(rows)))
        for row in rows:
            ET.SubElement(node, "TRACK", Key=str(row["id"]))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        ET.parse(output_path)  # validate well-formed
        logger.info(f"Exported {len(rows)} tracks to Rekordbox XML: {output_path}")
        return output_path
    finally:
        conn.close()
