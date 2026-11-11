"""Tests for the Rekordbox XML exporter."""

import sqlite3
import xml.etree.ElementTree as ET

import pytest

from src.rekordbox.exporter import (
    export_rekordbox_xml, key_to_tonality, path_to_location, segments_to_marks,
)


@pytest.mark.parametrize("key,expected", [
    ("A minor", "Am"), ("C# major", "C#"), ("Bb minor", "A#m"), ("", ""), (None, ""), ("junk", ""),
])
def test_key_to_tonality(key, expected):
    assert key_to_tonality(key) == expected


def test_path_to_location_encodes_spaces():
    loc = path_to_location("C:/Music/My Track.mp3")
    assert loc.startswith("file://localhost/")
    assert "My%20Track.mp3" in loc


def test_segments_to_marks():
    segs = [
        {"segment_type": "drop", "start_time": 60.0},
        {"segment_type": "intro", "start_time": 0.0},
        {"segment_type": "drop", "start_time": 200.0},
    ]
    marks = segments_to_marks(segs)
    hot = [m for m in marks if m["Num"] >= 0]
    assert len(hot) == 1 and hot[0]["Num"] == 0 and hot[0]["Start"] == "60.000"
    assert len([m for m in marks if m["Num"] == -1]) == 3


def test_export_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE tracks (id INTEGER PRIMARY KEY, file_path TEXT, file_name TEXT,
            artist TEXT, title TEXT, album TEXT, duration REAL);
        CREATE TABLE features (track_id INTEGER, bpm REAL, key TEXT);
        CREATE TABLE segments (track_id INTEGER, segment_type TEXT, start_time REAL);
        INSERT INTO tracks VALUES (1, 'C:/m/a.mp3', 'a.mp3', 'Art', 'Tit', NULL, 300.4);
        INSERT INTO features VALUES (1, 128.0, 'A minor');
        INSERT INTO segments VALUES (1, 'drop', 64.0), (1, 'outro', 250.0);
    """)
    conn.commit()
    conn.close()

    out = export_rekordbox_xml(str(db), str(tmp_path / "o.xml"))
    root = ET.parse(out).getroot()
    track = root.find("COLLECTION/TRACK")
    assert track.get("Tonality") == "Am"
    assert track.get("AverageBpm") == "128.00"
    assert track.get("TotalTime") == "300"
    assert {m.get("Num") for m in track.findall("POSITION_MARK")} == {"0", "3", "-1"}
    assert root.find("PLAYLISTS/NODE/NODE/TRACK").get("Key") == "1"
