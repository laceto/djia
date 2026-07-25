"""Tests for the DJUCED hot-cue exporter (temp DB mimicking DJUCED's schema)."""

import sqlite3

import pytest

from src.djuced.exporter import (
    CUE_PREFIX,
    crosscheck_keys,
    export_mix_cues,
    load_djuced_keys,
    load_djuced_library,
    match_djuced_tracks,
    normalize_track_name,
    write_track_cues,
)


@pytest.fixture
def djuced_db(tmp_path):
    """Minimal DJUCED.db clone: tracks + trackCues with one user cue."""
    db = tmp_path / "DJUCED.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE tracks (
            id INTEGER PRIMARY KEY,
            filename CHARACTER VARYING(255),
            absolutepath CHARACTER VARYING(1024)
        );
        CREATE TABLE trackCues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trackId CHARACTER VARYING(100),
            cuename CHARACTER VARYING(100),
            cuenumber INTEGER,
            cuepos DECIMAL(5,1),
            loopLength DECIMAL(5,1),
            cueColor INTEGER,
            isSavedLoop INTEGER
        );
        INSERT INTO tracks (filename, absolutepath) VALUES
            ('01-ambivalent-nineteen (0daymusic.org).mp3',
             'C:/Users/x/musica/01-ambivalent-nineteen (0daymusic.org).mp3'),
            ('01. SIS - Nu Wim De Wa (Original Mix) -.mp3',
             'C:/Users/x/musica/01. SIS - Nu Wim De Wa (Original Mix) -.mp3'),
            (' Hermanez - Marrakech.mp3',
             'C:/Users/x/musica/ Hermanez - Marrakech.mp3'),
            (' Hermanez - Marrakech.mp3',
             'D:/musica/backup/ Hermanez - Marrakech.mp3');
        INSERT INTO trackCues
            (trackId, cuename, cuenumber, cuepos, loopLength, cueColor, isSavedLoop)
        VALUES
            ('C:/Users/x/musica/01-ambivalent-nineteen (0daymusic.org).mp3',
             'my cue', 1, 12.5, 0, 4, 0);
        """
    )
    conn.commit()
    conn.close()
    return str(db)


class TestMatching:
    def test_normalize_strips_tags_and_punctuation(self):
        assert (
            normalize_track_name("01-ambivalent-nineteen (0daymusic.org).mp3")
            == normalize_track_name("01 - ambivalent - nineteen.mp3")
        )
        assert (
            normalize_track_name("01. SIS - Nu Wim De Wa (Original Mix) -.mp3")
            == normalize_track_name("01. SIS - Nu Wim De Wa.mp3")
        )

    def test_match_renamed_copy(self, djuced_db):
        library = load_djuced_library(djuced_db)
        assert match_djuced_tracks("01 - ambivalent - nineteen.mp3", library) == [
            "C:/Users/x/musica/01-ambivalent-nineteen (0daymusic.org).mp3"
        ]

    def test_match_returns_all_duplicate_copies(self, djuced_db):
        library = load_djuced_library(djuced_db)
        matches = match_djuced_tracks(" Hermanez - Marrakech.mp3", library)
        assert sorted(matches) == [
            "C:/Users/x/musica/ Hermanez - Marrakech.mp3",
            "D:/musica/backup/ Hermanez - Marrakech.mp3",
        ]

    def test_no_match_returns_empty(self, djuced_db):
        library = load_djuced_library(djuced_db)
        assert match_djuced_tracks("totally unknown track.mp3", library) == []


class TestWriteCues:
    TRACK = "C:/Users/x/musica/01-ambivalent-nineteen (0daymusic.org).mp3"

    def _cues(self, db):
        conn = sqlite3.connect(db)
        try:
            return conn.execute(
                "SELECT cuename, cuenumber, cuepos FROM trackCues "
                "WHERE trackId = ? ORDER BY cuenumber",
                (self.TRACK,),
            ).fetchall()
        finally:
            conn.close()

    def test_writes_on_free_pads_preserving_user_cues(self, djuced_db):
        written = write_track_cues(
            djuced_db, self.TRACK, [("mix-in", 0.0), ("bass in", 60.0)]
        )
        assert written == 2

        cues = self._cues(djuced_db)
        # user cue untouched on pad 1; DJIA cues took the next free pads
        assert ("my cue", 1, 12.5) in cues
        assert (f"{CUE_PREFIX}mix-in", 2, 0.0) in cues
        assert (f"{CUE_PREFIX}bass in", 3, 60.0) in cues

    def test_rewrite_replaces_only_djia_cues(self, djuced_db):
        write_track_cues(djuced_db, self.TRACK, [("mix-in", 0.0), ("bass in", 60.0)])
        write_track_cues(djuced_db, self.TRACK, [("mix-in", 7.5)])

        cues = self._cues(djuced_db)
        assert ("my cue", 1, 12.5) in cues
        assert (f"{CUE_PREFIX}mix-in", 2, 7.5) in cues
        assert len(cues) == 2  # old DJIA cues gone, user cue kept


class TestExportMixCues:
    def test_dry_run_matches_but_writes_nothing(self, djuced_db):
        report = export_mix_cues(
            {"01 - ambivalent - nineteen.mp3": [("mix-in", 0.0)]},
            db_path=djuced_db,
            dry_run=True,
        )
        entry = report["01 - ambivalent - nineteen.mp3"]
        assert entry["matched"]
        assert entry["written"] == 0

    def test_real_run_writes_and_backs_up(self, djuced_db, tmp_path):
        report = export_mix_cues(
            {
                "01 - ambivalent - nineteen.mp3": [("mix-in", 0.0)],
                " Hermanez - Marrakech.mp3": [("mix-in", 5.0)],
                "unknown.mp3": [("mix-in", 0.0)],
            },
            db_path=djuced_db,
            dry_run=False,
        )
        assert report["01 - ambivalent - nineteen.mp3"]["written"] == 1
        # duplicate copies each get the cue
        assert report[" Hermanez - Marrakech.mp3"]["written"] == 2
        assert report["unknown.mp3"]["matched"] == []
        # a timestamped backup landed next to the DB
        assert list(tmp_path.glob("DJUCED.db.djia-backup-*"))


@pytest.fixture
def djuced_db_with_keys(tmp_path):
    """DJUCED.db clone whose tracks table carries a `key` column (Open Key strings)."""
    db = tmp_path / "DJUCED.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE tracks (id INTEGER PRIMARY KEY, filename TEXT, "
        "absolutepath TEXT, key TEXT)"
    )
    conn.executemany(
        "INSERT INTO tracks (filename, absolutepath, key) VALUES (?, ?, ?)",
        [
            ("2000_and_one-pak_pak.mp3", "/m/2000_and_one-pak_pak.mp3", "5m"),   # -> 12A
            ("some_track.mp3", "/m/some_track.mp3", "8A"),                       # Camelot form
            ("weird.mp3", "/m/weird.mp3", "C#m"),                                # musical -> unreadable
            ("nokey.mp3", "/m/nokey.mp3", None),                                 # no key
        ],
    )
    conn.commit()
    conn.close()
    return str(db)


class TestDjucedKeyCrosscheck:
    def test_load_keys_reads_key_column(self, djuced_db_with_keys):
        lib = load_djuced_keys(djuced_db_with_keys)
        keys = {t["filename"]: t["key_raw"] for t in lib}
        assert keys["2000_and_one-pak_pak.mp3"] == "5m"
        assert keys["nokey.mp3"] is None

    def test_load_keys_without_key_column(self, djuced_db):
        """Falls back gracefully to key_raw=None when the schema has no key column."""
        lib = load_djuced_keys(djuced_db)
        assert lib and all(t["key_raw"] is None for t in lib)

    def test_crosscheck_statuses(self, djuced_db_with_keys):
        lib = load_djuced_keys(djuced_db_with_keys)
        djia = [
            {"file_name": "2000_and_one-pak_pak.mp3", "key": "C#/Db minor", "camelot_key": "12A"},
            {"file_name": "some_track.mp3", "key": "A#/Bb minor", "camelot_key": "3A"},
            {"file_name": "weird.mp3", "key": "C#/Db minor", "camelot_key": "12A"},
            {"file_name": "nokey.mp3", "key": "C major", "camelot_key": "8B"},
            {"file_name": "not_in_djuced.mp3", "key": "D minor", "camelot_key": "7A"},
        ]
        status = {r["file_name"]: r["status"] for r in crosscheck_keys(djia, lib)}
        assert status["2000_and_one-pak_pak.mp3"] == "match"   # 12A == open-key 5m
        assert status["some_track.mp3"] == "diff"              # 3A != 8A
        assert status["weird.mp3"] == "unreadable"             # "C#m" not Camelot/Open Key
        assert status["nokey.mp3"] == "no_djuced_key"
        assert status["not_in_djuced.mp3"] == "no_djuced_match"

    def test_crosscheck_normalizes_djuced_open_key(self, djuced_db_with_keys):
        """DJUCED's Open Key '5m' is recognized as Camelot 12A on the DJUCED side."""
        lib = load_djuced_keys(djuced_db_with_keys)
        row = next(r for r in crosscheck_keys(
            [{"file_name": "2000_and_one-pak_pak.mp3", "key": "C#/Db minor", "camelot_key": "12A"}],
            lib,
        ))
        assert row["djuced_camelot"] == "12A"
