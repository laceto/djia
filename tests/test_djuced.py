"""Tests for the DJUCED hot-cue exporter (temp DB mimicking DJUCED's schema)."""

import sqlite3

import pytest

from src.djuced.exporter import (
    CUE_PREFIX,
    export_mix_cues,
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
