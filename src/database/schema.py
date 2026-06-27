"""SQLite database schema for DJIA."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# SQL table definitions
SCHEMA = """
-- Tracks table: core track information
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,
    format TEXT NOT NULL,
    artist TEXT,
    title TEXT,
    album TEXT,
    duration REAL NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Features table: audio analysis features
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER UNIQUE NOT NULL,
    bpm REAL,
    spectral_centroid_mean REAL,
    spectral_centroid_std REAL,
    spectral_rolloff_mean REAL,
    spectral_flux_mean REAL,
    harmonic_ratio REAL,
    percussive_ratio REAL,
    mfcc_mean REAL,
    mfcc_std REAL,
    mfcc_delta_mean REAL,
    chroma_variance REAL,
    chroma_entropy REAL,
    rms_mean REAL,
    rms_std REAL,
    rms_peak REAL,
    mfcc_vector TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

-- Mood table: track mood classification with confidence scores
CREATE TABLE IF NOT EXISTS mood (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER UNIQUE NOT NULL,
    dark REAL DEFAULT 0.0,
    hypnotic REAL DEFAULT 0.0,
    euphoric REAL DEFAULT 0.0,
    aggressive REAL DEFAULT 0.0,
    industrial REAL DEFAULT 0.0,
    minimal REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

-- Segments table: track structure segments
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    segment_type TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks (file_path);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks (artist);
CREATE INDEX IF NOT EXISTS idx_tracks_title ON tracks (title);
CREATE INDEX IF NOT EXISTS idx_features_track_id ON features (track_id);
CREATE INDEX IF NOT EXISTS idx_mood_track_id ON mood (track_id);
CREATE INDEX IF NOT EXISTS idx_segments_track_id ON segments (track_id);
CREATE INDEX IF NOT EXISTS idx_segments_type ON segments (segment_type);
"""


def init_db(db_path: str = "data/djia.db") -> sqlite3.Connection:
    """
    Initialize database with schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Database connection
    """
    try:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys

        # Create all tables
        for statement in SCHEMA.split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(statement)

        conn.commit()
        logger.info(f"Database initialized at {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def get_connection(db_path: str = "data/djia.db") -> sqlite3.Connection:
    """
    Get or create database connection.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Database connection
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn
