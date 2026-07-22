"""One-off backfill: crest_factor = rms_peak / rms_mean for existing rows.

Unlike the other new DSP metrics (spectral_flatness, onset_strength, beat_strength,
zero_crossing_rate, roughness), crest_factor is fully derivable from rms_mean/rms_peak,
which were already persisted before this feature shipped — so existing tracks don't
need to be re-analyzed to get it.
"""
import sqlite3

DB_PATH = "data/djia.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.execute(
    "SELECT id, rms_mean, rms_peak FROM features "
    "WHERE crest_factor IS NULL AND rms_mean IS NOT NULL AND rms_mean != 0 AND rms_peak IS NOT NULL"
)
rows = cur.fetchall()

updates = [(rms_peak / rms_mean, fid) for fid, rms_mean, rms_peak in rows]
conn.executemany("UPDATE features SET crest_factor = ? WHERE id = ?", updates)
conn.commit()

print(f"Backfilled crest_factor for {len(updates)} row(s)")
conn.close()
