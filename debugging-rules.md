# Debugging Rules

Known failure modes and environment quirks. Read when a script errors or output looks wrong.

## Gotchas

- **Run the CLI as a module** (`python -m src.cli ...`) from the repo root. The package uses relative
  imports and won't run as a loose script — `ModuleNotFoundError` / `ImportError` almost always means
  you ran it the wrong way.

- **LangGraph deps are NOT in `requirements.txt`.** The Track Tuner needs `langgraph` and
  `langchain-core`; install them separately. `src/ai/__init__.py` deliberately does **not** import the
  track_tuner modules (so the CLI keeps working without those packages) — import them directly from
  `src.ai.track_tuner_graph`. An `ImportError` from `src.ai` on tuner code means the deps are missing.

- **Reset a locked/corrupt DB** by deleting `data/djia.db`; it is recreated on the next `analyze` run.

- **`data/` and `results/` are gitignored** (audio + generated artifacts were removed from the repo in
  commit 9099368). Don't commit tracks, `.db` files, or NML output.

- **Demucs first run downloads ~1GB** (stem separation model), then caches on disk. A slow/hanging
  first stem separation is usually the model download, not a bug.

- **`NULL swing_score` / zero `segments` rows on older tracks**: swing-score and structure-segment
  persistence (`features.swing_score`, the `segments` table) were added after some tracks were
  already analyzed. If a track shows `NULL`/empty for either, re-run `analyze` on it — the column
  or table isn't the problem, the track just predates that feature. The same applies to the 7
  density/onset/timbre columns (`spectral_flatness`, `onset_strength_mean/std`, `beat_strength`,
  `zero_crossing_rate`, `roughness`, and `crest_factor` on tracks analyzed before the backfill —
  see `docs/schemas.md`).

- **Structure segments were silently never persisted via the `analyze` CLI, even though the
  feature "shipped"**: `orchestrator.py` called `self._add_segments(track_id, y, sr,
  features.get('bpm'), ...)`, but `analyze_audio()` (`audio_analysis.py`) only ever sets a
  `'tempo'` key, never `'bpm'` — so `features.get('bpm')` was always `None` and every analysis run
  hit the "No BPM ... skipping segment detection" warning and silently no-op'd. Fixed by aliasing
  `features.setdefault('bpm', features.get('tempo'))` right after `analyze_audio()` returns. If you
  see this warning again after that fix, something upstream (not this key mismatch) is actually
  failing to produce a BPM.

- **`generate-setlist` raises `ValueError: Setlist needs N tracks but library has M`** — `n_tracks`
  (`--tracks`) exceeds the number of fully-analyzed tracks (BPM present) in the DB. Lower
  `--tracks` or analyze more of the library first.

- **Setlist mix sheets look stale after re-analyzing a track** — `generate_setlist`'s mix points are
  cached by file name in `results/mix_points_cache.json` and never recomputed once cached. Delete
  the relevant entry (or the whole file) to force a recompute on the next run.

## Where to look

- Data flow / who-produces-what → `docs/architecture.md`.
- Data shapes (Track dataclass, DB schema) → `docs/schemas.md`.
