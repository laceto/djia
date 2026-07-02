# API Reference

Public entry points for programmatic use. Import the package from the repo root.

## DSP pipeline (`src/dsp/`)

- **`extractor.extract_track_features(path, config) -> Track`** — master orchestrator; runs
  Groove → Phrasing → Mood → Curation and returns a `Track`. `config` comes from the preset system.
- **`extract_feature_vector(track) -> dict`** — flattens a `Track` into the numeric dict used for
  similarity matching.
- **`config.get_config(preset) -> config`** — named presets: `default`, `minimal`, `house`,
  `techno`, `aggressive`.
- **`config.custom_config(...) -> config`** — build a config from explicit phrasing params
  (`novelty_threshold`, `min_segment_duration`, `breakdown_duration_threshold`).
- **`phrasing_engine.analyze_structure(y, sr, bpm, novelty_threshold, min_segment_duration, breakdown_threshold, include_beats=True) -> PhrasingResult`**
  — spectral-novelty segmentation; segment labels carry beat/bar ranges when `include_beats=True`.
- **`phrasing_engine.create_phrase_locked_segments(duration, bpm, bars_per_phrase, include_beats=True) -> List[Segment]`**
  — alternative fixed-bar segmentation (every segment exactly N bars).
- **Time/beat/bar helpers** (`phrasing_engine`): `time_to_bar`, `bar_to_time`, `time_to_beat`,
  `beat_to_bar_group`, `snap_to_bar_boundary` — conversions used by phrasing and for snapping cues
  to bar boundaries. See `PARAMETER_REFERENCE.md` for the beat/bar and phrase-locking model.

## Database (`src/database/`)

- **`schema.init_db(db_path="data/djia.db") -> Connection`** — create schema.
- **`schema.get_connection(db_path="data/djia.db") -> Connection`** — connection with
  `sqlite3.Row` factory.
- **`store.TrackStore(db_path)`** — CRUD over tracks/features/mood/segments (default `data/djia.db`).

## Matching (`src/matching/`)

- **`similarity`** — cosine similarity over feature vectors; filter by BPM / key / mood.

## AI layer (`src/ai/`)

- **`transition_mapper`** — scores track-to-track transition compatibility.
- **`playlist_generator`** — builds optimal DJ sequences from transition scores.
- **`stem_separator`** — Demucs stems (Drums/Bass/Vocals/Melody), on-disk cached.
- **`classifier`** — 6-dimension mood classification.
- **`segmentation`** — structural detection (drop/breakdown/outro) with confidence.

## Track Tuner (`src/ai/track_tuner_graph.py`) — optional, LangGraph

- **`run_single_track(path, preset, max_iterations)`** — tune one track's phrasing params until
  quality ≥ 0.70 or `max_iterations`.
- **`run_batch_tracks(paths, preset)`** — batch version.

> Import tuner entry points directly from `src.ai.track_tuner_graph` — `src/ai/__init__.py` does not
> re-export them, and their deps are not in `requirements.txt`.

## Orchestration

- **`orchestrator.Orchestrator`** — ties ingestion → DSP → AI → DB; drives the CLI.

Data shapes returned by these APIs are documented in `docs/schemas.md`.
