# Architecture

The system is a **DSP feature-extraction pipeline** with a database/export layer on top, plus an
**optional LangGraph agent** for auto-tuning segmentation parameters. Everything hangs off the
`Track` dataclass in `src/features/schema.py` (see `docs/schemas.md`).

## Data flow

```
Audio files
  → ingestion (scan + librosa load, 22,050 Hz mono)
  → DSP pipeline (Groove → Phrasing → Mood → Curation)  →  Track
  → AI layer (stems, mood classification, segmentation)
  → database (SQLite) + matching (similarity) + traktor (NML export)
```

`src/orchestrator.py` (`Orchestrator`) ties ingestion → DSP → AI → DB together and is what the CLI
(`src/cli.py`) drives.

## DSP pipeline (`src/dsp/`) — the core

`extractor.extract_track_features(path, config)` is the master orchestrator. **Engine order matters
because of data dependencies:**

1. **Groove** (`groove_engine.py`) runs first → BPM, beat grid, swing. BPM feeds everything downstream.
2. **Phrasing** (`phrasing_engine.py`) → structural segments (intro/build/drop/breakdown/outro) +
   hot-cue positions. Takes BPM and the tunable phrasing params. Segments carry beat/bar ranges when
   `include_beats=True`. `time_to_bar(seconds, bpm)` is the shared time↔bar conversion.
3. **Mood** (`mood_engine.py`) → Camelot key + brightness. Independent.
4. **Curation** (`curation_engine.py`) → danceability, energy curve, semantic tags. Consumes BPM,
   swing, brightness.

`extract_feature_vector(track)` flattens a `Track` into the numeric dict used for similarity matching.

## Config / preset system (`src/dsp/config.py`)

Segmentation behavior is driven entirely by three phrasing parameters: `novelty_threshold` (peak
sensitivity), `min_segment_duration`, `breakdown_duration_threshold`. Named presets (`default`,
`minimal`, `house`, `techno`, `aggressive`) trade off segment count vs. length — lower threshold +
shorter min duration = more, shorter segments. Get a config via `get_config(preset)` or build one
with `custom_config(...)`, then pass it as the `config=` arg to `extract_track_features`. Parameter
meanings are documented in `PARAMETER_REFERENCE.md`.

## LangGraph Track Tuner (`src/ai/track_tuner_*.py`) — optional, self-contained

An agent that iteratively tunes the phrasing params per track until segmentation quality is "good"
(≥0.70) or `max_iterations` is hit. Flow:

```
load_track → initialize_config → analyze_track → evaluate_quality
           → (suggest_tuning → analyze_track)* → finalize
```

State is a `TypedDict` with `operator.add` / `add_messages` reducers (`track_tuner_state.py`); nodes
are pure `(state, config) -> dict` and emit `[NodeName]`-prefixed `AIMessage`s for tracing. Entry
points: `run_single_track(path, preset, max_iterations)` and `run_batch_tracks(paths, preset)`.
Quality scoring lives in `evaluate_quality`; full rubric in `LANGGRAPH_TRACK_TUNER_README.md`.

The preset dicts here (`DEFAULT_CONFIGS` in `track_tuner_state.py`) mirror `dsp/config.py` — keep
them consistent.

## AI layer (`src/ai/`)

- `stem_separator.py` — Demucs (Drums/Bass/Vocals/Melody) with on-disk caching; first run downloads ~1GB.
- `classifier.py` — 6-dimension mood classification.
- `segmentation.py` — structural detection (drop/breakdown/outro) with confidence.
- `processor.py` — runs DSP on separated stems.
- `transition_mapper.py` — scores track-to-track transition compatibility.
- `playlist_generator.py` — builds optimal DJ sequences from transition scores.

## Data store & export

- `database/schema.py` + `database/store.py` — SQLite (`TrackStore`); default DB is `data/djia.db`.
- `matching/similarity.py` — cosine similarity over feature vectors, filterable by BPM/key/mood.
- `traktor/exporter.py` — writes Traktor NML with BPM, key, and auto hot cues.

## Ingestion

`src/ingestion/{scanner,loader}.py` handle file discovery and librosa loading (resampled to
22,050 Hz mono).

## Legacy

`src/main.py`, `audio_analysis.py`, `mixing_metrics.py`, `structure_detection.py` are an earlier
standalone implementation kept for backward compatibility; the `src/dsp` + `src/ai` pipeline
supersedes it. Don't extend these — add to the phased pipeline instead.
