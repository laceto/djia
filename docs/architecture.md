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
  → database (SQLite) + matching (similarity) + traktor (NML export) + djuced (hot-cue export)
```

`src/orchestrator.py` (`Orchestrator`) ties ingestion → DSP → AI → DB together and is what the CLI
(`src/cli.py`) drives.

## DSP pipeline (`src/dsp/`) — the core

`extractor.extract_track_features(path, config)` is the master orchestrator. **Engine order matters
because of data dependencies:**

1. **Groove** (`groove_engine.py`) runs first → BPM, beat grid, swing. BPM feeds everything downstream.
   `compute_swing_score` measures swing from **offbeat beat-phase** (median phase of off-grid onsets
   relative to the beat, not an off-grid-onset ratio) — 0.0 straight/machine grid, 1.0 full triplet
   swing. Also computes **onset strength** (`compute_onset_strength_stats` — mean/std of the onset
   envelope already needed for beat tracking; transient hardness/kick punch) and **beat strength**
   (`compute_beat_strength` — 0-1, tempogram-based: how dominant the detected tempo's periodicity is
   vs. any other periodicity in the track).
2. **Phrasing** (`phrasing_engine.py`) → structural segments (intro/build/drop/breakdown/outro) +
   hot-cue positions. Takes BPM and the tunable phrasing params. Segments carry beat/bar ranges when
   `include_beats=True`. `time_to_bar(seconds, bpm)` is the shared time↔bar conversion.
   Also hosts **element-onset detection** (`detect_element_onsets` — per-band additive novelty
   marking where new sound elements enter; opt-in via `analyze_structure(detect_elements=True)` or
   direct call) and `derive_mix_points(onsets, bpm, duration)` which turns onsets into named
   mix points (mix_in / bass_in / full_on / mix_out) for DJ use.
3. **Mood** (`mood_engine.py`) → Camelot key + brightness. Independent. Also computes **zero-crossing
   rate** (`compute_zero_crossing_rate`) and **timbral roughness** (`compute_roughness` — a pragmatic
   Sethares/Plomp-Levelt pairwise-dissonance approximation over each frame's loudest spectral peaks,
   tanh-squashed to 0-1: smooth/consonant vs. rough/dissonant).
4. **Curation** (`curation_engine.py`) → danceability, energy curve, semantic tags. Consumes BPM,
   swing, brightness. Also computes **spectral flatness** (`compute_spectral_flatness` — Wiener
   entropy, 0=tonal/clean to 1=noise-like/saturated) and **crest factor** (`compute_crest_factor` —
   peak-to-average RMS ratio; high = punchy/dynamic, near 1 = compressed).

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
- `transition_mapper.py` — scores track-to-track transition compatibility (BPM/key/mood/energy,
  plus a groove/swing term for pair-mix scoring).
- `playlist_generator.py` — builds optimal DJ sequences from transition scores.
- `setlist_generator.py` — data-driven 5-phase set builder (warm-up → build → peak → breakdown →
  comeback). Pipeline: `phase_quotas` (proportional track counts per phase) → `assign_phases`
  (global greedy affinity matching — strongest track/phase fits claim slots first) →
  `order_setlist` (greedy transition-chaining within/across phases, same BPM/key/mood/energy
  weights as `transition_mapper`) → `render_report` (markdown phase plan + per-transition mix
  sheets using cached element-onset mix points). Entry point: `generate_setlist(db_path, n_tracks,
  output_path, with_mix_sheets)`.

## Data store & export

- `database/schema.py` + `database/store.py` — SQLite (`TrackStore`); default DB is `data/djia.db`.
  `insert_features` persists `swing_score`/Camelot key plus 7 density/onset/timbre columns
  (`spectral_flatness`, `crest_factor`, `onset_strength_mean/std`, `beat_strength`,
  `zero_crossing_rate`, `roughness`) on the `features` table; `replace_segments` persists
  phrasing-engine structure segments to the `segments` table (idempotent per `method` —
  re-analysis replaces rather than duplicates). All are merged into the features dict by
  `orchestrator.py`'s `_add_tonality`/`_add_swing`/`_add_density` (best-effort, called from both
  `analyze_library` and `analyze_single_track`) before `insert_features` during `analyze`, so
  tracks analyzed before a given feature shipped have `NULL`/zero values until re-analyzed.
  Note: `orchestrator.py` calls `groove_engine`/`mood_engine`/two standalone `curation_engine`
  functions directly for the DB-persisted path — it does **not** run the full
  `curation_engine.analyze_curation` (danceability/energy_type/semantic_tags/complexity_score are
  only computed via the standalone `extractor.extract_track_features` path, not persisted to the DB
  through `analyze`).
- `matching/similarity.py` — cosine similarity over feature vectors, filterable by BPM/key/mood.
- `traktor/exporter.py` — writes Traktor NML with BPM, key, and auto hot cues.
- `djuced/exporter.py` — writes `DJIA …`-prefixed hot cues directly into DJUCED's own
  `DJUCED.db` (Hercules controllers), matched by fuzzy filename. Dry-run by default,
  auto-backup before the first real write; DJUCED must be closed while writing.

## Ingestion

`src/ingestion/{scanner,loader}.py` handle file discovery and librosa loading (resampled to
22,050 Hz mono).

## Legacy

`src/main.py`, `audio_analysis.py`, `mixing_metrics.py`, `structure_detection.py` are an earlier
standalone implementation kept for backward compatibility; the `src/dsp` + `src/ai` pipeline
supersedes it. Don't extend these — add to the phased pipeline instead.
