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
2. **Phrasing** (`phrasing_engine.py`) → structural segments (intro/drop/breakdown/outro) +
   hot-cue positions. Structure is detected from **kick+bass (20–150 Hz) low-band energy**:
   `compute_lowband_energy` → `smooth_lowband_energy` → `detect_energy_sections` (kick-on = drop,
   kick-off = breakdown) → `label_energy_sections` → `map_segments_to_hotcues`. Takes BPM and the
   tunable phrasing params (`min_bars`, `thresh_frac`, `max_pads`). `time_to_bar(seconds, bpm)` is the
   shared time↔bar conversion. Also hosts **element-onset detection** (`detect_element_onsets` —
   per-band additive novelty marking where new sound elements enter; opt-in via
   `analyze_structure(detect_elements=True)` or direct call) and
   `derive_mix_points(onsets, bpm, duration)` which turns onsets into named mix points
   (mix_in / bass_in / full_on / mix_out) for DJ use. `create_phrase_locked_segments` remains as an
   alternative fixed-N-bar strategy.
3. **Mood** (`mood_engine.py`) → Camelot key + brightness. Independent. Also computes **zero-crossing
   rate** (`compute_zero_crossing_rate`) and **timbral roughness** (`compute_roughness` — a pragmatic
   Sethares/Plomp-Levelt pairwise-dissonance approximation over each frame's loudest spectral peaks,
   tanh-squashed to 0-1: smooth/consonant vs. rough/dissonant).
   **Key backend:** by default a Krumhansl chroma-template match (weak on electronic music — near-constant
   confidence). If the optional **S-KEY** deep-learning model (`skey` package) is installed and a
   `file_path` reaches `analyze_mood`, it is used instead and `MoodResult.key_source` becomes `"skey"`
   with a real softmax confidence; otherwise the chroma method runs and `key_source` is `"chroma"`.
   Camelot codes convert to/from DJUCED's **Open Key Notation** via `camelot_to_open_key` /
   `open_key_to_camelot` (12A ↔ 5m).
4. **Curation** (`curation_engine.py`) → danceability, energy curve, semantic tags. Consumes BPM,
   swing, brightness. Also computes **spectral flatness** (`compute_spectral_flatness` — Wiener
   entropy, 0=tonal/clean to 1=noise-like/saturated) and **crest factor** (`compute_crest_factor` —
   peak-to-average RMS ratio; high = punchy/dynamic, near 1 = compressed).

Beyond the four ordered engines, `stem_profile.compute_stem_profile` derives **model-free
stem-proxy features** (Demucs-independent) from one STFT + one HPSS: `sub_ratio`/`bass_ratio` (low-end
energy shares), `kick_rate`/`perc_rate`/`hat_rate` (onset rates in the low/mid/high transient bands),
and `vocal_presence` (harmonic energy in the vocal band). It is not part of the ordered pipeline;
`dsp/worker.py` runs it per track for the DB-persisted `analyze` path (see Data store & export), and
its columns feed the similarity/clustering vector.

`extract_feature_vector(track)` flattens a `Track` into the numeric dict used for similarity matching.

## Config / preset system (`src/dsp/config.py`)

Segmentation behavior is driven by three phrasing parameters: `min_bars` (minimum section length;
shorter sections merge into the previous), `thresh_frac` (kick-on threshold as a fraction of peak
low-band energy), and `max_pads` (limit hot cues to a controller's pad count, `None` = one per
section). Named presets (`default`, `minimal`, `house`, `techno`, `aggressive`) trade off section
count vs. length — lower `thresh_frac` + smaller `min_bars` = more, shorter sections. Get a config
via `get_config(preset)` or build one with `custom_config(min_bars, thresh_frac, max_pads)`, then
pass it as the `config=` arg to `extract_track_features`. Parameter meanings are documented in
`PARAMETER_REFERENCE.md`.

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

- `stem_separator.py` — stems (Drums/Bass/Vocals/Melody) with on-disk caching; first run downloads the
  model. Backend is auto-selected from the model name: `demucs` (default, ~1GB htdemucs) or
  `audio-separator` (MelBand/BS Roformer; `pip install "djia[roformer]"`) — the preferred choice for
  techno. Both share the same `{name: (channels, samples)}` output contract.
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
  weights as `transition_mapper`, plus a groove/swing term applied as a multiplicative penalty
  rather than a competing weight — no-op when either track's `swing_score` is missing) →
  `render_report` (markdown phase plan + per-transition mix sheets using cached element-onset mix
  points). Entry point: `generate_setlist(db_path, n_tracks, output_path, with_mix_sheets)` — falls
  back to a timestamped output filename if the primary path is locked, instead of raising.

## Data store & export

- `database/schema.py` + `database/store.py` — SQLite (`TrackStore`); default DB is `db/djia.db`.
  `insert_features` persists `swing_score`/Camelot key plus the density/onset/timbre columns
  (`spectral_flatness`, `crest_factor`, `onset_strength_mean/std`, `beat_strength`,
  `zero_crossing_rate`, `roughness`) and the model-free stem-proxy columns (`sub_ratio`,
  `bass_ratio`, `kick_rate`, `perc_rate`, `hat_rate`, `vocal_presence` — from `dsp/stem_profile.py`)
  on the `features` table; `replace_segments` persists
  phrasing-engine structure segments to the `segments` table (idempotent per `method` —
  re-analysis replaces rather than duplicates). All are merged into the features dict by
  `dsp/worker.py`'s `_add_tonality`/`_add_swing`/`_add_density`/`_add_stem_profile` (best-effort, called from
  `analyze_one_track` — the shared compute-only pipeline both `Orchestrator.analyze_library` and
  `Orchestrator.analyze_single_track` run per file) before `insert_features` during `analyze`, so
  tracks analyzed before a given feature shipped have `NULL`/zero values until re-analyzed.
  Note: `dsp/worker.py` calls `groove_engine`/`mood_engine`/two standalone `curation_engine`
  functions directly for the DB-persisted path — it does **not** run the full
  `curation_engine.analyze_curation` (danceability/energy_type/semantic_tags/complexity_score are
  only computed via the standalone `extractor.extract_track_features` path, not persisted to the DB
  through `analyze`). `Orchestrator.analyze_single_track` persists exactly like `analyze_library`
  (registers/reuses the track's `track_id`, runs `analyze_one_track`, writes through the same
  `_persist_result` path) — it is not a preview-only/dry-run method.
- **Parallel analyze (`workers>1`)**: `Orchestrator.analyze_library` can fan out per-track compute
  (audio load through mood classification) to a `ProcessPoolExecutor`, dispatching the module-level
  `analyze_one_track` (`src/dsp/worker.py`). DB writes remain exclusively serial in the main process —
  every `insert_features`/`replace_segments`/`insert_mood` call happens only after a worker's result
  comes back, for any `workers` value. This invariant must not be violated by future changes: a worker
  process must never open its own connection to `db/djia.db`.
- `matching/similarity.py` — cosine similarity over feature vectors, filterable by BPM/key/mood.
- `matching/clustering.py` — hierarchical (agglomerative) clustering of the library over the same
  normalized vectors (`cluster_library` / `describe_clusters`); powers the `cluster-library` CLI.
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
