# API Reference

Public entry points for programmatic use. Import the package from the repo root.

## DSP pipeline (`src/dsp/`)

- **`extractor.extract_track_features(path, config) -> Track`** — master orchestrator; runs
  Groove → Phrasing → Mood → Curation and returns a `Track`. `config` comes from the preset system.
- **`extract_feature_vector(track) -> dict`** — flattens a `Track` into the numeric dict used for
  similarity matching.
- **`config.get_config(preset) -> config`** — named presets: `default`, `minimal`, `house`,
  `techno`, `aggressive`.
- **`mood_engine.analyze_mood(y, sr, file_path=None, prefer_skey=True) -> MoodResult`** —
  key/Camelot/brightness/roughness. When the optional `skey` package is installed **and** a
  `file_path` is given, key detection uses the S-KEY deep-learning model (much better on
  electronic music; `MoodResult.key_source == "skey"`, `key_confidence` is a real softmax
  probability); otherwise it falls back to the chroma template (`key_source == "chroma"`).
  Pass `prefer_skey=False` to force chroma.
- **`mood_engine.convert_to_camelot(note, key_type) -> str`** — musical key → Camelot code
  (e.g. `"F#/Gb", "minor" -> "11A"`). **`camelot_to_open_key("12A") -> "5m"`** and
  **`open_key_to_camelot("5m") -> "12A"`** convert between Camelot and Open Key Notation
  (what DJUCED / Rekordbox "Open Key" display). **`normalize_key_to_camelot(value) -> str|None`**
  best-effort-parses a Camelot or Open Key string to Camelot (None if unrecognized), used for
  cross-tool key comparison.
- **`mood_engine.detect_key_skey(file_path, device="cpu") -> (key, camelot, confidence)`** —
  run the S-KEY model directly (raises if `skey`/torch missing; callers catch and fall back).
  **`skey_label_to_key_camelot("C# minor") -> ("C#/Db minor", "12A")`** maps S-KEY labels to
  this repo's conventions (pure, no torch).
- **`config.custom_config(min_bars=4, thresh_frac=0.4, max_pads=None) -> config`** — build a config
  from explicit phrasing params.
- **`phrasing_engine.analyze_structure(y, sr, bpm, hop_length=512, min_bars=4, thresh_frac=0.4, max_pads=None, detect_elements=False, ...) -> PhrasingResult`**
  — low-band (kick+bass) drop/breakdown segmentation. Hot cues sit at section starts; `max_pads=N`
  limits them to N controller pads (intro, main drop and outro always kept). `detect_elements=True`
  also returns element onsets. Helper functions: `compute_lowband_energy`, `smooth_lowband_energy`,
  `detect_energy_sections`, `label_energy_sections`, `map_segments_to_hotcues(segments, max_pads)`.
- **`phrasing_engine.create_phrase_locked_segments(duration, bpm, bars_per_phrase, include_beats=True) -> List[Segment]`**
  — alternative fixed-bar segmentation (every segment exactly N bars).
- **`phrasing_engine.detect_element_onsets(y, sr, bpm, n_bands, threshold, min_sustain_bars) -> List[ElementOnset]`**
  — per-band additive novelty: the bar-snapped moments a new sound element (kick, hat, synth line)
  enters, with frequency band and confidence. One-shot FX are rejected via the sustain window.
- **`phrasing_engine.derive_mix_points(onsets, bpm, duration, mix_out_bars=32) -> dict`**
  — turns element onsets into named mix points: `mix_in` (first element entry), `bass_in` (first
  sub/low entry — swap the lows here), `full_on` (all bands in), `mix_out` (phrase math before the end).
- **Time/beat/bar helpers** (`phrasing_engine`): `time_to_bar`, `bar_to_time`, `time_to_beat`,
  `beat_to_bar_group`, `snap_to_bar_boundary` — conversions used by phrasing and for snapping cues
  to bar boundaries. See `PARAMETER_REFERENCE.md` for the beat/bar and phrase-locking model.
- **`spectrogram.compute_and_save_spectrogram(y, sr, key, hop_length=512, base_dir=DEFAULT_SPECTROGRAM_DIR) -> Path`**
  — computes a log-magnitude (dB) STFT and saves it to `base_dir/{key}.npy` (`key` is typically the
  DB `track_id`); returns the saved path. `compute_spectrogram(y, sr, hop_length)` and
  `save_spectrogram(S, key, base_dir)` are the split-out steps. Default dir: `data/spectrograms`.
- **`worker.analyze_one_track(file_path, segment_preset, bars_per_phrase, spectrogram_dir, spectrogram_key) -> dict`**
  — module-level (picklable), compute-only per-track pipeline used by `Orchestrator.analyze_library`;
  never touches the DB, never raises (failures surface via the returned dict's `"error"` key). Not
  normally called directly — see `docs/architecture.md` for the parallel-analyze design.

## Database (`src/database/`)

- **`schema.init_db(db_path="db/djia.db") -> Connection`** — create schema.
- **`schema.get_connection(db_path="db/djia.db") -> Connection`** — connection with
  `sqlite3.Row` factory.
- **`store.TrackStore(db_path)`** — CRUD over tracks/features/mood/segments (default `db/djia.db`).

## Matching (`src/matching/`)

- **`similarity`** — cosine similarity over the corpus-normalized feature vector
  (`SIMILARITY_FEATURES`); filter by BPM / key / mood. The vector includes the model-free
  stem-proxy features (`sub_ratio`, `bass_ratio`, `kick_rate`, `perc_rate`, `hat_rate`,
  `vocal_presence`), so similarity is sensitive to low-end weight, the kick/perc/hat split, and
  vocal presence.
- **`clustering.cluster_library(db_path, method="average", n_clusters=None, distance_threshold=None) -> {track_id: label}`**
  — hierarchical (agglomerative) clustering over cosine distance of the same normalized vectors; cut
  the dendrogram by `n_clusters` or `distance_threshold` (defaults to 0.25 when neither is given).
  `describe_clusters(labels, db_path)` summarizes each cluster (size, BPM mean/spread, modal Camelot
  key, dominant mood, mean `sub_ratio`/`vocal_presence`, example titles).

## AI layer (`src/ai/`)

- **`transition_mapper`** — scores track-to-track transition compatibility.
- **`playlist_generator`** — builds optimal DJ sequences from transition scores.
- **`setlist_generator.generate_setlist(db_path, n_tracks=28, output_path, with_mix_sheets=True) -> path`**
  — data-driven 5-phase set (warm-up→build→peak→breakdown→comeback): phase quotas by proportion,
  phase assignment from measured mood/energy/brightness/BPM, transition-optimized ordering, and a
  markdown report with element-onset mix sheets per transition. Pure core: `build_setlist(tracks, n)`;
  its `camelot_score(a, b)` scores Camelot codes ('7A'), unlike `transition_mapper`'s note-name scorer.
  `transition_score(a, b, ascending=False)` blends BPM 35% / key 30% / mood 20% / energy 15%, then
  applies a groove/swing compatibility term (`swing_score`, from the groove engine) as a
  **multiplicative** penalty rather than a competing weight — same swing is a no-op (factor 1.0),
  full clash floors the score at `GROOVE_PENALTY_FLOOR` (0.7); either track missing `swing_score`
  is also a no-op. If `output_path` is locked (e.g. open elsewhere), the write falls back to a
  timestamped filename instead of raising.
- **`stem_separator`** — Demucs stems (Drums/Bass/Vocals/Melody), on-disk cached.
- **`classifier`** — 6-dimension mood classification.
- **`segmentation`** — structural detection (drop/breakdown/outro) with confidence.

## Track Tuner (`src/ai/track_tuner_graph.py`) — optional, LangGraph

- **`run_single_track(path, preset, max_iterations)`** — tune one track's phrasing params until
  quality ≥ 0.70 or `max_iterations`.
- **`run_batch_tracks(paths, preset)`** — batch version.

> Import tuner entry points directly from `src.ai.track_tuner_graph` — `src/ai/__init__.py` does not
> re-export them, and their deps are not in `requirements.txt`.

## DJUCED export (`src/djuced/`)

- **`exporter.export_mix_cues(track_cues, db_path=DEFAULT_DJUCED_DB, dry_run=True) -> report`**
  — write `DJIA …`-prefixed hot cues straight into DJUCED's own `DJUCED.db` (matched by fuzzy
  filename; all duplicate copies get the cues). Dry-run by default; auto-backup before the first
  real write; only DJIA-named cues are ever replaced. Close DJUCED before writing.
- **`exporter.match_djuced_tracks(file_name, library)`** / **`load_djuced_library(db_path)`** —
  filename-normalized matching between DJIA's library and DJUCED's.
- **`exporter.load_djuced_keys(db_path) -> list`** — like `load_djuced_library` but also returns
  DJUCED's stored `key_raw` per track (PRAGMA-detected column; `None` when absent).
  **`exporter.crosscheck_keys(djia_tracks, djuced_library) -> list`** — compares each DJIA
  track's Camelot key against DJUCED's (both normalized to Camelot), returning per-track
  `status`: `match` / `diff` / `unreadable` / `no_djuced_key` / `no_djuced_match`. Drives the
  `crosscheck-djuced` CLI command.

## Traktor export (`src/traktor/`)

- **`exporter.export_all_tracks(nml_path, db_path, output_path)`** — add BPM/cues from the DB to a
  Traktor `Collection.nml` copy.

## Orchestration

- **`orchestrator.Orchestrator(db_path="db/djia.db", segment_preset="minimal", bars_per_phrase=16, spectrogram_dir=DEFAULT_SPECTROGRAM_DIR)`**
  — ties ingestion → DSP → AI → DB; drives the CLI.
- **`Orchestrator.analyze_library(data_dir="data", skip_existing=False, workers=1) -> dict`** —
  `workers<=1` runs the per-track compute sequentially in-process; `workers>1` fans it out across a
  `ProcessPoolExecutor` (via `dsp.worker.analyze_one_track`). All DB writes stay serial in the main
  process regardless of `workers`. Returns `{'analyzed', 'skipped', 'errors'}`.
- **`Orchestrator.analyze_single_track(file_path) -> dict | None`** — analyzes one file through the
  same worker pipeline as `analyze_library` (`dsp.worker.analyze_one_track`) and **persists** it:
  registers the track (or reuses its `track_id` if already in the DB), writes features/segments/mood,
  and returns the feature dict plus `track_id`. Returns `None` if the file doesn't exist or analysis
  fails. Backs `cli.py`'s `analyze --track PATH`.

Data shapes returned by these APIs are documented in `docs/schemas.md`.
