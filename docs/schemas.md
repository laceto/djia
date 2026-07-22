# Schemas

## `Track` dataclass — the data contract (`src/features/schema.py`)

`Track` is the object every part of the pipeline produces or consumes.

```python
Track(
    file_path: str,
    duration: float,                    # seconds
    phrasing: PhrasingResult,           # Phase 2 DSP
    groove: GrooveResult,
    mood: MoodResult,
    curation: CurationResult,
    sample_rate: int = 22050,
    analysis_timestamp: Optional[str] = None,   # ISO; auto-set in __post_init__
)
```

Component results:

- **`PhrasingResult`** — `segment_boundaries: List[float]`, `segments: List[Segment]`,
  `cue_points: List[CuePoint]`, `structure_confidence: float`,
  `element_onsets: List[ElementOnset]` (empty unless element detection was run).
- **`GrooveResult`** — `bpm: float`, `beat_grid: np.ndarray`, `beat_times: List[float]`,
  `swing_score: float` (0=stiff, 1=groovy), `tempo_stability: bool`, `stability_variance: float`,
  `onset_strength_mean/std: float` (transient hardness — how punchy/dominant the kick is),
  `beat_strength: float` (0-1, how dominant the detected tempo's periodicity is vs. any other
  in the track — from a tempogram).
- **`MoodResult`** — `key: str` (e.g. "A minor"), `camelot_key: str` (e.g. "7A"),
  `brightness: float` (0=dark, 1=bright), `key_confidence: float`,
  `zero_crossing_rate: float` (waveform sign-change rate; higher = noisier/acid-ish),
  `roughness: float` (0-1, Plomp-Levelt-style timbral roughness — smooth/consonant vs.
  rough/dissonant).
- **`CurationResult`** — `danceability: float`, `energy_curve: np.ndarray`, `energy_type: str`
  ("flat"|"dynamic"|"gradual"), `semantic_tags: List[str]`, `complexity_score: float`,
  `spectral_flatness: float` (0=tonal/clean, 1=noise-like/saturated — Wiener entropy),
  `crest_factor: float` (peak-to-average RMS ratio; high = punchy/dynamic, near 1 = compressed).

Building blocks:

- **`Segment`** — `start_time`, `end_time` (seconds), `label` (intro/build/drop/breakdown/outro),
  `confidence`.
- **`CuePoint`** — `time` (seconds), `label` (e.g. "Pad 1"), `type` (default "hotcue").
- **`ElementOnset`** — the moment a new sound element enters: `time` (seconds, bar-snapped),
  `band` (frequency-band index, 0 = lowest), `freq_low`/`freq_high` (Hz), `confidence` (0-1),
  `label` (DJ-EQ band hint: "sub"/"low"/"low-mid"/"mid"/"high-mid"/"high").

Wrappers / helpers:

- **`AnalysisResult`** — `track: Track`, `status: str` ("success"|"error"), `error_message`.
- **`create_test_track(file_path, duration)`** — builds a minimal valid `Track` for tests.

## SQLite schema (`src/database/schema.py`)

Default DB: `data/djia.db`. Foreign keys ON; `ON DELETE CASCADE` from `tracks`.

- **`tracks`** — `id` PK, `file_path` UNIQUE, `file_name`, `format`, `artist`, `title`, `album`,
  `duration`, `analysis_date`, `created_at`, `updated_at`.
- **`features`** — `track_id` UNIQUE FK, `bpm`, spectral stats (`spectral_centroid_mean/std`,
  `spectral_rolloff_mean`, `spectral_flux_mean`), `harmonic_ratio`, `percussive_ratio`, MFCC stats
  (`mfcc_mean/std/delta_mean`, `mfcc_vector` as TEXT), chroma (`chroma_variance`, `chroma_entropy`),
  RMS (`rms_mean/std/peak`), `key`/`camelot_key`/`key_confidence`, `swing_score`
  (0=straight, 1=swung, from the groove engine), `spectral_flatness` (curation engine),
  `crest_factor` (curation engine — derivable from `rms_mean`/`rms_peak`, so backfillable without
  re-analysis), `onset_strength_mean`/`onset_strength_std`/`beat_strength` (groove engine),
  `zero_crossing_rate`/`roughness` (mood engine). These 7 columns are NULL on tracks analyzed
  before this feature shipped — see `debugging-rules.md`.
- **`mood`** — `track_id` UNIQUE FK + 6 mood dimensions: `dark`, `hypnotic`, `euphoric`,
  `aggressive`, `industrial`, `minimal`.
- **`segments`** — `track_id` FK, `segment_type`, `start_time`, `end_time`, `confidence`,
  `method` (`'spectral'` = novelty detection with the orchestrator's preset, default `minimal`;
  `'phrase<N>'` = fixed N-bar phrase grid, default `phrase16`). `analyze` writes both sets per
  track via `TrackStore.replace_segments` (idempotent per method).

Indices exist on `tracks(file_path/artist/title)`, `features(track_id)`, `mood(track_id)`,
`segments(track_id/segment_type)`.

`init_db(db_path)` creates the schema; `get_connection(db_path)` returns a connection with
`row_factory = sqlite3.Row`.
