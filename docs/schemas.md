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
  `cue_points: List[CuePoint]`, `structure_confidence: float`.
- **`GrooveResult`** — `bpm: float`, `beat_grid: np.ndarray`, `beat_times: List[float]`,
  `swing_score: float` (0=stiff, 1=groovy), `tempo_stability: bool`, `stability_variance: float`.
- **`MoodResult`** — `key: str` (e.g. "A minor"), `camelot_key: str` (e.g. "7A"),
  `brightness: float` (0=dark, 1=bright), `key_confidence: float`.
- **`CurationResult`** — `danceability: float`, `energy_curve: np.ndarray`, `energy_type: str`
  ("flat"|"dynamic"|"gradual"), `semantic_tags: List[str]`, `complexity_score: float`.

Building blocks:

- **`Segment`** — `start_time`, `end_time` (seconds), `label` (intro/build/drop/breakdown/outro),
  `confidence`.
- **`CuePoint`** — `time` (seconds), `label` (e.g. "Pad 1"), `type` (default "hotcue").

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
  RMS (`rms_mean/std/peak`).
- **`mood`** — `track_id` UNIQUE FK + 6 mood dimensions: `dark`, `hypnotic`, `euphoric`,
  `aggressive`, `industrial`, `minimal`.
- **`segments`** — `track_id` FK, `segment_type`, `start_time`, `end_time`, `confidence`.

Indices exist on `tracks(file_path/artist/title)`, `features(track_id)`, `mood(track_id)`,
`segments(track_id/segment_type)`.

`init_db(db_path)` creates the schema; `get_connection(db_path)` returns a connection with
`row_factory = sqlite3.Row`.
