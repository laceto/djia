# DJIA: DJ Mixing Analytics

Analytics system for techno DJing to improve mixing quality through data-driven insights on audio tracks.

## Overview

DJIA analyzes your audio library to extract features, classify mood, detect structure, and suggest optimal track transitions for DJ sets. It provides both a command-line interface and programmatic API.

> **Working on the code?** Start at `CLAUDE.md` — it is a task router that points you to the right
> rule file (`coding-rules.md`, `testing-rules.md`, `debugging-rules.md`) and reference docs under
> `docs/` (`architecture.md`, `schemas.md`, `api-reference.md`, `scripts-reference.md`). Segmentation
> tuning lives in `PARAMETER_REFERENCE.md`; the LangGraph Track Tuner in `LANGGRAPH_TRACK_TUNER_README.md`.

**Key Features:**
- Automatic audio feature extraction (BPM, spectral analysis, harmonic content)
- Mood classification (dark, hypnotic, euphoric, aggressive, industrial, minimal)
- Track structure detection (drops, breakdowns, transitions)
- Smart transition scoring based on BPM, key, mood, and energy
- DJ playlist generation with optimal transitions
- Element-onset detection (where new sound elements enter) with derived mix points
- Traktor NML export for direct DJ software integration
- DJUCED hot-cue export (mix marks straight onto Hercules controller pads)
- SQLite database for persistent analysis results

## System Architecture

DJIA consists of 5 phases integrated through a master orchestrator:

### Phase 1: Ingestion
- Recursive directory scanning for audio files
- Multi-format metadata extraction (MP3, FLAC, OGG, WAV, M4A, AAC, WMA)
- Audio loading and resampling to 22050 Hz

**Modules:** `src/ingestion/scanner.py`, `src/ingestion/loader.py`

### Phase 2: DSP (Digital Signal Processing)
Four ordered engines (Groove → Phrasing → Mood → Curation), orchestrated by `extractor.py`:
- **Groove** — decimal BPM, beat grid, swing (runs first; BPM feeds everything downstream)
- **Phrasing** — structural segments (intro/build/drop/breakdown/outro) + hot-cue positions,
  plus opt-in element-onset detection and mix-point derivation (`derive_mix_points`)
- **Mood** — musical key (Camelot) + brightness
- **Curation** — danceability, energy curve, semantic tags

**Modules:** `src/dsp/` (`extractor.py`, `groove_engine.py`, `phrasing_engine.py`,
`mood_engine.py`, `curation_engine.py`, `config.py`). See `docs/architecture.md`.

### Phase 3: AI (Artificial Intelligence)
- Deep learning mood classification (dark, hypnotic, euphoric, aggressive, industrial, minimal)
- Danceability scoring
- Stem separation for advanced analysis
- Acoustic feature interpretation

**Modules:** `src/ai/classifier.py`, `src/ai/`, `src/dsp/mood_engine.py`

### Phase 4: Database & Export
- SQLite storage of tracks, features, mood, and segments
- Query interface for track analysis results
- Traktor NML export for DJ software integration
- DJUCED hot-cue export: writes DJIA mix points into DJUCED's own database (dry-run by
  default, auto-backup, only DJIA-named cues replaced)

**Modules:** `src/database/`, `src/traktor/`, `src/djuced/`

### Phase 5: Advanced AI (Optional)
- Transition quality scoring between tracks
- Transition graph construction
- Smart playlist generation with optimization
- Data-driven 5-phase setlist generation (warm-up→build→peak→breakdown→comeback) with
  per-transition mix sheets

**Modules:** `src/ai/transition_mapper.py`, `src/ai/playlist_generator.py`, `src/ai/setlist_generator.py`

### Optional: LangGraph Track Tuner
A self-contained agent (`src/ai/track_tuner_*.py`) that iteratively tunes the phrasing parameters
per track until segmentation quality is good. Its deps (`langgraph`, `langchain-core`) are **not** in
`requirements.txt` — install separately. See `LANGGRAPH_TRACK_TUNER_README.md`.

## Installation

### Requirements
- Python 3.10+
- Virtual environment recommended

### Setup

1. **Clone and activate environment:**
```bash
cd djia
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Create data directory:**
```bash
mkdir data
# Place your MP3/WAV files in data/
```

## Quick Start

### 1. Analyze Your Library

```bash
# Analyze all tracks in data/ directory
python -m src.cli analyze

# Analyze a specific directory
python -m src.cli analyze --data-dir /path/to/tracks

# Skip already-analyzed tracks
python -m src.cli analyze --skip-existing

# Parallel analysis across worker processes (default: os.cpu_count(); use --workers 1 for
# the old sequential behavior)
python -m src.cli analyze --workers 8
```

**Output:** Analyzed tracks stored in `data/djia.db`

### 2. List Analyzed Tracks

```bash
python -m src.cli list-tracks
python -m src.cli list-tracks --limit 20  # Show first 20
```

### 3. Find Similar Tracks

```bash
python -m src.cli find-similar 1 --top-k 5
# Find 5 tracks similar to track ID 1
```

### 4. Generate a DJ Playlist

```bash
python -m src.cli generate-playlist 1 10 5
# Create 5-track playlist from track 1 to track 10
```

**Output:**
- Suggested track sequence
- BPM arc (start → end)
- Average transition quality score

### 5. Generate a Full Setlist

```bash
python -m src.cli generate-setlist --tracks 28
# 28-track, 5-phase set (warm-up→build→peak→breakdown→comeback) with per-transition mix sheets
```

**Output:** Markdown report at `results/setlist_5phase.md` — phase plan (tracks, BPM, key, energy,
top mood per phase) plus a mix sheet for every transition (beatmatch pitch %, blend length, bass
swap / full-on timings from element-onset detection). Add `--skip-mix-sheets` to skip the audio
loads for a fast phase-plan-only preview.

### 6. Export to Traktor

```bash
python -m src.cli export-traktor djia_export.nml
# Export library to Traktor NML format
```

### 7. Regenerate a Spectrogram

```bash
python -m src.cli spectrogram 1
# Recompute and save the .npy log-magnitude spectrogram for an already-analyzed track (ID 1)
```

## CLI Reference

### Commands

#### `analyze`
Analyze audio library or single track.

```bash
python -m src.cli analyze [OPTIONS]
  --data-dir PATH       Directory to scan (default: data/)
  --track PATH          Analyze single track by path
  --db PATH             Database path (default: data/djia.db)
  --skip-existing       Skip already-analyzed tracks
  --workers N           Parallel worker processes (default: os.cpu_count(); use 1 for the
                        old sequential behavior; ignored with --track)
```

#### `list-tracks`
List all analyzed tracks with features.

```bash
python -m src.cli list-tracks [OPTIONS]
  --db PATH             Database path (default: data/djia.db)
  --limit N             Max tracks to show (default: 100)
```

#### `find-similar`
Find tracks similar to a given track.

```bash
python -m src.cli find-similar TRACK_ID [OPTIONS]
  --db PATH             Database path (default: data/djia.db)
  --top-k N             Number of results (default: 5)
```

#### `generate-playlist`
Generate an optimal DJ playlist.

```bash
python -m src.cli generate-playlist START_ID END_ID [STEPS] [OPTIONS]
  --db PATH             Database path (default: data/djia.db)
  # Example: generate-playlist 1 10 5  → 5-track path from track 1 to 10
```

#### `generate-setlist`
Generate a data-driven 5-phase setlist with mix sheets.

```bash
python -m src.cli generate-setlist [OPTIONS]
  --tracks N            Number of tracks in the set (default: 28)
  --output PATH         Output markdown path (default: results/setlist_5phase.md)
  --db PATH             Database path (default: data/djia.db)
  --skip-mix-sheets     Skip audio-based mix points (fast; phase plan only)
  # Phase quotas scale with --tracks: warm-up 20%, build 29%, peak 36%, comeback 15%,
  # breakdown fixed at 1-2 tracks. Mix points cache to results/mix_points_cache.json.
```

#### `export-traktor`
Export library to Traktor NML format.

```bash
python -m src.cli export-traktor [NML_PATH] [OPTIONS]
  --db PATH             Database path (default: data/djia.db)
  # Example: export-traktor traktor_library.nml
```

#### `spectrogram`
Regenerate and save the `.npy` log-magnitude spectrogram for an already-analyzed track.

```bash
python -m src.cli spectrogram TRACK_ID [OPTIONS]
  --db PATH             Database path (default: data/djia.db)
  --spectrogram-dir PATH  Output directory (default: data/spectrograms)
  # Example: spectrogram 1
```

## Programmatic API

### Orchestrator: End-to-End Analysis

```python
from src.orchestrator import Orchestrator

# Initialize
orchestrator = Orchestrator(db_path="data/djia.db")

# Analyze library (workers=1 is sequential; workers>1 fans compute out to a ProcessPoolExecutor)
result = orchestrator.analyze_library("data/", workers=4)
print(f"Analyzed: {result['analyzed']} tracks")

# Analyze single track
features = orchestrator.analyze_single_track("data/track.wav")
print(f"BPM: {features['tempo']}, Key: {features['key']}")

# Get all tracks
tracks = orchestrator.get_all_tracks_dict()
```

### Transition Scoring: Find Best Transitions

```python
from src.ai import score_transition

track_a = {'tempo': 128, 'key': 'C', 'rms_mean': 0.1, 'mood': {...}}
track_b = {'tempo': 130, 'key': 'G', 'rms_mean': 0.11, 'mood': {...}}

score = score_transition(track_a, track_b)
print(f"Transition quality: {score.overall_score:.2%}")
print(f"  BPM: {score.bpm_score:.2%}")
print(f"  Key: {score.key_score:.2%}")
print(f"  Mood: {score.mood_score:.2%}")
print(f"  Energy: {score.energy_score:.2%}")
```

### Playlist Generation: Create Sets

```python
from src.ai import generate_playlist

tracks = {
    1: {'tempo': 128, 'key': 'C', 'mood': {...}, ...},
    2: {'tempo': 130, 'key': 'G', 'mood': {...}, ...},
    # ... more tracks
}

playlist = generate_playlist(
    all_tracks=tracks,
    start_track_id=1,
    end_track_id=10,
    num_steps=8  # 8-track set
)

print(f"Playlist: {playlist}")  # [1, 3, 7, 5, 9, 6, 4, 10]
```

### Database: Direct Access

```python
from src.database.store import TrackStore

store = TrackStore("data/djia.db")

# Get track by ID
track = store.get_track(1)
print(f"Title: {track['title']}, Artist: {track['artist']}")

# Get all tracks
all_tracks = store.get_all_tracks()

# Get features
features = store.get_track_features(1)
print(f"BPM: {features['bpm']}")

# Get mood
mood = store.get_track_mood(1)
print(f"Mood: {mood}")
```

## Performance Benchmarks

### Typical Analysis Times
- Single track: 2-3 minutes (20-30 minutes if audio is long)
  - Ingestion: 10-20 seconds
  - DSP features: 30-60 seconds
  - Mood classification: 30-60 seconds
  - Database storage: <1 second

- 70-track library: 2-4 hours (first run, parallel analysis possible)
  - Subsequent runs with caching: <1 hour

### Database Performance
- Track insertion: <100ms per track
- Feature queries: <50ms
- Playlist generation (100 tracks): <1 second
- Full library export: <5 seconds

### Storage
- Database size: ~1-2 MB per track with full features
- Audio file: Varies (MP3 4-8 MB, WAV 50+ MB for 6-min song)

## Data Format

### Features Extracted
Each track analysis produces:

```python
{
    'id': 1,
    'file_name': 'track.wav',
    'duration': 300.0,  # seconds
    'tempo': 128.5,      # BPM
    'key': 'D',          # Camelot: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    'camelot_key': '7A',
    'key_confidence': 0.72,
    'spectral_centroid_mean': 3400,
    'spectral_centroid_std': 800,
    'spectral_rolloff_mean': 12000,
    'spectral_flux_mean': 0.02,
    'harmonic_ratio': 1.3,
    'percussive_ratio': 0.6,
    'mfcc_mean': 50.0,
    'mfcc_std': 12.0,
    'mfcc_delta_mean': 0.8,
    'chroma_variance': 0.4,
    'chroma_entropy': 3.2,
    'rms_mean': 0.11,
    'rms_std': 0.02,
    'rms_peak': 0.5,
    'swing_score': 0.35,          # 0=straight, 1=fully swung (groove engine)
    'onset_strength_mean': 2.1,   # transient hardness / kick punch (groove engine)
    'onset_strength_std': 0.8,
    'beat_strength': 0.83,        # 0-1, how dominant the detected tempo's pulse is (groove engine)
    'zero_crossing_rate': 0.06,   # waveform sign-change rate; higher = noisier/acid (mood engine)
    'roughness': 0.22,            # 0-1 timbral roughness, smooth to harsh (mood engine)
    'spectral_flatness': 0.15,    # 0=tonal/clean, 1=noise-like/saturated (curation engine)
    'crest_factor': 4.6,          # peak-to-average RMS ratio; high = punchy/dynamic (curation engine)
}
```

### Mood Classification
Normalized probabilities (sum to ~1.0):

```python
{
    'dark': 0.2,
    'hypnotic': 0.4,
    'euphoric': 0.2,
    'aggressive': 0.1,
    'industrial': 0.05,
    'minimal': 0.05,
}
```

### Transition Score
Quality of transition from track A to B (0.0-1.0):

```python
score = {
    'bpm_score': 0.95,       # BPM compatibility
    'key_score': 0.90,       # Harmonic distance (Camelot wheel)
    'mood_score': 0.85,      # Mood continuity (cosine similarity)
    'energy_score': 0.88,    # Energy arc smoothness
    'overall_score': 0.90,   # Weighted average
}
```

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Suite
```bash
pytest tests/test_full_pipeline.py -v
pytest tests/test_ai.py -v
pytest tests/test_database.py -v
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Features
- End-to-end pipeline tests with synthetic audio
- Transition scoring accuracy tests
- Database operations tests
- Playlist generation tests
- 15+ test cases covering all phases

## Configuration

### Environment Variables
```bash
# .env file (optional)
OPENAI_API_KEY=sk-...   # optional, for OpenAI audio embeddings
DEMUCS_MODEL=htdemucs   # stem separation model (Phase 3)
```
Read via `os.getenv()`. Never commit secrets.

### Custom Database Path
The database path is set per call, not via env var — pass `--db` on the CLI or `db_path` to the
`Orchestrator` / `TrackStore` constructor (default `data/djia.db`):
```python
orchestrator = Orchestrator(db_path="/path/to/database.db")
```

## Troubleshooting

### Issue: "No audio files found"
- Verify audio files are in data/ directory
- Check file permissions
- Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, WMA

### Issue: Audio analysis is slow
- First run takes longer (builds features database)
- Use `--skip-existing` flag for subsequent analyses
- Consider using SSD for better I/O performance

### Issue: "Database is locked"
- Close other database connections
- Delete `.db-journal` file if it exists
- Restart analysis

### Issue: Memory issues with large library
- Analyze directory in batches
- Close other applications
- Use `--skip-existing` to avoid reprocessing

## Contributing

Areas for improvement:
1. GPU acceleration for feature extraction
2. Real-time analysis dashboard
3. Advanced ML models for mood classification
4. Sync with other DJ software (Serato, rekordbox)
5. Machine learning for personalized mixing suggestions

## Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Single track analysis | 2-3 min | ✓ Achieved |
| Library analysis (70 tracks) | 2-4 hours | ✓ Achieved |
| CLI queries | <1 sec | ✓ Achieved |
| Playlist generation (100 tracks) | <1 sec | ✓ Achieved |
| Database size per track | ~1-2 MB | ✓ Achieved |

## Files Overview

```
djia/
├── src/
│   ├── cli.py                 # Command-line interface
│   ├── orchestrator.py        # Master orchestrator (ingestion → DSP → AI → DB)
│   ├── features/schema.py     # Track dataclass — the data contract
│   ├── ingestion/             # Phase 1: scanner.py, loader.py
│   ├── dsp/                   # Phase 2: extractor + groove/phrasing/mood/curation engines, config.py
│   │   ├── worker.py                # picklable per-track analyze step for ProcessPoolExecutor
│   │   └── spectrogram.py           # log-magnitude STFT computation + .npy persistence
│   ├── ai/
│   │   ├── classifier.py            # Phase 3: mood classification
│   │   ├── stem_separator.py        # Phase 3: Demucs stems
│   │   ├── segmentation.py          # Phase 3: structural detection
│   │   ├── transition_mapper.py     # Phase 5A: transition scoring
│   │   ├── playlist_generator.py    # Phase 5B: playlist generation
│   │   ├── setlist_generator.py     # Phase 5C: 5-phase setlist + mix sheets
│   │   └── track_tuner_*.py         # optional LangGraph Track Tuner
│   ├── database/              # Phase 4: schema.py, store.py (SQLite)
│   ├── matching/similarity.py # Phase 4: cosine similarity
│   ├── traktor/exporter.py    # Phase 4: Traktor NML export
│   ├── djuced/exporter.py     # Phase 4: DJUCED hot-cue export (Hercules)
│   └── main.py, audio_analysis.py, ...  # LEGACY (do not extend)
├── tests/                     # pytest suite (one file per subsystem)
├── docs/                      # architecture, schemas, api-reference, scripts-reference, archive/
├── coding-rules.md, testing-rules.md, debugging-rules.md   # task rule files
├── PARAMETER_REFERENCE.md     # segmentation tuning reference
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── CLAUDE.md                  # Agent Router (task-first entry point for contributors)
```

## License

This project is part of the DJIA analytics system.

## Support

For issues or questions, check:
1. CLAUDE.md for development guidelines
2. `tests/` for usage examples
3. CLI help: `python -m src.cli --help`
4. Individual module docstrings for API details

## Version History

- **v1.0** (Current)
  - All 5 phases implemented and integrated
  - CLI interface complete
  - Comprehensive test coverage
  - Database backend with SQLite
  - Traktor NML export support
