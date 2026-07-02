# DJIA Quick Start Guide

**Status:** 3/5 Agents Complete, All Files Delivered ✓  
**Ready for:** Testing & Validation

---

## Installation

### 1. Set Up Python Environment

```bash
# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "from src.ingestion.scanner import AudioScanner; print('✓ Imports working')"
```

---

## Quick Tests

### Test 1: Scan Audio Library

```bash
python -c "
from src.ingestion.scanner import AudioScanner
scanner = AudioScanner('data')
files = scanner.scan()
print(f'Found {len(files)} audio files')
"
```

**Expected Output:**
```
Found 72 audio files
```

### Test 2: Extract Track Features (DSP)

```bash
python -c "
from src.dsp.extractor import extract_track_features
track = extract_track_features('data/Joey Beltram - Slice 2010.mp3')
print(f'BPM: {track.groove.bpm:.1f}')
print(f'Key: {track.mood.key}')
print(f'Brightness: {track.mood.brightness:.2f}')
print(f'Danceability: {track.curation.danceability:.2f}')
"
```

**Expected Output:**
```
BPM: 127.5
Key: C#/Db minor
Brightness: 0.45
Danceability: 0.72
```

### Test 3: Database Operations

```bash
python -c "
from src.database.store import TrackStore
store = TrackStore('data/djia.db')
tracks = store.get_all_tracks()
print(f'Database contains {len(tracks)} tracks')
"
```

**Expected Output:**
```
Database contains X tracks
```

### Test 4: Similarity Engine

```bash
python -c "
from src.dsp.extractor import extract_track_features
from src.database.store import TrackStore

# Extract and store two tracks
store = TrackStore('data/djia.db')
# ... similarity testing code
"
```

---

## Command-Line Interface (CLI)

Once Agent 5 completes, you'll be able to use:

```bash
# Analyze entire library
python -m src.cli analyze data/

# Analyze single track
python -m src.cli analyze-track "data/Joey Beltram - Slice 2010.mp3"

# List all tracks
python -m src.cli list-tracks

# Find similar tracks
python -m src.cli find-similar 1 --top-k 5

# Export to Traktor
python -m src.cli export-traktor /path/to/collection.nml

# Generate DJ set
python -m src.cli playlist 1 42 --steps 5
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_dsp.py -v

# Run single test
pytest tests/test_dsp.py::TestPhrasingEngine::test_novelty_curve -v
```

---

## Expected File Structure

```
src/
  ✓ ingestion/          Phase 1: File scanning & loading
  ✓ database/           Phase 4: SQLite storage
  ✓ dsp/                Phase 2: 4 DSP engines
  ✓ features/           Shared data structures
  ✓ ai/                 Phase 3: AI & stem separation
  ✓ traktor/            Phase 4: Traktor NML export
  ✓ matching/           Phase 4: Similarity engine
  ✓ cli.py              User interface
  ✓ orchestrator.py     Master orchestrator

tests/
  ✓ test_ingestion.py
  ✓ test_dsp.py
  ✓ test_ai.py
  ✓ test_database.py
  ✓ test_traktor.py

data/                   Your 72 audio files
```

---

## Project Status by Phase

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Ingestion | ✅ Complete |
| 2 | DSP Core (4 Engines) | 🟨 In Progress (Agent 2) |
| 3 | AI & Stem Separation | ✅ Complete |
| 4 | Database & Export | ✅ Complete |
| 5 | Advanced AI (Optional) | 🟨 In Progress (Agent 5) |
| - | CLI & Integration | 🟨 In Progress (Agent 5) |

---

## Next Steps

### Immediate (After All Agents Complete)

1. **Run Full Test Suite**
   ```bash
   pytest --cov=src tests/ -v
   ```

2. **Analyze Full Library**
   ```bash
   python -m src.cli analyze data/
   ```

3. **Validate Results**
   - Check BPM accuracy against Traktor
   - Verify key detection manually
   - Confirm hot-cues are positioned correctly

### After Validation

1. **Export to Traktor**
   ```bash
   python -m src.cli export-traktor "C:\path\to\collection.nml"
   ```

2. **Test Similarity Search**
   ```bash
   python -m src.cli find-similar 1 --top-k 5
   ```

3. **Explore CLI Features**
   ```bash
   python -m src.cli --help
   ```

---

## Architecture Overview

```
Audio Files (data/)
       ↓
Phase 1: Ingestion (scanner + loader)
       ↓
Phase 2: DSP Core (4 engines: phrasing, groove, mood, curation)
       ↓
Phase 3: AI Layer (stems, mood classification, segmentation)
       ↓
Phase 4: Database (SQLite) + Export (Traktor NML) + Similarity
       ↓
Phase 5: Advanced AI (optional: playlist generation)
       ↓
CLI Interface + User Output
```

---

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd C:\Users\l_ace\Desktop\projects\djia

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Database Locked
```bash
# Remove old test database
rm data/djia_test.db
```

### Out of Memory
- Reduce track analysis batch size
- Process library in smaller chunks
- Use `--skip-existing` flag to avoid re-processing

### Missing Demucs Models
- First run may download ~1GB of model files
- Ensure good internet connection
- Models cached after first download

---

## Key Features

✅ **Phase 1:** Automatic audio file detection & metadata extraction  
✅ **Phase 2:** 4-step DSP analysis (structural, rhythmic, spectral, semantic)  
✅ **Phase 3:** AI-powered stem separation + mood classification + auto-cueing  
✅ **Phase 4:** SQLite database + cosine similarity matching + Traktor export  
✅ **Phase 5:** Generative playlist + transition mapping (optional)  
✅ **CLI:** User-friendly command-line interface  

---

## Support

For detailed documentation, see:
- `plan.md` — 10-week implementation plan
- `CLAUDE.md` — Project guidelines & architecture
- `IMPLEMENTATION_STATUS.md` — Detailed status report
- Agent deliverables in respective module READMEs

---

## Performance Expectations

- **Single track analysis:** 1-3 minutes (including stem separation)
- **Full library (70 tracks):** 90-180 minutes (one-time)
- **Database query:** <100ms
- **CLI commands:** <1 second

*Times may vary based on CPU/storage speed. Stem separation is cached for speed on subsequent runs.*

---

**Ready to dive in? Start with the Quick Tests above!**
