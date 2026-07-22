# Phase 4: Track Similarity Engine & Traktor Export

This document describes Phase 4 of DJIA — building a cosine similarity engine for finding similar tracks and exporting hot cues to Traktor Pro 3+.

## Overview

Phase 4 provides two key capabilities:

1. **Similarity Matching**: Find similar tracks based on audio features using cosine similarity
2. **Traktor NML Export**: Export analyzed tracks to Traktor collection format with hot cues and metadata

## Architecture

### Modules

```
src/
├── matching/
│   ├── __init__.py
│   └── similarity.py        # Similarity engine
├── traktor/
│   ├── __init__.py
│   └── exporter.py          # Traktor NML exporter
└── database/
    └── store.py             # Enhanced with Phase 4 features
```

## 1. Similarity Engine (`src/matching/similarity.py`)

### Core Functions

#### `normalize_features(track_dict) → np.ndarray`

Normalizes track features using Z-score normalization (mean=0, std=1).

**Why normalize?**
- Features have different scales (BPM: 0-200, RMS peak: 0-1)
- Without normalization, large-scale features would dominate
- Normalization ensures all features contribute equally to similarity

**Features included:**
- BPM (tempo)
- Spectral centroid (frequency balance)
- Spectral rolloff (brightness)
- Spectral flux (change rate)
- Harmonic ratio (harmonic content)
- Percussive ratio (drum content)
- MFCC mean/std (timbre)
- Chroma variance/entropy (pitch content)
- RMS mean/std/peak (loudness)

**Example:**
```python
from src.matching.similarity import normalize_features

features = {
    'bpm': 128.0,
    'spectral_centroid_mean': 2500.0,
    # ... other features
}

normalized_vector = normalize_features(features)
# Returns: np.array([-0.15, 0.32, ...])  # Shape: (15,)
```

#### `compute_similarity(vector_a, vector_b) → float`

Computes cosine similarity between two normalized feature vectors.

**Returns:** Score in range [0.0, 1.0]
- 1.0 = identical vectors
- 0.5 = moderately similar
- 0.0 = completely different

**Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```

**Example:**
```python
from src.matching.similarity import compute_similarity

v1 = normalize_features(track1_features)
v2 = normalize_features(track2_features)

score = compute_similarity(v1, v2)
print(f"Similarity: {score:.2f}")  # 0.87
```

#### `find_similar_tracks(track_id, top_k=5, bpm_tolerance=None, mood_filter=None, db_path="data/djia.db") → List[Tuple]`

Finds similar tracks to a given track with optional filtering.

**Parameters:**
- `track_id`: ID of query track
- `top_k`: Number of results (default: 5)
- `bpm_tolerance`: Optional BPM range (e.g., 2 for ±2 BPM)
- `mood_filter`: Optional mood filter (e.g., 'hypnotic', 'aggressive')
- `db_path`: Path to SQLite database

**Returns:** List of (track_dict, similarity_score) tuples, sorted by score descending

**Example:**
```python
from src.matching.similarity import find_similar_tracks

# Find 5 most similar tracks
matches = find_similar_tracks(track_id=42, top_k=5)

# With BPM filter (±2 BPM)
matches = find_similar_tracks(track_id=42, top_k=5, bpm_tolerance=2)

# With mood filter
matches = find_similar_tracks(track_id=42, top_k=5, mood_filter="hypnotic")

# Process results
for track, score in matches:
    print(f"{track['title']}: {score:.2f}")
```

## 2. Traktor NML Exporter (`src/traktor/exporter.py`)

### Traktor Integration

Traktor Pro 3 uses XML-based `.nml` files to store collections. Phase 4 reads and modifies these files to add:

- **BPM** from Phase 2 analysis
- **Hot Cues** from Phase 3 structure detection
- **Metadata** (brightness, danceability, mood)

### Core Functions

#### `parse_traktor_nml(nml_path) → ET.Element`

Parses Traktor collection.nml file and returns XML root element.

**Raises:** 
- `FileNotFoundError` if file doesn't exist
- `ET.ParseError` if XML is malformed

**Example:**
```python
from src.traktor.exporter import parse_traktor_nml

root = parse_traktor_nml("Collection.nml")
print(f"Collection has {root.get('ENTRIES')} entries")
```

#### `add_track_analysis(nml_root, track_id, track_analysis, db_path="data/djia.db") → ET.Element`

Adds analysis data (BPM, hot cues, metadata) to a track entry in the NML tree.

**Track Analysis Dict:**
```python
{
    'bpm': 128.5,                       # From Phase 2
    'brightness': 75,                   # 0-100
    'danceability': 88,                 # 0-100
    'mood': 'hypnotic',                 # Optional
    'cue_points': [                     # From Phase 3
        {'time': 45.2, 'type': 'drop'},
        {'time': 120.5, 'type': 'breakdown'},
        {'time': 300.0, 'type': 'outro'},
    ]
}
```

**Hot Cue Mapping:**
```
Phase 3 Type  →  Traktor Pad  →  Description
drop          →  Pad 1        →  First major energy return
breakdown     →  Pad 2        →  First percussive reduction
outro         →  Pad 4        →  Track ending section
```

**Example:**
```python
root = parse_traktor_nml("Collection.nml")

analysis = {
    'bpm': 130.0,
    'brightness': 75,
    'danceability': 85,
    'cue_points': [
        {'time': 45.2, 'type': 'drop'},
        {'time': 120.5, 'type': 'breakdown'},
    ]
}

track_entry = add_track_analysis(root, track_id=42, track_analysis=analysis)
```

#### `export_nml(nml_root, output_path) → bool`

Writes modified NML tree back to file with validation.

**Validates:**
- XML is well-formed
- Can be re-parsed after writing
- Compatible with Traktor Pro 3+

**Example:**
```python
success = export_nml(root, "results/collection_analyzed.nml")
if success:
    print("✓ Exported to Traktor format")
```

#### `export_all_tracks(traktor_nml_path, db_path="data/djia.db", output_path="results/collection_analyzed.nml") → str`

Batch export: reads all analyzed tracks from database and exports to Traktor NML.

**Workflow:**
1. Parse original Traktor collection.nml
2. Query database for all analyzed tracks
3. Retrieve features and structure points
4. Add hot cues, BPM, and metadata to each entry
5. Validate XML
6. Write to output_path

**Example:**
```python
from src.traktor.exporter import export_all_tracks

output = export_all_tracks(
    traktor_nml_path="Collection.nml",
    db_path="data/djia.db",
    output_path="results/collection_analyzed.nml"
)
print(f"Exported to: {output}")
```

## 3. Database Integration

### TrackStore Enhancements

Added methods for Phase 4:

#### `get_segments_by_type(track_id, segment_type) → Dict`

Get the first segment of a specific type.

```python
from src.database.store import TrackStore

store = TrackStore()
drop_segment = store.get_segments_by_type(track_id=42, segment_type='drop')
```

#### `get_all_tracks_with_features() → List[Dict]`

Get all tracks with features joined for batch operations.

```python
tracks = store.get_all_tracks_with_features()
for track in tracks:
    print(f"{track['title']}: {track['bpm']} BPM")
```

## Performance Requirements

Phase 4 targets these performance benchmarks:

| Operation | Target | Status |
|-----------|--------|--------|
| Single similarity query | <100ms | ✓ |
| Cosine similarity (2 vectors) | <1ms | ✓ |
| Find top-5 similar (1000 tracks) | <50ms | ✓ |
| NML export (500 tracks) | <10s | ✓ |
| Feature normalization | <1s per track | ✓ |

## DJ Workflow

### Complete Pipeline

```
Phase 1-3: Analysis
  └─ Audio loaded
  └─ Features extracted
  └─ Structure detected
  
Phase 4a: Similarity
  ├─ Query: "Find similar to Track X"
  ├─ Normalize features
  ├─ Compute cosine similarity to all
  └─ Return top-K matches
  
Phase 4b: Traktor Export
  ├─ Load Collection.nml
  ├─ For each analyzed track:
  │   ├─ Add BPM
  │   ├─ Add hot cues (drop, breakdown, outro)
  │   └─ Add metadata
  ├─ Validate XML
  └─ Save collection_analyzed.nml
  
Use in Traktor:
  ├─ Import collection_analyzed.nml
  ├─ Open track
  ├─ Hot cues auto-populated
  └─ Use for mixing
```

### Example: Find Similar & Export

```python
from src.matching.similarity import find_similar_tracks
from src.traktor.exporter import export_all_tracks
from src.database.store import TrackStore

# Step 1: Find similar track
store = TrackStore()
matches = find_similar_tracks(track_id=42, top_k=3, bpm_tolerance=2)

print("Suggested mixes:")
for track, score in matches:
    print(f"  • {track['title']} (similarity: {score:.0%})")

# Step 2: Export to Traktor
output_nml = export_all_tracks("Collection.nml", db_path="data/djia.db")
print(f"Ready to import in Traktor: {output_nml}")
```

## Testing

Run Phase 4 tests:

```bash
pytest tests/test_traktor.py -v
```

### Test Coverage

```
TestTraktorParsing
  ✓ Parse valid NML
  ✓ Handle missing files
  ✓ Detect malformed XML

TestTraktorExport
  ✓ Add analysis to track
  ✓ Create valid NML output
  ✓ Batch export all tracks

TestSimilarityEngine
  ✓ Normalize features
  ✓ Handle missing values
  ✓ Compute similarity (identical, orthogonal, bounds)

TestSimilarityMatching
  ✓ Find similar tracks (basic)
  ✓ Sort by similarity
  ✓ Apply BPM filter
  ✓ Handle nonexistent track

TestUtilities
  ✓ Millisecond to Traktor offset conversion
```

## Examples

See `examples/phase4_example.py` for complete working examples:

```bash
python examples/phase4_example.py
```

Demonstrates:
1. Feature normalization
2. Finding similar tracks
3. Traktor NML export workflow

## Traktor NML Format (Reference)

### Collection Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<COLLECTION ENTRIES="123">
    <ENTRY AUDIO_ID="..." TITLE="..." ARTIST="...">
        <TITLE>Track Title</TITLE>
        <ARTIST>Artist Name</ARTIST>
        <TEMPO BPM="128.5" />
        <INFO KEY="8A" BRIGHTNESS="75" DANCEABILITY="85" MOOD="hypnotic" />
        
        <!-- Hot Cues (max 8 per Traktor version) -->
        <CUE_V2 NAME="HotCue_1" DISPL_ORDER="0" TYPE="0" START="2304000" LEN="0" REPEATS="-1" HOTCUE="1" />
        <CUE_V2 NAME="HotCue_2" DISPL_ORDER="1" TYPE="0" START="5760000" LEN="0" REPEATS="-1" HOTCUE="2" />
        <CUE_V2 NAME="HotCue_4" DISPL_ORDER="3" TYPE="0" START="14400000" LEN="0" REPEATS="-1" HOTCUE="4" />
    </ENTRY>
</COLLECTION>
```

### Notes

- `START` is in samples at 48kHz (48 samples = 1ms)
- `HOTCUE` value determines pad number (1-8)
- Hot cues are automatically colored by Traktor based on pad number
- BPM is optional (auto-detected if missing)

## Troubleshooting

### "Track not found in NML"
- Ensure Collection.nml contains the track
- Check title/artist match between database and Traktor
- Re-export Collection.nml from Traktor to get latest data

### "No similar tracks found"
- Need at least 2 tracks in database
- Filters (BPM, mood) may be too restrictive
- Check that features are properly analyzed

### "Exported NML won't open in Traktor"
- Validate XML with `xmllint collection_analyzed.nml`
- Check file encoding (should be UTF-8)
- Try with smaller test collection first

## Next Steps (Phase 5)

Phase 5 can build on Phase 4:
- **AI-powered recommendations**: Use LLM for "vibe matching"
- **Playlist generation**: Auto-build mixes from similar tracks
- **Energy mapping**: BPM progression suggestions for sets
- **Mood analysis**: Classify energy arcs in mixes

## References

- Traktor Pro 3 Documentation
- Cosine Similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
- Z-score Normalization (https://en.wikipedia.org/wiki/Standard_score)
- Scikit-learn: Cosine Similarity (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)

## Dependencies

- `scikit-learn>=1.4.0` - Cosine similarity
- `numpy>=1.26.0` - Vector operations
- `xml.etree.ElementTree` - NML parsing (stdlib)
- `sqlite3` - Database (stdlib)
