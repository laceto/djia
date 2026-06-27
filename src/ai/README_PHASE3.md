# Phase 3: Advanced AI Features - Implementation Guide

## Overview

Phase 3 implements advanced audio analysis using stem separation, mood classification, and structural segmentation. These features enhance the core DSP analysis from Phase 2 with AI-driven insights.

## Components

### 1. Stem Separator (`stem_separator.py`)

Separates audio tracks into individual stems using Demucs (Facebook's Music Source Separation library).

#### Key Features:
- **Stem Types**: Drums, Bass, Vocals, Melody
- **Model**: Uses `htdemucs` (high-quality temporal separation) by default
- **Caching**: Automatic disk caching to avoid re-processing
- **Loudness Normalization**: RMS-based normalization for balanced stems
- **Fallback**: Graceful degradation if Demucs unavailable

#### Usage:

```python
from src.ai.stem_separator import StemSeparator, separate_stems

# Option 1: Using the class
separator = StemSeparator(model='htdemucs')
stems = separator.separate_stems('path/to/track.wav')
# Returns: {
#   'drums': np.ndarray (stereo),
#   'bass': np.ndarray (stereo),
#   'vocals': np.ndarray (stereo),
#   'melody': np.ndarray (stereo)
# }

# Option 2: Using convenience function
stems = separate_stems('path/to/track.wav', model='htdemucs')
```

#### Cache Location:
- Cached stems stored in: `results/stems/{track_hash}/`
- Each stem saved as `{stem_name}.wav`
- Metadata in `metadata.json`

#### Performance:
- ~30 seconds per track (highly dependent on audio length and GPU availability)
- First-time processing is slow; subsequent runs use cache
- GPU acceleration recommended for production use

### 2. Mood Classifier (`classifier.py`)

Classifies mood and acoustic characteristics using spectral analysis and rule-based scoring.

#### Features:
- **Mood Categories**: dark, hypnotic, euphoric, aggressive, industrial, minimal
- **Energy Levels**: low, medium, high
- **Danceability Score**: 0-1 (higher = more danceable)
- **Fallback**: Pure rule-based when Essentia unavailable

#### Rule-Based Classification:

The classifier uses spectral and temporal features:

| Mood | Indicators |
|------|-----------|
| **Dark** | Low frequencies, low energy, stable spectrum |
| **Hypnotic** | Mid-range frequencies, steady rhythm, moderate variation |
| **Euphoric** | Bright frequencies, high energy, dynamic |
| **Aggressive** | High contrast, high energy, harsh highs |
| **Industrial** | Very high-frequency content, metallic, harsh transients |
| **Minimal** | Low complexity, stable, sparse elements |

#### Usage:

```python
from src.ai.classifier import MoodClassifier, classify_mood

# Option 1: Using the class
classifier = MoodClassifier(use_essentia=False)  # False = use rule-based
result = classifier.classify_mood(y, sr)
# Returns: {
#   'moods': {
#     'dark': 0.15,
#     'hypnotic': 0.35,
#     'euphoric': 0.10,
#     'aggressive': 0.20,
#     'industrial': 0.15,
#     'minimal': 0.05
#   },
#   'energy': 'medium',
#   'danceability': 0.68
# }

# Option 2: Using convenience function
result = classify_mood(y, sr)
```

#### Danceability Calculation:

Combines:
- **Onset Strength** (30%): Rhythm detection via onset envelope
- **Energy** (30%): RMS mean and variation
- **Spectral Flux** (20%): Frequency change intensity
- **Base Score** (20%): Baseline 0.14

### 3. Structural Segmentation (`segmentation.py`)

Detects musical sections and structural landmarks using novelty curves and energy analysis.

#### Detected Structures:
- **Intro**: Low energy at track start
- **Build**: Gradual energy increase
- **Drop**: Sharp energy increase with strong rhythm
- **Breakdown**: Energy decrease, sparse elements
- **Bridge**: Mid-track transitions
- **Outro**: Low energy at track end

#### Implementation:

Uses two main curves:
1. **Novelty Curve**: Spectral flux detection for section boundaries
2. **Energy Curve**: Mel-spectrogram energy tracking

Peak detection finds structure points; classification based on:
- Energy change (before/after)
- Drum intensity
- Peak prominence

#### Usage:

```python
from src.ai.segmentation import StructureSegmenter, detect_structure, StructurePoint

# Option 1: Using the class
segmenter = StructureSegmenter()
points = segmenter.detect_structure(y, sr, y_drums=y_drums)
# Returns: [
#   StructurePoint(0.0s, 'intro', conf=0.8),
#   StructurePoint(30.5s, 'build', conf=0.75),
#   StructurePoint(65.2s, 'drop', conf=0.92),
#   ...
# ]

# Option 2: Using convenience function
points = detect_structure(y, sr, y_drums=y_drums)

# Access data
for point in points:
    print(f"{point.time:.1f}s: {point.structure_type} (conf={point.confidence:.2f})")
    data = point.to_dict()  # {'time': ..., 'type': ..., 'confidence': ...}
```

### 4. AI Processor (`processor.py`)

Orchestrates all Phase 3 components into a unified pipeline.

#### Pipeline:
1. Stem separation
2. Individual stem analysis
3. Mood classification (full mix)
4. Structural detection
5. Feature merging with Phase 2 results

#### Usage:

```python
from src.ai.processor import AIProcessor, process_with_stems

# Option 1: Using the class
processor = AIProcessor(stem_model='htdemucs', cache_stems=True)
result = processor.process_with_stems(
    'path/to/track.wav',
    features_dict={'bpm': 120, 'key': 'A'},  # Phase 2 features (optional)
    sr=22050
)

# Option 2: Using convenience function
result = process_with_stems(
    'path/to/track.wav',
    features_dict={'bpm': 120, 'key': 'A'}
)

# Result structure
result = {
    'audio_path': str,
    'stems_separated': bool,
    'full_audio': {'duration': float, 'sample_rate': int},
    'stems_data': {
        'drums': {...stem analysis...},
        'bass': {...stem analysis...},
        'vocals': {...stem analysis...},
        'melody': {...stem analysis...}
    },
    'mood_classification': {
        'moods': {...},
        'energy': str,
        'danceability': float
    },
    'structural_landmarks': [
        {'time': float, 'type': str, 'confidence': float},
        ...
    ],
    'enhanced_features': {...Phase 2 + Phase 3 combined...}
}
```

## Dependencies

### Required:
- `librosa>=0.10.0` - Audio analysis
- `soundfile>=0.12.0` - Audio I/O
- `numpy>=1.26.0` - Numerical computing
- `scipy>=1.12.0` - Scientific computing
- `demucs>=4.0.0` - Stem separation
- `torch>=2.0.0` - Required by Demucs

### Optional:
- `essentia>=2.1.0` - For TensorFlow-based mood models (requires special setup)

### Installation:

```bash
pip install -r requirements.txt
```

For GPU acceleration with Demucs (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Performance Targets

| Component | Time | Status |
|-----------|------|--------|
| Stem Separation | ~30s | Slow; caching essential |
| Mood Classification | <5s | Fast; rule-based |
| Structure Detection | <10s | Fast; novelty-based |
| **Total per track** | 45-60s | Acceptable |

## Caching Strategy

### Stem Cache:
- **Location**: `results/stems/{track_hash}/`
- **Contents**: Stereo WAV files + metadata.json
- **Validation**: Source path + model stored in metadata
- **TTL**: Indefinite (manual cleanup required)

### Cache Miss Recovery:
- Automatic on first run
- Metadata validation on subsequent runs
- Graceful fallback if cache corrupted

## Error Handling

### Demucs Unavailable:
```python
# Returns empty/zero stems
stems = {
    'drums': np.zeros((1, 1)),
    'bass': np.zeros((1, 1)),
    ...
}
```

### Essentia Unavailable:
- Automatically falls back to rule-based classification
- No performance degradation; results still valid

### File Errors:
- Invalid audio paths logged with warning
- Processor continues with available data
- No hard crashes

## Integration with Phase 2

Phase 3 is designed as an enhancement layer over Phase 2:

```python
# Phase 2 features
phase2_features = {
    'bpm': 120.5,
    'key': 'A minor',
    'spectral_centroid': 2500,
    ...
}

# Phase 3 enhancement
result = process_with_stems('track.wav', features_dict=phase2_features)

# Result contains both Phase 2 and Phase 3 data
enhanced = result['enhanced_features']
# {
#   'bpm': 120.5,  # Preserved
#   'key': 'A minor',  # Preserved
#   'stems': {...},  # New Phase 3
#   'mood': {...},  # New Phase 3
#   'structure': [...]  # New Phase 3
# }
```

## Testing

### Run All Tests:
```bash
pytest tests/test_ai.py -v
```

### Run Specific Component Tests:
```bash
# Mood classification
pytest tests/test_ai.py::TestMoodClassifier -v

# Structural segmentation
pytest tests/test_ai.py::TestStructureSegmenter -v

# Stem separation (note: slow without GPU)
pytest tests/test_ai.py::TestStemSeparator::test_stem_separator_initialization -v

# AI processor
pytest tests/test_ai.py::TestAIProcessor::test_processor_initialization -v
```

### Test Coverage:
- 25+ unit tests covering all components
- Fixture-based sample audio generation
- Integration tests with real audio I/O
- Performance benchmarks included

## Future Enhancements (Phase 4+)

### Phase 4: Database & Export
- Store Phase 3 results in SQLite
- Export to Traktor NML format
- Cue point generation from structure

### Phase 5: Playlist Generation
- Use stems + mood + structure for DJ set construction
- Transition quality scoring between tracks
- Automated setlist generation

### Phase 6: Real-Time Analysis
- Web interface for single-track analysis
- Batch processing dashboard
- Live mixing recommendations

## Troubleshooting

### "Demucs model not found"
```
Solution: First run will download model (~500MB). Requires internet.
```

### "Permission denied" on cache write
```
Solution: Ensure results/ directory is writable
chmod -R u+w results/
```

### "CUDA out of memory"
```
Solution: Use CPU processing instead
export TORCH_DEVICE=cpu
```

### "Librosa resampling slow"
```
Solution: Pre-resample audio to 22050 Hz before processing
```

## File Structure

```
src/ai/
├── __init__.py              # Module exports
├── stem_separator.py        # Demucs wrapper
├── classifier.py            # Mood classification
├── segmentation.py          # Structure detection
├── processor.py             # Orchestrator
└── README_PHASE3.md         # This file

tests/
├── test_ai.py               # 25+ unit tests
└── ...

results/
├── stems/
│   └── {track_hash}/        # Cache directory
│       ├── drums.wav
│       ├── bass.wav
│       ├── vocals.wav
│       ├── melody.wav
│       └── metadata.json
└── ...
```

## References

- **Demucs**: https://github.com/facebookresearch/demucs
- **Librosa**: https://librosa.org
- **Essentia**: https://essentia.upf.edu
- **Novelty Detection**: Foote, J. "Visualizing Music Structure and Identifying Important Sections" (2005)

## License

Same as DJIA project (see project root)
