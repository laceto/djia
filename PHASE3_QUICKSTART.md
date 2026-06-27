# Phase 3 Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

First run will download Demucs model (~500MB). Requires internet connection.

## Quick Usage Examples

### 1. Basic Processing
```python
from src.ai.processor import process_with_stems

result = process_with_stems('data/track.wav')
print(f"Mood: {result['mood_classification']}")
print(f"Structure: {result['structural_landmarks']}")
```

### 2. Stem Separation Only
```python
from src.ai.stem_separator import separate_stems

stems = separate_stems('data/track.wav')
# Returns: {'drums': array, 'bass': array, 'vocals': array, 'melody': array}

print(f"Drums shape: {stems['drums'].shape}")  # (2, samples)
```

### 3. Mood Classification
```python
from src.ai.classifier import classify_mood
import librosa

y, sr = librosa.load('data/track.wav')
mood = classify_mood(y, sr)

print(f"Top mood: {max(mood['moods'].items(), key=lambda x: x[1])[0]}")
print(f"Danceability: {mood['danceability']:.2f}")
print(f"Energy: {mood['energy']}")
```

### 4. Structure Detection
```python
from src.ai.segmentation import detect_structure
import librosa

y, sr = librosa.load('data/track.wav')
structure = detect_structure(y, sr)

for point in structure:
    print(f"{point.time:.1f}s: {point.structure_type} (confidence={point.confidence:.2f})")
```

### 5. With Phase 2 Features
```python
from src.ai.processor import process_with_stems

# Combine with Phase 2 DSP features
phase2_data = {
    'bpm': 120.5,
    'key': 'A',
    'spectral_centroid': 2500
}

result = process_with_stems('data/track.wav', features_dict=phase2_data)
enhanced = result['enhanced_features']
# Now contains both Phase 2 and Phase 3 data
```

## Running Tests

```bash
# All tests
pytest tests/test_ai.py -v

# Specific component
pytest tests/test_ai.py::TestMoodClassifier -v

# Skip slow tests (stem separation requires model)
pytest tests/test_ai.py -k "not process_with_stems" -v

# With coverage
pytest tests/test_ai.py --cov=src.ai
```

## Output Structure

```python
result = process_with_stems('track.wav')

# result['stems_separated']: bool
#   Whether stem separation succeeded

# result['full_audio']: dict
#   'duration': float (seconds)
#   'sample_rate': int (Hz)

# result['stems_data']: dict
#   'drums': {...analysis...}
#   'bass': {...analysis...}
#   'vocals': {...analysis...}
#   'melody': {...analysis...}

# result['mood_classification']: dict
#   'moods': {
#     'dark': 0.15,
#     'hypnotic': 0.35,
#     'euphoric': 0.10,
#     'aggressive': 0.20,
#     'industrial': 0.15,
#     'minimal': 0.05
#   },
#   'energy': 'medium',  # or 'low' / 'high'
#   'danceability': 0.68

# result['structural_landmarks']: list
#   [
#     {'time': 0.0, 'type': 'intro', 'confidence': 0.80},
#     {'time': 30.5, 'type': 'build', 'confidence': 0.75},
#     {'time': 65.2, 'type': 'drop', 'confidence': 0.92},
#     ...
#   ]

# result['enhanced_features']: dict
#   Merged Phase 2 + Phase 3 features
```

## Mood Categories

| Mood | Indicators |
|------|-----------|
| **dark** | Low frequencies, low energy, stable spectrum |
| **hypnotic** | Mid-range frequencies, steady rhythm, moderate variation |
| **euphoric** | Bright frequencies, high energy, dynamic |
| **aggressive** | High contrast, high energy, harsh highs |
| **industrial** | Very high-frequency content, metallic, harsh |
| **minimal** | Low complexity, stable, sparse elements |

## Structure Types

| Type | Description |
|------|-----------|
| **intro** | Low energy at track start |
| **build** | Gradual energy increase |
| **drop** | Sharp energy increase with strong rhythm |
| **breakdown** | Energy decrease, sparse elements |
| **bridge** | Mid-track transition |
| **outro** | Low energy at track end |

## Performance Tips

### Improve Speed:
1. **Use cached stems:** Second run uses cached results (~15 seconds)
2. **Enable GPU:** Install PyTorch with CUDA support
3. **Batch processing:** Process multiple files in parallel

### Reduce Memory:
1. **Disable stem caching:** `use_cache=False` in separate_stems()
2. **Lower sample rate:** Use `sr=16000` instead of 22050
3. **Process in chunks:** For very long tracks

### Improve Accuracy:
1. **Provide drum stem:** Pass `y_drums` to detect_structure()
2. **Use Essentia:** Install essentia-tensorflow for better mood classification
3. **Normalize audio:** Ensure consistent loudness before processing

## Troubleshooting

### "Demucs model not found"
```
Solution: First run will auto-download model (~500MB)
Ensure internet connection is available
```

### "CUDA out of memory"
```
Solution: Use CPU processing instead
export TORCH_DEVICE=cpu
```

### "Permission denied" on cache write
```
Solution: Ensure results/ directory is writable
chmod -R u+w results/
```

### "ImportError: No module named 'demucs'"
```
Solution: Install demucs
pip install demucs>=4.0.0 torch>=2.0.0
```

## Key Modules

**Stem Separation:**
```python
from src.ai.stem_separator import StemSeparator, separate_stems

separator = StemSeparator(model='htdemucs', cache_dir='results/stems')
stems = separator.separate_stems('track.wav')
```

**Mood Classification:**
```python
from src.ai.classifier import MoodClassifier, classify_mood

classifier = MoodClassifier(use_essentia=False)
result = classifier.classify_mood(y, sr)
```

**Structure Detection:**
```python
from src.ai.segmentation import StructureSegmenter, detect_structure, StructurePoint

segmenter = StructureSegmenter()
points = segmenter.detect_structure(y, sr)
```

**Full Processing:**
```python
from src.ai.processor import AIProcessor, process_with_stems

processor = AIProcessor(stem_model='htdemucs')
result = processor.process_with_stems('track.wav')
```

## API Reference

### StemSeparator
```python
separator = StemSeparator(cache_dir=Path, model='htdemucs')
stems = separator.separate_stems(audio_path, sr=16000, use_cache=True, normalize=True)
# Returns: Dict[stem_name, np.ndarray]
```

### MoodClassifier
```python
classifier = MoodClassifier(use_essentia=True)
result = classifier.classify_mood(y, sr)
# Returns: {'moods': {...}, 'energy': str, 'danceability': float}
```

### StructureSegmenter
```python
segmenter = StructureSegmenter()
points = segmenter.detect_structure(y, sr, y_drums=None)
# Returns: List[StructurePoint]
```

### AIProcessor
```python
processor = AIProcessor(stem_model='htdemucs')
result = processor.process_with_stems(audio_path, features_dict=None, sr=22050)
# Returns: Dict with stems_data, mood_classification, structural_landmarks, enhanced_features
```

## File Locations

```
Source Code:
  src/ai/stem_separator.py      - Stem separation
  src/ai/classifier.py          - Mood classification
  src/ai/segmentation.py        - Structure detection
  src/ai/processor.py           - Orchestration

Tests:
  tests/test_ai.py              - 25 unit tests

Documentation:
  src/ai/README_PHASE3.md       - Detailed guide
  PHASE3_COMPLETE.md            - Deliverables
  IMPLEMENTATION_SUMMARY.md     - Metrics

Cache:
  results/stems/{track_hash}/   - Cached stems
```

## Next Steps

1. **Integrate with Phase 2:** Merge DSP features with AI results
2. **Build Phase 4:** Database storage and similarity matching
3. **Export Data:** Traktor NML format support
4. **Create Dashboard:** Streamlit web interface

## Getting Help

1. Check `src/ai/README_PHASE3.md` for detailed documentation
2. Run tests with verbose output: `pytest tests/test_ai.py -vv`
3. Enable debug logging in code
4. Check error messages and traceback

## Version Info

- **Phase 3 Status:** Complete & Tested
- **Test Pass Rate:** 100% (25/25 tests)
- **Python:** 3.10+
- **Main Dependencies:** librosa, demucs, torch, numpy, scipy

---

Ready to use! For detailed API documentation, see `src/ai/README_PHASE3.md`
