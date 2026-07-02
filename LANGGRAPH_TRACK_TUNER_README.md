# LangGraph Track Tuner Agent

Automated track analysis and parameter tuning using LangGraph multi-agent orchestration.

## Overview

The Track Tuner is a LangGraph-based agent system that:
- Analyzes tracks and evaluates segmentation quality
- Automatically tunes phrasing engine parameters
- Iteratively improves results until satisfied or max iterations reached
- Processes single tracks or batches of tracks
- Provides detailed quality metrics and recommendations

## Architecture

### Single Track Agent

```
┌─────────────┐
│ LoadTrack   │ Extract BPM, duration
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ InitializeConfig │ Load preset parameters
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ AnalyzeTrack     │ Run phrasing engine
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ EvaluateQuality  │ Score segmentation (0-1)
└──────┬───────────┘
       │
    ┌──┴──────────────┐
    │                 │
    ▼                 ▼
 Satisfied?      (Max iterations?)
    │                 │
   YES               YES
    │                 │
    ▼                 ▼
┌──────────┐    ┌──────────┐
│ Finalize │    │ Finalize │
└────┬─────┘    └────┬─────┘
     │              │
     └──────┬───────┘
            │
    NO      ▼
    ├────────────────┐
    │                │
    ▼                │
┌────────────────┐   │
│ SuggestTuning  │   │
└────────┬───────┘   │
         │           │
         └─────┬─────┘
               │
               ▼
          (back to Analyze)
```

### Batch Agent

```
┌──────────────┐
│ ProcessTrack │ Run single-track agent for each track
└──────┬───────┘
       │
  For each track
       │
       ▼
┌─────────────────────┐
│ Single Track Agent  │ (repeated N times)
└──────┬──────────────┘
       │
       ▼
┌──────────────────┐
│ AggregateResults │ Collect all results
└────────┬─────────┘
         │
         ▼
      [Results]
```

## Components

### 1. State (`track_tuner_state.py`)

**TrackTunerState** — Single track analysis state
```python
{
    'track_path': str,
    'track_name': Optional[str],
    'bpm': Optional[float],
    'duration': Optional[float],
    'current_segments': Optional[List[dict]],
    'current_quality': Optional[SegmentQuality],
    'current_config': Optional[TuneConfig],
    'iterations_completed': int,
    'max_iterations': int,
    'satisfied': bool,
    'reason': Optional[str],
    'analysis_history': List[dict],  # Accumulates
    'recommendations': List[str],    # Accumulates
    'messages': List[BaseMessage],   # Accumulates
}
```

**BatchTrackTunerState** — Batch processing state
```python
{
    'track_paths': List[str],
    'current_track_index': int,
    'results': List[TrackAnalysisResult],  # Accumulates
    'config_preset': str,
    'messages': List[BaseMessage],
}
```

### 2. Nodes (`track_tuner_nodes.py`)

| Node | Purpose | Output |
|------|---------|--------|
| `load_track` | Extract BPM, duration, metadata | track_name, bpm, duration |
| `initialize_config` | Load preset parameters | current_config, initial_config |
| `analyze_track` | Run phrasing engine | current_segments, current_quality |
| `evaluate_quality` | Score segmentation | satisfied, reason, quality_score |
| `suggest_tuning` | Recommend parameter changes | current_config, recommendations |
| `finalize` | Complete analysis | (no state changes) |

### 3. Graph (`track_tuner_graph.py`)

- **`build_single_track_graph()`** — Single track analysis
- **`build_batch_graph()`** — Batch processing
- **`run_single_track(path, preset, max_iterations)`** — Easy API for single track
- **`run_batch_tracks(paths, preset)`** — Easy API for batch

## Quality Scoring

Quality score (0-1) combines:

1. **Segment Count Score** (30%)
   - Ideal: 2-10 segments
   - Penalty: too many (>10) or too few (<2)

2. **Bar Distribution Score** (30%)
   - Ideal: 16-256 bars per segment
   - Penalty: too short (<16) or too long (>256)

3. **Regularity Score** (40%)
   - Based on standard deviation of bar counts
   - Higher = more consistent segments

4. **False Breakdown Penalty** (-30%)
   - If >3 breakdown labels detected
   - Indicates poor parameter tuning

**Quality Thresholds:**
- Excellent: ≥ 0.85
- Good: ≥ 0.70 (tuning stops here)
- Fair: ≥ 0.50
- Poor: < 0.50

## Tuning Strategy

### Decision: Too Many Segments?

**Symptoms:** >10 segments detected, low quality score

**Fix:**
```python
novelty_threshold += 0.1  # (e.g., 0.5 → 0.6)
min_segment_duration += 2  # (e.g., 8.0 → 10.0)
```

**Why:** Reduces sensitivity to noise, filters out false positives

### Decision: Too Many Breakdowns?

**Symptoms:** >3 "breakdown" labels, has_false_breakdowns = True

**Fix:**
```python
breakdown_duration_threshold += 8  # (e.g., 24 → 32)
```

**Why:** Only very short segments are labeled as breakdowns

### Decision: Irregular Distribution?

**Symptoms:** high std deviation (>50), uneven segment lengths

**Fix:**
```python
novelty_threshold -= 0.05  # (e.g., 0.5 → 0.45)
```

**Why:** Increases sensitivity to detect better structural points

## Presets

### Default (General Purpose)
```python
novelty_threshold: 0.5
min_segment_duration: 8.0
breakdown_duration_threshold: 24.0
```

### Minimal (Minimal/Ambient/House)
```python
novelty_threshold: 0.65
min_segment_duration: 12.0
breakdown_duration_threshold: 32.0
```

### House (Deep/Tech House)
```python
novelty_threshold: 0.55
min_segment_duration: 10.0
breakdown_duration_threshold: 28.0
```

### Techno (Progressive/Techno)
```python
novelty_threshold: 0.45
min_segment_duration: 6.0
breakdown_duration_threshold: 20.0
```

### Aggressive (Complex Structure)
```python
novelty_threshold: 0.3
min_segment_duration: 4.0
breakdown_duration_threshold: 16.0
```

## Usage

### Single Track Analysis

```python
from src.ai.track_tuner_graph import run_single_track

result = run_single_track(
    track_path="data/My Track.mp3",
    preset="minimal",      # or "default", "house", "techno"
    max_iterations=3
)

print(f"Quality: {result['current_quality']['quality_score']:.2f}")
print(f"Segments: {result['current_quality']['num_segments']}")
print(f"Satisfied: {result['satisfied']}")
```

### Batch Processing

```python
from src.ai.track_tuner_graph import run_batch_tracks
from pathlib import Path
import glob

tracks = glob.glob("data/**/*.mp3", recursive=True)

result = run_batch_tracks(
    track_paths=tracks[:10],  # First 10 tracks
    preset="minimal"
)

for r in result['results']:
    print(f"{r['track_name']}: {r['quality_score']:.2f} | {r['satisfied']}")
```

### Use in Jupyter Notebook

See `DJIA_LangGraph_TrackTuner.ipynb` for complete examples:
- Single track analysis with detailed output
- Batch processing over multiple tracks
- Preset comparison
- Segment detailed breakdown
- Results export to JSON

## Output

### Single Track Result

```python
{
    'track_name': 'Hermanez - Marrakech',
    'bpm': 129.2,
    'duration': 635.2,
    'current_quality': {
        'num_segments': 2,
        'avg_bars_per_segment': 171.0,
        'regularity_std': 40.5,
        'has_false_breakdowns': False,
        'quality_score': 0.72,
    },
    'satisfied': True,
    'reason': 'Quality threshold met',
    'iterations_completed': 1,
    'current_config': {
        'novelty_threshold': 0.65,
        'min_segment_duration': 12.0,
        'breakdown_duration_threshold': 32.0,
        'iteration': 0,
    },
    'current_segments': [
        {
            'label': 'drop (beats 0-844)',
            'start_time': 0.0,
            'end_time': 392.88,
            'start_bar': 0.0,
            'end_bar': 211.5,
        },
        # ...
    ],
    'analysis_history': [
        {
            'iteration': 0,
            'config': {...},
            'quality_score': 0.72,
            'reason': 'Quality threshold met',
        },
    ],
}
```

### Batch Result

```python
{
    'results': [
        TrackAnalysisResult(...),  # Track 1
        TrackAnalysisResult(...),  # Track 2
        # ...
    ],
    'messages': [AIMessage(...), ...],
}
```

## LangGraph Patterns Used

✅ **Canonical State Pattern**
- TypedDict with operator.add for accumulating lists
- add_messages for message history

✅ **Canonical Node Pattern**
- All nodes accept (state, config)
- Return dict (never mutate state)
- Add AIMessage with [NodeName] prefix for traceability

✅ **Routing Functions**
- should_iterate() decides: analyze again or finalize?
- Based on quality score and iteration count

✅ **Configuration Injection**
- Presets passed via config["configurable"]["preset"]
- Models selected via get_llm(provider, tier)

## Integration with DJIA

The Track Tuner integrates with existing DJIA components:

```python
from src.dsp.extractor import extract_track_features
from src.dsp.config import custom_config
from src.dsp.phrasing_engine import time_to_bar

# Use your existing DSP engines
dsp_config = custom_config(
    novelty_threshold=0.65,
    min_segment_duration=12.0,
    breakdown_duration_threshold=32.0,
)

track = extract_track_features("data/track.mp3", config=dsp_config)
```

## Performance

- Single track: ~5-30s (depending on audio length and iterations)
- Batch of 10 tracks: ~1-5 min
- Each iteration adds ~2-3s per track

## Monitoring

All agent activity logged via LangGraph messages:

```python
for msg in result['messages']:
    print(msg.content)

# Output:
# [LoadTrack] Loaded Marrakech | BPM: 129.2 | Duration: 635.2s
# [InitializeConfig] Using preset: minimal
# [AnalyzeTrack] 2 segments | Avg bars: 171.0 | Std: 40.5
# [EvaluateQuality] Score: 0.72 | Quality threshold met
# [Finalize] Analysis complete | Quality: 0.72 | Iterations: 1
```

## Next Steps

1. **Run the notebook** — Execute cells in `DJIA_LangGraph_TrackTuner.ipynb`
2. **Analyze your library** — Batch process all tracks
3. **Export results** — Save analysis to JSON/CSV
4. **Use with CLI** — Integrate with `src/cli.py` for end-to-end workflow
5. **Visualize** — Create spectrograms with tuned parameters

## Files

```
src/ai/
├── track_tuner_state.py      TypedDict state definitions
├── track_tuner_nodes.py      Node functions (load, analyze, evaluate, etc.)
├── track_tuner_graph.py      LangGraph assembly & API

notebooks/
└── DJIA_LangGraph_TrackTuner.ipynb  Complete usage examples

docs/
└── LANGGRAPH_TRACK_TUNER_README.md  This file
```

## Troubleshooting

### Agent runs but quality is low

→ Adjust preset or increase max_iterations

### Takes too long

→ Reduce max_iterations or use "default" preset

### Results vary between runs

→ Agent is non-deterministic (uses LLM for evaluation) - normal

### Want deterministic results

→ Implement score evaluation in nodes instead of using LLM

## Future Enhancements

- [ ] LLM-based quality evaluation (GPT/Claude for semantic scoring)
- [ ] Multi-agent consensus (vote on tuning decisions)
- [ ] Persistent tuning memory (learn from previous tracks)
- [ ] Streaming output for Streamlit UI
- [ ] Export to Traktor NML format
- [ ] A/B testing different preset combinations
