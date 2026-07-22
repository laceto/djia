# Parameter Reference Card

The current, authoritative reference for segmentation tuning and behavior. Covers the three tunable
parameters, how detection works, beat/bar labels, and phrase-locking.

## How segmentation works (spectral novelty)

Segments come from **spectral novelty analysis** — measuring frame-to-frame change in the frequency
content and treating peaks as structural boundaries. Pipeline in `phrasing_engine.analyze_structure`:

1. **Novelty curve** — STFT (512-sample hop, ~23.2 ms @ 22,050 Hz) → per-frame normalized magnitude
   → spectral flux (L2 norm of frame differences) → novelty curve in 0–1.
2. **Peak detection** — `scipy.signal.find_peaks` with height = `novelty_threshold` and minimum
   spacing = `min_segment_duration`. Peaks become boundaries.
3. **Segments** — boundaries (plus 0.0 s start and track end) become labeled time ranges.
4. **Auto-labeling** — rule-based: first short section → `intro`, last short → `outro`, short
   mid-track (< `breakdown_duration_threshold`) → `breakdown`, first-half → `build`, second-half →
   `drop`. Confidence is currently a fixed 0.8.

Lower `novelty_threshold` and shorter `min_segment_duration` = more, shorter segments. This is why a
minimal track like Marrakech yields 94 segments at defaults but ~8 with the `minimal` preset.

## Three Tunable Parameters

### 1. `novelty_threshold` (0.0 to 1.0)

**Controls:** Sensitivity of spectral change detection

```
0.3  ├─── AGGRESSIVE (detects every change)
     │    → Many segments (100+)
     │    → Use: Complex breakdowns, drum & bass
     │
0.45 ├─── TECHNO (good for structured tracks)
     │    → Moderate segments (15-30)
     │    → Use: Progressive, techno, house
     │
0.5  ├─── DEFAULT (balanced)
     │    → Medium segments (10-20)
     │    → Use: General purpose
     │
0.55 ├─── HOUSE (less sensitive)
     │    → Fewer segments (8-15)
     │    → Use: Tech house, deep house
     │
0.65 ├─── MINIMAL (conservative)
     │    → Very few segments (5-10)
     │    → Use: Minimal, ambient, hypnotic
     │
0.7+ └─── ULTRA-CONSERVATIVE (only major changes)
         → Minimal segments (3-8)
         → Use: Ambient, very minimal tracks
```

**For Marrakech:** Use **0.65** or higher

---

### 2. `min_segment_duration` (seconds)

**Controls:** Minimum gap required between detected boundaries

```
4.0  ├─── AGGRESSIVE (allows short gaps)
     │    → Short, many segments
     │    → Use: Fast-paced, complex tracks
     │
6.0  ├─── TECHNO (moderate gap)
     │    → Medium segments
     │    → Use: Progressive, techno
     │
8.0  ├─── DEFAULT (balanced gap)
     │    → Standard segments
     │    → Use: General purpose
     │
10.0 ├─── HOUSE (larger gap)
     │    → Fewer, longer segments
     │    → Use: House, tech house
     │
12.0 ├─── MINIMAL (conservative gap)
     │    → Long segments
     │    → Use: Minimal, hypnotic
     │
16.0 └─── ULTRA-CONSERVATIVE (large gap)
         → Very long segments only
         → Use: Ambient, minimal
```

**For Marrakech:** Use **12.0** or higher

---

### 3. `breakdown_duration_threshold` (seconds)

**Controls:** Which segments get labeled "breakdown"

```
16   ├─── AGGRESSIVE (any short section = breakdown)
     │    → Many "breakdown" labels (5-10+)
     │    → Use: Tracks with lots of short sections
     │
20   ├─── TECHNO (moderate threshold)
     │    → Some "breakdown" labels (2-4)
     │    → Use: Progressive, techno
     │
24   ├─── DEFAULT (balanced)
     │    → Few "breakdown" labels (1-3)
     │    → Use: General purpose
     │
28   ├─── HOUSE (conservative)
     │    → Few "breakdown" labels (1-2)
     │    → Use: House, deep house
     │
32   ├─── MINIMAL (very conservative)
     │    → Almost no "breakdown" labels (0-1)
     │    → Use: Minimal, hypnotic, ambient
     │
40+  └─── ULTRA-CONSERVATIVE (very long sections)
         → Minimal "breakdown" labels (0)
         → Use: Very minimal, ambient
```

**For Marrakech:** Use **32.0** or higher

---

## Preset Combinations

### Minimal (House, Ambient, Hypnotic)
```
novelty_threshold: 0.65
min_segment_duration: 12.0
breakdown_duration_threshold: 32.0
→ Result: 5-10 segments, 0-1 breakdowns
```

### House (Deep House, Tech House)
```
novelty_threshold: 0.55
min_segment_duration: 10.0
breakdown_duration_threshold: 28.0
→ Result: 8-15 segments, 1-2 breakdowns
```

### Techno (Progressive, Modern Techno)
```
novelty_threshold: 0.45
min_segment_duration: 6.0
breakdown_duration_threshold: 20.0
→ Result: 15-30 segments, 2-4 breakdowns
```

### Aggressive (Complex, Drum & Bass)
```
novelty_threshold: 0.3
min_segment_duration: 4.0
breakdown_duration_threshold: 16.0
→ Result: 30-100+ segments, 5-10+ breakdowns
```

### Default (Balanced)
```
novelty_threshold: 0.5
min_segment_duration: 8.0
breakdown_duration_threshold: 24.0
→ Result: 10-20 segments, 1-3 breakdowns
```

---

## Quick Decision Tree

### "My track has too many segments"
```
Is novelty_threshold < 0.5?
  YES → Increase to 0.6
        Is it still too many?
          YES → Increase to 0.7
  NO → Increase min_segment_duration from 8 to 12
```

### "My track has too many 'breakdown' labels"
```
Increase breakdown_duration_threshold:
  24 → 28 → 32 → 40
```

### "I'm missing a major drop"
```
Decrease novelty_threshold:
  0.5 → 0.4 → 0.3
```

### "Segments are too short"
```
Increase min_segment_duration:
  8.0 → 12.0 → 16.0
```

---

## Parameter Interaction

| Change | Effect | When to Use |
|--------|--------|------------|
| ↑ `novelty_threshold` | Fewer segments | Track has too many false peaks |
| ↓ `novelty_threshold` | More segments | Missing major structural points |
| ↑ `min_segment_duration` | Longer, fewer segments | Segments too short/crowded |
| ↓ `min_segment_duration` | Shorter, more segments | Need to capture fast changes |
| ↑ `breakdown_duration_threshold` | Fewer "breakdown" labels | Too many false "breakdown" labels |
| ↓ `breakdown_duration_threshold` | More "breakdown" labels | Short sections not labeled as breakdowns |

---

## Real-World Examples

### Example 1: Hermanez - Marrakech (Minimal House)

**Problem:**
- Default → 94 segments, 89 breakdowns ❌

**Solution:**
```python
config = custom_config(
    novelty_threshold=0.65,
    min_segment_duration=12.0,
    breakdown_duration_threshold=32.0
)
```

**Result:** 8 segments, 1 breakdown ✅

---

### Example 2: Typical Progressive House (8 min)

**Expected:** 8-12 segments, 1-2 breakdowns

**Use:**
```python
config = get_config("house")
# or
config = custom_config(
    novelty_threshold=0.55,
    min_segment_duration=10.0,
    breakdown_duration_threshold=28.0
)
```

---

### Example 3: Complex Techno (6 min)

**Expected:** 15-25 segments, 2-4 breakdowns

**Use:**
```python
config = get_config("techno")
# or
config = custom_config(
    novelty_threshold=0.45,
    min_segment_duration=6.0,
    breakdown_duration_threshold=20.0
)
```

---

## Code Usage

### Using Presets
```python
from src.dsp.config import get_config
from src.dsp.extractor import extract_track_features

config = get_config("minimal")  # or "house", "techno", "aggressive"
track = extract_track_features("data/track.mp3", config=config)
```

### Custom Configuration
```python
from src.dsp.config import custom_config
from src.dsp.extractor import extract_track_features

config = custom_config(
    novelty_threshold=0.65,
    min_segment_duration=12.0,
    breakdown_duration_threshold=32.0
)
track = extract_track_features("data/track.mp3", config=config)
```

### Direct Function Call
```python
from src.dsp.phrasing_engine import analyze_structure
import librosa

y, sr = librosa.load("data/track.mp3", sr=22050)
phrasing = analyze_structure(
    y, sr,
    bpm=120,
    novelty_threshold=0.65,
    min_segment_duration=12.0,
    breakdown_threshold=32.0
)
```

---

## Validation Checklist

After tuning, verify:
- ✅ Segment count is reasonable (5-30 for 6-min track)
- ✅ Hot cues land on **real** structural points (not false positives)
- ✅ No more than 3 "breakdown" labels per track (unless track actually has many)
- ✅ Intro/outro are correctly labeled
- ✅ Major drops are detected (not missing)
- ✅ Steady sections don't have false boundaries

---

## Performance Impact

| Parameter | Impact on Speed |
|-----------|-----------------|
| `novelty_threshold` | None (post-processing only) |
| `min_segment_duration` | None (peak detection parameter) |
| `breakdown_duration_threshold` | None (labeling only) |

**Conclusion:** Tuning parameters has **zero performance impact**. Analyze freely!

---

## When to Retune

- ❌ Different audio codec (MP3 vs WAV) → No change needed
- ❌ Different sample rate (44.1kHz vs 22.05kHz) → No change needed
- ✅ Different genre → Retune
- ✅ Different production style → Retune
- ✅ Your preference changes → Retune

---

## Save Your Settings

Once you find good parameters, add them to `src/dsp/config.py`:

```python
PRESETS["my_minimal"] = DSPConfig(
    phrasing=PhrasingConfig(
        novelty_threshold=0.65,
        min_segment_duration=12.0,
        breakdown_duration_threshold=32.0,
    )
)
```

Then use everywhere:
```python
config = get_config("my_minimal")
```

---

## Element-onset detection parameters

Separate from segmentation: `detect_element_onsets` marks **where a new sound element enters**
(kick, hat, synth line) by splitting the spectrum into log-spaced bands and keeping only per-band
energy *increases*. Opt-in — `analyze_structure(..., detect_elements=True)` or a direct call.
Three tunables live in `PhrasingConfig` (`src/dsp/config.py`):

| Parameter | Default | Effect |
|---|---|---|
| `element_n_bands` | 8 | Log-spaced frequency bands watched independently. More bands = finer frequency localization, more potential onsets. |
| `element_onset_threshold` | 0.4 | Peak height (0-1) on the per-band additive-novelty curve. Lower = more sensitive (quieter elements, more false positives). |
| `element_min_sustain_bars` | 2.0 | Bars a new element must persist to count — rejects one-shot FX. |

Onsets are bar-snapped and carry a band label in DJ-EQ language (`sub`/`low`/…/`high`).
`derive_mix_points(onsets, bpm, duration, mix_out_bars=32)` reduces them to named mix points:
`mix_in` (first entry), `bass_in` (first sub/low entry), `full_on` (all bands in), `mix_out`
(N bars before the end). The track-pairing notebook and the DJUCED cue export are the consumers.

```python
from src.dsp.phrasing_engine import detect_element_onsets, derive_mix_points

onsets = detect_element_onsets(y, sr, bpm=126, threshold=0.4, min_sustain_bars=2.0)
points = derive_mix_points(onsets, bpm=126, duration=360.0)
```

---

## Beat & bar labels

Segment labels include **beat ranges** aligned to groups of 4 beats (one 4/4 bar), so cues land on
bar boundaries and match Traktor/Serato beat-grid conventions. Instead of `breakdown: 10.5s–20.3s`
you get `breakdown (beats 40-80)`.

- **Time → beat:** `beat = round(time_seconds * bpm / 60)`
- **Bar grouping:** `group = (beat // 4) * 4`
- Enabled by default; pass `include_beats=False` to `analyze_structure` for plain labels
  (`intro`, `drop`, …).

Helpers in `phrasing_engine.py`: `time_to_beat(seconds, bpm)`, `beat_to_bar_group(beat, group=4)`,
`time_to_bar(seconds, bpm)`, `bar_to_time(bar, bpm)`.

Quick reference: `1 bar = 4 beats`, `8 bars = 32 beats` (typical intro/outro),
`32 bars = 128 beats` (major structural boundary). At 126 BPM, `time_seconds = beats * 60 / bpm`
(e.g. 128 beats ≈ 61 s). If beat ranges look wrong, verify `track.groove.bpm` — phrasing uses
whatever the groove engine detected.

---

## Phrase-locking vs. spectral detection

Two segmentation strategies are available:

| | Spectral detection (default) | Phrase-locked |
|---|---|---|
| Boundaries | Real structural changes | Fixed N-bar phrases (8/16/32) |
| Segment length | Variable | Exactly N bars |
| Cue spacing | Irregular | Perfectly regular |
| Reflects the track | ✅ | ❌ |
| DJ-predictable | ⚠️ | ✅ |

- **Spectral** (`analyze_structure`) — use for real structure / non-standard tracks (e.g. minimal
  house). Requires parameter tuning.
- **Phrase-locked** (`create_phrase_locked_segments(duration, bpm, bars_per_phrase, include_beats)`)
  — forces every segment to N bars. Predictable but ignores actual morphology.
- **Hybrid (recommended for most DJ workflows):** run spectral detection, then snap cue points to
  the nearest bar with `snap_to_bar_boundary(time, bpm, bars_per_phrase=16)` — real structure **and**
  beat-grid-aligned cues.

```python
from src.dsp.phrasing_engine import analyze_structure, snap_to_bar_boundary

phrasing = analyze_structure(y, sr, bpm=126, include_beats=True)
snapped = [(c.label, snap_to_bar_boundary(c.time, bpm=126, bars_per_phrase=16))
           for c in phrasing.cue_points]
```

A comparison script lives in `test_phrase_locking.py`.
