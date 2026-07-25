# Parameter Reference Card

The current, authoritative reference for segmentation tuning and behavior. Covers the tunable
phrasing parameters, how detection works, hot-cue mapping, element onsets, and bar helpers.

## How segmentation works (low-band energy)

Structure comes from **kick+bass (low-band) energy** — for four-on-the-floor techno/house the low
end carries the arrangement: it is present during **drops** and falls away during **breakdowns**.
This is far more reliable than generic spectral novelty, which over-segments steady material.
Pipeline in `phrasing_engine.analyze_structure`:

1. **Low-band envelope** — `compute_lowband_energy`: STFT (512-sample hop, ~23.2 ms @ 22,050 Hz),
   keep only the 20–150 Hz bins, average per frame.
2. **Smooth + threshold** — `smooth_lowband_energy` averages over ~1 bar; frames above
   `thresh_frac × peak` are "kick on". `off→on` starts a **drop**, `on→off` starts a **breakdown**
   (`detect_energy_sections`).
3. **Merge + label** — sections shorter than `min_bars` are merged into the previous one so blips
   don't fragment the structure; sections stay contiguous and cover the whole track.
   `label_energy_sections` labels them `intro` (first kick-off), `outro` (last kick-off),
   `breakdown` (mid kick-off), `drop` (kick-on). Confidence is a fixed 0.85.
4. **Hot cues** — `map_segments_to_hotcues` places one cue at each section **start** (the point a DJ
   jumps to).

Lower `thresh_frac` and smaller `min_bars` = more, shorter sections (more drop/breakdown
transitions). This replaces the old spectral-novelty model, which produced dozens of spurious
"breakdown" labels on minimal tracks.

## Three Tunable Parameters

All three live in `PhrasingConfig` (`src/dsp/config.py`) and are passed straight to
`analyze_structure`.

### 1. `min_bars` (bars, integer)

**Controls:** minimum section length; shorter sections are merged into the previous one.

```
2   ├─── AGGRESSIVE — keeps short sections (more, shorter drops/breakdowns)
4   ├─── DEFAULT / TECHNO — balanced
6   ├─── HOUSE — fewer, longer sections
8   └─── MINIMAL / CONSERVATIVE — forces long sections (few boundaries)
```

### 2. `thresh_frac` (0.0 to 1.0)

**Controls:** the kick-on threshold as a fraction of peak low-band energy.

```
0.30 ├─── AGGRESSIVE — much of the track reads as "kick on"; many transitions
0.35 ├─── TECHNO
0.40 ├─── DEFAULT — balanced
0.45 ├─── HOUSE
0.50 └─── MINIMAL / CONSERVATIVE — only clear, loud drops register
```

Lower = more sensitive (catches partial kick sections as drops). Higher = only full-energy drops
count, so more of the track reads as breakdown.

### 3. `max_pads` (integer or `None`)

**Controls:** how many hot cues / physical performance pads to emit.

- `None` (default) — one cue per structural section, labelled by type (intro→Pad 1, main drop→Pad 4,
  later drops→Pad 3, breakdown→Pad 2, outro→Pad 1).
- `N` (e.g. **4** for a 4-pad controller) — keep the **N most important** sections and re-label them
  `Pad 1..N` in chronological order. The **intro**, the **first (main) drop** and the **outro** are
  always kept; remaining slots go to the longest drops/breakdowns. Because pads are numbered by time,
  the outro naturally takes the last pad.

---

## Presets

`get_config(preset)` returns a `DSPConfig`; the phrasing block trades off `min_bars` / `thresh_frac`:

| Preset | `min_bars` | `thresh_frac` | Character |
|---|---|---|---|
| `default` | 4 | 0.40 | Balanced |
| `techno` | 4 | 0.35 | Slightly more sensitive |
| `house` | 6 | 0.45 | Fewer, longer sections |
| `minimal` | 8 | 0.50 | Very few boundaries |
| `aggressive` | 2 | 0.30 | Many short drop/breakdown transitions |

`max_pads` defaults to `None` in every preset; set it via `custom_config(max_pads=4)` or by editing
`PhrasingConfig`.

---

## Code Usage

### Using presets
```python
from src.dsp.config import get_config
from src.dsp.extractor import extract_track_features

config = get_config("minimal")  # or "house", "techno", "aggressive", "default"
track = extract_track_features("data/track.mp3", config=config)
```

### Custom configuration
```python
from src.dsp.config import custom_config
from src.dsp.extractor import extract_track_features

config = custom_config(min_bars=8, thresh_frac=0.5, max_pads=4)
track = extract_track_features("data/track.mp3", config=config)
```

### Direct function call
```python
from src.dsp.phrasing_engine import analyze_structure
import librosa

y, sr = librosa.load("data/track.mp3", sr=22050)
phrasing = analyze_structure(y, sr, bpm=123, min_bars=4, thresh_frac=0.4, max_pads=4)
for cue in phrasing.cue_points:
    print(cue.label, round(cue.time, 2), cue.type)
```

### From the CLI helper
```bash
# bar-level structure with 8-bar phrase snapping, limited to a 4-pad controller
python detect_structure.py data/track.mp3 --phrase 8 --pads 4
```

---

## Quick Decision Tree

**Too many sections / breakdowns** → raise `thresh_frac` (0.4 → 0.5) and/or `min_bars` (4 → 8).

**Missing a real drop** → lower `thresh_frac` (0.4 → 0.3).

**Sections too short / choppy** → raise `min_bars` (4 → 8).

**More cues than my controller has pads** → set `max_pads` to the pad count (e.g. 4).

| Change | Effect |
|--------|--------|
| ↑ `thresh_frac` | Fewer drops, more of the track reads as breakdown |
| ↓ `thresh_frac` | More drop/breakdown transitions |
| ↑ `min_bars` | Longer, fewer sections |
| ↓ `min_bars` | Shorter, more sections |
| set `max_pads=N` | Exactly N hot cues, most important sections |

Tuning has **zero performance impact** — the envelope is computed once; the parameters only affect
thresholding, merging, and labelling.

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

## Bar helpers & snapping

Cue times can be converted to bars and snapped to phrase boundaries so they match Traktor/Serato
beat grids. Helpers in `phrasing_engine.py`:

- **Time → beat:** `time_to_beat(seconds, bpm)` = `round(seconds * bpm / 60)`
- **Time → bar:** `time_to_bar(seconds, bpm)` (1 bar = 4 beats); inverse `bar_to_time(bar, bpm)`
- **Bar grouping:** `beat_to_bar_group(beat, beats_per_group=4)`
- **Snap to phrase:** `snap_to_bar_boundary(time, bpm, bars_per_phrase=16)`

Quick reference: `1 bar = 4 beats`, `8 bars = 32 beats` (typical intro/outro),
`32 bars = 128 beats` (major boundary). At 126 BPM, `seconds = beats * 60 / bpm` (128 beats ≈ 61 s).
If bar numbers look wrong, verify `track.groove.bpm` — phrasing uses whatever the groove engine
detected. `detect_structure.py --phrase N` applies this snapping to the reported bars.

---

## Low-band detection vs. phrase-locking

Two segmentation strategies are available:

| | Low-band detection (default) | Phrase-locked |
|---|---|---|
| Boundaries | Real drop/breakdown transitions | Fixed N-bar phrases (8/16/32) |
| Segment length | Variable | Exactly N bars |
| Cue spacing | Irregular | Perfectly regular |
| Reflects the track | ✅ | ❌ |
| DJ-predictable | ⚠️ | ✅ |

- **Low-band** (`analyze_structure`) — real structure; the default and what feeds the Traktor export.
- **Phrase-locked** (`create_phrase_locked_segments(duration, bpm, bars_per_phrase, include_beats)`)
  — forces every segment to N bars. Predictable but ignores actual morphology.
- **Hybrid:** run `analyze_structure`, then snap cue points to the nearest bar with
  `snap_to_bar_boundary(time, bpm, bars_per_phrase=16)` — real structure **and** grid-aligned cues.

```python
from src.dsp.phrasing_engine import analyze_structure, snap_to_bar_boundary

phrasing = analyze_structure(y, sr, bpm=126)
snapped = [(c.label, snap_to_bar_boundary(c.time, bpm=126, bars_per_phrase=16))
           for c in phrasing.cue_points]
```

---

## Validation Checklist

After tuning, verify:
- ✅ Section count is reasonable (a handful of drops/breakdowns for a 6–9 min track, not dozens)
- ✅ Hot cues land on **real** drops/breakdowns (not false positives)
- ✅ Intro and (if present) outro are correctly labelled
- ✅ Major drops are detected (not missing)
- ✅ With `max_pads=N`, exactly N cues emit and the outro takes the last pad
