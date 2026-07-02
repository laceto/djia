# Coding Rules

How to write code in this repo. Read before adding or changing any Python in `src/`.

## General

- Everything hangs off the `Track` dataclass in `src/features/schema.py` — it is the data
  contract. Read it (or `docs/schemas.md`) before touching anything that produces or consumes track
  features.
- All modules use `logging.getLogger(__name__)`; no module configures handlers — callers do.
- Run and import the package as a module from the repo root (relative imports). See
  `debugging-rules.md` if imports fail.

## DSP core (`src/dsp/`)

- `extractor.extract_track_features(path, config)` is the master orchestrator. **Engine order is
  load-bearing** because of data dependencies — do not reorder:
  1. **Groove** first → BPM feeds everything downstream.
  2. **Phrasing** → needs BPM + the phrasing params; segments carry beat/bar ranges when
     `include_beats=True`. Use `time_to_bar(seconds, bpm)` for time↔bar conversion, don't reinvent it.
  3. **Mood** → independent (Camelot key + brightness).
  4. **Curation** → consumes BPM, swing, brightness.
- See `docs/architecture.md` for the full data-flow rationale.

### Segmentation / tuning

- Segmentation behavior is driven **only** by three phrasing params: `novelty_threshold`,
  `min_segment_duration`, `breakdown_duration_threshold`.
- When changing segmentation, **change parameters in `src/dsp/config.py`** (or via
  `custom_config(...)`) and pass a `config=` — never hardcode thresholds in the engines.
- Named presets (`default`, `minimal`, `house`, `techno`, `aggressive`) trade segment count vs.
  length. See `PARAMETER_REFERENCE.md` for what each parameter does.

## LangGraph Track Tuner (`src/ai/track_tuner_*.py`)

- Optional, self-contained agent. Nodes are pure `(state, config) -> dict` and emit
  `[NodeName]`-prefixed `AIMessage`s for tracing — keep that pattern for new nodes.
- State is a `TypedDict` with `operator.add` / `add_messages` reducers in `track_tuner_state.py`.
- The preset dicts here (`DEFAULT_CONFIGS` in `track_tuner_state.py`) **mirror `dsp/config.py`** —
  if you change one, change the other to keep them consistent.
- LangGraph deps are **not** in `requirements.txt` and `src/ai/__init__.py` deliberately does not
  import these modules. Import them directly from `src.ai.track_tuner_graph`. See
  `debugging-rules.md`.
- Scoring rubric lives in `evaluate_quality`; full rubric in `LANGGRAPH_TRACK_TUNER_README.md`.

## Legacy — do not extend

`src/main.py`, `audio_analysis.py`, `mixing_metrics.py`, `structure_detection.py` are an earlier
standalone implementation kept for backward compatibility. The `src/dsp` + `src/ai` pipeline
supersedes them. **Don't extend the legacy files — add to the phased pipeline instead.**
