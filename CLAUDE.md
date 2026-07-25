# DJIA — Agent Router

DSP feature-extraction pipeline for techno/house tracks: extracts BPM, key, structural segments,
mood, danceability → SQLite → similarity/playlists → Traktor NML export. This file routes you to the
right context. The data contract for everything is the `Track` dataclass in `src/features/schema.py`.

## Identify Your Task

**CODING (DSP core)** — Editing engines, extractor, or the config/preset system in `src/dsp/`
→ READ: `coding-rules.md`
→ ALSO READ: `docs/architecture.md` for engine order and data dependencies

**CODING (AI layer)** — stem separation, mood classifier, segmentation, transition mapper, playlist generator
→ READ: `coding-rules.md`
→ ALSO READ: `docs/architecture.md` for how the AI layer consumes DSP output

**TESTING** — Writing tests or running the suite
→ READ: `testing-rules.md`

**DEBUGGING** — A script errored, the DB is locked, imports fail, or output looks wrong
→ READ: `debugging-rules.md`
→ ALSO READ: `docs/architecture.md` for pipeline flow

**TUNING SEGMENTATION** — Changing segment count/behavior via phrasing parameters
→ READ: `PARAMETER_REFERENCE.md` for what each parameter does
→ ALSO READ: `coding-rules.md` (segmentation/tuning section)

**DATA / EXPORT** — DB queries, similarity search, Traktor NML export, or DJUCED cue export
→ READ: `docs/api-reference.md`
→ ALSO READ: `docs/scripts-reference.md` for CLI commands

**RUNNING THE PIPELINE** — CLI, orchestrator, ingestion, analyzing a library
→ READ: `docs/scripts-reference.md`

## Repo Layout

```
src/
  dsp/          core pipeline: extractor + groove/phrasing/mood/curation engines, config.py
  ai/           stem_separator, classifier, segmentation, processor, transition_mapper,
                playlist_generator, setlist_generator
  features/     schema.py — the Track dataclass, THE data contract
  database/     SQLite schema + store (TrackStore)
  matching/     cosine similarity over feature vectors
  traktor/      NML exporter with hot cues
  djuced/       hot-cue exporter writing directly into DJUCED.db (Hercules)
  ingestion/    scanner + loader (librosa, 22,050 Hz mono)
  orchestrator.py   ties ingestion → DSP → AI → DB
  cli.py            argparse subcommands
  main.py, audio_analysis.py, mixing_metrics.py, structure_detection.py   LEGACY — do not extend
tests/          pytest suite (one file per subsystem)
docs/           reference documentation (see table below)
data/ results/ db/  gitignored — never commit tracks, .db, or NML
```

Root holds only: `pyproject.toml`/`requirements.txt`, `CLAUDE.md`, the `*-rules.md` files,
`README.md`, the current reference guide (`PARAMETER_REFERENCE.md`), and standalone analysis scripts
(`demo_capabilities.py` — full 5-phase tour; `detect_structure.py` — drop/breakdown bars).
Older phase/tuning docs are archived under `docs/archive/`.

## Reference Docs (load only what your task requires)

| File | Contents |
|---|---|
| `docs/architecture.md` | Data flow, engine order + dependencies, config/preset system, AI layer |
| `docs/schemas.md` | `Track` dataclass + component results, SQLite table schema |
| `docs/api-reference.md` | Public entry points: extract_track_features, TrackStore, similarity, orchestrator |
| `docs/scripts-reference.md` | All CLI commands with flags, env setup, `.env` config |
| `PARAMETER_REFERENCE.md` | What each phrasing/segmentation parameter does (current) |

## Instructions

1. Identify your task above
2. Load the rule file for that task
3. Load only the reference docs the task actually requires — do not load all docs
4. Always-true, no rule file needed: run the CLI as a module from repo root (`python -m src.cli ...`);
   `data/` + `results/` + `db/` are gitignored
5. If unsure which category fits, ask — do not guess
