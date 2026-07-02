# Debugging Rules

Known failure modes and environment quirks. Read when a script errors or output looks wrong.

## Gotchas

- **Run the CLI as a module** (`python -m src.cli ...`) from the repo root. The package uses relative
  imports and won't run as a loose script — `ModuleNotFoundError` / `ImportError` almost always means
  you ran it the wrong way.

- **LangGraph deps are NOT in `requirements.txt`.** The Track Tuner needs `langgraph` and
  `langchain-core`; install them separately. `src/ai/__init__.py` deliberately does **not** import the
  track_tuner modules (so the CLI keeps working without those packages) — import them directly from
  `src.ai.track_tuner_graph`. An `ImportError` from `src.ai` on tuner code means the deps are missing.

- **Reset a locked/corrupt DB** by deleting `data/djia.db`; it is recreated on the next `analyze` run.

- **`data/` and `results/` are gitignored** (audio + generated artifacts were removed from the repo in
  commit 9099368). Don't commit tracks, `.db` files, or NML output.

- **Demucs first run downloads ~1GB** (stem separation model), then caches on disk. A slow/hanging
  first stem separation is usually the model download, not a bug.

## Where to look

- Data flow / who-produces-what → `docs/architecture.md`.
- Data shapes (Track dataclass, DB schema) → `docs/schemas.md`.
