# Scripts Reference

All commands, env setup, and external config. Run everything from the repo root.

## Environment (Windows)

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

> LangGraph deps (`langgraph`, `langchain-core`) are **not** in `requirements.txt` — install them
> separately if you use the Track Tuner.

## CLI (module form — always run from repo root)

Verified against `src/cli.py` (argparse). Flags are exactly as below.

```bash
# analyze — directory (default data/) or a single track
python -m src.cli analyze                          # analyze data/
python -m src.cli analyze --data-dir "path/to/dir" # custom directory
python -m src.cli analyze --track "path/to.mp3"    # single track
python -m src.cli analyze --workers 8              # parallel analysis with 8 worker processes
#   also: --db PATH   --skip-existing   --workers N (default: os.cpu_count(), min 1; workers<=1
#         is the old sequential path; ignored/no-op with --track)

# list-tracks
python -m src.cli list-tracks [--limit N] [--db PATH]      # --limit default 100

# find-similar (track_id positional; no bpm-tolerance flag)
python -m src.cli find-similar <track_id> [--top-k 5] [--db PATH]

# generate-playlist (start_id end_id positional; steps positional, default 5)
python -m src.cli generate-playlist <start_id> <end_id> [steps] [--db PATH]
#   e.g. generate-playlist 1 10 5   → 5-step path from track 1 → 10

# generate-setlist — data-driven 5-phase set (warm-up→build→peak→breakdown→comeback)
python -m src.cli generate-setlist [--tracks 28] [--output results/setlist_5phase.md] [--db PATH]
#   writes a markdown phase plan + per-transition mix sheets (element-onset mix points)
#   --skip-mix-sheets: skip the audio loads (fast; transitions lose the deck timings)
#   mix points are cached in results/mix_points_cache.json — first run is slow, reruns instant

# export-traktor (output nml_path positional, default djia_export.nml)
python -m src.cli export-traktor [out.nml] [--traktor-input Collection.nml] [--db PATH]
#   --traktor-input: existing Traktor Collection.nml to source hot cues from

# spectrogram (track_id positional) — regenerate the .npy spectrogram for an already-analyzed track
python -m src.cli spectrogram <track_id> [--db PATH] [--spectrogram-dir data/spectrograms]
#   e.g. spectrogram 1   → loads the track's audio and saves data/spectrograms/1.npy
```

## Direct DSP (no DB)

```bash
# Analyze one file straight through the DSP orchestrator
python -m src.dsp.extractor "path/to.mp3"
```

## Tests / lint

See `testing-rules.md`:

```bash
pytest tests/ -v
ruff check src/
ruff format src/
```

## External config (`.env`)

```
OPENAI_API_KEY=sk-...     # optional, for OpenAI audio embeddings
DEMUCS_MODEL=htdemucs     # stem separation model
```

Read via `os.getenv()`. Never commit secrets.
