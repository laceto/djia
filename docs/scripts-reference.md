# Scripts Reference

All commands, env setup, and external config. Run everything from the repo root.

## Environment (Windows)

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

> Dependencies live in **`pyproject.toml`** (single source of truth); `requirements.txt` just
> installs it editable with the dev+dashboard extras (`-e .[dev,dashboard]`).

> **Optional: S-KEY key detection** (deep-learning, much better than the chroma template for
> electronic music). Not on PyPI yet — install its deps via the extra, then the package from GitHub:
> ```bash
> pip install -e ".[skey]"
> git clone https://github.com/deezer/skey && pip install --no-deps ./skey
> ```
> When installed, `analyze` uses it automatically (`key_source="skey"`); otherwise DJIA falls back
> to the chroma method. LangGraph deps (`langgraph`, `langchain-core`) are likewise **not** bundled —
> install separately if you use the Track Tuner.

## CLI (module form — always run from repo root)

Verified against `src/cli.py` (argparse). Flags are exactly as below.

```bash
# analyze — directory (default data/) or a single track
python -m src.cli analyze                          # analyze data/
python -m src.cli analyze --data-dir "path/to/dir" # custom directory
python -m src.cli analyze --track "path/to.mp3"    # single track (persists to --db, same as a scan of just this file)
python -m src.cli analyze --workers 8              # parallel analysis with 8 worker processes
#   also: --db PATH   --skip-existing   --workers N (default: os.cpu_count(), min 1; workers<=1
#         is the old sequential path; ignored/no-op with --track)

# list-tracks — includes a "Camelot (Open Key)" column, e.g. 12A (5m)
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

# crosscheck-djuced — compare DJIA's detected keys against the keys DJUCED has stored
python -m src.cli crosscheck-djuced [--db PATH] [--djuced-input DJUCED.db] [--output report.md]
#   --djuced-input: path to DJUCED.db (default ~/Documents/DJUCED/DJUCED.db); read-only
#   matches tracks by normalized filename, normalizes both keys to Camelot (DJUCED's Open
#   Key "5m" == Camelot "12A"), and flags match / DIFFERS / unreadable / not-in-DJUCED
#   --output: also write a Markdown report (diffs highlighted)
```

## Direct DSP (no DB)

```bash
# Analyze one file straight through the DSP orchestrator
python -m src.dsp.extractor "path/to.mp3"
```

## Standalone analysis scripts (repo root)

Two self-contained scripts run the real DSP/AI code without the DB or CLI. Both write to `results/`
(gitignored) and are headless-safe (plots saved as PNGs; nothing opens a window unless `--show`).

```bash
# detect_structure.py — breakdown/drop structure reported in BAR numbers
python detect_structure.py [track.mp3] [--phrase 8] [--min-bars 4] [--thresh 0.4] [--pads 4] [--no-plot]
#   --phrase N : snap reported bars to an N-bar phrase grid (0 = raw)
#   --min-bars : merge sections shorter than N bars
#   --thresh   : kick-on threshold as a fraction of peak low-band energy
#   --pads N   : also print an N-pad hot-cue mapping (intro, main drop, outro always kept)
#   writes results/structure_bars.txt + results/plots/structure_bars.png
#   Uses src.dsp.phrasing_engine (same code as the Traktor hot-cue export) — single source of truth.

# demo_capabilities.py — end-to-end tour of all 5 phases on one track
python demo_capabilities.py [track.mp3] [--show]
#   ingestion → 4 DSP engines → mood classifier → SQLite + similarity + Traktor NML → transitions/playlist
#   mirrors console output to results/demo_report.txt; plots to results/plots/*.png
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
