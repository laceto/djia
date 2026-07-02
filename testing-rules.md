# Testing Rules

How to test and check code quality. Read before writing tests or running the suite.

```bash
# Tests (run from repo root)
pytest tests/ -v                       # all
pytest tests/test_dsp.py -v            # one file
pytest tests/test_dsp.py::test_name -v # one test
pytest --cov=src tests/                # with coverage

# Lint / format
ruff check src/
ruff format src/
```

## Conventions

- Tests live in `tests/`, one file per subsystem: `test_dsp.py`, `test_ai.py`, `test_ingestion.py`,
  `test_database.py`, `test_traktor.py`, `test_full_pipeline.py`, `test_mixing_metrics.py`.
- Shared fixtures are in `tests/conftest.py`.
- Build test `Track` objects with `create_test_track(...)` from `src/features/schema.py` rather than
  constructing the nested dataclasses by hand.
- Run `ruff check` / `ruff format` on any `src/` code you touch before considering it done.
