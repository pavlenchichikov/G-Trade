# Contributing

Thanks for your interest in G-Trade. This is a personal research project licensed under
**CC BY-NC 4.0** (non-commercial) - please keep that in mind for any reuse or contribution.

## Development setup

```bash
pip install -r requirements.txt
pip install ruff pytest
```

The web UI and the test suite do **not** need TensorFlow-heavy training. Only
`train_hybrid.py` (and anything that retrains) is CPU-bound and slow.

## Before opening a pull request

- `pytest -q` - the test suite passes
- `ruff check .` - lint is clean
- **ASCII-only** source (no smart quotes, em-dashes, or arrows); match the style,
  naming and comment density of the file you are editing
- Keep each PR focused on **one topic**; write a clear description of the problem solved
- **Do not commit** secrets, `.env`, model artifacts, `market.db`, logs, or the local
  research journals (`_ar_*.json`, `_ar_wiki/`) - these are git-ignored on purpose

## Good to know

- `ab_labeling.py` (and its test) is a **local, git-ignored** experiment script - edit it
  locally, but it is not part of the tracked repository.
- Heavy retrains are best run in chunks (`GTRADE_ASSETS`, ~15 assets per process) - see
  [Training](README.md#training) in the README.
- The project follows a design-first, test-driven workflow: a short spec and plan come
  before the code, and changes are small, well-tested, and additive behind env flags
  (default-off and byte-identical) wherever possible.
