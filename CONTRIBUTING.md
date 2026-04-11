# Contributing to SPECTRAX-GK

Thanks for helping build SPECTRAX-GK.

## Development setup

```bash
pip install -e .
pip install -e .[dev]
```

## Running tests

```bash
pytest
```

By default, integration tests are excluded to keep local runs fast. To run the
full integration suite:

```bash
pytest -m integration
```

For quick local feedback with a 5-minute per-file cap:

```bash
python tools/run_tests_fast.py
```

## Building docs

```bash
python -m sphinx -W -b html docs docs/_build/html
```

## Packaging checks

```bash
python -m build
twine check dist/*
```

## Code layout

- `src/spectraxgk`: library code
- `examples`: runnable examples
- `tests`: unit tests
- `docs`: Sphinx docs

## Style

- Keep functions small and pure when possible.
- Favor JAX-friendly control flow (`lax.scan`, `vmap`, `jit`).
- Add tests for new functionality.
