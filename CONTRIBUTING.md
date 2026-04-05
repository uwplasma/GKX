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
