# GKX 1.8.2 dependency and runtime-import inventory

Baseline packaging commit: `7c4d4598bbd1263fd451a57dc814ac50c65579f3` after the optional-dependency repair. Counts cover Python files under `src/gkx`; test, documentation, and developer-tool imports are not runtime-package requirements.

## Core declarations

| Requirement | Import root | Static source files | Literal dynamic files | Finding |
| --- | --- | ---: | ---: | --- |
| `jax>=0.10.1` | `jax` | 114 | 0 | Core array/JIT/autodiff runtime. |
| `jaxlib>=0.10.1` | `jaxlib` | 0 | 0 | Declared companion lower bound; no direct Python import. |
| `numpy` | `numpy` | 106 | 0 | Host arrays, serialization, and validation numerics. |
| `matplotlib` | `matplotlib` | 9 | 0 | Plotting and figure artifacts. |
| `scipy` | `scipy` | 4 | 0 | Reference numerics and selected analysis routines. |
| `netCDF4` | `netCDF4` | 14 | 0 | Runtime and artifact NetCDF I/O. |
| `diffrax` | `diffrax` | 1 | 0 | Legacy/promoted-audit integration route pending Phase 2 consolidation. |
| `equinox` | `equinox` | 1 | 0 | PyTree/module utilities used by the Diffrax route. |
| `solvax>=0.12.0` | `solvax` | 7 | 0 | Matrix-free eigensolver and implicit-solver ownership. |
| `booz_xform_jax` | `booz_xform_jax` | 0 | 3 | Loaded literally through importlib in geometry bridges. |
| `tqdm` | `tqdm` | 0 | 0 | No package-source import found; deletion candidate pending CLI/user audit. |

## Optional extras

| Extra | Declared requirement | Package-source relation |
| --- | --- | --- |
| `docs` | `sphinx` | development/documentation/release only |
| `docs` | `sphinx-rtd-theme` | development/documentation/release only |
| `docs` | `matplotlib` | also a core dependency |
| `release` | `build` | development/documentation/release only |
| `release` | `twine` | development/documentation/release only |
| `dev` | `pytest` | development/documentation/release only |
| `dev` | `pytest-cov` | development/documentation/release only |
| `dev` | `ruff==0.16.4` | development/documentation/release only |
| `dev` | `mypy` | development/documentation/release only |
| `dev` | `mpmath` | development/documentation/release only |
| `dev` | `gmpy2` | development/documentation/release only |
| `dev` | `pandas` | lazy zonal dataframe/CSV helpers; no base import required |
| `dev` | `Pillow` | lazy image optimization in `gkx.artifacts.figure_style` |
| `dev` | `mkdocs` | development/documentation/release only |
| `dev` | `sphinx` | development/documentation/release only |
| `dev` | `sphinx-rtd-theme` | development/documentation/release only |
| `dev` | `matplotlib` | also a core dependency |
| `dev` | `build` | development/documentation/release only |
| `dev` | `twine` | development/documentation/release only |
| `assets` | `Pillow` | lazy image optimization in `gkx.artifacts.figure_style` |
| `validation` | `pandas` | lazy zonal dataframe/CSV helpers; no base import required |

## Imported roots outside core declarations

| Import root | Static source files | Literal dynamic files | Classification |
| --- | ---: | ---: | --- |
| `PIL` | 1 | 0 | covered by the `assets`/`dev` Pillow extra; lazy function import |
| `booz_xform` | 0 | 1 | optional backend fallback, not supplied by `booz_xform_jax` and not declared by GKX |
| `cycler` | 1 | 0 | direct import supplied transitively by core matplotlib; declare directly or import through matplotlib |
| `pandas` | 1 | 1 | TYPE_CHECKING plus lazy import; covered by the `validation`/`dev` extra |
| `tomli` | 1 | 0 | Python <3.11 fallback is unreachable under requires-python >=3.11; cleanup candidate |
| `vmex` | 1 | 4 | optional geometry/optimization bridge loaded statically in one local scope and dynamically elsewhere; ownership matrix required before packaging decision |

## Literal dynamic-import files

### `booz_xform_jax`

- `src/gkx/geometry/booz_xform_bridge.py`
- `src/gkx/geometry/vmec_boozer_constants.py`
- `src/gkx/geometry/vmec_boozer_core.py`

### `pandas`

- `src/gkx/diagnostics/zonal_validation.py`

### `vmex`

- `src/gkx/geometry/vmec_boozer_constants.py`
- `src/gkx/geometry/vmec_boozer_core.py`
- `src/gkx/geometry/vmec_state_sensitivity.py`
- `src/gkx/geometry/vmec_tensor_mapping.py`

## Decisions exposed by the inventory

- `tqdm` is declared but unused under `src/gkx`; verify console and downstream obligations before deleting it.
- `cycler` is imported directly but only transitively declared through matplotlib; Phase 1 packaging should either declare it or route the import through the owner dependency.
- `booz_xform` is an undeclared optional fallback; make it an explicit extra or remove the fallback after the JAX bridge owns the supported path.
- `tomli` is dead compatibility code at the Python 3.11 floor.
- VMEX remains optional in package metadata even though several promoted geometry/objective paths load it; the VMEX/GKX ownership matrix must decide whether those paths move to VMEX, become an explicit extra, or leave the GKX 3 surface.
- `jaxlib` is explicitly lower-bounded without a direct import, while JAX also owns its runtime dependency; retain only if the independent compatibility rationale survives the minimum-stack audit.

## Reproduction

Parse `pyproject.toml` with `tomllib`; parse every `src/gkx/*.py` file with `ast`; count top-level roots from `import`/`from` nodes and literal string roots passed to `import_module`. Conditional and type-checking imports are then classified by source inspection, not counted as unconditional imports.
