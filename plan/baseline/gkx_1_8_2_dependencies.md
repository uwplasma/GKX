# GKX 1.8.2 dependency and runtime-import inventory

Status: Phase 0 packaging snapshot; documentation only. Baseline:
`4104bf4a2d7463fcd56e9c38434d88510377d2b4`, after the pandas/Rich base-wheel
repair. Counts are AST-derived from the 199 Python files under `src/gkx`.

## Base requirements

| Declaration | Static import files | Literal dynamic import owners | Observed role | Phase 1 disposition to prove |
|---|---:|---|---|---|
| `jax>=0.10.1` | 114 | none | core arrays, transforms, compilation, AD, devices | retain core with tested floor/latest stacks |
| `jaxlib>=0.10.1` | 0 | none | compiled JAX backend and version compatibility | retain explicit compatible floor even without a Python import |
| `numpy` | 106 | none | host arrays, I/O, validation, plotting | retain core |
| `matplotlib` | 9 | none | plotting and demo modules | decide core versus plotting extra through clean-wheel workflow tests |
| `scipy` | 4 | none | interpolation/integration and linear-solver helpers | retain only exercised owners after consolidation |
| `netCDF4` | 14 | none | canonical/runtime and foreign NetCDF I/O | retain core while NetCDF is the canonical result format |
| `diffrax` | 1 | none | temporary integrator oracle through `diffrax_core.py` | remove after native explicit/IMEX migration parity |
| `equinox` | 1 | none | Diffrax runtime seam | remove with Diffrax unless another promoted owner remains |
| `solvax>=0.12.0` | 7 | none | Krylov/implicit solves and custom derivatives | retain only validated algorithms and keep a released-version lane |
| `booz_xform_jax` | 0 | three direct `import_module` owners plus backend discovery | optional VMEC/Boozer bridge | move out of GKX core after geometry ownership/parity and clean-wheel gates |
| `tqdm` | 0 | none | no executable use under `src/gkx` | remove unless a promoted workflow demonstrates an owner |

The static count includes imports under type-checking or guarded scopes; the
exceptions are called out below. A zero import count is not automatically an
unused dependency: `jaxlib` supplies JAX's executable backend. Conversely,
declaration does not justify retention: `tqdm` has no source owner.

## Optional extras

| Extra | Declarations | Source ownership |
|---|---|---|
| `docs` | Sphinx, RTD theme, matplotlib | documentation build configuration and plots |
| `release` | build, twine | distribution build/metadata checks; no installable-source import |
| `dev` | pytest, pytest-cov, pinned Ruff, MyPy, mpmath, gmpy2, pandas, Pillow, MkDocs, Sphinx/theme, matplotlib, build, twine | tests, lint/type tools, validation, assets, docs, and release |
| `assets` | Pillow | optional PNG palette quantization in `save_figure` |
| `validation` | pandas | zonal dataframe/CSV helpers through `_require_pandas` |

The base-wheel repair makes pandas truly optional: `DataFrame` is type-only at
runtime and dataframe/CSV helpers load pandas with one actionable error.
Pillow is imported inside the palette-quantization branch only. Neither is a
base import requirement.

## Imported but not directly declared

| Import | Files | Boundary | Finding |
|---|---:|---|---|
| `cycler` | 1 | unconditional module import in `artifacts/figure_style.py` | supplied transitively by matplotlib but used directly; either declare it, use matplotlib's public exposure, or remove the direct dependency |
| `PIL` | 1 | guarded function-local import | correctly covered by `assets`/`dev`, not base |
| `pandas` | 1 static type-only plus 1 literal dynamic owner | guarded optional feature | correctly covered by `validation`/`dev` after #130 |
| `tomli` | 1 fallback branch | Python `<3.11` only | unreachable under current `requires-python >=3.11`; the module docstring still claims Python 3.10 support and is stale |
| `vmex` | 1 guarded import plus 4 literal dynamic owners | optional live-equilibrium integration | intentionally not declared in GKX core; must remain an optional one-way adapter |
| `booz_xform` | 1 literal dynamic owner | optional legacy/Fortran-compatible backend | not declared; auto-discovery must fail clearly when absent |

Standard-library and internal `gkx` imports are excluded from dependency
decisions. `PIL` maps to the Pillow distribution, and `tomli` maps to the
backport distribution rather than its import spelling.

## Literal dynamic-import ownership

- `booz_xform_jax.jax_api` is loaded directly by
  `geometry/booz_xform_bridge.py`, `geometry/vmec_boozer_constants.py`, and
  `geometry/vmec_boozer_core.py`; `backend_discovery.py` also resolves the
  package through its generic search helper.
- VMEX is loaded by `geometry/booz_xform_bridge.py`,
  `vmec_boozer_constants.py`, `vmec_boozer_core.py`,
  `vmec_state_sensitivity.py`, and `vmec_tensor_mapping.py`.
- pandas is loaded only by
  `diagnostics/zonal_validation.py:_require_pandas` outside type checking.
- The runtime imports no `tqdm` symbol and no literal dynamic `tqdm` module.

These paths reinforce the geometry ownership audit: live VMEX and Boozer
integration should become a small optional composition, not define the base
installation.

## Direct-import file owners

Compact source ownership for the smaller declared dependencies is:

- SciPy: `geometry/imported_vmec.py`,
  `geometry/vmec_boozer_derivatives.py`,
  `geometry/vmec_field_line_sampling.py`, and `solvers/linear/krylov.py`.
- Diffrax and Equinox: `solvers/time/diffrax_core.py` only.
- SOLVAX: `geometry/autodiff_checks.py`, `objectives/core.py`,
  `solvers/linear/adaptive_propagator.py`, `solvers/linear/implicit.py`,
  `solvers/linear/krylov.py`, `solvers/linear/krylov_algorithms.py`, and
  `solvers/nonlinear/imex.py`.
- Matplotlib: eight `artifacts` modules plus `workflows/demo.py`.
- NetCDF4: fourteen artifact, calibration, geometry, and WOUT owners. This is
  consistent with the frozen canonical output contract, but should consolidate
  behind the result/I/O owner in Phase 2.

JAX and NumPy are intentionally pervasive (114 and 106 files). Their counts
are architecture evidence: consolidation must reduce import sites with module
ownership, not add wrapper dependencies around arrays.

## Packaging obligations

1. Test Python 3.11 with minimum compatible versions and a current supported
   stack. Import-only tests are insufficient for JAX/JAXLIB and SOLVAX.
2. A base wheel must run analytic/Miller linear and nonlinear smoke workflows,
   write/read canonical outputs, and plot only if matplotlib remains core.
3. Validation and asset extras must be tested both absent (actionable errors)
   and present (dataframe/CSV and palette output work).
4. VMEX and Boozer workflows need their own explicit optional install contract;
   local-checkout discovery is not a packaging interface.
5. Native explicit/IMEX parity must land before removing Diffrax/Equinox.
6. Remove `tqdm` only in a focused dependency PR with clean-wheel and CLI
   evidence; this audit does not change metadata.
7. Resolve the direct `cycler` import instead of relying silently on a
   transitive dependency.
8. Delete the obsolete Python 3.10 `tomli` fallback and stale support wording
   only in a separate source-neutral cleanup after the floor gate confirms
   Python 3.11.

## Reproduction

```bash
python - <<'PY'
import ast
import tomllib
from pathlib import Path

root = Path('.')
project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
print(project['dependencies'])
print(project['optional-dependencies'])

owners = {}
for path in sorted((root / 'src/gkx').rglob('*.py')):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split('.')[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module.split('.')[0]]
        else:
            names = []
        for name in names:
            owners.setdefault(name, set()).add(str(path))
for name, paths in sorted(owners.items()):
    print(name, len(paths))
PY
```

Literal `importlib.import_module` calls and generic backend-discovery calls must
also be reviewed; the static loop alone deliberately cannot infer their
runtime requirement level.
