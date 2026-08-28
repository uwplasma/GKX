# GKX 1.8.2 geometry ownership and deletion inventory

Status: Phase 0 audit; documentation only. This inventory freezes observed
interfaces and an approved target boundary. It does not claim that the target
deletions or cross-repository parity gates are complete.

## Frozen revisions

| Repository | Revision | Role observed at this revision |
|---|---|---|
| GKX | `4104bf4a2d7463fcd56e9c38434d88510377d2b4` | Flux-tube contract, imported-WOUT path, solver, generic objectives, and legacy equilibrium-specific bridges |
| VMEX | `f7bd9469a059d2c54b6d85a125205c8c245c0a10` | Equilibrium state/WOUT and live-state field-line geometry |
| booz_xform_jax | `1d5e8c8a72db8a745e7cb56fb077b64bb85d0763` | Boozer transform kernels and file-compatible driver |

The checkouts were clean when audited. The names below come from executable
modules and package metadata, not from roadmap prose.

## Approved one-way boundary

```text
VMEX state/runtime ──> VMEX field-line arrays ──> GKX contract ──> GKX solver/objective
       │
       └──> VMEX Boozer input tables ──> booz_xform_jax transform ──> consumers

WOUT file ──> one file adapter (missing target seam) ──> same field-line arrays
```

- VMEX owns equilibrium solution physics, VMEC-compatible WOUT data, and the
  transformation from a live equilibrium state to sampled field-line arrays.
- booz_xform_jax owns Boozer-coordinate transformation. VMEX may prepare its
  equilibrium-specific input tables and call that public transform.
- GKX owns the generic normalized flux-tube data contract, validation of that
  contract, grid/operator consumption, turbulence solves, and objectives that
  operate on GKX arrays/results.
- Equilibrium-specific optimization callables belong next to the equilibrium
  state in VMEX. They may call GKX through its optional `turbulence` extra;
  neither VMEX geometry nor booz_xform_jax imports GKX.
- Arrays flow into GKX. GKX must not reconstruct VMEX equilibrium physics or
  Boozer tables after the replacement seams and parity gates exist.

This boundary avoids a core dependency cycle: VMEX depends on booz_xform_jax;
its optional `turbulence` extra depends on GKX; GKX should eventually consume
VMEX mappings without making VMEX a core dependency.

## Current public seams

### VMEX

`vmex.core.turbulence` exposes:

- `gk_fieldline_geometry(state, rt, ...)`: pure-JAX mapping with no GKX
  import. It supports symmetric/asymmetric states, PEST field-line sampling,
  and optional equal-arc resampling.
- `flux_tube_geometry(state, rt, ...)`: thin optional wrapper around
  `gkx.flux_tube_geometry_from_mapping`.
- `turbulence_objective_vector`, `turbulent_growth_rate`,
  `quasilinear_flux_proxy`, and `nonlinear_heat_flux_proxy`: equilibrium-side
  wrappers over GKX objective seams.

The geometry function is a public symbol of the domain module, but is not a
top-level `vmex` lazy export. The GKX-dependent functions are packaged behind
VMEX's `turbulence = ["gkx>=1.7.1", ...]` optional extra.

`vmex.core.boozer_tables.boozer_input_tables(state, rt, j)` owns the traceable
conversion from one VMEX half-mesh surface to WOUT-convention spectral inputs.
`vmex.WoutData`, `vmex.read_wout`, `vmex.write_wout`, and
`vmex.wout_from_state` own VMEC-compatible file data and state serialization.

Missing seam: there is no `gk_fieldline_geometry_from_wout(...)` (or
equivalent same-contract function). A user with an existing WOUT therefore
cannot obtain the VMEX live-state mapping contract without either reconstructing
a state or using GKX's separate imported-WOUT implementation.

### booz_xform_jax

The top-level package owns and exports:

- `Booz_xform` for the legacy/file-compatible object route;
- `BoozXformConstants`, `BoozXformGrids`,
  `prepare_booz_xform_constants`, and
  `prepare_booz_xform_constants_from_inputs`;
- `booz_xform_jax`, `booz_xform_jax_impl`, and
  `booz_xform_from_inputs` for JAX-native transforms;
- plotting and CLI compatibility utilities.

It imports neither GKX nor VMEX. VMEX directly imports this package in its
Boozer driver, neoclassical, omnigenity, optimization, and plotting modules.

### GKX

The stable owner seam is
`gkx.flux_tube_geometry_from_mapping(mapping, ...)`. Required one-dimensional
profiles are:

`theta`, `gradpar`, `bmag`, `bgrad`, `gds2`, `gds21`, `gds22`, `cvdrift`,
`gbdrift`, `cvdrift0`, and `gbdrift0`.

Optional profiles are `jacobian` and `grho`. Scalar metadata includes `q`,
`s_hat`/`shat`, `epsilon`, `R0`, `B0`, `alpha`, `drift_scale`, `kxfac`, and
`theta_scale`; structural metadata includes `nfp`, `kperp2_bmag`,
`bessel_bmag_power`, and `theta_closed_interval`.

GKX also owns the generic geometry-input objective seams
`solver_linear_operator_matrix_from_geometry`,
`solver_growth_rate_from_geometry`, `solver_objective_vector_from_geometry`,
and `solver_scalar_objective_from_vector`.

The current facade has 51 top-level exports whose name or target is
VMEC/VMEX/Boozer-specific. These are Phase 1 compatibility debt, not the target
public surface.

## Observed validation

At the frozen revisions on CPU:

- six focused VMEX geometry tests pass: stability conventions, internal
  identities, vacuum drift identity, equal-arc consistency, validation
  failures, and the GKX mapping contract;
- three focused booz_xform_jax tests pass: small reference parity, finite
  covariant-field gradients, and differentiable Jacobian harmonics;
- GKX's `vmec_tensor_mapping.py` is already a thin adapter over
  `vmex.core.turbulence.gk_fieldline_geometry`.

These results establish that usable seams exist. They do not establish full
WOUT/live-state equivalence, all normalization parity, or end-to-end optimizer
gradient parity.

## Duplication and deletion candidates

The inventory below is prospective. No file may be deleted solely because it
appears here; its replacement and gates must land first.

### Live-state and Boozer geometry

Nine GKX modules contain 5,176 lines at the frozen revision:

| GKX module | Lines | Target replacement | Required gate before deletion/consolidation |
|---|---:|---|---|
| `geometry/booz_xform_bridge.py` | 712 | booz_xform_jax public array API plus one GKX contract adapter | symmetric/asymmetric value and AD parity |
| `geometry/vmec_tensor_mapping.py` | 84 | retain only the smallest `from_vmex` adapter, or call VMEX directly | exact mapping-key/value parity and error-contract test |
| `geometry/vmec_boozer_constants.py` | 91 | booz_xform_jax prepared constants | prepared/unprepared transform parity and cache-key test |
| `geometry/vmec_boozer_core.py` | 948 | VMEX `boozer_input_tables` plus booz_xform_jax | surface/mode/value parity and traced state-gradient parity |
| `geometry/vmec_boozer_drifts.py` | 206 | VMEX field-line geometry | drift convention parity including finite beta |
| `geometry/vmec_boozer_derivatives.py` | 810 | VMEX field-line geometry/Boozer input ownership | radial derivative and equal-arc parity |
| `geometry/vmec_state_controls.py` | 546 | VMEX state/control API | parameter-vector round trip and tangent parity |
| `geometry/vmec_state_sensitivity.py` | 841 | VMEX-side derivative gates plus generic GKX AD checks | AD/FD parity on the same public composition |
| `geometry/vmec_flux_tube_reports.py` | 938 | compact cross-repository contract tests/artifacts | live-state versus WOUT parity at declared tolerances |

`geometry/backend_discovery.py` is not counted above. Its 319 lines mix local
checkout discovery with backend import policy; it should shrink after the
dependency boundary is explicit, but some optional-backend discovery may
remain.

### Equilibrium-specific objectives and optimization

Nine GKX modules contain another 5,099 lines:

| GKX module group | Target owner | Gate before removal from GKX |
|---|---|---|
| `objectives/solver_vmec.py`, `vmec_boozer.py`, `vmec_boozer_context.py` | VMEX wrappers composed with generic GKX geometry objectives | scalar/vector/table value parity |
| `objectives/vmec_boozer_gradients.py`, `vmec_boozer_fd.py` | VMEX optimization validation plus generic GKX AD/FD utilities | conditioning-aware AD/FD parity on named controls |
| `objectives/vmec_boozer_line_search.py`, `vmec_transport_branch.py`, `vmec_transport_optimization.py` | VMEX optimizer/problem interfaces | candidate ordering, constraint, and holdout replay parity |
| `objectives/vmec_transport.py` | VMEX `turbulence` wrappers | linear/quasilinear/nonlinear-window objective value parity and claim labels |

Together the two candidate groups are 18 files and 10,275 lines, about eleven
percent of the frozen 91,494 installable Python lines. This is an upper bound:
generic GKX objective reducers or audit logic that has no VMEX equivalent must
be retained under a generic owner rather than deleted.

## Required cross-repository gates

Land these in small PRs and freeze tolerances before removing an old path:

1. **Mapping schema:** exact required keys, shapes, dtypes, finite values,
   endpoint convention, and constant equal-arc `gradpar`.
2. **Normalization:** `B0`, `R0`, signed flux, `q`, `s_hat`, `gds21`,
   `gbdrift0`, pressure drift, Jacobian, and `grho` conventions.
3. **Live-state parity:** VMEX public mapping versus the surviving GKX adapter
   for symmetric, asymmetric, vacuum, and finite-beta cases.
4. **WOUT parity:** the proposed VMEX WOUT mapping versus live state and the
   current GKX imported-WOUT route, separating spectral truncation differences
   from implementation defects.
5. **Boozer parity:** VMEX input tables plus booz_xform_jax versus the current
   GKX VMEC/Boozer route for values, mode ordering, equal-arc profiles, and
   finite-beta drifts.
6. **Derivative parity:** JVP/VJP and centered finite differences for selected
   boundary, profile, and field-line controls with explicit conditioning.
7. **Solver parity:** the same mapping produces matching linear eigenvalues,
   physical flux weights, and bounded nonlinear-window values in GKX.
8. **Optimization parity:** VMEX-side objectives reproduce accepted GKX
   objective values, gradients where supported, and line-search decisions.
9. **Import boundary:** analytic/Miller/imported geometry does not import VMEX;
   non-Boozer GKX paths do not import booz_xform_jax; VMEX live geometry does
   not import GKX.
10. **Packaging:** base GKX and base VMEX install/import independently; only
    VMEX's turbulence extra activates the equilibrium-to-GKX composition.

Raw WOUTs and comparator output remain local. Permanent tests should use
small redistributable fixtures or self-contained mathematical identities.

## Dependency decisions exposed by the audit

- GKX currently declares booz_xform_jax as a core dependency even though the
  approved target makes Boozer transformation a companion path. Move it out of
  GKX core only after a clean-wheel matrix proves analytic, Miller, imported
  WOUT, live VMEX, and Boozer workflows have truthful extras and messages.
- VMEX correctly keeps GKX behind its `turbulence` extra. Its core live-state
  mapping remains GKX-independent.
- Do not make GKX depend on VMEX merely to preserve legacy discovery helpers.
  The eventual `from_vmex` adapter should accept arrays/mappings or be an
  optional integration seam.
- Do not copy booz_xform_jax kernels into either repository.

## Reproduction commands

From the three frozen checkouts under one parent directory:

```bash
git -C GKX rev-parse HEAD
git -C VMEX rev-parse HEAD
git -C BOOZ_XFORM_JAX rev-parse HEAD
rg -n "vmex|booz_xform" GKX/src/gkx --glob '*.py'
rg -n "gkx|booz_xform" VMEX/vmex --glob '*.py'
rg -n "gkx|vmex" BOOZ_XFORM_JAX/src --glob '*.py'
python -m pytest -q VMEX/tests/test_turbulence.py \
  -k 'geometry_matches_stability_conventions or geometry_internal_identities or vacuum_limit or equal_arc or surface_index_validation or contract_passes_gkx_validation'
python -m pytest -q BOOZ_XFORM_JAX/tests/test_jax_api.py \
  -k 'jax_api_matches_reference_small or jax_api_covariant_field_gradients_are_finite or jacobian_harmonics_are_differentiable'
```

The focused results recorded above used a shared Python 3.11 environment with
JAX on CPU. They are interface checks, not performance claims.
