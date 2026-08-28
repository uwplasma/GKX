# GKX 1.8.2 equilibrium ExB-shear readiness decision

Status: Phase 0 audit complete. Equilibrium-flow shear remains a Python
research API and is deferred from GKX 3.0 stable to the first GKX 3.1 physics
lane.

This decision applies the preapproved roadmap rule. It does not remove the
research implementation, weaken a gate, or claim that a single treatment must
represent all possible flow-shear regimes.

## Scope

The audited model is perpendicular equilibrium `E x B` decorrelation in a
local flux tube,

```text
kx*(t) = kx(0) - ky gamma_E t.
```

It does not include toroidal rotation, parallel-velocity-gradient drive,
parallel-flow shear, momentum flux, or momentum transport. Those require
separate equations, normalizations, and validation.

Frozen repository baseline:
`4104bf4a2d7463fcd56e9c38434d88510377d2b4`. The tracked promotion artifact
is `docs/_static/flow_shear_fixed_step_response_gate.json`; its GKX campaign
ran at revision `5def1e499701a3a63ddc8eaaaf4c0075e23e7043` and its independent
comparison at source revision `bc2fe552`.

## Readiness matrix

| Layer | Current implementation/evidence | Status for 3.0 |
|---|---|---|
| Coordinate trajectory | `advance_shearing_coordinates`; analytic `kx*(t)` trajectory | Closed foundation |
| Integer remap | nearest radial Fourier cell, inverse-remap and retained-norm checks | Closed foundation |
| Fractional remap | residual radial phase retained through split nonlinear transforms | Closed foundation |
| Dealias/Hermitian policy | modes leaving the two-thirds band are zeroed; physical symmetry projected after remap/stages | Closed foundation |
| Linear cache | `update_linear_cache_for_sheared_kx` rebuilds `kperp`, drifts, gyroaverages, field solve, bracket multipliers, and hyperdiffusion | Closed foundation |
| Boundary topology | periodic and standard linked tubes gated; non-twist tubes fail closed | Bounded research scope |
| Explicit integration | stage-consistent midpoint RK2 and Heun RK3; zero-shear identity and designed order | Closed research API |
| IMEX integration | fixed-step first-order endpoint operator; zero-shear identity, order, diagnostics, JVP/VJP | Closed research API |
| Adaptive policy | physical time, CFL, chunk continuation, and tangents gated internally | Research only; external stage-time policy differs |
| Nonlinear bracket | full-complex and compressed-real paths agree in the scoped gates | Closed research foundation |
| Transport diagnostics | JAX-native heat-flux trace and matched-window statistics | Closed measurement path |
| Full-resolution response | predeclared fixed-step `64x64x24`, `Nl=4`, `Nm=8`, `t=[240,300]` gate | Failed promotion gate |
| Input/CLI surface | no TOML key or executable workflow | Correctly withheld |

The implementation is real and extensively gated. The missing item is not a
coordinate or AD primitive; it is trustworthy model-level nonlinear response
evidence.

## Promotion evidence

The prospective full-resolution case used a periodic Cyclone ITG domain,
adiabatic electrons, x64 precision, `gamma_E=0` versus `gamma_E=0.01`,
`64x64x24`, `Nl=4`, `Nm=8`, fixed `dt=0.02`, final time 300, and the
independently selected late window `t=[240,300]`.

Predeclared gates required:

- each baseline/treatment window to pass finite-sample, running-drift,
  terminal-mean, block-count, and SEM checks;
- at least 5 percent relative heat-flux reduction;
- at least two combined SEM of uncertainty separation.

### GKX fixed-step IMEX

Both RTX A4000 runs remained finite, but both late windows failed stationarity:

| Quantity | Baseline | `gamma_E=0.01` treatment |
|---|---:|---:|
| Late mean | 15.45084977 | 16.19484182 |
| SEM | 0.26283183 | 0.16017286 |
| Running-mean relative drift | 0.21338996 | 0.09788862 |
| Terminal-mean relative delta | 0.14632727 | 0.11363462 |
| Runtime (minutes) | 43.053191 | 43.027167 |

The provisional relative reduction is -4.815 percent (an increase), with
-2.417 combined-SEM separation. Because both windows fail, this is not promoted
as a stationary transport comparison.

### Independent fixed-step RK4 comparison

Both comparison windows pass every independent window gate:

| Quantity | Baseline | `gamma_E=0.01` treatment |
|---|---:|---:|
| Late mean | 11.71543849 | 14.62361010 |
| SEM | 0.21567202 | 0.14072891 |
| Running-mean relative drift | 0.13399613 | 0.05175159 |
| Terminal-mean relative delta | 0.07456769 | 0.02317517 |
| Runtime (minutes) | 10.839066 | 10.699767 |

The relative reduction is -24.823 percent, resolved at -11.293 combined SEM.
This stationary result fails both the direction and magnitude of the
predeclared promotion gate.

An earlier adaptive GKX campaign showed a 6.10 percent reduction, while its
external adaptive comparison showed no resolved reduction. The source audit
found different shearing-basis timing within adaptive RK stages. The final
fixed-step audit was designed to remove that policy ambiguity and did not
recover the suppression. Short startup or strong-shear pilots cannot override
the prospective full-resolution result.

## Decision

Equilibrium ExB shear does **not** enter the GKX 3.0 stable capability matrix.
It remains accessible from the explicit research functions
`integrate_nonlinear_sheared` and `integrate_nonlinear_sheared_transport` and
their lower-level coordinate/cache owners. No configuration key, standard CLI
workflow, or broad physical-response claim is added.

This is a model-promotion decision, not a finding that the coordinate/remap or
integrator implementation is defective. Existing invariant, convergence, and
derivative tests remain valuable and must stay green during consolidation.

## GKX 3.1 reopening gate

Reconsider promotion only in a dedicated physics PR sequence that:

1. states the equilibrium-flow convention and normalization in a user-facing
   case schema without adding rotation or parallel-flow physics by implication;
2. preserves the coordinate, remap, topology, cache, convergence, and AD gates;
3. prospectively selects at least one regime with an independently supported
   response expectation and freezes resolution/timestep/window policies first;
4. obtains stationary baseline and treatment windows with uncertainty-aware
   transport statistics on CPU/NVIDIA-supported code paths;
5. performs a model-identical local external comparison or explains and gates
   every remaining discretization difference;
6. documents regimes where transport increases, decreases, or is unresolved,
   rather than making universal suppression wording;
7. demonstrates that the user workflow, restart, memory, and runtime costs are
   acceptable for a stable feature.

Retuning the old 5 percent/two-SEM thresholds after seeing the failed result is
not a reopening path.

## Reproduction

The compact decision artifact is self-contained and raw states remain local:

```bash
python -m pytest -q \
  tests/validation/nonlinear/test_nonlinear_window_artifact_contracts.py \
  -k flow_shear
python -m pytest -q \
  tests/unit/nonlinear/test_nonlinear_exb.py \
  tests/unit/nonlinear/test_nonlinear_helpers_extra.py \
  tests/unit/linear/test_linear_helpers_extra.py
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path(
    "docs/_static/flow_shear_fixed_step_response_gate.json"
).read_text())
assert p["passed"] is False
assert p["conclusion"]["promotion_gate_passed"] is False
assert p["conclusion"]["input_file_exposure_allowed"] is False
print(p["gkx_gk"]["matched"]["statistics"])
print(p["comparison"]["matched"]["statistics"])
PY
```

The raw comparison binary and input digests, campaign digest, device,
precision, integrators, and runtimes remain recorded in the JSON artifact.
