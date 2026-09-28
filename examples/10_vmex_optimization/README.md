# QA nonlinear transport

## Maintained turbulence-optimization examples (VMEX)

The runnable QA turbulence optimizations now live in VMEX, next to
`QA_optimization.py`, and each adds one GKX objective tuple to its list,
evaluated on flux tubes chosen by physical radius `s` and field-line label
`alpha` (mean or softmax over tubes):

| VMEX script | GKX objective | Derivative route |
| --- | --- | --- |
| [`QA_optimization_turbulence_linear.py`](https://github.com/uwplasma/vmex/blob/main/examples/optimization/QA_optimization_turbulence_linear.py) | dominant linear growth rate | forward-mode implicit Jacobian, least squares |
| [`QA_optimization_turbulence_quasilinear.py`](https://github.com/uwplasma/vmex/blob/main/examples/optimization/QA_optimization_turbulence_quasilinear.py) | mixing-length quasilinear heat flux | forward-mode implicit Jacobian (eigenvector derivatives), least squares |
| [`QA_optimization_turbulence_nonlinear.py`](https://github.com/uwplasma/vmex/blob/main/examples/optimization/QA_optimization_turbulence_nonlinear.py) | post-saturation window heat flux, saturation-gated | one reverse equilibrium adjoint per gradient, L-BFGS-B |

```bash
pip install "vmex[turbulence]"   # gkx>=2.4.2
```

VMEX assembles least-squares Jacobians in forward mode. Every dense GKX
eigen-objective (`solver_growth_rate_from_geometry`, the
`solver_objective_vector_from_geometry` entries, the nonlinear-window proxy
built on them) differentiates in both modes through `dominant_eigenpair`: one
dense `eig` and one bordered LU per Jacobian, plus one operator JVP at the
eigenvector per column. `eigensolver="sparse-direct"` is forward-capable too;
`eigensolver="adaptive-propagator"` is reverse-only (use VMEX's scalar
`from_loss` route with it). The nonlinear window is differentiable in both
modes; the nonlinear script takes the scalar route because one reverse sweep
through the window is cheaper than one forward tangent per boundary
coefficient.
[run.py](run.py) below stays as the pinned record of the earlier
single-tube campaign.

## The pinned campaign script

[run.py](run.py) (formerly `QA_optimization.py`) follows VMEX's boundary-mode ladder and
adds physical GKX heat flux to its objective tuples:

```python
objective_function_terms = [
    (qs, 0.0, QA_PRIORITY),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY),
    (opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY),
    (turbulent_transport, 0.0, transport_weight),
]
```

The equilibrium is vacuum; `A_OVER_LT=3` and `A_OVER_LN=1` supply finite
gyrokinetic drive. The script uses exact discrete differentiation of a physical
post-saturation heat-flux window,
not a state norm. VMEX's equilibrium response and GKX's finite-window
response are different parts of the derivative.
SciPy's `least_squares` consumes the residuals and their analytic Jacobian.

```bash
pip install vmex
python examples/10_vmex_optimization/run.py
```

This is an expensive research calculation. `VMEX_EXAMPLES_CI=1` selects a
tiny wiring smoke test, not saturation. Controls: `SATURATION_STEPS`,
`WINDOW_STEPS`, `DT`, `NX,NY,NZ,NL,NM`.

## Warm restart and derivatives

The existing `SaturationWarmStart` policy is wired but **disabled**
(`max_reuse=0`). It can reuse distribution amplitude between accepted stages;
it does not change refresh points during a local objective evaluation.
Its 5% geometry threshold and quarter-spin-up budget are heuristics.
A warm-start speedup at the same accuracy has **not** been established.

The nonlinear objective detaches the initial distribution and differentiates a
fixed RK window. It is not the exact derivative of long-time mean transport.
Keep seed/state and numerical policy fixed during each optimizer stage;
validate accepted candidates with independent cold runs.

## Results and next qualification

The retained QA candidate's nominal reduction is 12.26%, but **not statistically
resolved**: 4 of 48 nominal traces fail the final-drift test.
The conditional interval and overlapping refinement intervals alone do not
establish convergence.

[Stellarator optimization](../../docs/stellarator_optimization.rst) contains
initial/final inputs, shapes, Boozer plots, heat-flux traces, campaign scripts
and all results. [The active plan](../../plan.md) specifies warm/cold tests,
linear and quasilinear companion examples, and the validation gates.
