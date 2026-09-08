# QA nonlinear transport

[QA_optimization.py](QA_optimization.py) follows VMEX's boundary-mode ladder and
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
python examples/optimization/QA_optimization.py
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
