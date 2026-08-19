# QA nonlinear transport

[`QA_optimization.py`](QA_optimization.py) follows VMEX's QA boundary-mode
ladder and appends one physical GKX heat-flux tuple:

```python
objective_function_terms = [
    (qs, 0.0, QA_PRIORITY),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY),
    (opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY),
    (turbulent_transport, 0.0, transport_weight),
]
```

The equilibrium is vacuum. `A_OVER_LT=3` and `A_OVER_LN=1` provide finite ITG
transport. The script carries the VMEX parallel scale into GKX, runs to
saturation, differentiates an actual post-saturation heat-flux window with
exact discrete differentiation, and refreshes the state after each VMEX stage.

```bash
python examples/optimization/QA_optimization.py
```

The default run is a GPU calculation. Use `VMEX_EXAMPLES_CI=1` for a short
end-to-end smoke test. The main accuracy/cost controls are
`SATURATION_STEPS`, `WINDOW_STEPS`, `NX`, `NY`, `NZ`, `NL`, and `NM`.

The spin-up between stages can be warm-started by
`gkx.workflows.runtime.warm_start.SaturationWarmStart`, which is wired into
`saturate()` but **off by default** (`max_reuse = 0`). Switching it on does not
move the refresh points: the saturated state is still detached and still
replaced once per accepted VMEX stage, so the objective stays a fixed function
of `(state, runtime)` for the whole of each stage and never becomes a function
of the optimizer's within-stage history. Only the cost of the refresh changes.
A stage that moved the flux tube by less than
`geometry_tolerance = 0.05` (relative L2 over the metric profiles the nonlinear
operator reads) reseeds from the previous saturated state and runs
`warm_step_fraction = 0.25` of `SATURATION_STEPS`, because a saturated seed
does not have to climb out of a 1e-3 perturbation again. The full cold spin-up
comes back the moment the geometry moves further than that, or after
`max_reuse` consecutive warm spin-ups, so a chain of small accepted steps
cannot drift away from the attractor unchecked.

It is off by default because the saving is real but its cost has not been
measured on this objective: a shortened spin-up still has to re-equilibrate to
the new geometry, and whether a quarter budget suffices is a question about
this objective's sensitivity that only a full optimization run answers. Raise
`max_reuse` to opt in, and compare against a `max_reuse = 0` run when you do.

The analytic Jacobian includes both the implicit VMEX equilibrium response and
the exact GKX window derivative; SciPy's `least_squares` consumes it directly.
Independent, replicated post-transient runs validate the accepted direction:
24 nominal pairs reduce transport by 12.26% (95% CI 10.64--13.88%), and the
20x20 and stationary 24x24 refinement intervals overlap above zero. The full
protocol and CSV data are in the [stellarator optimization
documentation](../../docs/stellarator_optimization.rst).

The accepted initial/final VMEX inputs, restartable ensemble driver, statistical
estimator, reproduction commands, and figures are linked from that page.
