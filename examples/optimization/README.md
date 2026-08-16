# QA nonlinear transport

[`QA_optimization.py`](QA_optimization.py) follows VMEX's QA boundary-mode
ladder and appends one physical GKX heat-flux tuple:

```python
objective_function_terms = [
    (qs, 0.0, 1.0),
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (opt.mean_iota, IOTA_TARGET, 10.0),
    (turbulent_transport, 0.0, transport_weight),
]
```

The equilibrium is vacuum. `A_OVER_LT=2.49` and `A_OVER_LN=0.8` provide a
finite ITG drive in GKX. The script runs to saturation, differentiates an
actual post-saturation heat-flux window with exact discrete differentiation,
and refreshes the saturated state after each accepted VMEX stage.

```bash
python examples/optimization/QA_optimization.py
```

The default run is a GPU calculation. Use `VMEX_EXAMPLES_CI=1` for a short
end-to-end smoke test. The main accuracy/cost controls are
`SATURATION_STEPS`, `WINDOW_STEPS`, `NX`, `NY`, `NZ`, `NL`, and `NM`.

The analytic Jacobian includes both the implicit VMEX equilibrium response and
the exact GKX window derivative; SciPy's `least_squares` consumes it directly.
Final transport claims still require independent, replicated post-transient
baseline/candidate runs.
