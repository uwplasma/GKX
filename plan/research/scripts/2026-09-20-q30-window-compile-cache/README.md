# Q30, the adjoint heat-flux window recompiled on every call

`nonlinear_heat_flux_window` is the route a transport optimization takes, and
on `main` it recompiled **13 XLA modules on every call** — four scans and nine
conds — so every objective evaluation paid full compile time, 60 to 74 per cent
of its wall.

**The cause is not the linked cache, a per-call closure, or a non-hashable
static argument.** `jax._src.core.ClosedJaxpr` inherits `object.__eq__` and
`object.__hash__`, so a jaxpr compares by *identity*, and an eager `lax.scan`
or `lax.cond` is dispatched on the jaxpr its body was just traced into. A jaxpr
rebuilt per call is a new key and every lowering cache under it misses.
`jaxpr_identity.py` shows it in four lines with no GKX in them.

The fix gives the differentiated scan one module-level `jax.jit` whose arrays
are all **arguments**, so the compile is reused across calls *and* across the
geometries an optimizer proposes. The host-side work — the linked cache build,
the quadrature weights, the chain-cover projection, the projector's `ky` axis
layout — stays outside the graph.

## Scripts

| file | what it does |
|---|---|
| `jaxpr_identity.py` | the minimal reproduction: three identical eager scans, three compiles |
| `bench_q30.py` | compiles per call and wall per call for one tree |
| `run_timing.sh` | the A/B above, both `ky` layouts, arms interleaved, two repetitions |
| `gate_bitwise.py` | the window's value and both adjoint components as raw bytes, for one tree |
| `run_gate.sh` | the 13-case × 2-precision identity matrix across two trees |
| `scan_body_probe.py` | equation and primitive counts of every scan body the window stages |

Each takes a GKX checkout on `PYTHONPATH` and runs one tree per process, so no
two trees ever share a `jax` import. Pass the tree paths to the two `run_*.sh`
wrappers:

```bash
PY=/path/to/venv/bin/python ./run_timing.sh <main-tree> <branch-tree> out
PY=/path/to/venv/bin/python ./run_gate.sh   <main-tree> <branch-tree> out
```

## Results

`out/timing.txt` and `out/time_*.json` — Cyclone at Nx=Ny=16, Nz=12, Nl=2,
Nm=4, a six-step RK3 window, `checkpoint=True`, `jax.value_and_grad` wrt
`tprim`, float32, jax 0.10.2 on XLA:CPU:

| `ky` layout | tree | first call | compiles | steady call | compiles |
|---|---|---:|---:|---:|---:|
| two-sided | `main` | 11.03, 11.13 s | 405 | 4.92–5.36 s | **13** |
| two-sided | branch | 7.31, 7.34 s | 123 | **0.139–0.141 s** | **0** |
| half | `main` | 12.31, 12.43 s | 365 | 7.04–7.36 s | **13** |
| half | branch | 6.78, 6.82 s | 105 | **0.112–0.116 s** | **0** |

37× and 63× per call. Host load was 3.45 before and 4.41 after, so the wall
times are not idle-host numbers; the compile counts are exact regardless.

`out/gate_matrix.txt` — bitwise in 20 of 26 configurations, every one using the
shipped default `checkpoint=True` except RK2 under x64. The six that move are 1
to 11 float32 ulps and 1 to 3 double ulps, against the window's own measured
float32 roundoff of 48 ulps (queue row Q14). `plan/log.md` has the case table
and why the residual cannot be removed from a graph that is reusable across
geometries.
