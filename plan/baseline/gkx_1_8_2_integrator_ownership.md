# GKX 1.8.2 time-integrator ownership audit

This inventory freezes the integration routes at GKX
`4104bf4a2d7463fcd56e9c38434d88510377d2b4`. It describes current code, not
the intended GKX 3 API. “Implemented” below does not mean scientifically
validated. Phase 2 may remove Diffrax only after every applicable migration
gate in this document is executable and green.

## Runtime selection

`TimeConfig` exposes two independent method names:

- `use_diffrax=True` selects the Diffrax family; this is the dataclass default.
- `diffrax_solver="Dopri8"` selects the Diffrax solver; it is ignored by native
  routes.
- `use_diffrax=False` selects the native family.
- `method="rk2"` selects the native method; it is ignored by Diffrax routes.

`integrate_linear_from_config` and `integrate_nonlinear_from_config` in
`src/gkx/solvers_time_runners.py` own that branch. The checked-in runtime TOML
decks overwhelmingly set `use_diffrax=false`; the linear TEM and kinetic-
electron decks are the two checked-in runtime decks that deliberately select
Diffrax with Tsit5. Python benchmarks may select Diffrax unless their
`--no-diffrax` option is supplied.

The selector is therefore not yet ready for dependency deletion: production
examples favor native integration, but a default-constructed `TimeConfig`
still selects Diffrax and two scientific examples rely on it explicitly.

## Call graph and owners

| Scope | Selector/caller | Current owner | Public seam | Result contract |
|---|---|---|---|---|
| Linear native | `integrate_linear_from_config`, `use_diffrax=False` | `solvers/linear/integrators.py` | `integrate_linear` | sampled field/history tuple, with native parallel and custom-collision seams |
| Linear explicit diagnostics | workflow linear route | `solvers/time/explicit.py`, `explicit_diagnostics.py`, `explicit_steps.py` | `integrate_linear_explicit`, `integrate_linear_explicit_diagnostics` | time, selected observable/growth diagnostics, optional state |
| Linear Diffrax | `integrate_linear_from_config`, `use_diffrax=True` | `solvers/time/diffrax_core.py`, `diffrax_linear.py` | `integrate_linear_diffrax` | optional sampled state and selected field/density observable |
| Linear streaming Diffrax | direct call only | `solvers/time/diffrax_streaming.py` | `integrate_linear_diffrax_streaming` | optional final state plus fitted growth rate and frequency |
| Nonlinear native explicit | `integrate_nonlinear_from_config`, native non-IMEX method | `solvers/nonlinear/explicit.py`, `state_integration.py`, `diagnostic_integration.py` | `integrate_nonlinear`, `integrate_nonlinear_cached`, `integrate_nonlinear_explicit_diagnostics` | final state/fields or time, diagnostics, final state, fields |
| Nonlinear native IMEX | native method `imex` | `solvers/nonlinear/imex.py`, `imex_diagnostics.py`, facade modules above | same nonlinear public seams | final state/fields or time, diagnostics, final state, fields |
| Nonlinear Diffrax | `integrate_nonlinear_from_config`, `use_diffrax=True` | `solvers/time/diffrax_core.py`, `diffrax_nonlinear.py` | `integrate_nonlinear_diffrax` | final state and fields |
| Nonlinear sharded native | non-null supported `state_sharding`, `use_diffrax=False` | `parallel/integrators.py` | `integrate_nonlinear_sharded` | final state and optional fields |

`integrate_linear_diffrax_streaming` has no production caller outside its own
tests and documentation at this revision. It remains a top-level export and a
specialized fitted-observable implementation, so “unused by runners” is not a
safe deletion argument.

## Native explicit capability matrix

| Capability | Linear native | Nonlinear native | Evidence/status |
|---|---:|---:|---|
| Fixed-step Euler | yes | yes | implemented and unit tested |
| Fixed-step RK2 | yes | yes | implemented; current native configuration default |
| Fixed-step RK3/Heun | no in `linear/integrators.py` | yes | nonlinear accepts `rk3`/`rk3_heun` |
| Fixed-step classic RK3 | no in `linear/integrators.py` | yes | nonlinear accepts `rk3_classic` |
| Fixed-step RK4 | yes | yes | implemented and unit tested |
| SSPX3 | yes | yes | implemented; supported by nonlinear sharded path |
| K10 | low-level explicit stage only | low-level serial path | not accepted by the sharded validator; parity must not be inferred |
| Adaptive CFL | separate explicit workflow | diagnostic/sheared runtime routes | implemented with `dt_min`, `dt_max`, `cfl`, and method-specific `cfl_fac`; not tolerance controlled |
| JAX scan/JIT | yes | yes | implemented |
| Discrete-adjoint checkpointing | checkpoint flag | blocked two-level scan checkpointing | implemented; nonlinear retention is documented as O(sqrt(N)) states |
| Custom collision operator | yes | explicit only | runners reject unsupported Diffrax, sharded, and nonlinear IMEX combinations |
| Runtime diagnostics/sample stride | yes | yes | native diagnostic owners retain production diagnostic payloads |
| Prepared/cached execution | yes | yes | production profiling and objective seams use these paths |
| State sharding/parallel RHS | yes | yes | native-only runner lanes; supported axes remain separately gated |
| Sheared coordinates | n/a | yes | fixed/adaptive explicit plus fixed IMEX; adaptive IMEX is rejected |

The native linear `imex` and `imex2` labels treat collision and hypercollision
damping implicitly while evaluating the remaining RHS explicitly. They are not
equivalent to the nonlinear native IMEX owner.

## Native nonlinear IMEX boundary

The nonlinear IMEX implementation builds a matrix-free full linear operator,
advances the nonlinear bracket explicitly, and solves the implicit stage with
SOLVAX. It owns fixed-step state-only and diagnostic scans, implicit restart
and preconditioner controls, JAX checkpointing, field reconstruction, and the
runtime diagnostic seams.

It deliberately rejects adaptive sheared IMEX and custom collision operators.
Collision splitting in the diagnostic route is a separate configured seam and
must not be confused with arbitrary `CollisionOperator` support. The owner is
therefore the target native stiff route, but its scientific stability and
accuracy envelope must be established case by case before it replaces any
Diffrax KenCarp workflow.

## Diffrax capability matrix

The four Diffrax modules contain 1,636 physical lines at this revision:
`diffrax_core.py` 188, `diffrax_linear.py` 449, `diffrax_nonlinear.py` 427, and
`diffrax_streaming.py` 572.

| Capability | Linear | Nonlinear | Streaming | Migration consequence |
|---|---:|---:|---:|---|
| Euler/Heun | yes | yes | yes | native value/order tests required |
| Tsit5/Dopri5/Dopri8 | yes | yes | yes | no like-for-like native adaptive RK owner yet |
| ImplicitEuler/Kvaerno3/4/5 | yes | yes | yes | native stiff comparison required; nonlinear native owner is IMEX, not DIRK |
| KenCarp3/4/5 | yes | yes | yes | compare against native nonlinear IMEX over stiff production cases |
| Constant step | yes | yes | yes | exact saved-time and endpoint gates required |
| PID `rtol`/`atol` adaptation | yes | yes | yes | must be replaced or explicitly retired with affected examples migrated |
| Direct adjoint | yes | yes | yes | native gradient parity required |
| Recursive checkpoint adjoint | yes | yes | yes | peak-memory and gradient parity required |
| Forward-mode JVP policy | yes | not exposed | no explicit public mode | linear JVP finite-difference test must move to native owner |
| Packed complex state | yes | yes | yes | native paths use complex state directly; value/dtype parity required |
| Selected phi/density/mode saves | yes | fields only | fitted phi/density | preserve observable shapes and mode-selection semantics |
| Single-device sharding seam | yes | yes | no | native replacement must cover the supported device layouts |
| Progress meter | yes | yes | yes | behavior may be simplified, but commands must remain usable |
| Custom collision operator | rejected by runner | rejected by runner | no | native capability is broader; no parity blocker |
| Specialized fitted growth/frequency | no | no | yes | migrate to canonical post-processing or retain a small native facade |

The solver-name aliases are part of the current behavior: `rk4` maps to Tsit5,
`rk2` to Heun, `implicit` to Kvaerno5, and `imex`/`semi-implicit` to KenCarp4.
They are compatibility debt, not evidence that the native tableaus match.

## Existing test ownership

The focused source suites at the frozen revision contain:

| Suite | Test functions | What it presently establishes |
|---|---:|---|
| `test_diffrax_integrators_core.py` | 14 | helper/error branches, packing, linear modes/fields, adaptive linear JVP finite differences, streaming, nonlinear explicit/IMEX, JIT smoke |
| `test_explicit_time_integrators_lowlevel.py` | 16 | native linear tableaus, adaptive-CFL controls, progress, full loop, error paths |
| `test_nonlinear_explicit_scan.py` | 11 | explicit scan/checkpoint/diagnostic mechanics |
| `test_nonlinear_explicit_step.py` | 10 | explicit stage values and method dispatch |
| `test_nonlinear_imex.py` | 12 | implicit operator, SOLVAX handoff, scan and diagnostic behavior |
| `test_runners.py` | 8 | configuration routing, parallel/sharding and collision rejection/forwarding |

These unit suites exercise mechanics. They do not by themselves establish
production-order accuracy, long-time nonlinear transport identity, stiff
stability, restart identity, CPU/GPU identity, or performance acceptance.

## Required Phase 2 migration gates

All comparisons must freeze precision, grid, state seed, terms, geometry,
device, sample times, and output normalization. A gate is applicable to every
route whose behavior is being migrated, not merely one smoke case.

| Gate | Required comparison | Acceptance evidence |
|---|---|---|
| VALUE | one-step and fixed-horizon states/fields for each retained native tableau | dtype-aware absolute/relative tolerances, with per-observable norms and no NaN/Inf |
| ORDER | manufactured linear and nonlinear problems over at least three `dt` values | fitted observed order consistent with the declared tableau before asymptotic saturation |
| STABILITY | collisional/streaming stiff linear cases and nonlinear IMEX production cases | stable horizon and bounded invariant/energy error over a documented timestep envelope |
| RESTART | uninterrupted versus state/artifact restart at an interior sample | identical time grid and within-tolerance final state, fields, diagnostics, and adaptive-state metadata |
| DIAGNOSTIC | phi/density/mode saves, sample/diagnostic stride, growth/frequency fits, heat/energy traces | exact shapes/times/labels and within-tolerance numerical values |
| AD | forward JVP, reverse gradient, checkpointed reverse gradient on linear and nonlinear objectives | centered-FD agreement, step/block agreement, no gradient NaN/Inf, bounded peak memory |
| DEVICE | matched Apple/CPU and NVIDIA runs | same configuration fingerprint and dtype-aware trajectory/objective/gradient identity |
| PERF | cold prepare/compile/first, five warm repeats, host RSS and device peak/live bytes | no unexplained regression; publish medians and raw profiles for value and gradient lanes |
| WORKFLOW | TEM and kinetic-electron examples, benchmark flags, public direct calls | commands complete with preserved outputs or an explicit documented retirement |
| API | top-level Diffrax seams and configuration keys | deprecation window and migration target before removal from the GKX 3 surface |

The nonlinear objective gate must include linear, quasilinear, and nonlinear
VMEX consumers. Value-only solver parity is insufficient if compile latency,
warm runtime, checkpoint memory, or gradient stability regresses.

## Migration sequence and deletion candidates

1. Add native parity tests for the two Diffrax-selected linear example decks
   and the supported direct linear save modes.
2. Move streaming growth/frequency extraction onto the canonical diagnostic
   post-processing path, or implement a small native fitted-observable facade.
3. Establish native explicit and native IMEX value/order/stability/restart/AD
   gates on CPU and NVIDIA, including optimization objectives.
4. Change `TimeConfig` and generated input defaults only after those gates pass;
   retain a deprecation period for `use_diffrax`, `diffrax_solver`, and the
   adaptive-tolerance keys.
5. Remove the Diffrax routes and then remove `diffrax` and `equinox` from core
   dependencies in the same release tranche whose lock/build evidence proves
   the base install.

After the gates close, the named deletion candidates are:

- `src/gkx/solvers_time/diffrax_core.py`
- `src/gkx/solvers_time/diffrax_linear.py`
- `src/gkx/solvers_time/diffrax_nonlinear.py`
- `src/gkx/solvers_time/diffrax_streaming.py`
- their exports in `src/gkx/__init__.py` and `src/gkx/solvers_time.py`
- their branches/imports in `src/gkx/solvers_time_runners.py`
- Diffrax-only tests after their scientific assertions move to native tests
- `diffrax` and `equinox` in `pyproject.toml`
- deprecated `TimeConfig` Diffrax fields after the compatibility window
- Diffrax-specific benchmark flags and documentation after replacements exist

No deletion is authorized by this audit alone.

## Reproduction

Run the focused owner tests:

```console
python -m pytest -q \
  tests/unit/solvers/test_diffrax_integrators_core.py \
  tests/unit/solvers/test_explicit_time_integrators_lowlevel.py \
  tests/unit/solvers/test_nonlinear_explicit_scan.py \
  tests/unit/solvers/test_nonlinear_explicit_step.py \
  tests/unit/solvers/test_nonlinear_imex.py \
  tests/unit/solvers/test_runners.py
```

Reproduce route declarations and direct streaming callers with:

```console
rg -n 'use_diffrax|diffrax_solver|diffrax_(adaptive|rtol|atol|max_steps)' \
  src/gkx examples benchmarks docs tests
rg -n 'integrate_linear_diffrax_streaming\\(' \
  --glob '*.py' --glob '!src/gkx/solvers_time/diffrax_streaming.py'
wc -l src/gkx/solvers_time/diffrax_{core,linear,nonlinear,streaming}.py
```
