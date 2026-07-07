# SPECTRAX-GK

[![Release](https://img.shields.io/github/v/release/uwplasma/SPECTRAX-GK?display_name=tag)](https://github.com/uwplasma/SPECTRAX-GK/releases)
[![PyPI](https://img.shields.io/pypi/v/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![CI](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/uwplasma/SPECTRAX-GK/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://github.com/uwplasma/SPECTRAX-GK/blob/main/pyproject.toml)
[![Coverage](https://codecov.io/gh/uwplasma/SPECTRAX-GK/graph/badge.svg)](https://codecov.io/gh/uwplasma/SPECTRAX-GK)

SPECTRAX-GK is a JAX-native gyrokinetic solver designed for differentiability,
accelerator-ready execution, and stellarator-optimization research workflows.
The code employs a Hermite-Laguerre velocity space, Fourier perpendicular 
coordinates, and field-aligned flux-tube geometry to simulate linear and 
nonlinear electrostatic and electromagnetic turbulence in magnetized plasmas.
The validated release claim is narrower than the full feature surface; use the
claim-scope ledger below before citing benchmark, quasilinear, autodiff,
refactor, or manuscript results.

## Installation

```bash
pip install spectraxgk
```

or install the development checkout directly:

```bash
git clone https://github.com/uwplasma/SPECTRAX-GK
cd SPECTRAX-GK
pip install -e .
```

## Quickstart (Executable)

```bash
# Run the built-in default example.
spectraxgk

# The hyphenated entry point works too.
spectrax-gk

# Run directly from a checked-in TOML.
spectraxgk examples/linear/axisymmetric/cyclone.toml

# Compute linear quasilinear transport weights and write JSON/CSV outputs.
spectraxgk run-runtime-linear \
  --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml \
  --out tools_out/cyclone_quasilinear

# Write a restartable nonlinear NetCDF bundle.
spectraxgk run-runtime-nonlinear \
  --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
  --steps 200 \
  --out tools_out/cyclone_release.out.nc

# Generate small VMEC-JAX equilibria locally, then run prefilled VMEC TOMLs.
pip install vmec-jax
cd examples/vmec
./generate_wouts.sh
cd ../..

spectraxgk run \
  --config examples/linear/axisymmetric/runtime_circular_vmec_linear.toml \
  --out tools_out/circular_vmec_linear

spectraxgk run \
  --config examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml \
  --out tools_out/qhs_nonlinear_run

# Turn supported saved outputs into review figures.
spectraxgk --plot tools_out/cyclone_release.out.nc
spectraxgk --plot spectraxgk_default_linear.summary.json
```

Running `spectraxgk` with no TOML starts a short Cyclone initial-value linear
demo (equivalent to the standard `examples/linear/axisymmetric/cyclone.toml`
surface), prints setup and live time-integration progress with elapsed time and
ETA, and writes the demo outputs in the current directory:

- `spectraxgk_default_linear.toml`: the input file that reproduces the run
- `spectraxgk_default_linear.summary.json`
- `spectraxgk_default_linear.timeseries.csv`
- `spectraxgk_default_linear.eigenfunction.csv`
- `spectraxgk_default_linear.png`

The figure shows the linear `|\phi|^2` time history on a log scale with the
fitted `(\gamma, \omega)` annotation, plus the normalized real and imaginary
eigenfunction. Re-run the same numerical case with
`spectraxgk run-linear --config spectraxgk_default_linear.toml --progress`.

Longer runtime commands also print live status lines with step/time progress,
wall elapsed time, and an estimated wall-clock time remaining when progress is
enabled. Adaptive nonlinear runs emit chunk-level elapsed/ETA updates.

The `--plot` mode reads saved outputs directly:

- linear bundles: `*.summary.json` + `*.timeseries.csv` + `*.eigenfunction.csv`
- nonlinear bundles: `*.summary.json` + `*.diagnostics.csv` or `*.out.nc`

Linear plots reproduce the two-panel growth/eigenfunction layout. Nonlinear
plots produce a three-panel diagnostic view with field amplitude/energy,
resolved diagnostics, and heat flux.

## Highlights

- **Differentiable JAX-native kernels** for gradient-based optimization and sensitivity analysis.
- **Hermite-Laguerre spectral velocity basis** providing efficient kinetic closures and multi-fidelity modeling.
- **Accelerator-ready execution** on CPUs and GPUs with JIT compilation.
- **Flexible geometry interface** supporting analytic s-alpha, Miller, and direct VMEC equilibrium imports.
- **Electromagnetic field-channel support** including $(\phi, A_\parallel, B_\parallel)$ fluctuations, with validation claims limited to tracked release lanes.
- **Multi-species support** with kinetic electrons and advanced collision operators.
- **Quasilinear transport diagnostics** from linear states, with explicit
  saturation-rule metadata and electrostatic channel validation gates.
- **Automated benchmark workflows** for reproducible validation and regression tracking.
- **Modular runtime/refactor surfaces** with focused tests for restart artifacts,
  diagnostics, validation gates, nonlinear RHS routing, nonlinear diagnostic
  state assembly, explicit/IMEX nonlinear stepping, solver-objective gradient
  gates, VMEC/Boozer sensitivity gates, and public API boundaries.

## Runtime and Memory

![Runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

SPECTRAX-GK is optimized for performance across CPU and GPU backends. The
runtime panel above compares cold wall time and peak memory usage for the
shipped benchmark cases, including startup/compilation for JAX rows. Treat these
as release-accounting measurements for the listed workloads, not as a universal
throughput model for longer warm runs.

Performance tracking covers:

- **Cyclone ITG** (linear/nonlinear)
- **KBM** and **ETG** configurations
- **W7-X** and **HSX** stellarator geometries
- **Miller** geometry models

The refreshed shipped panel includes the W7-X and HSX linear and nonlinear
rows. It is generated only from reviewed measured artifacts:

- `docs/_static/runtime_memory_summary_ship_refresh.json`
- `docs/_static/runtime_memory_results_ship_refresh.csv`
- `docs/_static/runtime_memory_benchmark.png`

Regenerate the public panel from the shipped refresh summary with:

```bash
python tools/benchmark_runtime_memory.py \
  --summary-glob docs/_static/runtime_memory_summary_ship_refresh.json \
  --csv-out docs/_static/runtime_memory_results_ship_refresh.csv \
  --summary-out docs/_static/runtime_memory_summary_ship_refresh.json \
  --plot-out docs/_static/runtime_memory_benchmark.png
```

Representative shipped rows from `docs/_static/runtime_memory_results_ship_refresh.csv`:

| Case | SPECTRAX-GK CPU | SPECTRAX-GK GPU | Reference backend | Peak RSS range |
| --- | ---: | ---: | ---: | ---: |
| Cyclone ITG linear | 39.6 s | 24.2 s | 981.7 s | 1.1-2.0 GiB |
| W7-X nonlinear | 474.0 s | 50.8 s | 111.0 s | 2.0-6.3 GiB |
| HSX nonlinear | 646.5 s | 49.3 s | 135.7 s | 2.1-6.4 GiB |
| Cyclone Miller nonlinear | 339.2 s | 47.1 s | 77.8 s | 2.1-4.9 GiB |

## QA ITG Optimization Panel

SPECTRAX-GK ships VMEC-JAX-style QA optimization examples that append one differentiable ITG transport residual to the usual aspect-ratio, iota, and quasisymmetry objective tuples. The strict baseline below follows the upstream VMEC-JAX `QA_optimization.py` max-mode-5 simple-seed setup; the transport rows restart from that solved QA state and optimize one representative ITG residual for linear growth, quasilinear flux, or nonlinear-window screening. Full equations, gates, and audit provenance are in the [stellarator optimization docs](docs/stellarator_optimization.rst).

![VMEC-JAX QA max-mode-5 optimizer sweep](docs/_static/vmec_jax_qa_full_sweep_panel.png)

The baseline is an admitted QA design (`A = 5.0000`, mean `iota = 0.41020`, QS residual `8.9e-6`). The transport-optimized rows are optimizer-output comparisons, not promoted turbulent-flux designs: their solved-WOUT rows narrowly miss the strict mean-`iota >= 0.41` gate, and the matched long post-transient nonlinear `Q(t)` audits below do not show a significant heat-flux reduction. Optimizer comparisons are valid only within matched `setup_summary.json` comparison fingerprints.

The true `t=1500` office nonlinear audits now pass the late-window
seed/timestep gates for the strict QA baseline, growth-objective candidate,
quasilinear-objective candidate, and nonlinear-window candidate. Their
`t=[1100,1500]` replicated means are `<Q_i>=11.58`, `11.51`, `11.64`, and
`11.61`, respectively. Matched comparisons do not promote any transport
candidate: the growth row reduces late-window heat flux by only `0.60%`
(`z=0.26`, below the `4%` gate), while the quasilinear and nonlinear-window
rows are slightly worse than baseline (`-0.49%`, `z=-0.19`; `-0.25%`,
`z=-0.09`). The strict QA `t=1500` set is therefore closed as robust negative
optimization-transfer evidence, not as a successful nonlinear turbulent-flux
optimization. The earlier bookkeeping audit that stopped near `t=400` is kept
only as negative launch-contract evidence.

The companion `RBC(1,1)` landscape scans the strict QA baseline over `[-75%, +75%]` with 31 points. The top panel shows the linear growth rate and every shipped electrostatic quasilinear heat-flux rule over the same multi-surface, multi-field-line, multi-`k_y` sample used by the optimizer examples. The lower panel overlays 24 long-window post-transient nonlinear heat-flux ensembles (`t=[1100,1500]`, seed/timestep replicated): the negative side, the zero-offset baseline, and eight positive coefficients (`+5%`, `+10%`, `+15%`, `+20%`, `+25%`, `+30%`, `+35%`, `+40%`). The `+20%` point is admitted only under the scoped diagnostic `20%` seed/timestep-spread gate (`15.48%` spread, `<Q_i>=9.2545`); it would not satisfy the stricter `15%` production-style gate. The remaining higher positive-side coefficients are treated as stability-boundary/open long-window points, not inferred from reduced metrics. Reduced/startup nonlinear-window estimators are intentionally excluded from this plot and from optimization-promotion claims.

![QA RBC(1,1) transport landscape](docs/_static/vmec_boundary_transport_landscape_rbc11_full.png)

The optimizer-strategy report below is built from the tracked QA optimizer and
`RBC(1,1)` artifacts. It shows that the current deterministic transport rows
reduce their internal objectives but remain diagnostic-only, while the
converged landscape contains a real lower-`Q_i` direction. The landscape is a
noise/convergence diagnostic, not an admission source for optimized QA
stellarators. The next campaign should use
exact-adjoint least squares for the strict QA baseline, adjoint trust/L-BFGS
with continuation for linear/quasilinear residuals, and SPSA/CMA-ES/Bayesian
outer-loop comparators only for noisy long-window nonlinear heat-flux
objectives.

![QA optimizer strategy report](docs/_static/vmec_jax_qa_optimizer_strategy_report.png)

```bash
python examples/optimization/QA_optimization_linear_ITG.py
python examples/optimization/QA_optimization_quasilinear_ITG.py
python examples/optimization/QA_optimization_nonlinear_ITG.py
python examples/optimization/QA_nonlinear_ITG_matched_audit.py
python examples/optimization/QA_nonlinear_ITG_transport_matrix.py
python examples/optimization/QA_parameter_scan.py
```

## Self-Contained VMEC Geometry Examples

The VMEC-backed examples no longer require users to generate separate EIK
geometry files. The repository ships small `vmec_jax` input decks under
`examples/vmec/`; users generate `wout_*.nc` locally and the runtime TOMLs
already point to the expected relative paths. To build every bundled demo
equilibrium, run:

```bash
pip install vmec-jax
cd examples/vmec
./generate_wouts.sh
cd ../..
```

Then run the VMEC-backed examples from the repository root:

```bash
spectraxgk run --config examples/linear/axisymmetric/runtime_circular_vmec_linear.toml
spectraxgk run --config examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml
spectraxgk run --config examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml
```

The bundled QHS/QI/QA VMEC decks are self-contained demonstrators. Exact
machine-specific HSX or W7-X validation should use the same TOMLs with
`--vmec-file` pointing to the corresponding benchmark `wout_*.nc`.

## Current claim scope

The current release surface is deliberately scoped:

- Linear and nonlinear benchmark claims are tied to tracked gates and figures
  under `docs/_static`.
- The large runtime/diagnostic refactor is an infrastructure claim: extracted
  runtime startup/chunk/result helpers, validation-gate helpers, and restart
  artifact schema tests preserve public behavior. It is not a new physics,
  validation, nonlinear-optimization, or speedup claim.
- Electrostatic quasilinear weights and spectra are validated diagnostics. The
  one-constant and simple saturation-rule absolute-flux models are rejected on
  the current 12-case train/holdout portfolio. After admitting the replicated
  Solovev external-VMEC holdout, the one-constant positive-growth mixing-length
  model fails more strongly (`6.49 > 0.35`). The richer
  `spectral_envelope_ridge` candidate closes the scoped core portfolio when the
  declared Solovev and shaped-pressure stress outliers are kept outside the
  claim: core mean relative error is `0.280`, core holdout error is `0.275`,
  and interval coverage is `10/10`. The full 12-case universal predictor and
  promoted rank screener remain deferred because the stress cases and full-rank
  gates do not pass. SPECTRAX-GK therefore ships QL diagnostics, a scoped core
  model-development diagnostic, and guarded optimization-screening examples,
  not a runtime/TOML universal absolute-flux predictor.
  Electromagnetic quasilinear calibration remains deferred.
- The `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` path is artifact-bound:
  zero-beta equal-arc geometry parity is claimable for the rows that pass the
  current `mboz=nboz=21` parity matrix, and reduced
  linear/quasilinear/nonlinear-window-estimator gradients are claimable only on
  the tracked QH/Li383 gates. The multi-surface/alpha/`k_y` portfolio gate is
  reduced/model-development evidence for objective plumbing, not a nonlinear
  heat-flux optimization claim. A half-mesh Boozer radial-index convention fix
  restored the fixed-resolution QI row (`drift=7.13e-2 < 8e-2`) and the
  evaluated QI robustness variants at `ntheta=8,16` now pass. The broader QI
  seed campaign remains artifact-limited because three input-only QI seeds have
  no bundled `wout` references; this is not broad QI transport validation, QI
  quasilinear calibration, or QI nonlinear optimization. The actual nonlinear
  finite-difference audits are startup plumbing checks with
  `transport_average_gate = false`; they are not production turbulence-gradient
  or nonlinear heat-flux optimization claims.
- Production parallelization is currently the independent-work path for `k_y`
  scans, quasilinear studies, and UQ ensembles. Sensitivity sweeps can use the
  same deterministic independent-work reconstruction, but they need a dedicated
  scaling artifact before making a speedup claim. Whole-state
  nonlinear sharding and nonlinear domain sharding remain diagnostic unless the
  exact workload has passing identity and profiler promotion gates.
- W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron
  extensions are deferred from the current manuscript/release scope.

The detailed claim ledger is in
[`docs/release_scope.rst`](docs/release_scope.rst).

![SPECTRAX-GK linear benchmark panel](docs/_static/benchmark_core_linear_atlas.png)

![SPECTRAX-GK nonlinear benchmark panel](docs/_static/benchmark_core_nonlinear_atlas.png)

The figures above represent the validated benchmark suite, covering linear
microinstabilities and nonlinear transport across diverse magnetic
configurations. The shipped nonlinear atlas emphasizes the longest archived
windows currently tracked in the repo: KBM to about `t=400`, W7-X to about
`t=200`, and Cyclone Miller to about `t=122`. HSX is currently archived on the
closed `t=50` window; no longer-window HSX nonlinear audit artifact is currently
tracked for the release panel.

Quasilinear transport diagnostic example:

![SPECTRAX-GK quasilinear Cyclone spectrum](docs/_static/quasilinear_cyclone_spectrum.png)

This panel is generated from `examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml`.
It shows linear growth/frequency, eigenfunction-weighted `k_perp`, amplitude-normalized
heat/particle flux weights, and an explicitly uncalibrated mixing-length output. The
absolute saturated-flux claim remains gated on nonlinear train/holdout calibration.
The first Cyclone nonlinear audit is tracked in `docs/quasilinear.rst` and is
kept at `training_or_audit_only` until a held-out calibration set passes.

The manuscript-facing quasilinear calibration panel now uses the full admitted
electrostatic portfolio: two training geometries and ten held-out nonlinear
windows spanning tokamak, stellarator, and external-VMEC cases. The newest
admitted entry is the replicated Solovev external-VMEC window, which passes
seed/timestep readiness and the explicit `20%` spread gate (`15.99%` spread).
The QI candidate is finite but remains excluded because its `n48/n64` heat-flux
means differ by about `38%` at `t=250`, above the `15%` grid/window gate.

![SPECTRAX-GK quasilinear stellarator train/holdout calibration](docs/_static/quasilinear_stellarator_train_holdout.png)

The current training set is Cyclone plus the external-VMEC ITERModel case; the
holdouts are Cyclone Miller, HSX, W7-X, D-shaped VMEC, up-down asymmetric VMEC,
circular VMEC, CTH-like VMEC, shaped-pressure VMEC, QP VMEC, and Solovev VMEC.
This is a stronger transfer test than the earlier Cyclone-only fit: nonlinear
input validation passes, but the fitted one-constant mixing-length model fails
the held-out absolute-flux gate with mean relative error about `6.49`. The
simple-rule sweep is also negative: linear-weight is the least-bad simple rule
at `4.42`, positive-growth mixing length is `6.49`, absolute-growth mixing
length is `6.85`, and even the training-mean null is `1.80`. SPECTRAX-GK does
not promote any simple or user-facing absolute QL flux predictor from this
one-constant saturation-rule family.

The richer held-out candidate below is the reduced `spectral_envelope_ridge`
model. It uses only two linear-spectrum envelope features. After adding the
Solovev replicated holdout, its leave-one-geometry-out mean relative error is
about `0.697`, with interval coverage `11/12` on the full ledger. The declared
core portfolio excludes the Solovev repair and shaped-pressure stress cases and
passes the scoped transport diagnostic: mean relative error `0.280`, held-out
mean `0.275`, maximum error `0.575`, and interval coverage `10/10`. Rank
screening remains borderline (`Spearman ~0.745`, just below the `0.75` gate),
so the current claim is a scoped core absolute-flux diagnostic for examples and
model development, not a universal runtime predictor.

![SPECTRAX-GK quasilinear candidate uncertainty gate](docs/_static/quasilinear_candidate_uncertainty.png)

The residual-anatomy panel below explains the failed gate instead of hiding it
behind one aggregate number. External axisymmetric VMEC holdouts now dominate
the full-ledger error budget. The Solovev replicated holdout and
shaped-pressure VMEC are retained as declared stress outliers, while the
remaining 10-case core portfolio passes the transport and interval-coverage
diagnostic. This closes the current QL lane for scoped examples while preserving
the honest boundary that universal stress-case absolute-flux prediction needs
richer saturation physics.

![SPECTRAX-GK quasilinear residual anatomy](docs/_static/quasilinear_error_anatomy.png)

The regularization audit below sweeps the ridge strength for the same
spectral-envelope candidate. It is a guardrail against over-tuning: the best
setting is now `lambda = 0.5`, with mean relative error `0.689 > 0.35`, so
regularization choice does not rescue absolute-flux promotion.

![SPECTRAX-GK quasilinear candidate regularization audit](docs/_static/quasilinear_candidate_regularization_sweep.png)

The stellarator-specific summary below compares the admitted HSX and W7-X
nonlinear windows with the current quasilinear candidates, while keeping QA and
QH explicitly scoped: QA has matched nonlinear-audit evidence only, and QH is
excluded until grid/window convergence passes. Simple one-constant
quasilinear rules do not transfer as absolute stellarator heat-flux predictors
on the admitted portfolio. The spectral-envelope ridge candidate is the best
current scoped model-selection result, but the frozen ledger now points to
missing saturation physics rather than additional holdout collection before
universal stellarator-flux claims.

![SPECTRAX-GK quasilinear stellarator usefulness summary](docs/_static/quasilinear_stellarator_usefulness.png)

The companion screening-skill panel separates correlation/ranking usefulness
from absolute-flux promotion. On the 12-case portfolio,
`spectral_envelope_ridge` remains the best screened model, but it does not pass
the full or held-out rank/correlation gates (`Spearman ~0.636` full, `~0.624`
held-out; pairwise order accuracy `~0.697` full, `~0.689` held-out; all below
the `0.75` gates). On the outlier-declared core portfolio, the same candidate
passes the transport/coverage diagnostic but remains just below the strict rank
gate, so absolute-flux runtime promotion remains `none`.

![SPECTRAX-GK quasilinear screening skill summary](docs/_static/quasilinear_screening_skill.png)

The companion holdout-gap report makes the remaining promotion blockers
explicit instead of hiding them in the calibration plot. Ten holdouts are
admitted, but both the strict model-selection gate and the current absolute
heat-flux calibration remain failed. The one-constant/simple rules miss the
aggregate holdout gate (`6.49 > 0.35` for positive-growth mixing length), and
the best spectral-envelope candidate still misses the uncertainty transport
gate (`0.697 > 0.35`). The independent-holdout-count blocker is closed; the
remaining blockers are transport error and screening/ranking skill, so the next
step is better saturation physics rather than threshold loosening.

![SPECTRAX-GK quasilinear holdout gap report](docs/_static/quasilinear_holdout_gap_report.png)

The shaped-tokamak-pressure external-VMEC repair is now admitted only through
the explicit high-grid policy. The full `n48/n64/n80`, `t=450`, `dt=0.04`
ladder fails because the coarse `n48` trace is not converged (`0.469` pairwise
heat-flux shift), but the retained `n64/n80` pair passes at `t=450`
(`0.0789`) and `t=650` (`0.0983` common, `0.0981` least). The high-grid
horizon gate passes (`0.0418` common, `0.1237` least), and the `n80`
seed/timestep ensemble passes on `t=[325,650]` with mean heat flux `7.16`,
mean-relative spread `0.0939`, and combined SEM/mean `0.0463`. This is a
scoped high-grid holdout under coarse-grid exclusion, not a full
`n48/n64/n80` convergence claim and not an absolute-flux promotion.

![SPECTRAX-GK external-VMEC next holdout runbook](docs/_static/external_vmec_next_holdout_runbook.png)

The external-VMEC runbook remains fail-closed after the corrected QH
warm-start audit: the staged `n64/n80`, `dt=0.04` QH ladder reached `t=250`,
`450`, and `700`, but the long-window high-grid disagreements stayed above the
relaxed 20% gate (`t700`: `0.349` common-window, `0.367` least-window). QH is
negative grid-convergence evidence, not a quasilinear calibration holdout. A
new bounded linear screen identified the independent Solovev VMEC candidate
(`gamma=0.0944` at `ky=0.2857`). Its repaired `n48/t250` seed/timestep
ensemble now passes the finite-output, readiness, and explicit `20%` spread
gates with `<Q_i>=1.409`, but it worsens absolute QL transfer; Solovev is
therefore admitted as negative evidence, not as an absolute-flux promotion.

The completed CTH-like modified-protocol harvest is now a scoped high-grid
transport holdout, not a normal full-ladder convergence claim. The full
`n48/n64/n80` grid gates fail at every tracked horizon because the coarse
`n48` trace is not converged; the tracked t=350 full-grid sidecar fails only
the common/least grid-difference metrics (`0.296` and `0.272`).
The `t=150` high-grid check has close `n64/n80` means (`0.026` common-window
and `0.009` least-window relative differences) but still fails the late-window
slope gate, so that window is treated as transient. The later high-grid
windows pass: `t=250` and `t=350` agree in time to `0.018` common-window and
`0.019` least-window relative change. The explicit admission checker accepts
only the retained `n64/n80` evidence, and records that the case must not be
described as full `n48/n64/n80` convergence.

![SPECTRAX-GK CTH-like high-grid candidate gate](docs/_static/external_vmec_cth_like_modified_t350_high_grid_convergence_gate.png)

![SPECTRAX-GK CTH-like late high-grid time-horizon gate](docs/_static/external_vmec_cth_like_modified_late_high_grid_time_horizon_gate.png)

The follow-up `n80` seed/timestep audit confirms why the longer window matters.
The first `t=[250,350]` replicate extraction was individually converged but
failed the strict ensemble spread gate (`0.182 > 0.15`). Restart-continuing the
same four variants to `t=700` gives a passed `t=[350,700]` ensemble gate:
mean heat flux `9.60`, mean relative spread `0.041`, and combined SEM/mean
`0.052`. Together with the full-grid failure sidecar, the passed high-grid
gates, and the late time-horizon gate, this passes
`tools/release/check_external_vmec_high_grid_admission.py` and enters the quasilinear
ledger as `split = holdout`. The claim remains bounded: this improves the
model-development holdout set but does not promote absolute quasilinear fluxes.

![SPECTRAX-GK CTH-like long-window replicate gate](docs/_static/external_vmec_cth_like_modified_replicates_t700/replicate_ensemble_gate.png)

Two of the strongest admitted external-VMEC nonlinear holdouts are shown below.
These figures are part of the publication-facing evidence that the nonlinear
inputs are converged enough to be used as negative transfer constraints rather
than as exploratory pilots.

![SPECTRAX-GK ITERModel external-VMEC nonlinear convergence gate](docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.png)

The ITERModel external-VMEC case closes at `t=350` on the `48x48x32` to
`64x64x40` ladder. Its common-window grid difference is about `0.0165`, the
least-window difference is about `0.1415`, and the trend/CV/sample-count gates
all pass.

![SPECTRAX-GK up-down asymmetric external-VMEC nonlinear convergence gate](docs/_static/external_vmec_updown_asym_t450_high_grid_convergence_gate.png)

The up-down asymmetric external-VMEC tokamak closes at `t=450` on the same
ladder. Its common-window and least-window relative differences are about
`0.0435` and `0.0242`, respectively.

![SPECTRAX-GK circular external-VMEC nonlinear convergence gate](docs/_static/external_vmec_circular_t450_high_grid_convergence_gate.png)

The circular external-VMEC tokamak initially failed the shorter `t=150` and
`t=250` admission gates, then closes at `t=450` on the same high-grid ladder:
the common-window and least-window grid differences are about `0.0128` and
`0.0468`. These admitted windows strengthen the quasilinear calibration dataset
without changing the core conclusion: the current absolute-flux model is still
a rejected research candidate, not a shipped predictive transport law.

The follow-up seed/timestep replicate gate initially failed at `t=450` because
one seed still had a drifting terminal window. Extending the same three
replicas to `t=700` closes the physical readiness gate on `t=[350,700]`: the
ensemble mean heat flux is `18.97`, mean relative spread is `0.035`, and
combined SEM/mean is `0.043`.

![SPECTRAX-GK circular VMEC nonlinear replicate gate](docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.png)

Autodiff validation (inverse/sensitivity demo):

![SPECTRAX-GK autodiff inverse demo](docs/_static/autodiff_inverse_growth.png)

This single-mode figure checks that the JAX derivatives are correct and shows how one measured mode constrains the gradients locally. The expected outcome is small observable and Jacobian errors, not exact parameter recovery; the shipped result is a near-perfect match in `(γ, ω)` but a visibly non-unique recovered `(R/L_Ti, R/L_n)` pair.

Autodiff validation (two-mode inverse demo):

![SPECTRAX-GK autodiff two-mode demo](docs/_static/autodiff_inverse_twomode.png)

This two-mode figure is the actual parameter-recovery validation, where the goal is to recover the planted gradients from two independent mode observables. The shipped result reaches the target to numerical precision and the autodiff Jacobian matches finite differences, which is the behavior expected from an identifiable inverse problem.

Single-point runtime TOMLs can also carry their own artifact prefix:

```toml
[output]
path = "tools_out/runtime_case"
```

The executable `--out` flag overrides the TOML value when both are present.

The shipped nonlinear W7-X and HSX runtime TOMLs already set this lightweight
artifact prefix, so long stellarator parity runs leave ``tools_out/...``
diagnostics and summaries behind without extra command-line flags. The direct Python
case wrappers now honor that TOML output contract as well, so chunked
nonlinear runs persist their evolving diagnostics through the same path.

When the nonlinear target ends in `.out.nc` or another `.nc` suffix,
SPECTRAX-GK writes a restartable NetCDF bundle, compatible with the comparison
tooling, instead of the lightweight JSON/CSV sidecars:

- `case.out.nc`: resolved nonlinear diagnostics and metadata
- `case.big.nc`: final fields and moments in real and spectral layouts
- `case.restart.nc`: restart state for continuation runs

The same runtime input can then resume from the saved restart file by setting
restart controls in the TOML:

```toml
[time]
nstep_restart = 100

[output]
path = "tools_out/cyclone_release.out.nc"
restart_if_exists = true
save_for_restart = true
append_on_restart = true
restart_with_perturb = false
```

With that configuration, rerunning the same command resumes from
`tools_out/cyclone_release.restart.nc` when it already exists and appends the
new samples to `tools_out/cyclone_release.out.nc`. Restart appends preserve the
persisted NetCDF diagnostic schema; transient in-memory traces that are not
written to `.out.nc` are not reintroduced when an existing artifact is loaded
for continuation.

## Quickstart (Python)

```python
from spectraxgk import CycloneBaseCase, LinearParams, integrate_linear_from_config
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
import jax.numpy as jnp

cfg = CycloneBaseCase()
grid = build_spectral_grid(cfg.grid)
geom = SAlphaGeometry.from_config(cfg.geometry)
params = LinearParams()

G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

G_t, phi_t = integrate_linear_from_config(G0, grid, geom, params, cfg.time)
```

## Autodiff demo and parallelization notes

The autodiff inverse/sensitivity example lives at
`examples/theory_and_demos/autodiff_inverse_growth.py` and generates the
figure shown above. It uses JAX autodiff on a short linear ITG window, reports
gradients against a finite-difference check, and writes a summary JSON plus
parameter sweeps for both `R/L_Ti` and `R/L_n` alongside the plot. The
single-mode panel should be read as a local inverse demo, not as a global
identifiability claim; in the shipped figure the observable errors are small
while the parameter errors remain finite for exactly that reason.
The two-mode inverse example in
`examples/theory_and_demos/autodiff_inverse_twomode.py` uses two ky modes to
stabilize the inverse problem and provides the release-grade parameter
recovery panel, closing the identifiability gap present in the single-mode
demo. Both autodiff examples now report finite-difference Jacobian checks,
Jacobian rank/conditioning, covariance, standard deviations, correlations, and
one-sigma UQ ellipse area in their summary JSON files.

The differentiable geometry bridge example lives at
`examples/theory_and_demos/differentiable_geometry_bridge.py` and writes the
publication artifact below. It validates the in-memory
`vmec_jax`/`booz_xform_jax` bridge contract used by stellarator optimization
workflows: solver-ready field-line arrays remain JAX-traceable, geometry
observable sensitivities match central finite differences, a two-parameter
inverse design recovers the target observables, and the local UQ covariance is
reported. When `vmec_jax` is available, the same artifact also checks a real
VMEC boundary-aspect derivative through its boundary Fourier API and real VMEC
metric-tensor derivatives through `vmec_jax.geom.eval_geom`. It also samples a
real stellarator VMEC field line from `vmec_jax` metric and magnetic-field
tensors to check that state-level geometry sensitivities reach field-line
observables before any SPECTRAX-GK closure approximation is introduced. The
same path now emits a direct VMEC tensor-derived SPECTRAX-GK flux-tube mapping
and checks its geometry-observable sensitivities against finite differences,
so the differentiability chain starts at `vmec_jax` state coefficients rather
than only at a Boozer spectral adapter. The validation artifact also records a
direct-VMEC-tensor vs imported-VMEC/EIK array-parity audit. A new
`vmec_jax -> booz_xform_jax` Boozer equal-arc core audit now matches the
imported convention for `bmag`, `bgrad`, `gradpar`, `q`, `s_hat`, and the
solver Jacobian at the percent level on the tracked stellarator fixture; the
same audit now reconstructs the zero-beta Boozer metric profiles `gds*`/`grho`
with worst normalized mismatch `3.45e-2` and the loaded-convention zero-beta
drift profiles `cvdrift`/`gbdrift`/`cvdrift0`/`gbdrift0` with worst normalized
mismatch `3.50e-2`. The remaining geometry promotion work is finite-beta and
broader production-runtime drift parity beyond the tracked zero-beta equal-arc
fixtures. When
`booz_xform_jax` is available, it also runs a bounded JAX-native Boozer
spectral transform, samples the resulting Boozer `|B|` spectrum onto a
field-line flux-tube mapping, and checks both derivative paths against central
finite differences. When both optional backends are available, the artifact
also starts from a real `vmec_jax` `VMECState`, perturbs VMEC Fourier
coefficients, converts that state through `booz_xform_jax`, and differentiates
the resulting SPECTRAX-GK field-line geometry observables against central
finite differences. The remaining promotion gate is exact production drift
parity with the imported VMEC/EIK runtime path and then multi-equilibrium
transport-gradient and nonlinear-window gates through the solver.

![SPECTRAX-GK differentiable geometry bridge](docs/_static/differentiable_geometry_bridge.png)

A separate mode-21 parity matrix checks the same Boozer equal-arc path on the
tracked QH, fixed-resolution QI, and shaped-tokamak fixtures. The matrix is
generated by `tools/artifacts/build_vmec_boozer_parity_matrix.py` and enforces
`mboz,nboz >= 21`. The current regenerated artifact passes all matrix rows. The
evaluated QI robustness variants (`ntheta=8,16`) pass, while three QI input
seeds remain explicitly marked `missing_bundled_wout_reference` rather than
being silently promoted. This is a field-line geometry convention gate, not a
production stellarator-transport-gradient claim.

![SPECTRAX-GK VMEC/Boozer parity matrix](docs/_static/vmec_boozer_parity_matrix.png)

The solver-objective geometry-gradient gate differentiates actual
electrostatic linear-RHS eigenpair observables with respect to solver-ready
geometry arrays and checks the implicit left/right eigenpair sensitivities
against central finite differences. This closes the production solver contract
for `FluxTubeGeometryData` gradients. The companion full-chain gate starts
from a real `vmec_jax` state coefficient, maps through `booz_xform_jax`
with `mboz=nboz=21`, builds the SPECTRAX-GK linear RHS, and verifies the
linear eigenfrequency gradient against central finite differences. The
full-chain quasilinear gate uses a richer `Nl=2, Nm=3` moment basis and
checks `gamma`, `omega`, `<k_perp^2>`, the electrostatic heat-flux weight, and
`gamma Q_i/k_perp^2` against central finite differences with maximum relative
error `4.3e-3`. These are differentiability checks on reduced solver
observables and an uncalibrated heat-flux proxy, not calibrated absolute-flux
predictions. This closes the reduced linear/quasilinear stellarator
objective-gradient path on the tracked all-surface QH fixture. A second Li383
holdout now passes the same frequency and quasilinear VMEC/Boozer gradient
contracts at `mboz=nboz=21`; the combined holdout matrix has maximum relative
AD/finite-difference mismatch `4.9e-3` across the reduced linear/quasilinear
objectives. Companion QH and Li383 reduced nonlinear-window estimator gates
differentiate a smooth late-window heat-flux envelope through the same
`vmec_jax -> booz_xform_jax -> SPECTRAX-GK` state path; the expanded matrix
including those estimator rows has maximum relative mismatch `2.7e-2`. That
closes a multi-equilibrium bounded differentiability check for
nonlinear-window-style reduced objectives, but it is not a converged nonlinear
turbulence-gradient or optimized-equilibrium transport claim.

A compact nonlinear finite-difference audit now runs actual SPECTRAX-GK
nonlinear Cyclone startup windows at `R/LTi = base +/- step` plus a repeated
base run. It passes finite-output, repeatability, monotonic drive-response,
startup-window CV/trend, and resolved finite-difference-response gates with
response/base about `0.111`. This is only a startup-response plumbing and
conditioning check. It is not a production heat-flux average, VMEC/Boozer
nonlinear state-gradient, or optimized-equilibrium transport claim.

A companion VMEC/Boozer-perturbed audit starts from the real mode-21
`vmec_jax -> booz_xform_jax` QH state bridge, writes perturbed sampled
geometries to temporary NetCDF files, and runs compact nonlinear startup windows
at `Rcos_mid_surface_m1 = base +/- 1e-5`. It passes finite-output,
repeatability, startup-window conditioning, geometry-response, and resolved
central finite-difference response gates with response/base about `0.040`. The
forward/backward response is asymmetric and not monotone, so this is only a
VMEC/Boozer geometry-perturbed startup observable-path audit. It is not promoted
as a local nonlinear gradient, optimized-equilibrium audit, or production
heat-flux stellarator optimization claim. A memory-bounded Boozer surface
stencil exists for diagnostics and large-equilibrium probes, but it is not used
for the published linear/quasilinear accuracy claim.

For nonlinear transport claims, heat flux must be measured as a long-time
post-transient running average. The gate for future production heat-flux
optimization requires discarding the initial transient, retaining enough
post-transient samples, checking that the cumulative running mean and
independent late blocks are stable, and comparing the same late window against
the tracked nonlinear reference cases. The short FD audits above explicitly
record `transport_average_gate = false` to avoid treating startup-scale fluxes
as saturated transport.

![SPECTRAX-GK solver-objective geometry-gradient gate](docs/_static/solver_objective_gradient_gate.png)

![SPECTRAX-GK VMEC/Boozer solver-frequency gradient gate](docs/_static/vmec_boozer_solver_frequency_gradient_gate.png)

![SPECTRAX-GK VMEC/Boozer quasilinear-gradient gate](docs/_static/vmec_boozer_quasilinear_gradient_gate.png)

![SPECTRAX-GK VMEC/Boozer reduced nonlinear-window-gradient gate](docs/_static/vmec_boozer_nonlinear_window_gradient_gate.png)

![SPECTRAX-GK VMEC/Boozer gradient holdout matrix](docs/_static/vmec_boozer_gradient_holdout_matrix.png)

The reduced VMEC/Boozer optimization path also has aggregate guardrails. The
multi-point gate below checks a quasilinear objective over two field lines and
two `k_y` samples at `mboz=nboz=21`; the growth-vs-quasilinear comparison shows
that growth-rate and quasilinear objectives can choose different initial VMEC
coefficient directions. The VMEC/Boozer aggregate promotion gate now has the
missing production-scope nonlinear holdout: a QH `vmec_jax -> booz_xform_jax`
surface/field-line transport run at `torflux=0.78`, `alpha=1.2`, and
`ky rho_i ~= 0.2` passes the `t=[350,700]` replicated window with
`<Q_i>=7.998`, mean-relative spread `0.0837`, and combined SEM/mean `0.0242`.
This closes the VMEC/Boozer held-out plumbing gate. It is still not, by itself,
a broad nonlinear turbulent-transport optimization claim. The alpha-heldout
split shown below is a positive reduced field-line generalization check, the
surface-heldout split extends this to a true held-out `surface_index`, and the
Li383 panel checks that the same aggregate finite-difference plus line-search
machinery works on a second equilibrium.

![SPECTRAX-GK VMEC/Boozer physical-torflux aggregate gate](docs/_static/vmec_boozer_torflux_aggregate_objective_gate.png)

![SPECTRAX-GK VMEC/Boozer multi-alpha aggregate-objective gate](docs/_static/vmec_boozer_multi_point_objective_gate.png)

![SPECTRAX-GK VMEC/Boozer growth-vs-quasilinear line-search comparison](docs/_static/vmec_boozer_aggregate_line_search_comparison.png)

![SPECTRAX-GK VMEC/Boozer aggregate alpha-heldout gate](docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.png)

![SPECTRAX-GK VMEC/Boozer aggregate surface-heldout gate](docs/_static/vmec_boozer_aggregate_surface_holdout_gate.png)

![SPECTRAX-GK VMEC/Boozer second-equilibrium aggregate gate](docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.png)

![SPECTRAX-GK VMEC/Boozer QH held-out nonlinear transport gate](docs/_static/vmec_boozer_holdout_transport/vmec_boozer_qh_torflux078_alpha120_holdout_ensemble_gate.png)

The backend-free portfolio reducer below is the lightweight contract that
multi-surface, multi-field-line, multi-`k_y` stellarator optimization drivers
should satisfy before they rely on expensive VMEC/Boozer row producers. It checks
normalized sample/objective weights and AD/JVP/finite-difference consistency
for the aggregate scalar objective; it is not a VMEC/Boozer or nonlinear
turbulent-transport optimization claim by itself.

![SPECTRAX-GK stellarator objective portfolio reducer gate](docs/_static/stellarator_objective_portfolio_gate.png)

![SPECTRAX-GK nonlinear startup-window finite-difference audit](docs/_static/nonlinear_window_fd_audit.png)

![SPECTRAX-GK VMEC/Boozer nonlinear startup-window finite-difference audit](docs/_static/vmec_boozer_nonlinear_window_fd_audit.png)

The nonlinear time-horizon audit below separates long post-transient transport
windows from startup plumbing checks and reduced nonlinear-envelope examples.
The external nfp4 QH pilot has now been extended to `t=150`, where its late
heat-flux window is meaningful rather than noise-floor-scale; it remains a
feasibility result because the `48x48x32` grid check changes the late
heat-flux level by about `52%`, and the follow-on `64x64x40` check changes it
again by about `63%`. A later corrected warm-start ladder extended the
`64x64x40` and `80x80x48` comparison to `t=250`, `450`, and `700`; the
`t=700` late-window heat-flux shift remains about `35-37%`, so QH is closed as
negative grid-convergence evidence for the current quasilinear ledger. A new D-shaped
tokamak external-VMEC candidate now passes the longer `t=250` high-grid gate:
`48x48x32` and `64x64x40` differ by `13.9%` on the common late window and
`10.8%` on independently selected least-trending windows. A follow-up
seed/timestep replicate campaign on the `64x64x40`, `t=250` D-shaped case
passes the late-window ensemble gate on `t=[170,250]`: the three accepted
windows have mean heat fluxes `18.8`, `20.8`, and `18.1`, with mean relative
spread `0.141` below the `0.15` gate. A circular external-VMEC replicate
campaign required a longer horizon: the `t=450` ensemble spread was already
small, but seed31 failed terminal-window stationarity, so the accepted artifact
is the `t=700`, `t=[350,700]` replicate with mean relative spread `0.035` and
combined SEM/mean `0.043`. The selected optimized QA equilibrium was then run
through the same long-window protocol at `n64` with two seed replicates and one
timestep replicate. Its accepted `t=[350,700]` window has ensemble mean ion
heat flux `10.19`, mean-relative spread `0.038`, and combined SEM/mean `0.021`.
This is a passed post-transient optimized-equilibrium audit; it is not a
universal absolute-flux model and should be compared case-by-case against the
chosen baseline objective and geometry family.

![SPECTRAX-GK nonlinear transport time-horizon audit](docs/_static/nonlinear_transport_time_horizon_audit.png)

![SPECTRAX-GK D-shaped VMEC nonlinear replicate gate](docs/_static/external_vmec_dshape_replicates/dshape_replicate_t250_ensemble_gate.png)

![SPECTRAX-GK circular VMEC nonlinear replicate gate](docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.png)

![SPECTRAX-GK optimized-equilibrium nonlinear replicate gate](docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.png)

The matched no-ESS reference from the same `vmec_jax` QA campaign also passes
the same `t=[350,700]` seed/timestep ensemble gate. Against that finite-transform
reference, the optimized QA/ESS equilibrium reduces the late-window ion heat
flux from `12.50` to `10.19`, a relative reduction of `18.4%` with `7.82`
combined-SEMs separation.

![SPECTRAX-GK matched no-ESS to optimized QA/ESS nonlinear audit](docs/_static/qa_no_ess_to_optimized_nonlinear_audit.png)

An earlier aspect-6 projected VMEC-JAX transport-gradient step is documented
separately as a negative transfer audit: the reduced single-sample transport
metric improves by `3.55%`, but the matched long-window `t=[350,700]`
nonlinear ensemble comparison changes the mean heat flux from `9.833` to
`9.891` (`-0.585%` relative reduction). That older candidate is not promoted
as a nonlinear heat-flux optimum. The newer full max-mode-5 projected-weight
single-point audits remain scoped candidate evidence only. The strict broad
18-point matrix campaign did not promote a general optimized stellarator:
accepted QA/ESS passed `9/18` samples, projected weight `1e-3` failed early with
only `1/18` passing samples and mean reduction `0.748%`, and projected weight
`5e-4` increased heat flux by `2.99%` on its first completed sample. Broad
nonlinear turbulent-flux optimization is therefore deferred rather than claimed.

The production nonlinear optimization guard below is the enforced claim
boundary. It passes as a release-safety check because startup/reduced nonlinear
artifacts are scoped correctly, three long post-transient replicated holdout
ensembles pass (D-shaped, circular, and QH VMEC/Boozer), the selected
optimized-equilibrium window has explicit `seed31`, `seed32`, and `dt0p04`
provenance, and the matched no-ESS-to-optimized audit shows an `18.4%`
heat-flux reduction with `7.82` combined-SEMs separation. The claim remains
bounded: optimized-equilibrium replicated trace evidence satisfies the strict
count (`4` qualifying ensembles), and `3/3` required matched
baseline-to-optimized audits now pass the explicit `2%` late-window reduction
policy. The accepted matched audits are the no-ESS-to-optimized QA/ESS audit
(`18.4%`) and two max-mode-5 projected-weight audits (`2.68%` and `3.35%`).
Three strict `t=1500` QA objective candidates remain recorded as negative
transfer evidence. The generic replicated-holdout lane is frozen at the three
accepted long-window holdout ensembles; no additional holdouts are being
collected to rescue this claim. This does not prove that the current
quasilinear model is a universal absolute-flux predictor, that nonlinear
turbulence gradients are available, or that the optimization generalizes across
three surfaces, two field-line labels, and multiple `k_y` values.
That broader gate is now reproducible with
`tools/artifacts/build_matched_nonlinear_transport_matrix.py`, which writes and reports
the required 18-point matched baseline/candidate nonlinear matrix before any
multi-surface turbulent-flux optimization language is allowed. The current
negative-evidence ledger is tracked at
`docs/_static/broad_nonlinear_transport_matrix_negative_evidence.json`.

For the next nonlinear optimizer campaign, the current `RBC(1,1)` landscape is
used as a deterministic launch diagnostic from the strict max-mode-5 QA
baseline, not as a promoted nonlinear heat-flux result. The refreshed scan
covers `[-75%, +75%]` in 31 points and evaluates growth plus all explicit
quasilinear rules on a three-surface, two-field-line, three-`k_y` sample set.
The nonlinear row is populated only by long post-transient nonlinear
heat-flux ensembles, using `t_max=1500`, the averaging window `t=[1100,1500]`,
the `n64:64:64:40:40` grid, and seed/timestep variants. The first `-75%`
point showed why this matters: its earlier `t=[350,700]` window was still
drifting. A neighboring `-70%` point still failed the `t=[700,1100]`
timestep-spread gate but passed after continuation to `t=[1100,1500]`, so the
full scan now uses that stricter late window. Until those ensembles finish for
the refreshed 31-point scan, the plot is a landscape/noise diagnostic and
optimizer-design input, not a nonlinear turbulent-flux optimization claim.
The separate nonlinear turbulence-gradient evidence gate is stricter and
remains fail-closed after the completed QA/ESS overdetermined control campaign,
the targeted `RBC(1,1)` seed follow-up, and the bounded `ZBS(1,0)` `7.5%`
follow-up. The overdetermined `RBC(1,1)` candidate is local and
response-resolved but remains too uncertain after five-member state ensembles
(`gradient_uncertainty_rel = 0.683 > 0.5`). The newer `ZBS(1,0)` `7.5%`
follow-up is the clearest locality result: all 12 `t=900` outputs pass, the
response is resolved (`response_fraction = 0.0319`), and the finite-difference
bracket is local (`fd_asymmetry_rel = 0.044`). It still fails promotion because
the plus-state spread is too large (`mean_rel_spread = 0.196 > 0.15`) and the
propagated uncertainty is too high (`gradient_uncertainty_rel = 1.81 > 0.5`).
The current release therefore documents nonlinear turbulence-gradient evidence
as rigorous negative/model-development results, not as a promoted production
nonlinear-gradient or full nonlinear turbulent-flux optimization claim.

The next scientifically efficient step is not another blind single-coefficient
rerun. The tracked design artifact recommends keeping the claim fail-closed and
moving to explicit variance reduction, a control-variate observable, or a
better-conditioned multi-control direction before another expensive production
campaign.
A companion composite-direction manifest defines a smaller descent-oriented
QA/ESS boundary direction with the same long-window contract; that audit also
remains fail-closed after its plus-state spread and central-FD gates.
The newer QL-seeded VMEC-state screen admits `Rsin_mid_surface_m1` and
`Zcos_mid_surface_m1` as internal state-control seeds only. The first
state-to-input attempt deliberately failed closed: stellarator-symmetric
`RBC/ZBS` perturbations produced zero response in those asymmetric
`Rsin/Zcos` controls. The follow-up `LASYM=true` branch now writes and solves
four `RBS/ZBC` perturbation families, measures a full-rank `2 x 4` response
matrix with condition number `1.02`, and updates the state-control runbook with
two mapped input-control directions. This closes the mapping guardrail for
short-bracket nonlinear-gradient launches; it is not yet a promoted converged
nonlinear turbulence-gradient or optimized-equilibrium transport claim. The
checked short-bracket launch contract has also been written and its VMEC decks
have solved normally, preparing two bounded nonlinear campaign manifests for
the next evidence step. Those short-bracket nonlinear campaigns have now been
run on the office GPUs: all `18` nonlinear outputs completed, all output and
replicated-window gates passed, but both central finite-difference gates remain
blocked because `alpha_delta=1e-3` gives small response fractions
(`0.0045` and `0.0015`) with large finite-difference asymmetry. The follow-up
bracket-amplitude sweep also completed all `36` office GPU runs at
`alpha_delta=3e-3` and `1e-2` with no runtime failures. It still passes output
and ensemble gates but fails all four central finite-difference gates; the best
response fraction is only `0.0045`, far below the `0.03` resolved-response
gate. This closes the larger-single-bracket hypothesis as negative evidence;
the next nonlinear-gradient step is variance reduction, longer replicated
windows, or a better-conditioned multi-control observable, not promotion of
this single-control gradient. The bounded `ZBS(1,0)` follow-up at a `7.5%`
bracket has now been run with `12` long `t=900` office-GPU outputs. All output
gates pass over `t=[450,900]`; the baseline and minus ensembles pass, but the
plus ensemble fails the spread gate (`mean_rel_spread = 0.196 > 0.15`) and the
central finite-difference gate remains blocked by propagated uncertainty
(`gradient_uncertainty_rel = 1.81 > 0.5`). This is useful negative evidence:
the response is finally resolved (`response_fraction = 0.0319`) and local
(`fd_asymmetry_rel = 0.044`), but the plus-state variance is still too large
for a production nonlinear turbulence-gradient claim. The refreshed
next-campaign design panel now includes all `16` tracked central-FD artifacts:
zero promoted nonlinear-gradient controls, one bounded-replica follow-up
candidate, and `15` cases that need replacement, locality repair, or variance
reduction before further long-window GPU time is justified. The current
top-level action is now paired-seed or control-variate variance reduction for
the plus-state limiter, not another blind long-window replica campaign. The
paired-seed runbook confirms that common-label plus-minus differences reduce
some common noise but are still too uncertain
(`paired_response_uncertainty_rel = 0.984`). A midpoint common-mode
control-variate screen is promising
(`adjusted_response_uncertainty_rel = 0.238`, `sem_reduction_fraction = 0.759`),
and the independent follow-up now closes that specific uncertainty blocker.
The `21` matched plus/minus control-mean pairs (`42` nonlinear continuations)
pass the strict late-window gate over `t=[600,1100]`: plus/minus ensemble
spread is below `0.15`, no per-seed window rows fail, and the combined
response uncertainty is `0.311 < 0.5`. This closes the
rel7.5 variance-reduced nonlinear-gradient evidence lane, not a broad
nonlinear turbulent-flux optimization claim.

![SPECTRAX-GK VMEC-state nonlinear-gradient launch runbook](docs/_static/nonlinear_gradient_state_control_runbook.png)

![SPECTRAX-GK VMEC state-to-input mapping campaign](docs/_static/nonlinear_gradient_state_to_input_mapping_campaign.png)

![SPECTRAX-GK measured VMEC state-to-input mapping response](docs/_static/nonlinear_gradient_state_to_input_mapping_response.png)

![SPECTRAX-GK asymmetric VMEC state-to-input mapping campaign](docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_campaign.png)

![SPECTRAX-GK asymmetric measured VMEC state-to-input response](docs/_static/nonlinear_gradient_asymmetric_state_to_input_mapping_response.png)

![SPECTRAX-GK VMEC-state short-bracket launch status](docs/_static/nonlinear_gradient_state_control_short_bracket_launch_status.png)

![SPECTRAX-GK VMEC-state short-bracket nonlinear audit](docs/_static/nonlinear_gradient_state_control_short_bracket_nonlinear_audit_status.png)

![SPECTRAX-GK VMEC-state bracket-amplitude sweep status](docs/_static/nonlinear_gradient_state_control_bracket_sweep_status.png)

![SPECTRAX-GK production nonlinear optimization guard](docs/_static/production_nonlinear_optimization_guard.png)

![SPECTRAX-GK QA/ESS ZBS(1,0) nonlinear gradient gate](docs/_static/qa_ess_zbs10_rel5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.png)

![SPECTRAX-GK QA/ESS ZBS(1,0) bounded nonlinear gradient follow-up](docs/_static/qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.png)

![SPECTRAX-GK QA/ESS ZBS(1,0) variance-reduction plan](docs/_static/qa_ess_zbs10_rel7p5_variance_reduction_plan.png)

![SPECTRAX-GK QA/ESS ZBS(1,0) control-variate campaign plan](docs/_static/qa_ess_zbs10_rel7p5_control_variate_campaign_plan.png)

![SPECTRAX-GK QA/ESS ZBS(1,0) independent control-mean gate](docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.png)

The control-variate campaign has both a launch contract and a completed
independent control-mean gate for the rel7.5 evidence lane. The post-run
reduction is automated by
`tools/postprocess_nonlinear_gradient_control_mean_campaign.py`, which requires
the full matched plus/minus seed set with outputs reaching the final
post-transient window before producing the final control-mean gate. It accepts
stride-rounded final times but rejects intermediate checkpoint chunks.

![SPECTRAX-GK QA/ESS overdetermined RBC(1,1) nonlinear gradient gate](docs/_static/qa_ess_profile_gradient_rbc_1_1_nonlinear_gradient_rbc_1_1_central_fd_gradient_gate.png)

![SPECTRAX-GK QA/ESS composite nonlinear gradient gate](docs/_static/qa_ess_descent_profile_rel2_nonlinear_gradient_profile_direction_zbs_1_1_zbs_1_0_rbc_1_1_central_fd_gradient_gate.png)

![SPECTRAX-GK QA/ESS targeted nonlinear gradient follow-up](docs/_static/qa_ess_descent_profile_rel2_plus_delta_followup_replicate_spread_diagnostic.png)

Differentiable stellarator ITG optimization examples live in
`examples/optimization/` and are restricted to actual VMEC-JAX QA workflows:
linear-growth, quasilinear-flux, nonlinear-window transport-objective scripts,
the matched nonlinear ITG audit, and the guarded VMEC-JAX QA driver. Full
`vmec_jax -> booz_xform_jax -> SPECTRAX-GK` nonlinear optimization remains
unpromoted unless production-grade nonlinear turbulence-gradient or robust
finite-difference audits pass with converged post-transient heat-flux windows, continued
curvature/drift parity on additional equilibria, and matched
baseline-to-optimized nonlinear audits for broader geometry families.

For production parallelization of independent work, use
`spectraxgk.batch_map` / `spectraxgk.ky_scan_batches` for ky scans,
quasilinear/UQ ensembles, and sensitivity-sweep plumbing that does not claim a
new scaling result. Runtime `k_y` scans can select the same independent-worker
path from TOML:

```toml
[parallel]
strategy = "batch"
axis = "ky"
num_devices = 4      # or batch_size = 4
backend = "auto"     # "thread" or "process" are explicit alternatives
```

This path preserves serial ordering and uses independent solver calls; it does
not change the solver layout. Whole-state fixed-step nonlinear sharding through
`TimeConfig.state_sharding = "auto"` (or `"ky"` / `"kx"`) remains a
correctness/profiler path for partitioning the packed state array across JAX
devices. It is intentionally limited to state axes: sharding across the `z` FFT
axis is tracked as a future domain-decomposition lane because it requires a
separate communication/layout design. The current profiler-backed artifacts are
`docs/_static/nonlinear_sharding_profile.json` for the local control-flow gate
and `docs/_static/nonlinear_sharding_profile_office_gpu.json` for the two-GPU
office identity gate. Treat both as engineering gates, not as runtime speedup
claims. The matched large strong-scaling sweep in `docs/performance.rst`
confirms this conservative stance: older whole-state nonlinear sharding
artifacts were identity-correct but did not produce a production CPU/GPU speedup
claim, and the newer device-z pencil route passes RHS and fixed-step
transport-window identity but remains below the two-GPU speedup gate. The final
release decision for this tranche is to ship production independent-work
parallelization and defer production nonlinear domain decomposition until the
RHS/update route streams scalar diagnostics in-place and clears matched CPU/GPU
identity plus speedup gates. Production parallelization should therefore focus
on independent `k_y` scans, quasilinear studies, sensitivity sweeps, and
UQ/ensemble batching until that communication-aware nonlinear route is complete.

On current JAX/XLA CPU backends, the nonlinear whole-state `pjit` profiler
skips active multi-device CPU sharding by default because FFT layout/collective
failures can abort the process before Python can catch the error. CPU
forced-device runs remain useful for independent-work and non-FFT-axis checks;
they are not used as production nonlinear domain-speedup evidence.

For UQ and optimization portfolios, `spectraxgk.independent_ensemble_provenance_gate`
is the production identity/provenance check. It runs serial and
`independent_map` ensemble members, verifies result ordering and numerical
identity, checks worker clipping for oversubscribed requests, validates
deterministic reconstruction through the independent-work decomposition
contract, and confirms worker-exception metadata.

![SPECTRAX-GK ky-batch parallelization identity gate](docs/_static/parallel_ky_scan_gate.png)

The ky-batch gate above is generated by
`python tools/generate_parallel_ky_scan_gate.py`. It runs the real Cyclone
linear solver serially and with fixed-shape ky batching, verifies numerical
identity for `gamma` and `omega`, and reports the observed batch speedup for
engineering tracking.

![SPECTRAX-GK independent ky scan strong scaling](docs/_static/independent_ky_scan_scaling_large.png)

The large independent-`k_y` strong-scaling panel uses the real Cyclone linear
solver on 64 modes with `Ny=128`, `Nz=96`, `Nl=4`, `Nm=8`, and `240` RK2
steps per mode. It passes exact `gamma`/`omega` identity against the one-worker
reference. The refreshed release artifact reaches `7.18x` on eight local CPU
workers and `1.88x` on two RTX A4000 GPUs on `ssh office`. This is the preferred
production parallelization path for linear scans, quasilinear studies, and UQ
ensembles. Sensitivity sweeps are covered by the same independent-work
ordering/provenance utilities, but not by a standalone scaling claim yet.

![SPECTRAX-GK parallelization closure status](docs/_static/parallelization_completion_status.png)

The closure status above is regenerated by
`python tools/artifacts/build_parallelization_completion_status.py`. It marks independent
`k_y` scans and quasilinear/UQ ensembles as production-closed, while keeping
whole-state nonlinear sharding and FFT-axis decomposition diagnostic until they
have runtime communication, conservation, transport-window, and profiler-backed
speedup gates. The status JSON also embeds the UQ/optimization ensemble
provenance gate so the production independent-work lane is closed on ordering,
worker clipping, exception metadata, and deterministic reconstruction, not only
speedup and scalar identity.

The decomposition-contract gate below is the lower-level correctness ledger for
parallel work partitioning. It confirms deterministic shard assignment and
serial reconstruction identity for independent `k_y` and UQ portfolios, while
labeling nonlinear state-domain partitioning as diagnostic metadata only.

![SPECTRAX-GK parallel decomposition contract status](docs/_static/parallel_decomposition_status.png)

The latest nonlinear device-z observable split also stays diagnostic: the
auto-chunked two-GPU transport-window route preserves final-state and transport
observable identity, but the compute-only speedup is `1.19x` and the scalar
observable gate is about `42.6x` more expensive than the sharded compute row.
The refreshed fixed-window device-z profile reaches `1.48x` on two GPUs with
identity preserved, still below the `1.5x` production speedup gate.
The next performance tranche is therefore fused device-side diagnostic
accumulation inside the nonlinear RHS/update, followed by full-solver
serial-vs-decomposed transport-window gates.

![SPECTRAX-GK quasilinear UQ ensemble strong scaling](docs/_static/quasilinear_uq_ensemble_scaling_large.png)

The quasilinear/UQ ensemble panel applies the same independent-worker policy
to six late-time Cyclone ITG gradient samples and five `k_y` values per sample
at `Ny=96`, `Nz=64`, `Nl=3`, `Nm=6`, and `2000` RK2 steps. It computes real
linear growth/frequency fits and a reduced mixing-length feature observable,
then checks exact serial identity. On `ssh office`, CPU process scaling reaches
`5.41x` on eight requested workers using six actual ensemble chunks, and the
two-RTX-A4000 GPU run reaches `1.71x`. This is a parallelization and UQ
plumbing result, not a promoted absolute nonlinear heat-flux model.

## Benchmarks

SPECTRAX-GK is validated against standard gyrokinetic benchmarks within the
tracked release scope:

- **Linear growth rates, frequencies, and eigenfunctions:** release-atlas cases
  including Cyclone ITG, ETG, KBM, W7-X, HSX, and shaped tokamak coverage.
- **Nonlinear transport windows:** release-gated heat-flux and energy statistics
  for Cyclone, Cyclone Miller, KBM, W7-X, and HSX.

The root `benchmarks/` directory contains the maintained benchmark drivers,
runtime TOMLs, and the small result index used by the documentation. The
tooling in `tools/` regenerates the atlas and runtime/memory panels from those
inputs while writing raw solver outputs to `tools_out/` or another scratch
directory.
Current promoted benchmark artifacts are indexed in
`benchmarks/results/manifest.toml` and displayed in `docs/benchmarks.rst`.
For the current release pass, the accepted nonlinear validation set is Cyclone,
Cyclone Miller, KBM, W7-X, and HSX. Full-GK ETG nonlinear pilots, TEM/KAW stress
lanes, kinetic-electron extensions, and W7-X zonal-flow recurrence/damping stay
outside the active release parity claim unless a gate-indexed artifact promotes
them explicitly.
The window-statistics artifact uses case-specific mean-relative gates: KBM
`0.02`, HSX `0.05`, Cyclone Miller `0.095`, and the broader release envelope
`0.10` for Cyclone and W7-X while their paper-level tightening lanes remain
open.

## Runtime and Memory Details

Experimental or not-yet-closed lanes such as KAW, TEM, and kinetic-electron
Cyclone are tracked separately and do not appear in the shipped runtime panel.
For the stellarator rows on `office`, the shipped panel uses pre-generated
`*.eik.nc` geometry files rather than on-the-fly VMEC regeneration. The GX
reference rows also run against a consistent local `netcdf-c` / `hdf5`
runtime stack there, because the default `office` stellarator environment
mixed incompatible HDF5 / NetCDF libraries and lacked the VMEC Python helper
dependencies needed for live geometry generation.

These shipped runtime rows are cold wall-time measurements, so the SPECTRAX-GK
nonlinear GPU entries include JAX startup/compile cost. Targeted `office` GPU
profiles on the same short nonlinear cases measured:

- Cyclone nonlinear: `warmup_time_s = 33.957`, `run_time_s = 15.054`
- KBM nonlinear: `warmup_time_s = 27.485`, `run_time_s = 9.725`

This means the current short-run Cyclone and KBM gaps are dominated much more
by cold-start overhead than by steady-state timestep throughput. In steady
state, Cyclone GPU is faster than the shipped GX runtime row, and KBM GPU is
close to parity.
The hollow diamond markers in the runtime subplot show those warm second-run
timings on top of the cold wall-time bars.

### Kernel profiling and gated fast modes

![Nonlinear RHS kernel profile](docs/_static/nonlinear_rhs_profile.png)

The current profiler splits the nonlinear RHS into field solve, linear RHS,
nonlinear bracket, and full RHS kernels on CPU and GPU. The latest bounded
Cyclone profile shows the compiled linear RHS, nonlinear bracket, and full RHS
are the dominant warm-throughput targets, while GPU execution reduces all
measured RHS kernels. The
companion JSON artifact records dominant kernels and grid-to-spectral speedups
so the optimization lane remains traceable and machine-checkable.

The next profiler layer resolves the linear RHS into individual term kernels.
The tracked Cyclone CPU artifact (`docs/_static/linear_rhs_terms_profile.json`)
now includes the zero-collision fast path and linked-FFT refactor and reports
`full_linear_rhs=1.08e-1 s` in the bounded CPU harness. The active-state
companion
(`docs/_static/linear_rhs_terms_profile_z_wave_cpu.json`) injects resolved
parallel variation and reports `full_linear_rhs=1.27e-1 s` while showing
linked-`|k_z|` hypercollisions becoming active; apart from the accepted
zero-collision guard, zero-norm initial-state rows
remain enabled until a state-window identity gate proves they remain inactive
after nonlinear evolution. The
matching `office` GPU artifact
(`docs/_static/linear_rhs_terms_profile_gpu.json`) reports
`full_linear_rhs=5.50e-3 s` on one RTX A4000, and the active-state GPU
companion reports `5.48e-3 s` while reproducing the linked-`|k_z|`/
hypercollision norm match.

The tracked state-window gate
(`docs/_static/linear_rhs_zero_norm_state_window_gate.json`) now makes that
policy executable: it accepts a zero-collision skip for the `nu=0` Cyclone
window but rejects skipping linked-`|k_z|` hypercollisions once a resolved
parallel perturbation is present.

A larger Cyclone Miller companion profile is documented in
`docs/performance.rst` and tracked as
`docs/_static/nonlinear_rhs_profile_miller.{png,json}`. It uses
`Nx=192`, `Ny=64`, `Nz=24`, `Nl=4`, `Nm=8`. After the grid-Laguerre
`einsum` refactor, the matched one-GPU profile gives `full_rhs=1.28e-2 s` in
grid mode and `1.48e-2 s` in spectral mode. Spectral mode still reduces the
GPU nonlinear bracket by about `1.63x`, but the full-RHS timing is limited by
the combined linear-RHS/bracket graph, so the next optimization target is
linear-RHS fusion/cache layout before any broader nonlinear speedup claim.
The matched W7-X/HSX runtime-mode stellarator profiler artifact
(`docs/_static/nonlinear_rhs_profile_stellarator_runtime.json`) records W7-X
and HSX GPU full-RHS calls near `2.7e-2 s` versus CPU calls near `3.1e-1 s`;
those rows close the release-level performance evidence while keeping broader
production nonlinear speedup claims scoped to future identity- and
profiler-gated work.

The full fused linear-RHS trace artifact
(`docs/_static/full_linear_rhs_trace_summary.json`) now records the Cyclone
Miller graph-level profile after electrostatic field specialization:
`warm_seconds=8.09e-2`, first compile+execute `1.40 s`, and `2225` HLO
lines. The matching pre-specialization local artifact had `warm_seconds=1.19e-1`
and `2425` HLO lines, so this is a bounded CPU graph-localization improvement,
not a broad runtime claim. The active `z_wave` companion
(`docs/_static/full_linear_rhs_trace_z_wave_summary.json`) uses the same
specialized graph and reports `warm_seconds=1.29e-1` after resolved parallel
variation is injected; that timing is not promoted as a speedup until a matched
GPU and nonlinear full-RHS profile is refreshed.

![Spectral Laguerre mode gate](docs/_static/laguerre_mode_gate_gpu.png)

The optional spectral Laguerre nonlinear mode is gated, not a default. On the
bounded local CPU and `office` GPU gates it preserves scalar nonlinear
diagnostics across Cyclone, KBM, W7-X, and HSX. The refreshed CPU gate has
maximum relative differences below `8.9e-4` and grid/spectral runtime ratios
of `2.90`, `3.31`, `1.67`, and `0.66` for Cyclone, KBM, W7-X, and HSX,
respectively. The tracked GPU gate has maximum relative differences below
`2.2e-5` and ratios `1.66`, `2.69`, `1.63`, and `0.74`. HSX is slower on both
backends in these bounded gates, so users should treat spectral Laguerre mode
as an opt-in engineering mode and rerun
`python tools/gate_laguerre_nonlinear_modes.py` for their production case
before relying on it for performance claims.

Regenerate the runtime figure from collected per-case summaries with:

```bash
python tools/benchmark_runtime_memory.py \
  --summary-glob tools_out/runtime_memory_*linear.json \
  --summary-glob tools_out/runtime_memory_*nonlinear.json

# For a long office sweep, keep going after a failed row and save per-row logs.
python tools/benchmark_runtime_memory.py --continue-on-error --log-dir tools_out/runtime_memory_logs
```

Parallelization scaling figures are kept in the performance docs rather than
the top-level README. The shipped public claim is the independent-work path for
`k_y` scans, quasilinear studies, and UQ ensembles; sensitivity sweeps use the
same deterministic work partitioning but do not yet have a standalone scaling
claim. Whole-state nonlinear sharding remains an identity/profiler artifact
until a communication-aware nonlinear decomposition has matched CPU/GPU
identity, transport-window, and profiler evidence.

## Examples

The `examples/` directory is organized by physics and configuration:

- **`linear/`**: Linear microinstability drivers for axisymmetric (Tokamak) and non-axisymmetric (Stellarator) geometries.
- **`nonlinear/`**: Nonlinear turbulence simulations and transport analysis.
- **root `benchmarks/`**: Scripts, TOMLs, and result-index pointers for replicating published benchmark results and parameter scans.
- **`theory_and_demos/`**: Pedagogical examples and demonstrations of the underlying numerical methods.

Release-gated nonlinear example lanes include:

- Cyclone ITG
- Cyclone Miller
- KBM
- W7-X
- HSX

A full-GK ETG nonlinear pilot lane is also available at
`examples/nonlinear/axisymmetric/runtime_etg_nonlinear.toml`, but it remains a
pilot until its benchmark operating point, observable contract, and gate-indexed
artifact are promoted.

Reduced collisional ETG workflows have been retired from `main`; the maintained examples now use the full-GK runtime.

## Documentation

Comprehensive documentation is available in `docs/`. Start with
`docs/quickstart.rst`, then use `docs/theory.rst`, `docs/operators.rst`,
`docs/numerics.rst`, `docs/quasilinear.rst`,
`docs/stellarator_optimization.rst`, `docs/parallelization.rst`, and
`docs/release_scope.rst` for the detailed equations, numerical algorithms,
validation gates, examples, and current claim boundaries.

## Testing

Default `pytest` runs skip integration tests for faster feedback. Use:

```bash
pytest
pytest -m integration
python tools/run_tests_fast.py
python tools/run_wide_coverage_gate.py --shards 48 --timeout 300 --fail-under 95 --pytest-arg=-o --pytest-arg=addopts= --pytest-arg=-m --pytest-arg="not slow"
```

`tools/run_tests_fast.py` runs per-file pytest shards with a 300 s per-file
timeout and a 300 s total local budget by default. Use
`--total-timeout 0` only when you explicitly want the full sequential local
pass.

For laptops or shared workstations, run the same wide gate one bounded shard at
a time with `--only-shard N --keep-existing-coverage --skip-combine`, then
finish with `--combine-only --fail-under 95`. CI adds
`--require-shard-data --shard-manifest coverage-wide-shard-manifest.json` so
the final coverage badge cannot be refreshed from an incomplete shard upload.
This keeps every local pytest process under the release timeout instead of
launching one long run.

## Plotting outputs

Use `spectraxgk --plot <artifact>` for supported saved linear summary bundles
and nonlinear diagnostic/NetCDF bundles. For a nonlinear NetCDF-specific
diagnostic figure:

```bash
python examples/utilities/plot_runtime_outputs.py tools_out/cyclone_nonlinear.out.nc \
  --out tools_out/cyclone_nonlinear_diagnostics.png
```

## Contributing

SPECTRAX-GK is an open-source project welcoming contributions. Whether it's improving runtimes, reducing memory usage, or expanding the physics models, your help is appreciated.

## License

MIT License.
