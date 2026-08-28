# GKX 3.0: Research-Grade Modernization Plan

**Planning branch:** `plan/research-grade-roadmap`
**Ground-truth file:** `plan.md` at the repository root on that branch
**Baseline:** GKX 1.8.2, commit `e89c7fed31657f32b638e653c7b266e33cded805`, 25 August 2026
**Status:** Maintainer decisions incorporated; implementation-ready
**Primary audience:** GKX maintainers, contributors, and coding agents

## 1. Mission

Turn GKX into the reference local gyrokinetic platform for tokamak and stellarator core-turbulence calculations and design workflows. GKX should combine:

- trustworthy linear, quasilinear, and nonlinear turbulent-transport calculations;
- electrostatic and electromagnetic physics;
- adiabatic and kinetic electrons;
- arbitrary kinetic species, impurities, and trace-species diagnostics;
- validated model, Sugama, improved-Sugama, and Coulomb collision operators;
- efficient Hermite-Laguerre velocity-space resolution and physically controlled closures;
- JAX-native differentiation, uncertainty quantification, and optimization;
- efficient CPU and NVIDIA GPU execution, with multi-GPU work promoted only after measured value;
- direct coupling to equilibrium, optimization, transport, and interoperability tools;
- a small, deliberate, maintainable Python codebase with a clear user experience.

GKX must stop being described or organized as a Python reproduction of GX. GX remains a principal reference implementation, comparison code, and source of numerical provenance. GKX 3 should have its own product contract, methods, evidence, and research program.

## 2. Product position

GKX 3 occupies a specific category:

> A differentiable, research-grade, local delta-f flux-tube gyrokinetic solver for tokamak and stellarator turbulent transport, designed for repeated simulation, uncertainty analysis, transport coupling, and high-dimensional optimization.

This scope is now approved. GKX 3.x remains a local core-turbulence code. It does not expand into global, full-f, edge, scrape-off-layer, particle-in-cell, or whole-device gyrokinetics.

GKX should no longer be described as a JAX clone of GX. GX remains a principal formulation, performance, validation, and provenance reference. GKX owns its own API, numerical methods, evidence, optimization workflows, and capability boundaries.

### 2.1 Primary strengths to build around

1. **Hermite-Laguerre velocity space.** Retain the compact spectral representation and the ability to move between low-moment and high-fidelity kinetic calculations.
2. **JAX execution.** Use compilation, automatic differentiation, vectorization, and explicit sharding as scientific capabilities rather than marketing labels.
3. **Three-dimensional geometry.** Make tokamak and stellarator geometry equally ordinary user paths.
4. **End-to-end design coupling.** Make VMEX -> flux-tube geometry -> GKX -> objective -> gradient a first-class, tested workflow.
5. **Advanced linearized collisions and closures.** Use the moment representation to make collision physics and velocity-space closure a differentiating numerical program.
6. **Evidence discipline.** Preserve fail-closed scientific claims while removing campaign governance from the installable solver.
7. **A learnable product.** Keep one obvious CLI, one obvious Python API, one result schema, and a small number of maintained examples.

### 2.2 Explicit non-goals through GKX 3.x

The following are out of scope unless this plan is amended by Rogerio Jorge:

- global full-radius core turbulence;
- full-f edge and scrape-off-layer turbulence;
- magnetic-axis, separatrix, divertor, or wall-resolved simulations;
- particle-in-cell algorithms;
- replacing VMEX, DESC, SIMSOPT, ESSOS, Pyrokinetics, T3D, TORAX, TGYRO, or PORTALS;
- a universal quasilinear claim outside a model's declared and tested domain;
- a new general-purpose optimization framework;
- retaining undocumented GKX 1.x internal import paths;
- keeping one-off research campaigns, internal evidence dashboards, or release-evidence builders in `src/gkx`;
- implementing the bilinear perturbation collision term `C[delta f, delta f]` while GKX remains a Maxwellian-background local delta-f code;
- committing to AMD GPU or TPU support before maintainers, hardware, and CI exist.

GENE-X, XGC, GTC, and ORB5 already cover global or particle-in-cell regimes. GKX should interoperate with broader modeling ecosystems rather than dilute its local design focus.

## 3. Current-state audit

### 3.1 What is already good

The 1.8.2 repository is not a failed starting point. Preserve these gains:

- PyPI packaging and a working `gkx` executable;
- Read the Docs configuration;
- a repository below 20 MiB after history and media cleanup;
- compressed figures and movies with larger media moved to release assets;
- direct execution from TOML and VMEC/VMEX WOUT files;
- progress, elapsed-time, and ETA reporting for long runs;
- restartable NetCDF output and automatic figures;
- linear, nonlinear, quasilinear, collision, geometry, differentiation, and optimization pathways;
- named tokamak and stellarator benchmark cases;
- package-wide coverage gates at or above 95 percent;
- artifact-backed claim boundaries and the willingness to retain negative results;
- current CI success on the 1.8.2 release commit.

The modernization must not erase working physics in pursuit of a prettier tree.

### 3.2 Main structural problems

The current architecture manifest records approximately:

- 199 installable Python source files, with a target of 45;
- more than 91,000 source lines, with a target of 45,000;
- 101 Python test files;
- 95 Python files under `tools`;
- a public API registry with well over one hundred exports, including low-level kernels, report builders, campaign diagnostics, and implementation-specific helpers;
- separate `runtime`, `workflows`, `solvers`, `operators`, `terms`, `diagnostics`, `artifacts`, `objectives`, and `parallel` layers whose ownership frequently overlaps;
- several solver and time-integration routes that do not support the same collision, sharding, differentiation, and output capabilities;
- documentation that mixes user learning, API reference, scientific derivation, release governance, research status, release evidence, and internal architecture policy;
- tests that partly protect user behavior and physics, and partly protect repository tools and historical artifact machinery.

A green architecture gate currently means “no regression beyond the audited baseline,” not “the target architecture has been reached.” GKX 3 must finish the consolidation instead of perpetually carrying the migration framework.

### 3.3 User-facing problems

- The obvious CLI path is obscured by names such as `run-runtime-nonlinear` and files prefixed with `runtime_`.
- There are too many ways to invoke similar calculations.
- The top-level API exposes implementation details that users should not memorize.
- Examples are organized by the history of features rather than a short learning path.
- The README is much improved but remains long enough to act as a second documentation site.
- Documentation pages are numerous, large, and often written for maintainers or release evidence rather than users.
- Some capability labels are broader than the supported runtime scope, especially around “full Coulomb” collisions.

### 3.4 Scientific gaps that matter most

The plan should distinguish implementation from promotion. Some paths exist but need broader evidence.

1. Kinetic-electron and electromagnetic nonlinear validation is not yet as broad as the electrostatic adiabatic-electron path.
2. A robust absolute-flux quasilinear model is not yet established.
3. Advanced collision operators are strongest in low-order or offline algebraic form; arbitrary-order, multispecies, finite-perpendicular-wavelength production use remains incomplete.
4. Production multi-GPU nonlinear scaling remains weaker than the correctness work.
5. Nonlinear differentiation is a useful finite-window discrete derivative, not a converged derivative of the invariant turbulent measure.
6. Broad, multi-surface, multi-field-line nonlinear stellarator optimization remains unproven.
7. Rotation, equilibrium ExB shear, momentum transport, and some kinetic-electron instabilities require a deliberate capability audit and validation program.

## 4. Lessons from established codes

### 4.1 GX: time to solution and spectral convergence

GX establishes the closest numerical and performance reference. It demonstrates that a Fourier-Laguerre-Hermite local solver can perform useful nonlinear tokamak and stellarator calculations in minutes on one or a few GPUs, with higher-fidelity results in hours. Its design lessons are:

- optimize the entire algorithm around accelerator execution;
- make resolution a controllable accuracy-cost knob;
- use full phase-space pseudospectral structure where it is advantageous;
- minimize communication by decomposing species and Hermite modes;
- validate nonlinear convergence, not only individual operators;
- report time to a physically useful answer.

GKX must preserve numerical parity where the models are intended to match, but should not inherit every implementation choice or public convention without review.

### 4.2 GENE: physics breadth and mature scalability

GENE is the breadth benchmark. Its documented capabilities include arbitrary kinetic species, electrostatic and magnetic fluctuations, advanced inter-species collision models, tokamak and stellarator equilibrium interfaces, rotation and ExB shear, local and global calculations, and broad phase-space parallelism. GKX does not need to reproduce global GENE. It should use GENE to define the local-physics validation matrix and to identify missing production features.

### 4.3 stella: center the code on a clear numerical method

The stella implementation is centered on an operator-split implicit-explicit method, its stability and efficiency, and linear and nonlinear verification in axisymmetric and non-axisymmetric geometry. The lesson for GKX is direct: a feature inventory is not a coherent computational design. GKX capability work should be organized around identifiable advances such as:

- a memory-bounded discrete adjoint for nonlinear gyrokinetics;
- a matrix-free arbitrary-order finite-wavelength collision algorithm;
- an adaptive or asymptotically correct Hermite-Laguerre closure;
- a scalable species-Hermite decomposition with measured communication bounds;
- a validated differentiable stellarator-design workflow.

### 4.4 CGYRO, TGLF, TGYRO, and GACODE: ecosystems win users

The GACODE stack shows the value of a coherent path from nonlinear gyrokinetics to reduced transport models and profile evolution. GKX should not only be a solver binary. It should expose stable inputs, outputs, metadata, and Python functions that transport and optimization codes can call repeatedly.

### 4.5 Pyrokinetics: interoperability should not be reimplemented

Pyrokinetics standardizes local geometry, species, numerics, input/output conversion, normalization, and analysis across GS2, CGYRO, GENE, stella, TGLF, GKW, and GX. A GKX adapter is a high-return community task. GKX should not create another universal conversion framework inside its own package.

### 4.6 GTC, XGC, ORB5, and GENE-X: preserve the scope boundary

These codes demonstrate the computational and physical specialization required for global, edge, SOL, wall, and particle-in-cell calculations. Their existence reinforces the GKX product boundary: local core turbulence, rapid repeated solves, and differentiable design.

### 4.7 iGENE and gyaradax: JAX or differentiation is not enough

By 2026, differentiable nonlinear local gyrokinetics and JAX-native local gyrokinetics already exist in the literature. GKX must distinguish itself through three-dimensional stellarator geometry, Hermite-Laguerre methods, advanced collisions, evidence quality, and end-to-end design results. “Written in JAX” is an implementation fact, not the product definition.

## 5. Product contracts

GKX 3 should be designed around five user personas.

### 5.1 New user

Needs to install GKX, run a known case, see progress, obtain one output bundle, and make a correct plot without understanding package internals.

### 5.2 Production turbulence user

Needs stable TOML inputs, preflight diagnostics, restart, reproducible metadata, resolution guidance, convergence tools, parameter scans, and trustworthy CPU/GPU execution.

### 5.3 Physics or numerical-method developer

Needs pure kernels, clear equations, small modules, invariant tests, manufactured solutions, profiling, and explicit provenance.

### 5.4 Optimization and UQ user

Needs pure PyTree-compatible inputs, value/gradient contracts, batched execution, common-random-number controls, uncertainty-aware objectives, and composition with VMEX and other codes.

### 5.5 Integrated-modeling user

Needs a stable programmatic `solve(case) -> result` contract, versioned NetCDF, units and normalization metadata, error estimates, and adapter-level compatibility with Pyrokinetics and transport solvers.

## 6. One obvious public workflow

### 6.1 Python API

The primary API should converge on:

```python
import gkx

case = gkx.load("cyclone.toml")
result = gkx.solve(case)
gkx.plot(result)
```

Scans and objectives should be equally direct:

```python
scan = gkx.scan(case, ky=[0.1, 0.2, 0.3])

objective = gkx.objective(case, diagnostic="ion_heat_flux")
value, gradient = gkx.value_and_grad(objective, design)
```

Target top-level public API: no more than 30 documented names. Everything else remains available from deliberate subpackages for advanced users, but is not re-exported from `gkx`.

Suggested top-level names:

- `load`, `solve`, `scan`, `plot`;
- `Case`, `Grid`, `Species`, `Geometry`, `Physics`, `Numerics`, `Output`;
- `LinearResult`, `NonlinearResult`, `ScanResult`;
- `objective`, `value_and_grad`;
- `available_devices`, `doctor`;
- selected collision-model constructors only if they are stable user concepts.

Do not expose report builders, validation-policy classes, raw RHS functions, sharded kernels, cache internals, or campaign utilities at the package root.

### 6.2 Command-line interface

Replace historical command names with one small command family:

```console
gkx run case.toml
gkx scan case.toml --ky 0.1 0.2 0.3
gkx plot result.nc
gkx check case.toml
gkx estimate case.toml
gkx info result.nc
gkx examples
gkx doctor
```

`gkx case.toml` may remain a shorthand for `gkx run case.toml`.

The CLI must:

- use `argparse` as the parsing dependency;
- use Rich for progress, tables, warnings, and diagnostics;
- avoid keeping both Rich and tqdm;
- print the chosen device, precision, model, geometry, state shape, estimated memory, timestep policy, output path, and restart status before a long run;
- show compilation separately from integration;
- show simulated time, timestep or chunk, wall time, throughput, estimated completion, and the monitored physical diagnostic;
- fail before compilation on invalid or unsupported combinations;
- record every resolved input and automatic choice in the output bundle.

### 6.3 Input policy

Use TOML as the only first-class human-authored input format.

- TOML is readable, versionable, and already established in GKX.
- JSON is appropriate for machine-readable summaries, provenance, and API interchange, but maintaining two equal user input schemas would double validation and documentation burden.
- Every TOML has `schema_version`.
- Unknown keys are errors by default.
- Deprecated keys produce one actionable warning and a migration suggestion.
- Physical or engineering units should be accepted at the user boundary where practical, then normalized once into an internal convention.
- Every result stores the complete normalized case and original input text.

### 6.4 Output policy

The canonical result is versioned NetCDF with:

- coordinates and dimensions with stable names;
- normalization and units metadata;
- fields, moments, fluxes, spectra, fit windows, convergence state, and uncertainty metadata;
- geometry provenance and equilibrium checksums;
- code, dependency, device, precision, and git/release version;
- restart state in a documented companion group or file;
- a compact JSON summary for quick inspection and workflow systems.

CSV is allowed only for deliberately tabular exports, never as the only production result.

## 7. Target package architecture

The package should be shallow and organized by scientific ownership, not execution history.

```text
src/gkx/
  __init__.py          # small lazy public surface
  api.py               # load, solve, scan, plot, objective
  case.py              # immutable user-facing case PyTrees
  cli.py               # argparse + Rich; no physics

  physics/
    equations.py       # normalized gyrokinetic equation and term assembly
    fields.py          # quasineutrality, Ampere, b_parallel solves
    collisions.py      # collision protocol and promoted models
    moments.py         # physical moments, fluxes, free energy
    closures.py        # physical and numerical hierarchy closures

  geometry/
    analytic.py        # slab, s-alpha, simple models
    miller.py          # Miller geometry
    vmec.py            # VMEC/VMEX/imported flux-tube geometry
    boundaries.py      # linked, generalized twist-shift, non-twisting policy

  numerics/
    grids.py           # spatial and velocity grids
    basis.py           # Hermite-Laguerre transforms and gyroaverages
    spectral.py        # FFTs, dealiasing, nonlinear brackets, projection
    linear.py          # matrix-free linear operator and eigen solve
    nonlinear.py       # state RHS and nonlinear step
    time.py            # explicit, IMEX, checkpointing, adaptive chunks
    parallel.py        # batching and promoted domain decomposition

  solve/
    linear.py          # linear point, scan, eigenmode workflows
    nonlinear.py       # saturation, restart, diagnostics
    diagnostics.py     # convergence and uncertainty orchestration

  quasilinear/
    weights.py         # physical eigenmode flux weights
    saturation.py      # explicit saturation-model interface
    model.py           # calibrated prediction and uncertainty

  optimize/
    objectives.py      # stable scalar/vector objectives
    derivatives.py     # eigen, trajectory, implicit and FD contracts
    stellarator.py     # optional VMEX composition

  io/
    config.py          # TOML schema and migration
    results.py         # NetCDF and JSON summary
    plotting.py        # standard user plots
```

This is a ceiling, not a requirement to create every file. A new module is justified only when it owns a coherent equation, algorithm, or user contract and replaces more complexity than it adds.

### 7.1 Architecture budgets

Proposed GKX 3 release gates:

- no more than 45 installable Python files;
- no more than 45,000 nonblank, noncomment source lines;
- median production module below 500 lines;
- no production module above 900 lines;
- public facades below 300 lines;
- no more than 30 top-level public names;
- no more than three package-directory levels below `src/gkx`;
- no folder with only one implementation file;
- no parallel “reports,” “contracts,” “policies,” “strategies,” or “artifacts” hierarchies;
- import of `gkx` remains lazy and lightweight;
- no generated code, raw campaign output, or publication data in the wheel.

### 7.2 Deletion test

For every module, class, function, script, and exported name, answer:

1. Which equation, numerical method, user workflow, or release contract owns it?
2. Which production module imports it?
3. Which user or maintained example needs it?
4. Which scientific or behavioral test would fail if it disappeared?
5. Why can it not live in the nearest scientific owner?

Delete or internalize it if these questions have no concrete answer.

## 8. Source-code standards

### 8.1 Style

- Prefer small pure functions over policy objects and registries.
- Use immutable dataclasses or explicit PyTrees for case and solver state.
- Separate host orchestration from compiled numerical kernels.
- Keep docstrings short: purpose, arguments, returns, important convention, and reference when needed.
- Comments explain an equation, convention, or nonobvious numerical choice; they do not narrate obvious Python.
- Avoid repeated wrappers whose only purpose is renaming or forwarding arguments.
- Avoid strings that select internal behavior when a small typed value or enum-like literal is clearer.
- Avoid class hierarchies unless runtime polymorphism is genuinely needed.
- Keep user-facing errors typed, specific, and actionable.
- Keep internal array axes and normalization conventions explicit in names and documentation.
- Use one canonical term assembly and one canonical field solve.

### 8.2 JAX boundary rules

Compiled functions must not:

- perform file I/O;
- print or invoke progress callbacks unless a specifically measured callback is required;
- convert traced arrays through NumPy;
- branch on traced values in Python;
- create data-dependent shapes;
- recreate jitted function objects in repeated calls;
- hide static topology in mutable global state;
- silently change precision;
- donate buffers whose ownership is not clear to the caller.

### 8.3 Provenance

GX-derived and literature-derived code must have concise, checkable provenance.

Create `PROVENANCE.md` with one row per adapted scientific component:

| GKX symbol | Source | Revision | Original file/symbol | Author or contributor | Equation/reference | Nature of adaptation |
|---|---|---|---|---|---|---|

A derived function should also contain a short local marker, for example:

```python
def miller_metric(...):
    """Return Miller metric coefficients.

    Provenance: GX `geometry.cpp::...`, revision `<hash>`, implementation
    attributed there to Rahul Gaur; equations follow Miller et al. (1998).
    GKX changes the storage layout and uses JAX array operations.
    """
```

Do not paste long provenance essays into every docstring. The local marker points to the central ledger. Record source license compatibility before retaining copied or closely translated code.

### 8.4 Tooling

Keep the required developer stack small:

- Ruff for formatting and linting;
- mypy or pyright for package typing, with a plan to remove broad ignores;
- pytest and pytest-cov;
- Sphinx, MyST-Parser if adopted, and one theme;
- build and twine for releases.

Do not add a tool unless it replaces manual review or an existing dependency.

## 9. Dependency and packaging policy

### 9.1 Runtime dependencies

Audit every dependency against actual imports and user value. The target core is:

- `jax`;
- `numpy`;
- `scipy` only where JAX or the standard library lacks a required host operation;
- one NetCDF implementation;
- `matplotlib` for standard plotting;
- `rich` for CLI output.

Approved dependency decisions:

- Python 3.11 remains the minimum supported version.
- `jaxlib` should normally arrive through the selected JAX installation rather than be declared independently.
- Replace `tqdm` with Rich.
- Promote the native explicit Runge-Kutta implementation as the explicit owner.
- Promote one native operator-split IMEX implementation as the stiff owner.
- Retain Diffrax temporarily only as a migration oracle while value, order, stability, restart, diagnostics, and derivative parity are established; then remove it from the runtime dependency set unless it proves a unique promoted capability.
- Retain Equinox only if it materially simplifies stable PyTrees or transformations after the architecture rewrite.
- Retain SOLVAX only for algorithms GKX exercises and validates.
- Treat VMEX as the supported live-equilibrium and optimization integration, not as a source-code copy inside GKX.
- `booz_xform_jax` owns the Boozer transform. It should not be a direct core GKX dependency merely to construct turbulence geometry.
- Keep documentation, release, comparison, and development tools out of runtime dependencies.

The preferred installation layers are:

1. `pip install gkx`: analytic, Miller, imported flux-tube/EIK geometry, linear/nonlinear solves, outputs, and plotting;
2. `pip install gkx vmex`: live differentiable VMEX geometry and stellarator optimization;
3. install `booz_xform_jax` only when Boozer spectra, `boozmn` I/O, quasisymmetry/omnigenity diagnostics, or another Boozer-specific workflow requires it.

Do not physically merge the VMEX, `booz_xform_jax`, and GKX repositories. Integrate them through small public array contracts and delete duplicated implementations.

### 9.2 Version policy

GKX 3 requires Python 3.11 or newer.

The preferred `project.dependencies` list has no speculative upper bounds. Bare names are preferred, but compatibility claims must remain truthful.

Use this rule:

- use a bare dependency name when GKX works throughout the maintained support window;
- use a lower bound only when an older version lacks an API or correctness fix required by GKX;
- never use speculative upper bounds;
- document every lower bound in `docs/reference/compatibility.md`;
- test Python 3.11 with a minimum supported dependency stack and a current stack;
- test the supported NVIDIA CUDA stack separately;
- keep exact reproducible CPU and NVIDIA GPU environments in constraints files outside package metadata;
- do not add AMD GPU or TPU release gates in GKX 3.0.

This keeps normal installation lightweight without claiming compatibility that the code does not have.

### 9.3 Distribution

Required release outputs:

- PyPI wheel and source distribution;
- conda-forge feedstock after the stable API is frozen;
- a clean-install wheel smoke test;
- a clean-install source-distribution smoke test;
- a release checksum manifest;
- `CITATION.cff`, license, code of conduct, contributing guide, security policy, and changelog;
- semantic versioning with an explicit compatibility policy;
- signed or trusted-publisher PyPI release automation;
- release notes that separate physics, numerics, API, performance, and fixes.

### 9.4 Size budgets

- fresh clone below 15 MiB, hard ceiling 20 MiB;
- wheel below 10 MiB unless a reviewed scientific table requires more;
- tracked documentation media below 5 MiB total;
- individual tracked figures normally below 300 KiB;
- compressed movies may use release assets; large GKX-owned validation data remains local unless `rogeriojorge` explicitly approves publication;
- no WOUT, full NetCDF run, profiler trace, or raw ensemble in git.

## 10. Testing and scientific evidence

Line coverage is a maintenance metric, not a scientific validation claim. GKX 3 keeps at least 95 percent package-wide statement coverage and organizes evidence into the following ladder.

### 10.1 Evidence ladder

- **E0 - contract:** shapes, dtypes, errors, schema, and public behavior.
- **E1 - mathematics:** identities, symmetry, null spaces, conservation, adjointness, and free-energy signs.
- **E2 - numerics:** manufactured solutions, observed order, convergence, conditioning, restart identity, and serial/parallel identity.
- **E3 - analytic physics:** Landau damping, fluid and collisionless limits, zonal-flow residuals, conductivity, and known asymptotics.
- **E4 - literature benchmark:** reproduce a published case with the same normalization, geometry, resolution policy, and observable.
- **E5 - independent code comparison:** matched GKX/GX, GENE, stella, GS2, or CGYRO inputs and postprocessing.
- **E6 - nonlinear statistics:** stationary windows, autocorrelation-aware uncertainty, seed and timestep replicates, and resolution ladders.
- **E7 - performance:** synchronized cold/warm timing, memory, transfers, compile count, and scaling at matched accuracy.

Every promoted scientific claim must point to at least one appropriate evidence level. A frozen JSON report is not a substitute for the solver run that produced it.

### 10.2 Target test layout

```text
tests/
  conftest.py
  test_api.py
  test_case_config.py
  test_io_and_restart.py
  test_grids_and_basis.py
  test_geometry.py
  test_fields.py
  test_linear_operator.py
  test_nonlinear_operator.py
  test_time_integration.py
  test_diagnostics.py
  test_collisions.py
  test_closures.py
  test_quasilinear.py
  test_autodiff.py
  test_parallel.py
  test_cli_and_examples.py

  physics/
    test_linear_benchmarks.py
    test_zonal_and_landau.py
    test_electromagnetic_modes.py
    test_collision_transport.py
    test_nonlinear_transport.py

  integration/
    test_tokamak_workflows.py
    test_stellarator_workflows.py
    test_optimization_and_coupling.py
```

Proposed gate: no more than 30 Python test files and 35,000 test lines while preserving or improving detection power. Use parametrization and shared fixtures rather than one file per option or tool.

### 10.3 What to delete from tests

Delete tests that only assert:

- the existence of a historical wrapper;
- the continued presence of a one-off tool;
- a report schema no user consumes;
- a generated dashboard fingerprint without regenerating its source calculation;
- a deleted example or deprecated internal import;
- architecture policy that becomes unnecessary after the target architecture is reached.

Keep tests for public compatibility adapters during their declared deprecation period.

### 10.4 Physics validation matrix

The stable release matrix should include, at minimum:

**Analytic and reduced cases**

- free streaming and phase mixing;
- Landau damping and recurrence control;
- slab secondary instability;
- energy and free-energy conservation;
- collision invariants and H-theorem;
- Rosenbluth-Hinton and stellarator zonal-flow responses;
- Spitzer-Harm conductivity and high-collisionality limits.

**Tokamak linear cases**

- Cyclone ITG with adiabatic electrons;
- kinetic-electron ITG;
- ETG;
- TEM;
- electromagnetic KBM;
- microtearing or another electron electromagnetic branch;
- Miller shaping and ExB shear once implemented.

**Stellarator linear cases**

- W7-X and HSX ITG;
- at least one QA, QH, and QI configuration;
- multiple field lines where rational or low-shear effects matter;
- kinetic-electron and electromagnetic stellarator cases.

**Nonlinear cases**

- Cyclone adiabatic-electron ITG;
- Cyclone kinetic-electron ITG;
- Miller-shaped tokamak;
- one electromagnetic case;
- W7-X and HSX or equivalent stellarator cases;
- one optimized-equilibrium holdout;
- one collision-model comparison.

For each nonlinear case, specify prospectively:

- resolution ladder;
- timestep ladder;
- transient exclusion and stationarity criteria;
- minimum autocorrelation times in the averaging window;
- seed or timestep replicates;
- uncertainty definition;
- cross-code observable and tolerance.

### 10.5 CI tiers

**Pull-request fast tier**

- import and wheel smoke;
- Ruff and typing;
- small mathematics and numerical tests;
- public CLI/examples;
- target runtime below 10 minutes.

**Pull-request broad CPU tier**

- complete non-slow suite;
- combined coverage at least 95 percent;
- docs warnings-as-errors;
- package and repository-size gates;
- minimum and latest dependency stacks.

**Scheduled or self-hosted GPU tier**

- representative linear and nonlinear runs;
- CPU/GPU numerical parity;
- memory and performance regression;
- advanced collision kernels;
- multi-device identity and scaling.

**Release tier**

- full physics validation matrix;
- clean wheel and source installs;
- paper/release figure regeneration from immutable inputs;
- external comparison artifacts;
- repository, wheel, and documentation size checks.

Make the required CI contexts actual protected-branch requirements. A protected branch with no required contexts is not a release gate.

### 10.6 External-code comparison policy

GX, GENE, stella, CGYRO, and other maintained codes are important local comparison tools, but they are not permanent executable dependencies or unquestionable golden oracles for GKX.

Rules:

- external-code binaries and raw outputs stay in the maintainers' local comparison environment;
- do not commit or publish raw comparator outputs without a separate explicit decision;
- every comparison records the comparator commit, build options, input, normalization, resolution, solver residual, and postprocessing version;
- a disagreement is investigated through normalization, residual, resolution, timestep, and diagnostic definitions before either code is declared correct;
- permanent GKX CI must remain self-contained and future-proof;
- convert lessons from an external comparison into one or more of:
  - an analytic or asymptotic test;
  - a manufactured solution;
  - a conservation, symmetry, or free-energy test;
  - an independently implemented reference formula;
  - an observed-order or convergence test;
  - a literature-anchored scalar or interval;
  - an internal serial/parallel or alternate-algorithm identity test;
- a compact historical cross-code summary may appear in documentation, but it must not become a version-agnostic test oracle;
- local external comparisons remain required before promoting broad physics claims, even though their raw outputs are not part of the repository.

This policy protects GKX from inheriting silent changes or bugs in another code while preserving the scientific value of independent comparison.

## 11. JAX and performance program

### 11.1 Performance contract

Performance claims must always report:

- cold process startup;
- device transfer;
- trace and compile time;
- first execution;
- warm prepared execution;
- simulated time per wall-clock second;
- peak host and device memory;
- input/output transfer volume;
- compile count and cache hit behavior;
- numerical precision;
- physical resolution and achieved error;
- device model, software stack, and concurrency state.

All JAX timing must synchronize with `block_until_ready()`.

### 11.2 Prepared simulations

The main performance abstraction should be a prepared simulation with fixed topology and dynamic physical arrays:

```python
simulation = gkx.prepare(case)
result = simulation.run(initial_state=state, parameters=parameters)
```

Preparation owns:

- grid and static shapes;
- geometry layout;
- transform plans and masks;
- compiled step or scan;
- sharding specification;
- output sampling policy.

Repeated scans and optimizations must not recreate Python function objects or recompile unchanged topology.

### 11.3 Kernel rules

- JIT the outermost physical step, scan, or objective.
- Use `lax.scan`, `fori_loop`, or `while_loop` for hot loops.
- Use `vmap` for independent species, modes, field lines, or ensemble members when it lowers overhead.
- Fuse term application only when profiling shows a benefit; prevent pathological XLA fusion when it increases data rereads.
- Keep array layout deliberate and documented.
- Avoid materializing spatially varying dense matrices when a matrix-free contraction is available.
- Use buffer donation at clear ownership boundaries.
- Use persistent compilation cache in the CLI, configured before first compilation.
- Use exact or highest-precision contractions only where conservation or conditioning requires them.
- Keep production float32, validation float64, and any mixed-precision policy explicit.
- Profile before introducing Pallas or custom kernels.

### 11.4 Solver strategy

Consolidate to these promoted routes:

1. native low-storage RK2/RK3/RK4 for standard explicit calculations;
2. one native operator-split IMEX route for kinetic electrons, stiff parallel streaming, and collisions;
3. one matrix-free eigensolver with implicit differentiation;
4. one checkpointed native discrete adjoint for finite nonlinear windows;
5. one adaptive chunk/saturation orchestration around fixed-shape compiled kernels.

The current Diffrax and native routes are not allowed to remain duplicate half-supported products. The native route is the default owner because it already carries the advanced-collision, sharding, checkpointed-adjoint, and production runtime seams. Use Diffrax as a temporary independent integration oracle during consolidation, then remove its public API, configuration switches, tests, source modules, and dependency unless a unique promoted use survives the audit.

Every promoted physics option must work on the explicit and/or IMEX owner that claims it. Unsupported combinations fail in case validation before compilation.

### 11.5 Parallelization and supported hardware

GKX 3.0 requires:

- a validated CPU route;
- a validated NVIDIA GPU route;
- consistent numerical behavior and supported precision policies on both.

Promote parallel work in this order:

1. independent `ky`, radius, field-line, parameter, seed, and UQ batches;
2. species decomposition;
3. Hermite decomposition with the required halo;
4. combined species-Hermite meshes;
5. only then consider perpendicular or parallel domain decomposition.

Acceptance requires:

- serial identity at validation precision;
- no unsupported collision or diagnostic substitution;
- an HLO and communication census;
- physical transport-window identity;
- peak-memory improvement or measured throughput improvement;
- a prospectively fixed scaling target.

Multi-GPU execution is valuable but is not a GKX 3.0 release blocker. Promote it in 3.x only when it improves representative time-to-solution or memory capacity. AMD GPU and TPU execution may work through JAX, but GKX makes no support, CI, performance, or compatibility promise for them at this stage.

### 11.6 Comparative performance targets

Use GX as the primary same-formulation performance reference and GENE/stella for selected method comparisons.

Phase 0 should measure a representative matrix before fixing numeric targets. Proposed progression:

- **GKX 3 alpha:** no more than 2x GX warm time to matched accuracy on core cases;
- **GKX 3 beta:** parity or a documented advantage in repeated prepared solves, memory, differentiation, or CPU portability;
- **GKX 3 stable:** no more than 10 percent unexplained performance regression from the beta baseline, with competitive time to a converged physical answer.

Compare time to physical error, not only time per step.

## 12. Physics capability roadmap

### 12.1 Core model contract

The governing-model documentation and code must clearly specify:

- local Maxwellian-background delta-f ordering;
- normalization and reference units;
- gyrocenter state convention;
- adiabatic and kinetic species responses;
- electrostatic and electromagnetic field equations;
- geometry coefficients and boundary conditions;
- nonlinear bracket and dealiasing;
- linearized collisions, sources, sinks, and artificial dissipation;
- diagnostics and flux definitions;
- free-energy balance;
- model limitations.

Every runtime term has one equation, one implementation owner, one configuration key, and at least one mathematics or physics test.

### 12.2 Required GKX 3.0 stable capabilities

- linear initial-value evolution;
- dominant and selected eigenmodes;
- nonlinear turbulence and restart;
- electrostatic and electromagnetic fluctuations;
- adiabatic electrons;
- kinetic electrons;
- arbitrary kinetic ion species, including trace impurities;
- particle and heat fluxes by species and field channel;
- analytic, Miller, imported, and live VMEX tokamak/stellarator geometry;
- linked/generalized twist-and-shift and a documented low-shear option;
- a conserving model collision operator;
- original and improved Sugama at their promoted scope;
- linearized Coulomb at its promoted scope;
- artificial hypercollision/hyperdiffusion with convergence guidance;
- saturation-aware nonlinear execution and uncertainty-aware postprocessing;
- one promoted quasilinear model with an explicit domain and limitations;
- differentiable linear and finite-window nonlinear objectives.

**Conditional 3.0 item: equilibrium ExB shear.** Phase 0 must audit the existing implementation. Include it in 3.0 only if one bounded implementation/validation sequence can close the shearing-coordinate convention, remap conservation, timestep constraints, linear benchmarks, and nonlinear transport gates without destabilizing the core rewrite. Otherwise schedule it as the first 3.1 physics feature.

### 12.3 GKX 3.x follow-on capabilities

- equilibrium ExB shear, if the Phase-0 audit does not admit it to 3.0;
- rotation and parallel-flow shear;
- validated momentum flux and momentum transport;
- multi-scale ion/electron calculations if a viable algorithm and resource model are established;
- adaptive Hermite-Laguerre resolution;
- advanced reduced-electron models for kinetic-electron efficiency;
- broader finite-beta stellarator validation;
- transport-profile coupling;
- multi-GPU production paths after measured value is demonstrated.

## 13. Collision and closure program

### 13.1 Collision ordering for a local delta-f code

GKX evolves a perturbation about a prescribed Maxwellian background,

\[
f_s = F_{0s} + \delta f_s .
\]

For the bilinear Landau operator,

\[
C[f,f]
=
C[F_0,F_0]
+
C[F_0,\delta f]
+
C[\delta f,F_0]
+
C[\delta f,\delta f].
\]

The equilibrium term vanishes for a Maxwellian. The two cross terms form the linearized Landau operator. The final term is quadratic in the perturbation and is beyond the standard delta-f ordering retained by GKX.

Therefore:

- a nonlinear turbulence simulation does **not** require a nonlinear collision operator;
- the nonlinear ExB bracket and the linearized collision operator are entirely consistent in delta-f gyrokinetics;
- the production GKX collision target is the most accurate, scalable **linearized** gyrokinetic Coulomb/Sugama hierarchy;
- `C[delta f, delta f]` is not a GKX 3.x feature.

A nonlinear full Coulomb operator becomes relevant only after an explicit scope change to full-f or strongly non-Maxwellian physics, such as evolving backgrounds, strong tails/runaways, or edge/SOL distributions far from a Maxwellian. The collision protocol may remain extensible, but no source, test, documentation, or schedule budget is reserved for that implementation in this roadmap.

### 13.2 Accurate names and evidence levels

Use explicit labels:

- conserving Lenard-Bernstein/Dougherty-like model;
- drift-kinetic original Sugama;
- drift-kinetic improved Sugama;
- linearized drift-kinetic Coulomb;
- linearized finite-perpendicular-wavelength gyrokinetic Coulomb.

Never label the present linearized operator simply “nonlinear Coulomb” or imply that nonlinear time evolution changes the collision operator's mathematical order.

Separate five levels of evidence:

1. coefficient and algebra validation;
2. runtime-kernel validation;
3. integrated linear gyrokinetic validation;
4. nonlinear turbulence sensitivity to operator choice;
5. production performance and convergence.

### 13.3 Collision architecture

One collision protocol receives a typed context containing:

- evolved gyrocenter perturbation;
- field-coupled response where required;
- solved fields;
- species parameters;
- local perpendicular wavelength and geometry;
- normalization and collision frequency;
- moment layout.

The protocol must work through the same explicit, IMEX, differentiation, and promoted sharding routes or fail at case validation before compilation.

The production implementation must avoid materializing a dense collision matrix at every spatial point. Candidate designs include:

- matrix-free analytic contractions;
- sparse moment coupling;
- factored test-particle and field-particle pieces;
- low-rank invariant-restoring corrections;
- separable interpolation in target/source Bessel arguments;
- fused batched application on CPU and GPU;
- implicit solves that exploit the same structure.

### 13.4 Collision phases

**C0 - semantic consolidation**

- unify collision selection and normalization;
- remove duplicate diagonal/custom switches;
- document which state each operator acts on;
- make unsupported solver combinations fail in preflight;
- align TOML, Python API, output metadata, and documentation.

**C1 - arbitrary-order drift-kinetic hierarchy**

- generate or apply original Sugama, improved Sugama, and Coulomb at arbitrary retained Hermite-Laguerre order;
- remove fixed eight-moment runtime restrictions;
- validate pair frequencies, signs, invariants, conductivity, and convergence;
- implement a matrix-free or structured application.

**C2 - finite-wavelength like-species linearized Coulomb**

- retain test, field, and polarization terms;
- support production moment resolutions;
- replace full spatial dense-table materialization with separable, low-rank, or on-the-fly contraction;
- validate the drift-kinetic limit, finite-Larmor classical diffusion, ITG stabilization, and zonal damping;
- add implicit treatment for collisionally stiff cases.

**C3 - multispecies finite-wavelength linearized Coulomb**

- support independent target/source Larmor radii, mass ratios, temperature ratios, charges, and directed frequencies;
- conserve each species particle number and total momentum and energy;
- validate electron-ion, ion-ion, impurity, equal-species, and disparate-temperature limits;
- compare locally with at least one independent implementation, without making its raw output a permanent GKX oracle.

**Scope-change watch item - nonlinear full Coulomb**

Track relevant algorithms and literature, including tensor-free methods, only for possible reuse in linearized matrix-free contractions. Do not implement the bilinear full-f hierarchy unless the scientific scope and state model are formally changed.

### 13.5 Collision acceptance tests

- Maxwellian null space;
- particle, momentum, and energy conservation;
- H-theorem or discrete dissipativity at the declared level;
- Onsager/self-adjointness where applicable;
- exact published low-order coefficients;
- `k_perp -> 0` reduction;
- Spitzer-Harm conductivity;
- Braginskii/Pfirsch-Schluter limits;
- multispecies temperature and flow relaxation;
- collisional zonal-flow damping;
- finite-wavelength ITG scan without the spurious short-wave branch;
- TEM and electromagnetic sensitivity where literature comparisons exist;
- nonlinear heat-flux comparison among model, Sugama, improved Sugama, and Coulomb;
- moment, wavelength-grid, timestep, and factorization convergence;
- JIT, JVP/VJP, CPU/NVIDIA parity, and memory scaling.

### 13.6 Closure program

Keep physical collisions distinct from numerical closure.

Promoted closure interface candidates:

- hard truncation, for reference only;
- hypercollision/hyperdiffusion with resolution-aware coefficients;
- outgoing Hermite-flux or phase-mixing closure;
- generalized Hammett-Perkins/Landau-fluid closure at low moment order;
- high-collisionality Chapman-Enskog closure;
- adaptive moment refinement based on tail energy or free-energy flux.

A closure is promoted only if it:

- preserves the declared low moments and free-energy behavior;
- reduces recurrence without hiding under-resolution;
- converges to the unclosed high-resolution hierarchy;
- improves time to a fixed error on representative linear and nonlinear cases;
- behaves continuously as collisionality and moment count vary.

## 14. Quasilinear model program

### 14.1 Goal and claim tiers

GKX should find and ship the **best supported quasilinear model**, not stop automatically at the simplest screening proxy and not wait for impossible universal accuracy.

A model may be promoted when it is useful, reproducible, and honest about its limits. Every released model receives one claim tier:

- **Tier Q1 - ranking/screening:** reliable ordering and mode attribution in a declared domain;
- **Tier Q2 - calibrated flux:** quantitative particle/heat/momentum fluxes with uncertainty in a declared domain;
- **Tier Q3 - transport coupling:** stable profile-evolution use with demonstrated robustness and uncertainty propagation.

The default model is the highest-scoring model that passes its declared tier. It may have known failures outside that domain. Those failures must be visible, and the runtime should emit an out-of-domain or low-confidence diagnostic rather than silently extrapolate.

### 14.2 Architecture

Separate four components:

1. **linear response:** eigenvalues, eigenfunctions, physical particle/heat/momentum flux weights, and field-channel decomposition;
2. **saturation model:** an explicit named model with parameters and physical assumptions;
3. **calibration and uncertainty:** training/validation/holdout datasets, fitted parameters, prediction intervals, and calibration diagnostics;
4. **model card:** version, domain, required physics, expected errors, known failures, and refusal conditions.

The saturation model is a protocol, not a hidden scalar constant.

### 14.3 Candidate model families

Evaluate rather than assume:

- mixing-length forms based on `gamma/k_perp^2`;
- mode-structure-weighted effective `k_perp`;
- multi-mode spectral envelopes;
- zonal-flow or secondary-instability-informed suppression;
- geometry-, shear-, or collisionality-aware saturation parameters;
- separate ion-scale and electron-scale models;
- separate tokamak and stellarator calibration when one universal model loses predictive value;
- ensembles or model averaging when uncertainty is better represented by several physically distinct closures.

Use TGLF, QuaLiKiz, and related theory as references, not code to copy.

### 14.4 Dataset design

The GKX-owned dataset should include:

- tokamak circular and Miller cases;
- kinetic-electron ITG/TEM;
- electromagnetic KBM cases;
- W7-X, HSX, QA, QH, and QI configurations;
- several radii and field lines;
- profile-gradient, magnetic-shear, beta, and collisionality scans;
- multiple nonlinear seeds and autocorrelation-aware uncertainty;
- resolution and timestep metadata.

External GX, GENE, stella, or CGYRO calculations may be run locally as independent checks. Their raw outputs are not part of the repository or public archive and are not permanent test oracles.

Split by equilibrium or device family, not random rows, so holdouts test generalization.

### 14.5 Model selection and promotion

Freeze the dataset split, metrics, and weights before fitting the final candidates. Score each candidate on:

- rank correlation and pairwise ordering;
- median and tail relative error where absolute flux is claimed;
- interval calibration;
- physical sign and symmetry constraints;
- robustness to linear-resolution refinement;
- performance and compilation cost;
- out-of-domain detection;
- stability when coupled to an optimizer or transport iteration.

A Q1 model may be promoted without Q2 accuracy. A Q2 model may be promoted for a named subset of devices, regimes, or physics even when it fails elsewhere. There is no requirement that one model cover every tokamak and stellarator regime.

Suggested initial gates, to be frozen in Phase 8:

- Q1: held-out rank correlation and pair-order accuracy sufficient for design screening, with calibrated confidence flags;
- Q2: held-out median error and bias within a declared tolerance, with interval coverage and no systematic device-family failure;
- Q3: stable coupled iterations and profile predictions within the propagated nonlinear uncertainty.

Stress cases remain in the model card. They may be classified as out of domain but may not be silently removed after results are seen.

## 15. Differentiation, optimization, and uncertainty

### 15.1 Derivative contract

Document each derivative as one of:

- exact derivative of the implemented discrete function;
- implicit derivative of a converged algebraic solve;
- finite-window trajectory derivative conditional on a fixed initial state;
- statistical or ensemble sensitivity estimate;
- finite-difference validation or fallback.

Never describe a finite-window nonlinear derivative as the derivative of the infinite-time turbulent invariant measure.

### 15.2 Linear derivatives

Promote:

- eigenvalue and eigenvector derivatives with branch tracking;
- left/right eigenvector conditioning diagnostics;
- mode-crossing and degeneracy handling;
- AD/FD and adjoint/forward consistency;
- cost versus parameter count;
- geometry and profile derivatives.

### 15.3 Nonlinear derivatives

The production finite-window method should include:

- block checkpointing with measured memory/runtime tradeoff;
- saturation-state detachment stated explicitly;
- measured autocorrelation and trajectory-divergence horizons;
- multiple finite-difference step sizes;
- gradient ensembles over independent saturated states and separated windows;
- covariance, direction cosine, and signal-to-noise diagnostics;
- line-search validation on held-out perturbations;
- final independent long nonlinear audits.

Compare under a matched GPU-hour budget with:

- central finite differences;
- SPSA;
- common-random-number ensemble finite differences;
- a shadowing or least-squares method if a numerically viable formulation is developed.

### 15.4 Flagship optimization result

The strongest near-term scientific target is an end-to-end stellarator campaign with:

- 50-200 boundary or equilibrium degrees of freedom;
- several radii;
- several field lines;
- several `ky` values or a justified spectral aggregate;
- aspect, iota, quasisymmetry/quasi-isodynamicity, MHD, and engineering constraints;
- a training set and held-out surfaces/field lines;
- uncertainty-aware nonlinear gradient aggregation;
- a comparison against SPSA or another noisy-objective method;
- matched, replicated, long post-transient baseline/candidate nonlinear runs;
- an independent GX, GENE, or stella audit for at least one headline result.

This is the result that can make GKX category-defining. A single-surface, single-field-line, eight-control result remains proof of principle.

## 16. Coupling and ecosystem integration

### 16.1 VMEX and Boozer ownership

The code audit establishes this ownership boundary:

- **VMEX owns equilibrium physics and live-state field-line geometry.** Its field-line adapter already computes the full GKX geometry contract: `bmag`, `gradpar`, metric coefficients, grad-B and curvature drifts, pressure corrections, `bgrad`, `grho`, `q`, `s_hat`, reference scales, equal-arc mapping, finite-beta terms, and asymmetric geometry.
- **`booz_xform_jax` owns the Boozer coordinate transformation, Boozer spectra, and `boozmn` I/O.**
- **GKX owns the generic flux-tube geometry contract, normalization validation, boundary/topology policy, and consumption of those arrays by gyrokinetics.**

A local gyrokinetic flux tube does not require GKX to recompute a Boozer transform. A consistent straight-field-line coordinate with the required metric and drift coefficients is sufficient. Boozer coordinates remain valuable for quasisymmetry, quasi-isodynamicity, omnigenity, plotting, and other magnetic-configuration diagnostics, but those are not reasons to duplicate the transform inside GKX.

#### Target live-state interface

The canonical differentiable path is:

```text
VMEX state/runtime
    -> vmex.gk_fieldline_geometry(...)
    -> gkx.FluxTubeGeometry.from_mapping(...)
    -> gkx.prepare/solve/objective
```

GKX should provide only a thin convenience adapter such as:

```python
geometry = gkx.geometry.from_vmex(state, runtime, surface=0.6, alpha=0.0)
```

The adapter calls the public VMEX array API and validates the GKX contract. It does not contain VMEC spectral geometry, Boozer tables, radial derivatives, or drift formulas.

#### Standard WOUT path

Reading a standard VMEC-compatible WOUT without solving an equilibrium remains a core user requirement.

Preferred resolution:

1. add one targeted public VMEX function, `gk_fieldline_geometry_from_wout(...)`, that returns the same mapping from any compatible WOUT;
2. keep one small read-only GKX imported-EIK/WOUT adapter until that API is released and validated;
3. then decide whether the remaining file adapter belongs in GKX, VMEX, or Pyrokinetics.

Do not require a user to reconstruct and re-converge a VMEX equilibrium merely to run GKX from an existing WOUT.

#### GKX deletion candidates

After value, derivative, finite-beta, asymmetric, and normalization parity gates pass, retire the duplicated live-state implementation represented by:

- `geometry/booz_xform_bridge.py`, especially its synthetic smooth metric/drift closure;
- `geometry/vmec_boozer_core.py`;
- `geometry/vmec_boozer_constants.py`;
- `geometry/vmec_boozer_derivatives.py`;
- duplicated VMEC drift, tensor, state-control, sensitivity, field-line-sampling, and report modules;
- VMEC/Boozer objective/report modules whose only role is to reconstruct geometry or campaign admission.

Do not delete generic GKX objective assembly, the flux-tube contract, analytic/Miller geometry, or the temporary standard-file adapter.

The current geometry inventory indicates that this boundary can remove more than eight thousand lines of direct geometry duplication before counting VMEC/Boozer objective and report code.

#### Acceptance gates

- VMEX and GKX mappings agree under the final normalization contract;
- vacuum and finite-beta cases pass;
- stellarator-symmetric and `LASYM` cases pass;
- equal-arc and non-equal-arc conventions are explicit;
- geometry values and VJPs pass against finite differences;
- imported WOUT and live-state routes agree where they represent the same equilibrium;
- no Boozer-specific dependency is imported during an analytic, Miller, or ordinary imported-geometry run;
- failures identify whether they came from equilibrium, geometry conversion, or gyrokinetics.

### 16.2 Pyrokinetics

Contribute a GKX plugin or adapter supporting:

- GKX TOML read/write;
- local geometry and species conversion;
- normalization conversion;
- linear and nonlinear output reading;
- fields, fluxes, eigenvalues, eigenfunctions, and spectra;
- round-trip tests on canonical cases.

This is a higher-return community feature than implementing more private converters in GKX.

### 16.3 Transport solvers

Define a stable adapter returning fluxes and uncertainties as functions of local profiles and geometry. Target integrations, in priority order:

1. T3D or another stellarator-capable transport solver;
2. TORAX for differentiable tokamak profile workflows;
3. TGYRO/PORTALS-style steady-profile workflows;
4. broader IMAS or Fusion Data Platform interfaces as community demand appears.

The adapter must support parallel independent radii and persistent prepared simulations.

### 16.4 Optimization codes

Keep the solver optimizer-neutral. Provide examples or adapters for:

- VMEX;
- SciPy optimizers;
- JAXopt or Optax only as optional examples;
- DESC/SIMSOPT geometry exchange where practical;
- Bayesian/UQ workflows through stable array inputs and outputs.

Do not embed an optimizer framework in the scientific core.

## 17. Examples

Keep 8-12 canonical user examples. Suggested sequence:

```text
examples/
  README.md
  01_linear_itg.py
  01_linear_itg.toml
  02_kinetic_electron_itg.py
  03_electromagnetic_kbm.py
  04_nonlinear_itg.py
  04_nonlinear_itg.toml
  05_stellarator_w7x.py
  06_collision_models.py
  07_quasilinear_scan.py
  08_restart_and_plot.py
  09_vmex_geometry.py
  10_stellarator_optimization.py
  11_parameter_ensemble.py
  12_multi_gpu.py
  data/
```

Delete `runtime_` prefixes and historical taxonomy such as `theory_and_demos` and `utilities` when their contents can be placed in the numbered path or documentation.

Each Python example must:

- be a direct top-level script, with no `main()` or `if __name__ == "__main__"`;
- put user parameters near the top;
- show the imports and data structures users need;
- print what is being built and run;
- run the simulation through the public API;
- print key physical results;
- save a result;
- make and save a plot;
- state expected CPU/GPU runtime and optional dependencies;
- avoid private local paths and large bundled inputs;
- distinguish tutorial, benchmark, and long research settings.

The TOML and Python versions should describe the same case where both are provided.

## 18. Documentation rewrite

### 18.1 Information architecture

Use the Diataxis separation:

```text
docs/
  index.md

  tutorials/
    first-linear-run.md
    first-nonlinear-run.md
    first-stellarator-run.md
    first-parameter-scan.md

  how-to/
    choose-resolution.md
    run-kinetic-electrons.md
    run-electromagnetic.md
    restart.md
    compare-collision-models.md
    run-on-gpu.md
    run-on-multiple-gpus.md
    couple-vmex.md
    optimize.md
    troubleshoot.md

  reference/
    input-schema.md
    output-schema.md
    cli.md
    python-api.md
    normalization.md
    compatibility.md
    capability-matrix.md

  explanation/
    gyrokinetic-model.md
    geometry.md
    hermite-laguerre.md
    fields-and-diagnostics.md
    time-integration.md
    collisions-and-closures.md
    quasilinear-model.md
    nonlinear-statistics.md
    autodiff.md
    parallelization.md

  developer/
    architecture.md
    provenance.md
    testing.md
    benchmarking.md
    release.md
```

Remove `research_grade_program`, `research_grade_plan`, release-readiness dashboards, internal figure inventories, and campaign status from the published user documentation. The planning branch and benchmark archive own those records.

### 18.2 Tooling and theme

- retain Sphinx and Read the Docs;
- adopt the PyData Sphinx Theme for a large scientific documentation set;
- optionally adopt MyST Markdown for lower-friction contributions while retaining Sphinx equations, references, and API directives;
- use accessible high-contrast code highlighting;
- use one restrained project color and no decorative dashboard styling;
- enable link checking and warnings-as-errors;
- test code snippets and selected tutorials in CI;
- keep API pages generated from the small public surface, not every internal module.

### 18.3 Content requirements

The documentation must include:

- complete governing equations and normalization;
- term-by-term mapping to source modules;
- numerical algorithms and stability restrictions;
- input and output examples;
- resolution and convergence guidance;
- validation cases with citations and reproducible commands;
- performance interpretation, including compilation;
- limitations and unsupported combinations;
- plots and compressed movies;
- source provenance and contributor attribution;
- applications to tokamaks, stellarators, optimization, and transport coupling.

The original GX notes and relevant equations should be rewritten into a coherent GKX explanation, not copied as disconnected implementation notes.

### 18.4 Writing standard

Documentation prose must:

- lead with the user’s question or the scientific point;
- use active voice;
- use concrete names, numbers, equations, and examples;
- avoid vague importance claims and marketing language;
- avoid repeated binary contrasts, faux-insight openings, synonym cycling, and generic summary paragraphs;
- distinguish demonstrated, validated, experimental, and planned capabilities;
- use “to our knowledge” only after a current literature check;
- receive a human line-by-line review before release.

### 18.5 README target

README ceiling: approximately 150-200 lines.

Order:

1. badges;
2. one-sentence purpose;
3. one strong image or compact animation;
4. install;
5. first run;
6. first Python example;
7. supported capabilities with precise qualifiers;
8. validation/performance summary with links to docs;
9. documentation, citation, contributing, and license.

No derivations, release-governance prose, full benchmark tables, or research roadmaps in README.

## 19. Benchmark and research-data organization

### 19.1 Repository boundaries

- `examples/`: user learning and representative workflows;
- `benchmarks/`: small self-contained inputs, analytic/literature references, local comparison drivers, and compact GKX-owned summaries;
- `tests/`: self-contained assertions that run in CI;
- `scripts/`: no more than 8-12 maintained developer commands;
- local or explicitly approved GKX release assets: raw GKX nonlinear runs, large GKX tables, movies, and profiler traces;
- maintainers' local comparison workspace: external-code binaries, inputs requiring private installations, and raw external-code outputs;
- `plan/research-grade-roadmap`: plan, decision log, audit snapshots, and work log.

Delete `tools/` after its maintained functionality is consolidated into `benchmarks/`, `scripts/`, CI, or local campaign infrastructure. The existing architecture target of zero Python files under `tools` should be completed.

### 19.2 Reproducible validation and release bundle

One maintained command should:

1. verify the GKX environment and input checksums;
2. run or replay each GKX validation calculation;
3. verify resolution and statistical gates;
4. produce reviewed tables and figures;
5. write a machine-readable manifest with the GKX code and data versions.

External-code comparisons are a separate local command. It records comparator provenance and emits a local report, but no raw comparator output is committed or published by default.

Normal users must not download validation campaigns or external-code data to install GKX.

## 20. Phased execution plan

Each phase is complete only when its exit gates pass. Dates may be assigned after maintainer review and resource allocation.

### Phase 0 - Freeze the 1.8.2 baseline and rewrite the ground-truth plan

**Goal:** establish a trustworthy starting point before moving code.

Tasks:

- replace the front of the existing planning-branch `plan.md` with this approved charter;
- move the current long audit log under `plan/archive/` or preserve it in git history;
- record source, test, tool, documentation, media, wheel, and public-API counts;
- freeze self-contained numerical fingerprints for selected linear, nonlinear, geometry, collision, restart, differentiation, and optimization cases;
- record cold/warm CPU and NVIDIA GPU runtime, memory, compile count, and transfer metrics;
- inventory every public name and downstream use;
- inventory every dependency and actual import;
- inventory GX-derived functions and begin `PROVENANCE.md`;
- freeze current output schemas and compatibility obligations;
- audit whether equilibrium ExB shear is close enough for the 3.0 gate;
- freeze the VMEX/GKX/`booz_xform_jax` ownership matrix and deletion candidates;
- freeze the native explicit and native IMEX ownership decision and the Diffrax migration tests;
- define the local-only external-code comparison protocol;
- set actual required protected-branch CI contexts.

Exit gates:

- baseline can be reproduced from a clean wheel and source install;
- plan contains current metrics and commands;
- no architecture-changing feature PR enters until the baseline gates pass;
- every future PR can state which frozen behavior it preserves or intentionally changes;
- no permanent test requires an external gyrokinetic executable or raw external output.

### Phase 1 - Define GKX 3 product surface

**Goal:** make one stable API, CLI, case schema, and result schema.

Tasks:

- introduce immutable `Case` and typed submodels;
- introduce `load`, `solve`, `scan`, `plot`, and `prepare` contracts;
- define `LinearResult`, `NonlinearResult`, and `ScanResult`;
- define versioned TOML and NetCDF schemas;
- implement new CLI command family with Rich output;
- reduce top-level API to no more than 30 names;
- mark GKX 1.x public adapters with one declared deprecation window;
- remove report and campaign utilities from top-level imports;
- add migration documentation and a configuration converter where needed.

Exit gates:

- canonical examples use only the new public surface;
- wheel smoke test exercises linear, nonlinear dry-run/check, and plot paths;
- old promoted inputs either work through an adapter or fail with a migration message;
- public API documentation fits on one navigable reference page.

### Phase 2 - Consolidate the scientific core

**Goal:** remove duplicate pathways while preserving physics.

Work order:

1. case/config and normalization;
2. grids and Hermite-Laguerre basis;
3. field solve and physical moments;
4. linear term assembly;
5. nonlinear bracket and projection;
6. native explicit and native IMEX integration;
7. geometry ownership migration;
8. collisions and closures;
9. diagnostics and result writing;
10. supported parallel execution.

For each tranche:

- identify one owner module;
- move or rewrite functions into that owner;
- preserve numerical fingerprints;
- delete old wrappers and existence-only tests;
- update docs, provenance, and import adapters;
- measure source files and lines before and after.

Geometry tranche:

- introduce the thin `from_vmex` mapping adapter;
- add or request the targeted VMEX WOUT-to-field-line API;
- retain one small standard-file path until replacement parity is complete;
- delete the synthetic Boozer bridge and duplicate live-state VMEC/Boozer geometry;
- convert VMEC-specific objectives into generic objectives over the flux-tube contract.

Integrator tranche:

- promote native explicit RK;
- promote one native IMEX owner;
- use Diffrax only for migration parity;
- remove Diffrax configuration, public exports, source, tests, and dependency after the owner gates pass.

Intermediate gate:

- no more than 75 source files and 65,000 source lines.

Final gate:

- no more than 45 source files and 45,000 source lines;
- no duplicate solver route for the same promoted calculation;
- no duplicate VMEC/Boozer geometry implementation;
- every runtime physics switch works through the canonical case and result paths.

### Phase 3 - Consolidate tests and evidence

**Goal:** maximize detection power while cutting test machinery.

Tasks:

- reorganize tests into the target layout;
- combine configuration matrices with parametrization;
- replace artifact-only assertions with solver-backed tests where affordable;
- move slow reproducible comparisons to benchmark/release tiers;
- define the evidence level of every promoted claim;
- add manufactured-solution and observed-order tests where missing;
- add prospectively defined nonlinear statistical policies;
- delete tests for retired wrappers and tools.

Exit gates:

- at most 30 test files and 35,000 test lines;
- package coverage at least 95 percent;
- every public name has direct behavioral coverage;
- every promoted physics lane has E3-E6 evidence appropriate to the claim;
- mutation or fault-injection spot checks show that major physics gates detect planted errors.

### Phase 4 - Rewrite examples and documentation

**Goal:** make a new user productive without reading internal architecture.

Tasks:

- replace the current examples hierarchy with the numbered canonical set;
- rewrite README to the target order and length;
- rebuild documentation with the Diataxis structure;
- adopt the chosen theme and accessible styles;
- rewrite equations and methods from first principles;
- add tested snippets, plots, compressed movies, and clear runtime expectations;
- remove research status and internal campaign governance from published docs;
- complete provenance and contributor attribution.

Exit gates:

- a clean user can install and finish the first linear tutorial in under ten minutes on CPU;
- the first bounded nonlinear tutorial visibly progresses and produces one result bundle and figure set;
- all canonical examples run in CI or have a smaller CI mode plus a checked full-run artifact;
- Sphinx builds with warnings as errors and link checking;
- documentation media remains within its budget;
- human no-slop review complete.

### Phase 5 - Single-device performance and memory

**Goal:** make the canonical solver small, prepared, and competitive before expanding parallelism.

Tasks:

- create prepared simulation objects with stable shapes;
- remove repeated tracing and compilation;
- configure persistent cache in the CLI;
- profile RHS, field solve, bracket, diagnostics, output sampling, and collision paths;
- fix pathological fusion, layout, transfer, and allocation behavior;
- add safe buffer donation;
- complete the native explicit/native IMEX consolidation and delete Diffrax after parity;
- improve IMEX treatment for kinetic electrons and collisions;
- record time-to-error against locally maintained GX comparisons for the core matrix.

Exit gates:

- no unexplained regression over the frozen baseline;
- cold, warm, prepared, and memory metrics are reproducible;
- repeated scans and optimization calls do not recompile fixed topology;
- core NVIDIA GPU cases are within the Phase-0-agreed range of GX at matched accuracy or have a documented GKX advantage;
- CPU remains a supported, validated route;
- no external comparator output is required to run the performance regression suite.

### Phase 6 - Physics completion and broad validation

**Goal:** close the stable local-model capability matrix.

Subphases:

- 6A: electrostatic adiabatic-electron linear/nonlinear validation;
- 6B: kinetic-electron ITG, TEM, and ETG;
- 6C: electromagnetic KBM, KAW, and microtearing lanes;
- 6D: stellarator kinetic-electron and electromagnetic validation;
- 6E: equilibrium ExB shear only if the Phase-0 readiness audit admits it to 3.0;
- 6F: zonal flows, sources/sinks, free energy, and long-window statistics.

Rotation, parallel-flow shear, momentum flux, and momentum transport are GKX 3.x features rather than 3.0 blockers.

Exit gates:

- the capability matrix names exactly which cases and observables are validated;
- permanent tests are self-contained and anchored in mathematics, numerics, analytic physics, or stable literature values;
- local matched comparisons cover at least two established-code families, with comparator versions and residuals recorded locally;
- no broad claim rests only on a reduced or startup-window artifact;
- unsupported combinations fail in preflight.

### Phase 7 - Advanced linearized collisions and closures

**Goal:** turn current algebraic work into scalable production delta-f collision physics.

Tasks follow C0-C3 and the closure program above. The nonlinear full-f Coulomb hierarchy is not part of this phase.

Exit gates for GKX 3.0 stable:

- arbitrary-order drift-kinetic original/improved Sugama and Coulomb, or an explicitly narrower stable scope approved by Rogerio Jorge;
- a production finite-wavelength like-species linearized operator at useful resolution without spatial dense-matrix explosion;
- conductivity, ITG, zonal, conservation, convergence, and nonlinear operator-choice gates;
- native explicit/IMEX and differentiation support;
- CPU/NVIDIA performance and memory documented.

Multispecies finite-wavelength Coulomb may be a 3.x milestone if it would delay the stable core excessively.

### Phase 8 - Quasilinear model

**Goal:** select and ship the best supported model with a precise model card.

Tasks:

- freeze the GKX-owned dataset and holdout policy;
- implement physical eigenmode flux weights;
- define the saturation-model protocol;
- evaluate candidate model families and ensembles;
- freeze a multi-objective selection score;
- quantify uncertainty and domain of applicability;
- compare against nonlinear GKX and local external checks;
- integrate the selected model with scans, VMEX objectives, and result metadata.

Exit gates:

- one model is selected as the default for its declared Q1, Q2, or Q3 tier;
- untouched holdouts pass the frozen tier-specific gates;
- known failures and out-of-domain regimes remain visible;
- the runtime emits confidence/domain information;
- model version, features, calibration provenance, and uncertainty are stored in every result;
- no wording implies universal accuracy outside the model card.

### Phase 9 - Differentiable optimization and coupling

**Goal:** demonstrate the capability that distinguishes GKX.

Tasks:

- complete the robust linear derivative portfolio;
- implement nonlinear gradient ensembles and uncertainty diagnostics;
- compare AD, finite differences, and SPSA under matched computational cost;
- run the high-dimensional, multi-surface, multi-field-line VMEX campaign;
- use the exact VMEX field-line mapping rather than duplicated GKX Boozer geometry;
- perform independent long nonlinear holdouts;
- run local cross-code checks without archiving comparator outputs;
- add a Pyrokinetics adapter;
- add a transport-solver adapter and one profile-evolution demonstration.

Exit gates:

- gradient cost versus parameter count and optimization progress are recorded as GKX-owned artifacts;
- the optimized design improves prospectively defined held-out nonlinear transport with statistical significance;
- geometric and MHD constraints pass;
- at least one local external calculation corroborates the final trend or transport change;
- transport adapter runs several radii in parallel through the stable API.

### Phase 10 - CPU/NVIDIA production and stable release

**Goal:** release a community-ready GKX 3.0.

Tasks:

- finish CPU and NVIDIA GPU performance/compatibility gates;
- promote multi-GPU paths only when they already meet measured value thresholds;
- finish packaging and conda-forge;
- publish compatibility, governance, and contribution policies;
- conduct an external beta with users outside the core team;
- resolve beta issues without expanding scope;
- regenerate all GKX-owned release evidence from the final candidate.

Exit gates:

- source, test, documentation, media, wheel, API, coverage, physics, performance, and release budgets pass;
- protected main requires all release-critical checks;
- clean CPU and NVIDIA GPU install instructions are independently reproduced;
- at least two external users complete linear and nonlinear workflows;
- no AMD GPU, TPU, or multi-GPU result is required for 3.0;
- no known P0/P1 correctness or documentation issue remains.

## 21. Pull-request decomposition

The modernization should be executed as a sequence of small reviewable PRs. The first candidate queue is:

1. rewrite the planning-branch `plan.md` and archive the old audit;
2. freeze 1.8.2 measurements, public API, dependency, and provenance inventories;
3. freeze external-comparison and self-contained-test policies;
4. create the VMEX/GKX/`booz_xform_jax` ownership and parity matrix;
5. introduce `Case`/`Result` contracts without moving kernels;
6. introduce the new CLI aliases and Rich progress;
7. add versioned TOML/NetCDF schemas and migration tests;
8. reduce top-level exports;
9. consolidate the field solve and moments;
10. consolidate the linear operator;
11. consolidate the nonlinear bracket and projection;
12. promote native explicit RK and delete its duplicate wrappers;
13. promote one native IMEX owner;
14. remove Diffrax after migration parity;
15. add the thin `from_vmex` geometry adapter;
16. add or request the targeted VMEX WOUT field-line API;
17. delete duplicate live-state VMEC/Boozer geometry and objectives;
18. consolidate the collision protocol;
19. consolidate diagnostics and result writing;
20. reorganize tests;
21. replace examples;
22. rewrite the documentation shell and theme;
23. complete single-device profiling and prepared simulations;
24. close physics lanes one at a time;
25. close quasilinear, advanced-collision, optimization, and coupling lanes with separate gates.

Never combine broad file movement, new physics, performance changes, and new scientific results in one PR.

## 22. Codex operating contract

Every Codex session working on this program must begin by reading `plan.md` on `plan/research-grade-roadmap` and the relevant source, tests, docs, and history.

### 22.1 Before editing

Codex must record in the current work-log entry:

- task and non-goals;
- baseline branch and commit;
- affected public behavior and scientific claims;
- files expected to be deleted, moved, or changed;
- acceptance tests and measurements;
- rollback condition.

### 22.2 During work

- Work on one small feature branch.
- Prefer deletion and consolidation over adding parallel abstractions.
- Do not copy a module merely to refactor it later.
- Preserve numerical fingerprints unless the task is an approved bug or model correction.
- A model change must cite equations and add a mathematics/physics gate.
- A performance change must include synchronized measurement and numerical identity.
- An AD change must include finite-difference and conditioning checks.
- A documentation claim must point to a tracked reproducible result.
- No private paths, generated raw data, caches, WOUTs, profiler traces, external-code binaries, or raw external-code outputs enter git.
- Keep examples on the public API.
- Update provenance when translating GX or literature code.

### 22.3 Before opening a PR

Codex must run the narrow tests, formatting, typing, architecture/size checks, docs if affected, and at least one direct user workflow.

The PR description must state:

- what user or scientific problem was fixed;
- what was deleted;
- equations or conventions affected;
- numerical results before and after;
- runtime and memory before and after if relevant;
- tests that would fail on the old defect;
- claim boundary and remaining limitations;
- exact reproduction commands.

Open one draft PR. Do not merge, force-push main, or broaden the PR after review without updating `plan.md`. Only GitHub user `rogeriojorge` gives final approval for and merges PRs into `main`.

### 22.4 After merge

Append to the work log:

- merge commit and PR;
- final measurements;
- changed baseline counts;
- decisions made;
- deferred issues;
- next unblocked task.

The work log is append-only. The plan body changes only through explicit maintainer decisions recorded in the decision log.

## 23. Ground-truth `plan.md` format

The file on the planning branch should use this stable order:

1. mission;
2. scope and non-goals;
3. current baseline table;
4. product and API contracts;
5. target architecture and budgets;
6. scientific capability matrix;
7. evidence and performance policies;
8. phased roadmap and exit gates;
9. active work board;
10. decision log;
11. risk register;
12. provenance and literature ledger;
13. reproduction commands;
14. append-only work log;
15. archived audit links.

The first 500 lines should be enough for a new human or agent to understand the program. Long campaign narratives belong in archived notes, not the governing plan.

## 24. Approved maintainer decisions

These decisions govern implementation unless Rogerio Jorge amends the plan.

| Decision | Approved rule |
|---|---|
| Scientific scope | GKX 3.x is explicitly local, Maxwellian-background delta-f, and flux-tube. |
| Compatibility | Broad and undocumented GKX 1.x internal imports may be removed. Preserve only promoted user workflows through a bounded migration layer. |
| Human input | TOML is the human-authored format. |
| Canonical output | Versioned NetCDF is the result/restart format; JSON is for summaries and machine interchange. |
| Python | Python 3.11 remains the minimum. |
| Core platforms | CPU and NVIDIA GPU are required. AMD GPU and TPU have no current support commitment. |
| Public API | No more than 30 top-level names. |
| Source budget | No more than 45 Python source files and 45,000 Python source lines. |
| Test budget | No more than 30 Python test files and 35,000 Python test lines, with at least 95 percent coverage. |
| Integrators | Native explicit RK and one native IMEX route become owners; duplicated Diffrax paths are removed after parity. |
| ExB shear | Include in 3.0 only if the readiness audit shows it can be closed in a bounded, validated sequence. |
| Rotation/momentum | Rotation, flow shear, momentum flux, and momentum transport are 3.x follow-ons. |
| Quasilinear | Select the best model and promote it at the highest passing claim tier, with explicit limitations and out-of-domain behavior. Universal accuracy is not required. |
| Collision ordering | Prioritize scalable arbitrary-order linearized Sugama/Coulomb. The bilinear `C[delta f, delta f]` operator is out of scope while GKX remains delta-f. |
| Geometry ownership | VMEX owns live equilibrium-to-field-line geometry; `booz_xform_jax` owns Boozer transforms; GKX owns the generic flux-tube contract and solver consumption. Delete duplicate GKX implementations after parity. |
| Flagship capability | High-dimensional, multi-surface nonlinear stellarator optimization is the first major end-to-end target. |
| External comparisons | Maintain external codes locally. Do not make raw external-code outputs public or permanent CI oracles. Convert findings into self-contained GKX tests. |
| Planning branch | Keep `plan/research-grade-roadmap` as the long-lived ground-truth and append-only work-log branch. |
| Main approval | GitHub user `rogeriojorge` is the final reviewer and sole merger into `main`. |
| Program scope | This roadmap is about the code. It does not prioritize or sequence publications. |

## 25. Risk register

### R1. Refactor changes physics silently

Mitigation: frozen fingerprints, term-by-term moves, independent comparisons, and no mixed refactor/physics PRs.

### R2. File-count targets create giant modules

Mitigation: 900-line hard ceiling, scientific ownership rules, and median-module budget.

### R3. Coverage falls while tests are consolidated

Mitigation: combine only after mapping each line and scientific claim to an owner; require detection-power tests, not count-only tests.

### R4. JAX compilation dominates user experience

Mitigation: prepared simulations, stable shapes, persistent cache, compile reporting, and cold/warm benchmarks.

### R5. Advanced collisions remain algebraically correct but unusably dense

Mitigation: make matrix-free/factorized complexity an entry condition for high-order production promotion.

### R6. Quasilinear calibration overfits a small internal dataset

Mitigation: device/equilibrium holdouts, external code data, frozen splits, uncertainty, and visible stress cases.

### R7. Short-window nonlinear gradients do not predict stationary transport

Mitigation: gradient ensembles, directional long-run holdouts, comparison with SPSA and ensemble finite differences, and precise derivative labels.

### R8. Optional integrations inflate core dependencies

Mitigation: adapters and extras, imported standard files, and dependency audit gates.

### R9. Documentation rewrite becomes another large parallel project

Mitigation: write pages only for stable product contracts, migrate one user journey at a time, delete superseded pages immediately.

### R10. Agents optimize metrics instead of the code

Mitigation: every budget is paired with physics, user, and performance acceptance gates; reject changes that game file counts, coverage, or dashboards.

### R11. Provenance cannot be reconstructed after translation

Mitigation: complete the GX/function ledger before deleting historical wrappers and require provenance in every translation PR.

### R12. Community adoption lags technical quality

Mitigation: Pyrokinetics, transport coupling, external beta users, stable schemas, concise tutorials, and responsive issue/PR governance.

### R13. External comparator drift becomes a hidden oracle

Mitigation: external runs are local, version-pinned, residual-checked, and diagnostic. Permanent CI is rebuilt from independent mathematics, numerics, analytic physics, and stable literature references.

### R14. VMEX integration creates circular ownership or dependency bloat

Mitigation: arrays flow one way through a documented flux-tube contract. VMEX does not import GKX in its geometry core, GKX does not reproduce VMEX geometry, and `booz_xform_jax` remains a Boozer-specific companion rather than a turbulence-core dependency.

## 26. Program-level success criteria

GKX 3.0 is complete when all of the following are true.

### Software

- no more than 45 source files and 45,000 source lines;
- no more than 30 test files and 35,000 test lines;
- at least 95 percent statement coverage;
- no more than 30 top-level public names;
- no production module above 900 lines;
- no Python files under `tools/`;
- clean wheel, source, and conda installs;
- repository below 15 MiB and wheel below 10 MiB.

### User experience

- install and first linear run in under ten minutes on a typical laptop;
- obvious CLI and Python paths;
- visible progress for every long run;
- one canonical result bundle and plotting command;
- 8-12 clear runnable examples;
- modern, organized, warning-free documentation.

### Scientific capability

- promoted linear and nonlinear electrostatic/electromagnetic tokamak and stellarator cases;
- adiabatic and kinetic electrons;
- arbitrary kinetic species at the stable supported scope;
- collision and closure capabilities labeled and validated precisely;
- one quasilinear default with a passing model card and declared claim tier;
- robust differentiation and optimization workflows;
- high-dimensional, multi-surface nonlinear stellarator optimization with held-out validation;
- locally reproducible external checks from at least two independent code families, without external raw-output test dependencies.

### Performance and platforms

- reproducible cold, warm, and prepared CPU/NVIDIA GPU metrics;
- competitive time to matched physical error against locally maintained GX comparisons for core cases;
- stable memory budgets;
- no uncontrolled recompilation in scans or optimization;
- no AMD GPU, TPU, or multi-GPU release requirement.

### Ecosystem and governance

- thin production VMEX coupling with no duplicate live-state geometry;
- Pyrokinetics adapter;
- one transport-solver integration;
- external beta users;
- versioned GKX-owned validation manifests;
- required protected-branch checks;
- final `main` approval and merge by `rogeriojorge`;
- clear contribution and provenance paths.

## 27. Literature and standards ledger

The plan should maintain complete bibliographic entries. Core starting references include:

### Gyrokinetic codes and algorithms

- Mandell et al., “GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design,” J. Plasma Phys. 90, 905900402 (2024), DOI 10.1017/S0022377824000631.
- Barnes, Parra, and Landreman, “stella: an operator-split, implicit-explicit delta-f gyrokinetic code for general magnetic field configurations,” J. Comput. Phys. 391, 365-380 (2019), DOI 10.1016/j.jcp.2019.01.025.
- Candy, Belli, and Bravenec, CGYRO reference, J. Comput. Phys. 324, 73-93 (2016).
- GENE 3 documentation and reference publications.
- GTC-P scalability publications.
- XGC documentation and whole-volume stellarator publications.
- GENE-X spectral and stellarator extensions, CPC 316, 109817 (2025) and CPC 324, 110138 (2026).

### Quasilinear and transport modeling

- Staebler et al., “Quasilinear theory and modelling of gyrokinetic turbulent transport in tokamaks,” Nucl. Fusion 64, 103001 (2024).
- Stephens et al., “Quasilinear gyrokinetic theory: a derivation of QuaLiKiz,” J. Plasma Phys. 87 (2021).
- Jorge et al., “Direct microstability optimization of stellarator devices,” Phys. Rev. E 110, 035201 (2024).
- Kim et al., “Optimization of nonlinear turbulence in stellarators,” J. Plasma Phys. 90, 905900210 (2024).
- Rodriguez-Fernandez et al., PORTALS profile-prediction work, arXiv:2312.12610 and subsequent publication.

### Collisions and moment methods

- Abel et al., “Linearized model Fokker-Planck collision operators for gyrokinetic simulations. I. Theory,” Phys. Plasmas 15, 122509 (2008).
- Jorge, Ricci, and Loureiro, arXiv:1709.01411.
- Jorge, Frei, and Ricci, “Nonlinear Gyrokinetic Coulomb Collision Operator,” J. Plasma Phys. 85, 905850604 (2019), arXiv:1906.03252.
- Frei et al., “Development of Advanced Linearized Gyrokinetic Collision Operators Using a Moment Approach,” J. Plasma Phys. 87, 905870501 (2021), arXiv:2104.11480.
- Frei, Ernst, and Ricci, arXiv:2202.06293.
- Frei, Hoffmann, and Ricci, arXiv:2201.02860.
- Frei et al., “Moment-Based Approach to the Flux-Tube Linear Gyrokinetic Model,” J. Plasma Phys. 89, 905890414 (2023), arXiv:2210.05799.

### Differentiable simulation

- iGENE, arXiv:2605.03086.
- gyaradax, arXiv:2604.06085.
- JAX-in-Cell, arXiv:2512.12160.
- Griewank and Walther, Revolve checkpointing, ACM TOMS 26, 19-45 (2000).
- least-squares shadowing, NILSS, and NILSAS references used in the nonlinear-sensitivity work.

### Software and documentation standards

- JAX official benchmarking, profiling, compilation-cache, donation, sharding, and autodiff documentation.
- Scientific Python SPEC 0 for minimum supported dependencies and SPEC 1 for lazy loading.
- Diataxis documentation framework.
- PyData Sphinx Theme accessibility guidance.
- Sphinx and MyST-Parser documentation.
- Peter Yang’s No AI Slop pattern guide.

---

## Work log

Append entries below this line. Do not rewrite historical entries.

### 2026-08-26 - Proposed GKX 3 modernization charter

- Audited GKX 1.8.2 at `e89c7fed31657f32b638e653c7b266e33cded805`.
- Preserved the current sub-20-MiB repository, packaging, progress, output, validation, and documentation gains as baseline contracts.
- Identified architecture, public API, duplicate pathways, test/tool volume, documentation organization, quasilinear promotion, collision scalability, and broad nonlinear optimization as the principal program gaps.
- Proposed retaining `plan/research-grade-roadmap` as the ground-truth branch and restructuring its `plan.md` around stable contracts and append-only logs.

### 2026-08-26 - Maintainer decisions incorporated

- Fixed the GKX 3.x scope as local Maxwellian-background delta-f and flux-tube.
- Approved removal of undocumented 1.x internal imports and the hard source, test, and public-API budgets.
- Kept Python 3.11 as the minimum and CPU/NVIDIA GPU as the required platforms.
- Made equilibrium ExB shear conditional on a bounded readiness audit; moved rotation and momentum transport to 3.x.
- Reframed the quasilinear lane around the best supported tiered model rather than screening-only or universal-accuracy requirements.
- Determined that the correct production collision hierarchy is linearized Sugama/Coulomb; `C[delta f, delta f]` is outside the GKX ordering and roadmap.
- Selected native explicit RK and one native IMEX route as owners, with Diffrax removed after migration parity.
- Audited VMEX and `booz_xform_jax` ownership. VMEX already owns exact differentiable field-line metrics and drifts, while `booz_xform_jax` owns the coordinate transform. GKX will retain only a generic flux-tube contract and thin adapters.
- Required external comparator codes and outputs to remain local; permanent GKX tests must be self-contained and future-proof.
- Recorded `rogeriojorge` as the final approver and sole merger into `main`.
- Removed publication sequencing from the code roadmap.

### 2026-08-28 - Phase 0 workspace and charter update

- Task: replace the ground-truth planning-branch roadmap with the approved GKX 3 modernization charter, bootstrap fresh upstream clones, and establish the Python 3.11 development environment for baseline work.
- Non-goals: no solver, physics, numerical fingerprint, public API, dependency, or release-claim change is part of this planning update.
- Baseline: `main` at GKX 1.8.2, commit `e89c7fed31657f32b638e653c7b266e33cded805`; planning branch at `bcb9fd86eb5a36abdc9d3486e9d8988c3b23d872` before this entry.
- Affected public behavior and scientific claims: none; this change governs later implementation and records approved claim boundaries.
- Files expected to change: root `plan.md` only. The superseded audit remains recoverable from planning-branch history and its supporting notes remain under `plan/`.
- Acceptance: the roadmap follows the approved stable order, retains an append-only work log, names explicit phase exit gates and PR boundaries, and leaves `main` untouched.
- Measurements: verify the plan structure and repository state; baseline correctness and cold/warm runtime, memory, compile, transfer, API, dependency, and provenance inventories remain Phase 0 follow-up work.
- Rollback condition: revert this planning commit if it differs from the maintainer-approved charter or weakens the recorded scientific, performance, or review gates.
