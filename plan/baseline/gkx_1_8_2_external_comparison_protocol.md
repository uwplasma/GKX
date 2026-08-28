# GKX 1.8.2 local external-comparison protocol

Status: Phase 0 policy and reproduction contract. This document does not
contain external output, does not promote a numerical result, and does not make
GX, stella, or GS2 a GKX dependency or CI oracle.

Frozen source revisions for the initial local workspace are:

| Code | Revision | Current GKX-side support |
|---|---|---|
| GKX | `cb2219bbf835a7f96817bf766bbbfc29c992a0b5` | reference owner and self-contained promotion target |
| GX | `3865a53778862e1686f414bf6f416339e24887c9` | detailed linear, nonlinear, exact-state, RHS-term, and diagnostic comparison tools |
| stella | `2b8e269f2addd0baa5991057eafa022135e04498` | literature/digitized references exist; no general raw-output adapter is promoted |
| GS2 | `4d8c94bcfd976ed5d04ec83e776c3d915038a589` | no general raw-output adapter is promoted |

The revisions identify the audited source trees. Every actual comparison must
record its own revisions again; this table is not a floating “latest” alias.

## Boundary and directory contract

Use two disjoint roots:

```text
GKX_project/
  GKX/                 tracked GKX checkout
  GX/                  external source checkout
  stella/              external source checkout
  GS2/                 external source checkout
  artifacts/
    external/<campaign-id>/
      manifest.json
      inputs/
      raw/<code>/
      extracted/<code>/
      comparison/
```

`artifacts/external` is outside every source repository. The campaign ID should
be a stable human label plus a UTC timestamp or manifest digest. Do not put the
following in GKX Git history, pull-request attachments, public CI artifacts, or
documentation downloads without a separate maintainer decision:

- external executables, objects, modules, containers, or build caches;
- generated external inputs that reveal private paths or installations;
- WOUTs or other equilibrium files not already approved for distribution;
- raw NetCDF, HDF5, binary state, restart, log, or profile output;
- extracted full trajectories, fields, eigenfunctions, or term dumps;
- absolute user, host, scratch, module, token, or credential paths.

Small independently authored benchmark inputs may live in `benchmarks/` only
when their license and provenance are clear and they contain no raw output.
Tracked compact summaries require explicit review under the promotion rule
below. `.gitignore` is not a publication boundary: check `git status` and
`git check-ignore` before and after every campaign.

## Required manifest

Write `manifest.json` before comparing numbers. It is local and must include:

| Group | Required fields |
|---|---|
| Campaign | schema version, campaign ID, creation time in UTC, purpose, operator, claim boundary |
| Source | code name, repository URL, full commit, dirty status, submodule commits, patch digest or explicit `none` |
| Build | compiler/CUDA/MPI versions, flags, precision, build-system command, executable SHA-256, linked numerical/NetCDF libraries, container digest if used |
| Machine | host label without private path, CPU/GPU model, visible devices, process/thread/MPI layout, memory, scheduler allocation |
| Input | local relative path, SHA-256, generator and version, equilibrium/data digests, parameter overrides |
| Physics | species, normalization, signs, geometry, electromagnetic switches, collisions/closures, gradients, beta, flow/shear, nonlinear switches |
| Resolution | spatial and velocity grids, domain/tube length, dealiasing, boundary/twist convention, retained modes |
| Numerics | integrator, timestep/adaptive controls, tolerance, solver residual/tolerance/restart, dissipation, initialization, random seed, end time, save cadence |
| Output | local relative raw paths and SHA-256 values, exit status, wall time, peak host/device memory when measured |
| Extraction | postprocessor repository/commit, command, script SHA-256, diagnostic definitions, selected windows/modes, interpolation and fit policies |
| Comparison | compared quantities, unit/normalization transforms, masks, tolerances or exploratory status, result location |

A dirty comparator is allowed for diagnosis only when the patch is saved
locally and hashed. A result from a dirty tree cannot support a release claim
until reproduced from a named clean revision or reviewed patch.

## Run protocol

1. **Freeze the case.** Write the shared physical case in code-independent
   terms. List every code-specific translation and expected non-equivalence.
2. **Freeze sources and builds.** Record full commits, submodules, dirty status,
   executable hashes, build flags, precision, libraries, and visible hardware.
3. **Validate each code independently.** Require successful input checks,
   finite outputs, stated solver convergence/residual, and code-owned smoke or
   numerical tests before cross-code interpretation.
4. **Run a resolution ladder.** Compare matched physical domains while varying
   timestep, spatial/velocity resolution, dissipation, and solver tolerance as
   applicable. One matched grid is a diagnostic, not convergence evidence.
5. **Extract without overwriting raw data.** Write code-native output only below
   `raw/<code>` and normalized values below `extracted/<code>`. Hash both.
6. **Compare definitions before values.** Resolve units, reference lengths,
   thermal-speed factors, Fourier signs/order, field normalization, flux sign,
   species sums, Jacobian/quadrature weights, time windows, and mode selection.
7. **Classify the outcome.** Use `agreement`, `explained_difference`,
   `unresolved_difference`, or `invalid_run`; never silently drop a failed or
   contradictory rung.
8. **Translate the lesson.** Create self-contained GKX evidence before using the
   comparison to gate GKX. External files and executables remain local.

Wall-clock comparisons additionally require identical device occupancy and a
clear distinction among process startup, preparation, compilation/first
execution, transfers, and warm execution. Cross-code runtime is not meaningful
when these phases or precision differ.

## Disagreement ladder

Investigate a mismatch in this order and retain each result in the local
manifest:

1. input translation, species ordering, signs, and normalization;
2. geometry arrays, coordinate orientation, tube length, twist/boundaries, and
   quadrature/Jacobian conventions;
3. initialization and the exact state/mode being compared;
4. diagnostic definition, sampling time, fit/window selection, and
   interpolation;
5. linear term/RHS decomposition and field solve on the same state;
6. solver residual, iteration/tolerance, timestep, and arithmetic precision;
7. spatial and velocity resolution, dealiasing, dissipation, and domain size;
8. alternate algorithm or manufactured/analytic limit within GKX;
9. only then a possible model or implementation defect.

No code is declared correct because it is older, more cited, or agrees with a
third code. A three-code majority can still share a convention or model error.

## Code-specific command templates

Commands below are templates derived from each frozen checkout's own build/run
documentation. Replace angle-bracket values locally and record the expanded
command in the manifest. Never paste private expanded paths into Git.

### GX

GX requires an NVIDIA build/runtime environment in the frozen checkout.

```console
git -C GX status --short
git -C GX rev-parse HEAD
make -C GX -j <jobs>
sha256sum GX/gx
(cd artifacts/external/<campaign-id>/raw/gx && \
  <workspace>/GX/gx ../../inputs/<case>.in)
```

The current GKX repository has GX-specific tools for imported-linear windows,
KBM branches, nonlinear diagnostics/terms, exact-state startup, RHS terms, and
parity matrices. Invoke `--help` first and direct every output option to the
local campaign tree:

```console
python GKX/tools/comparison/compare_gx_imported_linear.py --help
python GKX/tools/comparison/compare_gx_kbm.py --help
python GKX/tools/comparison/compare_gx_nonlinear.py --help
python GKX/tools/comparison/compare_gx_rhs_terms.py --help
python GKX/tools/comparison/compare_runtime.py --help
```

These tools parse local GX data; their presence does not make any historical GX
file a permanent oracle.

### stella

The frozen checkout supports CMake and make builds. The manifest must record
which route and options were used; the CMake double-precision option defaults
on but must still be recorded.

```console
git -C stella status --short
git -C stella rev-parse HEAD
cmake -S stella -B artifacts/external/<campaign-id>/build/stella \
  -DSTELLA_ENABLE_DOUBLE=ON
cmake --build artifacts/external/<campaign-id>/build/stella -j <jobs>
<stella-executable> <local-input>
```

There is no general promoted GKX stella-output adapter at the frozen revision.
Add a local extractor first, record its source digest and definitions, and turn
the finding into a self-contained GKX test before proposing a tracked gate.
Digitized literature curves are literature-anchored evidence, not a substitute
for a revision-bound local run.

### GS2

Clone/update submodules and select a supported `GK_SYSTEM`; record all linked
libraries and make variables.

```console
git -C GS2 status --short
git -C GS2 rev-parse HEAD
git -C GS2 submodule status --recursive
make -C GS2 submodules
make -C GS2 GK_SYSTEM=<system> -I Makefiles -j <jobs>
<gs2-executable> --check-input <local-input>
mpirun -n <ranks> <gs2-executable> <local-input>
```

As with stella, no general promoted raw-output adapter is frozen here. A local
extractor must state array ordering, normalization, and diagnostics rather than
reusing a GX assumption.

On platforms where `sha256sum` is unavailable, use
`shasum -a 256 <path>` and record the command.

## Promotion to permanent GKX evidence

An external finding may influence a release claim only after it becomes one or
more self-contained GKX gates:

- an analytic/asymptotic limit or independently implemented formula;
- a manufactured solution or observed-order test;
- a conservation, symmetry, free-energy, restart, or alternate-algorithm gate;
- a literature-anchored scalar/interval with citation and convention mapping;
- a compact, versioned cross-code summary approved for publication.

A compact summary may contain scalar metrics, uncertainty/convergence data,
full source/build/input/postprocessor provenance, hashes, and the explicit
claim boundary. It must not contain raw arrays or enough samples to reconstruct
an external output. Its schema must name the comparator revision; a summary is
never valid for another revision by default. Every summary needs a
self-contained GKX test whose failure message asks for investigation, not an
automatic declaration that the comparator is correct.

If no self-contained translation is scientifically honest, keep the result
local and classify the GKX claim as externally compared but not CI-gated.

## Cleanup and audit

Before closing a campaign:

```console
git -C GKX status --short
git -C GKX ls-files | rg -i '(\.out\.nc|\.big\.nc|restart|term_dump|diag_state)'
git -C GKX check-ignore -v ../artifacts/external/<campaign-id> || true
```

The first command must show only intended GKX changes. Review any result from
the second command as an already approved compact fixture or legacy debt; it is
not permission to add another raw file. Archive or delete the local campaign
under the maintainer's data-retention policy, preserving the manifest and
hashes wherever the raw data are retained.

## GKX-side verification

The protocol changes no executable code. Verify the current comparison-tool
parsers and self-contained fixtures with:

```console
python -m pytest -q \
  tests/tools/comparison/test_reference_comparison_tools.py \
  tests/tools/comparison/test_exact_state_audit.py
python tools/comparison/compare_gx_imported_linear.py --help
python tools/comparison/compare_gx_kbm.py --help
python tools/comparison/compare_gx_nonlinear.py --help
python tools/comparison/compare_gx_rhs_terms.py --help
python tools/comparison/compare_runtime.py --help
```

Permanent hosted CI runs only the self-contained GKX tests above. External
builds and executions remain an explicit local release activity.
