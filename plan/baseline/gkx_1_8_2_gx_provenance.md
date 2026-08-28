# GKX 1.8.2 GX-derived-function inventory

This inventory freezes the GX software-provenance boundary at GKX commit
`4104bf4a2d7463fcd56e9c38434d88510377d2b4`. It answers two separate
questions:

1. Which code did GKX history explicitly identify as a GX port?
2. Where do the descendants of that code live after later renames and splits?

It does not infer copying from numerical parity, shared variable names,
published equations, or interoperability with GX output.

## Evidence and upstream anchor

GKX commit `58ff86c859c1955faecdb3291745bc1d7712852a` added package
`spectraxgk.from_gx` with the module docstring “Internal geometry backends
progressively ported from GX.” Its VMEC module says it is “ported from GX
pyvmec (`gx_geo_vmec.py`)” and follows the original GX script for numerical
parity. This is direct repository evidence, not a conclusion based on similar
code.

The commit added 1,941 lines across three implementation files:

| Imported file at `58ff86c8` | Lines | Top-level symbols | SHA-256 |
| --- | ---: | ---: | --- |
| `src/spectraxgk/from_gx/kernels.py` | 228 | 7 | `b43fcfcf95fe99b636752265ab2f9da4330188692f26c434e482308eaa127e91` |
| `src/spectraxgk/from_gx/miller.py` | 668 | 17 | `9fafb31ad4dfe229de008ed930262d282cf8509eaca9a50bf4da39a510da4bd6` |
| `src/spectraxgk/from_gx/vmec.py` | 1,045 | 10 | `d73fc5a405a96a5ddd41e164628e7b816e987cbb1ab47cbb64f372465d5942be` |

The exact upstream revision used by the author was not recorded. In the local
GX history, `96e42569fa9ffc392a46ddedddf5d24a27b8de39` is the last commit
before the import date. It is therefore the comparison anchor, not asserted as
the exact source revision. Relevant files at that revision are:

| GX comparison file | Lines | SHA-256 |
| --- | ---: | --- |
| `geometry_modules/miller/gx_geo.py` | 675 | `91742d3d07a9ba705eb81ef374ca69786cb01eae069179b7738d526a32b46ad7` |
| `geometry_modules/miller/utils.py` | 288 | `1c0490bf2e11f6340260b1cac70513576f35758922e96929b2a90a4766a43553` |
| `geometry_modules/pyvmec/gx_geo_vmec.py` | 1,414 | `2b80e84542f9ee71cb524e0dc494a86e56533a9cf2f6e97973be007bfc5b0815` |

The GX license text is unchanged between comparison revision `96e42569` and
the local checkout at `3865a537`; its SHA-256 is recorded in root
`PROVENANCE.md`.

## Imported symbol to current owner

All 34 top-level imported symbols appear exactly once below. “Split” means
that later private helpers share the implementation while the named current
entry point remains the contract. “Removed” means no current equivalent was
found and no provenance claim is made for a replacement.

### Imported JAX/array kernels (7)

| Imported symbol | Current owner | Status |
| --- | --- | --- |
| `nperiod_mask` | `gkx.geometry.kernels.nperiod_mask` | retained |
| `nperiod_contract` | `gkx.geometry.kernels.nperiod_contract` | retained |
| `finite_diff_nonuniform` | `gkx.geometry.kernels.finite_diff_nonuniform` | retained |
| `gx_derm` | `gkx.geometry.kernels.centered_reflected_difference` | renamed |
| `gx_dermv` | `gkx.geometry.kernels.weighted_centered_difference` | renamed and split into dimensional helpers |
| `gx_nperiod_data_extend` | `gkx.geometry.kernels.extend_nperiod_data` | renamed |
| `gx_reflect_n_append` | `gkx.geometry.kernels.reflect_and_append` | renamed |

### Imported Miller pipeline (17)

| Imported symbol | Current owner | Status |
| --- | --- | --- |
| `_safe_denom` | `gkx.geometry.kernels._safe_denom` | retained |
| `internal_miller_backend_available` | none | removed; the installed backend no longer has this availability shim |
| `generate_miller_eik_internal` | `gkx.geometry.imported_miller.generate_miller_eik_internal` | retained/refactored |
| `derm` | `gkx.geometry.kernels.derm` | retained wrapper |
| `dermv` | `gkx.geometry.kernels.dermv` | retained wrapper |
| `nperiod_data_extend` | `gkx.geometry.kernels.nperiod_data_extend` | retained wrapper |
| `reflect_n_append` | `gkx.geometry.kernels.reflect_n_append` | retained wrapper |
| `MillerCoreParams` | `gkx.geometry.analytic.MillerCoreParams` | moved |
| `build_collocation_surfaces` | `gkx.geometry.analytic.build_collocation_surfaces` | moved |
| `compute_primary_gradients` | `gkx.geometry.imported_miller.compute_primary_gradients` | retained/refactored |
| `cumulative_trapezoid` | `gkx.geometry.kernels.cumulative_trapezoid` | retained |
| `compute_straight_field_theta` | `gkx.geometry.imported_miller.compute_straight_field_theta` | retained/refactored |
| `compute_equal_arc_theta` | `gkx.geometry.imported_miller.compute_equal_arc_theta` | retained/refactored |
| `rebuild_straight_theta_state` | `gkx.geometry.imported_miller.rebuild_straight_theta_state` | retained/refactored |
| `to_ballooning` | `gkx.geometry.kernels.to_ballooning` | retained |
| `assemble_miller_profiles` | `gkx.geometry.imported_miller.assemble_miller_profiles` | retained and split into private profile stages |
| `write_miller_eik_netcdf` | `gkx.geometry.imported_miller.write_miller_eik_netcdf` | retained/refactored |

### Imported VMEC/PyVMEC pipeline (10)

| Imported symbol | Current owner | Status |
| --- | --- | --- |
| `internal_vmec_backend_available` | `gkx.geometry.backend_discovery.internal_vmec_backend_available` | moved |
| `nperiod_set` | `gkx.geometry.vmec_field_line_sampling.nperiod_set` | moved |
| `dermv` | `gkx.geometry.vmec_field_line_sampling.dermv` | moved |
| `_Struct` | `gkx.geometry.vmec_field_line_sampling._Struct` | moved |
| `_vmec_splines` | `gkx.geometry.vmec_field_line_sampling._vmec_splines` | moved/refactored |
| `_vmec_fieldlines` | `gkx.geometry.imported_vmec._vmec_fieldlines` | retained facade; body split across sampling, derivative, and state modules |
| `_apply_flux_tube_cut` | `gkx.geometry.imported_vmec._apply_flux_tube_cut` | retained/refactored |
| `_equal_arc_remap` | `gkx.geometry.imported_vmec._equal_arc_remap` | retained/refactored |
| `write_vmec_eik_netcdf` | `gkx.geometry.imported_vmec.write_vmec_eik_netcdf` | retained/refactored |
| `generate_vmec_eik_internal` | `gkx.geometry.imported_vmec.generate_vmec_eik_internal` | retained/refactored |

## Current descendant boundary

Later commits made source-level correspondence less obvious:

- `51f4f190` removed `from_gx` naming and created neutral imported-geometry
  owners;
- `5bad73dc` consolidated the Miller backend into
  `geometry/imported_miller.py`;
- `e58b3d3c` consolidated the VMEC backend into
  `geometry/imported_vmec.py`;
- `11f7f646`, `1953dc8e`, and `2e18e8ba` split VMEC discovery, sampling,
  derivative algebra, and state sampling into focused owners;
- `75825eb3` renamed the package from SPECTRAX-GK to GKX.

For license/provenance purposes, the conservative current descendant set is:

- all translated finite-difference, reflection, period-extension,
  trapezoid, and ballooning helpers in `gkx.geometry.kernels`;
- `MillerCoreParams` and `build_collocation_surfaces` in
  `gkx.geometry.analytic`;
- the Miller profile and EIK pipeline in `gkx.geometry.imported_miller`;
- the Boozer/PyVMEC import helpers beginning at
  `_booz_xform_jax_search_paths` in `gkx.geometry.backend_discovery`;
- the `_Struct`, spline, period, derivative, Boozer-mode, metric, drift, and
  packing pathway in `gkx.geometry.vmec_field_line_sampling`;
- the private field-line/Hegna--Nakajima geometry pathway beginning at
  `_fieldline_boozer_coordinates` in
  `gkx.geometry.vmec_boozer_derivatives`;
- the Boozer load/sample pathway from `_new_boozer_object_with_auto_fallback`
  and `_VMECFieldlineScalars` onward in `gkx.geometry.vmec_state_controls`;
- the field-line, flux-tube-cut, equal-arc, EIK-write, and generation pathway
  in `gkx.geometry.imported_vmec`.

Some private dataclasses and stage helpers in those paths were created by GKX
while refactoring the imported algorithms. They remain inside the conservative
descendant boundary because they reorganize data and equations from the port.
Conversely, later generic backend discovery, VMEC state-control/optimization,
and public differentiable geometry utilities are not labeled copied merely
because they call this pathway.

## Explicit exclusions

The following are GX-related but not classified as copied or translated code
without separate history evidence:

- `gkx.artifacts.gx_output`, which reads and plots GX NetCDF output;
- tools and benchmarks that invoke GX or compare GKX and GX results;
- solver changes made to achieve GX numerical parity (grid ordering,
  dealiasing, linked boundaries, timestep policy, or normalization);
- NetCDF names chosen for interoperability;
- Laguerre--Hermite, collision, and gyrokinetic equations implemented from
  cited papers.

Those items require scientific citation or compatibility documentation, not a
software-copy assertion. Future ports must update root `PROVENANCE.md` in the
same PR and record an exact upstream revision and source path at import time.

## Reproduction commands

```console
git show 58ff86c8:src/spectraxgk/from_gx/kernels.py | shasum -a 256
git show 58ff86c8:src/spectraxgk/from_gx/miller.py | shasum -a 256
git show 58ff86c8:src/spectraxgk/from_gx/vmec.py | shasum -a 256

git -C ../GX show 96e42569:geometry_modules/miller/gx_geo.py | shasum -a 256
git -C ../GX show 96e42569:geometry_modules/miller/utils.py | shasum -a 256
git -C ../GX show 96e42569:geometry_modules/pyvmec/gx_geo_vmec.py | shasum -a 256
git -C ../GX show 96e42569:docs/License.rst | shasum -a 256
```
