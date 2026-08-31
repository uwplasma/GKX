# GKX 1.8.2 output-schema inventory

This file freezes the output contracts implemented at GKX commit
`4104bf4a2d7463fcd56e9c38434d88510377d2b4`. It is an inventory of the
current Python containers and artifact writers, not the proposed GKX 3 schema.
The product version is 1.8.2; the snapshot also contains the merged planning,
benchmark-import, and optional-dependency repairs, none of which changed these
contracts.

The implementation has no persisted runtime schema version: runtime TOML
inputs and summary JSON, CSV, NPY, and NetCDF outputs are versionless. The
NetCDF scalar `code_info=1` with attribute `value="gkx"` identifies the
producing code only; it is not read or documented as a schema version.
Consequently every structure below is a de facto compatibility surface until
Phase 1 introduces explicit versioning and migration tests.

## Python result containers

These frozen dataclasses are in `src/gkx/workflows/runtime/results.py`.
Annotations describe the in-memory contract; the artifact writers persist only
the subsets listed later.

| Container | Fields, in declaration order |
| --- | --- |
| `RuntimeLinearResult` | `ky: float`; `gamma: float`; `omega: float`; `selection: ModeSelection`; `t: ndarray | None`; `signal: ndarray | None`; `field_history: ndarray | None`; `state: ndarray | None`; `z: ndarray | None`; `eigenfunction: ndarray | None`; `fit_window_tmin: float | None`; `fit_window_tmax: float | None`; `fit_signal_used: str | None`; `gamma_stderr: float | None`; `omega_stderr: float | None`; `fit_r2: float | None`; `fit_settled: bool | None`; `quasilinear: dict | None` |
| `RuntimeLinearScanResult` | `ky: ndarray`; `gamma: ndarray`; `omega: ndarray`; `quasilinear: tuple[dict, ...] | None`; `parallel: dict | None`; `warm_start: dict | None` |
| `RuntimeParameterScanResult` | `parameter_name: str`; `values: ndarray`; `gamma: ndarray`; `omega: ndarray`; `runs: tuple[RuntimeLinearResult, ...]` |
| `RuntimeNonlinearResult` | `t: ndarray`; `diagnostics: SimulationDiagnostics | None`; `phi2: ndarray | None`; `fields: FieldState | None`; `state: ndarray | None`; `ky_selected: float | None`; `kx_selected: float | None`; `wall_seconds: float | None`; `saturation: dict | None` |

`RuntimeParameterScanResult` has no dedicated writer in the current runtime
artifact layer. `field_history` and the top-level nonlinear `t` are likewise
not directly persisted by the table writers.

`SimulationDiagnostics` contains required `t`, `dt_t`, `dt_mean`, `gamma_t`,
`omega_t`, `Wg_t`, `Wphi_t`, `Wapar_t`, `heat_flux_t`, `particle_flux_t`, and
`energy_t`; optional species heat/particle flux, turbulent-heating, field-mode,
CFL-scale, and `ResolvedDiagnostics` arrays supplement that core.

## Artifact names and returned path keys

For a prefix `case`, writers return string paths under the following keys.
Passing an explicit `.json` or `.csv` path makes that path the corresponding
summary or primary table; other sidecars still derive from the suffix-stripped
base.

| Writer | Required output | Conditional output |
| --- | --- | --- |
| linear | `summary` -> `case.summary.json` | `timeseries` -> `case.timeseries.csv`; `eigenfunction` -> `case.eigenfunction.csv`; `state` -> `case.state.npy`; quasilinear outputs below |
| linear scan | `summary` -> `case.summary.json`; `scan` -> `case.scan.csv` | `quasilinear_spectrum` -> `case.quasilinear_spectrum.csv` |
| standalone quasilinear | `quasilinear_summary` -> `case.quasilinear.summary.json` | `quasilinear_species` -> `case.quasilinear_species.csv` |
| nonlinear table | `summary` -> `case.summary.json` | `diagnostics` -> `case.diagnostics.csv`; `state` -> `case.state.npy` |
| nonlinear NetCDF | `out` -> `case.out.nc`; `summary` -> `case.summary.json` | `restart` defaults to `case.restart.nc`; `big` -> `case.big.nc` |

The NetCDF base resolver treats `.nc`, `.out.nc`, `.big.nc`, and `.restart.nc`
as one bundle base. A configured restart path overrides the default. Primary
NetCDF output requires diagnostics. Restart output requires state; big output
requires final fields.

## JSON summaries

JSON is UTF-8, indented by two spaces, key-sorted, and terminated by a newline.
Non-finite optional linear fit metrics are written as `null`.

### Linear summary

Required top-level keys are:

`kind`, `ky`, `gamma`, `omega`, `fit_window_tmin`, `fit_window_tmax`,
`fit_signal_used`, `gamma_stderr`, `omega_stderr`, `fit_r2`, `fit_settled`,
`selection`, `n_samples`, `n_state_shape`, `has_eigenfunction`, and
`has_quasilinear`.

`kind` is `"linear"`. `selection` contains integer `ky_index`, `kx_index`, and
`z_index`. `n_state_shape` is an integer list or `null`. When present,
`quasilinear` is an additional top-level key containing the payload unchanged.

### Linear-scan summary

Required keys are `kind="linear_scan"`, `n_ky`, `ky_min`, `ky_max`, and
`has_quasilinear`. Empty scans store `null` extrema. Optional `parallel` and
`warm_start` dictionaries are copied through unchanged when they are dicts.

### Quasilinear summary

The standalone summary serializes the supplied dictionary unchanged. The
producer's `QuasilinearTransportResult.to_dict()` emits:

`ky`, `gamma`, `omega`, `mode`, `saturation_rule`,
`amplitude_normalization`, `channels`, `kperp_average`, `kperp_eff2`,
`phi_norm2`, `amplitude2`, `heat_flux_weight_species`,
`particle_flux_weight_species`, `saturated_heat_flux_species`,
`saturated_particle_flux_species`, `species`, `metadata`,
`heat_flux_weight_total`, `particle_flux_weight_total`,
`saturated_heat_flux_total`, and `saturated_particle_flux_total`.

This writer accepts arbitrary dictionaries, so producer-added metadata is part
of observed output even though the writer does not validate a closed schema.

### Nonlinear summary

Always present: `kind="nonlinear"`, `ky_selected`, `kx_selected`, and
`n_state_shape`.

With diagnostics, the summary also contains `n_samples`, `t_last`, `dt_mean`,
`gamma_last`, `omega_last`, `Wg_last`, `Wphi_last`, `Wapar_last`,
`heat_flux_last`, `particle_flux_last`, and nested `timestep_cost`. Without
diagnostics but with scalar `phi2`, it instead contains `n_samples=0`,
`t_last=0.0`, and `phi2_last`. Optional `saturation` is copied through
unchanged.

## CSV and NPY contracts

CSV files use comma delimiters, an un-commented header row, and NumPy's text
number formatting.

| Artifact | Columns, in order |
| --- | --- |
| linear timeseries | `t`, `signal_real`, `signal_imag`, `signal_abs` |
| linear eigenfunction | `z`, `eigen_real`, `eigen_imag`, `eigen_abs` |
| linear scan | `ky`, `gamma`, `omega` |
| scan quasilinear spectrum | `ky`, `mode_ky`, `gamma`, `omega`, `kperp_eff2`, `heat_flux_weight_total`, `particle_flux_weight_total`, `amplitude2`, `saturated_heat_flux_total`, `saturated_particle_flux_total` |
| quasilinear species | `species_index`, `heat_flux_weight`, `particle_flux_weight`, `saturated_heat_flux`, `saturated_particle_flux` |
| nonlinear diagnostics, required prefix | `t`, `dt`, `gamma`, `omega`, `Wg`, `Wphi`, `Wapar`, `energy`, `heat_flux`, `particle_flux` |

Nonlinear diagnostics append `turbulent_heating` when available, then one
column per available species in this order: `heat_flux_s{i}`,
`particle_flux_s{i}`, and `turbulent_heating_s{i}`. Multi-component base
diagnostic series are flattened to one value per sample by averaging all
non-time axes. Quasilinear missing values are represented by `nan` in CSV.

`case.state.npy` is a standard NumPy `.npy` file. The writer preserves the
state's dtype and shape; it declares no axis names or schema version in the
file.

## Nonlinear NetCDF bundle

All numeric payloads below are NetCDF `f4` unless marked `f8` or `i4`.
Complex values use a trailing `ri=2` real/imaginary axis. The main and big
files use dealiased `kx` and non-negative/dealiased `ky` output axes even when
the in-memory solver state carries full spectral axes.

### `case.out.nc`

Dimensions are `ri=2`, `x=Nx`, `y=Ny`, `theta=Nz`, `kx=active Nx`,
`ky=active non-negative Ny`, `kz=Nz`, `m=Nm`, `l=Nl`, `s=Nspecies`, and
`time=number of diagnostic samples`.

Root scalar variables are `ny`, `nx`, `ntheta`, `nhermite`, `nlaguerre`,
`nspecies`, `nperiod`, `debug`, and `code_info`, all `i4`. `code_info` has the
string attribute `value="gkx"`. The four groups are `Grids`, `Geometry`,
`Diagnostics`, and `Inputs`.

`Grids` contains `time:f8(time)` and `kx`, `ky`, `kz`, `x`, `y`, and
`theta`, each `f4` on its namesake dimension.

`Geometry` always contains profile variables `bmag`, `bgrad`, `gbdrift`,
`gbdrift0`, `cvdrift`, `cvdrift0`, `gds2`, `gds21`, `gds22`, `grho`, and
`jacobian`, each `(theta)`. Scalars are `gradpar`, `nperiod:i4`, `q`, `shat`,
`shift`, `rmaj`, `aminor`, `kxfac`, `drhodpsi`, `theta_scale`, `nfp:i4`,
`alpha`, and `zeta_center`. Optional imported-geometry profiles are `Rplot`,
`Zplot`, and `zeta_plot`, each `(theta)`.

`Inputs` contains scalar `igeo:i4`, `slab:i4`, `const_curv:i4`,
`geofile_dum:i4`, `drhodpsi`, `kxfac`, `Rmaj`, `shift`, `eps`, `q`, `shat`,
`kappa`, `kappa_prime`, `tri`, `tri_prime`, `beta`, `zero_shat:i4`, `B_ref`,
`a_ref`, `grhoavg`, and `surfarea`.

`Diagnostics` always contains `Phi2_t(time)` plus `(time,s)` histories
`Wg_st`, `Wphi_st`, `Wapar_st`, `HeatFlux_st`, `ParticleFlux_st`,
`HeatFluxES_st`, `HeatFluxApar_st`, `HeatFluxBpar_st`, `ParticleFluxES_st`,
`ParticleFluxApar_st`, `ParticleFluxBpar_st`, and `TurbulentHeating_st`.

When resolved diagnostics exist, each corresponding non-null array adds its
variable:

- `Phi2_kxt(time,kx)`, `Phi2_kyt(time,ky)`,
  `Phi2_kxkyt(time,ky,kx)`, and `Phi2_zt(time,theta)`;
- `Phi2_zonal_t(time)`, `Phi2_zonal_kxt(time,kx)`,
  `Phi2_zonal_zt(time,theta)`, and complex
  `Phi_zonal_mode_kxt(time,kx,ri)` and
  `Phi_zonal_line_kxt(time,kx,ri)`;
- for prefixes `Wg`, `Wphi`, `Wapar`, `HeatFlux`, `ParticleFlux`,
  `HeatFluxES`, `HeatFluxApar`, `HeatFluxBpar`, `ParticleFluxES`,
  `ParticleFluxApar`, `ParticleFluxBpar`, and `TurbulentHeating`, optional
  `{prefix}_kxst(time,s,kx)`, `{prefix}_kyst(time,s,ky)`,
  `{prefix}_kxkyst(time,s,ky,kx)`, and `{prefix}_zst(time,s,theta)`;
- optional `Wg_lmst(time,s,m,l)`.

The reader reconstructs the in-memory diagnostics contract from this subset;
not every in-memory field is stored independently. In particular, the file
does not store `dt_t`, `dt_mean`, `gamma_t`, `omega_t`, `energy_t`,
`phi_mode_t`, or `cfl_scales` as named variables.

### `case.restart.nc`

Dimensions are `Nspecies`, `Nm`, `Nl`, `Nz`, `Nkx`, `Nky`, and `ri=2`.
`G:f4(Nspecies,Nm,Nl,Nz,Nkx,Nky,ri)` stores the dealiased complex state and
`time:f8()` stores the final diagnostic time. The in-memory state order is
`(species, l, m, ky, kx, z)`; read/write code transposes between that order and
the persisted order. The loader supports both positive-`ky` restart expansion
and a legacy/full-`ky` path, which is a migration obligation.

### `case.big.nc`

Dimensions and root metadata match `case.out.nc`, except `time=1`. Groups are
`Grids`, `Geometry`, and `Diagnostics`; there is no `Inputs` group. `Grids.time`
is the final diagnostic time.

Final fields are `Phi`, `Apar`, and `Bpar`, each
`(time,ky,kx,theta,ri)`, plus real-space `PhiXY`, `AparXY`, and `BparXY`, each
`(time,y,x,theta)`. Missing electromagnetic fields are written as zeros.

When state exists, basis moments `Density`, `Upar`, `Tpar`, `Tperp` and
particle moments `ParticleDensity`, `ParticleUpar`, `ParticleUperp`,
`ParticleTemp` are written both spectrally as
`(time,s,ky,kx,theta,ri)` and in real space with an `XY` suffix as
`(time,s,y,x,theta)`.

## Frozen compatibility obligations and Phase 1 debt

Until an explicitly versioned successor and migrations exist:

1. treat current path derivation and returned path-key names as observable;
2. preserve required JSON keys and CSV column ordering, and make additions
   consciously because consumers cannot negotiate a version;
3. preserve `.state.npy` dtype/shape and document axes before promoting it as a
   long-lived interchange format;
4. preserve NetCDF group, variable, dimension, dtype, complex `ri`, dealiased
   axis, and restart-transpose behavior for existing readers;
5. retain legacy/full-`ky` restart readability in any migration;
6. do not interpret arbitrary nested `parallel`, `warm_start`, `quasilinear`,
   `timestep_cost`, or `saturation` dictionaries as closed versioned schemas;
7. add explicit TOML and NetCDF result schema versions in Phase 1, then test
   old-reader/new-reader behavior before intentionally breaking this surface.

This freeze does not promise that every 1.8.2 field survives into GKX 3. It
creates the checklist needed to classify each later change as preserved,
migrated, deprecated, or intentionally removed.

## Reproduction and source anchors

Primary implementation anchors:

- `src/gkx/workflows/runtime/results.py`
- `src/gkx/diagnostics/metadata.py`
- `src/gkx/diagnostics/quasilinear_transport.py`
- `src/gkx/artifacts/io.py`
- `src/gkx/artifacts/spectral_layout.py`
- `src/gkx/artifacts/nonlinear_netcdf.py`
- `src/gkx/workflows/runtime/artifacts.py`

Focused executable contract tests:

```console
pytest -q tests/integration/runtime/test_runtime_artifacts.py
pytest -q tests/integration/runtime/test_runtime_runner.py
```

The inventory is documentation-only. Generated JSON, CSV, NPY, and NetCDF
probe files remain in pytest temporary directories and are not tracked.
