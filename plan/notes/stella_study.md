# stella source study and build report (GKX plan item 2.3, first half)

Date: 2026-08-18. Machine: macOS arm64 (Darwin 23.4), MacPorts at /opt/local.
Repo: /Users/rogeriojorge/local/stella (upstream https://github.com/stellaGK/stella.git)

## 1. Repository state and version

- The repo was **already cloned** at /Users/rogeriojorge/local/stella (not re-cloned; existing state preserved).
- Checked-out commit: **4cdc5fcd34a6bd9252873e2542fcd1c8c4636dff** ("Fix stellarator symmetric BC (#127)", 2023-10-02), `git describe` = **v0.5.1-240-g4cdc5fcd**.
- Submodules (initialized): externals/git_version @ b3794a1 (v0.6.0-4), externals/neasyf @ afb2b2f (v0.4.2-1), externals/pFUnit @ bc4de1f.
- Upstream master (fetched, NOT merged): **2b8e269f2addd0baa5991057eafa022135e04498** (2026-08-12); local checkout is 185 commits behind.
- Local working-tree modification (pre-existing, left untouched): Makefiles/Makefile.macosx-homebrew adapted for a previous homebrew/MacPorts build. The tree also contains many previous run outputs (cyclone `cyc_*`, `stella_cyclone_*`, `tune_*` cases) from earlier work — evidence this machine has run stella production scans before.

**Version gap that matters for GKX:** the pinned v0.5.1-era code is **electrostatic only** (`fapar`, `fbpar` are accepted in the input but documented "currently has no effect"). Current master (2026) has a fully restructured input system (new namelists: `&geometry_options`, `&gyrokinetic_terms`, `&electromagnetic` with `include_apar`, `include_bpar`, `beta`), electromagnetic capability, and a converter script `AUTOMATIC_TESTS/convert_input_files/convert_inputFile.py` for old inputs. A default new-format input lives at `STELLA_CODE/read_namelists_from_input_file/default_input_file.in` on master. If GKX cross-validation needs EM runs with stella, the office Linux host should build **master**; for electrostatic ITG (linear + nonlinear, kinetic electrons) the pinned local version is sufficient and already validated here.

## 2. Build on macOS: SUCCESS

Toolchain found (no `sudo port install` needed — nothing was missing):
- gfortran = MacPorts gcc13 (13.4.0); `mpif90` = mpich-clang16 wrapper driving gfortran-mp-13 (`port select mpi` = mpich-clang16-fortran)
- FFTW3 at /opt/local; LAPACK found; netCDF-C at /opt/local (nc-config)
- netCDF-Fortran 4.6.1: **user's local build** at /Users/rogeriojorge/local/netcdf-fortran/build (MacPorts does not have libnetcdff installed)

The pre-existing `stella` binary in the repo (built Feb 2025) **no longer runs**: it was linked against an older MacPorts MPICH whose `libmpifort.12.dylib` exported `_mpi_win_allocate_shared_cptr_`; the current mpich ports do not — a MacPorts MPI upgrade broke it. Backed it up to scratchpad (`stella_binary_backup_feb2025`) and rebuilt.

Working build recipe (out-of-tree CMake; the repo's Makefile-based `GK_SYSTEM` path also exists, incl. a `Makefile.macports`):

```sh
cd /Users/rogeriojorge/local/stella
make -I Makefiles clean GK_SYSTEM=macports   # REQUIRED: stale in-tree .o/.mod from old make build break CMake (documented in README)
FC=mpif90 cmake . -B build_cmake \
  -DnetCDFFortran_ROOT=/Users/rogeriojorge/local/netcdf-fortran/build \
  -DFFTW_ROOT=/opt/local
cmake --build build_cmake -j 8
# binary: /Users/rogeriojorge/local/stella/build_cmake/stella
```

macOS-specific gotchas (relevant when redoing this, less so on the Linux office host):
1. The in-tree artifacts from any previous plain-`make` build MUST be cleaned first, or CMake linking fails with `___spfunc_MOD_j0` undefined (stale `spfunc.mod` compiled with the `_SPLOCAL_` variant shadows the F200X-intrinsics variant). This is the exact failure documented in the README.
2. Do NOT set `-DSTELLA_ENABLE_LOCAL_SPFUNC=ON` — it is declared incompatible with the default `STELLA_ENABLE_F200X=ON`, and is unnecessary (gfortran's F2008 `bessel_j0/j1` intrinsics are used).
3. Run with the matching launcher: `/opt/local/bin/mpirun` (mpich-clang16). No DYLD path tricks needed for the fresh binary.
4. Rebuild whenever MacPorts mpich is upgraded (dylib symbol drift, see above).

**Verification:** `mpirun -np 4 build_cmake/stella example_linear.in` (CBC-like linear ITG, ky=0.5) reproduces the user's archived run in the repo **bit-for-bit**: final `omega = 0.18702523 + 0.09703135 i` (units v_th,ref/a) — identical to `example_linear.omega` from the Feb 2025 binary. The nonlinear box example (`example_nonlinear.in`, 32x32, Dougherty collisions, hyper-dissipation) also runs cleanly on 4 ranks. Test artifacts are in the scratchpad `stella_test/` dir, not in the repo.

On Linux (office ssh host) the equivalent is trivial: gcc + openmpi + netcdf-fortran + fftw3 + lapack, then the same CMake line (or `make -I Makefiles GK_SYSTEM=gnu_ubuntu`).

## 3. Evolved equations and normalizations

- **Model:** delta-f electrostatic gyrokinetics in a flux tube (options for full-flux-surface and radially global exist but default off). Operator-split implicit-explicit scheme: parallel streaming and mirror terms implicit (response-matrix approach), drifts/drive/nonlinearity explicit (SSP-RK2/3/4). Reference: Barnes, Parra & Landreman, JCP 391, 365 (2019).
- **Evolved distribution:** the guiding-center distribution g normalized by the Maxwellian. From `stella_diagnostics.f90`: `f/F0 = g + (Ze/T)(<phi>_R - phi)`, i.e. g = h/F0 - (Ze/T)<phi>_R with h the nonadiabatic part. This is the GS2-family "gbar" convention. (`run_parameters::maxwellian_normalization` toggles whether the extra Maxwellian factor is absorbed; default false.)
- **Field equation:** quasineutrality for phi only (`fields.fpp`); apar machinery exists but is inert in this version. Adiabatic-species closure via `adiabatic_option` — `'field-line-average-term'` (= `iphi00=2`): delta n_e/n = (e/T_e)(phi - <phi>_psi), the correct choice for ITG with adiabatic electrons; `'no-field-line-average-term'` for ETG with adiabatic ions. `tite` and `nine` enter this closure.
- **Normalizations** (GS2/GIST-compatible):
  - Reference length: **a_ref**. Miller: lengths in units of a (the CBC example uses rmaj = 1/0.36 = 2.77778). VMEC: **a_ref = Aminor_p** from the wout file.
  - Reference field: VMEC **B_ref = 2|psi_tor,LCFS|/a_ref^2** (explicitly "the choices made by Pavlos Xanthopoulos in GIST", `vmec_to_stella_geometry_interface.f90:459-463`). Same convention as GX, so a GX/GKX wout-based flux tube and a stella one share a_ref and B_ref directly.
  - Thermal speed: **v_th,s = sqrt(2 T_s/m_s)** (Maxwellian is exp(-vpa^2 - mu B) on the code grid; the g_exb doc string writes rates in units of sqrt(2)v_th,ref/R explicitly).
  - Time in a_ref/v_th,ref; omega and gamma in v_th,ref/a_ref (this is the unit of the `.omega` file and `omega` netCDF variable).
  - Wavenumbers kx, ky in 1/rho_ref (netCDF attribute `units = "1/rho_r"`), rho_ref = m_ref v_th,ref/(e B_ref).
  - rho* = rho_ref/a_ref set by `rhostar` in `&parameters` (only matters beyond the flux-tube limit: full-flux-surface, radially global, parallel nonlinearity, neoclassical).
  - Species inputs: `dens`, `temp`, `mass` relative to reference species; gradients `tprim` = a/L_T, `fprim` = a/L_n (note: a-normalized, not R-normalized — CBC's R/L_T=6.9 becomes tprim=2.49 with a/R=0.36).
  - Collisionality: `vnew_ref` in `&parameters`; per-species nu built as vnew_ref * dens * Z^4 / (sqrt(mass) temp^1.5). Operators: Dougherty (default, `&collisions_dougherty`, momentum/energy conserving options) or Fokker-Planck (`&collisions_fp`), implicit by default (`&dissipation`: include_collisions, collisions_implicit, hyper_dissipation + `&hyper` D_hyper).
- **Sign/flux caveats for comparison:** fluxes written per species vs time; with `flux_norm = .true.` (default) radial fluxes are divided by <|grad r|>_psi. Set `flux_norm = .false.` for the raw gyro-Bohm-normalized fluxes when comparing to GKX/GX (the W7-X example does exactly this).

## 4. Velocity space, and mapping GKX (Nl, Nm) to stella

- **Grid type: (vpa, mu)**, not pitch-angle/energy and not spectral:
  - vpa: **uniform** grid on [-vpa_max, vpa_max], `nvpa = 2*nvgrid` points, vpa=0 excluded. Defaults nvgrid=24 (48 points), vpa_max=3.
  - mu: **Gauss-Laguerre** nodes/weights (`get_laguerre_grids`), `nmu` points, mu=0 excluded, extent set by `vperp_max` (default 3); `equally_spaced_mu_grid=.true.` switches to uniform (needed by some collision configs; default off = recommended).
- **Mapping from GKX Hermite-Laguerre (Nm Hermite in vpar, Nl Laguerre in mu):**
  - mu direction is the clean one: stella's nmu-point Gauss-Laguerre integration is exact for Laguerre content up to degree 2*nmu-1, so **nmu >= Nl** resolves everything GKX carries; nmu = Nl is the natural like-for-like setting, nmu = 12 vs Nl <= 8 is comfortable.
  - vpar: stella's uniform grid with trapezoid-type weights is not spectral; a uniform N-point grid resolves roughly the structure of ~N/2 Hermite modes at vpa_max=3. Practical rule: **nvpa (=2*nvgrid) ≈ 2-3x Nm**. Defaults (nvgrid=24 → nvpa=48) comfortably cover Nm=16; for Nm=32 use nvgrid=36-48. For linear benchmarks, do a joint convergence scan (stella: nvgrid, nmu; GKX: Nm, Nl) and compare converged endpoints rather than matched "resolutions".
  - Collisionless linear ITG growth rates are insensitive above nvgrid~24, nmu~12; recycling-free nonlinear fluxes typically use nvgrid=24-36, nmu=12-18 in published stella work.

## 5. Geometry

- **Options** (`&geo_knobs`, `geo_option`): `'local'`/`'miller'` (Miller, `&millergeo_parameters`: rhoc, rmaj, rgeo, qinp, shat, shift, kappa/kapprim, tri/triprim, betaprim, nzed_local spline resolution), `'vmec'` (`&vmec_parameters`), `'input.profiles'` (GA gacode). Geometric coefficient overwrite hooks via `geo_file` ('input.geometry', same format as the `.geometry` output — handy for feeding identical geometry into two codes).
- **VMEC flux tube** (`geo/vmec_geo.f90` + `geo/vmec_interface/vmec_to_stella_geometry_interface.f90`, originally by Matt Landreman, modified by Barnes):
  - `vmec_filename` = wout NetCDF; `torflux` = s = normalized toroidal flux of the surface (stella's rhoc = rhotor = sqrt(s)); `alpha0` = field-line label (alpha = theta_pest - iota*zeta), `zeta_center` recenters the tube; `surface_option=0` interpolates to exactly torflux.
  - **Parallel coordinate:** zed follows the *toroidal* angle zeta along the field line by default; `zed_equal_arc=T` remaps so b.grad z is constant (recommended for stellarators; the W7-X example uses it, with `zgrid_refinement_factor`).
  - **Tube length:** set by `nfield_periods` (a real), the number of field periods spanned in zeta. Relation to poloidal turns N_theta (manual): `nfield_periods = q * Nfp * N_theta`. A GX/GKX `npol`-poloidal-turn tube corresponds to `nfield_periods ≈ npol * q(s) * Nfp` (W7-X example: nfield_periods=7.60868 at s=0.0625 ~ one poloidal turn; the master linear example uses 17 at s=0.49 for +/-3 turns). In `&zgrid_parameters`, `nzed` is the number of z points *per 2π segment scaled unit* — total grid is nzed*(2*nperiod-1); for VMEC runs nperiod=1 and the tube length comes from nfield_periods.
  - **Signs/conventions:** `sign_torflux` = VMEC `signgs`, tracked through kx (dx/dpsi_t carries sgn(psi_t)); shat = (r/q)dq/dr with r = a*sqrt(s); gds2/gds21/gds22 and gbdrift/cvdrift/gbdrift0/cvdrift0 follow GS2 conventions (definitions documented in `docs/pages/user_manual/namelist_files/geo_knobs.nl`); all geometry arrays are dumped both to the `.geometry` ASCII file and the output NetCDF (bmag, gradpar, gbdrift, cvdrift, gds2, gds21, gds22, kperp2, jacob, ...) — the cleanest cross-check against GKX's geometry module is to diff these arrays directly.
  - **Boundary conditions** (`&zgrid_parameters:boundary_option`): `'default'` (zero incoming, linear runs), `'linked'` (standard twist-and-shift, tokamak nonlinear; jtwist ~ round(2 pi shat), dkx from shat), `'stellarator'` (twist-and-shift generalized to local shear at the tube ends — required for low-global-shear stellarators like W7-X; `dkx_over_dky` tunes jtwist), `'periodic'`. NOTE: newer inputs write `boundary_option='linked'` plus `twist_shift_option='stellarator'`; in this v0.5.1 checkout the equivalent is `boundary_option='stellarator'` (no `twist_shift_option` variable exists here).
  - Bundled equilibria: `geo/vmec_interface/equilibria/` ships `wout_w7x_standardConfig.nc` and `wout_161s1.nc` plus GIST files for cross-checking.

## 6. Input namelists (old format, matches the local build)

Key namelists: `&zgrid_parameters`, `&geo_knobs`, `&millergeo_parameters` | `&vmec_parameters`, `&physics_flags`, `&parameters`, `&vpamu_grids_parameters`, `&dist_fn_knobs` (adiabatic_option), `&time_advance_knobs` (explicit_option rk2/rk3/rk4, xdriftknob/ydriftknob/wstarknob term switches), `&kt_grids_knobs` + `&kt_grids_range_parameters` (linear: naky, aky_min/max, theta0) or `&kt_grids_box_parameters` (nonlinear: nx, ny, y0, jtwist; 2/3 dealiasing), `&init_g_knobs` (ginit_option 'default'|'noise', phiinit, restart), `&knobs` (fphi/fapar/fbpar, delt, nstep|tend, CFL cushions, implicit-solve switches, zed/vpa/time upwinding 0.02-0.05, lu_option — use 'local' on clusters), `&species_knobs` + `&species_parameters_N`, `&dissipation` (+ `&hyper`, `&collisions_dougherty`/`&collisions_fp`), `&stella_diagnostics_knobs`, `&layouts_knobs`, `&neoclassical_input`/`&sfincs_input` (off by default).

Per-case settings:

**(a) Linear ITG flux tube (CBC-like)** — `example_linear.in` in the repo root is exactly this (validated above): Miller rhoc=0.5, shat=0.796, qinp=1.4, rmaj=rgeo=2.77778; one ion species tprim=2.49, fprim=0.8 (= R/LT=6.9, R/Ln=2.2); adiabatic electrons via `adiabatic_option="field-line-average-term"`; `grid_option='range'`, naky=1, aky_min=aky_max=0.5; nzed=24, nperiod=2, boundary 'default'; nvgrid=24, nmu=12; rk3; `write_omega=.true.` Result: omega = 0.1870 + 0.0970i (v_th/a units).

**(b) Nonlinear ITG flux tube** — `example_nonlinear.in`: same Miller CBC surface; `nonlinear=.true.` in `&physics_flags`; `grid_option='box'`, nx=ny=32 (production: 128+), y0=10; `boundary_option="linked"`; `ginit_option="noise"` + zf_init; hyper_dissipation=.true. with D_hyper=0.1 (strongly recommended); Dougherty collisions; delt auto-adjusted by CFL (cfl_cushion_*), `tend` preferred over nstep for production.

**(c) Kinetic electrons** — set `nspec=2` in `&species_knobs` and add `&species_parameters_2` with `z=-1.0, mass=2.7e-4` (m_e/m_D; use 5.44e-4 for hydrogen-normalized), `type='electron'`, its own tprim/fprim. The implicit parallel-streaming solve (stream_implicit=.true., default) is what makes kinetic electrons affordable — no electron-CFL-limited timestep. `zeff` available. (The `species_parameters_2` block present in `example_linear.in` is ignored there because nspec=1.)

**(d) Electromagnetic** — NOT available in this checkout (fapar/fbpar inert; beta "currently has no effect" except through Miller betaprim in geometry). On current master: `&electromagnetic` namelist with `include_apar`, `include_bpar`, `beta`. Plan: use master on the office Linux box for any EM cross-validation; keep the pinned version for electrostatic anchors.

**W7-X-like example (old format, runs with local build after replacing `twist_shift_option='stellarator'` by `boundary_option='stellarator'`):** `AUTOMATIC_TESTS/convert_input_files/example_VMEC_nonlinear_W7X_v0.5.in` on master — geo_option='vmec', wout_w7xr003.nc, torflux=0.0625, nfield_periods=7.60868, zed_equal_arc=T, box 6x6 y0=10 (toy resolution), kinetic electrons, adiabatic_option="iphi00=2", flux_norm=F, hyper_dissipation. A new-format linear W7-X single-mode example (H. Thienpondt) is at `POST_PROCESSING/stellapy/examples/LINEAR_W7X_SINGLEMODE/input.in` on master (wout_w7x_standardConfig.nc — the same file shipped in this repo's `geo/vmec_interface/equilibria/` — torflux=0.49, nfield_periods=17, +/-3 poloidal turns).

## 7. Outputs and post-processing

ASCII (always/optionally, run name prefix): `.out` (|phi|^2 vs t), `.omega` (columns: time, ky, kx, Re[om], Im[om], Re[omavg], Im[omavg]; omavg averaged over `navg` steps — the converged linear growth rate/frequency), `.fluxes` (columns: time, then pflx, vflx, qflx per species — the primary nonlinear time-trace), `.final_fields`, `.geometry` (all geometric coefficients vs zed), `.species.input`, `millerlocal.*` (Miller diagnostics).

NetCDF `<run>.out.nc` (via neasyf), verified by ncdump of an actual run:
- Dims: ky, kx, tube, zed, alpha, vpa, mu, species, t (+ri for complex).
- Grids/geometry: kx, ky (units 1/rho_r), zed, theta0, bmag, b_dot_grad_z, gradpar, gbdrift(0), cvdrift(0), gds2, gds21, gds22, grho, jacob, kperp2(zed,alpha,kx,ky), q, shat, drhodpsi, jtwist, beta; species charge/mass/dens/temp/tprim/fprim/vnew; full copy of the input file in `input_file`.
- Time series: `phi2(t)`; `omega(t,kx,ky,ri)` [write_omega]; `phi_vs_t(t,tube,zed,kx,ky,ri)` [write_phi_vs_time]; `phi2_vs_kxky(t,kx,ky)` [write_kspectra]; `gvmus(t,species,mu,vpa)` and `gzvs(t,species,vpa,zed,tube)` [velocity-space checks]; moments dens/upar/temp [write_moments]; `pflux_x/vflux_x/qflux_x(t,species,kx)` [write_radial_fluxes]; z- and mode-resolved fluxes `pflx_kxky/vflx_kxky/qflx_kxky(t,species,tube,zed,kx,ky)` [write_fluxes_kxkyz]. Note: in this version the *total* flux time trace lives in the ASCII `.fluxes` file; NetCDF holds the resolved versions. Restarts: `save_for_restart` + `nsave`, files under `restart_dir` (`nc/`), appended-safe on restart.
- Linear-run analysis: read `omega` (last navg-averaged value) or tail of `.omega`; both were used above to validate the build.
- Post-processing shipped in-repo: `post_processing/*.py` — `stella_data.py` (nc loader), `stella_plots.py`, `kspectra_plots.py`/`kspectra_movie.py`, `fluxes_stats.py`, `moments.py`, `gvmus_movie.py`, `gzvs_movie.py`, `RH.py` (Rosenbluth-Hinton residual check), `symmetry.py`, `zonal.py`, plus `fluxes.f90`. Separately, **stellapy** (in-repo python suite by Hanne Thienpondt, GUI+CLI; much expanded under `POST_PROCESSING/stellapy` on master with worked example runs W7-X linear/nonlinear).

## 8. Literature anchors for GKX cross-validation

1. **M. Barnes, F. I. Parra, M. Landreman, JCP 391, 365-380 (2019)** — the stella method paper. Anchors: linear CBC-type Miller benchmarks vs GS2 (growth-rate spectra, incl. kinetic electrons), Rosenbluth-Hinton zonal-flow residual, W7-X-geometry tests, nonlinear cyclone heat flux vs GS2. The repo's `example_linear.in`/`example_nonlinear.in` and `tests/regression/linear/RH/` map onto these.
2. **A. González-Jerez, P. Xanthopoulos, J. M. García-Regaña, I. Calvo, J. Alcusón, A. Bañón Navarro, A. von Stechow, H. Thienpondt, J. Plasma Phys. 88, 905880310 (2022)** — "Electrostatic gyrokinetic simulations in Wendelstein 7-X geometry: benchmark between the codes stella and GENE." The canonical W7-X flux-tube anchor: linear ITG growth-rate spectra, zonal-flow relaxation, and nonlinear ITG heat fluxes in W7-X standard configuration (adiabatic electrons). The bundled `wout_w7x_standardConfig.nc` is the right equilibrium family for reproducing this.
3. **Rosenbluth-Hinton residual** — built-in regression case `tests/regression/linear/RH/RH.in` + `post_processing/RH.py`; cheap first cross-check of zonal dynamics and geometry wiring against GKX.
4. **D. A. St-Onge, M. Barnes, F. I. Parra, JCP 468, 111498 (2022)** — radially global stella (useful later if GKX explores beyond-flux-tube effects; not needed for item 2.3).
5. **GX method paper (Mandell, Dorland et al., J. Plasma Phys. 2024; arXiv:2209.06731)** — GX's own CBC and W7-X benchmarks against GS2/GENE/stella. Since GKX shares GX's Hermite-Laguerre formulation and wout conventions (a_ref = Aminor_p, B_ref = 2|psi_a|/a^2 — identical to stella's GIST convention, confirmed in stella source), these cases triangulate GKX-vs-stella directly: same wout file, same s, same field line, ky grids in the same rho_ref units, times in the same a/v_th units (stella uses v_th = sqrt(2T/m); **CORRECTION 2026-08-18: GX does NOT — GX shares GKX's v_th=sqrt(T/m). See plan.md finding #1**; still worth a one-line unit sanity check in GKX before quoting discrepancies).
6. (Physics-application anchor, optional) **H. Thienpondt et al., Phys. Rev. Research 5, L022053 (2023)** and related stella W7-X turbulence papers from the Ciemat group — nonlinear W7-X flux levels with kinetic electrons, useful qualitative context.

## 9. Status vs plan item 2.3 (first half)

- Clone: pre-existing, preserved; commit recorded; submodules present.
- Build: SUCCESS on macOS arm64/MacPorts (CMake out-of-tree, `build_cmake/stella`); no system packages needed; documented gotchas above. Old broken binary backed up.
- Validation: linear CBC ITG reproduces the machine's own archived stella results exactly (omega = 0.18703 + 0.09703i at ky=0.5); nonlinear box smoke test passes on 4 MPI ranks.
- Deliverable next (second half of 2.3): pick the concrete GKX<->stella comparison cases (suggest: (i) RH residual, (ii) CBC linear ky scan with adiabatic + kinetic electrons, (iii) W7-X linear ky scan at s=0.49 with wout_w7x_standardConfig.nc, (iv) nonlinear CBC heat flux), and decide pinned-v0.5.1 vs master on the office Linux host (master required only for EM).
