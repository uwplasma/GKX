# Finite-beta QA equilibria + GKX geometry smoke test (plan item 2.2 start)

Date: 2026-08-18. Machine: macOS arm64, CPU JAX 0.9.2, vmex 0.5.0.
GKX: /Users/rogeriojorge/local/GKX @ feat/bounded-memory-nonlinear-adjoint (read-only).
All artifacts in this directory.

## 1. Input decks

Base: GKX `examples/vmec/input.LandremanPaul2021_QA_lowres` (Landreman-Paul 2021
precise QA, R0~1 m, a~1/6 m, PHIEDGE=0.08386 Wb -> <B>~1 T, NFP=2, MPOL=NTOR=8,
NS to 50). Copy kept as `base_input_copy`.

Common changes to both decks (convergence):
- `NITER_ARRAY = 2000 4000 12000` (base 600/1000/1000 hit the cap at finite beta)
- `FTOL_ARRAY = 1e-8 1e-10 1e-12` (base final 1e-13; 1e-12 converges in ~1 min,
  final fsqr/fsqz/fsql all < 1e-12)

### input.LandremanPaul2021_QA_beta2 (pressure only)
- `PMASS_TYPE='power_series'`, `AM = 2.2e4 -4.4e4 2.2e4`, `PRES_SCALE=1`
  i.e. p(s) = 2.2e4 * (1-s)^2 Pa - smooth, peaked, p(1)=0, p'(1)=0.
  Sizing: <p> = p0/3 ~ 7.3 kPa vs <B^2>/2mu0 ~ 3.7e5 Pa -> <beta> ~ 2%.
- `NCURR=1`, `CURTOR=0` (zero net toroidal current; wout ctor = -8e-12 A).

### input.LandremanPaul2021_QA_beta2_current (pressure + current)
- Same pressure profile.
- `NCURR=1`, `CURTOR=-1.0e4` A, `PCURR_TYPE='power_series'`, `AC = 1.0 -1.0`
  i.e. I'(s) ~ (1-s): on-axis-peaked current density, 10 kA total.
- Sign note: with CURTOR=+1e4 the axis iota collapsed to 0.055
  (q(s=0.25)~15, degenerate for flux tubes), so the sign was flipped so the
  current RAISES iota. Estimate mu0*I*R0/(2 pi a^2 B) ~ 0.075 at the edge;
  the on-axis effect is larger because the current density is peaked.

## 2. Equilibrium properties (from wout)

| quantity                | beta2 (pressure only) | beta2_current (p + 10 kA) |
|-------------------------|-----------------------|---------------------------|
| converged               | yes (fsq* < 1e-12)    | yes (fsq* < 1e-12)        |
| betatotal (<beta>)      | 1.906 %               | 1.892 %                   |
| ctor (net current)      | -8e-12 A (zero)       | -1.0003e4 A               |
| iota axis               | 0.2867                | 0.4565                    |
| iota edge               | 0.4388                | 0.4738                    |
| iota @ s=0.25 / 0.64    | 0.327 / 0.405         | 0.467 / 0.476             |
| aspect ratio            | 6.000                 | 6.000                     |
| volavgB                 | 0.999 T               | 0.999 T                   |
| p(0) / p(1)             | 21993 / ~0 Pa         | same                      |
| min interior DMerc      | -438                  | -815                      |

Vacuum LP QA reference iota ~ 0.42 (flat): pressure alone (NCURR=1, zero J)
depresses core iota to 0.287; the 10 kA co-iota current restores and raises
it (axis +0.17, edge +0.035) - a clearly noticeable iota change while staying
far from low-order rationals at the smoke-test surfaces. Beta within 5% of
the 2% target for both -> no pres_scale iteration needed. Negative DMerc
spots are expected for an unoptimized 2%-beta QA; harmless for geometry
smoke tests but worth recording.

Runs: `vmex input.<name>` (~50-65 s solver time each); logs `run_beta2.log`,
`run_beta2_current.log`; wouts `wout_LandremanPaul2021_QA_beta2.nc`,
`wout_LandremanPaul2021_QA_beta2_current.nc`.

## 3. GKX geometry smoke tests

Configs `runtime_{beta2,beta2cur}_tf{0p25,0p64}.toml` are the HSX linear
quasilinear template (`examples/linear/non-axisymmetric/
runtime_hsx_linear_quasilinear.toml`) with vmec_file -> the new wouts,
torflux in {0.25, 0.64}, output redirected here. GKX geometry defaults:
include_shear_variation=false, include_pressure_variation=false, betaprim unset.

### eik generation (`python3 -m gkx.cli geometry vmec --config ... --out ...`)
All 4 cases succeed; all 27 variables in each eik.nc are finite.

| eik file               | max abs(cvdrift-gbdrift) | bmag range    |
|------------------------|--------------------------|---------------|
| eik_beta2_tf0p25.nc    | 0.0988                   | 0.972 - 1.088 |
| eik_beta2_tf0p64.nc    | 0.0759                   | 0.972 - 1.157 |
| eik_beta2cur_tf0p25.nc | 0.0974                   | 0.979 - 1.092 |
| eik_beta2cur_tf0p64.nc | 0.0757                   | 0.974 - 1.149 |

cvdrift != gbdrift with the expected sign (dp/ds<0 makes gbdrift more
negative): the wout-file geometry path IS applying a finite-beta pressure
split between curvature and grad-B drifts.

### short linear runs (`run-runtime-linear`, ky=0.5238, t_max=2, rk4 dt=0.005)
All 4 complete without error; gamma/omega finite (no NaN anywhere):

| case              | gamma    | omega   |
|-------------------|----------|---------|
| beta2 tf=0.25     | -0.0896  | -0.2484 |
| beta2 tf=0.64     | -0.1009  | -0.2614 |
| beta2cur tf=0.25  | +0.0180  | -0.2108 |
| beta2cur tf=0.64  | +0.0331  | -0.1904 |

(t_max=2 is a smoke test, not a converged growth-rate fit; values are finite
and case-dependent, which is what was being checked. Logs linear_*.log,
outputs out_*/.)

## 4. Where finite-beta terms live / what remains open

GKX has TWO VMEC geometry paths:

A. wout-file runtime path (used by `gkx.cli geometry vmec` and runtime
   linear/nonlinear runs): `src/gkx/geometry/imported_vmec.py` ->
   `vmec_boozer_derivatives.py` + `vmec_field_line_sampling.py`.
   This path is ALREADY finite-beta aware, unconditionally:
   - pressure read from wout `pres` and splined:
     src/gkx/geometry/vmec_field_line_sampling.py:112-116
   - mu0*dp/ds in the normal curvature kappa_n:
     src/gkx/geometry/vmec_field_line_sampling.py:636-638
   - HNGC beta_b Boozer-mode correction in kappa_n:
     src/gkx/geometry/vmec_field_line_sampling.py:639-641 (beta_b built from
     d_pressure_d_s in src/gkx/geometry/vmec_boozer_derivatives.py:540-551)
   - gbdrift = cvdrift + 2*Bref*Lref^2*sqrt(s)*mu0*pfac*(dp/ds)*sign(psi)/
     (psi_edge/2pi * B^2): src/gkx/geometry/vmec_field_line_sampling.py:728-737
     (pfac=1 unless the HNGC betaprim override is on); gbdrift0 == cvdrift0
     by construction (lines 740-744), which is correct (both are grad-psi
     components of the same curvature drift).
   - The config flags `include_pressure_variation` / `betaprim`
     (src/gkx/config.py:136-137, default off) gate ONLY the Hegna-Nakajima
     local-equilibrium override (d_pressure_d_s_1 / pfac scaling,
     vmec_boozer_derivatives.py:354-372) - NOT the physical equilibrium
     dp/ds terms above. Smoke tests above ran with defaults and still got
     the pressure split.

B. vmex-state differentiable bridge (used by the adjoint/optimization stack:
   src/gkx/objectives/vmec_boozer_context.py:119, vmec_transport*.py,
   solver_vmec.py, and gkx/geometry/vmec_flux_tube_reports.py):
   `src/gkx/geometry/vmec_boozer_core.py` - this is the code carrying the
   "finite-beta pressure corrections and broad-equilibrium drift gates
   remain open" note (vmec_boozer_core.py:836-837 and 912-913). It is
   ZERO-BETA ONLY. It does not error on finite-beta input; it silently
   drops the pressure terms.

### Precise changes needed for correct finite-beta drifts in path B
(src/gkx/geometry/vmec_boozer_core.py; reference implementation is path A)

1. Plumb a pressure profile into the bridge:
   - `_BoozerRadialProfiles` dataclass (vmec_boozer_core.py:144-158): add
     `d_pressure_ds` (and optionally `pressure`).
   - `_interpolate_boozer_radial_profiles` (vmec_boozer_core.py:426-480):
     interpolate p(s) and dp/ds at `request.torflux` the same way iota /
     d_iota_ds are done at lines 449-455. Source: differentiably from the
     vmex input profile (`inp.am`, `pres_scale` - preferred for the adjoint
     path) or from wout `pres` as in path A.

2. Add the mu0*dp/ds term to the normal curvature:
   - `_raw_drift_profiles`, kappa_n at vmec_boozer_core.py:704-708.
     Currently: kappa_n = dB/ds/(B*etf) + L0*kappa_g.
     Needed:    kappa_n = (B*dB/ds + mu0*dp/ds)/(B^2*etf) + L0*kappa_g
     (compare vmec_field_line_sampling.py:636-638). This changes cvdrift and
     cvdrift0 through b_cross_kappa_dot_grad_alpha/psi (lines 709-713).

3. Split gbdrift from cvdrift:
   - `_pack_metric_drift_profiles` at vmec_boozer_core.py:763-766 currently
     aliases `gbdrift=cvdrift`, `gbdrift0=cvdrift0`. gbdrift needs the
     pressure offset of vmec_field_line_sampling.py:728-737:
     gbdrift = cvdrift + 2*Bref*L^2*sqrt(s)*mu0*(dp/ds)*sign(etf)/(etf*B^2)
     applied on the raw (pre-equal-arc-remap) profile in
     `_raw_drift_profiles` (vmec_boozer_core.py:714-731) so it is remapped
     consistently. `gbdrift0=cvdrift0` may stay aliased (matches path A).

4. Broad-equilibrium drift gates (second half of the open note; larger job):
   - the HNGC beta_b mode-correction term in kappa_n
     (vmec_field_line_sampling.py:639-641; beta_b assembled in
     vmec_boozer_derivatives.py:540-551 from gmnc_b, Vprime, dp/ds) has no
     counterpart in the bridge - requires carrying the Boozer gmnc_b
     spectrum and Vprime through `_run_boozer_transform_from_state`;
   - the local-shear correction D_HNGC (vmec_field_line_sampling.py:586-598,
     pressure part via d_pressure_d_s_1) is absent from `local_shear_l0`
     (vmec_boozer_core.py:702-703);
   - once implemented, update the scope strings at vmec_boozer_core.py:837
     and 913 and add a finite-beta parity gate comparing path B against
     path A on these two wouts (they are a ready-made fixture: same
     boundary, ~2% beta, one zero-current and one 10 kA).

Suggested first gate: items 1-3 give the leading-order finite-beta drift
physics (the cvdrift/gbdrift split here is ~0.08-0.10 in GKX units, i.e. a
20-25% effect on the drift amplitude); item 4 is a smaller correction at
this beta but is what "broad-equilibrium drift gates" refers to.

## 5. Deliverables in this directory

- input.LandremanPaul2021_QA_beta2, input.LandremanPaul2021_QA_beta2_current
- wout_LandremanPaul2021_QA_beta2.nc, wout_LandremanPaul2021_QA_beta2_current.nc
- run_beta2.log, run_beta2_current.log (vmex logs)
- runtime_*.toml (4 smoke-test configs), eik_*.nc (4 generated geometries)
- linear_*.log + out_*/ (4 short linear runs)
- base_input_copy (unmodified base deck, provenance)
- findings.md (this file)
