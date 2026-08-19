# stella vs GKX — CBC linear ITG cross-validation (rung 1)

Date: 2026-08-18. Machine: macOS arm64, CPU JAX 0.9.2 (x64 enabled), stella via
/opt/local/bin/mpirun -np 4, binary /Users/rogeriojorge/local/stella/build_cmake/stella.
GKX at /Users/rogeriojorge/local/GKX (READ-ONLY), PYTHONPATH=/Users/rogeriojorge/local/GKX/src.

## Case
CBC in a-units: Miller rhoc=0.5, shat=0.796, q=1.4, rmaj=rgeo=2.77778, kappa=1,
delta=0, tprim=2.49, fprim=0.8, adiabatic electrons (field-line-average term),
collisionless, electrostatic.

## Runs
- stella_single/cbc_ky0p5.in: copy of stella repo example_linear.in.
  Result (t=60, omavg): omega=0.187022, gamma=0.097030  [vth/a, vth=sqrt(2T/m)]
  Validated target 0.18703+0.09703i: REPRODUCED.
- stella_scan/cbc_scan6.in: grid_option='range', naky=6, aky_min=0.2, aky_max=0.7
  -> ky = 0.2,0.3,0.4,0.5,0.6,0.7. (An earlier 5-point scan cbc_scan.in gave
  evenly spaced 0.2..0.7 in steps of 0.125 — kept for reference.)
- gkx/cbc_miller_collisionless.toml: GKX runtime config matched to stella case.
  Miller path (model="miller"), collisionless (collisions=false,
  hypercollisions=false, [terms] collisions=0 hypercollisions=0, nu*=0),
  Nl=12 (Laguerre), Nm=48 (Hermite), solver=time, dt=0.002, t_max=40 (20000 steps),
  imex2, linked boundary, ntheta=32, nperiod=2 (Nz=96), x64.
  Single point: run-runtime-linear --ky 0.5 -> gamma=0.138381, omega=0.404540 [cs/a].
- gkx/cbc_miller_scan_sqrt2.toml: same but y0=14.14213562 (dky=1/y0=0.070711)
  so grid ky hit stella_ky/sqrt(2) exactly; scan ky =
  [0.14142136,0.21213203,0.28284271,0.35355339,0.42426407,0.49497475];
  scan-runtime-linear --batch-ky.
- gkx/cbc_miller_geom.eik.nc: gkx geometry miller export of the matched Miller
  geometry (GS2-convention coefficient names/values; agrees with stella
  .geometry to <=1.2% pointwise).

## Convention finding (verified in source + empirically)
GKX normalizes to v_ref = sqrt(T/m) and rho_ref = sqrt(T*m)/|q| —
src/gkx/operators/linear/params.py lines 264-265:
    vth=jnp.sqrt(temperature / mass),
    rho=jnp.sqrt(temperature * mass) / jnp.abs(charge),
i.e. the GENE-like cs/rho_s family, NOT the GS2/GX vth=sqrt(2T/m) family that
stella uses. Mapping for the same physical mode:
    ky_stella = sqrt(2) * ky_gkx_input
    (gamma, omega)[vth/a] = (gamma, omega)[cs/a] / sqrt(2)
Empirical check at GKX ky=0.5: (0.138381, 0.404540)/sqrt2 = (0.097850, 0.286053)
vs stella at ky=0.7071 (interp): (~0.0964, ~0.2802) -> ~1.5% / ~2.1%.

## Tracked GKX cyclone lane (docs/_static/cyclone_mismatch_table.csv)
Different physical case: s-alpha (eps=0.18, s_hat=0.8) + hypercollisional
settings (nu_hermite=1, nu_laguerre=2), ky in GKX rho_s units. Not directly
comparable to the collisionless Miller CBC without the sqrt(2) remap and a
collisionality caveat. Reproduction run of that lane at Nl24/Nm12, t=10,
ky=0.3: gamma=0.063224 omega=0.300887 (tracked CSV at converged resolution:
0.092646, 0.283882; GX audit quote: 0.101814, 0.286777).
