"""Does the zonal-flow response predict nonlinear flux where linear proxies fail?

Motivation, measured on the tracked RBC(1,1) boundary landscape (24 replicated
nonlinear windows, 1.6% noise floor, 2.6x spread in heat flux):

    proxy                              Pearson r   Spearman
    quasilinear linear-weight            -0.570     -0.707
    quasilinear shape-aware power law    -0.570     -0.707
    quasilinear mixing length            -0.205     -0.031
    linear growth rate                   -0.200      0.062

Every quasilinear proxy is **anticorrelated** with the nonlinear heat flux.
Minimizing them raises transport. That is not noise: the anticorrelation holds in
both halves of the scan (-0.54, -0.77) and is not explained by distance from the
baseline (+0.40).

The physics those proxies omit is the saturation mechanism. ITG turbulence
saturates through zonal flows, and along this landscape the shaping that mildly
raises linear drive also strengthens the zonal-flow response, which wins. A
predictor built on the drive alone therefore points the wrong way.

This script tests the alternative: the collisionless **Rosenbluth-Hinton zonal
residual**, which measures how much of an initial zonal perturbation survives
after geodesic-acoustic transients. It is a linear, collisionless initial-value
calculation -- seconds, not hours -- and it is differentiable, so it can drive an
optimizer.

Hypothesis: R_ZF rises along the landscape where nonlinear flux falls, giving a
*positive* predictor of confinement where the quasilinear proxies give a
negative one.

References for the residual and its role in stellarator saturation:
Rosenbluth & Hinton, Phys. Rev. Lett. 80, 724 (1998); Xiao & Catto, Phys.
Plasmas 13, 102311 (2006); Xanthopoulos et al., Phys. Rev. Lett. 125, 265001
(2020) on zonal-flow control of stellarator ITG transport.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

_LANDSCAPE = Path("docs/_static/vmec_boundary_transport_landscape_rbc11_full.json")
_RBC_LINE = re.compile(r"^(\s*RBC\(1,1\)\s*=\s*)(\S+)\s*$", re.MULTILINE)


def landscape_points() -> list[dict]:
    """Nonlinear windows joined to the reduced linear metrics at the same point."""

    data = json.loads(_LANDSCAPE.read_text())
    rows = {round(r["coefficient_value"], 10): r for r in data["rows"]}
    points = []
    for entry in data["nonlinear_ensemble_points"]:
        if not entry.get("passed"):
            continue
        row = rows.get(round(entry["coefficient_value"], 10))
        if row is None:
            continue
        points.append(
            {
                "fraction": row["relative_fraction"],
                "coefficient": entry["coefficient_value"],
                "flux": entry["mean"],
                "flux_sem": entry["sem"],
                "metrics": row.get("reduced_metrics") or {},
            }
        )
    points.sort(key=lambda p: p["fraction"])
    return points


def write_input(template: Path, coefficient: float, destination: Path) -> Path:
    """Clone a scan input with RBC(1,1) replaced.

    The shipped decks differ from one another in exactly this one line, so a
    reconstruction is faithful rather than approximate -- verified by diffing two
    of the surviving inputs.
    """

    text = template.read_text()
    replacement, count = _RBC_LINE.subn(
        lambda m: f"{m.group(1)}{coefficient:.16E}", text
    )
    if count != 1:
        raise ValueError(f"expected exactly one RBC(1,1) line, found {count}")
    destination.write_text(replacement)
    return destination


def zonal_residual(
    equilibrium, *, kx: float, hermite: int, laguerre: int, t_max: float, dt: float,
    s_index: int = 7, ntheta: int = 32,
) -> dict[str, float]:
    """Collisionless Rosenbluth-Hinton residual for one equilibrium.

    Protocol follows the pitfalls established for this measurement:

    * average the **signed complex** zonal potential, Jacobian-weighted along the
      field line -- ``|phi|`` rectifies the geodesic-acoustic oscillation and
      overestimates the residual by roughly 3x;
    * the GAM does not damp at moderate ``q``, so extract the residual by fitting
      ``R + A exp(-gamma t) cos(omega t + phi)`` rather than window-averaging,
      which swings with the window;
    * keep the fit window strictly before Hermite recurrence at
      ``t_rec ~ 2 sqrt(N_m) / k_par``.
    """

    import jax.numpy as jnp
    from scipy.optimize import curve_fit

    from vmex.core import turbulence as turb

    from gkx.config import GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.terms.fields import solve_fields

    # Same VMEC -> flux-tube mapping the quasilinear objective uses, so the
    # zonal measurement and the proxy see identical geometry.
    geometry = turb.flux_tube_geometry(
        equilibrium.state, equilibrium.runtime, s_index=s_index, alpha=0.0, ntheta=ntheta
    )
    grid = build_spectral_grid(
        GridConfig(Nx=3, Ny=2, Nz=ntheta, Lx=2.0 * np.pi / max(kx, 1e-12), Ly=1.0e4)
    )
    params = linear_params_for_geometry(geometry, tau_e=1.0)
    cache = build_linear_cache(grid, geometry, params, laguerre, hermite)

    # Zonal branch only: no drive, no collisions. The residual is a property of
    # the collisionless neoclassical polarization.
    terms = LinearTerms(
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )
    state = jnp.zeros(
        (1, laguerre, hermite, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex128,
    ).at[0, 0, 0, 0, 1, :].set(1.0e-6)

    unit = jnp.asarray([1.0])
    jacobian = np.asarray(cache.jacobian, dtype=float)
    weight = jacobian / max(jacobian.sum(), 1e-30)

    def zonal_potential(current) -> complex:
        fields = solve_fields(
            current,
            cache,
            params,
            charge=unit,
            density=unit,
            temp=unit,
            mass=unit,
            tz=unit,
            vth=unit,
            fapar=jnp.asarray(0.0),
            w_bpar=jnp.asarray(0.0),
        )
        # signed complex, Jacobian-weighted flux-surface average at (ky=0, kx>0)
        return complex(np.sum(np.asarray(fields.phi)[0, 1, :] * weight))

    # Fuse the RK4 step and sample sparsely. Converting the field to NumPy every
    # step dominated the cost; the residual fit needs a few hundred samples, not
    # one per step.
    import jax

    @jax.jit
    def advance(current):
        def rhs(s):
            return linear_rhs_cached(s, cache, params, terms=terms, use_jit=False)[0]

        k1 = rhs(current)
        k2 = rhs(current + 0.5 * dt * k1)
        k3 = rhs(current + 0.5 * dt * k2)
        k4 = rhs(current + dt * k3)
        return current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @jax.jit
    def chunk(current):
        return jax.lax.fori_loop(0, sample_stride, lambda _, s: advance(s), current)

    steps = int(round(t_max / dt))
    sample_stride = max(steps // 400, 1)
    n_samples = steps // sample_stride
    trace = np.empty(n_samples + 1, dtype=complex)
    trace[0] = zonal_potential(state)
    for index in range(n_samples):
        state = chunk(state)
        trace[index + 1] = zonal_potential(state)
    dt_sample = dt * sample_stride

    times = np.arange(n_samples + 1) * dt_sample
    signal = trace.real / (abs(trace[0]) + 1e-30)

    gradpar = float(np.asarray(geometry.gradpar()))
    t_rec = 2.0 * np.sqrt(hermite) / max(abs(gradpar), 1e-6)
    fit_mask = times < min(t_max, 0.85 * t_rec)

    def model(t, residual, amplitude, damping, omega, phase):
        return residual + amplitude * np.exp(-damping * t) * np.cos(omega * t + phase)

    try:
        popt, _ = curve_fit(
            model,
            times[fit_mask],
            signal[fit_mask],
            p0=(0.1, 0.9, 0.05, 1.0, 0.0),
            maxfev=200_000,
        )
        residual = float(popt[0])
    except Exception as err:
        raise RuntimeError(f"residual fit failed: {type(err).__name__}: {err}") from err

    return {
        "residual": residual,
        "t_rec": float(t_rec),
        "gradpar": gradpar,
        "final_signed_mean": float(signal[fit_mask][-len(signal[fit_mask]) // 4 :].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--kx", type=float, default=0.05)
    parser.add_argument("--hermite", type=int, default=25)
    parser.add_argument("--laguerre", type=int, default=11)
    parser.add_argument("--t-max", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/zonal_flow_saturation_model.json")
    )
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt

    args.workdir.mkdir(parents=True, exist_ok=True)
    points = landscape_points()
    if args.limit:
        points = points[:: max(len(points) // args.limit, 1)][: args.limit]
    print(f"landscape points: {len(points)}", flush=True)

    for point in points:
        label = f"f{point['fraction']:+.2f}".replace(".", "p")
        deck = write_input(
            args.template, point["coefficient"], args.workdir / f"input.{label}"
        )
        equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(deck))
        try:
            measured = zonal_residual(
                equilibrium,
                kx=args.kx,
                hermite=args.hermite,
                laguerre=args.laguerre,
                t_max=args.t_max,
                dt=args.dt,
            )
        except Exception as err:  # pragma: no cover - reported, not hidden
            measured = {"residual": float("nan"), "error": f"{type(err).__name__}: {err}"[:200]}
            print(f"  !! {label}: {measured['error']}", flush=True)
        point.update(measured)
        print(
            f"  f={point['fraction']:+.2f}  Q_nl={point['flux']:.2f}  "
            f"R_ZF={point.get('residual', float('nan')):.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(points, indent=2) + "\n")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
