"""Run a physical nonlinear heat-flux descent through a VMEC state control.

The differentiated path is

``vmex state coefficient -> booz_xform_jax geometry -> GKX cache -> projected
production RK2 map -> physical heat-flux window``.

The initial condition is a production-CFL saturated state at the base control.
AD is checked against a centered finite difference, then a short line search is
made in the negative-gradient direction.  This proves a finite-window
state-control descent only: the perturbed states are not re-equilibrated VMEC
boundaries and the window derivative is not an infinite-time turbulent
gradient.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts.build_vmec_boozer_nonlinear_window_fd_audit import (
    geometry_response_metrics,
    write_flux_tube_geometry_netcdf,
)
from tools.campaigns.nonlinear_saturated_state import saturation_report


def _runtime_config(
    geometry_file: Path,
    *,
    nx: int,
    ny: int,
    nz: int,
    t_max: float,
    random_seed: int,
):
    from gkx.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
    from gkx.workflows.runtime.config import (
        RuntimeCollisionConfig,
        RuntimeConfig,
        RuntimeNormalizationConfig,
        RuntimePhysicsConfig,
        RuntimeSpeciesConfig,
        RuntimeTermsConfig,
    )

    return RuntimeConfig(
        grid=GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=20.0, Ly=20.0, boundary="periodic"),
        time=TimeConfig(
            t_max=t_max,
            dt=0.01,
            method="rk2",
            sample_stride=20,
            diagnostics_stride=20,
            use_diffrax=False,
            fixed_dt=False,
            dt_max=0.05,
            cfl=0.9,
            cfl_fac=1.0,
            compressed_real_fft=True,
            laguerre_nonlinear_mode="grid",
        ),
        geometry=GeometryConfig(
            model="imported-netcdf", geometry_file=str(geometry_file)
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-3,
            init_single=False,
            random_seed=random_seed,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=2.49, fprim=0.8),),
        collisions=RuntimeCollisionConfig(
            D_hyper=0.05,
            p_hyper_kperp=2.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=1.0,
            nu_hyper_m=1.0,
            p_hyper_m=4.0,
            damp_ends_amp=0.1,
            damp_ends_widthfrac=0.125,
        ),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="rho_star"
        ),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            electrostatic=True,
            electromagnetic=False,
            adiabatic_electrons=True,
        ),
        terms=RuntimeTermsConfig(
            collisions=0.0,
            hypercollisions=1.0,
            hyperdiffusion=1.0,
            end_damping=1.0,
            apar=0.0,
            bpar=0.0,
            nonlinear=1.0,
        ),
    )


def _context(args: argparse.Namespace) -> dict[str, Any]:
    from gkx.objectives.vmec_boozer_context import (
        _mode21_vmec_boozer_linear_context,
    )

    return _mode21_vmec_boozer_linear_context(
        case_name=args.case_name,
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        parameter_family=args.parameter_family,
        surface_index=args.surface_index,
        ntheta=args.nz,
        mboz=args.mboz,
        nboz=args.nboz,
        surface_stencil_width=args.surface_stencil_width,
        n_laguerre=args.nl,
        n_hermite=args.nm,
    )


def _saturated_state(
    args: argparse.Namespace,
    context: dict[str, Any],
    base_geometry: Any,
) -> tuple[Any, float, float, dict[str, Any], Any]:
    """Load or generate the base-control production saturated state."""

    import jax.numpy as jnp

    if args.saturated_state is not None:
        archive = np.load(args.saturated_state)
        if not bool(archive["saturated"]):
            raise SystemExit("provided state is not marked saturated")
        recorded_method = str(archive["method"]) if "method" in archive else None
        if recorded_method is not None and recorded_method != "rk2":
            raise SystemExit(
                f"state was produced with {recorded_method}; this campaign uses rk2"
            )
        runtime = _runtime_config(
            Path("unused.nc"),
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            t_max=args.saturation_time,
            random_seed=args.random_seed,
        )
        return (
            jnp.asarray(archive["state"]),
            float(archive["adaptive_dt"]),
            float(archive["tau_ac"]),
            {"loaded": True, "saturated": True},
            runtime,
        )

    from gkx import run_runtime_nonlinear

    if args.state_out is None:
        raise SystemExit("--state-out is required when generating saturation")
    with tempfile.TemporaryDirectory(prefix="gkx-vmec-control-") as temp_dir:
        geometry_file = Path(temp_dir) / "base_geometry.nc"
        write_flux_tube_geometry_netcdf(base_geometry, geometry_file)
        runtime = _runtime_config(
            geometry_file,
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            t_max=args.saturation_time,
            random_seed=args.random_seed,
        )
        started = time.time()
        result = run_runtime_nonlinear(
            runtime,
            Nl=args.nl,
            Nm=args.nm,
            diagnostics=True,
            return_state=True,
        )
        elapsed = time.time() - started
    if result.diagnostics is None or result.state is None:
        raise RuntimeError("production saturation run returned no state/diagnostics")
    times = np.asarray(result.diagnostics.t, dtype=float)
    flux = np.asarray(result.diagnostics.heat_flux_t, dtype=float)
    steps_dt = np.asarray(result.diagnostics.dt_t, dtype=float)
    report = saturation_report(times, flux, min_tau_multiples=5.0)
    report["wall_seconds"] = elapsed
    if not report["saturated"]:
        raise SystemExit(f"VMEC/Boozer base run did not saturate: {report}")
    adaptive_dt = float(np.nanmedian(steps_dt))
    tau_ac = float(report["tau_ac"])
    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.state_out,
        state=np.asarray(result.state),
        saturated=True,
        t_end=float(times[-1]),
        adaptive_dt=adaptive_dt,
        method="rk2",
        tau_ac=tau_ac,
    )
    return result.state, adaptive_dt, tau_ac, report, runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--parameter-family", default="Rcos")
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=22)
    parser.add_argument("--saturation-time", type=float, default=400.0)
    parser.add_argument("--saturated-state", type=Path, default=None)
    parser.add_argument("--state-out", type=Path, default=None)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument(
        "--ad-mode", choices=("reverse", "forward", "both"), default="reverse"
    )
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument(
        "--line-steps", type=float, nargs="+", default=[1.0e-8, 3.0e-8, 1.0e-7]
    )
    parser.add_argument("--max-geometry-change", type=float, default=0.05)
    parser.add_argument("--max-ad-fd-relative-error", type=float, default=1.0e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.max_geometry_change <= 0.0
        or args.max_ad_fd_relative_error <= 0.0
        or any(step <= 0.0 for step in args.line_steps)
    ):
        raise SystemExit("descent tolerances must be positive")

    import jax
    import jax.numpy as jnp

    from gkx.core.grid import build_spectral_grid
    from gkx.diagnostics import fieldline_quadrature_weights, heat_flux_total
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.nonlinear.projection import _make_hermitian_projector
    from gkx.runtime import build_runtime_linear_params, build_runtime_term_config
    from gkx.solvers.nonlinear.explicit import (
        advance_explicit_nonlinear_state,
        checkpointed_explicit_scan,
    )
    from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached

    jax.config.update("jax_enable_x64", True)
    print(f"devices: {jax.devices()}", flush=True)
    context = _context(args)
    x0 = jnp.asarray([0.0], dtype=jnp.float64)
    base_geometry = context["geometry_for"](x0)
    state, dt, tau_ac, saturation, runtime = _saturated_state(
        args, context, base_geometry
    )
    grid = build_spectral_grid(runtime.grid)
    base_params = build_runtime_linear_params(runtime, Nm=args.nm, geom=base_geometry)
    term_cfg = build_runtime_term_config(runtime)
    project_state = _make_hermitian_projector(
        np.asarray(grid.ky), int(np.asarray(grid.kx).size)
    )
    expected_shape = (1, args.nl, args.nm, args.ny, args.nx, args.nz)
    if state.shape != expected_shape:
        raise SystemExit(f"state shape {state.shape} != expected {expected_shape}")
    base_cache = build_linear_cache(
        grid, base_geometry, base_params, Nl=args.nl, Nm=args.nm
    )
    probe, _fields = nonlinear_rhs_cached(state, base_cache, base_params, term_cfg)
    state = jnp.asarray(state, dtype=probe.dtype)

    def window_mean(control):
        geometry = context["geometry_for"](control)
        cache = build_linear_cache(grid, geometry, base_params, Nl=args.nl, Nm=args.nm)
        _volume_factor, flux_factor = fieldline_quadrature_weights(geometry, grid)

        def advance(current, _unused):
            current = project_state(current)

            def rhs(local_state):
                # Geometry/cache entries are active differentiation variables.
                # The optimized field-solve custom VJP is for state/parameter
                # adjoints; use the primitive field solve for cache cotangents.
                return nonlinear_rhs_cached(
                    local_state,
                    cache,
                    base_params,
                    term_cfg,
                    differentiable=True,
                )

            derivative, _fields_now = rhs(current)
            next_state = advance_explicit_nonlinear_state(
                current,
                derivative,
                jnp.asarray(dt, dtype=jnp.real(current).dtype),
                method="rk2",
                rhs_fn=rhs,
                project_state=project_state,
                state_dtype=current.dtype,
            )
            _next_derivative, fields = rhs(next_state)
            apar = jnp.zeros_like(fields.phi)
            bpar = jnp.zeros_like(fields.phi)
            heat = heat_flux_total(
                next_state,
                fields.phi,
                apar,
                bpar,
                cache,
                grid,
                base_params,
                flux_factor,
            )
            return next_state, heat

        _final, heat = checkpointed_explicit_scan(
            advance,
            jax.lax.stop_gradient(state),
            jnp.arange(int(args.window)),
            checkpoint=True,
        )
        return jnp.mean(heat)

    value_only = jax.jit(window_mean)
    started = time.time()
    reverse_gradient = None
    forward_gradient = None
    base_value = value_only(x0)
    if args.ad_mode in {"reverse", "both"}:
        _reverse_value, reverse_gradient = jax.jit(jax.value_and_grad(window_mean))(x0)
        reverse_gradient.block_until_ready()
    if args.ad_mode in {"forward", "both"}:
        _forward_value, forward_gradient = jax.jit(
            lambda control: jax.jvp(
                window_mean,
                (control,),
                (jnp.ones_like(control),),
            )
        )(x0)
        forward_gradient.block_until_ready()
    gradient_seconds = time.time() - started
    fd_step = float(args.fd_step)
    plus = value_only(x0 + fd_step)
    minus = value_only(x0 - fd_step)
    centered_fd = (plus - minus) / (2.0 * fd_step)
    reverse_finite = bool(
        reverse_gradient is not None and jnp.isfinite(reverse_gradient[0])
    )
    forward_finite = bool(
        forward_gradient is not None and jnp.isfinite(forward_gradient)
    )
    if reverse_finite:
        gradient_value = float(reverse_gradient[0])
        ad_method_used = "reverse"
    elif forward_finite:
        gradient_value = float(forward_gradient)
        ad_method_used = "forward"
    else:
        gradient_value = None
        ad_method_used = None
    fd_finite = bool(jnp.isfinite(centered_fd))
    fd_relative_error = (
        abs(gradient_value - float(centered_fd)) / max(abs(float(centered_fd)), 1.0e-30)
        if gradient_value is not None and fd_finite
        else None
    )

    direction = -float(np.sign(gradient_value)) if gradient_value is not None else None
    line_rows = []
    if direction is not None:
        for step_size in args.line_steps:
            control = x0 + direction * float(step_size)
            candidate = value_only(control)
            geometry = context["geometry_for"](control)
            geometry_response = geometry_response_metrics(base_geometry, geometry)
            line_rows.append(
                {
                    "step": float(step_size),
                    "control": float(control[0]),
                    "heat_flux": float(candidate),
                    "relative_change": float((candidate - base_value) / base_value),
                    "geometry_response": geometry_response,
                    "admissible_geometry": bool(
                        geometry_response["max_relative_change"]
                        <= float(args.max_geometry_change)
                    ),
                }
            )
    admissible_rows = [row for row in line_rows if row["admissible_geometry"]]
    best = (
        min(admissible_rows, key=lambda row: row["heat_flux"])
        if admissible_rows
        else None
    )
    descent_passed = bool(best is not None and best["heat_flux"] < float(base_value))
    gates = {
        "finite_ad_gradient": gradient_value is not None,
        "ad_matches_centered_fd": bool(
            fd_relative_error is not None
            and fd_relative_error <= float(args.max_ad_fd_relative_error)
        ),
        "admissible_local_descent": descent_passed,
    }

    payload = {
        "kind": "vmec_boozer_physical_nonlinear_window_descent",
        "claim_level": "finite_window_vmec_state_control_not_equilibrated_boundary_or_stationary_gradient",
        "case": args.case_name,
        "parameter_names": context["parameter_names"],
        "parameter_indices": context["parameter_indices"],
        "base_control": 0.0,
        "grid": {"Nx": args.nx, "Ny": args.ny, "Nz": args.nz},
        "moments": {"Nl": args.nl, "Nm": args.nm},
        "dt": dt,
        "method": str(runtime.time.method),
        "tau_ac": tau_ac,
        "window": int(args.window),
        "window_time_in_tau_ac": int(args.window) * dt / tau_ac,
        "base_heat_flux": float(base_value),
        "ad_mode_requested": args.ad_mode,
        "ad_method_used": ad_method_used,
        "ad_gradient": gradient_value,
        "reverse_gradient": (float(reverse_gradient[0]) if reverse_finite else None),
        "forward_gradient": float(forward_gradient) if forward_finite else None,
        "centered_fd_gradient": float(centered_fd) if fd_finite else None,
        "fd_step": fd_step,
        "ad_fd_relative_error": fd_relative_error,
        "max_ad_fd_relative_error": float(args.max_ad_fd_relative_error),
        "gradient_seconds": gradient_seconds,
        "negative_gradient_direction": direction,
        "max_geometry_change": float(args.max_geometry_change),
        "line_search": line_rows,
        "best_candidate": best,
        "descent_passed": descent_passed,
        "gates": gates,
        "passed": all(gates.values()),
        "saturation": saturation,
        "limitations": [
            "finite post-saturation window",
            "single VMEC internal-state coefficient",
            "perturbed controls are not re-equilibrated boundary shapes",
            "candidate windows start from the same detached base state",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        f"Q={float(base_value):.6e} AD={payload['ad_gradient']} "
        f"FD={payload['centered_fd_gradient']} rel={fd_relative_error} "
        f"descent={'PASS' if descent_passed else 'FAIL'}",
        flush=True,
    )
    print(f"written: {args.output}", flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
