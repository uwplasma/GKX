"""Autodiff inverse/sensitivity demo for a linear ITG growth proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from gkx.config import CycloneBaseCase, GridConfig
from gkx.objectives.autodiff_validation import covariance_diagnostics
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.solvers.linear.integrators import integrate_linear
from gkx.artifacts.plotting import set_plot_style
from gkx.operators.linear.params import Species, build_linear_params


def _estimate_growth(phi_t: jnp.ndarray, t: jnp.ndarray, start_idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi_win = phi_t[start_idx:]
    t_win = t[start_idx:]
    amp = jnp.abs(phi_win) + 1.0e-12
    log_amp = jnp.log(amp)
    phase = jnp.unwrap(jnp.angle(phi_win))
    t_centered = t_win - jnp.mean(t_win)
    log_centered = log_amp - jnp.mean(log_amp)
    phase_centered = phase - jnp.mean(phase)
    denom = jnp.sum(t_centered**2)
    gamma = jnp.sum(t_centered * log_centered) / denom
    omega = -jnp.sum(t_centered * phase_centered) / denom
    return gamma, omega


def _gauss_newton_solve(
    obs_fn,
    target: np.ndarray,
    params_init: np.ndarray,
    *,
    steps: int,
    damping: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jac_fn = jax.jit(jax.jacobian(obs_fn))
    params = np.asarray(params_init, dtype=float)
    path = [params.copy()]
    loss_hist: list[float] = []
    obs_hist: list[np.ndarray] = []

    for _ in range(steps):
        obs = np.asarray(obs_fn(jnp.asarray(params)))
        residual = obs - target
        loss = float(residual @ residual)
        loss_hist.append(loss)
        obs_hist.append(obs.copy())
        if not np.isfinite(loss) or loss < 1.0e-14:
            break

        jac = np.asarray(jac_fn(jnp.asarray(params)))
        lhs = jac.T @ jac + damping * np.eye(jac.shape[1])
        rhs = jac.T @ residual
        step = np.linalg.solve(lhs, rhs)
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step *= max_step / max(step_norm, 1.0e-12)

        alpha = 1.0
        accepted = False
        for _ in range(10):
            candidate = params - alpha * step
            cand_obs = np.asarray(obs_fn(jnp.asarray(candidate)))
            cand_residual = cand_obs - target
            cand_loss = float(cand_residual @ cand_residual)
            if np.isfinite(cand_loss) and cand_loss <= loss:
                params = candidate
                path.append(params.copy())
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    obs_final = np.asarray(obs_fn(jnp.asarray(params)))
    residual_final = obs_final - target
    path_arr = np.asarray(path, dtype=float)
    return params, obs_final, residual_final, path_arr


def _growth_from_params(
    params_vec: jnp.ndarray,
    *,
    grid,
    geom,
    cache,
    G0,
    dt: float,
    steps: int,
    ky_index: int,
    kx_index: int,
    z_index: int,
    start_idx: int,
) -> jnp.ndarray:
    tprim = params_vec[0]
    fprim = params_vec[1]
    params = LinearParams(
        charge_sign=jnp.asarray([1.0]),
        density=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        temp=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        rho=jnp.asarray([1.0]),
        tz=jnp.asarray([1.0]),
        tprim=jnp.asarray([tprim]),
        fprim=jnp.asarray([fprim]),
        tau_e=1.0,
    )
    _, phi_t = integrate_linear(G0, grid, geom, params, dt=dt, steps=steps, cache=cache)
    t = jnp.arange(steps) * dt
    phi_mode = phi_t[:, ky_index, kx_index, z_index]
    gamma, omega = _estimate_growth(phi_mode, t, start_idx)
    return jnp.asarray([gamma, omega])


def run_demo(
    *,
    outdir: Path,
    steps: int,
    dt: float,
    ky_index: int,
    kx_index: int,
    z_index: int,
    tprim_true: float,
    fprim_true: float,
    tprim_init: float,
    fprim_init: float,
    gd_steps: int,
    gd_lr: float,
    plot: bool = True,
    write_files: bool = True,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    grid_cfg = GridConfig(Nx=1, Ny=8, Nz=32, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    Nl, Nm = 2, 2
    G0 = jnp.zeros((Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, ky_index, kx_index, :].set(1.0e-3 + 0.0j)

    cache = build_linear_cache(
        grid,
        geom,
        build_linear_params([Species(1.0, 1.0, 1.0, 1.0, 2.0, 0.8)], tau_e=1.0),
        Nl,
        Nm,
    )
    start_idx = max(2, steps // 2)

    growth_fn = jax.jit(
        lambda params_vec: _growth_from_params(
            params_vec,
            grid=grid,
            geom=geom,
            cache=cache,
            G0=G0,
            dt=dt,
            steps=steps,
            ky_index=ky_index,
            kx_index=kx_index,
            z_index=z_index,
            start_idx=start_idx,
        )
    )

    target = np.asarray(growth_fn(jnp.asarray([tprim_true, fprim_true])))

    def loss_fn(params_vec):
        gamma = growth_fn(params_vec)
        return jnp.sum((gamma - target) ** 2)

    params_init = np.asarray([tprim_init, fprim_init], dtype=float)
    params_final, obs_final, residual, path = _gauss_newton_solve(
        growth_fn,
        target,
        params_init,
        steps=gd_steps,
        damping=max(1.0e-6, 1.0e-2 * gd_lr),
        max_step=max(0.25, gd_lr),
    )
    tprim_hist = path[:, 0]
    fprim_hist = path[:, 1]
    loss_hist = np.sum((np.asarray([np.asarray(growth_fn(jnp.asarray(p))) for p in path]) - target[None, :]) ** 2, axis=1)

    sweep_tprim = np.linspace(1.2, 3.8, 16)
    sweep_tprim_vals = np.asarray(
        jax.vmap(lambda val: growth_fn(jnp.asarray([val, fprim_true])))(jnp.asarray(sweep_tprim))
    )
    sweep_fprim = np.linspace(0.4, 1.6, 16)
    sweep_fprim_vals = np.asarray(
        jax.vmap(lambda val: growth_fn(jnp.asarray([tprim_true, val])))(jnp.asarray(sweep_fprim))
    )

    tprim_check = 2.2
    fprim_check = 0.9
    eps = 1.0e-3
    params_center = jnp.asarray([tprim_check, fprim_check])
    jac_ad = np.asarray(jax.jacobian(growth_fn)(params_center))
    jac_fd = np.zeros_like(jac_ad)
    for idx in range(2):
        shift = np.zeros(2, dtype=float)
        shift[idx] = eps
        gamma_plus = np.asarray(growth_fn(jnp.asarray(params_center + shift)))
        gamma_minus = np.asarray(growth_fn(jnp.asarray(params_center - shift)))
        jac_fd[:, idx] = (gamma_plus - gamma_minus) / (2.0 * eps)
    rel_err_cols = np.linalg.norm(jac_ad - jac_fd, axis=0) / (np.linalg.norm(jac_fd, axis=0) + 1.0e-12)

    uq = covariance_diagnostics(jac_ad, residual, regularization=1.0e-9)
    cov = np.asarray(uq["covariance"], dtype=float)

    summary = {
        "target_observables": target.tolist(),
        "tprim_init": float(tprim_init),
        "fprim_init": float(fprim_init),
        "tprim_final": float(params_final[0]),
        "fprim_final": float(params_final[1]),
        "observable_final": obs_final.tolist(),
        "observable_abs_error": np.abs(residual).tolist(),
        "parameter_abs_error": [float(abs(params_final[0] - tprim_true)), float(abs(params_final[1] - fprim_true))],
        "loss_final": float(loss_hist[-1]) if loss_hist.size else float(loss_fn(jnp.asarray(params_init))),
        "jac_autodiff": jac_ad.tolist(),
        "jac_finite_diff": jac_fd.tolist(),
        "jac_rel_error": rel_err_cols.tolist(),
        **uq,
    }

    if write_files:
        data_path = outdir / "autodiff_inverse_growth_tprim_sweep.csv"
        np.savetxt(
            data_path,
            np.column_stack([sweep_tprim, sweep_tprim_vals]),
            delimiter=",",
            header="tprim,gamma,omega",
            comments="",
        )
        data_path_f = outdir / "autodiff_inverse_growth_fprim_sweep.csv"
        np.savetxt(
            data_path_f,
            np.column_stack([sweep_fprim, sweep_fprim_vals]),
            delimiter=",",
            header="fprim,gamma,omega",
            comments="",
        )
        summary_path = outdir / "autodiff_inverse_growth_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    if plot:
        set_plot_style()
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

        ax0 = axes[0, 0]
        ax0.plot(sweep_tprim, sweep_tprim_vals[:, 0], marker="o", color="#1f77b4", label=r"$\gamma$")
        ax0.plot(sweep_tprim, sweep_tprim_vals[:, 1], marker="s", color="#ff7f0e", label=r"$\omega$")
        ax0.axhline(target[0], color="#1f77b4", linestyle="--", alpha=0.6, label="target γ")
        ax0.axhline(target[1], color="#ff7f0e", linestyle="--", alpha=0.6, label="target ω")
        ax0.set_xlabel(r"$a/L_{Ti}$")
        ax0.set_ylabel("observable")
        ax0.set_title("Sensitivity vs $a/L_{Ti}$")
        ax0.legend(loc="best", ncol=2, fontsize=9)

        ax1 = axes[0, 1]
        ax1.plot(sweep_fprim, sweep_fprim_vals[:, 0], marker="o", color="#1f77b4", label=r"$\gamma$")
        ax1.plot(sweep_fprim, sweep_fprim_vals[:, 1], marker="s", color="#ff7f0e", label=r"$\omega$")
        ax1.axhline(target[0], color="#1f77b4", linestyle="--", alpha=0.6, label="target γ")
        ax1.axhline(target[1], color="#ff7f0e", linestyle="--", alpha=0.6, label="target ω")
        ax1.set_xlabel(r"$a/L_{n}$")
        ax1.set_ylabel("observable")
        ax1.set_title("Sensitivity vs $a/L_{n}$")
        ax1.legend(loc="best", ncol=2, fontsize=9)

        ax2 = axes[1, 0]
        tprim_grid = np.linspace(1.2, 3.8, 80)
        fprim_grid = np.linspace(0.4, 1.6, 80)
        dt_grid, df_grid = np.meshgrid(tprim_grid - tprim_true, fprim_grid - fprim_true)
        quad_grid = (
            (jac_ad[0, 0] * dt_grid + jac_ad[0, 1] * df_grid) ** 2
            + (jac_ad[1, 0] * dt_grid + jac_ad[1, 1] * df_grid) ** 2
        )
        levels = np.geomspace(max(float(np.nanmin(quad_grid[quad_grid > 0.0])), 1.0e-10), max(float(np.nanmax(quad_grid)), 1.0e-4), 8)
        ax2.contour(tprim_grid, fprim_grid, quad_grid, levels=levels, colors="#cbd5e1", linewidths=1.0)
        ax2.plot(tprim_hist, fprim_hist, marker="o", color="#2ca02c", label="Gauss-Newton path")
        ax2.scatter([tprim_init], [fprim_init], color="#111827", marker="s", s=36, label="initial")
        ax2.scatter([tprim_true], [fprim_true], color="#d62728", marker="x", s=80, label="target")
        ax2.scatter([params_final[0]], [params_final[1]], color="#7c3aed", marker="o", s=42, label="recovered")
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2.0 * np.sqrt(np.maximum(vals, 0.0))
        ellipse = Ellipse((params_final[0], params_final[1]), width, height, angle=angle, fill=False, color="#9467bd")
        ax2.add_patch(ellipse)
        ax2.set_xlabel(r"$a/L_{Ti}$")
        ax2.set_ylabel(r"$a/L_{n}$")
        ax2.set_title("Inverse solve + loss contours")
        ax2.text(
            0.03,
            0.03,
            "One mode fits the observables,\nnot both gradients uniquely.",
            transform=ax2.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
        )
        ax2.legend(loc="best", fontsize=8)

        ax3 = axes[1, 1]
        ax3.bar(["$a/L_{Ti}$", "$a/L_{n}$"], rel_err_cols, color=["#9467bd", "#8c564b"])
        ax3.set_ylabel("Jacobian rel. error")
        ax3.set_title("Autodiff vs finite diff")
        ax3.text(
            0.03,
            0.95,
            f"|Δp| = ({abs(params_final[0]-tprim_true):.2e}, {abs(params_final[1]-fprim_true):.2e})\n"
            f"|Δobs| = ({abs(residual[0]):.2e}, {abs(residual[1]):.2e})",
            transform=ax3.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
        )

        fig.suptitle("Autodiff sensitivity and local inverse demo (single mode, non-unique)")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        fig_path = outdir / "autodiff_inverse_growth.png"
        fig.savefig(fig_path, dpi=200)
        fig.savefig(outdir / "autodiff_inverse_growth.pdf")
        print(f"Wrote {fig_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Autodiff inverse growth demo.")
    parser.add_argument("--outdir", type=Path, default=Path("docs/_static"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--ky-index", type=int, default=1)
    parser.add_argument("--kx-index", type=int, default=0)
    parser.add_argument("--z-index", type=int, default=0)
    parser.add_argument("--tprim-true", type=float, default=2.8)
    parser.add_argument("--fprim-true", type=float, default=0.8)
    parser.add_argument("--tprim-init", type=float, default=2.4)
    parser.add_argument("--fprim-init", type=float, default=0.95)
    parser.add_argument("--gd-steps", type=int, default=18)
    parser.add_argument("--gd-lr", type=float, default=0.7)
    args = parser.parse_args()

    summary = run_demo(
        outdir=args.outdir,
        steps=args.steps,
        dt=args.dt,
        ky_index=args.ky_index,
        kx_index=args.kx_index,
        z_index=args.z_index,
        tprim_true=args.tprim_true,
        fprim_true=args.fprim_true,
        tprim_init=args.tprim_init,
        fprim_init=args.fprim_init,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        plot=True,
        write_files=True,
    )
    print("summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
