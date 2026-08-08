"""Implicit eigenpair sensitivities for a reduced quasilinear objective.

This demo keeps the production linear solve matrix-free in normal use, but
builds a tiny dense RHS fixture so the isolated-branch implicit sensitivity
formula can be checked against central finite differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from gkx.objectives.autodiff_validation import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from gkx.config import CycloneBaseCase, GridConfig
from gkx.diagnostics import fieldline_quadrature_weights, heat_flux_species
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.artifacts.plotting import set_plot_style
from gkx.diagnostics.quasilinear_transport import effective_kperp2, phi_norm2, saturated_flux_from_linear_weight


OBSERVABLE_LABELS = (r"$\gamma$", r"$\omega$", r"$k_{\perp,\mathrm{eff}}^2$", r"$\hat Q_i$", r"$Q_i^{ML}$")
# The differentiated parameters are LinearParams.fprim/tprim, which are
# a/L_n and a/L_T -- not R/L. Label them as what is on the axis.
PARAMETER_LABELS = (r"$a/L_n$", r"$a/L_{Ti}$")


def _build_tiny_fixture():
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    n_laguerre = 2
    n_hermite = 3
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        fprim=2.2,
        tprim=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, n_laguerre, n_hermite)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    return grid, cache, state_shape, vol_fac, flux_fac, terms


def _params_from_features(x: jnp.ndarray) -> LinearParams:
    return LinearParams(
        fprim=x[0],
        tprim=x[1],
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )


def build_report(
    *,
    r_over_ln: float = 2.2,
    r_over_lti: float = 6.9,
    step: float = 1.0e-3,
    rtol: float = 5.0e-2,
    atol: float = 2.0e-3,
) -> dict[str, object]:
    """Return a JSON-friendly implicit-sensitivity report."""

    grid, cache, state_shape, vol_fac, flux_fac, terms = _build_tiny_fixture()

    def rhs_with_params(state, params):
        return linear_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x):
        params = _params_from_features(x)
        return explicit_complex_operator_matrix(lambda state: rhs_with_params(state, params)[0], state_shape)

    def objective_fn(eigenvalue, eigenvector, x):
        params = _params_from_features(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_with_params(state, params)
        zeros = jnp.zeros_like(phi)
        kperp_eff2 = effective_kperp2(phi, cache, vol_fac)
        norm = phi_norm2(phi, cache, params, vol_fac, normalization="phi_rms")
        heat_weight = jnp.sum(
            heat_flux_species(
                state,
                phi,
                zeros,
                zeros,
                cache,
                grid,
                params,
                flux_fac,
            )
        ) / norm
        gamma = jnp.real(eigenvalue)
        omega = -jnp.imag(eigenvalue)
        saturated_heat = saturated_flux_from_linear_weight(heat_weight, gamma, kperp_eff2)
        return jnp.asarray([gamma, omega, kperp_eff2, heat_weight, saturated_heat])

    params = jnp.asarray([r_over_ln, r_over_lti])
    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        params,
        step=step,
        rtol=rtol,
        atol=atol,
        gap_floor=1.0e-6,
    )

    matrix = matrix_fn(params)
    eigvals, eigvecs = jnp.linalg.eig(matrix)
    index = int(report["selected_index"])
    observables = np.asarray(objective_fn(eigvals[index], eigvecs[:, index], params), dtype=float)
    report.update(
        {
            "kind": "quasilinear_implicit_sensitivity_demo",
            "case": "tiny_cyclone_linear_rhs",
            "parameters": {"fprim": float(r_over_ln), "tprim": float(r_over_lti)},
            "observable_labels": list(OBSERVABLE_LABELS),
            "parameter_labels": list(PARAMETER_LABELS),
            "observables": observables.tolist(),
        }
    )
    return report


def write_figure(report: dict[str, object], out: Path) -> None:
    """Write the publication-facing sensitivity figure."""

    out.parent.mkdir(parents=True, exist_ok=True)
    jac_impl = np.asarray(report["jacobian_implicit"], dtype=float)
    jac_fd = np.asarray(report["jacobian_fd"], dtype=float)
    observables = np.asarray(report["observables"], dtype=float)
    abs_err = np.abs(jac_impl - jac_fd)
    rel_err = abs_err / np.maximum(np.abs(jac_fd), float(report["atol"]))

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))

    ax0 = axes[0, 0]
    colors = ["#2563eb", "#0891b2", "#16a34a", "#f97316", "#7c3aed"]
    ax0.bar(np.arange(len(observables)), observables, color=colors)
    ax0.axhline(0.0, color="0.25", linewidth=0.8)
    ax0.set_xticks(np.arange(len(observables)), OBSERVABLE_LABELS, rotation=20, ha="right")
    ax0.set_ylabel("value")
    ax0.set_title("Reduced quasilinear observables")

    ax1 = axes[0, 1]
    vmax = max(float(np.max(np.abs(jac_impl))), 1.0e-12)
    im = ax1.imshow(jac_impl, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xticks(np.arange(len(PARAMETER_LABELS)), PARAMETER_LABELS)
    ax1.set_yticks(np.arange(len(OBSERVABLE_LABELS)), OBSERVABLE_LABELS)
    ax1.set_title("Implicit sensitivity matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1, 0]
    ax2.scatter(jac_fd.ravel(), jac_impl.ravel(), color="#111827", s=40)
    lo = float(min(np.min(jac_fd), np.min(jac_impl)))
    hi = float(max(np.max(jac_fd), np.max(jac_impl)))
    pad = 0.05 * max(hi - lo, 1.0e-12)
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#dc2626", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("central finite difference")
    ax2.set_ylabel("implicit left/right derivative")
    ax2.set_title("Derivative parity")

    ax3 = axes[1, 1]
    x = np.arange(jac_impl.shape[0])
    width = 0.35
    ax3.bar(x - width / 2, rel_err[:, 0], width, label=PARAMETER_LABELS[0], color="#2563eb")
    ax3.bar(x + width / 2, rel_err[:, 1], width, label=PARAMETER_LABELS[1], color="#f97316")
    ax3.set_xticks(x, OBSERVABLE_LABELS, rotation=20, ha="right")
    ax3.set_yscale("log")
    ax3.set_ylabel("relative derivative error")
    ax3.set_title("Finite-difference check")
    ax3.legend(loc="best", fontsize=9)
    ax3.text(
        0.03,
        0.95,
        f"passed = {report['passed']}\n"
        f"branch gap = {float(report['eigenvalue_gap']):.2e}\n"
        f"max rel. err = {float(report['max_rel_error']):.2e}",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )

    fig.suptitle("Implicit quasilinear eigenpair sensitivity gate")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def run_demo(*, outdir: Path, plot: bool = True, write_files: bool = True) -> dict[str, object]:
    """Run the demo and optionally write JSON/figure artifacts."""

    outdir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    if write_files:
        (outdir / "quasilinear_implicit_sensitivity.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
    if plot:
        fig_path = outdir / "quasilinear_implicit_sensitivity.png"
        write_figure(report, fig_path)
        print(f"Wrote {fig_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("docs/_static"))
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG/PDF figure generation")
    args = parser.parse_args()
    report = run_demo(outdir=args.outdir, plot=not args.no_plot, write_files=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
