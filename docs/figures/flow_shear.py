"""Render the README flow-shear figures from ``docs/figures/flow_shear.json``.

    python docs/figures/flow_shear.py [--data JSON] [--out DIR]
    python docs/figures/flow_shear.py --run [--data JSON] [--nx 64]   # GPU, about 30 min

``flow_shear.png``: Cyclone adiabatic-electron heat flux against ``exb_shear``,
GKX and GX, each point a restart from saturated unsheared turbulence.
``zonal_flow.png``: the same case with an imposed sinusoidal zonal flow of four
periods across the box. ``--run`` regenerates the zonal-flow data (the shear data
come from the GX comparison and are stored in the JSON); it needs a GPU, the
Cyclone deck of ``examples/03_nonlinear_tokamak`` and the solver internals below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RC = {
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.5,
    "savefig.dpi": 200,
}


def shear_figure(d: dict):
    import matplotlib.pyplot as plt

    from gkx.artifacts.figure_style import GKX_COLORS

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for nx, color, shift in (
        ("192", GKX_COLORS["blue"], -0.002),
        ("64", GKX_COLORS["grey"], 0.002),
    ):
        for code, marker, ls, off in (("gkx", "o", "-", 0.0), ("gx", "s", "--", 0.004)):
            rows = d["shear"][nx][code]
            g = np.array([r[0] for r in rows]) + shift + off
            q = np.array([r[1] for r in rows])
            err = np.array([r[2] for r in rows])
            ax.errorbar(
                g,
                q,
                err,
                marker=marker,
                ls=ls,
                color=color,
                capsize=3,
                ms=5,
                mfc=color if code == "gkx" else "white",
                mew=1.2,
                label=f"{'GKX' if code == 'gkx' else 'GX'}, nx = {nx}",
            )
    ax.set(
        xlabel=r"$\gamma_E\,a/v_t$",
        ylabel=r"$Q_i$ (gyro-Bohm)",
        title="Cyclone, adiabatic electrons; restart from saturated state, one seed",
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def zonal_figure(d: dict):
    import matplotlib.pyplot as plt

    from gkx.artifacts.figure_style import GKX_COLORS

    fig = plt.figure(figsize=(8.4, 5.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.2, 1.0])
    ax_q = fig.add_subplot(grid[0, 0])
    peak = np.array(d["zonal"]["peak_shear"])
    for seed, rows in d["zonal"]["Q"].items():
        q = np.array([r[0] for r in rows])
        err = np.array([r[1] for r in rows])
        ax_q.errorbar(
            peak,
            q,
            err,
            marker="o",
            capsize=2,
            label=f"seed {seed}",
            color=GKX_COLORS["blue"] if seed == "22" else GKX_COLORS["vermillion"],
        )
    ax_q.set(
        xlabel=r"peak imposed shear $\max|\partial_x^2\phi_{ext}|$",
        ylabel=r"$Q_i$ (gyro-Bohm)",
        title="Heat flux against flow amplitude",
    )
    ax_q.grid(alpha=0.3)
    ax_q.legend(frameon=False)
    ax_x = fig.add_subplot(grid[1, 0])
    shear = np.abs(np.array(d["zonal"]["shear_abs"]))
    shear = shear / shear.max()
    edges = np.linspace(0.0, 1.0, 5)
    which = np.digitize(shear, edges[1:-1])
    centres = 0.5 * (edges[1:] + edges[:-1])
    colors = (
        GKX_COLORS["grey"],
        GKX_COLORS["sky"],
        GKX_COLORS["blue"],
        GKX_COLORS["vermillion"],
    )
    for key, color in zip(sorted(d["zonal"]["Qx"]), colors):
        qx = np.array(d["zonal"]["Qx"][key])
        qx = qx / qx.mean()
        ax_x.plot(
            centres,
            [qx[which == b].mean() for b in range(4)],
            marker="o",
            color=color,
            label=f"peak shear {peak[int(key)]:.2g}",
        )
    ax_x.set(
        xlabel=r"local $|\partial_x^2\phi_{ext}|$ / peak",
        ylabel=r"$Q(x)/\langle Q\rangle$, binned",
        title="Flux against local shear",
    )
    ax_x.grid(alpha=0.3)
    ax_x.legend(frameon=False, fontsize=6.5)
    for row, (key, title) in enumerate(
        (("phi_free", "no flow"), ("phi_flow", "imposed flow"))
    ):
        ax = fig.add_subplot(grid[row, 1:])
        phi = np.array(d["zonal"][key])
        lim = float(np.max(np.abs(np.array(d["zonal"]["phi_free"]))))
        ax.imshow(
            phi,
            origin="lower",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            aspect="auto",
            extent=(0.0, d["zonal"]["Lx"], 0.0, d["zonal"]["Ly"]),
        )
        ax.set(
            xlabel=r"$x/\rho_{ref}$",
            ylabel=r"$y/\rho_{ref}$",
            title=rf"$\phi(x,y)$, {title}",
        )
    fig.tight_layout()
    return fig


def _q_of_x(state, phi, cache, grid, params, flux_fac, flux_scale):
    """Electrostatic heat-flux density ``Q(x)`` whose mean over x is the usual flux."""

    import jax.numpy as jnp

    from gkx.operators.moments import _jl_family, _transport_mode_weight

    fac = _transport_mode_weight(grid, use_dealias=True)
    active = fac != 0.0
    ky = grid.ky[:, None, None]
    vphi = jnp.where(active[:, :, None], 1.0j * ky * phi, 0.0)
    jl, _, jfac = _jl_family(cache)
    state = state[0] if state.ndim == 6 else state
    g = jnp.where(active[None, None, :, :, None], state, 0.0)
    p_bar = jnp.sum(jfac[0] * g[:, 0] + jl[0] * g[:, 2] / jnp.sqrt(2.0), axis=0)
    nt = float(
        jnp.asarray(params.density).ravel()[0] * jnp.asarray(params.temp).ravel()[0]
    )
    weight = 2.0 * fac.max(axis=1)[:, None] * flux_fac[None, :] * nt * flux_scale
    nkx = phi.shape[1]
    a = jnp.fft.ifft(vphi, axis=1) * nkx
    b = jnp.fft.ifft(p_bar, axis=1) * nkx
    return np.asarray(jnp.sum((jnp.conj(a) * b).real * weight[:, None, :], axis=(0, 2)))


def run(path: Path, nx: int, seeds=(22, 23), peaks=(0.05, 0.1, 0.2), periods=4) -> None:
    import dataclasses as dc
    from functools import partial

    import gkx
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.moments import fieldline_quadrature_weights
    from gkx.operators.nonlinear.rhs import zonal_flow_modes
    from gkx.solvers.nonlinear.transport_window import (
        integrate_nonlinear_sheared_transport,
    )
    from gkx.terms.assembly import compute_fields_cached
    from gkx.workflows import nonlinear as nl

    case = gkx.load(ROOT / "examples/03_nonlinear_tokamak/case_full.toml")
    case = dc.replace(case, grid=dc.replace(case.grid, Nx=nx))
    out: dict = {"zonal": {"Q": {}, "traces": {}}}
    Qx_acc: dict[int, list] = {}
    for seed in seeds:
        c = dc.replace(case, init=dc.replace(case.init, random_seed=seed))
        ctx, cfg = nl._prepare_context(
            c,
            ky_target=0.3,
            kx_target=None,
            Nl=4,
            Nm=8,
            dt=None,
            steps=None,
            status=lambda s: None,
        )
        grid, kx = ctx.grid, np.asarray(ctx.grid.kx)
        length = 2.0 * np.pi / float(np.min(np.abs(kx[kx != 0.0])))
        k = 2.0 * np.pi * periods / length
        cache = build_linear_cache(
            grid, ctx.geom, ctx.params, ctx.G0.shape[-5], ctx.G0.shape[-4]
        )
        _, flux_fac = fieldline_quadrature_weights(ctx.geom, grid)
        scale = float(cfg.normalization.flux_scale)
        jl0 = (
            float(cache.Jl[0, 0, 0, periods, 0])
            if cache.Jl.ndim == 5
            else float(cache.Jl[0, 0, periods, 0])
        )
        amps = [p / (k * k * jl0) for p in peaks]

        def stage(state, params, tlen, record_x):
            run_ = partial(
                integrate_nonlinear_sheared_transport,
                grid=grid,
                geom=ctx.geom,
                params=params,
                shear_rate=0.0,
                method="rk3",
                terms=ctx.terms,
                compressed_real_fft=True,
                flux_scale=scale,
                differentiable=False,
                fixed_dt=False,
                dt_min=float(cfg.time.dt_min),
                dt_max=cfg.time.dt_max,
                cfl=float(cfg.time.cfl),
                cfl_fac=nl.resolve_nonlinear_cfl_fac(
                    "rk3", cfg.time.cfl_fac, int(state.shape[-5] * state.shape[-4])
                ),
            )
            t, dt, ts, qs, qx = 0.0, 0.05, [], [], []
            while t < tlen:
                tr = run_(state, dt=dt, steps=256, initial_time=t, initial_dt=dt)
                state = tr.final_state
                ts.append(np.asarray(tr.time))
                qs.append(np.asarray(tr.heat_flux).reshape(len(ts[-1]), -1).sum(-1))
                t, dt = float(ts[-1][-1]), float(np.diff(ts[-1][-2:])[0])
                if record_x:
                    fields = compute_fields_cached(
                        state, cache, params, terms=ctx.terms
                    )
                    qx.append(
                        _q_of_x(state, fields.phi, cache, grid, params, flux_fac, scale)
                    )
            return state, np.concatenate(ts), np.concatenate(qs), qx

        sat, t1, q1, _ = stage(ctx.G0, ctx.params, 300.0, False)
        print(
            f"seed {seed}: saturated, Q(last third) = {q1[t1 > 200].mean():.3f}",
            flush=True,
        )
        rows, snaps = [], {}
        for index, amp in enumerate([0.0, *amps]):
            params = dc.replace(
                ctx.params,
                zonal_flow=zonal_flow_modes([0, 0, 0, amp], grid) if amp else None,
            )
            state, t, q, qx = stage(sat, params, 250.0, True)
            late = t > t[-1] / 2
            batches = np.array_split(q[late], 5)
            mean = float(q[late].mean())
            sem = float(np.std([b.mean() for b in batches], ddof=1) / np.sqrt(5))
            rows.append([mean, sem])
            out["zonal"]["traces"][f"{seed}_{index}"] = [
                t[::20].tolist(),
                q[::20].tolist(),
            ]
            Qx_acc.setdefault(index, []).append(np.mean(qx[len(qx) // 2 :], axis=0))
            if seed == seeds[0] and index in (0, len(amps)):
                fields = compute_fields_cached(state, cache, params, terms=ctx.terms)
                phi = np.asarray(fields.phi)[:, :, phi_mid(fields)]
                xy = np.fft.irfft(
                    np.fft.ifft(phi, axis=1) * phi.shape[1],
                    n=2 * (phi.shape[0] - 1),
                    axis=0,
                )
                snaps["phi_free" if index == 0 else "phi_flow"] = xy.T.tolist()
            print(
                f"seed {seed} peak {([0.0, *peaks])[index]}: Q = {mean:.3f} +/- {sem:.3f}",
                flush=True,
            )
        out["zonal"]["Q"][str(seed)] = rows
        out["zonal"].update(snaps)
        out["zonal"]["Lx"] = length
        out["zonal"]["Ly"] = float(2.0 * np.pi * ctx.grid.y0)
        out["zonal"]["x"] = (np.arange(kx.size) * length / kx.size).tolist()
        x = np.arange(kx.size) * length / kx.size
        out["zonal"]["shear_abs"] = np.abs(
            amps[-1] * jl0 * k * k * np.cos(k * x)
        ).tolist()
    out["zonal"]["Qx"] = {
        str(i): np.mean(v, axis=0).tolist() for i, v in Qx_acc.items()
    }
    out["zonal"]["profile_index"] = [0, len(peaks)]
    out["zonal"]["peak_shear"] = [0.0, *peaks]
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(out)
    path.write_text(json.dumps(old))


def phi_mid(fields) -> int:
    return int(np.asarray(fields.phi).shape[-1] // 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "docs/figures/flow_shear.json"))
    ap.add_argument("--out", default=str(ROOT / "docs/_static"))
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--nx", type=int, default=64)
    args = ap.parse_args()
    if args.run:
        run(Path(args.data), args.nx)
        return
    from gkx.artifacts.figure_style import figure_style, save_figure

    d = json.loads(Path(args.data).read_text())
    with figure_style(**RC):
        save_figure(
            shear_figure(d), Path(args.out) / "flow_shear.png", palette_colors=96
        )
        save_figure(
            zonal_figure(d), Path(args.out) / "zonal_flow.png", palette_colors=96
        )


if __name__ == "__main__":
    main()
