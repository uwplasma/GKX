"""Jacobian of the nonlinear heat flux window with respect to input gradients.

For each saturated state and each consecutive window of ``window_steps``, the
mean heat flux ``Q`` of :func:`gkx.nonlinear_heat_flux_window` is
differentiated with respect to the per-species drives (``tprim`` = a/L_T,
``fprim`` = a/L_n) in three ways:

* reverse mode (``jax.grad``, one adjoint pass for the whole row),
* forward mode (``jax.jacfwd``, one tangent per parameter),
* central finite differences of the *same* finite-window objective, over a
  ladder of relative steps so that truncation and roundoff are both visible.

It records the AD/FD relative error at each step, wall time and compiled
temporary memory of each method, and the sensitivities with their spread over
windows and seeds. With ``[scan]`` traces available it also records the
long-time mean flux against a/L_T, the stiffness curve that the window
derivative is *not* guaranteed to reproduce: a window derivative is the
response of a finite trajectory from a fixed state, not the derivative of the
climate. That comparison is the point of the figure.

Workflow (see ``nonlinear_flux_jacobian.toml``)::

    python scripts/campaigns/nonlinear_flux_jacobian.py --write-scan-decks
    # saturate each seed and each scan deck with nonlinear_saturated_state.py
    python scripts/campaigns/nonlinear_flux_jacobian.py          # Jacobian + JSON
    python scripts/campaigns/nonlinear_flux_jacobian.py --plot   # figure only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_suffix(".toml")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def parameter_names(parameters, n_species: int) -> list[str]:
    return [f"{p}[{s}]" for p in parameters for s in range(n_species)]


def make_objective(params, parameters, evaluate):
    """``theta -> Q``; ``theta`` stacks each named per-species array."""

    import jax.numpy as jnp

    def objective(theta):
        chunks = jnp.split(theta, len(parameters))
        updated = dataclasses.replace(
            params,
            **{name: chunk for name, chunk in zip(parameters, chunks, strict=True)},
        )
        return evaluate(updated)

    theta0 = jnp.concatenate(
        [jnp.atleast_1d(jnp.asarray(getattr(params, p), float)) for p in parameters]
    )
    return objective, theta0


def _timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    import jax

    jax.block_until_ready(out)
    return out, time.perf_counter() - start


def _temp_bytes(fn, theta) -> int | None:
    import jax

    try:
        stats = jax.jit(fn).lower(theta).compile().memory_analysis()
        return int(stats.temp_size_in_bytes + stats.output_size_in_bytes)
    except Exception:  # noqa: BLE001 - backends without memory analysis
        return None


def differentiate(objective, theta0, fd_steps) -> dict[str, Any]:
    """Reverse, forward and central-FD Jacobians of one window, with costs."""

    import jax
    import jax.numpy as jnp

    rows: dict[str, Any] = {}
    value_fn = jax.jit(objective)
    grad_fn = jax.jit(jax.value_and_grad(objective))
    fwd_fn = jax.jit(jax.jacfwd(objective))
    for label, fn in (("value", value_fn), ("reverse", grad_fn), ("forward", fwd_fn)):
        _, compile_and_run = _timed(fn, theta0)
        out, run = _timed(fn, theta0)
        rows[label] = {
            "seconds": run,
            "first_call_seconds": compile_and_run,
            "temp_bytes": _temp_bytes(
                objective
                if label == "value"
                else (
                    jax.grad(objective) if label == "reverse" else jax.jacfwd(objective)
                ),
                theta0,
            ),
        }
        if label == "value":
            q0 = float(out)
        elif label == "reverse":
            reverse = np.asarray(out[1])
        else:
            forward = np.asarray(out)
    fd = []
    for h in fd_steps:
        column = []
        for i in range(theta0.size):
            step = h * max(abs(float(theta0[i])), 1.0)
            e = jnp.zeros_like(theta0).at[i].set(step)
            column.append(
                (float(value_fn(theta0 + e)) - float(value_fn(theta0 - e))) / (2 * step)
            )
        column = np.asarray(column)
        fd.append(
            {
                "relative_step": h,
                "jacobian": column.tolist(),
                "relative_error": (
                    np.abs(reverse - column) / np.maximum(np.abs(column), 1e-300)
                ).tolist(),
            }
        )
    # FD cost for the full row at one step: 2 evaluations per parameter.
    rows["finite_difference"] = {
        "seconds": 2 * theta0.size * rows["value"]["seconds"],
        "temp_bytes": rows["value"]["temp_bytes"],
    }
    return {
        "Q": q0,
        "theta": np.asarray(theta0).tolist(),
        "reverse": reverse.tolist(),
        "forward": forward.tolist(),
        "forward_reverse_relative_difference": float(
            np.max(np.abs(forward - reverse) / np.maximum(np.abs(reverse), 1e-300))
        ),
        "fd": fd,
        "cost": rows,
    }


def long_time_scan(cfg, case_params, r0: float) -> dict[str, Any] | None:
    """Mean flux of each scan trace after ``t_start``, with a block SEM."""

    scan = cfg.get("scan")
    if not scan:
        return None
    points = []
    base = float(np.atleast_1d(np.asarray(case_params.tprim))[0])
    for m in scan["tprim_multipliers"]:
        path = ROOT / scan["traces"].format(multiplier=m)
        if not path.exists():
            print(f"  scan trace missing: {path.name}", flush=True)
            continue
        trace = np.load(path)
        t, q = np.asarray(trace["time"]), np.asarray(trace["heat_flux"])
        q = q[t >= float(scan["t_start"])]
        if q.size < 8:
            continue
        blocks = np.array_split(q, 8)
        means = np.array([b.mean() for b in blocks])
        points.append(
            {
                "multiplier": m,
                "tprim": base * m,
                "R_over_LT": base * m * r0,
                "Q_mean": float(q.mean()),
                "Q_sem": float(means.std(ddof=1) / np.sqrt(means.size)),
            }
        )
    if len(points) < 2:
        return {"points": points}
    x = np.array([p["R_over_LT"] for p in points])
    y = np.array([p["Q_mean"] for p in points])
    w = 1.0 / np.maximum(np.array([p["Q_sem"] for p in points]), 1e-12)
    centre = int(np.argmin(np.abs(np.array([p["multiplier"] for p in points]) - 1.0)))
    near = slice(max(centre - 1, 0), min(centre + 2, len(points)))
    slope, _ = np.polyfit(x[near], y[near], 1, w=w[near])
    fit = np.polyfit(x, y, 1, w=w)
    return {
        "points": points,
        "secant_dQ_dRLT_at_base": float(slope),
        "linear_fit_threshold_R_over_LT": float(-fit[1] / fit[0]),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from gkx.solvers_nonlinear_state_integration import (
        integrate_nonlinear,
        nonlinear_heat_flux_window,
    )

    from nonlinear_gradient_window import build_window_case

    case = build_window_case(ROOT / cfg["deck"], dict(cfg["grid"]))
    params = case["params"]
    # a/L -> R/L: the deck's major radius in units of the minor radius.
    r0 = float(tomllib.loads((ROOT / cfg["deck"]).read_text())["geometry"]["R0"])
    n_species = case["shape"][0]
    names = parameter_names(cfg["parameters"], n_species)
    steps = int(cfg["window_steps"])
    windows = []
    for state_path in cfg["states"]:
        archive = np.load(ROOT / state_path)
        state = jax.numpy.asarray(archive["state"]).astype(np.complex128)
        if state.ndim == 5:
            state = state[None]
        dt = float(archive["adaptive_dt"])
        for k in range(int(cfg["windows_per_state"])):

            def evaluate(p, state=state, dt=dt):
                return nonlinear_heat_flux_window(
                    state,
                    case["grid"],
                    case["geometry"],
                    p,
                    dt=dt,
                    steps=steps,
                    method=case["method"],
                    terms=case["terms"],
                    divergence_knee_steps=None,
                )

            objective, theta0 = make_objective(params, cfg["parameters"], evaluate)
            record = differentiate(objective, theta0, cfg["fd_steps"])
            record |= {"state": Path(state_path).name, "window_index": k, "dt": dt}
            windows.append(record)
            best = min(min(r["relative_error"]) for r in record["fd"])
            print(
                f"{record['state']} w{k}: Q={record['Q']:.4g} "
                f"dQ/dtheta={np.round(record['reverse'], 4)} best AD/FD={best:.1e} "
                f"rev {record['cost']['reverse']['seconds']:.1f}s "
                f"fwd {record['cost']['forward']['seconds']:.1f}s "
                f"val {record['cost']['value']['seconds']:.1f}s",
                flush=True,
            )
            state = integrate_nonlinear(
                state,
                case["grid"],
                case["geometry"],
                params,
                dt,
                steps,
                method=case["method"],
                terms=case["terms"],
                return_fields=False,
            )
    jac = np.array([w["reverse"] for w in windows])
    q = np.array([w["Q"] for w in windows])
    per_seed = {}
    for w in windows:
        per_seed.setdefault(w["state"], []).append(w["reverse"])
    seed_means = np.array([np.mean(v, axis=0) for v in per_seed.values()])
    return {
        "kind": "nonlinear_flux_jacobian",
        "claim_level": "finite_window_discrete_adjoint_not_long_time_sensitivity",
        "deck": cfg["deck"],
        "grid": cfg["grid"],
        "method": case["method"],
        "window_steps": steps,
        "parameters": names,
        "R0_over_a": r0,
        "windows": windows,
        "summary": {
            "Q_mean": float(q.mean()),
            "Q_std": float(q.std(ddof=1)),
            "jacobian_mean": jac.mean(axis=0).tolist(),
            "jacobian_std_over_windows": jac.std(axis=0, ddof=1).tolist(),
            "jacobian_sem_over_seeds": (
                seed_means.std(axis=0, ddof=1) / np.sqrt(len(seed_means))
            ).tolist()
            if len(seed_means) > 1
            else None,
            "dQ_dRLT_mean": float(jac[:, 0].mean() / r0),
            "normalized_stiffness_dlnQ_dlnRLT": float(
                np.mean(jac[:, 0] * np.array([w["theta"][0] for w in windows]) / q)
            ),
            "best_ad_fd_relative_error": float(
                max(
                    min(r["relative_error"][i] for r in w["fd"])
                    for w in windows
                    for i in range(len(names))
                )
            ),
            "max_forward_reverse_relative_difference": float(
                max(w["forward_reverse_relative_difference"] for w in windows)
            ),
        },
        "long_time_scan": long_time_scan(cfg, params, r0),
    }


def write_scan_decks(cfg: dict[str, Any]) -> None:
    text = (ROOT / cfg["deck"]).read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    base = float(raw["species"][0]["tprim"])
    out = ROOT / cfg["scan"]["deck_dir"]
    out.mkdir(parents=True, exist_ok=True)
    for m in cfg["scan"]["tprim_multipliers"]:
        deck = text.replace(f"tprim = {base}", f"tprim = {base * m:.6g}", 1)
        (out / f"tprim_x{m}.toml").write_text(deck, encoding="utf-8")
        print(out / f"tprim_x{m}.toml")


def plot(record: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = record["parameters"]
    labels = {"tprim[0]": r"$a/L_T$", "fprim[0]": r"$a/L_n$"}
    colors = ["#2a6fdb", "#d9822b", "#3a9a5b", "#9b59b6"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    steps = [r["relative_step"] for r in record["windows"][0]["fd"]]
    for i, name in enumerate(names):
        errs = np.array(
            [[r["relative_error"][i] for r in w["fd"]] for w in record["windows"]]
        )
        ax.fill_between(steps, errs.min(0), errs.max(0), color=colors[i], alpha=0.2)
        ax.loglog(
            steps,
            np.median(errs, 0),
            "o-",
            color=colors[i],
            label=labels.get(name, name),
        )
    ax.set_xlabel("relative FD step $h$")
    ax.set_ylabel(r"$|\partial Q_{AD}-\partial Q_{FD}|/|\partial Q_{FD}|$")
    ax.set_title("(a) adjoint vs central differences")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    cost = [w["cost"] for w in record["windows"]]
    methods = [
        ("value", "one $Q$"),
        ("reverse", "reverse AD"),
        ("forward", "forward AD"),
        ("finite_difference", "central FD"),
    ]
    secs = [np.median([c[m]["seconds"] for c in cost]) for m, _ in methods]
    mem = [cost[0][m]["temp_bytes"] for m, _ in methods]
    bars = ax.bar([lab for _, lab in methods], secs, color=["#999999", *colors[:3]])
    for bar, b in zip(bars, mem, strict=True):
        if b:
            ax.annotate(
                f"{b / 2**20:.0f} MB",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("wall time per Jacobian row [s]")
    ax.set_title(f"(b) cost, {len(names)} parameters, {record['window_steps']} steps")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    scan = record.get("long_time_scan") or {}
    pts = scan.get("points", [])
    r0 = record["R0_over_a"]
    if pts:
        ax.errorbar(
            [p["R_over_LT"] for p in pts],
            [p["Q_mean"] for p in pts],
            yerr=[2 * p["Q_sem"] for p in pts],
            fmt="s",
            color="k",
            label="long-time mean (±2 SEM)",
        )
    s = record["summary"]
    x0 = record["windows"][0]["theta"][0] * r0
    slope = s["dQ_dRLT_mean"]
    err = s["jacobian_std_over_windows"][0] / r0
    dx = np.linspace(-0.9, 0.9, 3)
    ax.plot(
        x0 + dx,
        s["Q_mean"] + slope * dx,
        color=colors[0],
        label=rf"window adjoint $\partial Q/\partial(R/L_T)$ = {slope:.1f}±{err:.1f}",
    )
    if "secant_dQ_dRLT_at_base" in scan:
        ax.plot(
            [],
            [],
            " ",
            label=f"long-time secant = {scan['secant_dQ_dRLT_at_base']:.1f}",
        )
    ax.axvline(6.0, color="0.5", ls=":", lw=1)
    ax.text(
        6.02,
        ax.get_ylim()[1] * 0.95,
        "Dimits\nthreshold",
        fontsize=8,
        va="top",
        color="0.4",
    )
    ax.set_xlabel(r"$R/L_T$")
    ax.set_ylabel(r"heat flux $Q$ [gyroBohm]")
    ax.set_title("(c) window derivative vs stiffness")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"written: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--write-scan-decks", action="store_true")
    parser.add_argument("--plot", action="store_true", help="figure from the JSON")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ROOT / cfg["output"]
    if args.write_scan_decks:
        write_scan_decks(cfg)
        return 0
    if not args.plot:
        record = run(cfg)
        output.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"written: {output}")
    plot(json.loads(output.read_text(encoding="utf-8")), ROOT / cfg["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
