"""What ``run_to = "saturation"`` actually does, on three real runs.

The README describes the stop policy in words. Words are the wrong medium for
it: the policy is a statement about the *shape* of a heat-flux trace, and the
only way to see whether a reader has understood it is to show them a trace and
the window the policy chose on it.

This figure replays the shipped policy --
:func:`gkx.diagnostics.saturation.saturation_stop_decision`, unmodified, at its
default ``rel_sem = 0.05`` -- over heat-flux traces already tracked in
``docs/_static``. Nothing here is drawn by hand and nothing is simulated: every
curve is a saved GKX nonlinear run, and every window, mean, SEM and verdict is
what the policy returns when it is asked about that trace.

The runtime asks the same question after each integration chunk, on a trace
that grows under it, and stops at the first chunk whose answer is yes. The
replay asks it on every growing prefix instead -- finer than the runtime's
128-step chunking, so the stop time drawn is the earliest the policy could
fire, and the real run would stop at or just after it.

Three cases, because one is not enough to show what the gates are for:

* **a** stops early with room to spare -- the plateau is clean, the correlated
  SEM is well under threshold, and two thirds of ``t_max`` is never integrated.
* **b** takes longer to get there -- a long transient and larger excursions, so
  the window has to grow until the relative SEM lands exactly on ``0.05``.
* **c** never gets there. An under-resolved grid on a short horizon leaves the
  relative SEM at twice the threshold, so the run uses its whole ``t_max`` and
  the summary says *not saturated by the time horizon* rather than presenting a
  mean it cannot stand behind.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)
from gkx.diagnostics.saturation import SaturationStopConfig, saturation_stop_decision

TIME_LABEL = r"$t \, c_s/a$"
FLUX_LABEL = r"$Q/Q_{\mathrm{gB}}$"

TRACE = GKX_COLORS["blue"]
ACCEPTED = GKX_COLORS["green"]
REJECTED = GKX_COLORS["vermillion"]
SPINUP = GKX_COLORS["grey"]
UNUSED = "#B8B8B8"

#: Tracked pilot traces, chosen to make the three outcomes of the policy
#: visible side by side. Each ``csv`` has a companion ``.json`` recording the
#: ``.out.nc`` bundle it was extracted from; that provenance is copied into this
#: figure's own JSON rather than restated here.
CASES: tuple[dict[str, str], ...] = (
    {
        "panel": "a",
        "key": "saturates_early",
        "csv": "docs/_static/external_vmec_itermodel_nonlinear_t350_n64_pilot.traces.csv",
        "headline": "Saturates, horizon to spare",
    },
    {
        "panel": "b",
        "key": "saturates_late",
        "csv": "docs/_static/external_vmec_circular_holdout_nonlinear_t450_n64_pilot.traces.csv",
        "headline": "Slower: the window must grow",
    },
    {
        "panel": "c",
        "key": "never_saturates",
        "csv": "docs/_static/external_vmec_dshape_nonlinear_t150_n32_pilot.traces.csv",
        "headline": "Never saturates",
    },
)

#: The policy's internal gate names, in the words a reader of the README has.
GATE_NAMES: dict[str, str] = {
    "flux_indistinguishable_from_zero": "flux above the noise floor",
    "tau_ac_unresolved": "resolved $\\tau_{ac}$",
    "window_below_min_window": "window $\\geq 10\\tau_{ac}$",
    "rel_sem_above_threshold": "relative SEM",
    "window_not_stationary": "half-window agreement",
    "guard_not_stationary": "$W_\\phi$ guard stationarity",
    "Wg_guard_not_stationary": "$W_g$ guard stationarity",
}


def load_trace(path: Path) -> dict[str, Any]:
    """Read a tracked ``t,heat_flux[,wphi,wg]`` trace and its provenance."""

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: empty trace")
    columns = rows[0].keys()
    if "t" not in columns or "heat_flux" not in columns:
        raise ValueError(f"{path}: expected 't' and 'heat_flux' columns")

    def column(name: str) -> np.ndarray | None:
        if name not in columns:
            return None
        return np.array([float(row[name]) for row in rows])

    companion = path.with_suffix("").with_suffix(".json")
    meta: dict[str, Any] = {}
    if companion.exists():
        meta = json.loads(companion.read_text())
    return {
        "t": column("t"),
        "heat_flux": column("heat_flux"),
        "wphi": column("wphi"),
        "wg": column("wg"),
        "csv": str(path),
        "source": meta.get("source"),
        "label": meta.get("label"),
        "claim_level": meta.get("claim_level"),
    }


def replay(trace: dict[str, Any], cfg: SaturationStopConfig) -> dict[str, Any]:
    """Ask the shipped policy about every growing prefix of a tracked trace.

    Returns the first decision that says "saturated" together with the prefix
    it fired on, or -- if none does -- the decision the policy would report at
    ``t_max``, which is the verdict that reaches the run summary.
    """

    t, y = trace["t"], trace["heat_flux"]
    guard, free_energy = trace["wphi"], trace["wg"]
    previous: dict[str, Any] | None = None
    for n in range(max(int(cfg.min_samples), 8), t.size + 1):
        decision = saturation_stop_decision(
            t[:n],
            y[:n],
            guard=None if guard is None else guard[:n],
            free_energy_guard=None if free_energy is None else free_energy[:n],
            config=cfg,
        )
        if decision["saturated"]:
            return {
                "decision": decision,
                "stop_index": n,
                "stop_time": float(t[n - 1]),
                # Why the policy was still saying "not yet" one sample earlier:
                # the gate that was actually binding at the stop.
                "binding_gates": list(previous["reasons"]) if previous else [],
            }
        previous = decision
    return {
        "decision": saturation_stop_decision(
            t,
            y,
            guard=guard,
            free_energy_guard=free_energy,
            config=cfg,
        ),
        "stop_index": None,
        "stop_time": None,
        "binding_gates": [],
    }


def stats_box(result: dict[str, Any], cfg: SaturationStopConfig) -> str:
    """The numbers the policy decides on, plus which gate was last to give way.

    The binding gate is the interesting half of the answer and differs across
    these three runs: one is held back by the free-energy guard long after its
    flux is quiet, one by the relative SEM, and one never clears it at all.
    """

    decision = result["decision"]
    comparison = "$\\leq$" if decision["rel_sem"] <= cfg.rel_sem else "$>$"
    if decision["saturated"]:
        held = [GATE_NAMES.get(name, name) for name in result["binding_gates"]]
        gate_line = "last gate to clear: " + (", ".join(held) if held else "none")
    else:
        blocked = [GATE_NAMES.get(name, name) for name in decision["reasons"]]
        gate_line = "gate never cleared: " + ", ".join(blocked)
    return (
        f"mean {decision['mean']:.2f} $\\pm$ {decision['sem']:.2f}\n"
        f"rel SEM {decision['rel_sem']:.3f} {comparison} {cfg.rel_sem:.2f}\n"
        f"$\\tau_{{ac}}$ {decision['tau_ac']:.1f}, window "
        f"{decision['window_span']:.0f} vs "
        f"$10\\tau$ = {decision['min_window']:.0f}\n"
        f"{gate_line}"
    )


def draw_panel(ax, case: dict[str, Any], cfg: SaturationStopConfig) -> None:
    """One trace, the window the policy chose on it, and its verdict."""

    trace, result = case["trace"], case["result"]
    decision = result["decision"]
    t, y = trace["t"], trace["heat_flux"]
    saturated = decision["saturated"]
    accent = ACCEPTED if saturated else REJECTED
    stop_time = result["stop_time"] if saturated else float(t[-1])
    window_min = decision["window_tmin"]

    # Everything the run would never have integrated: drawn, but visibly not
    # part of the answer. This is the compute the policy saves.
    if saturated:
        tail = t >= stop_time
        ax.plot(t[tail], y[tail], color=UNUSED, lw=1.1, zorder=1)
    # Spin-up: discarded by the policy at the first crossing of the trace median.
    head = t <= window_min
    ax.plot(t[head], y[head], color=SPINUP, lw=1.4, zorder=2)
    body = (t >= window_min) & (t <= stop_time)
    ax.plot(t[body], y[body], color=TRACE, lw=1.5, zorder=3)

    ax.axvspan(window_min, stop_time, color=accent, alpha=0.13, lw=0, zorder=0)
    ax.axvline(window_min, color=SPINUP, lw=1.1, ls=(0, (3, 3)), zorder=4, alpha=0.9)
    ax.axvline(stop_time, color=accent, lw=1.8, ls="-", zorder=4)

    mean, sem = decision["mean"], decision["sem"]
    ax.hlines(mean, window_min, stop_time, color=accent, lw=2.0, zorder=6)
    ax.fill_between(
        [window_min, stop_time],
        mean - sem,
        mean + sem,
        color=accent,
        alpha=0.42,
        lw=0,
        zorder=5,
    )

    horizon = float(t[-1])
    # A little headroom past t_max so the stop marker never hides in the spine,
    # and enough above the trace that the numbers box sits on empty paper
    # rather than over the plateau it is describing.
    ax.set_xlim(0.0, horizon * 1.02)
    ax.set_ylim(0.0, float(np.max(y)) * 1.55)
    ax.set_xlabel(TIME_LABEL)
    if saturated:
        used = 100.0 * stop_time / horizon
        verdict = (
            f"{case['headline']}\nstopped at $t$ = {stop_time:.0f}, "
            f"{used:.0f}% of $t_{{\\max}}$ = {horizon:.0f}"
        )
    else:
        verdict = (
            f"{case['headline']}\nran the full $t_{{\\max}}$ = {horizon:.0f}, "
            "reported not saturated"
        )
    ax.set_title(verdict, fontsize=10.5)
    annotate_reference(ax, stats_box(result, cfg), loc="upper right")
    panel_label(ax, case["panel"], dx=-0.115, dy=1.18)


def figure(cases: list[dict[str, Any]], cfg: SaturationStopConfig, out: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    with figure_style(**{"savefig.dpi": 150, "font.size": 11.5}):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharey=False)
        for ax, case in zip(axes, cases):
            draw_panel(ax, case, cfg)
        axes[0].set_ylabel(FLUX_LABEL)

        handles = [
            Line2D([], [], color=SPINUP, lw=1.6, label="spin-up, discarded"),
            Line2D(
                [],
                [],
                color=SPINUP,
                lw=1.2,
                ls=(0, (3, 3)),
                label="window start (first median crossing)",
            ),
            Line2D([], [], color=TRACE, lw=1.6, label="averaged trace"),
            Line2D([], [], color=UNUSED, lw=1.6, label="never integrated"),
            Patch(facecolor=ACCEPTED, alpha=0.35, label="window: saturated, run stops"),
            Patch(
                facecolor=REJECTED,
                alpha=0.35,
                label="window: not saturated at $t_{\\max}$",
            ),
            Line2D([], [], color="#444444", lw=2.2, label="reported mean $\\pm$ SEM"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=9.5,
            bbox_to_anchor=(0.5, 0.012),
        )
        fig.suptitle(
            'run_to = "saturation": the stop policy replayed on three tracked '
            "GKX heat-flux traces",
            x=0.006,
            y=0.975,
            ha="left",
            fontsize=13,
        )
        # Explicit, because tight_layout reserves room above the axes for the
        # panel labels and leaves a band of empty paper under the suptitle.
        fig.subplots_adjust(
            left=0.058, right=0.995, top=0.805, bottom=0.225, wspace=0.20
        )
        return save_figure(fig, out, palette_colors=256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rel-sem",
        type=float,
        default=SaturationStopConfig.rel_sem,
        help="stop threshold on the autocorrelation-corrected relative SEM",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/saturation_examples.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/_static/saturation_examples.json"),
    )
    args = parser.parse_args()

    cfg = SaturationStopConfig(rel_sem=float(args.rel_sem))
    cases: list[dict[str, Any]] = []
    for spec in CASES:
        trace = load_trace(Path(spec["csv"]))
        cases.append({**spec, "trace": trace, "result": replay(trace, cfg)})

    records = []
    for case in cases:
        trace, result = case["trace"], case["result"]
        decision = result["decision"]
        horizon = float(trace["t"][-1])
        stop_time = result["stop_time"]
        records.append(
            {
                "panel": case["panel"],
                "key": case["key"],
                "headline": case["headline"],
                "data_origin": "tracked_gkx_nonlinear_run",
                "trace_csv": trace["csv"],
                "trace_source_bundle": trace["source"],
                "trace_label": trace["label"],
                "trace_claim_level": trace["claim_level"],
                "n_samples": int(trace["t"].size),
                "t_max": horizon,
                "guards_present": trace["wphi"] is not None,
                "saturated": bool(decision["saturated"]),
                "stop_time": stop_time,
                "stop_index": result["stop_index"],
                "horizon_fraction_used": (
                    1.0 if stop_time is None else stop_time / horizon
                ),
                "gates_binding_at_stop": result["binding_gates"],
                "decision": decision,
            }
        )

    print(f"{'panel':>5} {'saturated':>10} {'stop':>8} {'t_max':>7} {'rel_sem':>8}")
    for record in records:
        stop = "-" if record["stop_time"] is None else f"{record['stop_time']:.1f}"
        print(
            f"{record['panel']:>5} {str(record['saturated']):>10} {stop:>8} "
            f"{record['t_max']:7.1f} {record['decision']['rel_sem']:8.4f}"
        )

    written = figure(cases, cfg, args.output)
    print(f"written: {written}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "kind": "saturation_examples",
                "data_origin": "real",
                "claim_level": (
                    "shipped stop policy replayed unmodified over heat-flux "
                    "traces from tracked GKX nonlinear runs; no synthetic or "
                    "hand-drawn data"
                ),
                "policy_module": "src/gkx/diagnostics/saturation.py",
                "policy_entry_point": (
                    "gkx.diagnostics.saturation.saturation_stop_decision"
                ),
                "method": (
                    "The runtime evaluates the stop decision after each "
                    "integration chunk (<=128 steps) on a growing trace and "
                    "stops at the first chunk that saturates. This replay "
                    "evaluates it on every growing prefix of the tracked "
                    "trace, so the stop time reported is the earliest the "
                    "policy could fire; a real run stops at or just after it."
                ),
                "gates": [
                    "flux distinguishable from zero",
                    "tau_ac resolved (autocorrelation crosses zero inside the "
                    "window, and tau_ac > sampling interval)",
                    "window span >= min_window (10 tau_ac when unset)",
                    "autocorrelation-corrected relative SEM <= rel_sem",
                    "half-window means agree within 2x their combined SEM",
                    "same half-window stationarity on the Wphi and Wg guards",
                ],
                "config": {
                    "rel_sem": cfg.rel_sem,
                    "min_window": cfg.min_window,
                    "min_samples": cfg.min_samples,
                },
                "figure": str(args.output),
                "cases": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
