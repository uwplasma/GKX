"""Figures for :doc:`algorithms` -- the methods-and-decisions page.

Four figures, one subcommand each. Every number is read from a tracked
evidence artifact under ``plan/research/scripts/``; nothing is a literal here,
so a figure cannot drift away from the run that produced it.

``velocity-truncation``
    Laguerre ladder convergence and the free-energy l-spectra behind it
    (queue rows Q3/Q8 and Q16; ``2026-09-13-collisional-convergence`` and
    ``2026-09-18-eigen-laguerre-spectrum``).

``chain-ledger``
    Optimized-HLO op and byte counts before and after the shared linked-chain
    transforms, next to the idle-host A/B/A/B wall time of the same change
    (queue row Q9; ``2026-09-14-q9-batched-chain-fft`` and
    ``2026-09-18-q9-idle-host-timing``).

``eigen-cost``
    Wall time, peak RSS and certification verdict of the eigen routes on one
    deck (queue row Q21; ``2026-09-19-inner-solve-cost``).

``cross-code``
    Cyclone ky scans from four codes against the same GX goldens (queue row
    Q11; ``2026-09-14-cross-code-cyclone``).

``all`` runs the four in order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    REFERENCE_STYLE,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "plan" / "research" / "scripts"
STATIC = REPO_ROOT / "docs" / "_static"

COLLISIONAL = EVIDENCE / "2026-09-13-collisional-convergence"
EIGEN_SPECTRUM = EVIDENCE / "2026-09-18-eigen-laguerre-spectrum"
CHAIN_FFT = EVIDENCE / "2026-09-14-q9-batched-chain-fft"
IDLE_TIMING = EVIDENCE / "2026-09-18-q9-idle-host-timing"
INNER_SOLVE = EVIDENCE / "2026-09-19-inner-solve-cost"
CROSS_CODE = EVIDENCE / "2026-09-14-cross-code-cyclone"

#: One colour per collisionality, used by both velocity-truncation panels.
NU_COLORS: dict[float, str] = {
    0.0: GKX_COLORS["vermillion"],
    1e-3: GKX_COLORS["orange"],
    3e-3: GKX_COLORS["sky"],
    1e-2: GKX_COLORS["blue"],
}


def _nu_label(nu: float) -> str:
    return r"$\nu = 0$" if nu == 0.0 else rf"$\nu = {nu:g}$"


# --------------------------------------------------------------- data readers


def read_collisional_ladders() -> dict[float, list[tuple[int, float, bool]]]:
    """``{nu: [(Nl, gamma, settled), ...]}`` from the Q3/Q8 convergence CSV.

    Only GKX rows that are neither superseded nor re-run at a longer horizon
    are kept, so every point in a ladder shares ``t_end = 150``.
    """

    ladders: dict[float, list[tuple[int, float, bool]]] = {}
    with (COLLISIONAL / "summary.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["code"] != "GKX" or row["superseded"] != "False":
                continue
            if float(row["t_end"]) != 150.0:
                continue
            nu = float(row["nu"])
            ladders.setdefault(nu, []).append(
                (int(row["Nl"]), float(row["gamma"]), row["settled"] == "True")
            )
    for points in ladders.values():
        points.sort()
    if not ladders:
        raise SystemExit(f"no GKX ladder rows in {COLLISIONAL / 'summary.csv'}")
    return ladders


def read_eigen_results() -> list[dict]:
    """Every ``RESULT`` line of the Q16 certified eigenpair runs."""

    results: list[dict] = []
    for path in sorted((EIGEN_SPECTRUM / "results").glob("*.txt")):
        if path.name.endswith(".time.txt"):
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text.startswith("RESULT "):
            continue  # nu0-nl64 produced no result line; it is recorded as such
        results.append(json.loads(text[len("RESULT ") :]))
    if not results:
        raise SystemExit(f"no RESULT lines under {EIGEN_SPECTRUM / 'results'}")
    return results


def read_hlo_ledgers() -> dict[str, dict]:
    """The four Q9 runtime-route ledgers, keyed ``<variant>_<precision>``."""

    ledgers: dict[str, dict] = {}
    for variant in ("base", "shared_t"):
        for precision in ("32", "64"):
            path = CHAIN_FFT / "ledgers" / f"{variant}_runtime_{precision}.json"
            ledgers[f"{variant}_{precision}"] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return ledgers


_AB_ARM = re.compile(r"^==\s+(\S+)")
_AB_ALL = re.compile(
    r"^(\w+)\s+all\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"pass=(True|False)\s+verdict=(\w+)"
)


def read_idle_timing() -> list[tuple[str, str, float, str]]:
    """``(arm, kernel, shared/base ratio, verdict)`` from the Q9 A/B/A/B table."""

    rows: list[tuple[str, str, float, str]] = []
    arm = ""
    for line in (
        (IDLE_TIMING / "ab_tables.txt").read_text(encoding="utf-8").splitlines()
    ):
        header = _AB_ARM.match(line)
        if header:
            arm = header.group(1)
            continue
        match = _AB_ALL.match(line.strip())
        if match:
            rows.append((arm, match.group(1), float(match.group(4)), match.group(6)))
    if not rows:
        raise SystemExit(f"no aggregate rows in {IDLE_TIMING / 'ab_tables.txt'}")
    return rows


def read_inner_solve_arms(case: str) -> list[dict]:
    """The Q21 arm table, restricted to one case."""

    arms: list[dict] = []
    for line in (INNER_SOLVE / "summary.txt").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 15 or not fields[0].startswith(("screen/", "prod/")):
            continue
        if fields[1] != case:
            continue
        arms.append(
            {
                "arm": fields[0].split("/", 1)[1],
                "wall_s": float(fields[5]),
                "residual": float(fields[10]),
                "certified": fields[11] == "True",
                "rss_gb": float(fields[14]),
            }
        )
    if not arms:
        raise SystemExit(f"no {case} arms in {INNER_SOLVE / 'summary.txt'}")
    return arms


_CC_GEOMETRY = re.compile(r"^##\s+geometry\s+(\S+)")
_CC_GOLDEN = re.compile(r"^ky ([\d.]+): GX golden gamma ([\d.]+)")
_CC_CONVERGED = re.compile(r"^(\w+):.*->\s*converged\s+([\d.]+)")
_CC_UNCONVERGED = re.compile(r"^(\w+):.*->\s*unconverged")
_CC_LAST_RUNG = re.compile(
    r"r\d+\s+([\d.]+)(?:\s*\[[^\]]*\])?(?:\s*\([^)]*\))?\s*(?=;|->|$)"
)
_CC_GKX_RUNG = re.compile(r"Nl(\d+)\s+([\d.]+)")
_CC_OTHER = re.compile(r"^(gyaradax):\s+g\d+\s+([\d.]+)")


def read_cross_code() -> dict[str, dict]:
    """Per geometry: the GX goldens and each comparison code's ky scan.

    Returns ``{geometry: {"golden": {ky: gamma}, "codes": {code: {ky: (gamma,
    converged)}}, "gkx": [(ky, Nl, gamma), ...]}}``. GKX is kept separately
    because every certified rung is plotted: a single certified pair is not a
    resolution ladder, and where two rungs exist they disagree.
    """

    out: dict[str, dict] = {}
    geometry = ""
    ky = 0.0
    for raw in (
        (CROSS_CODE / "results" / "final_tables.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        head = _CC_GEOMETRY.match(raw)
        if head:
            geometry = head.group(1)
            out[geometry] = {"golden": {}, "codes": {}, "gkx": []}
            continue
        golden = _CC_GOLDEN.match(raw)
        if golden:
            ky = float(golden.group(1))
            out[geometry]["golden"][ky] = float(golden.group(2))
            continue
        line = raw.strip()
        if not line or ": not run" in line or not geometry:
            continue
        codes = out[geometry]["codes"]
        if line.startswith("gkx:"):
            for laguerre, gamma in _CC_GKX_RUNG.findall(line):
                out[geometry]["gkx"].append((ky, int(laguerre), float(gamma)))
            continue
        other = _CC_OTHER.match(line)
        if other:
            codes.setdefault(other.group(1), {}).setdefault(
                ky, (float(other.group(2)), True)
            )
            continue
        converged = _CC_CONVERGED.match(line)
        if converged:
            codes.setdefault(converged.group(1), {})[ky] = (
                float(converged.group(2)),
                True,
            )
            continue
        unconverged = _CC_UNCONVERGED.match(line)
        if unconverged:
            rungs = _CC_LAST_RUNG.findall(line)
            if rungs:
                codes.setdefault(unconverged.group(1), {})[ky] = (
                    float(rungs[-1]),
                    False,
                )
    if not out:
        raise SystemExit("no geometries parsed from the cross-code tables")
    return out


# ----------------------------------------------------------------- the figures


def velocity_truncation(output: Path) -> Path:
    import matplotlib.pyplot as plt

    ladders = read_collisional_ladders()
    spectra = {
        (result["nu"], result["Nl"]): result["laguerre_spectrum"]
        for result in read_eigen_results()
    }
    widest = max(nl for _, nl in spectra)

    with figure_style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.3))

        for nu in sorted(ladders):
            points = ladders[nu]
            color = NU_COLORS.get(nu, GKX_COLORS["grey"])
            left.plot(
                [nl for nl, _, _ in points],
                [gamma for _, gamma, _ in points],
                "-o",
                color=color,
                label=_nu_label(nu),
            )
            for nl, gamma, settled in points:
                if not settled:
                    left.plot([nl], [gamma], "x", color="black", ms=8, mew=1.4)
        left.plot(
            [], [], "x", color="black", ms=8, mew=1.4, label="half-window fit unsettled"
        )
        left.set_xlabel(r"Laguerre resolution $N_\ell$   ($N_m = 96$)")
        left.set_ylabel(r"growth rate $\gamma\,[v_{th}/a]$")
        left.set_title(r"Only $\nu = 0.01$ converges in $N_\ell$")
        left.legend(frameon=False, fontsize=8.5, loc="upper right")
        panel_label(left, "(a)")

        widest_nu = sorted(nu for nu, nl in spectra if nl == widest)
        for nu in widest_nu:
            values = spectra[(nu, widest)]
            right.semilogy(
                range(len(values)),
                values,
                color=NU_COLORS.get(nu, GKX_COLORS["grey"]),
                label=_nu_label(nu),
            )
        right.set_xlabel(r"Laguerre index $\ell$")
        right.set_ylabel("free energy per index (normalized)")
        right.set_title(rf"Spectrum at the widest ladder, $N_\ell = {widest}$")
        right.legend(frameon=False, fontsize=8.5, loc="lower left")
        annotate_reference(
            right,
            "certified eigenvectors, Cyclone s-alpha ITG, $k_y\\rho = 0.55$",
            loc="upper right",
        )
        panel_label(right, "(b)")

        fig.tight_layout()
        return save_figure(fig, output, palette_colors=256)


def _step_counts(ledger: dict) -> dict[str, float]:
    step = ledger["steps"]["rk3"]["counts"]
    return {
        "fft": float(step["fft"]),
        "transpose": float(step["transpose"]),
        "copy": float(step["copy"]),
        "gather": float(step["gather"]),
        "bytes": float(step["bytes_written"]),
    }


def chain_ledger(output: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    ledgers = read_hlo_ledgers()
    base = _step_counts(ledgers["base_32"])
    shared = _step_counts(ledgers["shared_t_32"])
    keys = ["fft", "transpose", "copy", "gather", "bytes"]
    labels = ["FFT", "transpose", "copy", "gather", "bytes\nwritten"]

    timing = read_idle_timing()

    with figure_style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.3))

        index = np.arange(len(keys))
        left.bar(
            index - 0.19,
            [1.0] * len(keys),
            width=0.36,
            color=GKX_COLORS["grey"],
            label="before",
        )
        left.bar(
            index + 0.19,
            [shared[key] / base[key] for key in keys],
            width=0.36,
            color=GKX_COLORS["green"],
            label="shared chain transforms",
        )
        for position, key in zip(index, keys, strict=True):
            left.text(
                position + 0.19,
                shared[key] / base[key] + 0.03,
                f"{base[key]:,.0f}\n→ {shared[key]:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        left.set_xticks(index)
        left.set_xticklabels(labels)
        left.set_ylim(0.0, 1.32)
        left.set_ylabel("per RK3 step, relative to before")
        left.set_title("Optimized HLO: the work the graph materializes")
        left.legend(frameon=False, fontsize=8.5, loc="upper right")
        panel_label(left, "(a)")

        kernels = sorted({kernel for _, kernel, _, _ in timing})
        offsets = np.linspace(-0.27, 0.27, len(kernels))
        arms = sorted({arm for arm, _, _, _ in timing})
        for offset, kernel in zip(offsets, kernels, strict=True):
            xs = [arms.index(arm) + offset for arm, k, _, _ in timing if k == kernel]
            ys = [ratio for _, k, ratio, _ in timing if k == kernel]
            right.plot(xs, ys, "o", ms=6, label=kernel)
        right.axhline(1.0, **REFERENCE_STYLE)
        right.set_xticks(range(len(arms)))
        right.set_xticklabels(arms, rotation=30, ha="right")
        right.set_ylabel("wall time, shared / before")
        right.set_title("Measured wall time on an idle host")
        right.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        annotate_reference(
            right,
            "A/B/A/B, blocked, idle host; 1.0 is no change",
            loc="lower right",
        )
        panel_label(right, "(b)")

        fig.tight_layout()
        return save_figure(fig, output, palette_colors=256)


def eigen_cost(output: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    prod = read_inner_solve_arms("prod")
    screen = read_inner_solve_arms("r96")

    with figure_style():
        fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.3))

        index = np.arange(len(prod))
        left.bar(
            index,
            [arm["wall_s"] for arm in prod],
            width=0.62,
            color=[
                GKX_COLORS["blue"]
                if arm["arm"].endswith("adaptive")
                else GKX_COLORS["orange"]
                for arm in prod
            ],
        )
        left.legend(
            handles=[
                Patch(color=GKX_COLORS["blue"], label="adaptive (the default)"),
                Patch(color=GKX_COLORS["orange"], label="matrix-free shift-invert"),
            ],
            frameon=False,
            fontsize=8.5,
            loc="upper left",
        )
        for position, arm in zip(index, prod, strict=True):
            left.text(
                position,
                arm["wall_s"] + 12.0,
                f"{arm['wall_s']:.0f} s\n{arm['rss_gb']:.1f} GB",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        left.set_xticks(index)
        left.set_xticklabels([arm["arm"] for arm in prod], rotation=30, ha="right")
        left.set_ylim(0.0, max(arm["wall_s"] for arm in prod) * 1.42)
        left.set_ylabel("wall time (s), peak RSS labelled")
        left.set_title("Certified routes on the production deck")
        panel_label(left, "(a)")

        for certified, color, marker in (
            (True, GKX_COLORS["green"], "o"),
            (False, GKX_COLORS["vermillion"], "X"),
        ):
            arms = [arm for arm in screen if arm["certified"] is certified]
            right.semilogy(
                [arm["wall_s"] for arm in arms],
                [arm["residual"] for arm in arms],
                marker,
                ms=7,
                linestyle="none",
                color=color,
                label="certified" if certified else "rejected by the gate",
            )
        right.axhline(1e-9, **REFERENCE_STYLE)
        right.text(
            0.02,
            1.4e-9,
            r"$10^{-9}$",
            transform=right.get_yaxis_transform(),
            fontsize=8,
        )
        right.set_xlabel("wall time (s)")
        right.set_ylabel("original-operator relative residual")
        right.set_title("Screening deck: cost against the residual reached")
        right.legend(frameon=False, fontsize=8.5, loc="center right")
        panel_label(right, "(b)")

        fig.tight_layout()
        return save_figure(fig, output, palette_colors=256)


CROSS_CODE_STYLE: dict[str, tuple[str, str, str]] = {
    "gs2": (GKX_COLORS["orange"], "s", "GS2"),
    "stella": (GKX_COLORS["purple"], "^", "stella"),
    "gyaradax": (GKX_COLORS["green"], "D", "gyaradax"),
}

GEOMETRY_NAMES = {"M": "M (circular Miller)", "S": "S (s-alpha)"}


def cross_code(output: Path) -> Path:
    import matplotlib.pyplot as plt

    scans = read_cross_code()
    geometries = sorted(scans)

    with figure_style():
        fig, axes = plt.subplots(1, len(geometries), figsize=(10.4, 4.3), sharey=False)
        for ax, geometry in zip(np.atleast_1d(axes), geometries, strict=True):
            golden = scans[geometry]["golden"]
            ax.plot(
                sorted(golden),
                [golden[ky] for ky in sorted(golden)],
                label="GX golden",
                **REFERENCE_STYLE,
            )
            for code, points in sorted(scans[geometry]["codes"].items()):
                color, marker, name = CROSS_CODE_STYLE.get(
                    code, (GKX_COLORS["grey"], "v", code)
                )
                for settled in (True, False):
                    kys = [
                        ky for ky, (_, ok) in sorted(points.items()) if ok is settled
                    ]
                    if not kys:
                        continue
                    ax.plot(
                        kys,
                        [points[ky][0] for ky in kys],
                        marker,
                        linestyle="none",
                        ms=7,
                        color=color,
                        markerfacecolor=color if settled else "white",
                        markeredgecolor=color,
                        markeredgewidth=1.4,
                        label=name if settled else None,
                    )
            rungs = scans[geometry]["gkx"]
            ax.plot(
                [ky for ky, _, _ in rungs],
                [gamma for _, _, gamma in rungs],
                "o",
                linestyle="none",
                ms=7,
                color=GKX_COLORS["blue"],
                label="GKX, certified" if rungs else None,
            )
            for ky, laguerre, gamma in rungs:
                ax.annotate(
                    rf"$N_\ell={laguerre}$",
                    xy=(ky, gamma),
                    xytext=(-6, 6),
                    textcoords="offset points",
                    ha="right",
                    fontsize=7.5,
                    color=GKX_COLORS["blue"],
                )
            ax.set_xlabel(r"$k_y \rho_i$")
            ax.set_title(f"geometry {GEOMETRY_NAMES.get(geometry, geometry)}")
            ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        first = np.atleast_1d(axes)[0]
        first.set_ylabel(r"growth rate $\gamma\,[v_{th}/a]$")
        annotate_reference(
            first,
            "open markers: that code's own resolution scan did not converge",
            loc="lower right",
        )
        fig.tight_layout()
        return save_figure(fig, output, palette_colors=256)


FIGURES = {
    "velocity-truncation": (velocity_truncation, "methods_velocity_truncation.png"),
    "chain-ledger": (chain_ledger, "methods_chain_transform_ledger.png"),
    "eigen-cost": (eigen_cost, "methods_eigen_route_cost.png"),
    "cross-code": (cross_code, "methods_cross_code_ky_scan.png"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=[*FIGURES, "all"])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=STATIC,
        help="directory the PNGs are written to (default: docs/_static)",
    )
    args = parser.parse_args(argv)

    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        builder, filename = FIGURES[name]
        print(f"written: {builder(args.out_dir / filename)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
