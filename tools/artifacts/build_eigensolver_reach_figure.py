"""What the matrix-free eigensolver buys, in one figure.

The README's case for this solver was 155 lines of prose and no picture, which
is the wrong ratio for the single largest capability in the code. Two panels
carry the whole argument:

**Left -- the memory wall.** A dense eigensolve stores the operator, so its cost
is ``16 n^2`` bytes in complex128. The matrix-free path stores a Krylov basis,
``16 n m``, which is linear in ``n``. The four points marked are truncations GKX
actually ran; the largest, ``n = 494,592``, would need 3.6 TiB as a dense matrix,
which is why "faster" is the wrong word for what changed -- the dense path cannot
represent that problem at any speed.

**Right -- the inner solve decides accuracy.** Shift-invert Arnoldi issues many
solves against one shifted operator. Measured against an exact direct inner solve
(the control), Krylov recycling reproduces it to machine precision while plain
GMRES stalls near 1e-2, at both sizes tested. This is what
``docs/solvax_defaults.rst`` records, and it is the reason the inner solver is
treated as a physics-accuracy choice rather than a performance knob.

Numbers come from the committed evidence JSON, never from literals here, so the
figure cannot drift away from what was measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)

DENSE = GKX_COLORS["vermillion"]
FREE = GKX_COLORS["green"]
SIZE_COLORS = (GKX_COLORS["blue"], GKX_COLORS["orange"])

# Truncations GKX has actually solved, with the label used in the docs.
RUN_POINTS = (
    (1536, "(4,8) QI"),
    (6144, "(8,16)"),
    (18816, "(14,28)"),
    (494592, "(hard trunc.)"),
)
KRYLOV_VECTORS = 24  # representative subspace size m for the O(n m) estimate
BYTES_COMPLEX128 = 16


def memory_panel(ax) -> None:
    """Dense n^2 storage against matrix-free n m, with the runs marked."""

    sizes = np.logspace(np.log10(500), np.log10(6.0e5), 200)
    dense = BYTES_COMPLEX128 * sizes**2 / 2**40
    free = BYTES_COMPLEX128 * sizes * KRYLOV_VECTORS / 2**40

    ax.loglog(sizes, dense, color=DENSE, lw=2.0, label="dense  $O(n^2)$")
    ax.loglog(
        sizes,
        free,
        color=FREE,
        lw=2.0,
        label=f"matrix-free  $O(nm)$, $m={KRYLOV_VECTORS}$",
    )

    for size, label in RUN_POINTS:
        ax.plot([size], [BYTES_COMPLEX128 * size**2 / 2**40], "o", color=DENSE, ms=5)
        ax.plot(
            [size],
            [BYTES_COMPLEX128 * size * KRYLOV_VECTORS / 2**40],
            "o",
            color=FREE,
            ms=5,
        )
        # The last point sits at the right edge, so its label goes above-left
        # instead of below-centre where it would lie on the line itself.
        last = size == RUN_POINTS[-1][0]
        ax.annotate(
            label,
            xy=(size, BYTES_COMPLEX128 * size * KRYLOV_VECTORS / 2**40),
            xytext=(-34, 9) if last else (0, -14),
            textcoords="offset points",
            ha="right" if last else "center",
            fontsize=7,
        )

    largest = RUN_POINTS[-1][0]
    ax.annotate(
        f"{BYTES_COMPLEX128 * largest**2 / 2**40:.1f} TiB\nas a dense matrix",
        xy=(largest, BYTES_COMPLEX128 * largest**2 / 2**40),
        xytext=(-96, -6),
        textcoords="offset points",
        fontsize=8,
        color=DENSE,
        arrowprops=dict(arrowstyle="->", color=DENSE, lw=0.9),
    )

    ax.axhline(0.048, color="0.45", ls=":", lw=1.0)  # ~48 GiB, one large GPU
    ax.annotate("one 48 GB GPU", xy=(6.0e2, 0.055), fontsize=7, color="0.35")

    ax.set_xlabel("state size $n = N_\\ell N_m N_\\theta$")
    ax.set_ylabel("operator storage (TiB)")
    ax.set_title("The dense path runs out of memory, not time")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def accuracy_panel(ax, records: list[dict]) -> None:
    """Error against the dense reference, per inner solver, at each size."""

    labels = {
        "exact-lu (control)": "exact LU\n(control)",
        "jax-gmres": "jax.scipy\nGMRES",
        "solvax-gmres": "solvax\nGMRES",
        "gcrot-fifo": "gcrot\nFIFO",
        "gcrot-harmonic": "gcrot\nharmonic",
    }
    order = list(labels)
    positions = np.arange(len(order))
    width = 0.36

    for offset, record in enumerate(records):
        errors = {row["name"]: row["relative_error"] for row in record["results"]}
        heights = [max(errors.get(name, np.nan), 1e-16) for name in order]
        ax.bar(
            positions + (offset - 0.5) * width,
            heights,
            width,
            color=SIZE_COLORS[offset % len(SIZE_COLORS)],
            label=f"$n={record['size']:,}$",
        )

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[name] for name in order], fontsize=7.5)
    ax.set_ylabel("relative error vs dense reference")
    ax.set_title("Recycling matches an exact inner solve")
    ax.set_ylim(1e-16, 3e2)
    ax.axhline(1e-12, color="0.45", ls=":", lw=1.0)
    ax.annotate("machine precision", xy=(-0.45, 2e-13), fontsize=7, color="0.35")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        nargs="+",
        default=[
            Path("docs/_static/shift_invert_recycling.json"),
            Path("docs/_static/shift_invert_recycling_n1536.json"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/eigensolver_reach.png")
    )
    args = parser.parse_args()

    records = [json.loads(path.read_text()) for path in args.evidence]
    records.sort(key=lambda record: record["size"])

    import matplotlib.pyplot as plt

    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        memory_panel(axes[0])
        accuracy_panel(axes[1], records)
        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
        annotate_reference(
            axes[1],
            "tools/campaigns/shift_invert_recycling.py",
            loc="upper right",
        )
        fig.tight_layout()
        written = save_figure(fig, args.output)

    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
