"""Draw ``docs/_static/flow_shear.png``: Cyclone ion heat flux against flow shear.

    python docs/figures/flow_shear.py

Left, constant shear (``[physics] exb_shear``), GKX and GX. Right, an imposed
sinusoidal zonal flow of four periods across the box (``[physics] zonal_flow``),
two seeds. Both cases are ``examples/03_nonlinear_tokamak/case_full.toml`` with
adiabatic electrons: a ``gamma_E = 0`` run to saturation, then each point
restarted from that state and averaged over the last half of 250 time units.
The numbers below are those runs, entered by hand, as (value, standard error);
regenerating them takes about 30 minutes on one GPU. GX values come from the GX
Cyclone deck with the same ``g_exb``.
"""

from __future__ import annotations

from pathlib import Path

from gkx.artifacts.figure_style import GKX_COLORS, figure_style, save_figure

SHEAR = (0.0, 0.05, 0.1, 0.2)  # gamma_E a/v_t
FLUX = {  # label: (color, marker, fill, Q_i in gyro-Bohm units, error)
    "GKX nx=192": ("blue", "o", True, (4.93, 5.23, 3.48, 0.34), (0, 0.12, 0.1, 0.01)),
    "GX nx=192": ("blue", "s", False, (4.46, 4.32, 2.84, 0.26), (0, 0.14, 0.06, 0.01)),
    "GKX nx=64": ("grey", "o", True, (6.07, 6.68, 3.98, 1.04), (0, 0.15, 0.08, 0.01)),
    "GX nx=64": ("grey", "s", False, (7.8, 5.33, 2.85, 0.81), (0, 0.1, 0.12, 0.03)),
}
ZONAL = {  # seed: (Q_i, error) at the peak shears in SHEAR
    22: ((7.37, 6.08, 4.78, 3.78), (0.2, 0.19, 0.09, 0.02)),
    23: ((7.67, 5.91, 5.02, 3.83), (0.28, 0.08, 0.1, 0.07)),
}


def main() -> None:
    import matplotlib.pyplot as plt

    with figure_style(**{"font.size": 8.5, "legend.fontsize": 7.5, "savefig.dpi": 200}):
        fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
        for i, (label, (c, m, fill, q, e)) in enumerate(FLUX.items()):
            x = [s + 0.002 * i for s in SHEAR]
            col = GKX_COLORS[c]
            a.errorbar(
                x,
                q,
                e,
                marker=m,
                ls="-" if fill else "--",
                color=col,
                mfc=col if fill else "white",
                capsize=3,
                ms=5,
                mew=1.2,
                mec=col,
                label=label,
            )
        for seed, color in zip(ZONAL, ("blue", "vermillion")):
            q, e = ZONAL[seed]
            b.errorbar(
                SHEAR,
                q,
                e,
                marker="o",
                capsize=3,
                color=GKX_COLORS[color],
                label=f"seed {seed}",
            )
        a.set(
            xlabel=r"$\gamma_E\,a/v_t$",
            ylabel=r"$Q_i$ (gyro-Bohm)",
            title="constant shear, exb_shear",
        )
        b.set(
            xlabel=r"peak shear $\max|\partial_x^2\phi_{ext}|\,a/v_t$",
            title="four-period zonal flow, zonal_flow",
        )
        for ax in (a, b):
            ax.grid(alpha=0.3)
            ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(
            fig,
            Path(__file__).resolve().parents[1] / "_static/flow_shear.png",
            palette_colors=96,
        )


if __name__ == "__main__":
    main()
