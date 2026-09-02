"""Build the documented VMEX closed-mirror GKX figure, movie, and record."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gkx.geometry import from_vmex_mirror  # noqa: E402
from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.objectives.core import (  # noqa: E402
    SOLVER_OBJECTIVE_NAMES,
    solver_objective_vector_from_geometry,
)
from vmex.mirror import (  # noqa: E402
    MirrorConfig,
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    gk_closed_fieldline_geometry,
    solve_fixed_boundary,
)

# Admission bars for the shipped closed racetrack. The first three are the ones
# VMEX's own closed lane asserts in
# ``tests/mirror/test_splines.py::test_closed_circular_limit_reaches_ftol_with_independent_strong_force``
# and this configuration meets them at machine precision. The strong-form bar is
# ours, and it is deliberately loose: on this racetrack ``force.normalized_rms``
# plateaus between 4.6e-3 and 5.1e-3 across every resolution measured (``ns``
# 5-9, ``mpol`` 4-6, ``coefficient_count`` 16-64) with no downward trend, so
# VMEX's 1.6e-4 circular-limit figure is not reachable here. Whether that
# plateau is the leg-return curvature junction or something undriven is asked at
# https://github.com/uwplasma/vmex/issues/211 and is unanswered; until it is
# answered this bar only certifies the two orders of magnitude that separate a
# solved state from the 0.61 seed, and must not be read as an accuracy claim.
MAX_FORCE_RESIDUAL_NORMALIZED_RMS = 1.0e-2
MAX_WEAK_RESIDUAL_MAXIMUM = 1.0e-12
MAX_NORMALIZED_DIVERGENCE_RMS = 1.0e-12
#: The published residual must beat the seed's by this factor. The shipped case
#: separates by 126 (0.61 seed against 4.8e-3 solved), five times this bar, so
#: it only catches a case rebuilt on its guess, not a merely marginal solve.
MIN_SEED_RESIDUAL_SEPARATION = 25.0


def equilibrium_admission_failures(equilibrium: dict) -> list[str]:
    """Return the reasons ``equilibrium`` is not a publishable solved state.

    The whole point of #173 is that shipping the solver's initial guess was
    invisible, so the builder and the release gate that reads the shipped record
    both check the same list rather than each carrying its own idea of solved.
    """

    failures = []
    if not equilibrium.get("converged"):
        failures.append("the solve did not report converged")
    if not equilibrium.get("solve_lambda"):
        failures.append(
            "solve_lambda was not set: with the default the solve stops after "
            "four iterations at a residual of 0.55 and is not a solve"
        )
    if not equilibrium.get("iterations", 0) > 0:
        failures.append("no solver iteration was taken, so this is a seed")
    checks = (
        ("force_residual_normalized_rms", MAX_FORCE_RESIDUAL_NORMALIZED_RMS),
        ("variational_maximum", MAX_WEAK_RESIDUAL_MAXIMUM),
        ("staggered_weak_force_maximum", MAX_WEAK_RESIDUAL_MAXIMUM),
        ("normalized_divergence_rms", MAX_NORMALIZED_DIVERGENCE_RMS),
    )
    for name, bound in checks:
        value = equilibrium.get(name)
        if value is None:
            failures.append(f"{name} is missing")
        elif not float(value) < bound:
            failures.append(f"{name} is {float(value):.6g}, not below {bound:g}")
    solved = equilibrium.get("force_residual_normalized_rms")
    seed = equilibrium.get("seed_force_residual_normalized_rms")
    if solved is not None and seed is not None:
        # The published state must be resolvably different from the one the
        # solver started at. Without this, a case whose seed happened to sit
        # under the bar could be shipped unsolved and still pass.
        if not float(solved) * MIN_SEED_RESIDUAL_SEPARATION < float(seed):
            failures.append(
                f"the solved residual {float(solved):.6g} is not "
                f"{MIN_SEED_RESIDUAL_SEPARATION:g}x below the seed residual "
                f"{float(seed):.6g}, so this state is indistinguishable from "
                "the initial guess"
            )
    return failures


def _revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _seed_force_residual(setup, inputs: dict) -> float:
    """Return the normalized force residual of the state this case used to ship."""

    from vmex.mirror.forces import isotropic_force_residual, mirror_energy

    flux, current = inputs["axial_flux_derivative"], inputs["current_derivative"]
    grid = setup.discretization.grid
    seed = setup.discretization.evaluate_state(setup.initial_state)
    energy = mirror_energy(
        seed,
        grid,
        axial_flux_derivative=flux,
        current_derivative=current,
        axis=setup.axis,
    )
    return float(
        np.asarray(
            isotropic_force_residual(
                energy,
                grid,
                state=seed,
                axial_flux_derivative=flux,
                current_derivative=current,
                axis=setup.axis,
                closed=True,
            ).normalized_rms
        )
    )


def _case():
    inputs = {
        "ns": 5,
        "mpol": 4,
        "nxi": 4,
        "coefficient_count": 16,
        "straight_length": 4.0,
        "return_radius": 2.0,
        "semi_major": 0.4,
        "semi_minor": 0.3,
        "section_turns": 0,
        "axial_flux_derivative": 0.02,
        "current_derivative": 0.0,
        "ntheta": 32,
        "arc_oversample": 8,
        "n_laguerre": 2,
        "n_hermite": 3,
        "quadrature_order": 3,
        "ftol": 1.0e-12,
        "max_iterations": 2000,
    }
    resolution = MirrorResolution(
        ns=inputs["ns"], mpol=inputs["mpol"], nxi=inputs["nxi"]
    )
    setup = build_stellarator_mirror_hybrid(
        resolution,
        coefficient_count=inputs["coefficient_count"],
        straight_length=inputs["straight_length"],
        return_radius=inputs["return_radius"],
        semi_major=inputs["semi_major"],
        semi_minor=inputs["semi_minor"],
        section_turns=inputs["section_turns"],
        axial_flux_derivative=inputs["axial_flux_derivative"],
        quadrature_order=inputs["quadrature_order"],
    )
    # Solve before measuring. Building on ``setup.discretization.evaluate_state(
    # setup.initial_state)`` published the seeded stream function on the
    # prescribed nested-ellipse surfaces -- a normalized force residual of 0.61 --
    # as though it were an equilibrium, which moved every objective this record
    # carries (#173). ``solve_lambda=True`` is required: with the default the
    # solve reports converged after four iterations at a residual of 0.55.
    solved = solve_fixed_boundary(
        setup.initial_state,
        setup.boundary,
        setup.discretization,
        MirrorConfig(
            resolution=resolution,
            ftol=inputs["ftol"],
            max_iterations=inputs["max_iterations"],
        ),
        axial_flux_derivative=inputs["axial_flux_derivative"],
        current_derivative=inputs["current_derivative"],
        solve_lambda=True,
        axis=setup.axis,
        require_convergence=True,
    ).evaluated
    state = solved.state
    equilibrium = {
        "converged": bool(solved.converged),
        "iterations": int(solved.iterations),
        "solve_lambda": True,
        "ftol": inputs["ftol"],
        "force_residual_normalized_rms": float(np.asarray(solved.force.normalized_rms)),
        "variational_maximum": float(np.asarray(solved.variational.maximum)),
        "staggered_weak_force_maximum": float(
            np.asarray(solved.staggered_weak_force.maximum)
        ),
        "normalized_divergence_rms": float(
            np.asarray(solved.normalized_divergence_rms)
        ),
        "resolution": {
            key: inputs[key]
            for key in ("ns", "mpol", "nxi", "coefficient_count", "quadrature_order")
        },
        "admission_bars": {
            "force_residual_normalized_rms": MAX_FORCE_RESIDUAL_NORMALIZED_RMS,
            "variational_maximum": MAX_WEAK_RESIDUAL_MAXIMUM,
            "staggered_weak_force_maximum": MAX_WEAK_RESIDUAL_MAXIMUM,
            "normalized_divergence_rms": MAX_NORMALIZED_DIVERGENCE_RMS,
        },
        # Measured on the state the builder used to publish, so the record shows
        # what the solve bought instead of asking a reader to take it on trust.
        "seed_force_residual_normalized_rms": _seed_force_residual(setup, inputs),
    }
    failures = equilibrium_admission_failures(equilibrium)
    if failures:
        raise SystemExit(
            "refusing to publish this case as an equilibrium: "
            + "; ".join(failures)
            + ". A seeded state was shipped as an equilibrium once (#173); the "
            "builder fails here rather than write the record."
        )
    geometry = from_vmex_mirror(
        state,
        setup.discretization,
        setup.axis,
        axial_flux_derivative=inputs["axial_flux_derivative"],
        current_derivative=inputs["current_derivative"],
        ntheta=inputs["ntheta"],
        arc_oversample=inputs["arc_oversample"],
    )
    objectives = solver_objective_vector_from_geometry(
        geometry,
        n_laguerre=inputs["n_laguerre"],
        n_hermite=inputs["n_hermite"],
        ny=4,
        selected_ky_index=1,
    )
    jax.block_until_ready(objectives)
    mapping = gk_closed_fieldline_geometry(
        state,
        setup.discretization,
        setup.axis,
        axial_flux_derivative=inputs["axial_flux_derivative"],
        current_derivative=inputs["current_derivative"],
        ntheta=inputs["ntheta"],
        arc_oversample=inputs["arc_oversample"],
    )
    return inputs, setup, geometry, mapping, np.asarray(objectives), equilibrium


def _style_axis_3d(axis) -> None:
    axis.set_axis_off()
    axis.set_box_aspect((1.0, 0.45, 1.3))


def build(output_dir: Path, *, movie: bool = True) -> dict:
    inputs, setup, geometry, mapping, objective, equilibrium = _case()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    theta = np.asarray(geometry.theta)
    bmag = np.asarray(geometry.bmag_profile)
    bgrad = np.asarray(geometry.bgrad_profile)
    xyz = np.asarray(setup.axis.centerline)
    # Field-line Cartesian points remain diagnostic metadata owned by VMEX.
    line_xyz = np.asarray(mapping["vmex_mirror"]["xyz"])
    # Roll the plotted parallel profiles onto the |B| minimum so the panel reads as the well its
    # periodicity implies rather than as a barrier with half a well at each edge. Display only:
    # the record uses the unrolled geometry and the objectives are invariant under the shift.
    # docs/geometry.rst carries that measurement.
    #
    # This used to take the fundamental's trough instead of argmin, because on the seeded state
    # the low-field leg was a flat plateau straddling the seam and argmin picked an arbitrary
    # sample within it. The solved equilibrium has no plateaus: it is a smooth two-hump profile
    # with a maximum at each return bend, so its fundamental is eight times smaller than its
    # second harmonic and that heuristic is now the ill-conditioned one. argmin is unambiguous
    # here and is what the axis label and the record's plot_convention actually claim.
    roll = bmag.size // 2 - int(bmag.argmin())

    def centered(values):
        return np.roll(np.asarray(values), roll)

    figure = plt.figure(figsize=(12.0, 4.2), constrained_layout=True)
    ax3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.6", lw=1.2)
    points = ax3d.scatter(
        line_xyz[:, 0], line_xyz[:, 1], line_xyz[:, 2], c=bmag, s=18, cmap="viridis"
    )
    figure.colorbar(points, ax=ax3d, shrink=0.62, pad=-0.03, label=r"$B/B_{ref}$")
    ax3d.set_title("closed VMEX mirror field line")
    ax3d.view_init(elev=24, azim=-58)
    _style_axis_3d(ax3d)

    # On the solved equilibrium |B| modulates by 3.8% about 1.0 while bgrad peaks at 0.0056, a
    # 200:1 range. Shared axes drew bgrad as a flat line on the zero of a 0-to-1.04 scale, so the
    # panel that is titled for the mirror force did not show it. Give bgrad its own axis.
    axb = figure.add_subplot(1, 3, 2)
    field = axb.plot(theta / np.pi, centered(bmag), lw=2.2, label=r"$B/B_{ref}$")
    # No y label on the left: the colourbar immediately to its left already reads B/B_ref, and
    # two of them stacked side by side read as a rendering fault. The legend carries the curve.
    axb.set(
        xlabel=r"equal-arc $z/\pi$ (origin at the $|B|$ minimum)",
        title="mirror force on the GKX grid",
    )
    axg = axb.twinx()
    force = axg.plot(
        theta / np.pi, centered(bgrad), lw=1.7, color="#d1495b", label="bgrad"
    )
    axg.axhline(0.0, color="0.75", lw=0.8)
    axg.set_ylabel("bgrad")
    axb.legend(handles=field + force, frameon=False, loc="upper right")

    axm = figure.add_subplot(1, 3, 3)
    axm.plot(theta / np.pi, centered(geometry.gds2_profile), label="gds2")
    axm.plot(theta / np.pi, centered(geometry.gds21_profile), label="gds21")
    axm.plot(theta / np.pi, centered(geometry.gds22_profile), label="gds22")
    axm.set(
        xlabel=r"equal-arc $z/\pi$ (origin at the $|B|$ minimum)",
        title="perpendicular metric",
    )
    axm.legend(frameon=False, ncol=3, fontsize=8)
    axm.text(
        0.03,
        0.04,
        rf"GKX: $\gamma={objective[0]:.4f}$, $\omega={objective[1]:.4f}$"
        "\n"
        rf"$Q_{{QL}}={objective[-1]:.4f}$, $B_{{max}}/B_{{min}}={bmag.max() / bmag.min():.3f}$",
        transform=axm.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    figure_path = output_dir / "vmex_mirror_gkx_showcase.webp"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    movie_path = output_dir / "vmex_mirror_gkx_rotation.mp4"
    snapshot_path = output_dir / "vmex_mirror_gkx_snapshot.webp"
    loop_path = output_dir / "vmex_mirror_gkx_loop.webp"
    if movie:
        movie_figure = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
        movie_3d = movie_figure.add_subplot(1, 2, 1, projection="3d")
        movie_profile = movie_figure.add_subplot(1, 2, 2)
        movie_3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.6", lw=1.0)
        movie_3d.scatter(
            line_xyz[:, 0], line_xyz[:, 1], line_xyz[:, 2], c=bmag, s=16, cmap="viridis"
        )
        _style_axis_3d(movie_3d)
        movie_profile.plot(theta / np.pi, centered(bmag), color="#31688e", lw=2.2)
        marker = movie_profile.axvline(theta[0] / np.pi, color="#d1495b", lw=1.5)
        movie_profile.set(
            xlabel=r"equal-arc $z/\pi$ (origin at the $|B|$ minimum)",
            ylabel=r"$B/B_{ref}$",
            title=rf"GKX mirror: $\gamma={objective[0]:.4f}$",
        )

        def frame(index: int):
            movie_3d.view_init(elev=24, azim=-60 + 360 * index / 72)
            marker.set_xdata([theta[index % theta.size] / np.pi] * 2)
            return (marker,)

        frame(0)
        movie_figure.savefig(snapshot_path, dpi=105)
        animation = FuncAnimation(movie_figure, frame, frames=72, interval=50)
        animation.save(movie_path, writer=FFMpegWriter(fps=18, bitrate=700), dpi=120)
        loop_frames = []
        for index in range(0, 72, 2):
            frame(index)
            buffer = BytesIO()
            movie_figure.savefig(buffer, format="png", dpi=80)
            buffer.seek(0)
            loop_frames.append(Image.open(buffer).convert("RGB").copy())
        loop_frames[0].save(
            loop_path,
            save_all=True,
            append_images=loop_frames[1:],
            duration=110,
            loop=0,
            quality=55,
            method=6,
        )
        plt.close(movie_figure)

    root = Path(__file__).resolve().parents[2]
    record = {
        "kind": "vmex_closed_mirror_gkx_showcase",
        "scope": "closed periodic mirror-hybrid; no open-end kinetic claim",
        "gkx_commit": _revision(root),
        "vmex_commit": _revision(
            Path(__import__("vmex").__file__).resolve().parents[1]
        ),
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "precision": "float64",
        "inputs": inputs,
        "objectives": dict(zip(SOLVER_OBJECTIVE_NAMES, map(float, objective))),
        "bmag_max_over_min": float(bmag.max() / bmag.min()),
        "bmag_max_over_min_note": "Field-strength modulation depth along the sampled field line, not a throat-to-midplane mirror ratio. On the solved equilibrium the line carries a maximum at each of the two return bends and a minimum on each straight leg. It does NOT track cross_section_elongation_squared: that identity held only on the seeded state this case used to ship (1.77849 against an elongation squared of 1.77778), and on the solved equilibrium the same inputs give 1.03801. It does move with return_radius. See the closed VMEX mirror geometry section of docs/geometry.rst.",
        "cross_section_elongation_squared": float(
            (inputs["semi_major"] / inputs["semi_minor"]) ** 2
        ),
        "equilibrium_state": "solved fixed-boundary equilibrium from solve_fixed_boundary(..., solve_lambda=True, require_convergence=True); not the seeded initial state",
        "equilibrium": equilibrium,
        "plot_convention": "the plotted parallel profiles are rolled so z=0 is the |B| minimum; the recorded numbers use the unrolled geometry",
        "gradpar": float(geometry.gradpar_value),
        "figure": figure_path.name,
        "snapshot": snapshot_path.name if movie else None,
        "animated_loop": loop_path.name if movie else None,
        "movie": movie_path.name if movie else None,
        "generator": str(Path(__file__).relative_to(root)),
    }
    (output_dir / "vmex_mirror_gkx_showcase.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static"),
        help="artifact directory",
    )
    parser.add_argument("--no-movie", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, movie=not args.no_movie), indent=2))


if __name__ == "__main__":
    main()
