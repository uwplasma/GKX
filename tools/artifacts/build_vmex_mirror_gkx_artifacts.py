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
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    gk_closed_fieldline_geometry,
)
from vmex.mirror.forces import isotropic_force_residual, mirror_energy  # noqa: E402


def _revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


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
        quadrature_order=3,
    )
    state = setup.discretization.evaluate_state(setup.initial_state)
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
    # The case runs on ``setup.initial_state``, the seeded stream function on the prescribed
    # nested-ellipse surfaces, not a solved equilibrium: measure how far that seed sits from
    # force balance rather than leave a reader to assume it is one.
    flux, current, grid = (
        inputs["axial_flux_derivative"],
        inputs["current_derivative"],
        setup.discretization.grid,
    )
    energy = mirror_energy(
        state,
        grid,
        axial_flux_derivative=flux,
        current_derivative=current,
        axis=setup.axis,
    )
    residual = isotropic_force_residual(
        energy,
        grid,
        state=state,
        axial_flux_derivative=flux,
        current_derivative=current,
        axis=setup.axis,
        closed=True,
    )
    return inputs, setup, geometry, mapping, np.asarray(objectives), residual


def _style_axis_3d(axis) -> None:
    axis.set_axis_off()
    axis.set_box_aspect((1.0, 0.45, 1.3))


def build(output_dir: Path, *, movie: bool = True) -> dict:
    inputs, setup, geometry, mapping, objective, residual = _case()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    theta = np.asarray(geometry.theta)
    bmag = np.asarray(geometry.bmag_profile)
    bgrad = np.asarray(geometry.bgrad_profile)
    xyz = np.asarray(setup.axis.centerline)
    # Field-line Cartesian points remain diagnostic metadata owned by VMEX.
    line_xyz = np.asarray(mapping["vmex_mirror"]["xyz"])
    # The equal-arc grid lands z=0 on the high-field leg, which draws this closed circuit as a
    # central barrier. Roll the plotted parallel profiles onto the |B| minimum so it reads as the
    # single well its periodicity implies. Display only: the record uses the unrolled geometry and
    # the objectives are invariant under the shift. docs/geometry.rst carries the measurement.
    # The low leg is a plateau straddling the seam, so take the fundamental's trough, not argmin.
    trough = np.angle(-np.sum(bmag * np.exp(-1j * theta)))
    roll = (
        bmag.size // 2
        - int(np.round((trough + np.pi) / (2.0 * np.pi) * bmag.size)) % bmag.size
    )

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

    axb = figure.add_subplot(1, 3, 2)
    axb.plot(theta / np.pi, centered(bmag), lw=2.2, label=r"$B/B_{ref}$")
    axb.plot(theta / np.pi, centered(bgrad), lw=1.7, label="bgrad")
    axb.axhline(0.0, color="0.75", lw=0.8)
    axb.set(
        xlabel=r"equal-arc $z/\pi$ (origin at the $|B|$ minimum)",
        title="mirror force on the GKX grid",
    )
    axb.legend(frameon=False)

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
        "bmag_max_over_min_note": "Ratio of the two flat legs of the closed racetrack on the sampled field line, not a throat-to-midplane mirror ratio: the modulation is produced by the rotating elliptical cross-section, so it tracks cross_section_elongation_squared and not straight_length or return_radius. See the closed VMEX mirror geometry section of docs/geometry.rst.",
        "cross_section_elongation_squared": float(
            (inputs["semi_major"] / inputs["semi_minor"]) ** 2
        ),
        "equilibrium_state": "seeded initial state from build_stellarator_mirror_hybrid, not a solved fixed-boundary equilibrium",
        "seed_force_residual_normalized_rms": float(
            np.asarray(residual.normalized_rms)
        ),
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
