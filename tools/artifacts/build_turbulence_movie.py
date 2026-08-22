"""Render a 3D turbulence movie of the electrostatic potential on a flux tube.

Continues GKX's production nonlinear solver in chunks, capturing ``phi`` after
each chunk, and
draws each frame twice: as the perpendicular cut a gyrokineticist reads, and as
the field-aligned tube in real space that shows why the cut looks the way it
does. The turbulence is elongated along ``B``, and a flux-tube movie that only
ever shows the perpendicular plane hides exactly that.

The tube maps the field-aligned coordinate ``z`` onto imported VMEC ``R``,
``Z``, and toroidal-angle profiles. Analytic decks use a labelled analytic
helix; imported geometry never silently falls back to it.

Pass ``--initial-state`` from ``nonlinear_saturated_state.py`` to film a
verified saturated trajectory. Chunked integration is deliberate: the cache,
parameters, CFL policy, and chunk shape are fixed, so JAX compiles once and
replays it without retaining a full state history.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gkx.artifacts import snapshots  # noqa: E402
from gkx.artifacts.figure_style import figure_style, save_figure  # noqa: E402
from gkx.geometry import (  # noqa: E402
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from gkx.solvers.nonlinear.diagnostic_integration import (  # noqa: E402
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.workflows.runtime.policies import (  # noqa: E402
    _runtime_external_phi,
    _select_nonlinear_mode_indices,
    build_runtime_nonlinear_diagnostics_kwargs,
)
from gkx.workflows.runtime.startup import (  # noqa: E402
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402


def _check_healthy(phi: np.ndarray, where: str, ceiling: float) -> None:
    """Abort on a run that has gone numerically unstable.

    This is a last-resort artifact guard. The integration path itself uses the
    production runtime's CFL policy, so non-finite output indicates a genuine
    numerical failure rather than an expected fixed-step limitation.

    The ceiling is calibrated against a measurement, not an assumption. An
    earlier version asserted that "saturated ITG turbulence is order 0.1-1" and
    aborted above 50, which is wrong for this normalization by two orders of
    magnitude: a state that passes every saturation check in
    ``tools/campaigns/nonlinear_saturated_state.py`` -- tau_ac 8.92, window 22.4
    tau_ac, late drift 1.6% -- has max|phi| = 137.7. That guard fired on correct
    physics and aborted three otherwise healthy runs partway up the linear phase.

    The mismatch had a cause worth stating: the shipped TOML sets
    diagnostic_norm = "rho_star", so this quantity is the rho-star-normalized
    potential. "Order 0.1 to 1" is right for ephi/T_i and wrong for
    (ephi/T_i)/rho_star by a factor 1/rho_star -- the ceiling and the field were
    two different quantities.
    """

    peak = float(np.abs(phi).max()) if phi.size else 0.0
    if not np.isfinite(phi).all():
        raise RuntimeError(
            f"{where}: solution contains non-finite values under the runtime "
            "CFL policy; inspect the deck and solver diagnostics"
        )
    if peak > ceiling:
        raise RuntimeError(
            f"{where}: max|phi| = {peak:.3e} exceeds the sanity ceiling "
            f"{ceiling:.3e}. Raise --ceiling if this case genuinely saturates "
            "higher -- measure it with "
            "tools/campaigns/nonlinear_saturated_state.py rather than guessing. "
            "A verified-saturated Cyclone state has max|phi| = 137.7, so a "
            "ceiling near 50 aborts healthy runs."
        )


#: Spin-up chunk size. Bounds peak memory: the integrator scans over steps and
#: stacks per-step fields, so one long call allocates proportionally to it.
_SPINUP_CHUNK = 100


def potential_real_space(fields, *, ny_full: int) -> np.ndarray:
    """Return ``phi(x, y, z)`` from the production integrator's field state."""

    return snapshots.potential_real_space(np.asarray(fields.phi), ny_full=ny_full)


def _movie_initial_state(
    path: Path | None,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    seed: int,
    amplitude: float,
) -> tuple[Any, float, bool, str]:
    """Restore a campaign state, or build the labelled seed fallback."""

    if path is not None:
        with np.load(path, allow_pickle=False) as archive:
            if "state" not in archive:
                raise ValueError(f"{path} has no 'state' array")
            restored = np.asarray(archive["state"])
            elapsed = float(archive["t_end"]) if "t_end" in archive else 0.0
            saturated = bool(archive["saturated"]) if "saturated" in archive else False
        if restored.shape != shape:
            raise ValueError(
                f"state shape {restored.shape} != movie grid shape {shape}"
            )
        return jnp.asarray(restored, dtype=dtype), elapsed, saturated, str(path)

    generator = np.random.default_rng(seed)
    spectral_seed = np.fft.fft2(generator.normal(size=shape[-3:]), axes=(0, 1))
    spectral_seed *= amplitude / (np.abs(spectral_seed).max() + 1e-30)
    state = (
        jnp.zeros(shape, dtype=dtype)
        .at[:, 0, 0]
        .set(jnp.asarray(spectral_seed, dtype=dtype))
    )
    return state, 0.0, False, "seeded continuation"


def _movie_moment_dims(
    raw: dict[str, Any], laguerre: int | None, hermite: int | None
) -> tuple[int, int]:
    """Resolve movie moments from CLI overrides or the production deck."""

    run = raw.get("run", {})
    nl = int(run.get("Nl", 4) if laguerre is None else laguerre)
    nm = int(run.get("Nm", 8) if hermite is None else hermite)
    return nl, nm


def _require_movie_geometry_profiles(geometry: Any, *, model: str) -> None:
    """Fail closed when an imported-geometry movie lacks physical coordinates."""

    normalized = model.strip().lower().replace("_", "-")
    if normalized in {"s-alpha", "salpha", "analytic", "slab"}:
        return
    names = (
        "cylindrical_R_profile",
        "cylindrical_Z_profile",
        "toroidal_angle_profile",
    )
    profiles = [getattr(geometry, name, None) for name in names]
    missing = [name for name, value in zip(names, profiles) if value is None]
    if missing:
        raise RuntimeError(
            "imported-geometry movies require physical R, Z, and toroidal-angle "
            f"profiles; missing {', '.join(missing)}"
        )
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in profiles]
    if len({array.size for array in arrays}) != 1 or arrays[0].size < 2:
        raise RuntimeError("physical movie coordinate profiles must have equal length >= 2")
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("physical movie coordinate profiles contain non-finite values")


def render_frame(
    phi_xy: np.ndarray,
    phi_yz: np.ndarray,
    geometry,
    *,
    output: Path,
    time: float,
    scale: float,
    label: str,
    extent: tuple[float, float] | None = None,
) -> None:
    """Draw one frame: perpendicular cut plus the field-aligned tube."""

    with figure_style():
        fig = plt.figure(figsize=(11.6, 5.0))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.25), wspace=0.18)

        # ---- perpendicular cut -------------------------------------------
        ax = fig.add_subplot(grid[0, 0])
        snapshots.draw_phi_xy_cut(ax, phi_xy, scale=scale, extent=extent)
        ax.set_title("Perpendicular cut at the outboard midplane")

        # ---- field-aligned tube -------------------------------------------
        ax3d = fig.add_subplot(grid[0, 1], projection="3d")
        snapshots.draw_flux_tube_3d(
            ax3d,
            phi_yz[None, :, :],
            geometry,
            scale=scale,
            azim=(-60.0 + time * 2.0) % 360.0,
        )
        ax3d.set_title(r"Flux tube along $\mathbf{B}$", y=0.94)

        fig.suptitle(f"{label}    $t\\,c_s/a = {time:.1f}$", fontsize=13)
        save_figure(fig, output)


def run(
    config: Path,
    output: Path | None,
    *,
    frames: int,
    steps_per_frame: int,
    dt: float | None,
    fps: int,
    keep_frames: bool,
    seed: int = 0,
    amplitude: float = 1.0e-3,
    laguerre: int | None = None,
    hermite: int | None = None,
    frames_only: bool = False,
    snapshots: Path | None = None,
    initial_state: Path | None = None,
    spinup_steps: int = 0,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    # Measured against a state that passes every saturation check (max|phi| =
    # 137.7 at 16^3), with room for a nonlinear overshoot.
    ceiling: float = 1.0e3,
) -> int:
    cfg, raw = load_runtime_from_toml(config)
    overrides = {
        key: value
        for key, value in (("Nx", nx), ("Ny", ny), ("Nz", nz))
        if value is not None
    }
    if nz is not None and cfg.grid.ntheta is not None:
        zp = cfg.grid.zp
        if zp is None:
            zp = 2 * cfg.grid.nperiod - 1 if cfg.grid.nperiod is not None else 1
        if nz % zp:
            raise ValueError(
                f"--nz={nz} is not divisible by parallel multiplier zp={zp}"
            )
        overrides["ntheta"] = nz // zp
    if overrides:
        cfg = dataclasses.replace(cfg, grid=dataclasses.replace(cfg.grid, **overrides))
        print(f"grid override: {overrides}", flush=True)
    geometry = build_runtime_geometry(cfg)
    params = build_runtime_linear_params(cfg, geom=geometry)
    # Must come from the config. TermConfig() defaults to nonlinear = 0.0 and
    # hyperdiffusion = 0.0, so omitting this runs a LINEAR case with no
    # small-scale dissipation: the ITG mode then grows exponentially forever,
    # which is what the earlier blow-ups actually were. No timestep can bound
    # a linear instability, which is why halving dt only delayed them.
    terms = build_runtime_term_config(cfg)
    if float(terms.nonlinear) == 0.0:
        raise ValueError(
            f"{config} sets [terms] nonlinear = 0.0; there is no turbulence to film"
        )

    from gkx.core.grid import build_spectral_grid
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.workflows.runtime.startup import _resolve_runtime_hl_dims

    # Movie-only grid overrides. The shipped stellarator configs run 96x96x48,
    # which is a production transport resolution and needs more memory than a
    # shared GPU can spare; a visualization does not. Any override is recorded
    # in the snapshot so a frame can never be mistaken for a production run.
    grid = build_spectral_grid(apply_geometry_grid_defaults(geometry, cfg.grid))
    geometry = ensure_flux_tube_geometry_data(geometry, grid.z)
    _require_movie_geometry_profiles(geometry, model=cfg.geometry.model)
    laguerre, hermite = _movie_moment_dims(raw, laguerre, hermite)
    nl, nm = _resolve_runtime_hl_dims(cfg, Nl=laguerre, Nm=hermite)
    cache = build_linear_cache(grid, geometry, params, nl, nm)
    step = float(dt if dt is not None else cfg.time.dt)

    perp = (grid.ky.size, grid.kx.size, grid.z.size)
    complex_dtype = (
        jnp.complex64 if jnp.zeros(1).dtype == jnp.float32 else jnp.complex128
    )
    shape = (len(cfg.species), nl, nm, *perp)
    state, elapsed, saturated_source, source_label = _movie_initial_state(
        initial_state,
        shape=shape,
        dtype=complex_dtype,
        seed=seed,
        amplitude=amplitude,
    )

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=0.3,
        kx_target=0.0,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    fixed_ky = int(cfg.expert.iky_fixed) if cfg.expert.fixed_mode else None
    fixed_kx = int(cfg.expert.ikx_fixed) if cfg.expert.fixed_mode else None
    prepared: dict[int, Any] = {}

    def advance(count: int):
        nonlocal state, elapsed
        simulation = prepared.get(count)
        if simulation is None:
            kwargs = build_runtime_nonlinear_diagnostics_kwargs(
                cfg,
                dt=step,
                steps=count,
                method=None,
                term_config=terms,
                sample_stride=count,
                diagnostics_stride=count,
                laguerre_mode=cfg.time.laguerre_nonlinear_mode,
                ky_index=ky_index,
                kx_index=kx_index,
                fixed_dt=bool(cfg.time.fixed_dt),
                fixed_mode_ky_index=fixed_ky,
                fixed_mode_kx_index=fixed_kx,
                external_phi=_runtime_external_phi(cfg),
                resolved_diagnostics=False,
                show_progress=False,
            )
            kwargs.pop("dt")
            kwargs.pop("steps")
            simulation = prepare_nonlinear_explicit_diagnostics(
                state, grid, geometry, params, step, count, cache=cache, **kwargs
            )
            prepared[count] = simulation
        time_chunk, _diag, state, fields = simulation.run(state)
        elapsed += float(np.asarray(time_chunk)[-1])
        return fields

    if spinup_steps > 0:
        # Advance through the linear growth phase without recording. A movie
        # that spends its length on exponential growth is showing an
        # instability, not turbulence; reaching saturation costs the same GPU
        # time either way, so recording only afterwards spends every frame
        # where the physics is.
        #
        print(f"spin-up: {spinup_steps} production steps", flush=True)
        done = 0
        while done < spinup_steps:
            chunk = min(_SPINUP_CHUNK, spinup_steps - done)
            fields = advance(chunk)
            done += chunk
            if done % (10 * _SPINUP_CHUNK) == 0 or done == spinup_steps:
                probe = potential_real_space(fields, ny_full=int(grid.ky.size))
                _check_healthy(probe, f"spin-up step {done}", ceiling)
                print(
                    f"  spin-up {done}/{spinup_steps}  "
                    f"max|phi| = {float(np.abs(probe).max()):.4e}",
                    flush=True,
                )

    label = config.stem.replace("_", " ")
    extent = (2.0 * np.pi * float(grid.x0), 2.0 * np.pi * float(grid.y0))
    xy_frames: list[np.ndarray] = []
    yz_frames: list[np.ndarray] = []
    times: list[float] = []
    scale = None
    written: list[Path] = []
    frame_dir = None
    if snapshots is None:
        assert output is not None
        frame_dir = output.parent / f"{output.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        fields = advance(steps_per_frame)
        phi = potential_real_space(fields, ny_full=int(grid.ky.size))
        _check_healthy(phi, f"frame {index + 1}", ceiling)
        phi_xy = phi[:, :, phi.shape[2] // 2]
        phi_yz = phi[phi.shape[0] // 2]
        magnitude = max(float(np.abs(phi_xy).max()), float(np.abs(phi_yz).max()))
        times.append(elapsed)
        if snapshots is not None:
            xy_frames.append(phi_xy.astype(np.float32))
            yz_frames.append(phi_yz.astype(np.float32))
        else:
            if scale is None or index < frames // 4:
                scale = max(magnitude, 1e-12)
            assert frame_dir is not None and scale is not None
            frame_path = frame_dir / f"frame_{index:04d}.png"
            render_frame(
                phi_xy,
                phi_yz,
                geometry,
                output=frame_path,
                time=elapsed,
                scale=scale,
                label=label,
                extent=extent,
            )
            written.append(frame_path)
        print(f"frame {index + 1}/{frames}  max|phi| = {magnitude:.4e}", flush=True)

    if snapshots is not None:
        snapshots.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snapshots,
            phi_xy=np.stack(xy_frames),
            phi_yz=np.stack(yz_frames),
            times=np.asarray(times),
            label=label,
            q=float(getattr(geometry, "q", 1.4) or 1.4),
            epsilon=float(getattr(geometry, "epsilon", 0.18) or 0.18),
            major_radius=float(getattr(geometry, "R0", 3.0) or 3.0),
            nfp=int(getattr(geometry, "nfp", 1) or 1),
            cylindrical_R_profile=_geometry_profile(geometry, "cylindrical_R_profile"),
            cylindrical_Z_profile=_geometry_profile(geometry, "cylindrical_Z_profile"),
            toroidal_angle_profile=_geometry_profile(
                geometry, "toroidal_angle_profile"
            ),
            extent=np.asarray(extent),
            snapshot_schema=np.array(3, dtype=np.int32),
            trajectory=np.array("production_runtime_continuation"),
            source_state=np.array(source_label),
            source_saturated=np.array(saturated_source),
            fixed_dt=np.array(bool(cfg.time.fixed_dt)),
            method=np.array(str(cfg.time.method)),
            resolution=np.array(
                [grid.kx.size, grid.ky.size, grid.z.size, nl, nm], dtype=np.int32
            ),
        )
        print(f"wrote {snapshots}")
        return 0

    assert output is not None and frame_dir is not None
    return _encode(frame_dir, written, output, fps, frames_only, keep_frames)


def _geometry_profile(geometry, name: str) -> np.ndarray:
    value = getattr(geometry, name, None)
    return np.empty(0, dtype=float) if value is None else np.asarray(value, dtype=float)


class _SnapshotGeometry:
    """Minimal geometry stand-in reconstructed from a snapshot file."""

    def __init__(
        self,
        q: float,
        epsilon: float,
        major_radius: float,
        nfp: int,
        cylindrical_R_profile: np.ndarray | None = None,
        cylindrical_Z_profile: np.ndarray | None = None,
        toroidal_angle_profile: np.ndarray | None = None,
    ) -> None:
        self.q = q
        self.epsilon = epsilon
        self.R0 = major_radius
        self.nfp = nfp
        self.cylindrical_R_profile = cylindrical_R_profile
        self.cylindrical_Z_profile = cylindrical_Z_profile
        self.toroidal_angle_profile = toroidal_angle_profile


def _snapshot_profile(data, name: str) -> np.ndarray | None:
    if name not in data:
        return None
    values = np.asarray(data[name], dtype=float)
    return values if values.size else None


def render_snapshots(
    snapshots: Path, output: Path, *, fps: int, frames_only: bool, keep_frames: bool
) -> int:
    """Render frames from a snapshot file written by the compute-only pass."""

    with np.load(snapshots, allow_pickle=False) as data:
        if "phi_xy" in data and "phi_yz" in data:
            phi_xy_series = data["phi_xy"]
            phi_yz_series = data["phi_yz"]
        else:
            phi_series = data["phi"]
            phi_xy_series = phi_series[:, :, :, phi_series.shape[3] // 2]
            phi_yz_series = phi_series[:, phi_series.shape[1] // 2, :, :]
        times = data["times"]
        label = str(data["label"])
        extent = tuple(data["extent"]) if "extent" in data else None
        geometry = _SnapshotGeometry(
            float(data["q"]),
            float(data["epsilon"]),
            float(data["major_radius"]),
            int(data["nfp"]),
            _snapshot_profile(data, "cylindrical_R_profile"),
            _snapshot_profile(data, "cylindrical_Z_profile"),
            _snapshot_profile(data, "toroidal_angle_profile"),
        )

    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Lock the colour scale on the first quarter, once, so the movie does not
    # silently renormalize while the amplitude is still growing.
    quarter = max(len(phi_xy_series) // 4, 1)
    scale = max(
        float(np.abs(phi_xy_series[:quarter]).max()),
        float(np.abs(phi_yz_series[:quarter]).max()),
        1e-12,
    )

    written: list[Path] = []
    for index, (phi_xy, phi_yz, time) in enumerate(
        zip(phi_xy_series, phi_yz_series, times)
    ):
        frame_path = frame_dir / f"frame_{index:04d}.png"
        render_frame(
            np.asarray(phi_xy, dtype=float),
            np.asarray(phi_yz, dtype=float),
            geometry,
            output=frame_path,
            time=float(time),
            scale=scale,
            label=label,
            extent=extent,
        )
        written.append(frame_path)
        print(f"rendered {index + 1}/{len(phi_xy_series)}", flush=True)

    return _encode(frame_dir, written, output, fps, frames_only, keep_frames)


def _encode(
    frame_dir: Path,
    written: list[Path],
    output: Path,
    fps: int,
    frames_only: bool,
    keep_frames: bool,
) -> int:
    if frames_only:
        print(f"wrote {len(written)} frames to {frame_dir}")
        return 0

    encode = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-vf",
            "scale=900:-2",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if encode.returncode != 0:
        print(encode.stderr[-2000:], file=sys.stderr)
        return 1
    print(f"wrote {output}")

    if not keep_frames:
        for path in written:
            path.unlink()
        frame_dir.rmdir()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--steps-per-frame", type=int, default=40)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amplitude", type=float, default=1.0e-3)
    parser.add_argument("--laguerre", type=int, default=None)
    parser.add_argument("--hermite", type=int, default=None)
    parser.add_argument(
        "--ceiling",
        type=float,
        default=1.0e3,
        help="abort if max|phi| exceeds this. Calibrated against a state that "
        "passes every check in tools/campaigns/nonlinear_saturated_state.py, "
        "which has max|phi| = 137.7 -- not against an assumed order of magnitude",
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument(
        "--spinup-steps",
        type=int,
        default=0,
        help="integrate this many steps before recording, to reach saturation",
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=None,
        help="state .npz from nonlinear_saturated_state.py; continue it with "
        "the deck's production method and CFL policy",
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=None,
        help="compute only: write lightweight phi cuts to this .npz and exit",
    )
    parser.add_argument(
        "--render-from",
        type=Path,
        default=None,
        help="render a movie from a snapshot .npz; needs no GPU and no config",
    )
    parser.add_argument(
        "--frames-only",
        action="store_true",
        help="render PNG frames and skip ffmpeg encoding",
    )
    args = parser.parse_args()

    if args.snapshots is None and args.output is None:
        parser.error("--output is required unless --snapshots is given")

    if args.render_from is not None:
        return render_snapshots(
            args.render_from,
            args.output,
            fps=args.fps,
            frames_only=args.frames_only,
            keep_frames=args.keep_frames,
        )
    if args.config is None:
        parser.error("config is required unless --render-from is used")

    return run(
        args.config,
        args.output,
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        dt=args.dt,
        fps=args.fps,
        keep_frames=args.keep_frames,
        seed=args.seed,
        amplitude=args.amplitude,
        laguerre=args.laguerre,
        hermite=args.hermite,
        frames_only=args.frames_only,
        snapshots=args.snapshots,
        initial_state=args.initial_state,
        spinup_steps=args.spinup_steps,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        ceiling=args.ceiling,
    )


if __name__ == "__main__":
    raise SystemExit(main())
