"""Render a 3D turbulence movie of the electrostatic potential on a flux tube.

Runs GKX's nonlinear solver in chunks, capturing ``phi`` after each chunk, and
draws each frame twice: as the perpendicular cut a gyrokineticist reads, and as
the field-aligned tube in real space that shows why the cut looks the way it
does. The turbulence is elongated along ``B``, and a flux-tube movie that only
ever shows the perpendicular plane hides exactly that.

The tube is drawn by mapping the field-aligned coordinate ``z`` onto the actual
field line: for an axisymmetric equilibrium that is a helix on a torus of
aspect ratio ``R0/a``; for a stellarator the same construction is used with the
device's field periods, so the twist shown is the geometry's own.

Chunked integration is deliberate: the cache and parameters are fixed, so JAX
compiles the chunk once and replays it, which is far cheaper than one long scan
with a snapshot callback inside it.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gkx.artifacts import snapshots  # noqa: E402
from gkx.artifacts.figure_style import figure_style, save_figure  # noqa: E402
from gkx.solvers.nonlinear.state_integration import (  # noqa: E402
    integrate_nonlinear_cached,
)
from gkx.operators.nonlinear.projection import (  # noqa: E402
    _make_nonlinear_state_projector,
)
from gkx.terms.fields import solve_fields  # noqa: E402
from gkx.workflows.runtime.startup import (  # noqa: E402
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402


def _check_healthy(phi: np.ndarray, where: str, ceiling: float) -> None:
    """Abort on a run that has gone numerically unstable.

    ``integrate_nonlinear_cached`` takes a fixed dt and does not enforce the CFL
    condition the adaptive runtime path applies, so a dt that is fine during the
    linear phase can go unstable once the nonlinearity bites. Without this check
    the failure surfaces as a movie of amplified noise, or as NaN written to a
    snapshot file 40 minutes later.

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
            f"{where}: solution contains non-finite values -- the timestep "
            "violates the CFL condition for this grid. Reduce --dt."
            " If the run was merely approaching saturation, raise --ceiling "
            "instead: measure the saturated amplitude rather than assuming it."
        )
    if peak > ceiling:
        raise RuntimeError(
            f"{where}: max|phi| = {peak:.3e} exceeds the sanity ceiling "
            f"{ceiling:.3e}. Raise --ceiling if this case genuinely saturates "
            "higher -- measure it with "
            "tools/campaigns/nonlinear_saturated_state.py rather than guessing. "
            "A verified-saturated Cyclone state has max|phi| = 137.7, so a "
            "ceiling near 50 aborts healthy runs. Reducing --dt is usually the "
            "wrong lever: this case ran stably under the adaptive stepper at "
            "dt = 0.031, larger than fixed steps that appeared to fail."
        )


#: Spin-up chunk size. Bounds peak memory: the integrator scans over steps and
#: stacks per-step fields, so one long call allocates proportionally to it.
_SPINUP_CHUNK = 100


def _species_arrays(cfg) -> dict[str, jnp.ndarray]:
    """Per-species arrays ``solve_fields`` needs, in config order."""

    species = list(cfg.species)
    pull = lambda name, default: jnp.asarray(  # noqa: E731
        [float(getattr(s, name, default)) for s in species]
    )
    charge = pull("charge", 1.0)
    density = pull("density", 1.0)
    temp = pull("temperature", 1.0)
    mass = pull("mass", 1.0)
    return {
        "charge": charge,
        "density": density,
        "temp": temp,
        "mass": mass,
        "tz": temp / charge,
        "vth": jnp.sqrt(2.0 * temp / mass),
    }


def potential_real_space(state, cache, params, cfg) -> np.ndarray:
    """Return ``phi(x, y, z)`` in real space from the spectral state."""

    fields = solve_fields(
        state,
        cache,
        params,
        fapar=jnp.asarray(0.0),
        w_bpar=jnp.asarray(0.0),
        **_species_arrays(cfg),
    )
    # The compressed-real-FFT ky handling lives in gkx.artifacts.snapshots so
    # every renderer applies the same transform the nonlinear bracket uses.
    return snapshots.potential_real_space(np.asarray(fields.phi))


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
    output: Path,
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
    spinup_steps: int = 0,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    # Measured against a state that passes every saturation check (max|phi| =
    # 137.7 at 16^3), with room for the overshoot a fixed-step run shows on the
    # way into saturation. A genuine numerical blow-up leaves this far behind.
    ceiling: float = 1.0e3,
) -> int:
    cfg, _ = load_runtime_from_toml(config)
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
    grid_cfg = cfg.grid
    overrides = {
        key: value
        for key, value in (("Nx", nx), ("Ny", ny), ("Nz", nz))
        if value is not None
    }
    if overrides:
        grid_cfg = dataclasses.replace(grid_cfg, **overrides)
        print(f"grid override: {overrides}", flush=True)
    grid = build_spectral_grid(grid_cfg)
    nl, nm = _resolve_runtime_hl_dims(cfg, Nl=laguerre, Nm=hermite)
    cache = build_linear_cache(grid, geometry, params, nl, nm)
    step = float(dt if dt is not None else cfg.time.dt)

    # A small broadband seed in the density moment. Generated in REAL space and
    # transformed forward, so the spectrum is Hermitian by construction: the
    # solver's projector only enforces ky <-> -ky, and a spectrum-side random
    # seed leaves the ky = 0 row not self-conjugate in kx, which shows up later
    # as a complex "real-space" potential. The movie is about the saturated
    # state, which forgets the seed; the fixed generator only keeps successive
    # renders of the same case comparable.
    generator = np.random.default_rng(seed)
    perp = (grid.ky.size, grid.kx.size, grid.z.size)
    spectral_seed = np.fft.fft2(generator.normal(size=perp), axes=(0, 1))
    spectral_seed *= amplitude / (np.abs(spectral_seed).max() + 1e-30)

    complex_dtype = (
        jnp.complex64 if jnp.zeros(1).dtype == jnp.float32 else jnp.complex128
    )
    shape = (len(cfg.species), nl, nm, *perp)
    state = (
        jnp.zeros(shape, dtype=complex_dtype)
        .at[:, 0, 0]
        .set(jnp.asarray(spectral_seed, dtype=complex_dtype))
    )

    # The production runtime applies this after every step; integrate_nonlinear_cached
    # does not. It is nearly a no-op here because the seed is generated in real
    # space and is already Hermitian, so it was not what fixed the blow-up -- but
    # production applies it and a run that drifts off the constraint should be
    # corrected rather than left to grow.
    project_state = _make_nonlinear_state_projector(
        state,
        ky_vals=np.asarray(grid.ky),
        nx=int(grid.kx.size),
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    )
    state = project_state(state)

    if spinup_steps > 0:
        # Advance through the linear growth phase without recording. A movie
        # that spends its length on exponential growth is showing an
        # instability, not turbulence; reaching saturation costs the same GPU
        # time either way, so recording only afterwards spends every frame
        # where the physics is.
        #
        # Chunked, because integrate_nonlinear_cached scans over steps and
        # stacks the per-step field history: one 12000-step call asked for
        # 9.4 GB and died. Memory is bounded by the chunk, not the spin-up.
        print(
            f"spin-up: {spinup_steps} steps to t = {spinup_steps * step:.1f}",
            flush=True,
        )
        done = 0
        while done < spinup_steps:
            chunk = min(_SPINUP_CHUNK, spinup_steps - done)
            result = integrate_nonlinear_cached(
                state,
                cache,
                params,
                step,
                chunk,
                method="rk4",
                terms=terms,
                return_fields=False,
            )
            state = project_state(result[0] if isinstance(result, tuple) else result)
            done += chunk
            if done % (10 * _SPINUP_CHUNK) == 0 or done == spinup_steps:
                probe = potential_real_space(state, cache, params, cfg)
                _check_healthy(probe, f"spin-up step {done}", ceiling)
                print(
                    f"  spin-up {done}/{spinup_steps}  "
                    f"max|phi| = {float(np.abs(probe).max()):.4e}",
                    flush=True,
                )

    if snapshots is not None:
        # Compute-only pass. Rendering is matplotlib on the CPU and takes far
        # longer than the physics, so holding a GPU allocation through it wastes
        # a shared device -- measured 0% utilization against 12.9 GB reserved.
        # Store only the xy and yz cuts the renderer consumes, then release the
        # GPU. A 96x96x48 frame is 32 times smaller in this representation.
        xy_frames = []
        yz_frames = []
        for index in range(frames):
            result = integrate_nonlinear_cached(
                state,
                cache,
                params,
                step,
                steps_per_frame,
                method="rk4",
                terms=terms,
                return_fields=False,
            )
            state = project_state(result[0] if isinstance(result, tuple) else result)
            phi = potential_real_space(state, cache, params, cfg)
            _check_healthy(phi, f"frame {index + 1}", ceiling)
            phi_xy = phi[:, :, phi.shape[2] // 2]
            phi_yz = phi[phi.shape[0] // 2]
            xy_frames.append(phi_xy.astype(np.float32))
            yz_frames.append(phi_yz.astype(np.float32))
            print(
                f"frame {index + 1}/{frames}  max|phi| = {np.abs(phi).max():.4e}",
                flush=True,
            )
        snapshots.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snapshots,
            phi_xy=np.stack(xy_frames),
            phi_yz=np.stack(yz_frames),
            times=(spinup_steps + np.arange(1, frames + 1) * steps_per_frame) * step,
            label=config.stem.replace("_", " "),
            q=float(getattr(geometry, "q", 1.4) or 1.4),
            epsilon=float(getattr(geometry, "epsilon", 0.18) or 0.18),
            major_radius=float(getattr(geometry, "R0", 3.0) or 3.0),
            nfp=int(getattr(geometry, "nfp", 1) or 1),
            cylindrical_R_profile=_geometry_profile(geometry, "cylindrical_R_profile"),
            cylindrical_Z_profile=_geometry_profile(geometry, "cylindrical_Z_profile"),
            toroidal_angle_profile=_geometry_profile(geometry, "toroidal_angle_profile"),
            extent=np.array(
                [2.0 * np.pi * float(grid.x0), 2.0 * np.pi * float(grid.y0)]
            ),
            snapshot_schema=np.array(2, dtype=np.int32),
            resolution=np.array(
                [grid.kx.size, grid.ky.size, grid.z.size, nl, nm], dtype=np.int32
            ),
        )
        print(f"wrote {snapshots}")
        return 0

    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    label = config.stem.replace("_", " ")
    extent = (2.0 * np.pi * float(grid.x0), 2.0 * np.pi * float(grid.y0))
    scale = None
    written: list[Path] = []
    for index in range(frames):
        result = integrate_nonlinear_cached(
            state,
            cache,
            params,
            step,
            steps_per_frame,
            method="rk4",
            terms=terms,
            return_fields=False,
        )
        state = project_state(result[0] if isinstance(result, tuple) else result)
        phi = potential_real_space(state, cache, params, cfg)
        _check_healthy(phi, f"frame {index + 1}", ceiling)
        phi_xy = phi[:, :, phi.shape[2] // 2]
        phi_yz = phi[phi.shape[0] // 2]

        magnitude = max(float(np.abs(phi_xy).max()), float(np.abs(phi_yz).max()))
        if scale is None or index < frames // 4:
            scale = max(magnitude, 1e-12)

        frame_path = frame_dir / f"frame_{index:04d}.png"
        render_frame(
            phi_xy,
            phi_yz,
            geometry,
            output=frame_path,
            time=(index + 1) * steps_per_frame * step,
            scale=scale,
            label=label,
            extent=extent,
        )
        written.append(frame_path)
        print(f"frame {index + 1}/{frames}  max|phi| = {magnitude:.4e}", flush=True)

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
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
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
        spinup_steps=args.spinup_steps,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        ceiling=args.ceiling,
    )


if __name__ == "__main__":
    raise SystemExit(main())
