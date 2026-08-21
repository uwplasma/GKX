"""Side-by-side turbulence movie across devices, for the README hero.

Takes the snapshot files written by ``build_turbulence_movie.py --snapshots``
and renders one frame per time index with every device beside the others, so a
reader sees in a single loop what changes between a tokamak and two very
different stellarators.

Each device keeps its **own** colour scale. Saturated amplitudes differ by more
than an order of magnitude across these cases, and a shared scale would render
the weaker ones as flat grey -- which would say "nothing happens in a
stellarator" rather than "the amplitude is lower". The scale is printed per
panel so the comparison stays quantitative.

Emits an mp4 for the documentation and an optimized GIF for the README, since
GitHub markdown will not play a repository mp4 inline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gkx.artifacts.figure_style import figure_style, save_figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_turbulence_movie import _field_line_tube, _torus_wireframe  # noqa: E402


class _Device:
    """One case loaded from a snapshot file."""

    def __init__(self, path: Path, title: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.phi = data["phi"]
        self.times = data["times"]
        self.title = title
        self.q = float(data["q"])
        self.epsilon = float(data["epsilon"])
        self.R0 = float(data["major_radius"])
        self.nfp = int(data["nfp"])
        # Lock the scale on the saturated tail, not the growth phase, or the
        # movie spends most of its length nearly blank.
        tail = self.phi[len(self.phi) // 2 :]
        # 99th percentile, not the max: a single hot cell sets the max and
        # washes every structure in the tube to pale grey. The reported number
        # below is still the true peak.
        self.scale = max(float(np.percentile(np.abs(tail), 99.0)), 1e-30)
        self.peak = max(float(np.abs(tail).max()), 1e-30)


def render(devices: list[_Device], output: Path, *, fps: int, gif: Path | None) -> int:
    frames = min(len(device.phi) for device in devices)
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index in range(frames):
        with figure_style():
            fig = plt.figure(figsize=(4.4 * len(devices), 3.9))
            for column, device in enumerate(devices):
                phi = np.asarray(device.phi[index], dtype=float)
                nx, ny, nz = phi.shape

                ax = fig.add_subplot(1, len(devices), column + 1, projection="3d")
                samples = max(4 * nz, 160)
                centre, outward, binormal, minor, major = _field_line_tube(
                    device, samples
                )
                source = np.linspace(0.0, 1.0, nz)
                target = np.linspace(0.0, 1.0, samples)
                slab = phi[nx // 2]
                resampled = np.stack(
                    [np.interp(target, source, slab[row]) for row in range(ny)], axis=0
                )

                angle = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
                radius = 0.85 * minor
                surface = (
                    centre[None, :, :]
                    + radius * np.cos(angle)[:, None, None] * outward[None, :, :]
                    + radius * np.sin(angle)[:, None, None] * binormal[None, :, :]
                )
                normed = 0.5 + 0.5 * np.clip(resampled / device.scale, -1.0, 1.0)

                wire = _torus_wireframe(major, minor)
                ax.plot_wireframe(
                    *wire,
                    rstride=6,
                    cstride=3,
                    color="#BBBBBB",
                    linewidth=0.3,
                    alpha=0.45,
                )
                ax.plot_surface(
                    surface[..., 0],
                    surface[..., 1],
                    surface[..., 2],
                    facecolors=plt.cm.RdBu_r(normed),
                    rstride=1,
                    cstride=1,
                    linewidth=0.0,
                    antialiased=False,
                    shade=False,
                )
                # The tube is wide and flat: equal limits on all three axes
                # would spend most of the panel on empty space above and below
                # it. Match the box aspect to the actual extents instead.
                span = (major + minor) * 1.02
                height = max(minor * 2.2, span * 0.28)
                ax.set_xlim(-span, span)
                ax.set_ylim(-span, span)
                ax.set_zlim(-height, height)
                # zoom fills the panel: matplotlib keeps generous internal
                # padding around a 3D axes even with the frame switched off.
                ax.set_box_aspect((1, 1, height / span), zoom=1.42)
                ax.set_axis_off()
                ax.view_init(elev=30, azim=(-60.0 + index * 1.6) % 360.0)
                ax.set_title(
                    f"{device.title}\n$|e\\phi/T_i|_{{\\max}}$ = {device.peak:.3f}",
                    y=0.99,
                    fontsize=11,
                )

            fig.suptitle(
                f"Ion-temperature-gradient turbulence     "
                f"$t\\,c_s/a = {devices[0].times[index]:.0f}$",
                fontsize=13,
            )
            fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.0, wspace=0.0)
            frame = frame_dir / f"frame_{index:04d}.png"
            save_figure(fig, frame)
            written.append(frame)
        if (index + 1) % 20 == 0:
            print(f"rendered {index + 1}/{frames}", flush=True)

    pattern = str(frame_dir / "frame_%04d.png")
    encode = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
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
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")

    if gif is not None:
        # Two-pass palette: a single-pass GIF of a smooth field bands badly.
        palette = frame_dir / "palette.png"
        chain = "fps=12,scale=900:-1:flags=lanczos"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output),
                "-vf",
                f"{chain},palettegen=stats_mode=diff",
                str(palette),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output),
                "-i",
                str(palette),
                "-lavfi",
                f"{chain}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                str(gif),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        palette.unlink()
        print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB)")

    shutil.rmtree(frame_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="TITLE=PATH",
        help="device title and snapshot file, repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    devices = []
    for item in args.snapshot:
        title, _, path = item.partition("=")
        devices.append(_Device(Path(path), title))
    return render(devices, args.output, fps=args.fps, gif=args.gif)


if __name__ == "__main__":
    raise SystemExit(main())
