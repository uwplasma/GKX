"""Run the secondary slab workflow in two stages."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.workflows.nonlinear import (
    build_secondary_stage2_config,
    run_secondary_modes,
    run_secondary_seed,
)


SECONDARY_MODES = (
    (0.0, -0.05),
    (0.0, 0.0),
    (0.0, 0.05),
    (0.1, -0.05),
    (0.1, 0.0),
    (0.1, 0.05),
)


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "runtime_secondary_slab.toml"
    cfg, _data = load_runtime_from_toml(cfg_path)

    with TemporaryDirectory(prefix="gkx_secondary_") as tmpdir:
        restart_path = Path(tmpdir) / "secondary_seed.bin"
        run_secondary_seed(
            cfg,
            restart_path=restart_path,
            ky_target=0.1,
            Nl=3,
            Nm=8,
        )
        stage2_cfg = build_secondary_stage2_config(cfg, restart_file=restart_path, t_max=2.0)
        rows = run_secondary_modes(
            stage2_cfg,
            modes=SECONDARY_MODES,
            Nl=3,
            Nm=8,
            steps=200,
            sample_stride=20,
        )

    print("ky      kx      omega        gamma")
    for row in rows:
        print(f"{row.ky:0.4f} {row.kx:0.4f} {row.omega: .6e} {row.gamma: .6e}")


if __name__ == "__main__":
    main()
