import argparse
from dataclasses import replace
from pathlib import Path

from gkx.artifacts.plotting import scan_comparison_figure
from gkx.benchmarking.shared import load_cyclone_reference_kinetic
from gkx.runtime import run_runtime_scan
from gkx.workflows.runtime.toml import load_runtime_from_toml


ROOT = Path(__file__).resolve().parents[1]
KINETIC_CONFIG = (
    ROOT / "examples" / "linear" / "axisymmetric" / "runtime_kinetic_electron.toml"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kinetic-electron ITG benchmark example."
    )
    parser.parse_args()

    ref = load_cyclone_reference_kinetic()
    ky_vals = ref.ky
    cfg, _raw = load_runtime_from_toml(KINETIC_CONFIG)
    cfg = replace(cfg, time=replace(cfg.time, t_max=8.0))
    scan = run_runtime_scan(
        cfg,
        ky_vals,
        Nl=12,
        Nm=32,
        solver="auto",
    )
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        x_label=r"$k_y \rho_i$",
        title="Kinetic-electron ITG",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig("kinetic_scan.png", dpi=200)


if __name__ == "__main__":
    main()
