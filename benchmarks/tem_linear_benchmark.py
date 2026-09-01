import argparse
from pathlib import Path

from gkx.artifacts.plotting import scan_comparison_figure
from gkx.benchmarking_shared import load_tem_reference
from gkx.runtime import run_runtime_scan
from gkx.workflows.runtime.toml import load_runtime_from_toml


ROOT = Path(__file__).resolve().parents[1]
TEM_CONFIG = ROOT / "examples" / "linear" / "axisymmetric" / "runtime_tem.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="TEM linear scan example.")
    parser.parse_args()

    ref = load_tem_reference()
    ky_vals = ref.ky
    cfg, _raw = load_runtime_from_toml(TEM_CONFIG)
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
        x_label=r"$k_y \rho_s$",
        title="TEM (s-alpha)",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig("tem_scan.png", dpi=200)


if __name__ == "__main__":
    main()
