"""Runtime command artifact output policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "timeseries",
    "eigenfunction",
    "state",
    "quasilinear_summary",
    "quasilinear_species",
)
COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS = ("summary", "scan", "quasilinear_spectrum")
COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS = (
    "quasilinear_summary",
    "quasilinear_species",
)
COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "diagnostics",
    "state",
    "out",
    "big",
    "restart",
)


def print_saved_paths(paths: Mapping[str, str], keys: Sequence[str]) -> None:
    """Print saved artifact paths in command-defined display order."""

    for key in keys:
        if key in paths:
            print(f"saved {paths[key]}")


def write_command_outputs(
    out_path: str | Path | None,
    payload: Any | None,
    *,
    writer: Callable[[str | Path, Any], dict[str, str]],
    display_keys: Sequence[str],
) -> dict[str, str]:
    """Write command artifacts when both destination and payload exist."""

    if out_path is None or payload is None:
        return {}
    paths = writer(out_path, payload)
    print_saved_paths(paths, display_keys)
    return paths


def write_linear_runtime_command_outputs(
    *,
    linear_out_path: str | Path | None,
    quasilinear_out_path: str | Path | None,
    result: Any,
    linear_writer: Callable[[str | Path, Any], dict[str, str]],
    quasilinear_writer: Callable[[str | Path, Any], dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Write all optional artifacts produced by one linear runtime command."""

    linear_paths = write_command_outputs(
        linear_out_path,
        result,
        writer=linear_writer,
        display_keys=COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    ql_paths = write_command_outputs(
        quasilinear_out_path,
        getattr(result, "quasilinear", None),
        writer=quasilinear_writer,
        display_keys=COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    return {"linear": linear_paths, "quasilinear": ql_paths}


def write_scan_runtime_command_outputs(
    out_path: str | Path | None,
    scan: Any,
    *,
    writer: Callable[[str | Path, Any], dict[str, str]],
) -> dict[str, str]:
    """Write optional artifacts produced by one linear-scan runtime command."""

    return write_command_outputs(
        out_path,
        scan,
        writer=writer,
        display_keys=COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS,
    )


def print_linear_run_header(
    *,
    label: str,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    solver: str,
    method: str,
    dt: float,
    steps: int,
    grid_shape: tuple[int, int, int],
    show_progress: bool,
    extra: str | None = None,
) -> None:
    """Print the standard executable header for linear initial-value runs."""

    print(f"starting {label}")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"solver={solver} method={method} dt={dt:.6g} steps={steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"progress={'on' if show_progress else 'off'}"
    )
    if extra is not None:
        print(extra)


def print_nonlinear_run_header(
    *,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    method: str,
    dt: float,
    steps: int | None,
    grid_shape: tuple[int, int, int],
    diagnostics: bool,
    show_progress: bool,
) -> None:
    """Print the standard executable header for nonlinear initial-value runs."""

    print("starting runtime nonlinear run")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"method={method} dt={dt:.6g} "
        f"steps={'auto' if steps is None else steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"diagnostics={'on' if diagnostics else 'off'} "
        f"progress={'on' if show_progress else 'off'}"
    )


def print_nonlinear_run_summary(result: Any) -> bool:
    """Print final nonlinear diagnostics and return whether diagnostics exist."""

    diag = result.diagnostics
    if diag is None:
        print("nonlinear run completed")
        return False
    t_values = np.asarray(diag.t)
    t_last = float(t_values[-1]) if t_values.size else 0.0
    print(
        "nonlinear: "
        f"t={t_last:.6g} "
        f"ky_sel={result.ky_selected:.6g} "
        f"kx_sel={result.kx_selected:.6g} "
        f"dt_mean={float(diag.dt_mean):.6g} "
        f"Wg={float(diag.Wg_t[-1]):.6g} "
        f"Wphi={float(diag.Wphi_t[-1]):.6g} "
        f"Wapar={float(diag.Wapar_t[-1]):.6g}"
    )
    return True


def print_nonlinear_command_outputs(paths: Mapping[str, str], *, enabled: bool) -> None:
    """Print nonlinear artifact paths after diagnostics confirm a saved run."""

    if enabled:
        print_saved_paths(paths, COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS)


__all__ = [
    "COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS",
    "print_linear_run_header",
    "print_nonlinear_command_outputs",
    "print_nonlinear_run_header",
    "print_nonlinear_run_summary",
    "print_saved_paths",
    "write_command_outputs",
    "write_linear_runtime_command_outputs",
    "write_scan_runtime_command_outputs",
]
