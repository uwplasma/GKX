"""Runtime command artifact output policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


def print_nonlinear_command_outputs(paths: Mapping[str, str], *, enabled: bool) -> None:
    """Print nonlinear artifact paths after diagnostics confirm a saved run."""

    if enabled:
        print_saved_paths(paths, COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS)


__all__ = [
    "COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS",
    "COMMAND_SCAN_ARTIFACT_DISPLAY_KEYS",
    "print_nonlinear_command_outputs",
    "print_saved_paths",
    "write_command_outputs",
    "write_linear_runtime_command_outputs",
    "write_scan_runtime_command_outputs",
]
