"""Structured runtime artifact writers for the executable and benchmark tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectraxgk.runtime import (
    RuntimeNonlinearResult,
    _concat_runtime_diagnostics,
    run_runtime_nonlinear,
)
from spectraxgk.runtime_artifact_diagnostics import (
    validate_finite_runtime_result as _validate_finite_runtime_result,
)
from spectraxgk.runtime_orchestration import (
    RuntimeArtifactHandoffDeps,
    run_runtime_nonlinear_artifact_handoff,
)

from spectraxgk.netcdf_spectral_layout import (
    _complex_to_ri as _complex_to_ri,
    _condense_kx as _condense_kx,
    _condense_kx_for_output as _condense_kx_for_output,
    _condense_ky as _condense_ky,
    _condense_ky_for_output as _condense_ky_for_output,
    _condense_kykx as _condense_kykx,
    _condense_kykx_for_output as _condense_kykx_for_output,
    _dealiased_spectral_field as _dealiased_spectral_field,
    _dealiased_kx_count as _dealiased_kx_count,
    _dealiased_kx_indices as _dealiased_kx_indices,
    _dealiased_kx_values as _dealiased_kx_values,
    _dealiased_ky_count as _dealiased_ky_count,
    _dealiased_ky_indices as _dealiased_ky_indices,
    _dealiased_ky_values as _dealiased_ky_values,
    _maybe_var as _maybe_var,
    _real_space_axis as _real_space_axis,
    _require_netcdf4 as _require_netcdf4,
    _restart_to_netcdf_layout as _restart_to_netcdf_layout,
    _species_matrix as _species_matrix,
    _spectral_species_to_ri as _spectral_species_to_ri,
    _spectral_to_ri as _spectral_to_ri,
    _spectral_to_xy as _spectral_to_xy,
    _state_basis_moments as _state_basis_moments,
    _take_axis as _take_axis,
    _write_runtime_root_metadata as _write_runtime_root_metadata,
)
from spectraxgk.nonlinear_output_netcdf import (
    _build_output_grid_and_geometry as _build_output_grid_and_geometry,
    _particle_moments as _particle_moments,
    _write_geometry_group as _write_geometry_group,
    _write_input_parameters_group as _write_input_parameters_group,
    _write_nonlinear_netcdf_outputs as _write_nonlinear_netcdf_outputs,
)
from spectraxgk.runtime_artifact_nonlinear_diagnostics import (
    _condense_diagnostics_for_netcdf_output as _condense_diagnostics_for_netcdf_output,
    _condense_resolved_for_output as _condense_resolved_for_output,
    _read_optional_var as _read_optional_var,
    _resolved_species_time as _resolved_species_time,
    _resolve_restart_path as _resolve_restart_path,
    load_nonlinear_netcdf_diagnostics as load_nonlinear_netcdf_diagnostics,
)
from spectraxgk.runtime_artifact_nonlinear import (
    _nonlinear_summary as _nonlinear_summary,
    write_runtime_nonlinear_table_artifacts as write_runtime_nonlinear_table_artifacts,
)
from spectraxgk.runtime_artifact_linear import (
    write_quasilinear_artifacts as write_quasilinear_artifacts,
    write_runtime_linear_artifacts as write_runtime_linear_artifacts,
    write_runtime_linear_scan_artifacts as write_runtime_linear_scan_artifacts,
)
from spectraxgk.runtime_artifact_io import (
    _artifact_base as _artifact_base,
    _ensure_parent as _ensure_parent,
    _flatten_series as _flatten_series,
    _netcdf_bundle_base,
    _is_netcdf_output_target,
    _write_csv as _write_csv,
    _write_json as _write_json,
    _write_state as _write_state,
)


def run_runtime_nonlinear_with_artifacts(
    cfg: Any,
    *,
    out: str | Path | None,
    ky_target: float,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    show_progress: bool = False,
    status_callback: Any = None,
) -> tuple[RuntimeNonlinearResult, dict[str, str]]:
    deps = RuntimeArtifactHandoffDeps(
        is_netcdf_output_target=_is_netcdf_output_target,
        resolve_restart_path=lambda path, run_cfg: _resolve_restart_path(
            path, run_cfg, for_write=False
        ),
        resolve_restart_write_path=lambda path, run_cfg: _resolve_restart_path(
            path, run_cfg, for_write=True
        ),
        netcdf_bundle_base=_netcdf_bundle_base,
        load_nonlinear_netcdf_diagnostics=load_nonlinear_netcdf_diagnostics,
        condense_diagnostics_for_netcdf_output=_condense_diagnostics_for_netcdf_output,
        concat_runtime_diagnostics=_concat_runtime_diagnostics,
        validate_finite_runtime_result=lambda result: _validate_finite_runtime_result(
            result, label="nonlinear runtime chunk"
        ),
        run_runtime_nonlinear=run_runtime_nonlinear,
        write_runtime_nonlinear_artifacts=write_runtime_nonlinear_artifacts,
    )
    return run_runtime_nonlinear_artifact_handoff(
        cfg,
        out=out,
        ky_target=ky_target,
        kx_target=kx_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        show_progress=show_progress,
        status_callback=status_callback,
        deps=deps,
    )


def write_runtime_nonlinear_artifacts(
    out: str | Path, result: Any, cfg: Any | None = None
) -> dict[str, str]:
    """Write summary/diagnostics/state artifacts for a nonlinear runtime run."""

    out_path = Path(out)
    if _is_netcdf_output_target(out_path):
        if cfg is None:
            raise ValueError(
                "cfg is required to write nonlinear NetCDF output artifacts"
            )
        return _write_nonlinear_netcdf_outputs(out_path, result, cfg)
    return write_runtime_nonlinear_table_artifacts(out_path, result)
