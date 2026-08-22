from __future__ import annotations

from dataclasses import fields
import math
from pathlib import Path
import re
import subprocess
from gkx.utils import tomlcompat as tomllib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from support.paths import REPO_ROOT, load_release_tool
from benchmarks import cyclone_linear_benchmark
from benchmarks.performance.benchmark_runtime_memory import (
    RuntimeBenchRun,
    _load_manifest,
    _load_summary_rows,
    _parse_profile_times,
    _parse_peak_rss_mb,
    _plot_results,
    _render,
    _run_command,
    _select_runs,
    _summary_row,
    _write_row_logs,
    _write_summary,
)
from gkx.core.velocity import J_l_all, single_precision_factorial
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear import dissipation as linear_dissipation_module
from gkx.terms import linear_terms as linear_terms_module
from gkx.terms.linear_terms import (
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
    linked_streaming_contribution,
    streaming_contribution,
)
from gkx.benchmarking.shared import (
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    TEM_OMEGA_D_SCALE,
    TEM_OMEGA_STAR_SCALE,
    TEM_RHO_STAR,
    ScanFitWindowPolicy,
    _build_initial_condition as build_benchmark_initial_condition,
    _two_species_params,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.runtime import (
    _build_initial_condition as build_runtime_initial_condition,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml


ROOT = REPO_ROOT
MANIFEST = ROOT / "benchmarks" / "results" / "manifest.toml"
MAX_TRACKED_RESULT_BYTES = 1_000_000
MAX_ROOT_BENCHMARK_PAYLOAD_BYTES = 200_000


def test_benchmark_readme_references_existing_python_drivers() -> None:
    """Keep researcher-facing reproduction commands synchronized with drivers."""

    readme = (ROOT / "benchmarks" / "README.md").read_text(encoding="utf-8")
    drivers = re.findall(r"python (benchmarks/[^\s]+\.py)", readme)

    assert drivers
    assert all((ROOT / driver).is_file() for driver in drivers)


def test_cyclone_publication_driver_uses_asymptotic_fit_window() -> None:
    """Exclude the measured startup transient from the Cyclone growth fit."""

    runtime_cfg, _ = load_runtime_from_toml(cyclone_linear_benchmark.CONFIG)
    assert cyclone_linear_benchmark.FIT_TMIN >= 0.7 * runtime_cfg.time.t_max
    assert cyclone_linear_benchmark.FIT_TMAX == pytest.approx(runtime_cfg.time.t_max)


def test_benchmark_public_exports_resolve() -> None:
    import gkx.benchmarking.shared as benchmark_api

    for name in benchmark_api.__all__:
        assert hasattr(benchmark_api, name), name


def test_runtime_tem_case_matches_transitional_operator_contract() -> None:
    """The canonical runtime case must preserve the established TEM operator."""

    runtime_cfg, _raw = load_runtime_from_toml(
        ROOT / "examples" / "linear" / "axisymmetric" / "runtime_tem.toml"
    )
    legacy_model = SimpleNamespace(
        tprim_i=20.0,
        tprim_e=20.0,
        fprim=20.0,
        Te_over_Ti=1.0,
        mass_ratio=370.0,
        nu_i=0.0,
        nu_e=0.0,
        beta=1.0e-4,
    )
    n_laguerre, n_hermite = 2, 4

    geometry = build_runtime_geometry(runtime_cfg)
    grid_full = build_spectral_grid(runtime_cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_index)

    runtime_params = build_runtime_linear_params(
        runtime_cfg,
        Nm=n_hermite,
        geom=geometry,
    )
    legacy_params = _two_species_params(
        legacy_model,
        kpar_scale=float(geometry.gradpar()),
        omega_d_scale=TEM_OMEGA_D_SCALE,
        omega_star_scale=TEM_OMEGA_STAR_SCALE,
        rho_star=TEM_RHO_STAR,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
        nhermite=n_hermite,
    )
    for field in fields(runtime_params):
        np.testing.assert_allclose(
            np.asarray(getattr(runtime_params, field.name)),
            np.asarray(getattr(legacy_params, field.name)),
            rtol=1.0e-7,
            atol=1.0e-9,
            err_msg=field.name,
        )

    runtime_state = build_runtime_initial_condition(
        grid,
        geometry,
        runtime_cfg,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        nspecies=2,
    )
    legacy_single = build_benchmark_initial_condition(
        grid,
        geometry,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=runtime_cfg.init,
    )
    legacy_state = np.zeros_like(np.asarray(runtime_state))
    legacy_state[1] = np.asarray(legacy_single)
    np.testing.assert_allclose(runtime_state, legacy_state, rtol=0.0, atol=1.0e-19)

    cache = build_linear_cache(
        grid,
        geometry,
        runtime_params,
        n_laguerre,
        n_hermite,
    )
    runtime_rhs, _ = linear_rhs_cached(
        runtime_state,
        cache,
        runtime_params,
        terms=build_runtime_linear_terms(runtime_cfg),
    )
    legacy_rhs, _ = linear_rhs_cached(
        runtime_state,
        cache,
        legacy_params,
        terms=LinearTerms(bpar=0.0),
    )
    np.testing.assert_allclose(runtime_rhs, legacy_rhs, rtol=1.0e-6, atol=1.0e-18)


def test_runtime_kinetic_case_matches_transitional_operator_contract() -> None:
    """The canonical kinetic-electron case preserves the executed operator."""

    runtime_cfg, _raw = load_runtime_from_toml(
        ROOT
        / "examples"
        / "linear"
        / "axisymmetric"
        / "runtime_kinetic_electron.toml"
    )
    model = SimpleNamespace(
        tprim_i=2.49,
        tprim_e=2.49,
        fprim=0.8,
        Te_over_Ti=1.0,
        mass_ratio=1.0 / 0.00027,
        nu_i=0.0,
        nu_e=0.0,
        beta=1.0e-5,
    )
    n_laguerre, n_hermite = 2, 4
    geometry = build_runtime_geometry(runtime_cfg)
    grid_full = build_spectral_grid(runtime_cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_index)

    runtime_params = build_runtime_linear_params(
        runtime_cfg, Nm=n_hermite, geom=geometry
    )
    legacy_params = _two_species_params(
        model,
        kpar_scale=float(geometry.gradpar()),
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.125,
        nhermite=n_hermite,
    )
    for field in fields(runtime_params):
        np.testing.assert_allclose(
            np.asarray(getattr(runtime_params, field.name)),
            np.asarray(getattr(legacy_params, field.name)),
            rtol=1.0e-7,
            atol=1.0e-9,
            err_msg=field.name,
        )

    runtime_state = build_runtime_initial_condition(
        grid,
        geometry,
        runtime_cfg,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        nspecies=2,
    )
    legacy_single = build_benchmark_initial_condition(
        grid,
        geometry,
        ky_index=0,
        kx_index=0,
        Nl=n_laguerre,
        Nm=n_hermite,
        init_cfg=runtime_cfg.init,
    )
    legacy_state = np.zeros_like(np.asarray(runtime_state))
    legacy_state[1] = np.asarray(legacy_single)
    phase_scale = np.vdot(legacy_state, runtime_state) / np.vdot(
        legacy_state, legacy_state
    )
    np.testing.assert_allclose(
        runtime_state,
        phase_scale * legacy_state,
        rtol=1.0e-7,
        atol=1.0e-12,
    )

    cache = build_linear_cache(
        grid, geometry, runtime_params, n_laguerre, n_hermite
    )
    runtime_rhs, _ = linear_rhs_cached(
        runtime_state,
        cache,
        runtime_params,
        terms=build_runtime_linear_terms(runtime_cfg),
    )
    legacy_rhs, _ = linear_rhs_cached(
        legacy_state,
        cache,
        legacy_params,
        terms=LinearTerms(bpar=0.0),
    )
    rhs_error = np.linalg.norm(
        np.asarray(runtime_rhs) - phase_scale * np.asarray(legacy_rhs)
    ) / np.linalg.norm(np.asarray(runtime_rhs))
    assert rhs_error < 2.0e-6


def test_runtime_kbm_case_matches_transitional_operator_contract() -> None:
    """The canonical runtime case preserves the established KBM operator."""

    runtime_cfg, _raw = load_runtime_from_toml(
        ROOT / "examples" / "linear" / "axisymmetric" / "runtime_kbm.toml"
    )
    model = SimpleNamespace(
        tprim_i=2.49, tprim_e=2.49, fprim=0.8,
        Te_over_Ti=1.0, mass_ratio=1.0 / 0.00027,
        nu_i=0.0, nu_e=0.0, beta=runtime_cfg.physics.beta,
    )
    n_laguerre, n_hermite = 2, 4
    geometry = build_runtime_geometry(runtime_cfg)
    grid_full = build_spectral_grid(runtime_cfg.grid)
    grid = select_ky_grid(
        grid_full, int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    )
    runtime_params = build_runtime_linear_params(
        runtime_cfg, Nm=n_hermite, geom=geometry
    )
    legacy_params = _two_species_params(
        model, kpar_scale=float(geometry.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        damp_ends_amp=0.1, damp_ends_widthfrac=0.125,
        nhermite=n_hermite,
    )
    for field in fields(runtime_params):
        np.testing.assert_allclose(
            np.asarray(getattr(runtime_params, field.name)),
            np.asarray(getattr(legacy_params, field.name)),
            rtol=1.0e-7, atol=1.0e-9, err_msg=field.name,
        )
    assert build_runtime_linear_terms(runtime_cfg).hypercollisions == 1.0

    runtime_state = build_runtime_initial_condition(
        grid, geometry, runtime_cfg, ky_index=0, kx_index=0,
        Nl=n_laguerre, Nm=n_hermite, nspecies=2,
    )
    legacy_single = build_benchmark_initial_condition(
        grid, geometry, ky_index=0, kx_index=0,
        Nl=n_laguerre, Nm=n_hermite, init_cfg=runtime_cfg.init,
    )
    legacy_state = np.zeros_like(np.asarray(runtime_state))
    legacy_state[1] = np.asarray(legacy_single)
    np.testing.assert_allclose(runtime_state, legacy_state, rtol=2.0e-7, atol=1.0e-17)

    cache = build_linear_cache(grid, geometry, runtime_params, n_laguerre, n_hermite)
    runtime_rhs, _ = linear_rhs_cached(
        runtime_state, cache, runtime_params,
        terms=build_runtime_linear_terms(runtime_cfg),
    )
    legacy_rhs, _ = linear_rhs_cached(
        legacy_state, cache, legacy_params, terms=LinearTerms(bpar=0.0),
    )
    np.testing.assert_allclose(runtime_rhs, legacy_rhs, rtol=1.0e-6, atol=2.0e-16)


def test_scan_policy_normalizes_keys_and_auto_fit_side_effects() -> None:
    assert normalize_solver_key(" Auto ") == "auto"
    assert normalize_solver_key(" gx_time ") == "gx_time"
    assert normalize_fit_signal(" Density ") == "density"
    assert apply_auto_fit_scan_policy("auto", streaming_fit=True, mode_only=True) == (
        False,
        False,
    )
    assert apply_auto_fit_scan_policy("phi", streaming_fit=True, mode_only=True) == (
        True,
        True,
    )

    with pytest.raises(ValueError, match="fit_signal"):
        normalize_fit_signal("bad")


def test_scan_policy_indexes_values_and_coerces_mode_method() -> None:
    assert indexed_float_value(None, 0) is None
    assert indexed_float_value([1, 2], 1) == pytest.approx(2.0)
    assert indexed_float_value(np.array([0.1, 0.2]), 0) == pytest.approx(0.1)
    assert indexed_scan_value(("a", "b"), 1) == "b"
    assert indexed_scan_value(np.array([3, 4]), 0) == 3
    assert indexed_scan_value(5, 9) == 5

    assert resolve_scan_mode_method("project", mode_only=True) == "z_index"
    assert resolve_scan_mode_method("max", mode_only=True) == "max"
    assert resolve_scan_mode_method("project", mode_only=False) == "project"


def test_scan_policy_window_and_batch_eligibility() -> None:
    t = np.linspace(0.0, 1.0, 5)
    assert scan_window_valid(t, 0.25, 0.75)
    assert not scan_window_valid(t, None, 0.75)
    assert not scan_window_valid(t, 0.2, 0.3)

    assert should_use_ky_batch(
        ky_batch=4, solver_key="time", dt=0.1, steps=10, tmin=None, tmax=None
    )
    assert not should_use_ky_batch(
        ky_batch=4,
        solver_key="krylov",
        dt=0.1,
        steps=10,
        tmin=None,
        tmax=None,
    )
    assert not should_use_ky_batch(
        ky_batch=4,
        solver_key="time",
        dt=np.array([0.1, 0.2]),
        steps=10,
        tmin=None,
        tmax=None,
    )
    with pytest.raises(ValueError, match="ky_batch"):
        should_use_ky_batch(
            ky_batch=0, solver_key="time", dt=0.1, steps=10, tmin=None, tmax=None
        )


def test_scan_fit_window_policy_recovers_synthetic_mode() -> None:
    dt = 0.1
    t = np.arange(120) * dt
    signal = np.exp((0.12 - 0.31j) * t)
    policy = ScanFitWindowPolicy(
        tmin=[2.0],
        tmax=[9.0],
        auto_window=False,
        min_points=10,
    )

    gamma, omega = policy.fit_signal(
        signal,
        idx=0,
        dt=dt,
        stride=1,
        params=LinearParams(),
        diagnostic_norm="none",
    )

    assert gamma == pytest.approx(0.12, rel=1e-3, abs=1e-3)
    assert omega == pytest.approx(0.31, rel=1e-3, abs=1e-3)


def test_scan_fit_window_policy_falls_back_to_auto_when_window_is_invalid() -> None:
    dt = 0.1
    t = np.arange(80) * dt
    signal = np.exp((0.08 - 0.2j) * t)
    policy = ScanFitWindowPolicy(
        tmin=100.0,
        tmax=101.0,
        auto_window=False,
        window_fraction=0.5,
        min_points=10,
        require_positive=True,
    )

    gamma, omega = policy.fit_signal(
        signal,
        idx=0,
        dt=dt,
        stride=1,
        params=LinearParams(),
        diagnostic_norm="none",
    )

    assert gamma == pytest.approx(0.08, rel=1e-3, abs=1e-3)
    assert omega == pytest.approx(0.2, rel=1e-3, abs=1e-3)


def _load_results_manifest() -> dict:
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def test_benchmark_results_manifest_is_root_level_and_small() -> None:
    assert MANIFEST.exists()
    assert MANIFEST.relative_to(ROOT).parts[:2] == ("benchmarks", "results")
    assert MANIFEST.stat().st_size < 20_000

    readme = MANIFEST.parent / "README.md"
    assert readme.exists()
    assert readme.stat().st_size < 20_000


def test_benchmark_results_manifest_points_to_tracked_docs_outputs() -> None:
    manifest = _load_results_manifest()
    rendered_suffixes = load_release_tool(
        "check_validation_coverage_manifest"
    ).RENDERED_ARTIFACT_SUFFIXES
    entries = [*manifest.get("figure", []), *manifest.get("table", [])]
    assert {entry["name"] for entry in entries} >= {
        "Core linear benchmark atlas",
        "Core nonlinear benchmark atlas",
        "Runtime and memory comparison",
        "Runtime and memory result rows",
    }

    for entry in entries:
        path = ROOT / entry["path"]
        rendered = path.suffix.lower() in rendered_suffixes
        assert ROOT / "tools_out" not in path.parents
        assert ROOT / "docs" / "_build" not in path.parents
        if path.exists():
            assert path.is_file(), entry["path"]
            assert path.stat().st_size <= MAX_TRACKED_RESULT_BYTES, entry["path"]
        else:
            assert rendered, entry["path"]

        source_manifest = ROOT / entry["source_manifest"]
        assert source_manifest.exists(), entry["source_manifest"]
        assert source_manifest.suffix == ".toml"
        assert entry["regenerate"].startswith("python ")
        assert entry["docs_page"].endswith((".rst", ".md"))
        assert entry["claim_scope"].strip()


def test_benchmark_results_manifest_documents_artifact_hygiene_policy() -> None:
    policy = _load_results_manifest()["policy"]
    assert policy["tracked_payload"] == "small pointers only"
    assert "tools_out" in policy["raw_outputs"]
    assert "docs/_static" in policy["docs_payload"]


def test_root_benchmark_manifest_is_reflected_in_docs() -> None:
    manifest = _load_results_manifest()
    docs_text = (ROOT / "docs" / "benchmarks.rst").read_text(encoding="utf-8")
    entries = [*manifest.get("figure", []), *manifest.get("table", [])]

    for entry in entries:
        assert entry["name"] in docs_text
        assert entry["path"] in docs_text
        assert entry["claim_scope"] in docs_text


def test_root_benchmark_payload_stays_lightweight() -> None:
    tracked_benchmark_files = [
        ROOT / path
        for path in subprocess.check_output(
            ["git", "ls-files", "benchmarks"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    ]
    assert tracked_benchmark_files
    total_bytes = sum(path.stat().st_size for path in tracked_benchmark_files)
    assert total_bytes <= MAX_ROOT_BENCHMARK_PAYLOAD_BYTES


def test_runtime_memory_manifest_loads_runs() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    assert any(
        run.case == "cyclone-linear" and run.backend == "gkx_cpu" for run in runs
    )
    assert any(run.backend == "gx" for run in runs)


def test_runtime_memory_selection_filters_case_and_backend(tmp_path: Path) -> None:
    manifest = tmp_path / "mini.toml"
    manifest.write_text(
        """
[[run]]
case = "a"
label = "A"
backend = "gkx_cpu"
command = "echo a"

[[run]]
case = "b"
label = "B"
backend = "gx"
command = "echo b"
profile_command = "echo profile"
host = "office"
enabled = false
""",
        encoding="utf-8",
    )
    runs = _load_manifest(manifest)
    assert runs[1].host == "office"
    assert runs[1].profile_command == "echo profile"
    selected = _select_runs(runs, {"a"}, {"gkx_cpu"})
    assert len(selected) == 1
    assert selected[0].case == "a"
    assert selected[0].backend == "gkx_cpu"


def test_parse_peak_rss_mb_supports_macos_and_linux_formats() -> None:
    assert _parse_peak_rss_mb("peak memory footprint: 1048576") == 1.0
    assert _parse_peak_rss_mb("Maximum resident set size (kbytes): 2048") == 2.0


def test_parse_profile_times_extracts_warmup_and_run_fields() -> None:
    parsed = _parse_profile_times("warmup_time_s=30.776 run_time_s=14.081")
    assert parsed == {"warmup_time_s": 30.776, "run_time_s": 14.081}


def test_load_summary_rows_merges_matching_json_files(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    first.write_text(
        '{"rows":[{"case":"a","backend":"gkx_cpu","status":"success"}]}\n',
        encoding="utf-8",
    )
    second = tmp_path / "b.json"
    second.write_text(
        '{"rows":[{"case":"a","backend":"gx","status":"success"}]}\n', encoding="utf-8"
    )
    rows = _load_summary_rows([str(tmp_path / "*.json")])
    assert len(rows) == 2
    assert {row["backend"] for row in rows} == {"gkx_cpu", "gx"}


def test_render_expands_root_and_env(monkeypatch) -> None:
    monkeypatch.setenv("GKX_BENCH_ROOT", "/tmp/bench")
    rendered = _render("{root}:${GKX_BENCH_ROOT}")
    assert str(ROOT) in rendered
    assert "/tmp/bench" in rendered


def test_gx_runtime_memory_manifest_runs_in_isolated_tempdir() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    gx_runs = [run for run in runs if run.backend == "gx"]
    assert gx_runs
    for run in gx_runs:
        assert "mktemp -d" in run.command
        assert "env " in run.command
        assert "-u DISPLAY" in run.command
        assert "HDF5_DISABLE_VERSION_CHECK=1" in run.command
        assert "CUDA_VISIBLE_DEVICES=${GKX_BENCH_CUDA_DEVICE}" in run.command


def test_gx_stellarator_runtime_manifest_uses_pregenerated_nc_geometry() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    stellarator = [
        run
        for run in runs
        if run.backend == "gx"
        and run.case in {"w7x-linear", "w7x-nonlinear", "hsx-linear", "hsx-nonlinear"}
    ]
    assert len(stellarator) == 4
    for run in stellarator:
        assert 'geo_option = "nc"' in run.command
        assert "vmec_file" in run.command
        assert 'geo_file = "' in run.command
        assert "REFERENCE_GK_NETCDF_LIBDIR" in run.command
        assert "REFERENCE_GK_PYTHON_BIN" in run.command


def test_gpu_runtime_memory_manifest_pins_configured_cuda_device() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    gpu_runs = [run for run in runs if run.backend == "gkx_gpu"]
    assert gpu_runs
    for run in gpu_runs:
        assert "CUDA_VISIBLE_DEVICES=${GKX_BENCH_CUDA_DEVICE}" in run.command


def test_short_nonlinear_gpu_rows_request_warm_profile_pass() -> None:
    runs = _load_manifest(ROOT / "tools" / "runtime_memory_manifest.toml")
    selected = {
        (run.case, run.backend): run.profile_command
        for run in runs
        if run.backend == "gkx_gpu"
        and run.case in {"cyclone-nonlinear", "kbm-nonlinear"}
    }
    assert "profile_runtime_kernels.py cyclone" in str(
        selected[("cyclone-nonlinear", "gkx_gpu")]
    )
    assert "profile_runtime_kernels.py cyclone" in str(
        selected[("kbm-nonlinear", "gkx_gpu")]
    )


def test_remote_runtime_memory_runs_disable_x11_forwarding(monkeypatch) -> None:
    captured = {}

    def fake_run(cmd, capture_output, text):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd

        class Proc:
            returncode = 0
            stdout = ""
            stderr = ""

        return Proc()

    monkeypatch.setattr(
        "benchmarks.performance.benchmark_runtime_memory.subprocess.run", fake_run
    )
    run = RuntimeBenchRun(
        case="c", label="C", backend="gx", command="echo hi", cwd="/tmp", host="office"
    )
    row = _run_command(run)
    assert row["status"] == "success"
    assert captured["cmd"][:2] == ["ssh", "-x"]


def test_runtime_memory_command_captures_profile_times(monkeypatch) -> None:
    def fake_run(cmd, shell, cwd, capture_output, text):  # type: ignore[no-untyped-def]
        class Proc:
            returncode = 0
            stdout = "warmup_time_s=12.5 run_time_s=3.25\n"
            stderr = "Maximum resident set size (kbytes): 2048\n"

        return Proc()

    monkeypatch.setattr(
        "benchmarks.performance.benchmark_runtime_memory.subprocess.run", fake_run
    )
    run = RuntimeBenchRun(
        case="c",
        label="C",
        backend="gkx_cpu",
        command="echo hi",
        cwd="/tmp",
        wrap_time=False,
    )
    row = _run_command(run)
    assert row["runtime_s"] >= 0.0
    assert row["peak_rss_mb"] == 2.0
    assert row["warmup_time_s"] == 12.5
    assert row["run_time_s"] == 3.25


def test_runtime_memory_command_runs_profile_subcommand(monkeypatch) -> None:
    calls = []

    def fake_run(cmd, shell, cwd, capture_output, text):  # type: ignore[no-untyped-def]
        calls.append(cmd)

        class Proc:
            returncode = 0
            stdout = "main\n" if len(calls) == 1 else "warmup_time_s=20 run_time_s=7\n"
            stderr = (
                "Maximum resident set size (kbytes): 2048\n" if len(calls) == 1 else ""
            )

        return Proc()

    monkeypatch.setattr(
        "benchmarks.performance.benchmark_runtime_memory.subprocess.run", fake_run
    )
    run = RuntimeBenchRun(
        case="c",
        label="C",
        backend="gkx_gpu",
        command="echo cold",
        profile_command="echo warm",
        cwd="/tmp",
        wrap_time=False,
    )
    row = _run_command(run)
    assert len(calls) == 2
    assert row["status"] == "success"
    assert row["warmup_time_s"] == 20.0
    assert row["run_time_s"] == 7.0
    assert "--- profile stdout ---" in row["stdout"]


def test_runtime_memory_row_logs_are_written(tmp_path: Path) -> None:
    row = {
        "case": "cyclone-linear",
        "backend": "gx",
        "stdout": "ok",
        "stderr": "warn",
    }
    logs = _write_row_logs(tmp_path, row)
    assert Path(logs["stdout_log"]).read_text(encoding="utf-8") == "ok"
    assert Path(logs["stderr_log"]).read_text(encoding="utf-8") == "warn"


def test_runtime_memory_summary_is_written(tmp_path: Path) -> None:
    rows = [
        {
            "case": "a",
            "backend": "gkx_cpu",
            "status": "success",
            "stdout": "long runtime log",
            "stderr": "warning log",
        }
    ]
    out = tmp_path / "summary.json"
    _write_summary(out, rows)
    text = out.read_text(encoding="utf-8")
    assert '"case": "a"' in text
    assert "long runtime log" not in text
    assert "warning log" not in text
    assert '"stdout_bytes": 16' in text
    assert '"stderr_bytes": 11' in text
    assert '"stdout_sha256"' in text


def test_runtime_memory_summary_row_prunes_existing_logs() -> None:
    row = {
        "case": "a",
        "backend": "gkx_cpu",
        "stdout": "ok",
        "stderr": "",
    }
    summary = _summary_row(row)
    assert "stdout" not in summary
    assert "stderr" not in summary
    assert summary["stdout_bytes"] == 2
    assert summary["stderr_bytes"] == 0
    assert summary["stdout_sha256"]
    assert summary["stderr_sha256"] == ""


def test_runtime_memory_plot_supports_warm_runtime_markers(tmp_path: Path) -> None:
    csv_path = tmp_path / "runtime.csv"
    csv_path.write_text(
        "\n".join(
            [
                "case,label,backend,status,returncode,runtime_s,warmup_time_s,run_time_s,peak_rss_mb,host,cwd,command,stdout_log,stderr_log",
                "cyclone-nonlinear,Cyclone ITG Nonlinear,gkx_gpu,success,0,35.3,33.2,14.4,1878.4,office,/tmp,cmd,out,err",
                "cyclone-nonlinear,Cyclone ITG Nonlinear,gx,success,0,21.1,,,1900.0,office,/tmp,cmd,out,err",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    png_path = tmp_path / "runtime.png"
    pdf_path = tmp_path / "runtime.pdf"

    _plot_results(csv_path, png_path, pdf_path)

    assert png_path.exists()
    assert pdf_path.exists()


def _gx_jflr(ell: int, b: jnp.ndarray) -> jnp.ndarray:
    """GX Jflr: exp(-b/2) * (-b/2)^ell / ell!."""

    return (
        jnp.exp(-0.5 * b)
        * ((-0.5 * b) ** ell)
        / jnp.asarray(math.factorial(ell), dtype=b.dtype)
    )


def test_gyroaverage_matches_gx_jflr():
    b = jnp.asarray([0.0, 0.3, 1.0], dtype=jnp.float32)
    Jl = J_l_all(b, l_max=4)
    for ell in range(5):
        expected = _gx_jflr(ell, b)
        assert jnp.allclose(Jl[ell], expected, rtol=1.0e-6, atol=1.0e-7)


def test_single_precision_factorial_matches_stirling_branch():
    m = jnp.asarray([7.0, 8.0, 12.0], dtype=jnp.float32)
    expected = jnp.asarray(
        [
            math.sqrt(2.0 * math.pi * x)
            * (x**x)
            * math.exp(-x)
            * (1.0 + 1.0 / (12.0 * x) + 1.0 / (288.0 * x * x))
            for x in (7.0, 8.0, 12.0)
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(
        single_precision_factorial(m), expected, rtol=1.0e-7, atol=1.0e-7
    )


def test_gyroaverage_matches_gx_jflr_stirling_branch():
    b = jnp.asarray([0.3, 1.0, 2.5], dtype=jnp.float32)
    ell = 7
    expected = (
        jnp.exp(-0.5 * b)
        * ((-0.5 * b) ** ell)
        / single_precision_factorial(jnp.asarray(float(ell), dtype=b.dtype))
    )
    Jl = J_l_all(b, l_max=ell)
    assert jnp.allclose(Jl[ell], expected, rtol=1.0e-7, atol=1.0e-8)


def test_salpha_geometry_matches_gx_formulas():
    geom = SAlphaGeometry(
        q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0, drift_scale=1.0
    )
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)

    bmag = 1.0 / (1.0 + geom.epsilon * jnp.cos(theta))
    bgrad = geom.gradpar() * geom.epsilon * jnp.sin(theta) * bmag
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat * geom.s_hat, dtype=jnp.float32)
    cv = (jnp.cos(theta) + shear * jnp.sin(theta)) / geom.R0
    gb = cv
    cv0 = (-geom.s_hat * jnp.sin(theta)) / geom.R0
    gb0 = cv0

    bmag_gx = geom.bmag(theta)
    bgrad_gx = geom.bgrad(theta)
    gds2_gx, gds21_gx, gds22_gx = geom.metric_coeffs(theta)
    cv_d, gb_d = geom.drift_components(jnp.asarray([0.1]), jnp.asarray([0.2]), theta)
    cv_d = cv_d[0, 0]
    gb_d = gb_d[0, 0]

    assert jnp.allclose(bmag_gx, bmag, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(bgrad_gx, bgrad, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds2_gx, gds2, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds21_gx, gds21, rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gds22_gx, gds22, rtol=1.0e-6, atol=1.0e-8)

    kx0 = jnp.asarray([0.1])
    ky0 = jnp.asarray([0.2])
    kx_hat = kx0 / geom.s_hat
    cv_d_expected = ky0[:, None] * cv + kx_hat[:, None] * cv0
    gb_d_expected = ky0[:, None] * gb + kx_hat[:, None] * gb0
    assert jnp.allclose(cv_d, cv_d_expected[0], rtol=1.0e-10, atol=1.0e-12)
    assert jnp.allclose(gb_d, gb_d_expected[0], rtol=1.0e-10, atol=1.0e-12)

    kperp2 = geom.k_perp2(kx0, ky0, theta)
    bmag_inv = 1.0 / bmag
    shat_inv = 1.0 / geom.s_hat
    gds22_match = jnp.asarray(gds22_gx, dtype=gds2.dtype)
    kperp2_expected = (
        ky0[:, None] * (ky0[:, None] * gds2 + 2.0 * kx0[:, None] * shat_inv * gds21)
        + (kx0[:, None] * shat_inv) ** 2 * gds22_match
    ) * (bmag_inv * bmag_inv)
    # Allow one-ulp level differences from mixed float32/float64 intermediates.
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-8, atol=5.0e-10)


def test_salpha_geometry_kperp2_matches_alternate_formula():
    """Alternate kperp2 convention omits the bmag^{-2} factor."""
    geom = SAlphaGeometry(
        q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0, alpha=0.0, kperp2_bmag=False
    )
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    gds2 = 1.0 + shear * shear
    gds21 = -geom.s_hat * shear
    gds22 = jnp.asarray(geom.s_hat * geom.s_hat, dtype=jnp.float32)

    kx0 = jnp.asarray([0.1])
    ky0 = jnp.asarray([0.2])
    kx_hat = kx0 / geom.s_hat
    kperp2 = geom.k_perp2(kx0, ky0, theta)
    kperp2_expected = (
        ky0[:, None] * (ky0[:, None] * gds2 + 2.0 * kx_hat[:, None] * gds21)
        + (kx_hat[:, None] ** 2) * gds22
    )
    # Allow one-ulp level differences from mixed float32/float64 intermediates.
    assert jnp.allclose(kperp2, kperp2_expected[0], rtol=1.0e-8, atol=5.0e-10)


def test_hypercollisions_matches_gx_formula():
    Nl, Nm = 6, 12
    G = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.complex64)
    ell = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=jnp.float32)[None, :, None, None, None]
    vth = jnp.asarray([1.0], dtype=jnp.float32)
    nu_hyper_l = jnp.asarray(0.5, dtype=jnp.float32)
    nu_hyper_m = jnp.asarray(0.5, dtype=jnp.float32)
    nu_hyper_lm = jnp.asarray(0.0, dtype=jnp.float32)
    p_hyper_l = jnp.asarray(6.0, dtype=jnp.float32)
    p_hyper_m = jnp.asarray(6.0, dtype=jnp.float32)
    p_hyper_lm = jnp.asarray(6.0, dtype=jnp.float32)
    nu_hyper = jnp.asarray(0.0, dtype=jnp.float32)
    hyper_ratio = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    ratio_l = (ell / float(Nl)) ** p_hyper_l
    ratio_m = (m / float(Nm)) ** p_hyper_m
    ratio_lm = ((2.0 * ell + m) / (2.0 * float(Nl) + float(Nm))) ** p_hyper_lm
    mask_const = (m > 2.0) | (ell > 1.0)
    mask_kz = jnp.zeros_like(mask_const)
    m_pow = m**p_hyper_m
    m_norm_kz = float(max(Nm - 1, 1))
    m_norm_kz_factor = (p_hyper_m + 0.5) / (m_norm_kz ** (p_hyper_m + 0.5))
    kz = jnp.asarray([0.0], dtype=jnp.float32)
    kpar_scale = jnp.asarray(1.0, dtype=jnp.float32)

    out = hypercollisions_contribution(
        G,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=hyper_ratio,
        ratio_l=ratio_l,
        ratio_m=ratio_m,
        ratio_lm=ratio_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=m_pow,
        m_norm_kz_factor=m_norm_kz_factor,
        kz=kz,
        kpar_scale=kpar_scale,
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    l_norm = float(Nl)
    m_norm = float(Nm)
    scaled_nu_l = l_norm * nu_hyper_l
    scaled_nu_m = m_norm * nu_hyper_m
    hyper_term = (
        -vth[:, None, None, None, None, None]
        * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m)
        - nu_hyper_lm * ratio_lm
    )
    expected = jnp.where(mask_const, hyper_term, 0.0) * G

    assert jnp.allclose(out, expected, rtol=1.0e-6, atol=1.0e-7)


def test_hypercollisions_skips_linked_abs_kz_when_kz_weight_is_zero(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(
            "abs_z_linked_fft should not run when hypercollisions_kz is zero"
        )

    monkeypatch.setattr(linear_dissipation_module, "abs_z_linked_fft", _fail)

    Nl, Nm = 2, 4
    G = jnp.ones((1, Nl, Nm, 1, 1, 2), dtype=jnp.complex64)
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask_const = jnp.zeros((1, Nl, Nm, 1, 1, 1), dtype=bool)
    mask_kz = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)

    out = hypercollisions_contribution(
        G,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray([0.0], dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
        linked_gather_map=jnp.asarray([0], dtype=jnp.int32),
        linked_gather_mask=jnp.asarray([True], dtype=bool),
        linked_use_gather=True,
    )

    assert jnp.allclose(out, jnp.zeros_like(G))


def test_hypercollisions_static_zero_operator_skips_linked_abs_kz(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError(
            "abs_z_linked_fft should not run for an exactly zero hypercollision operator"
        )

    monkeypatch.setattr(linear_dissipation_module, "abs_z_linked_fft", _fail)

    Nl, Nm = 2, 4
    G = jnp.ones((1, Nl, Nm, 1, 1, 2), dtype=jnp.complex64)
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)

    out = hypercollisions_contribution(
        G,
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask,
        mask_kz=mask,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
        linked_gather_map=jnp.asarray([0], dtype=jnp.int32),
        linked_gather_mask=jnp.asarray([True], dtype=bool),
        linked_use_gather=True,
    )

    assert jnp.allclose(out, jnp.zeros_like(G))


def test_linked_kz_hypercollisions_activate_for_z_varying_state():
    Nl, Nm, Nz = 2, 4, 4
    zeros_lm = jnp.zeros((Nl, Nm, 1, 1, 1), dtype=jnp.float32)
    mask_const = jnp.zeros((1, Nl, Nm, 1, 1, 1), dtype=bool)
    mask_kz = jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=bool)
    kwargs = dict(
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        nu_hyper=jnp.asarray([0.0], dtype=jnp.float32),
        nu_hyper_l=jnp.asarray(0.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_lm=jnp.asarray(0.0, dtype=jnp.float32),
        hyper_ratio=zeros_lm,
        ratio_l=zeros_lm,
        ratio_m=zeros_lm,
        ratio_lm=zeros_lm,
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=jnp.ones((1, Nl, Nm, 1, 1, 1), dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_const=jnp.asarray(0.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32),),
        linked_inverse_permutation=jnp.asarray([0], dtype=jnp.int32),
        linked_full_cover=True,
    )

    constant = jnp.ones((1, Nl, Nm, 1, 1, Nz), dtype=jnp.complex64)
    z_varying = constant * jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=jnp.complex64)

    constant_out = hypercollisions_contribution(constant, **kwargs)
    varying_out = hypercollisions_contribution(z_varying, **kwargs)

    assert jnp.linalg.norm(constant_out) < 1.0e-6
    assert jnp.linalg.norm(varying_out) > 1.0e-3


def test_static_zero_linear_term_guards_skip_expensive_operators(monkeypatch):
    def _fail_streaming(*args, **kwargs):
        raise AssertionError(
            "streaming_ladder_term should not run when streaming weight is zero"
        )

    def _fail_grad(*args, **kwargs):
        raise AssertionError(
            "grad_z_periodic should not run when linked streaming weight is zero"
        )

    monkeypatch.setattr(linear_terms_module, "streaming_ladder_term", _fail_streaming)
    monkeypatch.setattr(linear_terms_module, "grad_z_periodic", _fail_grad)

    G = jnp.ones((1, 2, 3, 1, 1, 4), dtype=jnp.complex64)
    out = streaming_contribution(
        G,
        kz=jnp.ones((4,), dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        sqrt_p=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(0.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out, jnp.zeros_like(G))

    field = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    out_linked = linked_streaming_contribution(
        G,
        phi=field,
        apar=field,
        bpar=field,
        Jl=jnp.ones((1, 2, 1, 1, 4), dtype=jnp.float32),
        JlB=jnp.ones((1, 2, 1, 1, 4), dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        sqrt_p=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((2, 3, 1, 1, 1), dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(0.0, dtype=jnp.float32),
        kz=jnp.ones((4,), dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(out_linked, jnp.zeros_like(G))


def test_disabled_em_fields_match_explicit_zero_arrays_in_streaming_and_diamagnetic():
    G = jnp.arange(1 * 2 * 4 * 2 * 1 * 3, dtype=jnp.float32).reshape(1, 2, 4, 2, 1, 3)
    G = (G + 1j * (G + 1.0)).astype(jnp.complex64) * 1.0e-3
    phi = jnp.ones((2, 1, 3), dtype=jnp.complex64) * (0.2 + 0.1j)
    zero_field = jnp.zeros_like(phi)
    Jl = jnp.ones((1, 2, 2, 1, 3), dtype=jnp.float32)
    JlB = 0.5 * Jl
    tz = jnp.asarray([1.0], dtype=jnp.float32)
    vth = jnp.asarray([1.2], dtype=jnp.float32)
    sqrt_p = jnp.ones((2, 4, 1, 1, 1), dtype=jnp.float32)
    sqrt_m = 0.75 * sqrt_p
    common_streaming = dict(
        G=G,
        phi=phi,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        kz=jnp.asarray([0.0, 1.0, -1.0], dtype=jnp.float32),
        dz=jnp.asarray(1.0, dtype=jnp.float32),
    )
    explicit_streaming = linked_streaming_contribution(
        apar=zero_field, bpar=zero_field, **common_streaming
    )
    pruned_streaming = linked_streaming_contribution(
        apar=None, bpar=None, **common_streaming
    )
    assert jnp.allclose(pruned_streaming, explicit_streaming, rtol=1.0e-6, atol=1.0e-7)

    l4 = jnp.arange(2, dtype=jnp.float32)[:, None, None, None]
    common_diamagnetic = dict(
        dG=jnp.zeros_like(G),
        phi=phi,
        Jl=Jl,
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        JlB=JlB,
        l4=l4,
        tprim=jnp.asarray([2.0], dtype=jnp.float32),
        fprim=jnp.asarray([0.8], dtype=jnp.float32),
        tz=tz,
        vth=vth,
        omega_star_scale=jnp.asarray(1.0, dtype=jnp.float32),
        ky=jnp.asarray([0.0, 0.3], dtype=jnp.float32),
        imag=jnp.asarray(1j, dtype=jnp.complex64),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    explicit_diamagnetic = diamagnetic_contribution(
        apar=zero_field, bpar=zero_field, **common_diamagnetic
    )
    pruned_diamagnetic = diamagnetic_contribution(
        apar=None, bpar=None, **common_diamagnetic
    )
    assert jnp.allclose(
        pruned_diamagnetic, explicit_diamagnetic, rtol=1.0e-6, atol=1.0e-7
    )


def test_diamagnetic_drive_populates_only_expected_hermite_modes():
    dG = jnp.zeros((1, 2, 4, 1, 1, 1), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 1), dtype=jnp.complex64)
    Jl = jnp.ones((1, 2, 1, 1, 1), dtype=jnp.float32)
    JlB = jnp.ones_like(Jl)
    out = diamagnetic_contribution(
        dG,
        phi=phi,
        apar=None,
        bpar=None,
        Jl=Jl,
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
        JlB=JlB,
        l4=jnp.arange(2, dtype=jnp.float32)[:, None, None, None],
        tprim=jnp.asarray([2.0], dtype=jnp.float32),
        fprim=jnp.asarray([0.8], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        omega_star_scale=jnp.asarray(1.0, dtype=jnp.float32),
        ky=jnp.asarray([0.3], dtype=jnp.float32),
        imag=jnp.asarray(1j, dtype=jnp.complex64),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )

    mode_norm = jnp.sum(jnp.abs(out), axis=(0, 1, 3, 4, 5))
    assert mode_norm[0] > 0.0
    assert mode_norm[2] > 0.0
    assert jnp.allclose(mode_norm[jnp.asarray([1, 3])], 0.0)


def test_static_zero_damping_guards_return_zero_without_profiles():
    G = jnp.ones((1, 2, 3, 2, 2, 4), dtype=jnp.complex64)

    hyperdiff = hyperdiffusion_contribution(
        G,
        kx=jnp.asarray([], dtype=jnp.float32),
        ky=jnp.asarray([], dtype=jnp.float32),
        dealias_mask=jnp.zeros((0, 0), dtype=bool),
        D_hyper=jnp.asarray(0.0, dtype=jnp.float32),
        p_hyper_kperp=jnp.asarray(2.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(hyperdiff, jnp.zeros_like(G))

    damp = end_damping_contribution(
        G,
        ky=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        damp_profile=jnp.ones((4,), dtype=jnp.float32),
        linked_damp_profile=jnp.ones((3, 5), dtype=jnp.float32),
        damp_amp=jnp.asarray(0.0, dtype=jnp.float32),
        weight=jnp.asarray(1.0, dtype=jnp.float32),
    )
    assert jnp.allclose(damp, jnp.zeros_like(G))
