from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from support.paths import REPO_ROOT, load_profiling_tool
from tools.profiling import profile_linear_rhs_parallel_slices as linear_slices
from tools.profiling._profiler_options import make_profile_options
from tools.profiling.profile_startup_and_cache import (
    PhaseTiming,
    _write_phase_csv,
    _write_phase_json,
    build_low_rank_moment_cache,
)

runtime_kernels = load_profiling_tool("profile_runtime_kernels")
linear_trace = runtime_kernels
nonlinear_trace = runtime_kernels


def test_cyclone_runtime_profiler_default_config_exists() -> None:
    args = runtime_kernels.build_cyclone_parser().parse_args([])

    assert (REPO_ROOT / args.config).is_file()
    assert args.repeats == 1
    assert args.resolved_diagnostics is True
    assert args.reuse_prepared_simulation is False
    assert args.out is None


def test_prepared_profile_summary_fingerprints_numerical_outputs() -> None:
    result = (
        jnp.asarray([1.0 + 2.0j, 3.0 - 1.0j]),
        SimpleNamespace(heat_flux_t=jnp.asarray([2.0, 4.0])),
        jnp.asarray([0.1, 0.2]),
        SimpleNamespace(phi=jnp.asarray([3.0j, 4.0])),
    )

    summary = runtime_kernels._prepared_result_summary(result)

    assert summary["final_state"]["shape"] == [2]
    assert summary["final_state"]["finite_fraction"] == 1.0
    np.testing.assert_allclose(summary["phi"]["l2_norm"], 5.0)
    np.testing.assert_allclose(summary["heat_flux"]["sum_real"], 6.0)
    np.testing.assert_allclose(summary["dt"]["max_abs"], 0.2)


def test_runtime_profile_normalizes_peak_rss_units() -> None:
    assert runtime_kernels._peak_rss_bytes(123, system="Darwin") == 123
    assert runtime_kernels._peak_rss_bytes(123, system="Linux") == 123 * 1024


def test_sspx3_stage_profile_preserves_identity_without_speedup_claim() -> None:
    profile = json.loads(
        (REPO_ROOT / "docs/_static/linear_sspx3_stage_profile.json").read_text(
            encoding="utf-8"
        )
    )

    assert profile["identity_gate"]["passed"] is True
    assert profile["before"]["finite"] is True
    assert profile["after"]["finite"] is True
    assert profile["performance_gate"]["measurable_speedup"] is False
    assert profile["performance_gate"]["passed"] is False
    assert abs(profile["performance_gate"]["relative_median_time_change"]) < 0.03


def test_prepared_nonlinear_cpu_gpu_profiles_are_matched_and_clean() -> None:
    cpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_cpu_profile.json"
        ).read_text(encoding="utf-8")
    )
    gpu = json.loads(
        (
            REPO_ROOT / "docs/_static/prepared_nonlinear_runtime_gpu_profile.json"
        ).read_text(encoding="utf-8")
    )

    for profile in (cpu, gpu):
        assert profile["git_revision"] == cpu["git_revision"]
        assert profile["git_dirty"] is False
        assert profile["reuse_prepared_simulation"] is True
        assert profile["resolved_diagnostics"] is False
        assert profile["steps"] == 200
        assert profile["method"] == "rk3"
        assert profile["fixed_dt"] is False
        assert profile["sample_stride"] == 10
        assert profile["diagnostics_stride"] == 10
        assert profile["software"] == cpu["software"]
        assert "result_summary" in profile
        assert profile["memory_summary"]["host_peak_rss_bytes"] > 0
    assert cpu["backend"] == "cpu"
    assert gpu["backend"] == "gpu"
    assert cpu["run_median_s"] / gpu["run_median_s"] >= 5.0
    for name in ("final_state", "phi", "heat_flux", "dt"):
        assert (
            cpu["result_summary"][name]["shape"] == gpu["result_summary"][name]["shape"]
        )
        assert cpu["result_summary"][name]["finite_fraction"] == 1.0
        assert gpu["result_summary"][name]["finite_fraction"] == 1.0
        np.testing.assert_allclose(
            cpu["result_summary"][name]["l2_norm"],
            gpu["result_summary"][name]["l2_norm"],
            rtol=1.0e-5,
            atol=1.0e-12,
        )


def test_resolved_diagnostic_profiles_are_identity_gated_and_bounded() -> None:
    for backend in ("cpu", "gpu"):
        compact = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_profile.json"
            ).read_text(encoding="utf-8")
        )
        resolved = json.loads(
            (
                REPO_ROOT
                / f"docs/_static/prepared_nonlinear_runtime_{backend}_resolved_profile.json"
            ).read_text(encoding="utf-8")
        )
        assert compact["git_revision"] == resolved["git_revision"]
        assert compact["software"] == resolved["software"]
        assert compact["resolved_diagnostics"] is False
        assert resolved["resolved_diagnostics"] is True
        assert compact["steps"] == resolved["steps"] == 200
        assert resolved["run_median_s"] / compact["run_median_s"] <= 1.25
        assert (
            resolved["memory_summary"]["host_peak_rss_bytes"]
            / compact["memory_summary"]["host_peak_rss_bytes"]
            <= 1.10
        )
        for name in ("final_state", "phi", "heat_flux", "dt"):
            np.testing.assert_allclose(
                compact["result_summary"][name]["l2_norm"],
                resolved["result_summary"][name]["l2_norm"],
                rtol=1.0e-7,
                atol=1.0e-12,
            )
        if backend == "gpu":
            compact_peak = compact["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            resolved_peak = resolved["memory_summary"]["device_stats"][
                "peak_bytes_in_use"
            ]
            assert resolved_peak / compact_peak <= 1.10


def test_make_profile_options_defaults_disable_python_and_host_tracers() -> None:
    opts = make_profile_options()
    assert opts.python_tracer_level == 0
    assert opts.host_tracer_level == 0


def test_make_profile_options_accepts_explicit_levels() -> None:
    opts = make_profile_options(python_tracer_level=1, host_tracer_level=2)
    assert opts.python_tracer_level == 1
    assert opts.host_tracer_level == 2


def test_profile_linear_cache_uses_low_rank_moment_factors() -> None:
    params = SimpleNamespace(
        nu_hermite=0.5,
        nu_laguerre=0.25,
        p_hyper=4,
        p_hyper_l=3,
        p_hyper_m=5,
        p_hyper_lm=2,
    )

    cache = build_low_rank_moment_cache(
        nl=3, nm=4, params=params, real_dtype=jnp.float32
    )

    assert cache["lb_lam"].shape == (3, 4)
    assert cache["collision_lam"].shape == (0,)
    assert cache["hyper_ratio"].shape == (3, 4, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 4, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_


def test_profile_runtime_startup_writes_csv_and_json(tmp_path: Path) -> None:
    phases = [
        PhaseTiming(phase="a", seconds=1.25, note="first"),
        PhaseTiming(phase="b", seconds=2.75, note="second"),
    ]
    csv_path = tmp_path / "startup.csv"
    json_path = tmp_path / "startup.json"

    _write_phase_csv(csv_path, phases)
    _write_phase_json(json_path, phases, {"config": "case.toml", "device_count": 1})

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "phase,seconds,note" in csv_text
    assert "a,1.25,first" in csv_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["config"] == "case.toml"
    assert payload["startup_total_s"] == 4.0
    assert payload["phases"][1]["phase"] == "b"


def test_full_linear_trace_hlo_token_counts_are_coarse_but_stable() -> None:
    hlo = """
    ROOT fusion.1 = f32[2] fusion(arg), kind=kLoop
    fft.2 = c64[2] fft(arg), fft_type=FFT
    scatter.3 = f32[2] scatter(arg)
    """
    counts = linear_trace._hlo_token_counts(hlo)

    assert counts["fusion"] >= 1
    assert counts["fft"] >= 2
    assert counts["scatter"] >= 1
    assert counts["gather"] == 0


def test_full_linear_trace_summary_contains_metadata() -> None:
    payload = linear_trace._build_summary(
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
        backend="cpu",
        nl=4,
        nm=8,
        repeats=3,
        state="z_wave",
        z_variation_norm=0.2,
        compile_execute_seconds=1.5,
        warm_seconds=0.1,
        rhs_norm=2.0,
        phi_norm=3.0,
        hlo_text="ROOT add.1 = f32[] add(a, b)\n",
        trace_dir=Path("tools_out/trace"),
        memory_profile=Path("tools_out/memory.prof"),
        hlo_out=Path("tools_out/hlo.txt"),
        force_electrostatic_fields=True,
        source="gkx.operators.linear.rhs.linear_rhs_cached",
    )

    assert payload["kind"] == "full_linear_rhs_trace_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["backend"] == "cpu"
    assert payload["warm_seconds"] == 0.1
    assert payload["hlo_token_counts"]["add"] >= 1
    assert payload["force_electrostatic_fields"] is True
    assert payload["source"] == "gkx.operators.linear.rhs.linear_rhs_cached"
    assert payload["trace_dir"] == "tools_out/trace"
    assert "kernel-level optimization targets" in payload["claim_scope"]


def test_full_linear_trace_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    linear_trace._write_summary_json({"kind": "full", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "full",
        "value": 2.0,
    }


def test_full_linear_trace_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = linear_trace._inject_z_wave(
        state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1
    )

    assert linear_trace._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_full_nonlinear_trace_summary_contains_metadata() -> None:
    payload = nonlinear_trace._build_nonlinear_summary(
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml",
        backend="gpu",
        nl=4,
        nm=8,
        repeats=5,
        state="initial",
        laguerre_mode="grid",
        compressed_real_fft=True,
        z_variation_norm=0.0,
        compile_execute_seconds=2.0,
        warm_seconds=0.01,
        rhs_norm=1.0,
        phi_norm=0.1,
        apar_norm=0.0,
        bpar_norm=0.0,
        hlo_text="ROOT multiply.1 = f32[] multiply(a, b)\nfft.2 = c64[] fft(c)\n",
        trace_dir=Path("tools_out/nonlinear_trace"),
        memory_profile=Path("tools_out/nonlinear.prof"),
        hlo_out=Path("tools_out/nonlinear.hlo.txt"),
        electrostatic_specialized=True,
    )

    assert payload["kind"] == "full_nonlinear_rhs_trace_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["backend"] == "gpu"
    assert payload["laguerre_mode"] == "grid"
    assert payload["compressed_real_fft"] is True
    assert payload["hlo_token_counts"]["multiply"] >= 1
    assert payload["hlo_token_counts"]["fft"] >= 1
    assert payload["electrostatic_specialized"] is True
    assert payload["trace_dir"] == "tools_out/nonlinear_trace"
    assert "transport runtime claim" in payload["claim_scope"]


def test_full_nonlinear_trace_field_norm_handles_missing_em_fields() -> None:
    assert nonlinear_trace._field_norm(None) == 0.0


# Parallel-scaling profiling contracts.
parallel_workloads = load_profiling_tool("profile_parallel_workloads")
independent_ky = parallel_workloads
linear_terms = load_profiling_tool("profile_linear_rhs_terms")
quasilinear_uq = parallel_workloads


def test_independent_ky_scan_scaling_parser_defaults_to_large_solver_case() -> None:
    args = independent_ky.build_independent_ky_parser().parse_args([])

    assert args.out_prefix == independent_ky.DEFAULT_INDEPENDENT_KY_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.ny == 128
    assert args.nz == 96
    assert args.steps == 240
    assert len(args.ky) == 12


def test_independent_ky_scan_scaling_splits_ky_without_empty_chunks() -> None:
    chunks = independent_ky._split_ky(np.asarray([0.1, 0.2, 0.3]), 8)

    assert [chunk.tolist() for chunk in chunks] == [[0.1], [0.2], [0.3]]


def test_independent_ky_scan_scaling_worker_env_is_backend_specific() -> None:
    cpu_env = independent_ky._worker_env({}, backend="cpu", worker_index=3)
    gpu_env = independent_ky._worker_env({}, backend="gpu", worker_index=1)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_independent_ky_scan_scaling_identity_metrics_pass_equal_values() -> None:
    reference = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2]}
    row = {"ky": [0.1, 0.2], "gamma": [0.3, 0.4], "omega": [-0.1, -0.2], "error": None}

    metrics = independent_ky._identity_metrics(reference, row)

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_gamma_abs_error"] == 0.0
    assert metrics["max_omega_abs_error"] == 0.0


def test_profile_linear_rhs_parallel_slices_builds_summary(monkeypatch) -> None:
    class FakeGrid:
        ky = np.asarray([0.0, 0.3])
        z = np.asarray([0.0, 1.0, 2.0, 3.0])

    def fake_problem(**_kwargs):  # type: ignore[no-untyped-def]
        return (
            jnp.ones((1, 4, 2, 1, 4), dtype=jnp.complex64),
            object(),
            object(),
            FakeGrid(),
        )

    def fake_rhs(state, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return 2.0 * state, jnp.ones((2, 1, 4), dtype=jnp.complex64)

    monkeypatch.setattr(linear_slices, "build_problem", fake_problem)
    monkeypatch.setattr("jax.devices", lambda _kind=None: [object(), object()])
    monkeypatch.setattr("gkx.operators.linear.rhs.linear_rhs_cached", fake_rhs)
    monkeypatch.setattr("gkx.solvers.linear.parallel.linear_rhs_parallel_cached", fake_rhs)

    summary = linear_slices.profile_linear_rhs_parallel_slices(
        platform="cpu",
        requested_devices=2,
        nx=1,
        ny=2,
        nz=4,
        nl=1,
        nm=4,
        warmups=0,
        repeats=1,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert summary["identity_passed"] is True
    assert summary["max_abs_error"] == 0.0
    assert summary["speedup"] > 0.0
    assert len(summary["rows"]) == 2


def test_profile_linear_rhs_parallel_slices_writes_artifacts(tmp_path: Path) -> None:
    summary = {
        "rows": [
            {"route": "serial", "median_s": 0.02, "samples_s": [0.02]},
            {"route": "sharded", "median_s": 0.03, "samples_s": [0.03]},
        ],
        "atol": 1.0e-8,
        "rtol": 1.0e-8,
        "identity_passed": True,
        "speedup": 0.67,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "max_phi_abs_error": 0.0,
    }
    out = tmp_path / "linear_rhs_parallel_slices_profile"
    paths = linear_slices.write_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "median_s" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()


def test_tracked_mixed_species_hermite_profile_is_scoped_and_identity_gated() -> None:
    artifact = (
        linear_slices.REPO_ROOT
        / "docs"
        / "_static"
        / "linear_rhs_species_hermite_profile_cpu.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["decomposition_axis"] == "species_hermite"
    assert payload["requested_devices"] == 4
    assert payload["actual_devices"] == 4
    assert payload["identity_passed"] is True
    integration = payload["integration"]
    assert integration["identity_passed"] is True
    assert integration["speedup_passed"] is (integration["speedup"] > 1.0)
    assert integration["state_identity"]["max_abs_error"] <= payload["atol"]
    assert integration["field_history_identity"]["max_abs_error"] <= payload["atol"]
    assert (
        "mixed species-Hermite collision-free integration" in integration["claim_scope"]
    )
    assert payload["max_rel_error"] <= payload["rtol"]
    assert payload["max_abs_error"] <= payload["atol"]
    assert payload["max_phi_abs_error"] <= payload["atol"]
    assert payload["speedup"] > 1.0
    assert "not a GPU or general scaling claim" in payload["claim_scope"]
    assert len(payload["git_revision"]) == 40


def test_linear_rhs_sweep_subcommand_builds_summary(
    monkeypatch,
) -> None:
    def fake_profile(**kwargs):  # type: ignore[no-untyped-def]
        devices = int(kwargs["requested_devices"])
        nm = int(kwargs["nm"])
        serial = float(nm) * 1.0e-3
        sharded = serial / max(devices, 1)
        return {
            "state_shape": (4, nm, 8, 1, 16),
            "serial_median_s": serial,
            "sharded_median_s": sharded,
            "speedup": serial / sharded,
            "identity_passed": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "max_phi_abs_error": 0.0,
        }

    monkeypatch.setattr(
        linear_slices, "profile_linear_rhs_parallel_slices", fake_profile
    )

    summary = linear_slices.run_sweep(
        platform="cpu",
        devices=[1, 2],
        nms=[8, 16],
        nx=1,
        ny=4,
        nz=8,
        nl=2,
        warmups=0,
        repeats=1,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert summary["identity_passed"] is True
    assert len(summary["rows"]) == 4
    assert {row["speedup"] for row in summary["rows"]} == {1.0, 2.0}


def test_linear_rhs_sweep_subcommand_writes_artifacts(
    tmp_path: Path,
) -> None:
    summary = {
        "identity_passed": True,
        "rtol": 1.0e-5,
        "rows": [
            {
                "platform": "cpu",
                "requested_devices": 1,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.02,
                "speedup": 1.0,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
            {
                "platform": "cpu",
                "requested_devices": 2,
                "nm": 8,
                "state_shape": (2, 8, 4, 1, 8),
                "serial_median_s": 0.02,
                "sharded_median_s": 0.012,
                "speedup": 1.67,
                "identity_passed": True,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "max_phi_abs_error": 0.0,
            },
        ],
    }
    out = tmp_path / "linear_rhs_parallel_slices_sweep"
    paths = linear_slices.write_sweep_artifacts(summary, out)

    assert (
        json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))[
            "identity_passed"
        ]
        is True
    )
    assert "requested_devices" in out.with_suffix(".csv").read_text(encoding="utf-8")
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()


def test_linear_rhs_terms_summary_reports_dominant_and_zero_norm_terms() -> None:
    payload = linear_terms._build_summary(
        [
            {"term": "field_solve", "seconds": 0.1, "norm": 1.0},
            {"term": "streaming", "seconds": 0.2, "norm": 2.0},
            {"term": "collisions", "seconds": 0.4, "norm": 0.0},
            {"term": "linked_abs_kz", "seconds": 0.3, "norm": 1.0e-16},
            {"term": "full_linear_rhs", "seconds": 1.2, "norm": 3.0},
        ],
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        repeats=5,
        backend="cpu",
        state="z_wave",
        z_variation_norm=0.25,
    )

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear"
    assert payload["state"] == "z_wave"
    assert payload["z_variation_norm"] == 0.25
    assert payload["dominant_measured_term"] == "collisions"
    assert payload["dominant_nonzero_norm_term"] == "streaming"
    assert payload["zero_norm_terms_by_time"][0]["term"] == "collisions"
    assert payload["full_over_sum_independently_measured_components"] == 1.2
    assert "initial state only" in payload["claim_scope"]


def test_linear_rhs_terms_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    linear_terms._write_summary_json({"kind": "linear", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "linear",
        "value": 2.0,
    }


def test_linear_rhs_terms_inject_z_wave_adds_parallel_variation() -> None:
    state = jnp.zeros((1, 4, 3, 2, 1, 5), dtype=jnp.complex64)

    out = linear_terms._inject_z_wave(
        state, ky_index=1, kx_index=0, amplitude=0.2, z_mode=1
    )

    assert linear_terms._z_variation_norm(out) > 0.0
    assert jnp.linalg.norm(out[0, 0, 2, 1, 0]) > 0.0


def test_linear_rhs_terms_hypercollision_kz_source_uses_nu_hyper_m_path() -> None:
    state = jnp.zeros((1, 2, 4, 1, 1, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 3, 0, 0, :].set(
        jnp.asarray([0.0, 1.0, 0.0, -1.0], dtype=jnp.complex64)
    )
    mask_kz = jnp.ones((2, 4, 1, 1, 1), dtype=bool)
    m_pow = jnp.ones((2, 4, 1, 1, 1), dtype=jnp.float32)

    source = linear_terms._hypercollision_kz_source(
        state,
        weight=jnp.asarray(1.0, dtype=jnp.float32),
        hypercollisions_kz=jnp.asarray(1.0, dtype=jnp.float32),
        nu_hyper_m=jnp.asarray(1.0, dtype=jnp.float32),
        m_norm_kz_factor=jnp.asarray(0.5, dtype=jnp.float32),
        vth=jnp.asarray([2.0], dtype=jnp.float32),
        kpar_scale=jnp.asarray(1.0, dtype=jnp.float32),
        mask_kz=mask_kz,
        m_pow=m_pow,
    )

    assert jnp.linalg.norm(source) > 0.0
    assert jnp.allclose(source[0, 0, 3, 0, 0, 1], -2.3 + 0.0j)


def test_linear_rhs_terms_tracked_miller_profile_is_active_artifact() -> None:
    path = REPO_ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"


def test_quasilinear_uq_ensemble_scaling_parser_defaults_to_bounded_solver_case() -> (
    None
):
    args = quasilinear_uq.build_quasilinear_uq_parser().parse_args([])

    assert args.out_prefix == quasilinear_uq.DEFAULT_QUASILINEAR_UQ_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2, 4]
    assert args.gradients == [2.20, 2.40, 2.60, 2.80, 3.00, 3.20]
    assert args.ky == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert args.steps == 2000
    assert args.sample_stride == 10
    assert args.fit_start_fraction == 0.5
    assert args.fit_end_fraction == 0.95


def test_quasilinear_uq_ensemble_scaling_reduced_observable_is_positive() -> None:
    obs = quasilinear_uq._quasilinear_reduced_observables(
        np.asarray([0.2, 0.4]),
        np.asarray([0.1, -0.1]),
        np.asarray([0.3, 0.1]),
    )

    assert obs["heat_flux_proxy"] > 0.0
    assert obs["weighted_growth"] == 0.1
    assert obs["omega_span"] == 0.19999999999999998


def test_quasilinear_uq_ensemble_scaling_identity_metrics_detect_equal_members() -> (
    None
):
    members = [
        {"tprim": 2.4, "heat_flux_proxy": 1.5, "gamma": [0.1, 0.2]},
        {"tprim": 2.7, "heat_flux_proxy": 2.5, "gamma": [0.3, 0.4]},
    ]

    metrics = quasilinear_uq._quasilinear_identity_metrics(
        {"members": members},
        {"members": list(reversed(members)), "error": None},
        value_rtol=1.0e-12,
        value_atol=1.0e-12,
    )

    assert metrics["identity_gate_pass"] is True
    assert metrics["max_heat_flux_proxy_abs_error"] == 0.0
    assert metrics["max_gamma_abs_error"] == 0.0


def test_quasilinear_uq_ensemble_scaling_worker_env_selects_device() -> None:
    gpu_env = quasilinear_uq._worker_env({}, backend="gpu", worker_index=1)
    cpu_env = quasilinear_uq._worker_env({}, backend="cpu", worker_index=0)

    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert cpu_env["OMP_NUM_THREADS"] == "1"
