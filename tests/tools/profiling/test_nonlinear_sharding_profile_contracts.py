from __future__ import annotations

from support.paths import load_profiling_tool
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from gkx.terms.config import FieldState
from gkx.terms.config import TermConfig


def _load_sharding_tool_module():
    return load_profiling_tool("profile_nonlinear_sharding")


def test_profile_nonlinear_sharding_parser_defaults_to_tracked_artifact() -> None:
    mod = _load_sharding_tool_module()
    args = mod.build_parser().parse_args([])

    assert args.out_json == mod.DEFAULT_OUT
    assert args.sharding == "auto"
    assert args.sharding_options is None
    assert args.method == "rk2"
    assert args.warmups == 1
    assert args.repeats == 3
    assert args.allow_unsafe_cpu_state_sharding is False
    assert (
        mod._artifact_path_for_contract(args.out_json)
        == "docs/_static/nonlinear_sharding_profile.json"
    )


def test_profile_nonlinear_sharding_documented_script_entrypoint() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/profiling/profile_nonlinear_sharding.py"),
            "--help",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "numerical identity gate" in result.stdout


def test_profile_nonlinear_sharding_problem_excites_nonlinear_bracket() -> None:
    mod = _load_sharding_tool_module()
    args = SimpleNamespace(nx=8, ny=8, nz=12, nl=2, nm=3, amplitude=1.0e-4)
    state, cache, params = mod._build_problem(args)
    nonlinear_terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=1.0,
    )
    rhs, _fields = mod.nonlinear_rhs_cached(
        state,
        cache,
        params,
        nonlinear_terms,
        compressed_real_fft=True,
        laguerre_mode="grid",
    )

    assert int(jnp.count_nonzero(jnp.abs(state) > 0.0)) > int(args.nz)
    assert float(jnp.max(jnp.abs(rhs))) > 0.0
    assert (
        mod._initial_nonlinear_activity(state, cache, params, laguerre_mode="grid")
        > 0.0
    )


def test_profile_nonlinear_sharding_source_contract_is_machine_readable(
    tmp_path: Path,
) -> None:
    mod = _load_sharding_tool_module()
    out_json = tmp_path / "profile.json"
    argv = [
        "--out-json",
        str(out_json),
        "--sharding",
        "kx",
        "--warmups",
        "0",
        "--repeats",
        "2",
    ]
    args = mod.build_parser().parse_args(argv)

    contract = mod._source_contract(args, argv, backend="gpu", device_count=2)

    assert contract["backend"] == "gpu"
    assert contract["source_contract_version"] == 1
    assert contract["device_count"] == 2
    assert contract["sharding_axis"] == "kx"
    assert contract["source_artifact"] == str(out_json.resolve())
    assert contract["timing_warmup_repeat"] == {"warmups": 0, "repeats": 2}
    assert contract["allow_unsafe_cpu_state_sharding"] is False
    assert contract["profile_command_argv"][-len(argv) :] == argv
    assert (
        "tools/profiling/profile_nonlinear_sharding.py" in contract["profile_command"]
    )
    assert {"python", "gkx", "jax", "jaxlib", "numpy"} <= set(
        contract["software_versions"]
    )
    assert all(contract["software_versions"].values())
    assert contract["git_revision"]
    assert isinstance(contract["git_dirty"], bool)


def test_profile_nonlinear_sharding_helpers_report_stats_and_unique_specs() -> None:
    mod = _load_sharding_tool_module()

    stats = mod._time_stats([3.0, 1.0, 2.0])

    assert stats["min"] == 1.0
    assert stats["median"] == 2.0
    assert stats["mean"] == 2.0
    assert stats["max"] == 3.0
    assert mod._sharding_specs("auto", "ky,kx,ky,z") == ["auto", "ky", "kx", "z"]
    assert mod._sharding_specs("auto,kx", None) == ["auto", "kx"]


def test_profile_nonlinear_sharding_reports_best_identity_candidate() -> None:
    mod = _load_sharding_tool_module()

    best = mod._best_identity_preserving_candidate(
        {
            "auto": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 0.8,
                "state_sharding_active": True,
            },
            "kx": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 1.2,
                "state_sharding_active": True,
            },
            "z": {
                "identity_gate_pass": False,
                "engineering_speedup_median": 3.0,
                "state_sharding_active": True,
            },
        }
    )

    assert best == {
        "spec": "kx",
        "engineering_speedup_median": 1.2,
        "state_sharding_active": True,
        "identity_gate_pass": True,
    }


def test_profile_nonlinear_sharding_excludes_inactive_speedup_candidates() -> None:
    mod = _load_sharding_tool_module()

    best = mod._best_identity_preserving_candidate(
        {
            "auto": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 4.0,
                "state_sharding_active": False,
            },
            "kx": {
                "identity_gate_pass": True,
                "engineering_speedup_median": 1.1,
                "state_sharding_active": True,
            },
        }
    )

    assert best["spec"] == "kx"
    assert best["engineering_speedup_median"] == 1.1


def test_profile_nonlinear_sharding_skips_unsafe_cpu_state_sharding() -> None:
    mod = _load_sharding_tool_module()

    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is True
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="cpu",
            device_count=4,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=True,
        )
        is False
    )
    assert (
        mod._skip_unsafe_cpu_state_sharding(
            backend="gpu",
            device_count=2,
            state_sharding_active=True,
            allow_unsafe_cpu_state_sharding=False,
        )
        is False
    )

    row = mod._candidate_failure(
        state_sharding_active=True,
        error=mod.CPU_WHOLE_STATE_SHARDING_SKIP_REASON,
        skip_reason="cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
    )

    assert row["identity_gate_pass"] is False
    assert row["state_sharding_active"] is True
    assert row["skip_reason"] == "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout"
    assert "unsafe_for_fft_layout" in row["error"]


def test_profile_nonlinear_sharding_diagnostic_metrics_compare_rhs_and_phi(
    monkeypatch,
) -> None:
    mod = _load_sharding_tool_module()

    def fake_rhs(
        state, cache, params, terms, *, compressed_real_fft=True, laguerre_mode="grid"
    ):
        del cache, params, terms, compressed_real_fft, laguerre_mode
        arr = jnp.asarray(state)
        return 2.0 * arr, FieldState(
            phi=jnp.sum(arr, axis=(0, 1)), apar=None, bpar=None
        )

    monkeypatch.setattr(mod, "nonlinear_rhs_cached", fake_rhs)

    reference = jnp.ones((2, 2, 1, 1, 3), dtype=jnp.complex64)
    candidate = reference.at[0, 0, 0, 0, 0].add(1.0e-3)

    metrics = mod._nonlinear_diagnostic_identity_metrics(
        reference,
        candidate,
        cache=object(),
        params=object(),
        terms=object(),
        compressed_real_fft=True,
        laguerre_mode="grid",
    )

    assert metrics["max_abs_rhs_error"] == pytest.approx(2.0e-3, rel=1.0e-4)
    assert metrics["max_abs_phi_error"] == pytest.approx(1.0e-3, rel=1.0e-4)
    assert metrics["max_rel_rhs_error"] > 0.0
    assert metrics["max_rel_phi_error"] > 0.0


# Sweep-driver contracts for the same nonlinear sharding profiling lane.
def _load_sweep_tool_module():
    return load_profiling_tool("profile_nonlinear_sharding")


def _load_device_z_pencil_tool_module():
    return load_profiling_tool("profile_device_z_pencil_transport_window")


def test_nonlinear_sharding_sweep_subcommand_parser_defaults_to_bounded_artifact() -> (
    None
):
    mod = _load_sweep_tool_module()

    args = mod.build_sweep_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_SWEEP_PREFIX
    assert args.backend == "cpu"
    assert args.devices == [1, 2]
    assert args.sharding_options == "auto,kx"
    assert args.timeout_s == 300.0
    assert args.office_gpu_xlarge is False


def test_nonlinear_sharding_sweep_subcommand_office_gpu_preset_is_canonical() -> None:
    mod = _load_sweep_tool_module()

    args = mod.apply_sweep_preset(
        mod.build_sweep_parser().parse_args(["--office-gpu-xlarge"])
    )

    assert args.backend == "gpu"
    assert args.devices == [1, 2]
    assert (args.nx, args.ny, args.nz, args.nl, args.nm, args.steps) == (
        48,
        96,
        128,
        4,
        8,
        12,
    )
    assert args.sharding_options == "auto,kx"
    assert args.out_prefix == mod.OFFICE_GPU_XLARGE_PREFIX
    assert args.trace is True


def test_nonlinear_sharding_sweep_subcommand_device_env_is_backend_specific() -> None:
    mod = _load_sweep_tool_module()

    cpu_env = mod._device_env({"XLA_FLAGS": "--foo=bar"}, backend="cpu", devices=4)
    replaced_cpu_env = mod._device_env(
        {"XLA_FLAGS": "--foo=bar --xla_force_host_platform_device_count=8"},
        backend="cpu",
        devices=2,
    )
    gpu_env = mod._device_env({}, backend="gpu", devices=2)

    assert cpu_env["JAX_PLATFORMS"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in cpu_env["XLA_FLAGS"]
    assert (
        "--xla_force_host_platform_device_count=8" not in replaced_cpu_env["XLA_FLAGS"]
    )
    assert "--xla_force_host_platform_device_count=2" in replaced_cpu_env["XLA_FLAGS"]
    assert "--foo=bar" in replaced_cpu_env["XLA_FLAGS"]
    assert gpu_env["JAX_PLATFORMS"] == "cuda"
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_nonlinear_sharding_sweep_subcommand_selects_fastest_identity_candidate() -> (
    None
):
    mod = _load_sweep_tool_module()
    payload = {
        "source_contract_version": 1,
        "backend": "gpu",
        "device_count": 2,
        "default_backend": "gpu",
        "sharding_axis": "kx",
        "profile_command": "python tools/profiling/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "tools/profiling/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "/tmp/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "gkx": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 1, "repeats": 3},
        "state_shape": [4, 8, 17, 32, 64],
        "state_sharding_requested": "auto",
        "serial_stats_s": {"median": 10.0},
        "best_identity_preserving_candidate": {"spec": "kx"},
        "sharded_results": {
            "auto": {
                "state_sharding_active": True,
                "stats_s": {"median": 8.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
            "kx": {
                "state_sharding_active": True,
                "stats_s": {"median": 5.0},
                "identity_gate_pass": True,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "error": None,
            },
        },
    }

    row = mod._row_from_payload(payload, requested_devices=2)

    assert row["best_spec"] == "kx"
    assert row["parallel_median_s"] == 5.0
    assert row["same_process_speedup"] == 2.0
    assert row["identity_gate_pass"] is True
    assert row["source_contract_version"] == 1
    assert row["profile_command"].startswith(
        "python tools/profiling/profile_nonlinear_sharding.py"
    )
    assert row["profile_command_argv"][-2:] == ["--sharding", "kx"]
    assert row["source_artifact"] == "/tmp/profile.json"
    assert row["software_versions"]["gkx"] == "test"
    assert row["timing_warmup_repeat"] == {"warmups": 1, "repeats": 3}
    assert row["profile_backend"] == "gpu"
    assert row["profile_device_count"] == 2
    assert row["profile_sharding_axis"] == "kx"


def test_nonlinear_sharding_sweep_subcommand_json_clean_replaces_nonfinite() -> None:
    mod = _load_sweep_tool_module()

    cleaned = mod._json_clean({"bad": math.inf, "ok": 1.0})

    assert cleaned == {"bad": None, "ok": 1.0}


def test_nonlinear_sharding_sweep_subcommand_records_timeout_rows(monkeypatch) -> None:
    mod = _load_sweep_tool_module()

    def _raise_timeout(*_args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd="profile",
            timeout=float(kwargs["timeout"]),
            output="stdout tail",
            stderr="stderr tail",
        )

    monkeypatch.setattr(mod.subprocess, "run", _raise_timeout)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=0.5,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][0]["parallel_median_s"] is None
    assert "timed out" in summary["rows"][0]["error"]
    assert "stderr tail" in summary["rows"][0]["error"]
    assert summary["speedup_blockers"] == ["cpu_2devices_identity_failed"]


def test_nonlinear_sharding_sweep_subcommand_marks_identity_only_slowdown(
    monkeypatch,
) -> None:
    mod = _load_sweep_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        device_count = 2 if "2devices" in out_json.name else 1
        spec = "kx" if device_count == 2 else "auto"
        median = 20.0 if device_count == 2 else 10.0
        payload = {
            "device_count": device_count,
            "default_backend": "gpu",
            "state_shape": [1],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {"spec": spec},
            "sharded_results": {
                spec: {
                    "state_sharding_active": device_count > 1,
                    "stats_s": {"median": median},
                    "identity_gate_pass": True,
                    "max_abs_state_error": 0.0,
                    "max_rel_state_error": 0.0,
                    "error": None,
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="gpu",
        devices=[1, 2],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto,kx",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is True
    assert summary["speedup_passed"] is False
    assert summary["status"] == "diagnostic_identity_only"
    assert summary["rows"][1]["strong_speedup_vs_1_device"] == 0.5
    assert summary["speedup_blockers"] == ["gpu_2devices_speedup_0.5_below_1"]


def test_nonlinear_sharding_sweep_subcommand_preserves_failed_profile_json(
    monkeypatch,
) -> None:
    mod = _load_sweep_tool_module()

    def _fake_run(cmd, **_kwargs):
        out_json = Path(cmd[cmd.index("--out-json") + 1])
        payload = {
            "device_count": 4,
            "default_backend": "cpu",
            "state_shape": [4, 8, 17, 32, 64],
            "state_sharding_requested": "auto",
            "serial_stats_s": {"median": 10.0},
            "best_identity_preserving_candidate": {
                "spec": None,
                "identity_gate_pass": False,
                "state_sharding_active": False,
                "engineering_speedup_median": None,
            },
            "sharded_results": {
                "auto": {
                    "state_sharding_active": True,
                    "stats_s": None,
                    "identity_gate_pass": False,
                    "max_abs_state_error": None,
                    "max_rel_state_error": None,
                    "error": "skipped: cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                    "skip_reason": "cpu_whole_state_pjit_sharding_unsafe_for_fft_layout",
                }
            },
        }
        out_json.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 2, "profile json written", "")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    summary = mod.run_sweep(
        backend="cpu",
        devices=[4],
        nx=4,
        ny=4,
        nz=4,
        nl=1,
        nm=1,
        dt=0.02,
        steps=1,
        method="rk2",
        sharding="auto",
        sharding_options="auto",
        laguerre_mode="grid",
        warmups=0,
        repeats=1,
        timeout_s=1.0,
        trace=False,
    )

    assert summary["identity_passed"] is False
    assert summary["rows"][0]["profile_returncode"] == 2
    assert summary["rows"][0]["state_sharding_active"] is True
    assert "unsafe_for_fft_layout" in summary["rows"][0]["error"]
    assert summary["profiles"]["4"]["profile_returncode"] == 2


# Device-z pencil transport-window profiler contracts for the same nonlinear
# domain-decomposition evidence lane.
def test_device_z_pencil_auto_chunk_records_fft_batch_pressure_model(
    monkeypatch,
) -> None:
    mod = _load_device_z_pencil_tool_module()
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    payload = mod.build_profile(
        shape=(1, 1, 4, 4, 8),
        device_counts=(1, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        observable_repeats=0,
        atol=5.0e-6,
        rtol=1.0e-4,
        min_speedup=1.5,
        z_chunk_size=None,
        auto_z_chunk_size=True,
        max_fft_batch_count=4,
        observable_mode="host_gather",
        trace_dir=None,
        trace_device_count=None,
        hlo_prefix=None,
    )

    model = payload["fft_batch_pressure_model"]
    assert payload["z_chunk_size"] == 1
    assert payload["auto_z_chunk_size"] is True
    assert model["chunking_required"] is True
    assert model["suggested_z_chunk_size"] == 1
    assert model["chunked_fft_batch_count"] == 4
    assert payload["rows"][1]["blocked_reasons"] == ["not_enough_devices"]
    assert payload["observable_repeats"] == 0
    assert payload["observable_mode"] == "host_gather"
    assert payload["rows"][0]["timing_scope"] == "compute_only_final_state_update"
    assert payload["rows"][1]["observable_gate_stats_s"] == {}


def test_device_z_pencil_rhs_mode_records_rhs_identity_gate(monkeypatch) -> None:
    mod = _load_device_z_pencil_tool_module()
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    payload = mod.build_rhs_profile(
        shape=(1, 1, 4, 4, 8),
        device_counts=(1, 2),
        warmups=0,
        repeats=1,
        atol=5.0e-6,
        rtol=1.0e-4,
        min_speedup=1.5,
    )

    assert payload["kind"] == "nonlinear_device_z_pencil_rhs_profile"
    assert payload["rows"][0]["rhs_max_abs_error"] == 0.0
    assert payload["rows"][1]["blocked_reasons"] == ["not_enough_devices"]
    assert payload["summary"]["status"] == "skipped_no_multidevice"


def test_device_z_pencil_parser_defaults_to_transport_window_mode() -> None:
    mod = _load_device_z_pencil_tool_module()

    args = mod.build_parser().parse_args([])

    assert args.mode == "transport-window"
    assert args.out_prefix is None


def test_device_z_pencil_main_dispatches_rhs_mode_to_rhs_profiler(
    monkeypatch,
) -> None:
    mod = _load_device_z_pencil_tool_module()
    calls: dict[str, object] = {}

    def fake_build_rhs_profile(**kwargs):
        calls["profile"] = kwargs
        return {
            "kind": "nonlinear_device_z_pencil_rhs_profile",
            "summary": {"status": "fake_rhs"},
        }

    def fake_write_artifacts(summary, out_prefix):
        calls["write"] = (summary, out_prefix)

    monkeypatch.setattr(mod, "build_rhs_profile", fake_build_rhs_profile)
    monkeypatch.setattr(mod, "write_artifacts", fake_write_artifacts)

    rc = mod.main(
        [
            "--mode",
            "rhs",
            "--shape",
            "1,1,4,4,8",
            "--device-counts",
            "1",
            "--warmups",
            "0",
            "--repeats",
            "1",
        ]
    )

    assert rc == 0
    assert calls["profile"] == {
        "shape": (1, 1, 4, 4, 8),
        "device_counts": (1,),
        "warmups": 0,
        "repeats": 1,
        "atol": 5.0e-6,
        "rtol": 1.0e-4,
        "min_speedup": 1.5,
    }
    summary, out_prefix = calls["write"]  # type: ignore[misc]
    assert summary["summary"]["status"] == "fake_rhs"
    assert out_prefix == mod.DEFAULT_RHS_OUT_PREFIX


def test_device_z_pencil_observable_repeats_rejects_negative_values(
    monkeypatch,
) -> None:
    mod = _load_device_z_pencil_tool_module()
    real_devices = tuple(jax.devices())
    monkeypatch.setattr(mod.jax, "devices", lambda: [real_devices[0]])

    with pytest.raises(ValueError, match="observable_repeats"):
        mod.build_profile(
            shape=(1, 1, 4, 4, 8),
            device_counts=(1,),
            steps=1,
            dt=0.001,
            warmups=0,
            repeats=1,
            observable_repeats=-1,
            atol=5.0e-6,
            rtol=1.0e-4,
            min_speedup=1.5,
            z_chunk_size=None,
            auto_z_chunk_size=False,
            max_fft_batch_count=4,
            observable_mode="host_gather",
            trace_dir=None,
            trace_device_count=None,
            hlo_prefix=None,
        )
