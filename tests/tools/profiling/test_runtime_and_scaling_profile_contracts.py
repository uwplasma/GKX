from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from support.paths import REPO_ROOT, load_profiling_tool
from scripts.profiling._profiler_options import make_profile_options
from scripts.profiling.profile_startup_and_cache import (
    PhaseTiming,
    _write_phase_csv,
    _write_phase_json,
    build_low_rank_moment_cache,
    main_runtime_startup,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
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
        jnp.asarray([0.0, 0.5]),
        SimpleNamespace(
            heat_flux_t=jnp.asarray([2.0, 4.0]),
            dt_t=jnp.asarray([0.1, 0.2]),
        ),
        jnp.arange(6, dtype=jnp.float32).reshape(1, 2, 3).astype(jnp.complex64),
        SimpleNamespace(phi=jnp.asarray([3.0j, 4.0])),
    )

    summary = runtime_kernels._prepared_result_summary(result)

    assert summary["time"]["shape"] == [2]
    assert summary["time"]["max_abs"] == 0.5
    assert summary["final_state"]["shape"] == [1, 2, 3]
    assert summary["final_state"]["finite_fraction"] == 1.0
    np.testing.assert_allclose(summary["phi"]["l2_norm"], 5.0)
    np.testing.assert_allclose(summary["heat_flux"]["sum_real"], 6.0)
    np.testing.assert_allclose(summary["dt"]["max_abs"], 0.2)


def test_runtime_profile_normalizes_peak_rss_units() -> None:
    assert runtime_kernels._peak_rss_bytes(123, system="Darwin") == 123
    assert runtime_kernels._peak_rss_bytes(123, system="Linux") == 123 * 1024


def test_runtime_startup_profiler_keywords_match_integration_contract() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(main_runtime_startup)))
    integration_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "integrate_nonlinear_explicit_diagnostics_state"
    ]

    assert len(integration_calls) == 1
    passed_keywords = {
        keyword.arg for keyword in integration_calls[0].keywords if keyword.arg
    }
    accepted_keywords = set(
        inspect.signature(integrate_nonlinear_explicit_diagnostics_state).parameters
    )
    assert passed_keywords <= accepted_keywords, passed_keywords - accepted_keywords


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
    expected_shapes = {
        "time": [21],
        "final_state": [1, 4, 8, 64, 64, 24],
        "phi": [64, 64, 24],
        "heat_flux": [21],
        "dt": [21],
    }
    for name, expected_shape in expected_shapes.items():
        assert (
            cpu["result_summary"][name]["shape"] == gpu["result_summary"][name]["shape"]
        )
        assert cpu["result_summary"][name]["shape"] == expected_shape
        assert cpu["result_summary"][name]["finite_fraction"] == 1.0
        assert gpu["result_summary"][name]["finite_fraction"] == 1.0
        # The full nonlinear state is a more sensitive accumulated trajectory
        # than the scalar diagnostics after 200 adaptive steps.  Keep its
        # observed CPU/GPU norm drift explicit instead of letting the old
        # mislabeled time vector provide a falsely exact state gate.
        rtol = 1.0e-3 if name == "final_state" else 1.0e-5
        np.testing.assert_allclose(
            cpu["result_summary"][name]["l2_norm"],
            gpu["result_summary"][name]["l2_norm"],
            rtol=rtol,
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
        for name in ("time", "final_state", "phi", "heat_flux", "dt"):
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


def test_hlo_op_counts_read_op_names_not_metadata() -> None:
    hlo = """
  %copy.6 = c64[2,4]{1,0} copy(%t), metadata={op_name="jit(f)/concatenate"}
  %concatenate.0 = c64[2,8]{1,0} concatenate(%copy.6, %c), dimensions={1}
  %gather.2 = f32[3] gather(%concatenate.0, %i), metadata={op_name="jit(f)/fft"}
  ROOT %copy.4 = f64[5] copy(%x)
  %while.1 = (c64[2], s32[]) while(%tuple), condition=%cond, body=%body
"""
    counts = runtime_kernels._hlo_op_counts(hlo)

    assert (counts["copy"], counts["concatenate"], counts["gather"]) == (2, 1, 1)
    assert counts["fft"] == 0
    assert counts["bytes_written"] == 8 * 8 + 16 * 8 + 5 * 8


def test_hlo_op_counts_price_captured_arrays_as_module_literals() -> None:
    """The ledger must price what a captured array costs the executable.

    The reference-route contract (queue row Q18) puts both nonlinear routes on
    the captured graph, so the module carries its cache, parameter and policy
    arrays as literals. ``constant_bytes`` is that payload and is counted apart
    from ``bytes_written``, which prices materialized copies.
    """

    hlo = """
  %constant.1 = f32[64]{0} constant({...})
  %constant.2 = s32[8]{0} constant({...})
  %copy.3 = f32[64]{0} copy(%constant.1)
"""
    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["constant_bytes"] == 64 * 4 + 8 * 4
    assert counts["bytes_written"] == 64 * 4
    assert counts["copy"] == 1


def test_hlo_op_counts_charge_only_buffers_that_are_written() -> None:
    """A copy inside a fusion body is an index, not a buffer (queue row Q27).

    XLA:CPU emits a fusion body as one loop nest, so a ``copy`` written inside
    one -- typically re-laying-out a fusion *parameter* so the consumer can
    read it in the order it wants -- allocates nothing; only the fusion's ROOT
    owns an output buffer.  ``bytes_written`` charges every such instruction
    the full size of its shape, which is why cloning one producer into more
    consumer fusions moved it by tens of per cent with no traffic behind it.
    ``materialized_bytes`` is the subset that owns a buffer, and the two
    partition ``bytes_written`` exactly.
    """

    hlo = """
%fused_computation.1 (param_0: f32[4]) -> f32[4] {
  %param_0.1 = f32[4]{0} parameter(0)
  %copy.1 = f32[4]{0} copy(%param_0.1)
  ROOT %copy.2 = f32[4]{0} copy(%copy.1)
}

%body.7 (arg: f32[8]) -> f32[8] {
  %arg.1 = f32[8]{0} parameter(0)
  ROOT %copy.3 = f32[8]{0} copy(%arg.1)
}

ENTRY %main (x: f32[4]) -> f32[4] {
  %x.1 = f32[4]{0} parameter(0)
  %fusion.1 = f32[4]{0} fusion(%x.1), kind=kLoop, calls=%fused_computation.1
  %while.1 = f32[8]{0} while(%w), condition=%cond.6, body=%body.7
  ROOT %copy.4 = f32[4]{0} copy(%fusion.1)
}
"""

    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["copy"] == 4
    # every copy, fusion interiors included -- the historical total
    assert counts["bytes_written"] == 4 * 4 + 4 * 4 + 8 * 4 + 4 * 4
    # the fusion ROOT, the while body's copy and the entry copy own buffers
    assert counts["materialized_bytes"] == 4 * 4 + 8 * 4 + 4 * 4
    # only %copy.1, written inside the fusion body and not its root
    assert counts["fused_interior_bytes"] == 4 * 4
    assert (
        counts["materialized_bytes"] + counts["fused_interior_bytes"]
        == counts["bytes_written"]
    )


def test_hlo_op_counts_without_computation_headers_are_all_materialized() -> None:
    """A bare instruction list has no fusion bodies, so nothing is interior.

    The older ledger snippets in this file are written that way, and their
    ``bytes_written`` must keep its value.
    """

    hlo = """
  %copy.6 = c64[2,4]{1,0} copy(%t)
  %concatenate.0 = c64[2,8]{1,0} concatenate(%copy.6, %c), dimensions={1}
"""

    counts = runtime_kernels._hlo_op_counts(hlo)

    assert counts["fused_interior_bytes"] == 0
    assert counts["materialized_bytes"] == counts["bytes_written"] == 8 * 8 + 16 * 8


def test_compiled_memory_stats_report_the_compilers_buffer_assignment() -> None:
    """``memory_analysis`` is the independent check on the text counts.

    It comes from XLA's buffer assignment rather than from the printed module,
    so an instruction a fusion emits as index arithmetic cannot inflate it. The
    step ledger records it beside the op counts for that reason.
    """

    stats: dict[str, int] = {}
    # An explicit dtype: the CI shards run with JAX_ENABLE_X64, under which an
    # unannotated zeros() is float64 and the argument is twice this size.
    runtime_kernels._compiled_hlo_text(
        lambda x: jnp.sum(x * 2.0),
        jnp.zeros((8, 8), dtype=jnp.float32),
        stats=stats,
    )

    assert set(stats) >= {
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    }
    assert stats["argument_size_in_bytes"] == 8 * 8 * 4
    assert all(value >= 0 for value in stats.values())


def test_nonlinear_step_hlo_routes_name_one_reference_graph() -> None:
    """``--route runtime`` and ``--route diagnostics`` are the same graph.

    Queue row Q18 gave ``integrate_nonlinear_explicit_diagnostics_state`` the
    prepared route's jit, so the ledger must not keep lowering a separate
    module for the runtime. ``--route eager-scan`` keeps the retired operand
    placement so the rejected option stays priceable.
    """

    parser = runtime_kernels.build_nonlinear_step_hlo_parser()
    choices = parser.parse_known_args(["--route", "eager-scan"])[0]
    assert choices.route == "eager-scan"

    source = textwrap.dedent(inspect.getsource(runtime_kernels.main_nonlinear_step_hlo))
    tree = ast.parse(source)
    selectors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Name)
        and node.body.id == "_eager_scan_hlo"
    ]
    assert len(selectors) == 1
    assert isinstance(selectors[0].orelse, ast.Name)
    assert selectors[0].orelse.id == "_diagnostics_scan_hlo"


def test_eager_scan_route_still_lowers_scan_body_arrays_as_arguments() -> None:
    """The retired placement must stay measurable, and stay the rejected one.

    A jit of the same function embeds the closed-over array as a constant.
    That captured graph is what both shipped routes now compile; the bound
    module below is the eager scan the contract retired.
    """

    weights = jnp.linspace(0.5, 1.5, 64, dtype=jnp.float32)

    def run(state: jnp.ndarray) -> jnp.ndarray:
        def step(carry: jnp.ndarray, _unused: None) -> tuple[jnp.ndarray, None]:
            return carry * weights, None

        return jax.lax.scan(step, state, None, length=3)[0]

    state = jnp.ones(64, dtype=jnp.float32)
    constant = re.compile(r"= f32\[64\]\S* constant\(")
    entry = re.compile(r"entry_computation_layout=\{\((?P<arguments>[^)]*)\)->")
    captured = runtime_kernels._compiled_hlo_text(run, state)
    bound = runtime_kernels._scan_equation_hlo(run, state)

    def entry_arguments(text: str) -> int:
        match = entry.search(text)
        assert match is not None
        return match.group("arguments").count("f32[64]")

    assert constant.search(captured) and entry_arguments(captured) == 1
    assert not constant.search(bound) and entry_arguments(bound) == 2


def test_full_linear_trace_summary_contains_metadata() -> None:
    payload = linear_trace._build_summary(
        config="benchmarks/cases/cyclone_nonlinear_miller.toml",
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
    assert payload["case"] == "cyclone_nonlinear_miller"
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
        config="benchmarks/cases/cyclone_nonlinear_miller.toml",
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
    assert payload["case"] == "cyclone_nonlinear_miller"
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


# Tracked parallel and RHS-term profile artifacts. Their profilers were retired
# in SLIM-SCRIPTS tranche 3 (recovery SHA in plan/research/2026-09-22-slim-tools/MAP.md);
# the committed JSONs stay as fixtures and keep these contracts.


def test_tracked_mixed_species_hermite_profile_is_scoped_and_identity_gated() -> None:
    artifact = (
        REPO_ROOT / "docs" / "_static" / "linear_rhs_species_hermite_profile_cpu.json"
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


def test_linear_rhs_terms_tracked_miller_profile_is_active_artifact() -> None:
    path = REPO_ROOT / "docs" / "_static" / "linear_rhs_terms_profile_miller_cpu.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear_miller"
    assert payload["state"] == "z_wave_linear_kick"
    assert payload["full_linear_rhs_seconds"] > 0.0
    assert payload["rows"]["streaming"]["norm"] > 0.0
    assert payload["dominant_nonzero_norm_term"] == "streaming"
