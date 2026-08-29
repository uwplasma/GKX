"""Unit contracts: parallel nonlinear."""

from __future__ import annotations


# ---- test_nonlinear_domain_parallel.py ----

import jax
import jax.numpy as jnp
import pytest

import gkx.operators.nonlinear.parallel as nonlinear_parallel
import gkx.operators.nonlinear.domain_decomposition as nonlinear_parallel_domain
from gkx.operators.nonlinear.parallel import (
    NonlinearDomainDecompositionPlan,
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    local_stencil_nonlinear_domain_decomposed_step,
    local_stencil_nonlinear_domain_serial_step,
)


def test_nonlinear_domain_facade_reexports_domain_gate_functions() -> None:
    public_domain_names = (
        "build_nonlinear_domain_decomposition_plan",
        "deterministic_nonlinear_domain_state",
        "nonlinear_domain_identity_report",
        "nonlinear_domain_parallel_identity_gate",
        "nonlinear_domain_transport_window_identity_gate",
        "local_stencil_nonlinear_domain_decomposed_step",
        "local_stencil_nonlinear_domain_serial_step",
    )

    for name in public_domain_names:
        assert getattr(nonlinear_parallel, name) is getattr(
            nonlinear_parallel_domain,
            name,
        )


def test_nonlinear_domain_plan_uses_static_halo_chunks() -> None:
    plan = build_nonlinear_domain_decomposition_plan(
        (7, 3),
        axis=-2,
        num_domains=3,
    )

    assert plan == NonlinearDomainDecompositionPlan(
        state_shape=(7, 3),
        axis=0,
        chunk_sizes=(3, 2, 2),
        halo=1,
    )
    assert plan.num_domains == 3
    assert plan.domain_size == 7
    assert plan.offsets == (0, 3, 5)
    assert plan.chunk_bounds == ((0, 3), (3, 5), (5, 7))
    assert plan.boundary_indices == (0, 2, 3, 4, 5, 6)
    assert plan.decomposition_metadata() == {
        "state_shape": (7, 3),
        "axis": 0,
        "chunk_sizes": (3, 2, 2),
        "halo": 1,
        "num_domains": 3,
        "domain_size": 7,
        "offsets": (0, 3, 5),
        "chunk_bounds": ((0, 3), (3, 5), (5, 7)),
        "boundary_indices": (0, 2, 3, 4, 5, 6),
    }
    assert plan.to_dict() == {
        "state_shape": (7, 3),
        "axis": 0,
        "chunk_sizes": (3, 2, 2),
        "halo": 1,
    }


def test_nonlinear_domain_identity_gate_enables_only_matching_decomposition() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    gated_state, report = nonlinear_domain_parallel_identity_gate(
        state,
        plan,
        dt=0.025,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    serial = local_stencil_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.025)
    decomposed = local_stencil_nonlinear_domain_decomposed_step(state, plan, dt=0.025)

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.gate_name == "nonlinear_domain_local_stencil_identity"
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.max_abs_error <= report.atol
    assert report.max_rel_error <= report.rtol
    assert report.boundary_indices == (0, 2, 3, 5)
    assert report.boundary_max_abs_error <= report.atol
    assert report.boundary_max_rel_error <= report.rtol
    assert "bounded local-stencil diagnostic" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)
    assert jnp.allclose(gated_state, decomposed, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_transport_window_gate_tracks_conservation_proxies() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    report = nonlinear_domain_transport_window_identity_gate(
        state,
        plan,
        dt=0.025,
        steps=5,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.gate_name == "nonlinear_domain_transport_window_identity"
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.steps == 5
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.boundary_indices == (0, 2, 3, 5)
    assert report.max_abs_state_error <= report.atol
    assert report.max_abs_boundary_error <= report.atol
    assert report.mass_trace_max_abs_error <= report.atol
    assert report.mass_trace_max_rel_error <= report.rtol
    assert report.free_energy_trace_max_abs_error <= report.atol
    assert report.free_energy_trace_max_rel_error <= report.rtol
    assert report.flux_proxy_trace_max_abs_error <= report.atol
    assert report.flux_proxy_trace_max_rel_error <= report.rtol
    assert len(report.serial_mass_trace) == report.steps + 1
    assert len(report.decomposed_free_energy_trace) == report.steps + 1
    assert report.serial_mass_drift == pytest.approx(report.decomposed_mass_drift)
    assert report.serial_free_energy_drift == pytest.approx(
        report.decomposed_free_energy_drift
    )
    assert "transport-window identity gate" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["identity_passed"] is True


def test_nonlinear_domain_transport_window_gate_fails_closed_on_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=0,
        chunk_sizes=(2, 2),
        halo=1,
    )

    report = nonlinear_domain_transport_window_identity_gate(
        state,
        invalid_plan,
        steps=2,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is False
    assert report.blocked_reasons == ("chunk_sizes_do_not_cover_axis",)
    assert report.max_abs_state_error == float("inf")
    assert report.mass_trace_max_abs_error == float("inf")


def test_nonlinear_domain_identity_report_fails_closed_on_mismatch() -> None:
    state = deterministic_nonlinear_domain_state((5, 3))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)
    serial = local_stencil_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    perturbed = serial.at[0, 0].add(1.0e-3)

    report = nonlinear_domain_identity_report(
        serial,
        perturbed,
        plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.max_abs_error > report.atol
    assert report.boundary_max_abs_error > report.atol
    assert report.to_dict()["identity_passed"] is False


def test_nonlinear_domain_identity_report_blocks_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=0,
        chunk_sizes=(2, 2),
        halo=1,
    )

    report = nonlinear_domain_identity_report(
        state,
        state,
        invalid_plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is False
    assert report.blocked_reasons == ("chunk_sizes_do_not_cover_axis",)


def test_nonlinear_domain_identity_report_blocks_shape_mismatch() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    candidate = deterministic_nonlinear_domain_state((5, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    report = nonlinear_domain_identity_report(
        state,
        candidate,
        plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is True
    assert report.blocked_reasons == ("decomposed_shape_does_not_match_serial",)
    assert report.max_abs_error == float("inf")
    assert report.max_rel_error == float("inf")


def test_nonlinear_domain_decomposed_step_is_jax_jittable_with_static_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, axis=0, num_domains=3)

    serial = local_stencil_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    jit_step = jax.jit(
        lambda item: local_stencil_nonlinear_domain_decomposed_step(item, plan, dt=0.05)
    )
    decomposed = jit_step(state)

    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_plan_rejects_unsupported_or_invalid_domains() -> None:
    with pytest.raises(ValueError, match="num_domains cannot exceed"):
        build_nonlinear_domain_decomposition_plan((2, 4), num_domains=3)
    with pytest.raises(ValueError, match="one-cell halo"):
        build_nonlinear_domain_decomposition_plan((4, 4), halo=2)
    with pytest.raises(ValueError, match="non-empty"):
        build_nonlinear_domain_decomposition_plan((0, 4))


def test_nonlinear_domain_decomposed_step_rejects_manual_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=-1,
        chunk_sizes=(3, 3),
        halo=1,
    )

    with pytest.raises(ValueError, match="axis_not_canonical"):
        local_stencil_nonlinear_domain_decomposed_step(state, invalid_plan)


def test_nonlinear_domain_fail_closed_edge_branches() -> None:
    with pytest.raises(ValueError, match="state_shape"):
        build_nonlinear_domain_decomposition_plan(())
    with pytest.raises(ValueError, match="num_domains"):
        build_nonlinear_domain_decomposition_plan((4, 4), num_domains=0)
    with pytest.raises(ValueError, match="shape"):
        deterministic_nonlinear_domain_state(())
    with pytest.raises(ValueError, match="shape entries"):
        deterministic_nonlinear_domain_state((4, 0))

    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)
    with pytest.raises(ValueError, match="state shape"):
        local_stencil_nonlinear_domain_decomposed_step(state[:5], plan)
    with pytest.raises(ValueError, match="steps"):
        nonlinear_domain_transport_window_identity_gate(state, plan, steps=0)

    single_chunk_plan = build_nonlinear_domain_decomposition_plan(
        state.shape,
        num_domains=1,
    )
    mass, free_energy, flux_proxy = (
        nonlinear_parallel_domain._nonlinear_domain_transport_observables(
            state,
            single_chunk_plan,
        )
    )
    assert mass > 0.0
    assert free_energy > 0.0
    assert flux_proxy > 0.0
    assert nonlinear_parallel_domain._relative_trace_error(
        (1.0, 2.0),
        (1.0,),
        floor=1.0e-6,
    ) == (float("inf"), float("inf"))
    assert nonlinear_parallel_domain._trace_drift((1.0,)) == 0.0

    shape_mismatch_report = nonlinear_domain_transport_window_identity_gate(
        state[:5],
        plan,
        steps=1,
    )
    assert shape_mismatch_report.identity_passed is False
    assert shape_mismatch_report.blocked_reasons == ("state_shape_does_not_match_plan",)


# ---- test_nonlinear_parallel.py ----

import json

import numpy as np

import gkx
import gkx.operators.nonlinear.parallel_contracts_domain as nonlinear_parallel_contracts_domain
import gkx.operators.nonlinear.parallel_contracts_spectral as nonlinear_parallel_contracts_spectral
import gkx.operators.nonlinear.parallel_contracts_strategy as nonlinear_parallel_contracts_strategy
import gkx.operators.nonlinear.device_z as nonlinear_parallel_device_z
import gkx.operators.nonlinear.spectral_core as nonlinear_parallel_spectral_core
import gkx.operators.nonlinear.spectral_identity_integrator as spectral_identity_integrator
import gkx.operators.nonlinear.spectral_identity_reports as spectral_identity_reports
import gkx.operators.nonlinear.spectral_identity_rhs as spectral_identity_rhs
from gkx.operators.nonlinear.parallel import (
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
    NonlinearParallelStrategy,
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
    NonlinearSpectralRHSIdentityReport,
    classify_nonlinear_parallel_strategy,
    nonlinear_parallel_strategies,
    nonlinear_parallel_strategy,
    release_ready_nonlinear_parallel_strategies,
)


def test_nonlinear_parallel_facade_reexports_device_z_core() -> None:
    assert (
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_rhs
        is nonlinear_parallel_device_z.device_z_pencil_nonlinear_spectral_rhs
    )
    assert (
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate
        is nonlinear_parallel_device_z.device_z_pencil_nonlinear_spectral_transport_window_identity_gate
    )


def test_nonlinear_parallel_facade_reexports_spectral_identity_core() -> None:
    assert (
        nonlinear_parallel.nonlinear_spectral_communication_identity_gate
        is spectral_identity_reports.nonlinear_spectral_communication_identity_gate
    )
    assert (
        nonlinear_parallel.nonlinear_spectral_rhs_identity_gate
        is spectral_identity_rhs.nonlinear_spectral_rhs_identity_gate
    )
    assert (
        nonlinear_parallel.integrate_logical_decomposed_nonlinear_spectral
        is spectral_identity_integrator.integrate_logical_decomposed_nonlinear_spectral
    )
    assert (
        nonlinear_parallel.nonlinear_spectral_pencil_transport_window_identity_gate
        is spectral_identity_integrator.nonlinear_spectral_pencil_transport_window_identity_gate
    )
    assert (
        nonlinear_parallel.nonlinear_spectral_pencil_rhs_identity_gate
        is spectral_identity_rhs.nonlinear_spectral_pencil_rhs_identity_gate
    )
    assert (
        nonlinear_parallel.pencil_decomposed_nonlinear_spectral_rhs
        is spectral_identity_rhs.pencil_decomposed_nonlinear_spectral_rhs
    )


def test_nonlinear_parallel_public_api_exports_are_stable() -> None:
    public_names = (
        "NonlinearDomainDecompositionPlan",
        "NonlinearDomainIdentityReport",
        "NonlinearDomainTransportWindowReport",
        "NonlinearParallelStrategy",
        "NonlinearSpectralCommunicationReport",
        "NonlinearSpectralDevicePencilFFTBatchModel",
        "NonlinearSpectralDevicePencilRHSIdentityReport",
        "NonlinearSpectralDevicePencilTransportWindowReport",
        "NonlinearSpectralDomainWorkModel",
        "NonlinearSpectralPencilRHSIdentityReport",
        "NonlinearSpectralPencilTransportWindowReport",
        "NonlinearSpectralPencilWorkModel",
        "NonlinearSpectralRHSIdentityReport",
        "build_nonlinear_domain_decomposition_plan",
        "classify_nonlinear_parallel_strategy",
        "deterministic_nonlinear_domain_state",
        "deterministic_nonlinear_spectral_state",
        "device_z_pencil_fft_batch_pressure_model",
        "device_z_pencil_nonlinear_spectral_rhs",
        "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
        "integrate_logical_decomposed_nonlinear_spectral",
        "nonlinear_domain_identity_report",
        "nonlinear_domain_parallel_identity_gate",
        "nonlinear_domain_transport_window_identity_gate",
        "nonlinear_parallel_strategies",
        "nonlinear_parallel_strategy",
        "nonlinear_spectral_communication_identity_gate",
        "nonlinear_spectral_communication_identity_report",
        "nonlinear_spectral_domain_work_model",
        "nonlinear_spectral_pencil_rhs_identity_gate",
        "nonlinear_spectral_pencil_transport_window_identity_gate",
        "nonlinear_spectral_pencil_work_model",
        "nonlinear_spectral_rhs_identity_gate",
        "nonlinear_spectral_rhs_identity_report",
        "pencil_decomposed_nonlinear_spectral_rhs",
        "local_stencil_nonlinear_domain_decomposed_step",
        "local_stencil_nonlinear_domain_serial_step",
        "release_ready_nonlinear_parallel_strategies",
    )

    assert set(public_names).isdisjoint(gkx.__all__)
    assert set(public_names) <= set(nonlinear_parallel.__all__)
    assert NonlinearParallelStrategy is nonlinear_parallel.NonlinearParallelStrategy
    assert (
        NonlinearDomainDecompositionPlan
        is nonlinear_parallel.NonlinearDomainDecompositionPlan
    )
    assert (
        NonlinearDomainIdentityReport
        is nonlinear_parallel.NonlinearDomainIdentityReport
    )
    assert (
        NonlinearDomainTransportWindowReport
        is nonlinear_parallel.NonlinearDomainTransportWindowReport
    )
    assert (
        NonlinearSpectralCommunicationReport
        is nonlinear_parallel.NonlinearSpectralCommunicationReport
    )
    assert (
        NonlinearSpectralDevicePencilFFTBatchModel
        is nonlinear_parallel.NonlinearSpectralDevicePencilFFTBatchModel
    )
    assert (
        NonlinearSpectralDevicePencilRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralDevicePencilRHSIdentityReport
    )
    assert (
        NonlinearSpectralDevicePencilTransportWindowReport
        is nonlinear_parallel.NonlinearSpectralDevicePencilTransportWindowReport
    )
    assert (
        NonlinearSpectralDomainWorkModel
        is nonlinear_parallel.NonlinearSpectralDomainWorkModel
    )
    assert (
        NonlinearSpectralPencilRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralPencilRHSIdentityReport
    )
    assert (
        NonlinearSpectralPencilTransportWindowReport
        is nonlinear_parallel.NonlinearSpectralPencilTransportWindowReport
    )
    assert (
        NonlinearSpectralPencilWorkModel
        is nonlinear_parallel.NonlinearSpectralPencilWorkModel
    )
    assert (
        NonlinearSpectralRHSIdentityReport
        is nonlinear_parallel.NonlinearSpectralRHSIdentityReport
    )
    for name in public_names:
        assert getattr(gkx, name) is getattr(nonlinear_parallel, name)


def test_nonlinear_parallel_facade_reexports_contract_objects() -> None:
    domain_names = (
        "NonlinearDomainDecompositionPlan",
        "NonlinearDomainIdentityReport",
        "NonlinearDomainTransportWindowReport",
    )
    spectral_names = (
        "NonlinearSpectralCommunicationReport",
        "NonlinearSpectralDevicePencilFFTBatchModel",
        "NonlinearSpectralDevicePencilRHSIdentityReport",
        "NonlinearSpectralDevicePencilTransportWindowReport",
        "NonlinearSpectralDomainWorkModel",
        "NonlinearSpectralIntegratorIdentityReport",
        "NonlinearSpectralPencilRHSIdentityReport",
        "NonlinearSpectralPencilTransportWindowReport",
        "NonlinearSpectralPencilWorkModel",
        "NonlinearSpectralRHSIdentityReport",
    )
    strategy_names = (
        "NonlinearParallelStrategy",
        "ParallelReadiness",
    )

    for name in domain_names:
        assert getattr(nonlinear_parallel, name) is getattr(
            nonlinear_parallel_contracts_domain,
            name,
        )
    for name in spectral_names:
        assert getattr(nonlinear_parallel, name) is getattr(
            nonlinear_parallel_contracts_spectral,
            name,
        )
    for name in strategy_names:
        assert getattr(nonlinear_parallel, name) is getattr(
            nonlinear_parallel_contracts_strategy,
            name,
        )


def test_nonlinear_parallel_facade_reexports_public_spectral_helpers() -> None:
    spectral_core_names = (
        "deterministic_nonlinear_spectral_state",
        "device_z_pencil_fft_batch_pressure_model",
        "nonlinear_spectral_domain_work_model",
        "nonlinear_spectral_pencil_work_model",
    )

    for name in spectral_core_names:
        assert getattr(nonlinear_parallel, name) is getattr(
            nonlinear_parallel_spectral_core,
            name,
        )


def test_nonlinear_parallel_contract_blockers_and_nested_reports() -> None:
    invalid_plan = nonlinear_parallel_contracts_domain.NonlinearDomainDecompositionPlan(
        state_shape=(),
        axis=2,
        chunk_sizes=(),
        halo=2,
    )

    assert invalid_plan.boundary_indices == ()
    assert nonlinear_parallel_contracts_domain._nonlinear_domain_plan_blockers(
        invalid_plan
    ) == (
        "state_shape_empty",
        "axis_not_canonical",
        "unsupported_halo",
        "chunk_sizes_empty",
    )

    non_positive_plan = (
        nonlinear_parallel_contracts_domain.NonlinearDomainDecompositionPlan(
            state_shape=(0, 4),
            axis=0,
            chunk_sizes=(0, 1),
        )
    )
    assert non_positive_plan.boundary_indices == ()
    assert set(
        nonlinear_parallel_contracts_domain._nonlinear_domain_plan_blockers(
            non_positive_plan
        )
    ) >= {"state_shape_non_positive", "chunk_size_non_positive"}
    assert (
        "serial_shape_does_not_match_plan"
        in nonlinear_parallel_contracts_domain._nonlinear_domain_identity_blockers(
            jnp.zeros((2, 4)),
            jnp.zeros((2, 3)),
            non_positive_plan,
        )
    )

    work_model = nonlinear_parallel.nonlinear_spectral_pencil_work_model(
        (2, 4, 16, 16, 2),
        y_chunks=(8, 8),
        x_chunks=(8, 8),
    )
    rhs_report = (
        nonlinear_parallel_contracts_spectral.NonlinearSpectralPencilRHSIdentityReport(
            state_shape=work_model.state_shape,
            y_chunks=work_model.y_chunks,
            x_chunks=work_model.x_chunks,
            y_offsets=work_model.y_offsets,
            x_offsets=work_model.x_offsets,
            atol=1.0e-8,
            rtol=1.0e-8,
            field_max_abs_error=0.0,
            field_max_rel_error=0.0,
            bracket_max_abs_error=0.0,
            bracket_max_rel_error=0.0,
            rhs_max_abs_error=0.0,
            rhs_max_rel_error=0.0,
            identity_passed=True,
            decomposed_path_enabled=True,
            work_model=work_model,
            claim_scope="unit nested report",
        )
    )
    transport_report = nonlinear_parallel_contracts_spectral.NonlinearSpectralPencilTransportWindowReport(
        state_shape=work_model.state_shape,
        y_chunks=work_model.y_chunks,
        x_chunks=work_model.x_chunks,
        y_offsets=work_model.y_offsets,
        x_offsets=work_model.x_offsets,
        steps=2,
        dt=0.01,
        atol=1.0e-8,
        rtol=1.0e-8,
        final_state_max_abs_error=0.0,
        final_state_max_rel_error=0.0,
        free_energy_trace_max_abs_error=0.0,
        free_energy_trace_max_rel_error=0.0,
        field_energy_trace_max_abs_error=0.0,
        field_energy_trace_max_rel_error=0.0,
        physical_flux_trace_max_abs_error=0.0,
        physical_flux_trace_max_rel_error=0.0,
        bracket_rms_trace_max_abs_error=0.0,
        bracket_rms_trace_max_rel_error=0.0,
        serial_free_energy_drift=0.0,
        pencil_free_energy_drift=0.0,
        identity_passed=True,
        decomposed_path_enabled=True,
        work_model=work_model,
        claim_scope="unit nested transport report",
    )
    device_rhs_report = nonlinear_parallel_contracts_spectral.NonlinearSpectralDevicePencilRHSIdentityReport(
        state_shape=work_model.state_shape,
        sharded_axis="z",
        axis_name="z",
        requested_device_count=2,
        active_device_count=0,
        atol=1.0e-8,
        rtol=1.0e-8,
        rhs_max_abs_error=float("inf"),
        rhs_max_rel_error=float("inf"),
        identity_passed=False,
        device_sharding_active=False,
        decomposed_path_enabled=False,
        claim_scope="unit device report",
        blocked_reasons=("requires_at_least_two_devices",),
    )
    device_transport_report = nonlinear_parallel_contracts_spectral.NonlinearSpectralDevicePencilTransportWindowReport(
        state_shape=work_model.state_shape,
        sharded_axis="z",
        axis_name="z",
        requested_device_count=2,
        active_device_count=0,
        steps=2,
        dt=0.01,
        atol=1.0e-8,
        rtol=1.0e-8,
        final_state_max_abs_error=float("inf"),
        final_state_max_rel_error=float("inf"),
        free_energy_trace_max_abs_error=float("inf"),
        free_energy_trace_max_rel_error=float("inf"),
        field_energy_trace_max_abs_error=float("inf"),
        field_energy_trace_max_rel_error=float("inf"),
        physical_flux_trace_max_abs_error=float("inf"),
        physical_flux_trace_max_rel_error=float("inf"),
        bracket_rms_trace_max_abs_error=float("inf"),
        bracket_rms_trace_max_rel_error=float("inf"),
        serial_free_energy_drift=0.0,
        device_free_energy_drift=0.0,
        identity_passed=False,
        device_sharding_active=False,
        decomposed_path_enabled=False,
        claim_scope="unit device transport report",
    )
    batch_model = nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
        work_model.state_shape,
        device_count=2,
    )
    blocked_strategy = nonlinear_parallel_contracts_strategy.NonlinearParallelStrategy(
        name="whole_state_kx_ky",
        readiness="blocked",
        independent_work=False,
        changes_solver_layout=True,
        identity_gates=("unit_identity",),
        physics_gates=("unit_physics",),
        profiler_gates=("unit_profile",),
        notes="unit blocked strategy",
    )

    assert rhs_report.to_dict()["work_model"]["state_shape"] == work_model.state_shape
    assert (
        transport_report.to_dict()["work_model"]["state_shape"]
        == work_model.state_shape
    )
    assert device_rhs_report.to_dict()["blocked_reasons"] == (
        "requires_at_least_two_devices",
    )
    assert device_transport_report.to_dict()["requested_device_count"] == 2
    assert batch_model.to_dict()["device_count"] == 2
    assert blocked_strategy.blocked is True


def test_fft_axis_domain_sharding_is_diagnostic_until_runtime_fft_gates_exist() -> None:
    strategy = nonlinear_parallel_strategy("fft_axis_domain")

    assert strategy.diagnostic_only is True
    assert strategy.release_ready is False
    assert strategy.readiness == "diagnostic"
    assert "distributed_fft_forward_inverse_identity" in strategy.identity_gates
    assert "distributed_fft_nonlinear_bracket_identity" in strategy.identity_gates
    assert "distributed_fft_field_solve_identity" in strategy.identity_gates
    assert "pencil_fft_fused_nonlinear_rhs_identity" in strategy.identity_gates
    assert "pencil_fft_physical_transport_window_identity" in strategy.identity_gates
    assert "device_z_pencil_fused_nonlinear_rhs_identity" in strategy.identity_gates
    assert (
        "device_z_pencil_physical_transport_window_identity" in strategy.identity_gates
    )
    assert "pencil-FFT fused-bracket identity" in strategy.notes
    assert "device-level transport-window routing" in strategy.notes


def test_logical_decomposed_spectral_integrator_routes_after_identity_gate() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 5))

    routed, report = nonlinear_parallel.integrate_logical_decomposed_nonlinear_spectral(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        dt=0.0025,
        steps=3,
        atol=5.0e-6,
        rtol=5.0e-6,
    )
    serial = state
    dt = jnp.asarray(0.0025, dtype=jnp.real(state).dtype)
    for _ in range(3):
        _field, _bracket, rhs = (
            nonlinear_parallel_spectral_core._serial_nonlinear_spectral_rhs(serial)
        )
        serial = serial + dt * rhs

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert routed.shape == state.shape
    assert jnp.allclose(routed, serial, atol=5.0e-6, rtol=5.0e-6)
    assert "no production distributed FFT" in report.claim_scope


def test_nonlinear_spectral_domain_work_model_blocks_global_reconstruction_speedup() -> (
    None
):
    model = nonlinear_parallel.nonlinear_spectral_domain_work_model(
        (2, 4, 32, 32, 4),
        y_chunks=(16, 16),
        x_chunks=(16, 16),
    )

    assert isinstance(model, NonlinearSpectralDomainWorkModel)
    assert model.num_tiles == 4
    assert model.state_elements == 32768
    assert model.field_elements == 4096
    assert model.owned_state_elements_per_step == model.state_elements
    assert model.state_allgather_elements_per_step == 3 * model.state_elements
    assert model.bracket_allgather_elements_per_step == 3 * model.state_elements
    assert model.field_broadcast_elements_per_step == 3 * model.field_elements
    assert model.total_communication_elements_per_step == 208896
    assert model.communication_to_owned_work_ratio == pytest.approx(6.375)
    assert model.parallel_efficiency_ceiling == pytest.approx(1.0 / 7.375)
    assert model.production_speedup_feasible is False
    assert model.feasibility_blockers == (
        "global_reconstruction_communication_dominates_owned_work",
    )
    assert "not a distributed FFT performance claim" in model.claim_scope
    assert model.to_dict()["production_speedup_feasible"] is False


def test_nonlinear_spectral_pencil_work_model_gates_plausible_scaling() -> None:
    model = nonlinear_parallel.nonlinear_spectral_pencil_work_model(
        (2, 4, 32, 32, 4),
        y_chunks=(16, 16),
        x_chunks=(16, 16),
    )

    assert isinstance(model, NonlinearSpectralPencilWorkModel)
    assert model.num_tiles == 4
    assert model.global_reconstruction_elements_per_step == 0
    assert (
        model.transform_payload_elements_per_step
        == 3 * model.state_elements + 2 * model.field_elements
    )
    assert model.pencil_transpose_elements_per_step > 0
    assert (
        model.communication_to_fft_work_ratio
        < model.max_communication_to_fft_work_ratio
    )
    assert model.predicted_speedup_ceiling >= model.min_predicted_speedup
    assert model.production_speedup_feasible is True
    assert model.feasibility_blockers == ()
    assert "no global reconstruction" in model.claim_scope

    blocked = nonlinear_parallel.nonlinear_spectral_pencil_work_model(
        (2, 4, 8, 8, 2),
        y_chunks=(8,),
        x_chunks=(8,),
    )
    assert blocked.production_speedup_feasible is False
    assert "single_tile_no_domain_decomposition" in blocked.feasibility_blockers


def test_device_z_pencil_fft_batch_pressure_model_suggests_profile_chunking() -> None:
    model = nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
        (4, 16, 96, 96, 64),
        device_count=2,
        max_fft_batch_count=65_536,
    )

    assert isinstance(model, NonlinearSpectralDevicePencilFFTBatchModel)
    assert model.local_z_extent == 32
    assert model.unchunked_fft_batch_count == 4 * 16 * 96 * 32
    assert model.chunking_required is True
    assert model.suggested_z_chunk_size == 8
    assert model.effective_z_chunk_size == 8
    assert model.chunked_fft_batch_count == 4 * 16 * 96 * 8
    assert model.chunking_active is True
    assert model.disable_gpu_preallocation_recommended is True
    assert model.profiling_candidate is True
    assert model.feasibility_blockers == ()
    assert "not constitute a nonlinear speedup claim" in model.claim_scope
    assert model.to_dict()["suggested_z_chunk_size"] == 8

    already_safe = nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
        (2, 3, 6, 4, 2),
        device_count=2,
    )
    assert already_safe.chunking_required is False
    assert already_safe.suggested_z_chunk_size is None
    assert already_safe.effective_z_chunk_size == 1


def test_device_z_pencil_fft_batch_pressure_model_fails_closed() -> None:
    blocked = nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
        (4, 16, 128, 128, 3),
        device_count=2,
    )
    assert blocked.profiling_candidate is False
    assert "z_extent_not_divisible_by_device_count" in blocked.feasibility_blockers

    too_large = nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
        (8, 32, 256, 256, 4),
        device_count=2,
        max_fft_batch_count=1_024,
    )
    assert too_large.suggested_z_chunk_size == 1
    assert "fft_batch_pressure_exceeds_single_z_plane" in too_large.feasibility_blockers
    assert "z_chunk_size_still_exceeds_fft_batch_cap" in too_large.feasibility_blockers

    with pytest.raises(ValueError, match="device_count"):
        nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
            (2, 3, 6, 4, 2),
            device_count=0,
        )
    with pytest.raises(ValueError, match="max_fft_batch_count"):
        nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
            (2, 3, 6, 4, 2),
            device_count=1,
            max_fft_batch_count=0,
        )
    with pytest.raises(ValueError, match="z_chunk_size"):
        nonlinear_parallel.device_z_pencil_fft_batch_pressure_model(
            (2, 3, 6, 4, 2),
            device_count=1,
            z_chunk_size=0,
        )


def test_spectral_physical_observable_sums_are_z_additive() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 4))
    _field, bracket, _rhs = (
        nonlinear_parallel_spectral_core._serial_nonlinear_spectral_rhs(state)
    )

    whole = nonlinear_parallel_device_z._spectral_physical_transport_observable_vector_from_sums(
        nonlinear_parallel_device_z._spectral_physical_transport_observable_sums(
            state, bracket
        )
    )
    split = nonlinear_parallel_device_z._spectral_physical_transport_observable_sums(
        state[..., :2],
        bracket[..., :2],
    ) + nonlinear_parallel_device_z._spectral_physical_transport_observable_sums(
        state[..., 2:],
        bracket[..., 2:],
    )
    reassembled = nonlinear_parallel_device_z._spectral_physical_transport_observable_vector_from_sums(
        split
    )

    np.testing.assert_allclose(np.asarray(reassembled), np.asarray(whole), rtol=1.0e-6)


_MIXED_RADIX_OBSERVABLE_SHAPE = (4, 16, 96, 96, 16)


def _mixed_radix_observable_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return serial-route and sharded-route observables at Ny = Nx = 96.

    The serial reference reduces already materialized arrays op by op, while
    the sharded reducer runs inside one jit, so its reduction is fused into its
    elementwise producer. Reproduce both shapes of that computation without
    devices: eager over the whole state versus jitted per z half, added.
    """

    state = nonlinear_parallel.deterministic_nonlinear_spectral_state(
        _MIXED_RADIX_OBSERVABLE_SHAPE
    )
    assert state.dtype == jnp.complex64

    def _fused_sums(local_state: jax.Array) -> jax.Array:
        _field, bracket, _rhs = (
            nonlinear_parallel_spectral_core._pencil_nonlinear_spectral_rhs(local_state)
        )
        return nonlinear_parallel_device_z._spectral_physical_transport_observable_sums(
            local_state,
            bracket,
        )

    _serial_field, serial_bracket, _serial_rhs = (
        nonlinear_parallel_spectral_core._serial_nonlinear_spectral_rhs(state)
    )
    serial = nonlinear_parallel_device_z._spectral_physical_transport_observable_vector_from_sums(
        nonlinear_parallel_device_z._spectral_physical_transport_observable_sums(
            state,
            serial_bracket,
        )
    )
    half = _MIXED_RADIX_OBSERVABLE_SHAPE[-1] // 2
    fused = jax.jit(_fused_sums)
    sharded = nonlinear_parallel_device_z._spectral_physical_transport_observable_vector_from_sums(
        fused(state[..., :half]) + fused(state[..., half:])
    )
    return np.asarray(serial), np.asarray(sharded)


def test_observable_sums_survive_fused_reduction_at_a_mixed_radix_grid() -> None:
    """Pin the single-precision observable reduction at Ny = Nx = 96 = 2**5 * 3.

    ``jnp.sum`` was not enough here. XLA lowers a reduction over a materialized
    array to a blocked vector reduction, but lowers the same reduction fused
    into its elementwise producer to a far longer accumulation chain, and this
    observable adds about 1e7 terms that are each around 5e-8 of the total. In
    single precision the longer chain absorbed roughly 1e-4 of the bracket
    observable, which is the whole identity budget, and it did so only at the
    grids whose shape produced the worse lowering -- 96 failed while both
    neighbouring powers of two passed, which reads like an FFT defect and is
    not one. Compensating the reduction removes the dependence on the lowering,
    so assert well inside the gate tolerance rather than at it.
    """

    serial, sharded = _mixed_radix_observable_vectors()

    np.testing.assert_allclose(sharded, serial, rtol=1.0e-6)


def test_device_z_sharded_reduce_transport_window_holds_at_a_mixed_radix_grid() -> None:
    """Pin the reported failure: ``sharded_reduce`` at 96 in single precision."""

    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local JAX devices")
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state(
        _MIXED_RADIX_OBSERVABLE_SHAPE
    )
    assert state.dtype == jnp.complex64

    report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=jax.devices()[:2],
        steps=2,
        dt=0.0025,
        observable_mode="sharded_reduce",
    )

    assert report.identity_passed is True
    assert report.blocked_reasons == ()
    assert report.final_state_max_abs_error <= report.atol
    # Relative, not absolute: the bracket observable is small enough that the
    # absolute floor alone would pass a reduction that is 1e-3 wrong.
    assert report.free_energy_trace_max_rel_error <= report.rtol
    assert report.field_energy_trace_max_rel_error <= report.rtol
    assert report.physical_flux_trace_max_rel_error <= report.rtol
    assert report.bracket_rms_trace_max_rel_error <= report.rtol


def test_device_z_transport_trace_helpers_build_fail_closed_reports() -> None:
    traces = nonlinear_parallel_device_z._new_transport_trace_dict()
    assert set(traces) == {
        "free_energy",
        "field_energy",
        "physical_flux",
        "bracket_rms",
    }
    traces["free_energy"].extend([2.0, 3.0])
    traces["field_energy"].append(0.5)
    traces["physical_flux"].append(0.25)
    traces["bracket_rms"].append(0.125)

    frozen = nonlinear_parallel_device_z._transport_trace_tuples(traces)
    assert frozen["free_energy"] == (2.0, 3.0)

    errors = nonlinear_parallel_device_z._transport_trace_error_pairs(
        frozen,
        frozen,
        floor=1.0e-6,
    )
    assert errors["free_energy"] == (0.0, 0.0)
    assert nonlinear_parallel_device_z._device_z_transport_identity_passed(
        state_abs=2.0e-7,
        state_rel=1.0e-2,
        trace_errors=errors,
        atol=1.0e-6,
        rtol=1.0e-5,
    )
    failing_errors = dict(errors)
    failing_errors["bracket_rms"] = (2.0e-4, 2.0e-3)
    assert not nonlinear_parallel_device_z._device_z_transport_identity_passed(
        state_abs=2.0e-7,
        state_rel=1.0e-2,
        trace_errors=failing_errors,
        atol=1.0e-6,
        rtol=1.0e-5,
    )

    report = nonlinear_parallel_device_z._blocked_device_z_transport_window_report(
        state_shape=(2, 3, 6, 4, 2),
        axis_name="z",
        requested_count=1,
        active_count=0,
        steps=2,
        dt=0.01,
        atol=1.0e-6,
        rtol=1.0e-5,
        blocked_reasons=("requires_at_least_two_devices",),
        serial_traces=traces,
    )
    assert report.identity_passed is False
    assert report.device_sharding_active is False
    assert report.decomposed_path_enabled is False
    assert report.final_state_max_abs_error == float("inf")
    assert report.serial_free_energy_trace == (2.0, 3.0)
    assert report.device_free_energy_trace == ()
    assert report.blocked_reasons == ("requires_at_least_two_devices",)


def test_device_z_observable_appender_routes_host_and_sharded_reduce() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((1, 1, 4, 4, 2))

    host_traces = nonlinear_parallel_device_z._new_transport_trace_dict()
    nonlinear_parallel_device_z._append_device_z_transport_observables(
        host_traces,
        state,
        observable_mode="host_gather",
        sharded_observables_fn=None,
    )
    assert all(len(values) == 1 for values in host_traces.values())
    assert all(np.isfinite(values[0]) for values in host_traces.values())

    reduced_traces = nonlinear_parallel_device_z._new_transport_trace_dict()
    nonlinear_parallel_device_z._append_device_z_transport_observables(
        reduced_traces,
        state,
        observable_mode="sharded_reduce",
        sharded_observables_fn=lambda _state: jnp.asarray([1.0, 2.0, 3.0, 4.0]),
    )
    assert reduced_traces == {
        "free_energy": [1.0],
        "field_energy": [2.0],
        "physical_flux": [3.0],
        "bracket_rms": [4.0],
    }

    with pytest.raises(ValueError, match="sharded observable reducer"):
        nonlinear_parallel_device_z._append_device_z_transport_observables(
            reduced_traces,
            state,
            observable_mode="sharded_reduce",
            sharded_observables_fn=None,
        )


def test_device_z_rhs_report_helpers_classify_identity_gates() -> None:
    blocked = nonlinear_parallel_device_z._blocked_device_z_rhs_report(
        state_shape=(2, 3, 6, 4, 2),
        axis_name="zdev",
        requested_count=1,
        active_count=0,
        atol=1.0e-6,
        rtol=1.0e-4,
        blocked_reasons=("requires_at_least_two_devices",),
    )

    assert blocked.identity_passed is False
    assert blocked.device_sharding_active is False
    assert blocked.decomposed_path_enabled is False
    assert blocked.axis_name == "zdev"
    assert blocked.blocked_reasons == ("requires_at_least_two_devices",)
    assert np.isinf(blocked.rhs_max_abs_error)

    passed = nonlinear_parallel_device_z._device_z_rhs_identity_report(
        state_shape=(2, 3, 6, 4, 2),
        axis_name="z",
        requested_count=2,
        active_count=2,
        atol=1.0e-6,
        rtol=1.0e-4,
        rhs_abs=5.0e-7,
        rhs_rel=2.0e-5,
    )
    assert passed.identity_passed is True
    assert passed.device_sharding_active is True
    assert passed.decomposed_path_enabled is True
    assert passed.blocked_reasons == ()

    failed = nonlinear_parallel_device_z._device_z_rhs_identity_report(
        state_shape=(2, 3, 6, 4, 2),
        axis_name="z",
        requested_count=2,
        active_count=2,
        atol=1.0e-6,
        rtol=1.0e-4,
        rhs_abs=2.0e-6,
        rhs_rel=2.0e-5,
    )
    assert failed.identity_passed is False
    assert failed.decomposed_path_enabled is False
    assert failed.blocked_reasons == ("device_z_pencil_rhs_identity_failed",)


def test_device_z_transport_window_observable_mode_fails_closed() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((1, 1, 4, 4, 2))

    with pytest.raises(ValueError, match="observable_mode"):
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
            state,
            devices=[jax.devices()[0]],
            observable_mode="device_get",  # type: ignore[arg-type]
        )


def test_pencil_fft_route_matches_serial_fft_and_rhs_without_reconstruction() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_parallel.nonlinear_spectral_pencil_rhs_identity_gate(
        state,
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    assert isinstance(report, NonlinearSpectralPencilRHSIdentityReport)
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.work_model.global_reconstruction_elements_per_step == 0
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol

    routed_rhs, routed_report = (
        nonlinear_parallel.pencil_decomposed_nonlinear_spectral_rhs(
            state,
            y_chunks=(3, 3),
            x_chunks=(2, 2),
            atol=5.0e-6,
            rtol=1.0e-5,
        )
    )
    _field, _bracket, serial_rhs = (
        nonlinear_parallel_spectral_core._serial_nonlinear_spectral_rhs(state)
    )
    assert routed_report.identity_passed is True
    assert routed_report.decomposed_path_enabled is True
    assert jnp.allclose(routed_rhs, serial_rhs, atol=5.0e-6, rtol=5.0e-6)


def test_z_chunked_pencil_bracket_matches_unchunked_route() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 5))
    field = nonlinear_parallel_spectral_core._field_from_state(state)

    unchunked = nonlinear_parallel_spectral_core._pencil_spectral_bracket(state, field)
    chunked = nonlinear_parallel_spectral_core._pencil_spectral_bracket_z_chunked(
        state,
        field,
        z_chunk_size=2,
    )
    chunked_rhs = (
        nonlinear_parallel_spectral_core._pencil_nonlinear_spectral_rhs_z_chunked(
            state,
            z_chunk_size=2,
        )[2]
    )
    unchunked_rhs = nonlinear_parallel_spectral_core._pencil_nonlinear_spectral_rhs(
        state
    )[2]

    assert jnp.allclose(chunked, unchunked, atol=5.0e-6, rtol=1.0e-5)
    assert jnp.allclose(chunked_rhs, unchunked_rhs, atol=5.0e-6, rtol=1.0e-5)
    assert jnp.allclose(
        nonlinear_parallel_spectral_core._pencil_spectral_bracket_z_chunked(
            state,
            field,
            z_chunk_size=99,
        ),
        unchunked,
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    with pytest.raises(ValueError, match="z_chunk_size must be at least one"):
        nonlinear_parallel_spectral_core._pencil_spectral_bracket_z_chunked(
            state,
            field,
            z_chunk_size=0,
        )


def test_device_z_pencil_route_fails_closed_without_two_devices() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    routed_rhs, report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_rhs(
        state,
        devices=(),
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    _field, _bracket, serial_rhs = (
        nonlinear_parallel_spectral_core._serial_nonlinear_spectral_rhs(state)
    )

    assert isinstance(report, NonlinearSpectralDevicePencilRHSIdentityReport)
    assert report.identity_passed is False
    assert report.device_sharding_active is False
    assert report.decomposed_path_enabled is False
    assert report.blocked_reasons == ("requires_at_least_two_devices",)
    assert jnp.allclose(routed_rhs, serial_rhs, atol=0.0, rtol=0.0)

    transport_report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=(),
        steps=2,
        atol=5.0e-6,
        rtol=1.0e-5,
    )
    assert isinstance(
        transport_report, NonlinearSpectralDevicePencilTransportWindowReport
    )
    assert transport_report.identity_passed is False
    assert transport_report.device_sharding_active is False
    assert transport_report.decomposed_path_enabled is False
    assert transport_report.blocked_reasons == ("requires_at_least_two_devices",)
    assert "cannot support the z shard" in transport_report.claim_scope


def test_host_staged_array_for_sharding_materializes_numpy_copy() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    staged = nonlinear_parallel_spectral_core._host_staged_array_for_sharding(state)

    assert isinstance(staged, np.ndarray)
    assert staged.shape == state.shape
    assert np.allclose(staged, np.asarray(jax.device_get(state)))


def test_abs_or_rel_tolerance_policy_matches_allclose_style_gate() -> None:
    assert nonlinear_parallel_spectral_core._within_abs_or_rel_tolerance(
        4.0e-6,
        2.0e-2,
        atol=5.0e-6,
        rtol=1.0e-4,
    )
    assert nonlinear_parallel_spectral_core._within_abs_or_rel_tolerance(
        1.0e-3,
        4.0e-7,
        atol=5.0e-6,
        rtol=1.0e-4,
    )
    assert not nonlinear_parallel_spectral_core._within_abs_or_rel_tolerance(
        1.0e-3,
        1.0e-2,
        atol=5.0e-6,
        rtol=1.0e-4,
    )


def test_device_z_transport_window_rejects_invalid_step_count() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    with pytest.raises(ValueError, match="steps must be at least one"):
        nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
            state,
            steps=0,
        )


def test_device_z_transport_window_blocks_nondivisible_z_extent() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 3))

    report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=(jax.devices()[0], jax.devices()[0]),
        steps=2,
    )

    assert report.identity_passed is False
    assert report.device_sharding_active is False
    assert report.active_device_count == 0
    assert report.requested_device_count == 2
    assert report.blocked_reasons == ("z_extent_not_divisible_by_device_count",)
    assert report.final_state_max_abs_error == float("inf")


def test_device_z_pencil_transport_window_identity_on_available_devices() -> None:
    if jax.local_device_count() < 2:
        pytest.skip("requires at least two local JAX devices")
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_parallel.device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=jax.devices()[:2],
        steps=2,
        dt=0.0025,
        atol=5.0e-6,
        rtol=1.0e-4,
    )

    assert report.identity_passed is True
    assert report.device_sharding_active is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.final_state_max_rel_error <= report.rtol
    assert report.physical_flux_trace_max_abs_error <= report.atol
    assert report.bracket_rms_trace_max_rel_error <= report.rtol
    assert len(report.serial_free_energy_trace) == report.steps + 1
    assert len(report.device_free_energy_trace) == report.steps + 1
    assert "transport-window identity gate" in report.claim_scope


def test_pencil_fft_physical_transport_window_identity_gate() -> None:
    state = nonlinear_parallel.deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = (
        nonlinear_parallel.nonlinear_spectral_pencil_transport_window_identity_gate(
            state,
            y_chunks=(3, 3),
            x_chunks=(2, 2),
            dt=0.001,
            steps=3,
            atol=5.0e-6,
            rtol=1.0e-5,
        )
    )

    assert isinstance(report, NonlinearSpectralPencilTransportWindowReport)
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.physical_flux_trace_max_abs_error <= report.atol
    assert len(report.serial_physical_flux_trace) == 4
    assert len(report.pencil_physical_flux_trace) == 4
    assert max(report.serial_physical_flux_trace) > 0.0
    assert "not an absolute nonlinear turbulent heat-flux claim" in report.claim_scope


def test_whole_state_kx_ky_is_diagnostic_only() -> None:
    strategy = nonlinear_parallel_strategy("whole_state_kx_ky")

    assert classify_nonlinear_parallel_strategy("whole_state_kx_ky") == "diagnostic"
    assert strategy.diagnostic_only is True
    assert strategy.release_ready is False
    assert strategy.independent_work is False
    assert strategy.changes_solver_layout is True
    assert "whole_state_kx_ky_final_state_identity" in strategy.identity_gates
    assert "nonlinear_domain_transport_window_identity" in strategy.physics_gates
    assert "matched_cpu_gpu_whole_state_scaling_profile" in strategy.profiler_gates
    assert "not a speedup claim" in strategy.notes


def test_independent_ky_and_uq_are_release_ready_independent_work_paths() -> None:
    release_ready_strategies = release_ready_nonlinear_parallel_strategies()
    release_ready = {strategy.name: strategy for strategy in release_ready_strategies}

    assert [strategy.name for strategy in release_ready_strategies] == [
        "independent_ky_scan",
        "uq_ensemble",
    ]
    assert set(release_ready) == {"independent_ky_scan", "uq_ensemble"}
    for name in ("independent_ky_scan", "uq_ensemble"):
        strategy = release_ready[name]
        assert strategy.release_ready is True
        assert strategy.independent_work is True
        assert strategy.changes_solver_layout is False
        assert strategy.identity_gates
        assert strategy.physics_gates
        assert strategy.profiler_gates


def test_velocity_species_hermite_exposes_identity_physics_and_profiler_gates() -> None:
    strategy = nonlinear_parallel_strategy("velocity_species_hermite")

    assert strategy.diagnostic_only is True
    assert "hermite_ghost_exchange_identity" in strategy.identity_gates
    assert "velocity_field_reduce_broadcast_identity" in strategy.identity_gates
    assert "nonlinear_fixed_step_identity_gate" in strategy.physics_gates
    assert "matched_velocity_species_hermite_scaling_profile" in strategy.profiler_gates
    assert set(strategy.required_gates) == set(
        strategy.identity_gates + strategy.physics_gates + strategy.profiler_gates
    )


def test_strategy_registry_is_complete_json_friendly_and_rejects_unknown_names() -> (
    None
):
    strategies = nonlinear_parallel_strategies()
    by_name = {strategy.name: strategy for strategy in strategies}

    assert set(by_name) == {
        "independent_ky_scan",
        "uq_ensemble",
        "whole_state_kx_ky",
        "velocity_species_hermite",
        "fft_axis_domain",
    }
    strategy_table = [strategy.to_dict() for strategy in strategies]
    decoded_table = json.loads(json.dumps(strategy_table, sort_keys=True))

    assert [row["name"] for row in decoded_table] == [
        strategy.name for strategy in strategies
    ]
    assert decoded_table[0]["name"] == "independent_ky_scan"
    assert decoded_table[0]["readiness"] == "release_ready"
    for row in decoded_table:
        assert set(row) == {
            "name",
            "readiness",
            "independent_work",
            "changes_solver_layout",
            "identity_gates",
            "physics_gates",
            "profiler_gates",
            "notes",
        }
        assert isinstance(row["identity_gates"], list)
        assert isinstance(row["physics_gates"], list)
        assert isinstance(row["profiler_gates"], list)
    with pytest.raises(ValueError, match="unknown nonlinear parallelization strategy"):
        nonlinear_parallel_strategy("banana")  # type: ignore[arg-type]


def test_strategy_table_keeps_release_ready_paths_independent_and_layout_preserving() -> (
    None
):
    strategies = nonlinear_parallel_strategies()

    assert len({strategy.name for strategy in strategies}) == len(strategies)
    for strategy in strategies:
        assert strategy.required_gates
        assert len(strategy.required_gates) == len(set(strategy.required_gates))
        if strategy.release_ready:
            assert strategy.independent_work is True
            assert strategy.changes_solver_layout is False
            assert "speedup claim" not in strategy.notes.lower()
        else:
            assert strategy.changes_solver_layout is True
            assert strategy.name not in {
                item.name for item in release_ready_nonlinear_parallel_strategies()
            }


def test_strategy_policy_separates_identity_physics_and_profiler_gates() -> None:
    for strategy in nonlinear_parallel_strategies():
        assert all("identity" in gate for gate in strategy.identity_gates)
        assert all(
            gate not in strategy.identity_gates for gate in strategy.physics_gates
        )
        assert all(
            "profile" in gate or "scaling" in gate for gate in strategy.profiler_gates
        )

    whole_state = nonlinear_parallel_strategy("whole_state_kx_ky")
    assert whole_state.diagnostic_only
    assert "final_state" in " ".join(whole_state.identity_gates)
    assert "nonlinear_window" in " ".join(whole_state.physics_gates)
    assert "whole_state_scaling_profile" in " ".join(whole_state.profiler_gates)


# ---- test_nonlinear_spectral_communication_gate.py ----


from gkx.operators.nonlinear.parallel import (
    NonlinearSpectralIntegratorIdentityReport,
    deterministic_nonlinear_spectral_state,
    device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
    logical_decomposed_nonlinear_spectral_rhs,
    nonlinear_spectral_communication_identity_gate,
    nonlinear_spectral_communication_identity_report,
    nonlinear_spectral_integrator_identity_gate,
    nonlinear_spectral_rhs_identity_gate,
    nonlinear_spectral_rhs_identity_report,
)


def test_nonlinear_spectral_communication_gate_closes_fft_bracket_and_field_layouts() -> (
    None
):
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_communication_identity_gate(
        state,
        y_chunks=(2, 2, 2),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralCommunicationReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_chunks == (2, 2, 2)
    assert report.x_chunks == (2, 2)
    assert report.y_offsets == (0, 2, 4)
    assert report.x_offsets == (0, 2)
    assert report.blocked_reasons == ()
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.fft_max_abs_error <= report.atol
    assert report.fft_max_rel_error <= report.rtol
    assert report.bracket_max_abs_error <= report.atol
    assert report.bracket_max_rel_error <= report.rtol
    assert report.field_max_abs_error <= report.atol
    assert report.field_max_rel_error <= report.rtol
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["identity_passed"] is True


def test_nonlinear_spectral_communication_report_fails_closed_on_mismatch() -> None:
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    perturbed = reference.at[0, 0, 0, 0, 0].add(1.0e-3)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)

    report = nonlinear_spectral_communication_identity_report(
        reference,
        perturbed,
        reference,
        reference,
        field,
        field,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.fft_max_abs_error > report.atol
    assert report.blocked_reasons == ()


def test_nonlinear_spectral_communication_report_fails_closed_on_shape_blocker() -> (
    None
):
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    communicated = jnp.ones((2, 3, 5, 4, 2), dtype=jnp.complex64)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)

    report = nonlinear_spectral_communication_identity_report(
        reference,
        communicated,
        reference,
        reference,
        field,
        field,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.fft_max_abs_error == float("inf")
    assert report.fft_max_rel_error == float("inf")
    assert report.blocked_reasons == ("communicated_fft_roundtrip_shape_mismatch",)


def test_nonlinear_spectral_rhs_identity_gate_reconstructs_ordered_logical_tiles() -> (
    None
):
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_rhs_identity_gate(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralRHSIdentityReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_chunks == (2, 1, 3)
    assert report.x_chunks == (1, 3)
    assert report.y_offsets == (0, 2, 3)
    assert report.x_offsets == (0, 1)
    assert report.tile_bounds == (
        (0, 2, 0, 1),
        (0, 2, 1, 4),
        (2, 3, 0, 1),
        (2, 3, 1, 4),
        (3, 6, 0, 1),
        (3, 6, 1, 4),
    )
    assert report.blocked_reasons == ()
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.reconstruction_max_abs_error <= report.atol
    assert report.reconstruction_max_rel_error <= report.rtol
    assert report.field_max_abs_error <= report.atol
    assert report.field_max_rel_error <= report.rtol
    assert report.bracket_max_abs_error <= report.atol
    assert report.bracket_max_rel_error <= report.rtol
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol
    assert "existing bracket contribution" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["tile_bounds"] == report.tile_bounds


def test_logical_decomposed_nonlinear_spectral_rhs_returns_gated_rhs() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    rhs, report = logical_decomposed_nonlinear_spectral_rhs(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralRHSIdentityReport)
    assert rhs.shape == state.shape
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.rhs_max_abs_error <= report.atol
    assert report.rhs_max_rel_error <= report.rtol


def test_nonlinear_spectral_integrator_identity_gate_closes_multistep_route() -> None:
    state = deterministic_nonlinear_spectral_state((2, 3, 6, 4, 2))

    report = nonlinear_spectral_integrator_identity_gate(
        state,
        y_chunks=(2, 1, 3),
        x_chunks=(1, 3),
        steps=3,
        dt=0.0025,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert isinstance(report, NonlinearSpectralIntegratorIdentityReport)
    assert report.state_shape == (2, 3, 6, 4, 2)
    assert report.y_offsets == (0, 2, 3)
    assert report.x_offsets == (0, 1)
    assert report.steps == 3
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.final_state_max_rel_error <= report.rtol
    assert report.free_energy_trace_max_abs_error <= report.atol
    assert report.field_energy_trace_max_abs_error <= report.atol
    assert report.flux_proxy_trace_max_abs_error <= report.atol
    assert len(report.serial_free_energy_trace) == report.steps + 1
    assert (
        "no production distributed FFT routing or speedup claim" in report.claim_scope
    )
    assert report.to_dict()["identity_passed"] is True


def test_device_z_transport_window_long_compute_route_passes_when_devices_available() -> (
    None
):
    devices = tuple(jax.devices())
    if len(devices) < 2:
        pytest.skip("requires at least two local JAX devices")

    state = deterministic_nonlinear_spectral_state((2, 4, 12, 10, 4))
    report = device_z_pencil_nonlinear_spectral_transport_window_identity_gate(
        state,
        devices=devices[:2],
        steps=8,
        dt=0.0025,
        z_chunk_size=2,
        atol=5.0e-6,
        rtol=1.0e-4,
    )

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.final_state_max_abs_error <= report.atol
    assert report.physical_flux_trace_max_abs_error <= report.atol


def test_nonlinear_spectral_rhs_report_fails_closed_on_rhs_mismatch() -> None:
    reference = jnp.ones((2, 3, 6, 4, 2), dtype=jnp.complex64)
    field = jnp.ones((6, 4, 2), dtype=jnp.complex64)
    bracket = 0.25j * reference
    rhs = -bracket
    perturbed_rhs = rhs.at[0, 0, 0, 0, 0].add(1.0e-3)

    report = nonlinear_spectral_rhs_identity_report(
        reference,
        reference,
        field,
        field,
        bracket,
        bracket,
        rhs,
        perturbed_rhs,
        state_shape=(2, 3, 6, 4, 2),
        y_chunks=(3, 3),
        x_chunks=(2, 2),
        tile_bounds=(
            (0, 3, 0, 2),
            (0, 3, 2, 4),
            (3, 6, 0, 2),
            (3, 6, 2, 4),
        ),
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.blocked_reasons == ()
    assert report.rhs_max_abs_error > report.atol


def test_nonlinear_spectral_state_and_chunk_validation_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="spectral state shape"):
        deterministic_nonlinear_spectral_state((2, 3, 4, 5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Ny and Nx >= 2"):
        deterministic_nonlinear_spectral_state((1, 1, 1, 4, 1))

    state = deterministic_nonlinear_spectral_state((2, 2, 6, 4, 1))
    with pytest.raises(ValueError, match="y_chunks must sum"):
        nonlinear_spectral_communication_identity_gate(
            state, y_chunks=(2, 2), x_chunks=(2, 2)
        )
    with pytest.raises(ValueError, match="x_chunks entries must be positive"):
        nonlinear_spectral_communication_identity_gate(
            state, y_chunks=(3, 3), x_chunks=(4, 0)
        )
    with pytest.raises(ValueError, match="y_chunks must sum"):
        nonlinear_spectral_rhs_identity_gate(state, y_chunks=(2, 2), x_chunks=(2, 2))
    with pytest.raises(ValueError, match="steps must be at least one"):
        nonlinear_spectral_integrator_identity_gate(
            state, y_chunks=(3, 3), x_chunks=(2, 2), steps=0
        )
