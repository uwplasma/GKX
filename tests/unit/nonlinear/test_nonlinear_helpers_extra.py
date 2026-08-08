from __future__ import annotations

from dataclasses import fields as dataclass_fields
import gc
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.solvers.nonlinear import (
    state_integration as nonlinear_state_integration_mod,
)
from gkx.operators.nonlinear import projection as nonlinear_projection
from gkx.config import CycloneBaseCase, GridConfig
from gkx.diagnostics import ResolvedDiagnostics
from gkx.diagnostics.transport import heat_flux_species
from gkx.diagnostics.moments import fieldline_quadrature_weights
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.operators.linear.cache_builder import (
    update_linear_cache_for_sheared_kx,
)
from gkx.operators.nonlinear.diagnostics import build_nonlinear_simulation_diagnostics, finalize_nonlinear_scan_diagnostics, maybe_emit_nonlinear_progress, run_sampled_explicit_diagnostic_scan, sampled_scan_intervals, select_nonlinear_step_diagnostics
from gkx.operators.nonlinear.policies import build_nonlinear_collision_split_policy, build_nonlinear_diagnostic_setup, build_nonlinear_imex_operator, build_nonlinear_time_step_policy
from gkx.solvers.nonlinear.diagnostic_integration import integrate_nonlinear_explicit_diagnostics, integrate_nonlinear_explicit_diagnostics_state, integrate_nonlinear_imex_diagnostics
from gkx.solvers.nonlinear.state_integration import integrate_nonlinear, integrate_nonlinear_cached, integrate_nonlinear_imex_cached, integrate_nonlinear_sheared, integrate_nonlinear_sheared_transport, nonlinear_rhs_cached
from gkx.operators.nonlinear.collisions import (
    _apply_collision_split,
    _collision_damping,
)
from gkx.operators.nonlinear.diagnostics import (
    _pack_resolved_diagnostics,
    _sample_indices_with_final,
)
from gkx.operators.nonlinear.policies import (
    _diagnostic_omega_mode_mask,
    _nonlinear_cfl_frequency_components,
)
from gkx.operators.nonlinear.projection import (
    _make_fixed_mode_projector,
    _make_hermitian_projector,
    _make_nonlinear_state_projector,
)
from gkx.solvers.nonlinear.diagnostic_integration import (
    _integrate_nonlinear_explicit_diagnostics_impl,
)
from gkx.terms.config import FieldState, TermConfig


def test_nonlinear_rhs_cached_prunes_disabled_em_fields(monkeypatch) -> None:
    G0 = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.zeros((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=jnp.zeros_like(phi), bpar=jnp.zeros_like(phi))
    cache = SimpleNamespace(
        Jl=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        JlB=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        kx_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        ky_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        dealias_mask=jnp.ones((1, 1), dtype=bool),
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        nonlinear_state_integration_mod,
        "assemble_rhs_cached_electrostatic_jit",
        lambda G, cache, params, terms, **kwargs: (jnp.zeros_like(G), fields),
    )
    monkeypatch.setattr(
        nonlinear_state_integration_mod,
        "assemble_rhs_cached_jit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic RHS should not run")
        ),
    )

    def _fake_nonlinear_em(G, **kwargs):
        seen["apar"] = kwargs["apar"]
        seen["bpar"] = kwargs["bpar"]
        return jnp.ones_like(G)

    monkeypatch.setattr(
        nonlinear_state_integration_mod, "nonlinear_em_contribution", _fake_nonlinear_em
    )
    rhs, rhs_fields = nonlinear_state_integration_mod.nonlinear_rhs_cached(
        G0,
        cache,
        params,
        TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0),
    )
    assert seen == {"apar": None, "bpar": None}
    np.testing.assert_allclose(np.asarray(rhs), 1.0)
    assert rhs_fields is fields


def test_nonlinear_rhs_cached_routes_generic_and_skips_disabled_bracket(
    monkeypatch,
) -> None:
    G0 = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    cache = SimpleNamespace()
    params = SimpleNamespace()
    calls: dict[str, int] = {"generic": 0, "electrostatic": 0, "nonlinear": 0}

    def _generic(G, cache, params, terms, **kwargs):
        calls["generic"] += 1
        return 4.0 * jnp.ones_like(G), fields

    def _electrostatic(*args, **kwargs):
        calls["electrostatic"] += 1
        raise AssertionError(
            "electrostatic fast path is invalid when Apar/Bpar terms are enabled"
        )

    def _nonlinear(*args, **kwargs):
        calls["nonlinear"] += 1
        raise AssertionError(
            "nonlinear bracket must not run when terms.nonlinear is zero"
        )

    monkeypatch.setattr(
        nonlinear_state_integration_mod, "assemble_rhs_cached_jit", _generic
    )
    monkeypatch.setattr(
        nonlinear_state_integration_mod,
        "assemble_rhs_cached_electrostatic_jit",
        _electrostatic,
    )
    monkeypatch.setattr(
        nonlinear_state_integration_mod, "nonlinear_em_contribution", _nonlinear
    )

    rhs, rhs_fields = nonlinear_state_integration_mod.nonlinear_rhs_cached(
        G0,
        cache,
        params,
        TermConfig(nonlinear=0.0, apar=1.0, bpar=1.0),
    )

    assert calls == {"generic": 1, "electrostatic": 0, "nonlinear": 0}
    np.testing.assert_allclose(np.asarray(rhs), 4.0)
    assert rhs_fields is fields


def test_nonlinear_rhs_cached_forwards_enabled_em_fields(monkeypatch) -> None:
    G0 = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    cache = SimpleNamespace(
        Jl=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        JlB=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        kx_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        ky_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        dealias_mask=jnp.ones((1, 1), dtype=bool),
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        nonlinear_state_integration_mod,
        "assemble_rhs_cached_jit",
        lambda G, cache, params, terms, **kwargs: (jnp.zeros_like(G), fields),
    )

    def _fake_nonlinear_em(G, **kwargs):
        seen["apar"] = kwargs["apar"]
        seen["bpar"] = kwargs["bpar"]
        seen["apar_weight"] = kwargs["apar_weight"]
        seen["bpar_weight"] = kwargs["bpar_weight"]
        seen["weight_dtype"] = kwargs["weight"].dtype
        return 2.0 * jnp.ones_like(G)

    monkeypatch.setattr(
        nonlinear_state_integration_mod, "nonlinear_em_contribution", _fake_nonlinear_em
    )
    rhs, _rhs_fields = nonlinear_state_integration_mod.nonlinear_rhs_cached(
        G0,
        cache,
        params,
        TermConfig(nonlinear=0.5, apar=1.0, bpar=1.0),
    )

    assert seen["apar"] is fields.apar
    assert seen["bpar"] is fields.bpar
    assert seen["apar_weight"] == pytest.approx(1.0)
    assert seen["bpar_weight"] == pytest.approx(1.0)
    assert seen["weight_dtype"] == jnp.float32
    np.testing.assert_allclose(np.asarray(rhs), 2.0)


def test_pack_resolved_diagnostics_and_fixed_mode_projector() -> None:
    names = [field.name for field in dataclass_fields(ResolvedDiagnostics)]
    assert len(names) == 58
    assert names[:9] == [
        "Phi2_kxt",
        "Phi2_kyt",
        "Phi2_kxkyt",
        "Phi2_zt",
        "Phi2_zonal_t",
        "Phi2_zonal_kxt",
        "Phi2_zonal_zt",
        "Phi_zonal_mode_kxt",
        "Phi_zonal_line_kxt",
    ]
    assert names[-4:] == [
        "TurbulentHeating_kxst",
        "TurbulentHeating_kyst",
        "TurbulentHeating_kxkyst",
        "TurbulentHeating_zst",
    ]
    resolved = tuple(np.full((1,), i, dtype=float) for i in range(len(names)))
    packed = _pack_resolved_diagnostics(resolved)
    for index, name in enumerate(names):
        np.testing.assert_allclose(getattr(packed, name), [float(index)])

    projector = _make_fixed_mode_projector(
        jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 2, 4),
        ky_index=1,
        kx_index=0,
    )
    G = jnp.zeros((1, 3, 2, 4), dtype=jnp.float32)
    out = projector(G)
    np.testing.assert_allclose(
        np.asarray(out[..., 1:2, 0:1, :]),
        np.asarray(
            jnp.arange(24, dtype=jnp.float32).reshape(1, 3, 2, 4)[..., 1:2, 0:1, :]
        ),
    )
    assert _make_fixed_mode_projector(None, ky_index=0, kx_index=0) is None


def test_sample_indices_with_final_preserves_last_step() -> None:
    np.testing.assert_array_equal(
        _sample_indices_with_final(6, 4), np.asarray([0, 4, 5])
    )
    np.testing.assert_array_equal(_sample_indices_with_final(6, 5), np.asarray([0, 5]))
    assert isinstance(_sample_indices_with_final(6, 1), slice)


def test_sampled_scan_intervals_and_runner_retain_final_step() -> None:
    np.testing.assert_array_equal(
        sampled_scan_intervals(5, 2), np.asarray([1, 2, 2], dtype=np.int32)
    )

    def step_fn(carry, _idx):
        G, _G_prev, fields_prev, _diag_prev, t_prev, dt_prev = carry
        G_new = G + 1
        fields_new = fields_prev + 10
        diag = G_new * 100
        t_new = t_prev + dt_prev
        return (G_new, G_new, fields_new, diag, t_new, dt_prev), (
            diag,
            t_new,
            dt_prev,
        )

    final_carry, diag_out = run_sampled_explicit_diagnostic_scan(
        step_fn,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.1, dtype=jnp.float32),
        ),
        steps=5,
        stride=2,
    )

    G_final, _G_prev, fields_final, diag_final, t_final, dt_final = final_carry
    diag_t, t_t, dt_t = diag_out
    np.testing.assert_allclose(np.asarray(G_final), 5)
    np.testing.assert_allclose(np.asarray(fields_final), 50)
    np.testing.assert_allclose(np.asarray(diag_final), 500)
    np.testing.assert_allclose(np.asarray(t_final), 0.5, rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(dt_final), 0.1, rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(diag_t), [100, 300, 500])
    np.testing.assert_allclose(np.asarray(t_t), [0.1, 0.3, 0.5], rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(dt_t), [0.1, 0.1, 0.1], rtol=1.0e-6)


def test_build_nonlinear_simulation_diagnostics_samples_scan_tuple() -> None:
    series = tuple(jnp.arange(5, dtype=jnp.float32) + offset for offset in range(12))
    resolved = tuple(
        jnp.full((5, 2), float(index), dtype=jnp.float32)
        for index, _field in enumerate(dataclass_fields(ResolvedDiagnostics))
    )
    diag = (*series, resolved)
    sample_idx = np.asarray([0, 3, 4], dtype=int)

    out = build_nonlinear_simulation_diagnostics(
        diag,
        t=jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32),
        dt_series=jnp.ones((5,), dtype=jnp.float32) * 0.1,
        resolved_diagnostics=True,
        sample_indices=sample_idx,
    )

    np.testing.assert_allclose(np.asarray(out.t), [0.1, 0.4, 0.5])
    np.testing.assert_allclose(np.asarray(out.gamma_t), [0.0, 3.0, 4.0])
    np.testing.assert_allclose(
        np.asarray(out.energy_t),
        np.asarray(out.Wg_t + out.Wphi_t + out.Wapar_t),
    )
    assert out.resolved is not None
    np.testing.assert_allclose(np.asarray(out.resolved.Phi2_kxt), 0.0)
    np.testing.assert_allclose(np.asarray(out.resolved.TurbulentHeating_zst), 57.0)


def test_finalize_nonlinear_scan_diagnostics_applies_output_sampling() -> None:
    series = tuple(jnp.arange(5, dtype=jnp.float32) + offset for offset in range(12))
    diag = (*series, ())
    t = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)
    dt = jnp.ones((5,), dtype=jnp.float32) * 0.1

    sampled = finalize_nonlinear_scan_diagnostics(
        diag,
        t=t,
        dt_series=dt,
        stride=3,
        sampled_scan=False,
        resolved_diagnostics=False,
    )
    retained = finalize_nonlinear_scan_diagnostics(
        diag,
        t=t,
        dt_series=dt,
        stride=3,
        sampled_scan=True,
        resolved_diagnostics=False,
    )

    np.testing.assert_allclose(np.asarray(sampled.t), [0.1, 0.4, 0.5])
    np.testing.assert_allclose(np.asarray(sampled.gamma_t), [0.0, 3.0, 4.0])
    np.testing.assert_allclose(np.asarray(retained.t), np.asarray(t))
    np.testing.assert_allclose(np.asarray(retained.gamma_t), np.arange(5))


def test_select_nonlinear_step_diagnostics_and_progress_noop() -> None:
    computed = (jnp.asarray(2.0), jnp.asarray(3.0), jnp.asarray(4.0), jnp.asarray(5.0))
    previous = (
        jnp.asarray(-1.0),
        jnp.asarray(-2.0),
        jnp.asarray(-3.0),
        jnp.asarray(-4.0),
    )

    used_compute = select_nonlinear_step_diagnostics(
        jnp.asarray(4, dtype=jnp.int32),
        diagnostics_stride=2,
        diag_prev=previous,
        compute_diag_fn=lambda: computed,
    )
    used_previous = select_nonlinear_step_diagnostics(
        jnp.asarray(3, dtype=jnp.int32),
        diagnostics_stride=2,
        diag_prev=previous,
        compute_diag_fn=lambda: computed,
    )

    np.testing.assert_allclose(np.asarray(used_compute[0]), 2.0)
    np.testing.assert_allclose(np.asarray(used_previous[0]), -1.0)

    state = jnp.asarray([7.0], dtype=jnp.float32)
    out = maybe_emit_nonlinear_progress(
        state,
        show_progress=False,
        diag=computed,
        idx=jnp.asarray(0, dtype=jnp.int32),
        steps=4,
        t_new=jnp.asarray(0.1, dtype=jnp.float32),
        progress_total=jnp.asarray(0.4, dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(out), [7.0])


def test_make_hermitian_projector_and_mode_mask() -> None:
    ky = np.array([0.0, 0.2, -0.2, -0.4], dtype=float)
    projector = _make_hermitian_projector(ky, nx=3)
    assert _make_hermitian_projector(ky.copy(), nx=3) is projector
    state = jnp.zeros((1, 4, 3, 2), dtype=jnp.complex64)
    state = state.at[..., 0:3, :, :].set(1.0 + 2.0j)
    out = projector(state)
    assert out.shape == state.shape
    np.testing.assert_allclose(
        np.asarray(out[..., 3, :, :]), np.asarray(jnp.conj(out[..., 1, [0, 2, 1], :]))
    )

    no_project = _make_hermitian_projector(np.array([0.0, 0.2], dtype=float), nx=1)
    same = no_project(state[..., :2, :1, :])
    np.testing.assert_allclose(np.asarray(same), np.asarray(state[..., :2, :1, :]))

    single_kx_projector = _make_hermitian_projector(
        np.array([0.0, 0.2, 0.4, -0.4, -0.2], dtype=float), nx=1
    )
    single_kx_state = (
        jnp.arange(5, dtype=jnp.float32).reshape(1, 5, 1, 1).astype(jnp.complex64)
    )
    single_kx_out = single_kx_projector(single_kx_state)
    np.testing.assert_allclose(
        np.asarray(single_kx_out[..., 3:, :, :]),
        np.asarray(jnp.conj(single_kx_out[..., 1:3, :, :])[..., ::-1, :, :]),
    )

    grid = SimpleNamespace(
        ky=np.array([0.0, 0.2, -0.2, -0.4]),
        kx=np.array([0.0, 0.5]),
        dealias_mask=np.array(
            [[True, False], [True, True], [True, True], [False, True]]
        ),
    )
    cache = SimpleNamespace(ky=jnp.asarray(grid.ky))
    mask = _diagnostic_omega_mode_mask(grid, cache, compressed_real_fft=True)
    assert mask.shape == (4, 2)
    assert bool(mask[0, 0]) is True
    assert bool(mask[3, 1]) is False

    signed_mask = _diagnostic_omega_mode_mask(grid, cache, compressed_real_fft=False)
    assert bool(signed_mask[1, 0]) is True
    assert bool(signed_mask[2, 0]) is False

    positive_grid = SimpleNamespace(
        ky=np.array([0.0, 0.2, 0.4]),
        kx=np.array([0.0, 0.5]),
        dealias_mask=np.array([[True, True], [True, False], [False, True]]),
    )
    positive_cache = SimpleNamespace(ky=jnp.asarray(positive_grid.ky))
    positive_mask = _diagnostic_omega_mode_mask(
        positive_grid, positive_cache, compressed_real_fft=True
    )
    np.testing.assert_array_equal(np.asarray(positive_mask), positive_grid.dealias_mask)


def test_make_nonlinear_state_projector_composes_fixed_mode_and_hermitian() -> None:
    fixed = jnp.zeros((1, 4, 3, 2), dtype=jnp.complex64)
    fixed = fixed.at[..., 1:2, 1:2, :].set(7.0 + 1.0j)
    trial = jnp.ones((1, 4, 3, 2), dtype=jnp.complex64) * (2.0 + 3.0j)

    projector = _make_nonlinear_state_projector(
        fixed,
        ky_vals=np.array([0.0, 0.2, 0.4, -0.2], dtype=float),
        nx=3,
        compressed_real_fft=True,
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=1,
    )
    projected = projector(trial)

    np.testing.assert_allclose(
        np.asarray(projected[..., 1:2, 1:2, :]),
        np.asarray(fixed[..., 1:2, 1:2, :]),
    )
    np.testing.assert_allclose(
        np.asarray(projected[..., 3, :, :]),
        np.conj(np.asarray(projected[..., 1, [0, 2, 1], :])),
    )

    no_hermitian = _make_nonlinear_state_projector(
        fixed,
        ky_vals=np.array([0.0, 0.2, 0.4, -0.2], dtype=float),
        nx=3,
        compressed_real_fft=False,
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
    )
    np.testing.assert_allclose(np.asarray(no_hermitian(trial)), np.asarray(trial))


def test_shearing_coordinates_follow_analytic_wave_and_inverse_remap() -> None:
    kx = jnp.asarray([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
    ky = jnp.asarray([0.0, 1.0])
    state = jnp.zeros((1, 2, 8, 1), dtype=jnp.complex64)
    state = state.at[0, 1, 0, 0].set(2.0 - 0.5j)

    update = nonlinear_projection.advance_shearing_coordinates(
        state,
        kx=kx,
        ky=ky,
        x0=1.0,
        shear_rate=1.0,
        previous_time=0.0,
        time=1.2,
    )

    assert int(update.cumulative_mode_shift[0]) == 0
    assert int(update.cumulative_mode_shift[1]) == -1
    np.testing.assert_allclose(update.state[0, 1, 7, 0], 2.0 - 0.5j)
    np.testing.assert_allclose(update.effective_kx[1, 7], -1.2, atol=2.0e-7)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(update.state)),
        np.linalg.norm(np.asarray(state)),
        atol=2.0e-7,
    )

    restored = nonlinear_projection.advance_shearing_coordinates(
        update.state,
        kx=kx,
        ky=ky,
        x0=1.0,
        shear_rate=1.0,
        previous_time=1.2,
        time=0.0,
    )
    np.testing.assert_allclose(restored.state, state, atol=2.0e-7)


def test_shearing_coordinates_zero_shear_and_dealias_boundary() -> None:
    kx = jnp.asarray([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
    ky = jnp.asarray([0.0, 1.0])
    state = (jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8, 1) + 0.25j).astype(
        jnp.complex64
    )
    identity = nonlinear_projection.advance_shearing_coordinates(
        state,
        kx=kx,
        ky=ky,
        x0=1.0,
        shear_rate=0.0,
        previous_time=0.0,
        time=4.0,
    )
    np.testing.assert_array_equal(identity.state, state)
    np.testing.assert_allclose(
        identity.effective_kx,
        jnp.broadcast_to(kx[None, :], (ky.size, kx.size)),
    )
    np.testing.assert_allclose(identity.phase, 1.0)

    edge = jnp.zeros_like(state).at[0, 1, 6, 0].set(1.0)
    mask = jnp.abs(kx)[None, :] <= 2.0
    mask = jnp.broadcast_to(mask, (2, 8))
    shifted = nonlinear_projection.advance_shearing_coordinates(
        edge,
        kx=kx,
        ky=ky,
        x0=1.0,
        shear_rate=0.5,
        previous_time=0.0,
        time=1.0,
        dealias_mask=mask,
    )
    assert int(shifted.cumulative_mode_shift[1]) == -1
    np.testing.assert_allclose(shifted.state, 0.0)


def test_shearing_coordinate_tangent_matches_finite_difference() -> None:
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0])
    ky = jnp.asarray([0.0, 0.75])
    state = jnp.ones((1, 2, 4, 1), dtype=jnp.complex64)

    def observables(rate):
        update = nonlinear_projection.advance_shearing_coordinates(
            state,
            kx=kx,
            ky=ky,
            x0=1.0,
            shear_rate=rate,
            previous_time=0.0,
            time=0.2,
        )
        return update.effective_kx[1, 0], update.phase[1, 1]

    rate = jnp.asarray(0.4, dtype=jnp.float32)
    _, tangent = jax.jvp(observables, (rate,), (jnp.ones_like(rate),))
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    plus = observables(rate + step)
    minus = observables(rate - step)
    finite_difference = tuple((hi - lo) / (2.0 * step) for hi, lo in zip(plus, minus))
    np.testing.assert_allclose(tangent[0], finite_difference[0], rtol=2.0e-4)
    np.testing.assert_allclose(tangent[1], finite_difference[1], rtol=3.0e-4)

    def radial_scale_observable(x0):
        return nonlinear_projection.advance_shearing_coordinates(
            state,
            kx=kx,
            ky=ky,
            x0=x0,
            shear_rate=rate,
            previous_time=0.0,
            time=0.2,
        ).phase[1, 1]

    x0 = jnp.asarray(1.1, dtype=jnp.float32)
    _, x0_tangent = jax.jvp(
        radial_scale_observable,
        (x0,),
        (jnp.ones_like(x0),),
    )
    x0_plus = radial_scale_observable(x0 + step)
    x0_minus = radial_scale_observable(x0 - step)
    x0_finite_difference = (x0_plus - x0_minus) / (2.0 * step)
    np.testing.assert_allclose(x0_tangent, x0_finite_difference, rtol=3.0e-4)


def test_sheared_integrator_zero_shear_identity_and_full_step_remap() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=4,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="periodic",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    state = jnp.zeros((1, 1, 4, 4, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(0.2 + 0.1j)
    project_state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)
    state = project_state(state)
    nonlinear_only = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=1.0,
    )
    for method in ("rk2", "rk3"):
        reference_state, reference_fields = integrate_nonlinear_cached(
            jnp.asarray(np.asarray(state).copy()),
            cache,
            params,
            dt=0.02,
            steps=2,
            method=method,
            terms=nonlinear_only,
            compressed_real_fft=True,
        )
        sheared_state, sheared_fields = integrate_nonlinear_sheared(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=0.0,
            method=method,
            cache=cache,
            terms=nonlinear_only,
        )
        np.testing.assert_allclose(sheared_state, reference_state, atol=2.0e-7)
        np.testing.assert_allclose(
            sheared_fields.phi, reference_fields.phi, atol=2.0e-7
        )
        np.testing.assert_allclose(
            sheared_state, project_state(sheared_state), atol=1.0e-7
        )
        state_only = integrate_nonlinear_sheared(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=0.0,
            method=method,
            cache=cache,
            terms=nonlinear_only,
            return_fields=False,
        )
        np.testing.assert_allclose(state_only, reference_state, atol=2.0e-7)

    disabled = TermConfig(*([0.0] * 12))
    remapped_state, _ = integrate_nonlinear_sheared(
        state,
        grid,
        geom,
        params,
        dt=0.4,
        steps=3,
        shear_rate=1.0,
        method="rk2",
        cache=cache,
        terms=disabled,
    )
    expected = nonlinear_projection.advance_shearing_coordinates(
        state,
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=1.2,
        dealias_mask=grid.dealias_mask,
    ).state
    expected = project_state(expected)
    np.testing.assert_allclose(remapped_state, expected, atol=2.0e-7)

    with pytest.raises(ValueError, match="steps must be at least one"):
        integrate_nonlinear_sheared(
            state,
            grid,
            geom,
            params,
            dt=0.1,
            steps=0,
            shear_rate=1.0,
            cache=cache,
        )
    with pytest.raises(ValueError, match="method must be"):
        integrate_nonlinear_sheared(
            state,
            grid,
            geom,
            params,
            dt=0.1,
            steps=1,
            shear_rate=1.0,
            method="rk4",
            cache=cache,
        )


@pytest.mark.parametrize("method", ["rk2", "rk3", "imex"])
def test_linked_sheared_integrator_has_exact_zero_shear_trajectory_identity(
    method: str,
) -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=8,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="linked",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=0.3, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    state = jnp.zeros((1, 2, 4, 8, 8), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(0.2 + 0.1j)
    state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(state)
    streaming_only = TermConfig(
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )
    state_host = np.asarray(state)

    reference_state, reference_fields = integrate_nonlinear_cached(
        jnp.asarray(state_host.copy()),
        cache,
        params,
        dt=0.01,
        steps=2,
        method=method,
        terms=streaming_only,
        compressed_real_fft=False,
    )
    sheared_state, sheared_fields = integrate_nonlinear_sheared(
        jnp.asarray(state_host.copy()),
        grid,
        geom,
        params,
        dt=0.01,
        steps=2,
        shear_rate=0.0,
        method=method,
        cache=cache,
        terms=streaming_only,
        compressed_real_fft=False,
    )

    np.testing.assert_allclose(sheared_state, reference_state, atol=2.0e-7)
    np.testing.assert_allclose(sheared_fields.phi, reference_fields.phi, atol=2.0e-7)


def test_linked_sheared_cache_preserves_chains_and_has_correct_tangent() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=8,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="linked",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=0.3, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)

    def shifted_cache(displacement):
        effective_kx = cache.kx_grid - cache.ky_grid * displacement
        return update_linear_cache_for_sheared_kx(
            cache, grid, geom, params, effective_kx
        )

    updated = shifted_cache(jnp.asarray(0.17, dtype=cache.kx_grid.dtype))
    np.testing.assert_array_equal(updated.kx_link_plus, cache.kx_link_plus)
    np.testing.assert_array_equal(updated.kx_link_minus, cache.kx_link_minus)
    for ky_index in range(cache.kx_link_plus.shape[0]):
        valid = np.asarray(cache.kx_link_mask_plus[ky_index])
        linked = np.asarray(cache.kx_link_plus[ky_index])[valid]
        source = np.arange(cache.kx_link_plus.shape[1])[valid]
        before = np.asarray(
            cache.kx_grid[ky_index, linked] - cache.kx_grid[ky_index, source]
        )
        after = np.asarray(
            updated.kx_grid[ky_index, linked] - updated.kx_grid[ky_index, source]
        )
        np.testing.assert_allclose(after, before, atol=2.0e-7)

    def objective(displacement):
        return jnp.sum(shifted_cache(displacement).kperp2**2)

    displacement = jnp.asarray(0.17, dtype=cache.kx_grid.dtype)
    _, tangent = jax.jvp(objective, (displacement,), (jnp.ones_like(displacement),))
    step = jnp.asarray(1.0e-3, dtype=displacement.dtype)
    finite_difference = (
        objective(displacement + step) - objective(displacement - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-3, atol=2.0e-4)


def _small_sheared_transport_case():
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=4,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="periodic",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        rho_star=1.0,
        nu_hyper=0.0,
        nu_hyper_m=0.0,
        tprim=2.0,
        fprim=0.5,
    )
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=4)
    state = jnp.zeros((1, 4, 4, 4, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(
        jnp.asarray([0.2 + 0.1j, 0.1 - 0.05j, 0.15 + 0.02j, 0.07 - 0.03j])
    )
    state = state.at[0, 2, 1, 1, :].set(0.03 + 0.02j)
    state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(state)
    terms = TermConfig(collisions=0.0, hypercollisions=0.0, end_damping=0.0)
    return grid, geom, params, cache, state, terms


def test_sheared_imex_rejects_unvalidated_adaptive_and_collision_routes() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()
    with pytest.raises(ValueError, match="requires fixed_dt=True"):
        integrate_nonlinear_sheared_transport(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=0.1,
            method="imex",
            cache=cache,
            terms=terms,
            fixed_dt=False,
        )
    with pytest.raises(NotImplementedError, match="custom collision operators"):
        integrate_nonlinear_sheared(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=0.1,
            method="imex",
            cache=cache,
            terms=terms,
            collision_operator=object(),
        )


@pytest.mark.parametrize("method", ["rk2", "imex"])
def test_sheared_transport_trace_matches_canonical_final_heat_flux(method: str) -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    trace = integrate_nonlinear_sheared_transport(
        state,
        grid,
        geom,
        params,
        dt=0.02,
        steps=3,
        shear_rate=0.0,
        method=method,
        cache=cache,
        terms=terms,
    )

    np.testing.assert_allclose(trace.time, [0.02, 0.04, 0.06], rtol=1.0e-6)
    assert trace.heat_flux.shape == (3, 1)
    _, flux_fac = fieldline_quadrature_weights(geom, grid)
    _, final_fields = nonlinear_rhs_cached(
        trace.final_state,
        cache,
        params,
        terms,
        compressed_real_fft=False,
    )
    apar = jnp.zeros_like(final_fields.phi)
    bpar = jnp.zeros_like(final_fields.phi)
    expected = heat_flux_species(
        trace.final_state,
        final_fields.phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
    )
    np.testing.assert_allclose(trace.heat_flux[-1], expected, rtol=2.0e-6, atol=2.0e-7)


def test_sheared_transport_compressed_bracket_matches_full_fractional_phase() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()
    options = dict(
        dt=0.017,
        steps=3,
        shear_rate=0.37,
        method="rk3",
        cache=cache,
        terms=terms,
        differentiable=False,
    )

    full = integrate_nonlinear_sheared_transport(
        state, grid, geom, params, compressed_real_fft=False, **options
    )
    compressed = integrate_nonlinear_sheared_transport(
        state, grid, geom, params, compressed_real_fft=True, **options
    )

    np.testing.assert_allclose(compressed.time, full.time, atol=1.0e-7)
    np.testing.assert_allclose(
        compressed.final_state, full.final_state, rtol=2.0e-5, atol=2.0e-7
    )
    np.testing.assert_allclose(
        compressed.heat_flux, full.heat_flux, rtol=2.0e-5, atol=2.0e-7
    )


def test_sheared_transport_preserves_x64_scan_carry_dtype() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64():
        state64 = state.astype(jnp.complex128)
        trace = integrate_nonlinear_sheared_transport(
            state64,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=0.01,
            method="rk3",
            cache=cache,
            terms=terms,
            differentiable=False,
            fixed_dt=False,
        )

    assert trace.final_state.dtype == jnp.complex128
    assert np.isfinite(np.asarray(trace.heat_flux)).all()


def test_sheared_imex_promotes_complex64_input_to_x64_operator_dtype() -> None:
    grid, geom, params, _cache, state, terms = _small_sheared_transport_case()
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64
    with enable_x64():
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=4)
        trace = integrate_nonlinear_sheared_transport(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=1,
            shear_rate=0.01,
            method="imex",
            cache=cache,
            terms=terms,
        )

    assert trace.final_state.dtype == jnp.complex128
    assert np.isfinite(np.asarray(trace.heat_flux)).all()


def test_sheared_transport_scale_does_not_change_trajectory() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    base = integrate_nonlinear_sheared_transport(
        state,
        grid,
        geom,
        params,
        dt=0.02,
        steps=2,
        shear_rate=0.2,
        cache=cache,
        terms=terms,
    )
    scaled = integrate_nonlinear_sheared_transport(
        state,
        grid,
        geom,
        params,
        dt=0.02,
        steps=2,
        shear_rate=0.2,
        cache=cache,
        terms=terms,
        flux_scale=3.0,
    )
    fast = integrate_nonlinear_sheared_transport(
        state,
        grid,
        geom,
        params,
        dt=0.02,
        steps=2,
        shear_rate=0.2,
        cache=cache,
        terms=terms,
        differentiable=False,
    )

    np.testing.assert_allclose(scaled.final_state, base.final_state, atol=2.0e-7)
    np.testing.assert_allclose(scaled.heat_flux, 3.0 * base.heat_flux, atol=2.0e-7)
    np.testing.assert_allclose(fast.final_state, base.final_state, atol=2.0e-7)
    np.testing.assert_allclose(fast.heat_flux, base.heat_flux, atol=2.0e-7)


def test_sheared_transport_adaptive_cfl_records_accepted_time_steps() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    def run(amplitude):
        return integrate_nonlinear_sheared_transport(
            amplitude * state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=4,
            shear_rate=0.1,
            method="rk3",
            cache=cache,
            terms=terms,
            fixed_dt=False,
            dt_min=1.0e-7,
            dt_max=0.02,
            cfl=0.9,
        )

    amplitude = jnp.asarray(1.0e5, dtype=jnp.float32)
    trace = run(amplitude)

    accepted_dt = np.diff(np.concatenate(([0.0], np.asarray(trace.time))))
    assert np.all(np.isfinite(accepted_dt))
    assert np.all(accepted_dt > 0.0)
    assert np.max(accepted_dt) <= 0.02 + 1.0e-7
    assert np.min(accepted_dt) < 0.019
    assert np.all(np.isfinite(np.asarray(trace.heat_flux)))

    def final_time(scale):
        return run(scale).time[-1]

    _, tangent = jax.jvp(final_time, (amplitude,), (jnp.ones_like(amplitude),))
    step = 100.0
    finite_difference = (
        final_time(amplitude + step) - final_time(amplitude - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1.0e-3, atol=1.0e-12)


def test_sheared_transport_restart_preserves_physical_time_and_state() -> None:
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()
    options = dict(
        shear_rate=0.2,
        method="rk3",
        cache=cache,
        terms=terms,
        differentiable=False,
    )

    complete = integrate_nonlinear_sheared_transport(
        state, grid, geom, params, dt=0.02, steps=4, **options
    )
    first = integrate_nonlinear_sheared_transport(
        state, grid, geom, params, dt=0.02, steps=2, **options
    )
    second = integrate_nonlinear_sheared_transport(
        first.final_state,
        grid,
        geom,
        params,
        dt=0.02,
        steps=2,
        initial_time=first.time[-1],
        initial_dt=first.time[-1] - first.time[-2],
        **options,
    )

    np.testing.assert_allclose(second.final_state, complete.final_state, atol=3.0e-7)
    np.testing.assert_allclose(
        jnp.concatenate((first.time, second.time)), complete.time, atol=1.0e-7
    )
    np.testing.assert_allclose(
        jnp.concatenate((first.heat_flux, second.heat_flux)),
        complete.heat_flux,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


@pytest.mark.parametrize("method", ["rk2", "imex"])
def test_sheared_transport_gradient_matches_tangent_and_finite_difference(
    method: str,
) -> None:
    jax.clear_caches()
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    def objective(shear_rate):
        trace = integrate_nonlinear_sheared_transport(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            shear_rate=shear_rate,
            method=method,
            cache=cache,
            terms=terms,
        )
        return jnp.mean(trace.heat_flux)

    shear_rate = jnp.asarray(0.2, dtype=jnp.float32)
    _, tangent = jax.jvp(
        objective,
        (shear_rate,),
        (jnp.ones_like(shear_rate),),
    )
    gradient = jax.grad(objective)(shear_rate)
    step = jnp.asarray(0.05, dtype=shear_rate.dtype)
    finite_difference = (
        objective(shear_rate + step) - objective(shear_rate - step)
    ) / (2.0 * step)

    assert np.isfinite(float(gradient))
    np.testing.assert_allclose(tangent, gradient, rtol=3.0e-5, atol=1.0e-11)
    np.testing.assert_allclose(gradient, finite_difference, rtol=5.0e-3, atol=1.0e-10)
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize(
    ("method", "minimum_order"),
    [("rk2", 1.8), ("rk3", 2.6), ("imex", 0.8)],
)
def test_sheared_runge_kutta_recovers_observed_order_on_physical_rhs(
    method: str,
    minimum_order: float,
) -> None:
    jax.clear_caches()
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=4,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="periodic",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        rho_star=1.0,
        nu_hyper=0.0,
        nu_hyper_m=0.0,
        tprim=2.0,
        fprim=0.5,
    )
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    initial = np.zeros((1, 2, 4, 4, 4), dtype=np.complex64)
    initial[0, 0, 1, 0, :] = np.asarray(
        [0.2 + 0.1j, 0.1 - 0.05j, 0.15 + 0.02j, 0.07 - 0.03j]
    )
    initial[0, 1, 1, 1, :] = 0.03 + 0.02j
    initial = np.asarray(
        _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(initial)
    )
    drift_drive = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )

    solutions = []
    final_time = 0.08
    with jax.disable_jit():
        for steps in (2, 4, 8):
            state, _ = integrate_nonlinear_sheared(
                jnp.asarray(initial.copy()),
                grid,
                geom,
                params,
                dt=final_time / steps,
                steps=steps,
                shear_rate=0.3,
                method=method,
                cache=cache,
                terms=drift_drive,
            )
            solutions.append(np.asarray(state))

    coarse_difference = np.linalg.norm(solutions[0] - solutions[1])
    fine_difference = np.linalg.norm(solutions[1] - solutions[2])
    observed_order = np.log(coarse_difference / fine_difference) / np.log(2.0)
    assert observed_order > minimum_order


def test_strong_flow_shear_suppresses_linear_itg_amplitude_after_dt_refinement() -> (
    None
):
    grid = build_spectral_grid(
        GridConfig(
            Nx=8,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi / 0.2,
            Ly=2.0 * np.pi / 0.3,
            boundary="periodic",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    params = LinearParams(
        rho_star=1.0,
        nu_hyper=0.0,
        nu_hyper_m=0.0,
        tprim=6.9,
        fprim=2.2,
        damp_ends_amp=0.0,
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    rng = np.random.default_rng(10)
    initial = np.zeros((2, 4, 4, 8, 8), dtype=np.complex64)
    initial[:, :, 1, 0, :] = 1.0e-4 * (
        rng.normal(size=(2, 4, 8)) + 1j * rng.normal(size=(2, 4, 8))
    )
    initial = np.asarray(
        _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(initial)
    )
    linear_itg = TermConfig(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )

    amplitudes: dict[tuple[float, float], float] = {}
    for dt in (0.02, 0.01):
        for shear_rate in (0.0, 0.5):
            _, fields = integrate_nonlinear_sheared(
                jnp.asarray(initial.copy()),
                grid,
                geom,
                params,
                dt=dt,
                steps=int(round(2.0 / dt)),
                shear_rate=shear_rate,
                method="rk2",
                cache=cache,
                terms=linear_itg,
            )
            amplitudes[(dt, shear_rate)] = float(jnp.linalg.norm(fields.phi[-1]))

    for shear_rate in (0.0, 0.5):
        coarse = amplitudes[(0.02, shear_rate)]
        fine = amplitudes[(0.01, shear_rate)]
        assert abs(coarse - fine) / fine < 0.01
    assert amplitudes[(0.01, 0.5)] / amplitudes[(0.01, 0.0)] < 0.8


def test_build_nonlinear_diagnostic_setup_uses_injected_policy() -> None:
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.2, -0.2], dtype=float),
        kx=np.asarray([0.0, 0.5], dtype=float),
        z=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=float),
    )
    cache = SimpleNamespace(ky=jnp.asarray(grid.ky), kx=jnp.asarray(grid.kx))
    calls: dict[str, object] = {}

    def _ensure_geometry(geom, z):
        calls["geom"] = geom
        calls["z"] = tuple(np.asarray(z, dtype=float))
        return SimpleNamespace(name="geometry")

    def _build_cache(_grid, geom, params, nl, nm):
        calls["cache_counts"] = (nl, nm)
        assert geom.name == "geometry"
        assert params.name == "params"
        return cache

    def _weights(geom, grid_in):
        assert geom.name == "geometry"
        return jnp.ones((grid_in.z.size,), dtype=jnp.float32), jnp.asarray(2.0)

    def _omega_mask(_grid, cache_in, *, compressed_real_fft):
        calls["compressed"] = compressed_real_fft
        assert cache_in is cache
        return jnp.ones((_grid.ky.size, _grid.kx.size), dtype=bool)

    setup = build_nonlinear_diagnostic_setup(
        jnp.zeros((2, 3, 3, 2, 4), dtype=jnp.complex64),
        grid,
        SimpleNamespace(name="raw"),
        SimpleNamespace(name="params"),
        cache=None,
        use_dealias_mask=True,
        z_index=None,
        compressed_real_fft=True,
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
        ensure_geometry_fn=_ensure_geometry,
        build_cache_fn=_build_cache,
        quadrature_weights_fn=_weights,
        omega_mask_fn=_omega_mask,
        midplane_index_fn=lambda nz: nz - 1,
    )

    assert calls["geom"].name == "raw"
    assert calls["z"] == (0.0, 0.5, 1.0, 1.5)
    assert calls["cache_counts"] == (2, 3)
    assert calls["compressed"] is True
    assert setup.cache is cache
    assert setup.z_idx == 3
    assert setup.use_dealias is True
    np.testing.assert_allclose(np.asarray(setup.vol_fac), np.ones(4))
    assert float(setup.flux_fac) == pytest.approx(2.0)


def test_build_nonlinear_time_step_policy_fixed_and_adaptive() -> None:
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.2, 0.4], dtype=float),
        kx=np.asarray([0.0, 0.5, 1.0], dtype=float),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        l=jnp.arange(2),
        m=jnp.arange(3)[None, :],
        kxfac=1.0,
    )
    params = SimpleNamespace(
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        tz=jnp.asarray([2.0], dtype=jnp.float32),
    )
    fields = FieldState(
        phi=jnp.ones((3, 3, 2), dtype=jnp.complex64),
        apar=None,
        bpar=None,
    )

    fixed = build_nonlinear_time_step_policy(
        grid,
        SimpleNamespace(),
        params,
        cache,
        method="rk3",
        dt=0.1,
        steps=4,
        fixed_dt=True,
        dt_min=0.01,
        dt_max=None,
        cfl=1.0,
        cfl_fac=None,
        compressed_real_fft=True,
        real_dtype=jnp.float32,
        resolve_cfl_fac_fn=lambda _method, _cfl_fac: 0.5,
        linear_frequency_bound_fn=lambda *args, **kwargs: np.asarray([1.0, 1.0, 1.0]),
        laguerre_velocity_max_fn=lambda _nl: 2.0,
        cfl_frequency_components_fn=lambda *args, **kwargs: (
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.asarray(3.0, dtype=jnp.float32),
        ),
    )
    np.testing.assert_allclose(np.asarray(fixed.dt_init), 0.1)
    np.testing.assert_allclose(np.asarray(fixed.progress_total), 0.4)
    np.testing.assert_allclose(
        np.asarray(fixed.update_dt(fields, jnp.asarray(0.07, dtype=jnp.float32))),
        0.07,
    )

    adaptive = build_nonlinear_time_step_policy(
        grid,
        SimpleNamespace(),
        params,
        cache,
        method="rk3",
        dt=0.1,
        steps=4,
        fixed_dt=False,
        dt_min=0.01,
        dt_max=0.2,
        cfl=1.0,
        cfl_fac=None,
        compressed_real_fft=True,
        real_dtype=jnp.float32,
        resolve_cfl_fac_fn=lambda _method, _cfl_fac: 0.5,
        linear_frequency_bound_fn=lambda *args, **kwargs: np.asarray([1.0, 1.0, 1.0]),
        laguerre_velocity_max_fn=lambda _nl: 2.0,
        cfl_frequency_components_fn=lambda *args, **kwargs: (
            jnp.asarray(2.0, dtype=jnp.float32),
            jnp.asarray(3.0, dtype=jnp.float32),
        ),
    )
    assert np.isnan(float(np.asarray(adaptive.progress_total)))
    np.testing.assert_allclose(
        np.asarray(adaptive.update_dt(fields, jnp.asarray(0.1, dtype=jnp.float32))),
        0.5 / 6.0,
        rtol=1.0e-6,
    )


def test_collision_damping_and_imex_operator_builder(monkeypatch) -> None:
    cache = SimpleNamespace(
        lb_lam=jnp.ones((2, 2, 1, 1, 1), dtype=jnp.float32),
    )
    params = SimpleNamespace(nu=0.1)
    term_cfg = TermConfig(collisions=0.5, hypercollisions=2.0)
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * 3.0,
    )
    damp = _collision_damping(
        cache, params, term_cfg, jnp.float32, squeeze_species=False
    )
    np.testing.assert_allclose(np.asarray(damp), 2.0 * 3.0)

    cache6 = SimpleNamespace(lb_lam=jnp.ones((1, 2, 2, 1, 1, 1), dtype=jnp.float32))
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype),
    )
    squeezed = _collision_damping(
        cache6,
        SimpleNamespace(nu=0.4),
        TermConfig(collisions=1.0, hypercollisions=1.0),
        jnp.float32,
        squeeze_species=True,
    )
    assert squeezed.shape == (2, 2, 1, 1, 1)

    cache_low_rank = SimpleNamespace(
        lb_lam=jnp.ones((2, 2), dtype=jnp.float32),
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones((1, 2, 2, 1, 1, 1), dtype=dtype),
    )
    squeezed_low_rank = _collision_damping(
        cache_low_rank,
        SimpleNamespace(nu=jnp.asarray([0.4], dtype=jnp.float32)),
        TermConfig(collisions=1.0, hypercollisions=1.0),
        jnp.float32,
        squeeze_species=True,
    )
    assert squeezed_low_rank.shape == (2, 2, 1, 1, 1)
    np.testing.assert_allclose(np.asarray(squeezed_low_rank), 1.0)

    monkeypatch.setattr(
        "gkx.operators.nonlinear.policies._build_implicit_operator",
        lambda *args, **kwargs: (
            jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
            (1, 2, 2, 1, 1, 1),
            4,
            jnp.asarray(0.1, dtype=jnp.float32),
            None,
            lambda x: x,
            True,
        ),
    )
    op = build_nonlinear_imex_operator(
        jnp.zeros((2, 2, 1, 1, 1), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
    )
    assert op.shape == (1, 2, 2, 1, 1, 1)
    assert op.squeeze_species is True


def test_build_nonlinear_collision_split_policy_controls_rhs_terms() -> None:
    term_cfg = TermConfig(collisions=0.5, hypercollisions=0.25, nonlinear=1.0)
    damping = jnp.asarray([2.0], dtype=jnp.float32)

    active = build_nonlinear_collision_split_policy(
        SimpleNamespace(name="cache"),
        SimpleNamespace(name="params"),
        term_cfg,
        jnp.float32,
        squeeze_species=True,
        collision_split=True,
        collision_damping_fn=lambda *args, **kwargs: damping,
    )
    inactive = build_nonlinear_collision_split_policy(
        SimpleNamespace(name="cache"),
        SimpleNamespace(name="params"),
        term_cfg,
        jnp.float32,
        squeeze_species=True,
        collision_split=False,
        collision_damping_fn=lambda *args, **kwargs: damping,
    )

    assert active.active is True
    assert active.rhs_terms.collisions == term_cfg.collisions
    assert active.rhs_terms.hypercollisions == 0.0
    assert active.rhs_terms.nonlinear == term_cfg.nonlinear
    np.testing.assert_allclose(np.asarray(active.damping), [2.0])
    assert inactive.active is False
    assert inactive.rhs_terms is term_cfg
    assert inactive.damping is None


def test_build_nonlinear_imex_operator_forwards_preconditioner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build(G0, cache, params, dt, terms, implicit_preconditioner):
        captured["preconditioner"] = implicit_preconditioner
        captured["terms"] = terms
        return (
            jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
            (1, 2, 2, 1, 1, 1),
            4,
            jnp.asarray(0.2, dtype=jnp.float32),
            lambda x: x,
            lambda x: x,
            False,
        )

    monkeypatch.setattr(
        "gkx.operators.nonlinear.policies._build_implicit_operator", _fake_build
    )
    op = build_nonlinear_imex_operator(
        jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.2,
        terms=TermConfig(nonlinear=1.0),
        implicit_preconditioner="identity",
    )
    assert captured["preconditioner"] == "identity"
    assert op.shape == (1, 2, 2, 1, 1, 1)


def test_nonlinear_cfl_frequency_components_zero_and_finite() -> None:
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=2, Nm=2)

    zeros = FieldState(
        phi=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        apar=None,
        bpar=None,
    )
    ox, oy = _nonlinear_cfl_frequency_components(
        zeros,
        grid,
        cache,
        compressed_real_fft=False,
        kx_max=1.0,
        ky_max=1.0,
        kxfac=1.0,
        vpar_max=1.0,
        muB_max=1.0,
    )
    assert float(ox) == pytest.approx(0.0)
    assert float(oy) == pytest.approx(0.0)

    phi = (
        jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
        .at[1, 1, 0]
        .set(1.0 + 0.0j)
    )
    fields = FieldState(phi=phi, apar=0.5 * phi, bpar=0.25 * phi)
    ox, oy = _nonlinear_cfl_frequency_components(
        fields,
        grid,
        cache,
        compressed_real_fft=False,
        kx_max=1.0,
        ky_max=1.0,
        kxfac=1.0,
        vpar_max=1.0,
        muB_max=1.0,
    )
    assert np.isfinite(float(ox))
    assert np.isfinite(float(oy))
    assert float(ox) >= 0.0
    assert float(oy) >= 0.0


def test_nonlinear_cfl_frequency_components_recovers_spectral_gradient_cfl() -> None:
    """CFL estimate should reduce to the pseudo-spectral derivative maximum."""

    ny = nx = 4
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    grid = SimpleNamespace(ky=ky, kx=kx)
    cache = SimpleNamespace(kx_grid=kx_grid, ky_grid=ky_grid)

    def _sin_x_hat(amplitude: float) -> jnp.ndarray:
        # Coefficients for amplitude * sin(x); irfft/ifft paths multiply by N.
        field = jnp.zeros((ny, nx, 1), dtype=jnp.complex64)
        field = field.at[0, 1, 0].set(-0.5j * amplitude)
        return field.at[0, -1, 0].set(0.5j * amplitude)

    phi_amp = 1.25
    apar_amp = 0.5
    bpar_amp = 0.25
    vpar_max = 2.0
    muB_max = 3.0
    fields = FieldState(
        phi=_sin_x_hat(phi_amp),
        apar=_sin_x_hat(apar_amp),
        bpar=_sin_x_hat(bpar_amp),
    )

    omega_x, omega_y = _nonlinear_cfl_frequency_components(
        fields,
        grid,
        cache,
        compressed_real_fft=False,
        kx_max=5.0,
        ky_max=3.0,
        kxfac=-2.0,
        vpar_max=vpar_max,
        muB_max=muB_max,
    )

    expected_vmax_y = phi_amp + vpar_max * apar_amp + muB_max * bpar_amp
    assert float(omega_x) == pytest.approx(0.0, abs=1.0e-6)
    assert float(omega_y) == pytest.approx(
        0.5 * 2.0 * 3.0 * expected_vmax_y, rel=1.0e-6
    )


def test_apply_collision_split_and_nonlinear_wrapper_routing(monkeypatch) -> None:
    G = jnp.ones((2, 2, 1, 1, 1), dtype=jnp.complex64)
    damping = jnp.ones_like(G.real)
    implicit = _apply_collision_split(
        G, damping, jnp.asarray(0.1, dtype=jnp.float32), "implicit"
    )
    exp = _apply_collision_split(G, damping, jnp.asarray(0.1, dtype=jnp.float32), "exp")
    imex = _apply_collision_split(
        G, damping, jnp.asarray(0.1, dtype=jnp.float32), "imex"
    )
    rkc = _apply_collision_split(
        G, damping, jnp.asarray(0.1, dtype=jnp.float32), "rkc2"
    )
    assert np.all(np.isfinite(np.asarray(implicit)))
    assert np.all(np.isfinite(np.asarray(exp)))
    np.testing.assert_allclose(np.asarray(imex), np.asarray(implicit))
    np.testing.assert_allclose(np.asarray(rkc), np.asarray(exp))
    with pytest.raises(ValueError):
        _apply_collision_split(G, damping, jnp.asarray(0.1, dtype=jnp.float32), "bad")

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.integrate_nonlinear_imex_cached",
        lambda *args, **kwargs: ("imex", "fields"),
    )
    assert integrate_nonlinear_cached(
        G,
        SimpleNamespace(
            ky=jnp.asarray([0.0, 0.2]),
            kx=jnp.asarray([0.0]),
            Jl=None,
            JlB=None,
            laguerre_to_grid=None,
            laguerre_to_spectral=None,
            laguerre_roots=None,
            laguerre_j0=None,
            laguerre_j1_over_alpha=None,
            b=None,
            dealias_mask=None,
            kxfac=1.0,
        ),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
    ) == ("imex", "fields")

    captured: dict[str, object] = {}

    def _fake_scan(rhs_fn, G0, dt, steps, **kwargs):
        captured["project_state"] = kwargs.get("project_state")
        captured["return_fields"] = kwargs.get("return_fields")
        return G0, FieldState(
            phi=jnp.zeros((4, 2, 2), dtype=jnp.complex64), apar=None, bpar=None
        )

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.integrate_nonlinear_scan",
        _fake_scan,
    )
    out_G, out_fields = integrate_nonlinear_cached(
        jnp.zeros((1, 4, 2, 2), dtype=jnp.complex64),
        SimpleNamespace(
            ky=jnp.asarray([0.0, 0.2, -0.2, -0.4]),
            kx=jnp.asarray([0.0, 0.5]),
            Jl=None,
            JlB=None,
            laguerre_to_grid=None,
            laguerre_to_spectral=None,
            laguerre_roots=None,
            laguerre_j0=None,
            laguerre_j1_over_alpha=None,
            b=None,
            dealias_mask=jnp.ones((4, 2), dtype=bool),
            kxfac=1.0,
        ),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk2",
        compressed_real_fft=True,
    )
    assert out_G.shape == (1, 4, 2, 2)
    assert captured["project_state"] is not None
    assert captured["return_fields"] is True
    assert out_fields.phi.shape == (4, 2, 2)


def test_integrate_nonlinear_builds_cache_and_rejects_bad_shape(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: "geom_eff",
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.build_linear_cache",
        lambda grid, geom, params, Nl, Nm: calls.append((Nl, Nm)) or "cache",
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.integrate_nonlinear_cached",
        lambda G0, cache, params, dt, steps, **kwargs: ("G_out", "fields_out"),
    )

    assert integrate_nonlinear(
        jnp.zeros((2, 3, 1, 1, 4), dtype=jnp.complex64),
        SimpleNamespace(z=np.array([-1.0, 0.0, 1.0, 2.0])),
        object(),
        object(),
        dt=0.1,
        steps=2,
    ) == ("G_out", "fields_out")
    assert integrate_nonlinear(
        jnp.zeros((1, 2, 3, 1, 1, 4), dtype=jnp.complex64),
        SimpleNamespace(z=np.array([-1.0, 0.0, 1.0, 2.0])),
        object(),
        object(),
        dt=0.1,
        steps=2,
    ) == ("G_out", "fields_out")
    assert calls == [(2, 3), (2, 3)]

    with pytest.raises(ValueError):
        integrate_nonlinear(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            SimpleNamespace(z=np.array([0.0])),
            object(),
            object(),
            dt=0.1,
            steps=2,
        )


def test_nonlinear_diagnostics_route_and_state_reject_imex(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.integrate_nonlinear_imex_diagnostics",
        lambda *args, **kwargs: ("t_imex", "diag_imex"),
    )
    assert integrate_nonlinear_explicit_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
    ) == ("t_imex", "diag_imex")

    with pytest.raises(ValueError):
        integrate_nonlinear_explicit_diagnostics_state(
            jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            method="imex",
        )


def test_integrate_nonlinear_explicit_diagnostics_explicit_and_state_routes(
    monkeypatch,
) -> None:
    payload = ("t_explicit", "diag_explicit", "G_final", "fields_final")
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._integrate_nonlinear_explicit_diagnostics_impl",
        lambda *args, **kwargs: payload,
    )

    out = integrate_nonlinear_explicit_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk3",
    )
    assert out == ("t_explicit", "diag_explicit")

    out_state = integrate_nonlinear_explicit_diagnostics_state(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk3",
    )
    assert out_state == payload


def test_explicit_diagnostics_impl_rejects_imex_and_bad_state_rank(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: geom,
    )
    grid = SimpleNamespace(z=np.array([0.0]))

    with pytest.raises(
        ValueError,
        match="Final-state runtime diagnostics helper only supports explicit methods",
    ):
        _integrate_nonlinear_explicit_diagnostics_impl(
            jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.complex64),
            grid,
            object(),
            object(),
            dt=0.1,
            steps=1,
            method="imex",
            cache=object(),
        )

    with pytest.raises(ValueError, match="G0 must have shape"):
        _integrate_nonlinear_explicit_diagnostics_impl(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            grid,
            object(),
            object(),
            dt=0.1,
            steps=1,
            method="rk2",
            cache=None,
        )


def test_integrate_nonlinear_explicit_diagnostics_forwarding_contracts(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_impl(*args, **kwargs):
        captured.update(kwargs)
        return ("t", "diag", "G_final", "fields_final")

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._integrate_nonlinear_explicit_diagnostics_impl",
        _fake_impl,
    )

    out = integrate_nonlinear_explicit_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk4",
        fixed_dt=False,
        dt_min=1.0e-4,
        dt_max=0.2,
        cfl=0.7,
        cfl_fac=0.5,
        collision_split=True,
        collision_scheme="exp",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    assert out == ("t", "diag")
    assert captured["fixed_dt"] is False
    assert captured["collision_split"] is True
    assert captured["collision_scheme"] == "exp"
    assert captured["fixed_mode_ky_index"] == 1
    assert captured["fixed_mode_kx_index"] == 0

    captured.clear()
    out_state = integrate_nonlinear_explicit_diagnostics_state(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="rk4",
        fixed_dt=False,
        fixed_mode_ky_index=0,
        fixed_mode_kx_index=1,
    )
    assert out_state == ("t", "diag", "G_final", "fields_final")
    assert captured["fixed_dt"] is False
    assert captured["fixed_mode_ky_index"] == 0
    assert captured["fixed_mode_kx_index"] == 1


def test_integrate_nonlinear_explicit_diagnostics_imex_forwarding_contracts(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_imex(*args, **kwargs):
        captured.update(kwargs)
        return ("t_imex", "diag_imex")

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.integrate_nonlinear_imex_diagnostics",
        _fake_imex,
    )

    out = integrate_nonlinear_explicit_diagnostics(
        jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=2,
        method="semi-implicit",
        collision_split=True,
        collision_scheme="exp",
        implicit_preconditioner="identity",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
        show_progress=True,
    )

    assert out == ("t_imex", "diag_imex")
    assert captured["collision_split"] is True
    assert captured["collision_scheme"] == "exp"
    assert captured["implicit_preconditioner"] == "identity"
    assert captured["fixed_mode_ky_index"] == 1
    assert captured["fixed_mode_kx_index"] == 0
    assert captured["show_progress"] is True


def test_explicit_diagnostics_impl_applies_fixed_mode_collision_and_stride(
    monkeypatch,
) -> None:
    grid = SimpleNamespace(
        ky=np.array([0.0, 0.2], dtype=float),
        kx=np.array([0.0], dtype=float),
        z=np.array([0.0, 1.0], dtype=float),
        dealias_mask=np.ones((2, 1), dtype=bool),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        kxfac=1.0,
        l=jnp.asarray([0], dtype=jnp.int32),
        m=jnp.asarray([[0]], dtype=jnp.int32),
        lb_lam=jnp.ones((1, 1, 1, 2, 1, 2), dtype=jnp.float32),
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]), nu=0.2)
    phi = jnp.ones((2, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=None, bpar=None)

    def _resolved_tuple():
        return (
            jnp.asarray(1.0),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )

    def _split_flux_tuple():
        return (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        )

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: geom,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.fieldline_quadrature_weights",
        lambda geom, grid: (
            jnp.ones((grid.z.size,), dtype=jnp.float32),
            jnp.asarray(1.0),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._diagnostic_omega_mode_mask",
        lambda grid, cache, **kwargs: jnp.ones((2, 1), dtype=bool),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._linear_frequency_bound",
        lambda *args, **kwargs: np.array([0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._laguerre_velocity_max",
        lambda nl: 0.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.nonlinear_rhs_cached",
        lambda G, cache, params, terms, **kwargs: (jnp.ones_like(G), fields),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.compute_fields_cached",
        lambda *args, **kwargs: fields,
    )

    def _fake_growth(phi, phi_prev, dt_step, z_index, mask):
        return (
            jnp.ones((2, 1), dtype=jnp.float32) * 2.0,
            jnp.ones((2, 1), dtype=jnp.float32) * -3.0,
        )

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._instantaneous_growth_rate_step",
        _fake_growth,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.phi2_resolved",
        lambda *args, **kwargs: _resolved_tuple(),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.distribution_free_energy_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.electrostatic_field_energy_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.magnetic_vector_potential_energy_resolved",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.heat_flux_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.heat_flux_channel_resolved_species",
        lambda *args, **kwargs: (
            _split_flux_tuple(),
            _split_flux_tuple(),
            _split_flux_tuple(),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.particle_flux_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.particle_flux_channel_resolved_species",
        lambda *args, **kwargs: (
            _split_flux_tuple(),
            _split_flux_tuple(),
            _split_flux_tuple(),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.turbulent_heating_resolved_species",
        lambda *args, **kwargs: (
            jnp.ones((1,), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._collision_damping",
        lambda *args, **kwargs: jnp.ones((1, 1, 2, 1, 2), dtype=jnp.float32),
    )

    def _fake_collision_split(G_state, damping, dt_local, scheme):
        assert scheme == "exp"
        return G_state + 5.0

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._apply_collision_split",
        _fake_collision_split,
    )

    G0 = jnp.zeros((1, 1, 2, 1, 2), dtype=jnp.complex64)
    G0 = G0.at[..., 1:2, 0:1, :].set(7.0 + 0.0j)

    for method in ("rk3_classic", "rk4", "k10"):
        t_branch, diag_branch, G_branch, _fields_branch = (
            _integrate_nonlinear_explicit_diagnostics_impl(
                G0,
                grid,
                SimpleNamespace(),
                params,
                dt=0.1,
                steps=1,
                method=method,
                cache=cache,
                terms=TermConfig(),
                sample_stride=1,
                diagnostics_stride=1,
                omega_ky_index=1,
                omega_kx_index=0,
            )
        )
        np.testing.assert_allclose(np.asarray(t_branch), [0.1])
        np.testing.assert_allclose(np.asarray(diag_branch.gamma_t), [2.0])
        assert G_branch.shape == G0.shape

    with pytest.raises(ValueError):
        _integrate_nonlinear_explicit_diagnostics_impl(
            G0,
            grid,
            SimpleNamespace(),
            params,
            dt=0.1,
            steps=1,
            method="not-a-method",
            cache=cache,
        )

    t, diag, G_final, fields_final = _integrate_nonlinear_explicit_diagnostics_impl(
        G0,
        grid,
        SimpleNamespace(),
        params,
        dt=0.1,
        steps=3,
        method="euler",
        cache=cache,
        terms=TermConfig(collisions=1.0, hypercollisions=0.0),
        sample_stride=1,
        diagnostics_stride=2,
        collision_split=True,
        collision_scheme="exp",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    np.testing.assert_allclose(np.asarray(t), [0.1, 0.3])
    np.testing.assert_allclose(np.asarray(diag.gamma_t), [2.0, 2.0])
    np.testing.assert_allclose(np.asarray(G_final[..., 1:2, 0:1, :]), 7.0)
    assert np.all(np.asarray(G_final[..., 0:1, 0:1, :]) > 0.0)
    assert diag.resolved is not None
    for field_info in dataclass_fields(ResolvedDiagnostics):
        resolved_value = getattr(diag.resolved, field_info.name)
        assert resolved_value is not None
        assert np.asarray(resolved_value).shape[0] == np.asarray(t).shape[0]
    assert fields_final.phi.shape == (2, 1, 2)


def test_explicit_diagnostics_resolved_schema_and_sample_axis(monkeypatch) -> None:
    grid = SimpleNamespace(
        ky=np.array([0.0, 0.2], dtype=float),
        kx=np.array([0.0, 0.5], dtype=float),
        z=np.array([0.0, 1.0], dtype=float),
        dealias_mask=np.ones((2, 2), dtype=bool),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        kxfac=1.0,
        l=jnp.asarray([0], dtype=jnp.int32),
        m=jnp.asarray([[0]], dtype=jnp.int32),
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    fields_state = FieldState(
        phi=jnp.ones((2, 2, 2), dtype=jnp.complex64) * (1.0 + 1.0j),
        apar=None,
        bpar=None,
    )

    def _marker(value: float) -> jnp.ndarray:
        return jnp.full((1,), value, dtype=jnp.float32)

    def _split_flux_tuple(
        base: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return tuple(_marker(base + offset) for offset in range(5))

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: geom,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.fieldline_quadrature_weights",
        lambda geom, grid: (
            jnp.ones((grid.z.size,), dtype=jnp.float32),
            jnp.asarray(1.0),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._diagnostic_omega_mode_mask",
        lambda grid, cache, **kwargs: jnp.ones((2, 2), dtype=bool),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._linear_frequency_bound",
        lambda *args, **kwargs: np.array([0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._laguerre_velocity_max",
        lambda nl: 0.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.nonlinear_rhs_cached",
        lambda G, cache, params, terms, **kwargs: (jnp.zeros_like(G), fields_state),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.compute_fields_cached",
        lambda *args, **kwargs: fields_state,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._instantaneous_growth_rate_step",
        lambda *args, **kwargs: (
            jnp.full((2, 2), 1.25, dtype=jnp.float32),
            jnp.full((2, 2), -0.75, dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.phi2_resolved",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(100, 108)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.zonal_phi_mode_kxt",
        lambda *args, **kwargs: _marker(108),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.zonal_phi_line_kxt",
        lambda *args, **kwargs: _marker(109),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.distribution_free_energy_resolved",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(110, 116)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.electrostatic_field_energy_resolved",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(116, 121)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.magnetic_vector_potential_energy_resolved",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(121, 126)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.heat_flux_resolved_species",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(126, 131)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.heat_flux_channel_resolved_species",
        lambda *args, **kwargs: (
            _split_flux_tuple(130),
            _split_flux_tuple(134),
            _split_flux_tuple(138),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.particle_flux_resolved_species",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(143, 148)),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.particle_flux_channel_resolved_species",
        lambda *args, **kwargs: (
            _split_flux_tuple(147),
            _split_flux_tuple(151),
            _split_flux_tuple(155),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.turbulent_heating_resolved_species",
        lambda *args, **kwargs: tuple(_marker(v) for v in range(160, 165)),
    )

    t, diag, _G_final, _fields_final = _integrate_nonlinear_explicit_diagnostics_impl(
        jnp.zeros((1, 1, 1, 2, 2, 2), dtype=jnp.complex64),
        grid,
        SimpleNamespace(),
        params,
        dt=0.1,
        steps=3,
        method="euler",
        cache=cache,
        terms=TermConfig(nonlinear=0.0),
        sample_stride=2,
        diagnostics_stride=1,
    )

    np.testing.assert_allclose(np.asarray(t), [0.1, 0.3])
    np.testing.assert_allclose(np.asarray(diag.energy_t), [347.0, 347.0])
    assert diag.resolved is not None
    for field_info in dataclass_fields(ResolvedDiagnostics):
        resolved_value = np.asarray(getattr(diag.resolved, field_info.name))
        assert resolved_value.shape[0] == 2

    np.testing.assert_allclose(np.asarray(diag.resolved.Phi2_kxt)[:, 0], [101.0, 101.0])
    np.testing.assert_allclose(
        np.asarray(diag.resolved.Phi_zonal_line_kxt)[:, 0], [109.0, 109.0]
    )
    np.testing.assert_allclose(np.asarray(diag.resolved.Wg_lmst)[:, 0], [115.0, 115.0])
    np.testing.assert_allclose(
        np.asarray(diag.resolved.HeatFluxBpar_zst)[:, 0], [142.0, 142.0]
    )
    np.testing.assert_allclose(
        np.asarray(diag.resolved.ParticleFluxBpar_zst)[:, 0], [159.0, 159.0]
    )
    np.testing.assert_allclose(
        np.asarray(diag.resolved.TurbulentHeating_zst)[:, 0], [164.0, 164.0]
    )


def test_fixed_small_amplitude_mode_gamma_omega_are_finite(monkeypatch) -> None:
    grid = SimpleNamespace(
        ky=np.array([0.0, 0.2, -0.2, -0.4], dtype=float),
        kx=np.array([0.0, 0.5], dtype=float),
        z=np.array([0.0, 1.0], dtype=float),
        dealias_mask=np.ones((4, 2), dtype=bool),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        kxfac=1.0,
        l=jnp.asarray([0], dtype=jnp.int32),
        m=jnp.asarray([[0]], dtype=jnp.int32),
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    amplitude = jnp.asarray(1.0e-7 + 2.0e-7j, dtype=jnp.complex64)
    G0 = jnp.zeros((1, 1, 1, 4, 2, 2), dtype=jnp.complex64)
    G0 = G0.at[..., 1, 0, :].set(amplitude)

    def _fields_from_state(G_state, *args, **kwargs):
        del args, kwargs
        return FieldState(phi=G_state[0, 0, 0], apar=None, bpar=None)

    def _rhs(G_state, cache, params, terms, **kwargs):
        del cache, params, terms, kwargs
        drive = jnp.ones_like(G_state) * jnp.asarray(
            3.0e-7 - 2.0e-7j, dtype=G_state.dtype
        )
        return drive, _fields_from_state(G_state)

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: geom,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.fieldline_quadrature_weights",
        lambda geom, grid: (
            jnp.ones((grid.z.size,), dtype=jnp.float32),
            jnp.asarray(1.0),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._diagnostic_omega_mode_mask",
        lambda grid, cache, **kwargs: jnp.ones((4, 2), dtype=bool),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._linear_frequency_bound",
        lambda *args, **kwargs: np.array([0.0, 0.0, 0.0], dtype=float),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration._laguerre_velocity_max",
        lambda nl: 0.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.nonlinear_rhs_cached", _rhs
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.compute_fields_cached",
        _fields_from_state,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.distribution_free_energy",
        lambda *args, **kwargs: jnp.asarray(0.0, dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.electrostatic_field_energy",
        lambda *args, **kwargs: jnp.asarray(0.0, dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.magnetic_vector_potential_energy",
        lambda *args, **kwargs: jnp.asarray(0.0, dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.heat_flux_species",
        lambda *args, **kwargs: jnp.zeros((1,), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.particle_flux_species",
        lambda *args, **kwargs: jnp.zeros((1,), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.turbulent_heating_species",
        lambda *args, **kwargs: jnp.zeros((1,), dtype=jnp.float32),
    )

    _t, diag, G_final, _fields_final = _integrate_nonlinear_explicit_diagnostics_impl(
        G0,
        grid,
        SimpleNamespace(),
        params,
        dt=0.05,
        steps=2,
        method="euler",
        cache=cache,
        terms=TermConfig(nonlinear=1.0),
        compressed_real_fft=False,
        z_index=0,
        omega_ky_index=1,
        omega_kx_index=0,
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
        resolved_diagnostics=False,
    )

    assert np.isfinite(np.asarray(diag.gamma_t)).all()
    assert np.isfinite(np.asarray(diag.omega_t)).all()
    np.testing.assert_allclose(np.asarray(diag.gamma_t), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(diag.omega_t), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(diag.phi_mode_t), np.asarray(amplitude), rtol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(G_final[..., 1:2, 0:1, :]), np.asarray(G0[..., 1:2, 0:1, :])
    )


def test_integrate_nonlinear_imex_diagnostics_rejects_bad_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.diagnostic_integration.ensure_flux_tube_geometry_data",
        lambda geom, z: geom,
    )
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_diagnostics(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            SimpleNamespace(z=np.array([0.0])),
            object(),
            object(),
            dt=0.1,
            steps=2,
        )


def test_integrate_nonlinear_imex_cached_shape_mismatch_and_zero_nonlinear(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    implicit_operator = SimpleNamespace(
        shape=(1, 2, 2, 1, 1, 2),
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_cached(
            G0,
            SimpleNamespace(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            implicit_operator=implicit_operator,
        )

    cache = SimpleNamespace(
        Jl=None,
        JlB=None,
        sqrt_m=None,
        sqrt_m_p1=None,
        kx_grid=None,
        ky_grid=None,
        dealias_mask=None,
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )
    good_operator = SimpleNamespace(
        shape=G0.shape,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )

    gmres_calls: list[int] = []

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.imex.gmres",
        lambda matvec, rhs, **kwargs: SimpleNamespace(
            x=gmres_calls.append(rhs.size) or rhs,
            converged=True,
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.assemble_rhs_cached_jit",
        lambda G, cache, params, terms, **kwargs: (
            jnp.zeros_like(G),
            FieldState(
                phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64), apar=None, bpar=None
            ),
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.compute_fields_cached",
        lambda G, cache, params, terms=None: (_ for _ in ()).throw(
            AssertionError("nonlinear path should stay off")
        ),
    )

    G_out, fields_t = integrate_nonlinear_imex_cached(
        G0,
        cache,
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        dt=0.1,
        steps=2,
        terms=TermConfig(nonlinear=0.0),
        implicit_operator=good_operator,
    )

    assert gmres_calls
    assert G_out.shape == G0.shape
    assert fields_t.phi.shape[0] == 2


def test_integrate_nonlinear_imex_cached_uses_electrostatic_linear_path(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(
        phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64), apar=None, bpar=None
    )
    implicit_operator = SimpleNamespace(
        shape=G0.shape,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )
    calls: list[str] = []

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.assemble_rhs_cached_electrostatic_jit",
        lambda G, cache, params, terms, **kwargs: (
            calls.append("electrostatic") or jnp.zeros_like(G),
            fields,
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.assemble_rhs_cached_jit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic linear RHS should not run")
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.imex.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )

    G_out, fields_t = integrate_nonlinear_imex_cached(
        G0,
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
        steps=1,
        terms=TermConfig(nonlinear=0.0, apar=0.0, bpar=0.0),
        implicit_operator=implicit_operator,
    )

    assert calls == ["electrostatic", "electrostatic"]
    assert G_out.shape == G0.shape
    assert fields_t.phi.shape[0] == 1


def test_integrate_nonlinear_imex_cached_builds_operator_and_nonlinear_term(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((1, 1, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        Jl=None,
        JlB=None,
        sqrt_m=None,
        sqrt_m_p1=None,
        kx_grid=None,
        ky_grid=None,
        dealias_mask=None,
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    fields = FieldState(
        phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64), apar=None, bpar=None
    )

    build_calls: list[float] = []
    nonlinear_calls: list[bool] = []

    def _fake_build_operator(
        G_in, cache_in, params_in, dt, linear_terms, implicit_preconditioner
    ):
        del cache_in, params_in, linear_terms, implicit_preconditioner
        build_calls.append(float(dt))
        return (
            G_in,
            tuple(G_in.shape),
            G_in.size,
            jnp.asarray(dt, dtype=jnp.float32),
            None,
            lambda x: x,
            False,
        )

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration._build_implicit_operator",
        _fake_build_operator,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.compute_fields_cached",
        lambda G, cache, params, terms=None, external_phi=None: fields,
    )

    def _fake_nonlinear_em(G, **kwargs):
        assert kwargs["weight"].shape == ()
        nonlinear_calls.append(True)
        return jnp.ones_like(G)

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.nonlinear_em_contribution",
        _fake_nonlinear_em,
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.state_integration.assemble_rhs_cached_jit",
        lambda G, cache, params, terms, **kwargs: (jnp.zeros_like(G), fields),
    )
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.imex.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )

    G_out, fields_t = integrate_nonlinear_imex_cached(
        G0,
        cache,
        params,
        dt=0.2,
        steps=2,
        terms=TermConfig(nonlinear=0.5),
    )

    assert build_calls == [0.2]
    assert nonlinear_calls == [True]
    np.testing.assert_allclose(np.asarray(G_out), 0.4)
    assert fields_t.phi.shape[0] == 2
