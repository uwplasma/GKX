"""Tests for nonlinear explicit scan integrator utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.diagnostics.analysis import estimate_observed_order
from gkx.terms.config import FieldState
from gkx.solvers.nonlinear.explicit import (
    _checkpoint_block_size,
    checkpointed_explicit_scan,
    integrate_nonlinear_scan,
)
from gkx.terms.nonlinear import (
    exb_nonlinear_contribution,
    placeholder_nonlinear_contribution,
)

_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _linear_rhs(rate: complex):
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        dG = rate * G
        phi = jnp.sum(G, axis=0)
        return dG, FieldState(phi=phi)

    return rhs_fn


def test_checkpointed_scan_matches_discrete_primal_and_reverse_with_tail() -> None:
    """Block checkpointing must change storage, not the discrete RK map."""

    steps = 19  # sqrt schedule has three full blocks and a non-empty tail
    indices = jnp.arange(steps, dtype=jnp.int32)

    def objective(parameter: jnp.ndarray, checkpoint: bool) -> jnp.ndarray:
        def step(carry: jnp.ndarray, index: jnp.ndarray):
            weight = jnp.asarray(index + 1, dtype=carry.dtype) / steps
            updated = jnp.tanh(carry + weight * parameter)
            return updated, (updated, updated * updated)

        final, (states, squares) = checkpointed_explicit_scan(
            step,
            jnp.asarray([0.2, -0.4], dtype=jnp.float32),
            indices,
            checkpoint=checkpoint,
        )
        return jnp.sum(final) + 0.01 * jnp.sum(states + squares)

    parameter = jnp.asarray(0.03, dtype=jnp.float32)
    plain = jax.value_and_grad(lambda value: objective(value, False))(parameter)
    blocked = jax.value_and_grad(lambda value: objective(value, True))(parameter)

    np.testing.assert_allclose(np.asarray(blocked[0]), np.asarray(plain[0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(blocked[1]), np.asarray(plain[1]), rtol=1e-6)
    assert _checkpoint_block_size(steps) == 5
    assert _checkpoint_block_size(2048) == 46


@pytest.mark.parametrize(
    ("method", "one_step_factor"),
    [
        ("euler", lambda a: 1.0 + a),
        ("rk2", lambda a: 1.0 + a + 0.5 * a * a),
        ("rk3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk3_heun", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        ("rk3_classic", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
        (
            "rk4",
            lambda a: 1.0
            + a
            + 0.5 * a * a
            + (a * a * a) / 6.0
            + (a * a * a * a) / 24.0,
        ),
        ("sspx3", lambda a: 1.0 + a + 0.5 * a * a + (a * a * a) / 6.0),
    ],
)
def test_integrate_nonlinear_scan_methods_match_linear_amplification(
    method, one_step_factor
) -> None:
    dt = 0.1
    steps = 4
    rate = 0.3 - 0.2j
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    G0_ref = jnp.array(G0)
    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(rate),
        G0,
        dt,
        steps,
        method=method,
        checkpoint=True,
    )
    a = rate * dt
    expected = (one_step_factor(a) ** steps) * G0_ref
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == steps
    assert jnp.allclose(G_final, expected, rtol=3.0e-3, atol=3.0e-3)


def test_integrate_nonlinear_scan_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        integrate_nonlinear_scan(
            _linear_rhs(0.1 + 0.0j),
            jnp.ones((2, 2), dtype=jnp.complex64),
            0.1,
            4,
            method="bad",
        )


def test_integrate_nonlinear_scan_projects_each_stage() -> None:
    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        flipped = jnp.flip(G, axis=-2)
        return 1j * flipped, FieldState(phi=jnp.sum(G, axis=0))

    def projector(G: jnp.ndarray) -> jnp.ndarray:
        pos = G[..., :3, :]
        neg = jnp.conj(pos[..., 1:2, :])[..., ::-1, :]
        return jnp.concatenate([pos, neg], axis=-2)

    G0 = jnp.asarray(
        [[1.0 + 0.0j], [2.0 + 1.0j], [-3.0 + 0.5j], [7.0 - 2.0j]], dtype=jnp.complex64
    )
    G_final, _fields = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="rk4",
        project_state=projector,
    )
    assert jnp.allclose(G_final[..., 3, :], jnp.conj(G_final[..., 1, :]))


def test_integrate_nonlinear_scan_rk3_alias_matches_heun_variant() -> None:
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)
    out_rk3, _ = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3"
    )
    out_heun, _ = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), jnp.array(G0), 0.1, 3, method="rk3_heun"
    )
    assert jnp.allclose(out_rk3, out_heun)


def test_integrate_nonlinear_scan_final_only_skips_field_history_rhs() -> None:
    calls = {"rhs": 0}

    def rhs_fn(G: jnp.ndarray) -> tuple[jnp.ndarray, FieldState]:
        calls["rhs"] += 1
        return 0.2 * G, FieldState(phi=jnp.sum(G, axis=0))

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    expected = (1.0 + 0.1 * 0.2) ** 2 * G0
    out = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="euler",
        return_fields=False,
    )

    assert out.shape == G0.shape
    assert jnp.allclose(out, expected)
    assert calls["rhs"] == 1


def test_integrate_nonlinear_scan_accepts_dynamic_and_static_rhs_arguments() -> None:
    def rhs_fn(
        G: jnp.ndarray, rate: jnp.ndarray, offset: float
    ) -> tuple[jnp.ndarray, FieldState]:
        return rate * G + offset, FieldState(phi=jnp.sum(G, axis=0))

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    expected = np.asarray(G0).copy()
    out = integrate_nonlinear_scan(
        rhs_fn,
        G0,
        0.1,
        2,
        method="euler",
        return_fields=False,
        rhs_args=(jnp.asarray(0.2, dtype=jnp.float32),),
        rhs_static_args=(0.05,),
    )

    for _ in range(2):
        expected = expected + 0.1 * (0.2 * expected + 0.05)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1.0e-6)


@pytest.mark.parametrize(
    ("method", "expected_order", "min_observed_order"),
    [
        ("euler", 1.0, 0.9),
        ("rk2", 2.0, 1.75),
        ("rk3", 3.0, 2.6),
        ("rk3_heun", 3.0, 2.6),
        ("rk3_classic", 3.0, 2.6),
        ("rk4", 4.0, 3.3),
        ("sspx3", 3.0, 2.6),
    ],
)
def test_integrate_nonlinear_scan_observed_order_against_exact_solution(
    method: str,
    expected_order: float,
    min_observed_order: float,
) -> None:
    rate = -1.1 + 0.7j
    t_final = 0.8
    G0 = jnp.asarray([[1.0 + 0.1j, -0.4 + 0.2j]], dtype=jnp.complex64)
    exact = np.exp(rate * t_final) * np.asarray(G0)

    errors: list[float] = []
    dts: list[float] = []
    for steps in (2, 4, 8, 16):
        dt = t_final / steps
        out, _ = integrate_nonlinear_scan(
            _linear_rhs(rate), jnp.array(G0), dt, steps, method=method
        )
        err = float(np.max(np.abs(np.asarray(out) - exact)))
        errors.append(err)
        dts.append(dt)

    metrics = estimate_observed_order(np.asarray(dts), np.asarray(errors))
    assert metrics.orders.size > 0
    assert metrics.asymptotic_order >= min_observed_order
    assert metrics.asymptotic_order <= expected_order + 0.6


def test_integrate_nonlinear_scan_k10_branch_is_finite_and_shape_preserving() -> None:
    G0 = jnp.asarray([[1.0 + 0.0j, 0.5 + 0.25j]], dtype=jnp.complex64)

    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(0.3 - 0.2j), G0, 0.1, 2, method="k10"
    )

    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == 2
    assert np.all(np.isfinite(np.asarray(G_final)))


def test_integrate_nonlinear_scan_show_progress_callback_path(monkeypatch) -> None:
    from gkx.utils import callbacks

    callback_calls: list[int] = []

    monkeypatch.setattr(
        callbacks, "should_emit_progress", lambda idx, steps: jnp.asarray(True)
    )

    def _fake_print_callback(state, idx, *args):
        del idx, args
        callback_calls.append(0)
        return state

    monkeypatch.setattr(callbacks, "print_callback", _fake_print_callback)

    G0 = jnp.asarray([[1.0 + 0.0j]], dtype=jnp.complex64)
    G_final, fields = integrate_nonlinear_scan(
        _linear_rhs(0.0 + 0.0j),
        G0,
        0.1,
        1,
        method="euler",
        show_progress=True,
    )

    assert callback_calls == [0]
    assert G_final.shape == G0.shape
    assert fields.phi.shape[0] == 1


def test_nonlinear_placeholders() -> None:
    G = jnp.ones((3, 4, 1), dtype=jnp.complex64)
    out = placeholder_nonlinear_contribution(G, weight=jnp.asarray(2.0))
    assert jnp.allclose(out, 0.0)
    exb = exb_nonlinear_contribution(
        G,
        phi=jnp.ones((3, 4, 1), dtype=jnp.complex64),
        dealias_mask=jnp.ones((3, 4), dtype=bool),
        kx_grid=jnp.ones((3, 4), dtype=jnp.float32),
        ky_grid=jnp.ones((3, 4), dtype=jnp.float32),
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    assert exb.shape == G.shape
