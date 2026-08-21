from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx.solvers.nonlinear.imex as imex_module
import gkx.solvers.nonlinear.imex_diagnostics as imex_diagnostics
from gkx.solvers.nonlinear.imex import (
    advance_imex_nonlinear_state,
    imex_fixed_point_guess,
    integrate_cached_imex_scan,
    make_imex_diagnostic_step,
    make_imex_nonlinear_term,
    make_imex_solve_step,
    run_imex_diagnostic_scan,
    solve_imex_step,
)
from gkx.terms.config import FieldState


def test_imex_diagnostic_helpers_have_canonical_owner() -> None:
    assert imex_module.advance_imex_nonlinear_state is (
        imex_diagnostics.advance_imex_nonlinear_state
    )
    assert imex_module.make_imex_diagnostic_step is (
        imex_diagnostics.make_imex_diagnostic_step
    )
    assert imex_module.run_imex_diagnostic_scan is (
        imex_diagnostics.run_imex_diagnostic_scan
    )


def test_imex_fixed_point_guess_applies_linear_predictor_iterations() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return g, None

    out = imex_fixed_point_guess(
        jnp.asarray([0.0], dtype=jnp.float32),
        jnp.asarray([1.0], dtype=jnp.float32),
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=2,
        implicit_relax=1.0,
    )

    np.testing.assert_allclose(np.asarray(out), [1.1], rtol=1e-6)


def test_solve_imex_step_identity_system_returns_rhs_shape() -> None:
    def linear_rhs(g, *_args, **_kwargs):
        return jnp.zeros_like(g), None

    G_rhs = jnp.asarray([[2.0]], dtype=jnp.float32)
    out = solve_imex_step(
        jnp.zeros_like(G_rhs),
        G_rhs,
        linear_rhs_fn=linear_rhs,
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        linear_cfg=SimpleNamespace(),
        external_phi=None,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        implicit_iters=0,
        implicit_relax=1.0,
        matvec=lambda flat: flat,
        shape=tuple(G_rhs.shape),
        implicit_tol=1.0e-8,
        implicit_maxiter=20,
        implicit_restart=5,
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(G_rhs), rtol=1e-6)


def test_make_imex_nonlinear_term_forwards_injected_kernels() -> None:
    seen: dict[str, object] = {}

    def fields_fn(*_args, **_kwargs):
        return "fields"

    def contribution_fn(*_args, **_kwargs):
        return "contribution"

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        seen["G"] = G
        seen["cache"] = cache
        seen["params"] = params
        seen["terms"] = terms
        seen["fields_fn"] = kwargs["fields_fn"]
        seen["contribution_fn"] = kwargs["nonlinear_contribution_fn"]
        seen["real_dtype"] = kwargs["real_dtype"]
        seen["external_phi"] = kwargs["external_phi"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return G + 2.0

    cache = SimpleNamespace(name="cache")
    params = SimpleNamespace(name="params")
    terms = SimpleNamespace(name="terms")
    term = make_imex_nonlinear_term(
        cache,
        params,
        terms,
        real_dtype=jnp.float32,
        external_phi=3.0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
    )
    G = jnp.asarray([1.0], dtype=jnp.float32)

    out = term(G)

    np.testing.assert_allclose(np.asarray(out), [3.0])
    assert seen["G"] is G
    assert seen["cache"] is cache
    assert seen["params"] is params
    assert seen["terms"] is terms
    assert seen["fields_fn"] is fields_fn
    assert seen["contribution_fn"] is contribution_fn
    assert seen["real_dtype"] is jnp.float32
    assert seen["external_phi"] == 3.0
    assert seen["compressed_real_fft"] is False
    assert seen["laguerre_mode"] == "spectral"


def test_make_imex_solve_step_forwards_solver_policy() -> None:
    seen: dict[str, object] = {}

    def solve_step_fn(G_in, G_rhs, **kwargs):
        seen["G_in"] = G_in
        seen["G_rhs"] = G_rhs
        seen.update(kwargs)
        return G_rhs + 4.0

    def linear_rhs_fn(g, *_args, **_kwargs):
        return g, None

    def matvec(flat):
        return flat

    def precond(flat):
        return flat

    solve_step = make_imex_solve_step(
        linear_rhs_fn=linear_rhs_fn,
        cache=SimpleNamespace(name="cache"),
        params=SimpleNamespace(name="params"),
        linear_cfg=SimpleNamespace(name="linear"),
        external_phi=None,
        dt_val=jnp.asarray(0.2, dtype=jnp.float32),
        implicit_iters=3,
        implicit_relax=0.5,
        matvec=matvec,
        shape=(1,),
        implicit_tol=1.0e-5,
        implicit_maxiter=7,
        implicit_restart=2,
        precond_op=precond,
        solve_step_fn=solve_step_fn,
    )
    G_in = jnp.asarray([1.0], dtype=jnp.float32)
    G_rhs = jnp.asarray([2.0], dtype=jnp.float32)

    out = solve_step(G_in, G_rhs)

    np.testing.assert_allclose(np.asarray(out), [6.0])
    assert seen["G_in"] is G_in
    assert seen["G_rhs"] is G_rhs
    assert seen["linear_rhs_fn"] is linear_rhs_fn
    assert seen["matvec"] is matvec
    assert seen["precond_op"] is precond
    assert seen["implicit_iters"] == 3
    assert seen["implicit_relax"] == 0.5
    assert seen["shape"] == (1,)
    assert seen["implicit_tol"] == 1.0e-5
    assert seen["implicit_maxiter"] == 7
    assert seen["implicit_restart"] == 2


def test_imex_operator_resolution_and_state_shape_policies() -> None:
    provided = SimpleNamespace(name="provided")
    built: list[dict[str, object]] = []

    def build_operator_fn(G, cache, params, dt, **kwargs):
        built.append(
            {
                "shape": tuple(G.shape),
                "cache": cache,
                "params": params,
                "dt": dt,
                **kwargs,
            }
        )
        return SimpleNamespace(name="built")

    reused = imex_module._resolve_imex_operator(
        implicit_operator=provided,
        G0=jnp.ones((2,), dtype=jnp.complex64),
        cache="cache",
        params="params",
        dt=0.2,
        linear_cfg="linear",
        implicit_preconditioner="diag",
        compressed_real_fft=False,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=None,
    )
    assert reused is provided
    assert built == []

    built_op = imex_module._resolve_imex_operator(
        implicit_operator=None,
        G0=jnp.ones((2,), dtype=jnp.complex64),
        cache="cache",
        params="params",
        dt=0.2,
        linear_cfg="linear",
        implicit_preconditioner="diag",
        compressed_real_fft=False,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=lambda *_args, **_kwargs: (),
    )
    assert built_op.name == "built"
    assert built[0]["terms"] == "linear"
    assert built[0]["implicit_preconditioner"] == "diag"
    assert built[0]["compressed_real_fft"] is False
    assert "build_implicit_operator_fn" in built[0]

    operator = SimpleNamespace(
        shape=(1, 2),
        squeeze_species=True,
        state_dtype=jnp.complex64,
    )
    state, shape, squeeze_species = imex_module._state_for_imex_operator(
        jnp.ones((2,), dtype=jnp.float32), operator
    )
    assert shape == (1, 2)
    assert squeeze_species is True
    assert state.shape == (1, 2)
    assert state.dtype == jnp.complex64

    with pytest.raises(ValueError, match="implicit_operator shape mismatch"):
        imex_module._state_for_imex_operator(
            jnp.ones((3,), dtype=jnp.float32), operator
        )


def test_advance_imex_nonlinear_state_default_method_solves_rhs() -> None:
    calls: list[float] = []

    def nonlinear_term(g):
        return 2.0 * g

    def solve_step(g_in, rhs):
        calls.append(float(np.asarray(g_in[0])))
        return rhs + 1.0

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="imex",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [2.5], rtol=1e-6)
    assert calls == [1.0]


def test_advance_imex_nonlinear_state_sspx3_matches_constant_rhs_step() -> None:
    def nonlinear_term(g):
        return jnp.ones_like(g) * 2.0

    def solve_step(_g_in, rhs):
        return rhs

    out = advance_imex_nonlinear_state(
        jnp.asarray([1.0], dtype=jnp.float32),
        dt_val=jnp.asarray(0.25, dtype=jnp.float32),
        method="sspx3",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda g: g,
    )

    np.testing.assert_allclose(np.asarray(out), [1.5], rtol=1e-6)


def test_make_imex_diagnostic_step_forwards_runtime_policies() -> None:
    seen: dict[str, object] = {}

    def nonlinear_term(G):
        return jnp.ones_like(G) * 2.0

    def solve_step(G_in, G_rhs):
        seen["solve"] = (G_in, G_rhs)
        return G_rhs + 3.0

    def compute_fields_fn(G, cache, params, **kwargs):
        seen["fields"] = (G, cache, params, kwargs)
        return "new_fields"

    def compute_diag_from_state(G, fields, G_prev, fields_prev, dt_val):
        seen["diag_args"] = (G, fields, G_prev, fields_prev, dt_val)
        return jnp.asarray(9.0, dtype=jnp.float32)

    def select_diagnostics_fn(idx, **kwargs):
        seen["select"] = (idx, kwargs)
        return kwargs["compute_diag_fn"]()

    def emit_progress_fn(G, **kwargs):
        seen["progress"] = (G, kwargs)
        return G + 1.0

    def apply_collision_split_fn(G, damping, dt_val, scheme):
        seen["collision"] = (G, damping, dt_val, scheme)
        return G + 2.0

    cache = SimpleNamespace(name="cache")
    params = SimpleNamespace(name="params")
    term_cfg = SimpleNamespace(name="terms")
    damping = SimpleNamespace(name="damping")
    step = make_imex_diagnostic_step(
        method="imex",
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        dt_val=jnp.asarray(0.5, dtype=jnp.float32),
        compute_fields_fn=compute_fields_fn,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        external_phi=4.0,
        compute_diag_from_state=compute_diag_from_state,
        diagnostics_stride=3,
        select_diagnostics_fn=select_diagnostics_fn,
        show_progress=True,
        steps=11,
        progress_total=jnp.asarray(5.5, dtype=jnp.float32),
        emit_progress_fn=emit_progress_fn,
        use_collision_split=True,
        damping=damping,
        collision_scheme="exact",
        apply_collision_split_fn=apply_collision_split_fn,
    )

    carry_out, step_out = step(
        (
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray([0.5], dtype=jnp.float32),
            "old_fields",
            jnp.asarray(-1.0, dtype=jnp.float32),
            jnp.asarray(2.0, dtype=jnp.float32),
        ),
        jnp.asarray(4, dtype=jnp.int32),
    )

    G_new, G_prev, fields_new, diag, t_new = carry_out
    np.testing.assert_allclose(np.asarray(G_new), [8.0])
    np.testing.assert_allclose(np.asarray(G_prev), [8.0])
    assert fields_new == "new_fields"
    np.testing.assert_allclose(np.asarray(diag), 9.0)
    np.testing.assert_allclose(np.asarray(t_new), 2.5)
    assert step_out[0] is diag
    assert step_out[1] is t_new
    solve_in, solve_rhs = seen["solve"]
    np.testing.assert_allclose(np.asarray(solve_in), [1.0])
    np.testing.assert_allclose(np.asarray(solve_rhs), [2.0])
    fields_args = seen["fields"]
    assert fields_args[1] is cache
    assert fields_args[2] is params
    assert fields_args[3]["terms"] is term_cfg
    assert fields_args[3]["external_phi"] == 4.0
    diag_args = seen["diag_args"]
    np.testing.assert_allclose(np.asarray(diag_args[0]), [7.0])
    assert diag_args[1] == "new_fields"
    np.testing.assert_allclose(np.asarray(diag_args[2]), [0.5])
    assert diag_args[3] == "old_fields"
    np.testing.assert_allclose(np.asarray(diag_args[4]), 0.5)
    select_idx, select_kwargs = seen["select"]
    np.testing.assert_allclose(np.asarray(select_idx), 4)
    assert select_kwargs["diagnostics_stride"] == 3
    np.testing.assert_allclose(np.asarray(select_kwargs["diag_prev"]), -1.0)
    collision_args = seen["collision"]
    np.testing.assert_allclose(np.asarray(collision_args[0]), [5.0])
    assert collision_args[1] is damping
    np.testing.assert_allclose(np.asarray(collision_args[2]), 0.5)
    assert collision_args[3] == "exact"
    progress_args = seen["progress"]
    np.testing.assert_allclose(np.asarray(progress_args[0]), [7.0])
    assert progress_args[1]["show_progress"] is True
    assert progress_args[1]["diag"] is diag
    assert progress_args[1]["steps"] == 11
    np.testing.assert_allclose(np.asarray(progress_args[1]["t_new"]), 2.5)
    np.testing.assert_allclose(np.asarray(progress_args[1]["progress_total"]), 5.5)


def test_make_imex_diagnostic_step_requires_collision_split_policy() -> None:
    step = make_imex_diagnostic_step(
        method="imex",
        nonlinear_term=lambda G: G,
        solve_step=lambda _G_in, G_rhs: G_rhs,
        project_state=lambda G: G,
        state_dtype=jnp.float32,
        real_dtype=jnp.float32,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        compute_fields_fn=lambda *_args, **_kwargs: "fields",
        cache=SimpleNamespace(),
        params=SimpleNamespace(),
        term_cfg=SimpleNamespace(),
        external_phi=None,
        compute_diag_from_state=lambda *_args: jnp.asarray(0.0, dtype=jnp.float32),
        diagnostics_stride=1,
        select_diagnostics_fn=lambda _idx, **kwargs: kwargs["compute_diag_fn"](),
        show_progress=False,
        steps=1,
        progress_total=jnp.asarray(0.1, dtype=jnp.float32),
        emit_progress_fn=lambda G, **_kwargs: G,
        use_collision_split=True,
        damping=SimpleNamespace(),
        apply_collision_split_fn=None,
    )

    with np.testing.assert_raises(ValueError):
        step(
            (
                jnp.asarray([1.0], dtype=jnp.float32),
                jnp.asarray([1.0], dtype=jnp.float32),
                "fields",
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )


def test_run_imex_diagnostic_scan_runs_fixed_step_policy() -> None:
    def step(carry, idx):
        G, G_prev, fields_prev, diag_prev, t_prev = carry
        del G_prev, fields_prev
        G_new = G + 1.0
        diag = diag_prev + idx + 1
        t_new = t_prev + 0.5
        return (G_new, G_new, jnp.asarray([2.0], dtype=jnp.float32), diag, t_new), (
            diag,
            t_new,
        )

    G_final, diag_out = run_imex_diagnostic_scan(
        step,
        (
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray([0.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        steps=3,
        checkpoint=False,
    )

    diag, t = diag_out
    np.testing.assert_allclose(np.asarray(G_final), [3.0])
    np.testing.assert_allclose(np.asarray(diag), [1.0, 3.0, 6.0])
    np.testing.assert_allclose(np.asarray(t), [0.5, 1.0, 1.5])


def test_integrate_cached_imex_scan_owns_cached_scan_policy(monkeypatch) -> None:
    G0 = jnp.zeros((1,), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1,), dtype=jnp.complex64), apar=None, bpar=None)
    build_calls: list[dict[str, object]] = []
    nonlinear_calls: list[str] = []
    linear_calls: list[str] = []

    def build_operator_fn(G, cache, params, dt, **kwargs):
        build_calls.append(kwargs)
        return SimpleNamespace(
            shape=tuple(G.shape),
            dt_val=jnp.asarray(dt, dtype=jnp.float32),
            precond_op=None,
            matvec=lambda x: x,
            squeeze_species=False,
            state_dtype=G.dtype,
        )

    def linear_rhs_fn(G, *_args, **_kwargs):
        linear_calls.append("linear")
        return jnp.zeros_like(G), fields

    def nonlinear_kernel(G, cache, params, terms, **kwargs):
        del cache, params, terms
        assert kwargs["fields_fn"] is fields_fn
        assert kwargs["nonlinear_contribution_fn"] is contribution_fn
        assert kwargs["compressed_real_fft"] is False
        assert kwargs["laguerre_mode"] == "spectral"
        nonlinear_calls.append("nonlinear")
        return jnp.ones_like(G)

    def fields_fn(*_args, **_kwargs):
        return fields

    def contribution_fn(*_args, **_kwargs):
        return jnp.asarray(0.0, dtype=jnp.float32)

    monkeypatch.setattr(
        "gkx.solvers.nonlinear.imex.jax.scipy.sparse.linalg.gmres",
        lambda matvec, rhs, **kwargs: (rhs, SimpleNamespace(success=True)),
    )

    G_out, fields_t = integrate_cached_imex_scan(
        G0,
        SimpleNamespace(name="cache"),
        SimpleNamespace(name="params"),
        0.2,
        2,
        term_cfg=SimpleNamespace(name="terms"),
        linear_cfg=SimpleNamespace(name="linear"),
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_operator_fn,
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_kernel,
        nonlinear_contribution_fn=contribution_fn,
        implicit_iters=0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
    )

    assert build_calls
    assert build_calls[0]["terms"].name == "linear"
    np.testing.assert_allclose(np.asarray(G_out), np.asarray([0.4]), rtol=1e-6)
    assert fields_t.phi.shape == (2, 1)
    assert nonlinear_calls == ["nonlinear"]
    assert linear_calls
