"""Guard the contractions that XLA would otherwise satisfy with TF32.

On Ampere and later NVIDIA GPUs XLA may answer an unpinned ``dot`` on the tensor
cores in TF32, which keeps 10 mantissa bits: a relative error of 2^-11 = 4.9e-04
where float32 would give ~1e-07. Two properties make the defect hard to see.

Nothing on CPU can observe it. There is no TF32 path there, so an unpinned dot
and ``Precision.HIGHEST`` produce bit-identical CPU numbers and no assertion on a
*value* can fail. The precision request, however, is recorded in the jaxpr on
every backend, so that is what these tests assert.

And it is shape-dependent. Measured on an RTX A4000, XLA uses TF32 only when the
contraction is a genuine matrix product -- a free dimension left on *both*
operands. Every vector-shaped contraction measured exact: a vdot of two length
4096 vectors at 4.2e-08, a ``(16,) x (16, 4096)`` tensordot at 1.2e-07, a
``(512, 512) x (512,)`` matvec at 1.1e-07, an ``n,kn->k`` einsum at 1.8e-07, and
the Arnoldi Gram-Schmidt vdot inside its ``fori_loop`` at 8.7e-07. The matrix
products on the same GPU: ``(16, 16) x (16, 4096)`` at 3.0e-04 and a real
``(m, m) x (m, m)`` at 3.1e-04 for m = 8, 64 and 512, every one of them back to
~1e-07 when pinned.

So the hazard is not "a dot" but "a dot that is matrix-shaped", and a contraction
can cross that line without being edited -- ``dominant_eigenpairs_propagator_cached``
lifts a ``(candidates, k) x (k, n)`` product that is exact at the default
``candidates = 1`` and TF32 from two candidates on. ``assert_matrix_dots_pinned``
therefore classifies by shape rather than by call site, which is what lets it
catch a contraction that becomes hazardous because a config value moved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkx.solvers.linear.krylov_algorithms as ka
import gkx.solvers.linear.krylov_propagator as kp
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.geometry.sensitivity import _damped_gauss_newton_step
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)

EXACT = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)

# The one matrix-shaped contraction that is deliberately left unpinned, with the
# measurement that justifies it. ``_propagator_arnoldi_restart_step`` lifts every
# Ritz vector at once purely to rank them by overlap; the winning index is then
# lifted again, on its own, by the exact vector-shaped contraction in
# ``_ritz_vector_from_index``. So the TF32 error never reaches a returned number,
# it only has to not reorder an argmax -- and it does not. Sweeping a synthetic
# branch crossing on an RTX A4000, the selected index was identical pinned and
# unpinned in 120/120 draws at each overlap gap of 1e-02, 1e-03, 3e-04, 1e-04 and
# 1e-05, plus 200/200 unconstrained random draws. The rounding is common-mode
# across the rows of one GEMM, so the differences that the argmax reads are far
# better conditioned than the entries themselves. Pinning it would cost a
# matrix product per restart to change nothing measurable.
ALLOWED_UNPINNED_MATRIX_DOTS = {
    "krylov_algorithms.py:673": "overlap ranking only; argmax provably unmoved",
}


def _iter_dots(jaxpr):
    """Yield every ``dot_general``, descending into nested jaxprs.

    ``fori_loop``, ``scan``, ``cond``, ``pjit`` and ``custom_vjp`` all park their
    equations in a sub-jaxpr, so a scan of ``jaxpr.eqns`` alone sees none of the
    Arnoldi contractions -- they live inside the loop body.
    """

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            yield eqn
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(item, "jaxpr", item)
                inner = getattr(inner, "jaxpr", inner)
                if hasattr(inner, "eqns"):
                    yield from _iter_dots(inner)


def _is_matrix_product(eqn) -> bool:
    """True when both operands keep a free axis, i.e. the tensor-core case."""

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    lhs, rhs = eqn.invars[0].aval, eqn.invars[1].aval
    lhs_free = [
        axis
        for axis in range(len(lhs.shape))
        if axis not in lhs_contract and axis not in lhs_batch
    ]
    rhs_free = [
        axis
        for axis in range(len(rhs.shape))
        if axis not in rhs_contract and axis not in rhs_batch
    ]
    return bool(lhs_free) and bool(rhs_free)


def _origin(eqn) -> str:
    """Return ``file.py:line`` inside gkx for one equation, for the allowlist."""

    traceback = eqn.source_info.traceback
    for frame in getattr(traceback, "frames", []) or []:
        name = getattr(frame, "file_name", "")
        if "/gkx/" in name and "site-packages" not in name:
            return f"{name.rsplit('/', 1)[-1]}:{frame.line_num}"
    return "<unknown>"


def matrix_dots(function, *args, **kwargs):
    """Return ``(origin, precision)`` for every matrix-shaped dot in a callable."""

    jaxpr = jax.make_jaxpr(function)(*args, **kwargs).jaxpr
    return [
        (_origin(eqn), eqn.params["precision"])
        for eqn in _iter_dots(jaxpr)
        if _is_matrix_product(eqn)
    ]


def assert_matrix_dots_pinned(label, function, *args, **kwargs) -> int:
    """Assert every matrix-shaped contraction is pinned; return how many there were."""

    found = matrix_dots(function, *args, **kwargs)
    unpinned = [
        origin
        for origin, precision in found
        if precision is None and origin not in ALLOWED_UNPINNED_MATRIX_DOTS
    ]
    assert not unpinned, (
        f"{label} contains matrix-shaped contractions with no precision request: "
        f"{sorted(set(unpinned))}. XLA satisfies these with TF32 on Ampere and "
        "later NVIDIA GPUs (10 mantissa bits, ~4.9e-04 relative). Pin them with "
        "precision=jax.lax.Precision.HIGHEST, or add the call site to "
        "ALLOWED_UNPINNED_MATRIX_DOTS with the measurement that says it is safe."
    )
    return len(found)


def _linear_setup():
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        nu=0.01,
        nu_hyper=0.01,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    nl, nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl=nl, Nm=nm)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    rng = np.random.default_rng(0)
    shape = (nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    v0 = jnp.asarray(
        rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=jnp.complex64
    )
    return cache, params, term_cfg, v0


@pytest.mark.parametrize("candidates", [1, 2, 4])
def test_propagator_candidate_lift_pins_exact_dot_precision(candidates: int) -> None:
    """The lift that produces the returned eigenvectors must never run in TF32.

    This is the contraction whose hazard depends on a config value: at
    ``candidates = 1`` it is a matvec and XLA leaves it exact, at two or more it
    is a matrix product and TF32 moved the returned eigenvector from 1.2e-08 to
    1.8e-05 against a float64 reference on an RTX A4000. Every candidate count is
    asserted so the pin cannot be dropped on the argument that the default is safe.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    found = matrix_dots(
        lambda value: kp.dominant_eigenpairs_propagator_cached(
            value,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            dt=0.01,
            propagator_steps=2,
            candidates=candidates,
        ),
        v0,
    )
    lifts = [
        precision
        for origin, precision in found
        if origin.startswith("krylov_propagator")
    ]
    assert lifts, "the candidate lift lowered to no matrix dot; the guard is vacuous"
    assert all(precision == EXACT for precision in lifts), (
        f"the candidate lift is unpinned at candidates={candidates} ({lifts}); it "
        "will be answered in TF32 on Ampere and later NVIDIA GPUs"
    )


def test_gauss_newton_normal_equations_pin_exact_dot_precision() -> None:
    """J^T J is the one matrix product in the inverse-design step.

    Forming the normal equations squares the conditioning, so TF32's 10 mantissa
    bits land directly in the step: 1.8e-04 on the entries and 1.7e-04 on the
    step against float64 on an RTX A4000, versus the 1.0e-04 rtol the geometry
    AD-versus-FD report gates on. The matvec and the vector dot on either side of
    it are vector-shaped and stay exact unpinned, so only one pin is expected.
    """

    jac = jnp.asarray(np.linspace(0.1, 1.6, 12).reshape(4, 3), dtype=jnp.float32)
    residual = jnp.asarray([0.3, -0.2, 0.5, 0.1], dtype=jnp.float32)
    found = matrix_dots(
        lambda j, r: _damped_gauss_newton_step(j, r, damping=1.0e-6), jac, residual
    )
    assert found, "the Gauss-Newton step lowered to no matrix dot; the guard is vacuous"
    assert all(precision == EXACT for _origin, precision in found), (
        f"the normal-equations product is unpinned ({found}); TF32 puts 1.7e-04 "
        "into a step whose own report gates at rtol 1.0e-04"
    )


def test_hot_path_matrix_contractions_are_pinned() -> None:
    """No matrix-shaped contraction in the solver hot paths may be left to TF32.

    A repo-wide ratchet scoped to the paths that actually run on a GPU. It is
    shape-aware on purpose: pinning every dot would cost throughput on the
    vector-shaped majority that measures exact anyway, while pinning by call site
    would miss the case this audit found -- a contraction that becomes a matrix
    product because a config value changed, with no edit to the line.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    checked = 0
    checked += assert_matrix_dots_pinned(
        "_arnoldi",
        lambda value: ka._arnoldi(
            value, ka._apply_operator, cache, params, term_cfg, 8
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "_operator_arnoldi_restart_step",
        lambda value: ka._operator_arnoldi_restart_step(
            value,
            value,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            omega_min_factor=0.0,
            omega_target_factor=2.0,
            omega_cap_factor=10.0,
            omega_sign=1,
            select_overlap=True,
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "_propagator_arnoldi_restart_step",
        lambda value: ka._propagator_arnoldi_restart_step(
            value,
            value,
            ka._apply_operator,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            horizon=jnp.asarray(1.0),
            growth_only=True,
            omega_min_factor=0.0,
            omega_target_factor=2.0,
            omega_cap_factor=10.0,
            omega_sign=1,
            select_overlap=True,
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "dominant_eigenpair_power",
        lambda value: ka.dominant_eigenpair_power(
            value, cache, params, term_cfg, iterations=2, dt=0.01
        ),
        v0,
    )
    for candidates in (1, 4):
        checked += assert_matrix_dots_pinned(
            f"dominant_eigenpairs_propagator_cached(candidates={candidates})",
            lambda value, count=candidates: kp.dominant_eigenpairs_propagator_cached(
                value,
                cache,
                params,
                term_cfg,
                krylov_dim=8,
                dt=0.01,
                propagator_steps=2,
                candidates=count,
            ),
            v0,
        )
    # Three today: the allowlisted overlap lift in the propagator restart, and the
    # candidate lift at each of the two candidate counts. The floor is here so that
    # a refactor which stops reaching these paths fails loudly instead of leaving a
    # sweep that asserts nothing.
    assert checked >= 3, (
        f"only {checked} matrix-shaped contractions were reached; the sweep has "
        "stopped exercising the paths it is supposed to guard"
    )


def test_allowlisted_contraction_is_still_matrix_shaped_and_unpinned() -> None:
    """Keep the allowlist honest.

    If the overlap lift is ever pinned, or stops being matrix-shaped, the entry
    above is stale and must be deleted rather than left to excuse some future
    contraction that happens to land on the same line.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    found = dict(
        matrix_dots(
            lambda value: ka._propagator_arnoldi_restart_step(
                value,
                value,
                ka._apply_operator,
                cache,
                params,
                term_cfg,
                krylov_dim=8,
                horizon=jnp.asarray(1.0),
                growth_only=True,
                omega_min_factor=0.0,
                omega_target_factor=2.0,
                omega_cap_factor=10.0,
                omega_sign=1,
                select_overlap=True,
            ),
            v0,
        )
    )
    for origin in ALLOWED_UNPINNED_MATRIX_DOTS:
        assert origin in found, (
            f"{origin} is allowlisted as an unpinned matrix contraction but no "
            "longer appears as one; delete the stale entry"
        )
        assert found[origin] is None, (
            f"{origin} is pinned now, so its allowlist entry is misleading; delete it"
        )
