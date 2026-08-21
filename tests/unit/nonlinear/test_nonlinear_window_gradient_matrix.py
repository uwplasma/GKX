"""AD-vs-finite-difference coverage matrix for the production heat-flux window.

Before this file the only gradient tests of ``nonlinear_heat_flux_window`` were
single-species, electrostatic, collisionless and RK2. Every other production
switch -- kinetic electrons, ``apar``/``bpar`` at finite beta, a custom
collision operator, hypercollisions, RK3/RK4 -- was wired but never exercised
under ``grad``, and the combination space was untested entirely.

Each row differentiates the same public entry point with respect to one scalar
design parameter and compares against a centered difference of that same
function. The finite-difference steps are tuned per row: they are the largest
step whose truncation error stays below the reverse-mode agreement, so a
regression in the adjoint shows up as a rel-error blow-up rather than being
masked by differencing noise.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.operators.linear.params import LinearParams, Species, build_linear_params
from gkx.solvers.nonlinear.state_integration import (
    DIVERGENCE_KNEE_STEPS,
    nonlinear_heat_flux_window,
)
from gkx.terms.config import TermConfig

NX, NY, NZ = 16, 16, 8
NL, NM = 2, 4
DT = 0.004
STEPS = 6
TAIL = 4

ES_TERMS = TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0)
EM_TERMS = TermConfig(nonlinear=1.0, apar=1.0, bpar=1.0)

IONS = Species(
    charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
)
#: Deuterium/electron mass ratio: the kinetic-electron case the window has to
#: survive, not a token second species.
ELECTRONS = Species(
    charge=-1.0, mass=2.7e-4, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
)
SECOND_ION = Species(
    charge=1.0, mass=2.0, density=0.2, temperature=1.0, tprim=1.8, fprim=0.5
)


@pytest.fixture(scope="module")
def case_grid():
    cfg = CycloneBaseCase(grid=GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=6.0, Ly=6.0))
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    return grid, geom


def _seed(grid, n_species: int, key: int) -> jnp.ndarray:
    """Deterministic low-amplitude state standing in for a saturated one."""

    rng = np.random.default_rng(key)
    shape = (n_species, NL, NM, grid.ky.size, grid.kx.size, grid.z.size)
    draw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jnp.asarray(1.0e-4 * draw, dtype=jnp.complex128)


class HermiteLaguerreDrag:
    """Custom collision model whose rate is a traced design parameter.

    ``integrate_nonlinear`` has always accepted ``collision_operator``; the
    differentiable window did not, so a run saturated with a custom operator and
    then differentiated returned the derivative of a *different* model. This
    operator exists to test that the rate reaches the gradient at all.
    """

    def __init__(self, nu):
        self.nu = nu

    def apply(self, context):
        state = context.distribution
        offset = state.ndim - 5
        n_laguerre, n_hermite = state.shape[offset : offset + 2]
        ell = jnp.arange(n_laguerre).reshape((n_laguerre, 1, 1, 1, 1))
        hermite = jnp.arange(n_hermite).reshape((1, n_hermite, 1, 1, 1))
        rate = jnp.asarray(self.nu) * (hermite + 2.0 * ell)
        return -rate * state


def _window(grid, geom, state, params, *, terms, method, collision, checkpoint):
    return nonlinear_heat_flux_window(
        state,
        grid,
        geom,
        params,
        dt=DT,
        steps=STEPS,
        method=method,
        tail_steps=TAIL,
        terms=terms,
        checkpoint=checkpoint,
        compressed_real_fft=True,
        collision_operator=collision,
    )


def _drive_case(grid, geom, params, terms, method, state, collision=None):
    """Differentiate a uniform multiplier on the per-species ``a/L_T`` drive."""

    def evaluate(scale, checkpoint=True):
        scaled = replace(params, tprim=jnp.asarray(params.tprim) * scale)
        return _window(
            grid,
            geom,
            state,
            scaled,
            terms=terms,
            method=method,
            collision=collision,
            checkpoint=checkpoint,
        )

    return evaluate, jnp.asarray(1.0), 1.0e-4


def _build_case(name, grid, geom):
    if name in ("baseline_es_rk2", "baseline_es_rk3", "baseline_es_rk4"):
        method = name.rsplit("_", 1)[1]
        return _drive_case(
            grid, geom, LinearParams(), ES_TERMS, method, _seed(grid, 1, 17)
        )

    if name == "multispecies_kinetic_electrons":
        params = build_linear_params([IONS, ELECTRONS], tau_e=1.0)
        return _drive_case(grid, geom, params, ES_TERMS, "rk2", _seed(grid, 2, 19))

    if name == "multispecies_two_ions":
        params = build_linear_params([IONS, SECOND_ION], tau_e=1.0)
        return _drive_case(grid, geom, params, ES_TERMS, "rk2", _seed(grid, 2, 19))

    if name == "electromagnetic_d_beta":
        base = LinearParams(beta=0.02, fapar=1.0)
        state = _seed(grid, 1, 23)

        def by_beta(beta, checkpoint=True):
            return _window(
                grid,
                geom,
                state,
                replace(base, beta=beta),
                terms=EM_TERMS,
                method="rk2",
                collision=None,
                checkpoint=checkpoint,
            )

        return by_beta, jnp.asarray(0.02), 1.0e-6

    if name == "electromagnetic_d_drive":
        return _drive_case(
            grid,
            geom,
            LinearParams(beta=0.02, fapar=1.0),
            EM_TERMS,
            "rk2",
            _seed(grid, 1, 23),
        )

    if name == "custom_collisions_d_nu":
        terms = TermConfig(nonlinear=1.0, collisions=1.0, apar=0.0, bpar=0.0)
        state = _seed(grid, 1, 29)

        def by_nu(nu, checkpoint=True):
            return _window(
                grid,
                geom,
                state,
                LinearParams(),
                terms=terms,
                method="rk2",
                collision=HermiteLaguerreDrag(nu),
                checkpoint=checkpoint,
            )

        return by_nu, jnp.asarray(0.3), 1.0e-5

    if name == "hypercollisions_d_nu_hyper_m":
        base = LinearParams(
            nu_hyper_m=0.5, nu_hyper_l=0.5, p_hyper_m=6.0, p_hyper_l=6.0
        )
        terms = TermConfig(
            nonlinear=1.0, hypercollisions=1.0, apar=0.0, bpar=0.0
        )
        state = _seed(grid, 1, 31)

        def by_hyper(nu_hyper_m, checkpoint=True):
            return _window(
                grid,
                geom,
                state,
                replace(base, nu_hyper_m=nu_hyper_m),
                terms=terms,
                method="rk2",
                collision=None,
                checkpoint=checkpoint,
            )

        # The hypercollisional sensitivity is ~3 decades below the window value,
        # so a 1e-5 step differences two nearly equal numbers; 1e-3 is where the
        # centered difference is truncation- rather than roundoff-limited.
        return by_hyper, jnp.asarray(0.5), 1.0e-3

    if name == "combined_ms_em_coll_hyper_rk3":
        params = build_linear_params(
            [IONS, ELECTRONS],
            tau_e=1.0,
            beta=0.02,
            fapar=1.0,
            nu_hyper_m=0.5,
            nu_hyper_l=0.5,
            p_hyper_m=6.0,
            p_hyper_l=6.0,
        )
        terms = TermConfig(
            nonlinear=1.0, apar=1.0, bpar=1.0, collisions=1.0, hypercollisions=1.0
        )
        return _drive_case(
            grid,
            geom,
            params,
            terms,
            "rk3",
            _seed(grid, 2, 37),
            collision=HermiteLaguerreDrag(0.2),
        )

    raise AssertionError(f"unknown gradient-matrix case {name}")


COVERAGE_MATRIX = (
    "baseline_es_rk2",
    "baseline_es_rk3",
    "baseline_es_rk4",
    "multispecies_kinetic_electrons",
    "multispecies_two_ions",
    "electromagnetic_d_beta",
    "electromagnetic_d_drive",
    "custom_collisions_d_nu",
    "hypercollisions_d_nu_hyper_m",
    "combined_ms_em_coll_hyper_rk3",
)


@pytest.mark.parametrize("case", COVERAGE_MATRIX)
def test_window_gradient_matches_centered_finite_difference(case, case_grid):
    """Every production switch must reach the adjoint of the physical flux."""

    grid, geom = case_grid
    evaluate, point, step = _build_case(case, grid, geom)
    value, gradient = jax.value_and_grad(evaluate)(point)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0, f"{case} left the design parameter disconnected"

    centered = (evaluate(point + step) - evaluate(point - step)) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered), rtol=1.0e-6, atol=0.0
    )


def test_block_checkpointed_window_matches_plain_reverse_pass(case_grid):
    """Block checkpointing must not change the value or the gradient.

    Run on the 6-D kinetic-electron state, where the retained block boundaries
    carry a species axis the existing single-species parity test never saw.
    """

    grid, geom = case_grid
    evaluate, point, _step = _build_case(
        "multispecies_kinetic_electrons", grid, geom
    )
    blocked = jax.value_and_grad(evaluate)(point)
    plain = jax.value_and_grad(lambda value: evaluate(value, False))(point)
    np.testing.assert_allclose(
        np.asarray(blocked), np.asarray(plain), rtol=1.0e-12, atol=0.0
    )


def test_custom_collision_operator_changes_the_differentiated_window(case_grid):
    """A window that ignored ``collision_operator`` would differentiate other physics."""

    grid, geom = case_grid
    evaluate, point, _step = _build_case("custom_collisions_d_nu", grid, geom)
    with_operator = float(evaluate(point))
    without_operator = float(
        _window(
            grid,
            geom,
            _seed(grid, 1, 29),
            LinearParams(),
            terms=TermConfig(nonlinear=1.0, collisions=1.0, apar=0.0, bpar=0.0),
            method="rk2",
            collision=None,
            checkpoint=True,
        )
    )
    assert with_operator != without_operator


def test_electromagnetic_window_actually_solves_apar_and_bpar(case_grid):
    """Guard the EM rows: a silently electrostatic field solve would still pass FD."""

    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.solvers.nonlinear.state_integration import nonlinear_rhs_cached

    grid, geom = case_grid
    params = LinearParams(beta=0.02, fapar=1.0)
    state = _seed(grid, 1, 23)
    cache = build_linear_cache(grid, geom, params, Nl=NL, Nm=NM)
    _derivative, fields = nonlinear_rhs_cached(
        state, cache, params, EM_TERMS, differentiable=True
    )
    assert fields.apar is not None and float(jnp.max(jnp.abs(fields.apar))) > 0.0
    assert fields.bpar is not None and float(jnp.max(jnp.abs(fields.bpar))) > 0.0


def test_window_beyond_the_divergence_knee_warns(case_grid):
    """A window past the measured knee must say so before it burns the compute.

    The knee is lowered to one step here so the guard is tested for the price of
    a two-step window rather than the 1025 the shipped default would cost.
    """

    grid, geom = case_grid
    state = _seed(grid, 1, 17)
    with pytest.warns(RuntimeWarning, match="divergence knee"):
        nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            LinearParams(),
            dt=DT,
            steps=2,
            terms=ES_TERMS,
            divergence_knee_steps=1,
        )


def test_window_at_or_below_the_knee_is_silent(case_grid, recwarn):
    """QA_optimization runs at exactly the knee; that must not warn."""

    grid, geom = case_grid
    state = _seed(grid, 1, 17)
    nonlinear_heat_flux_window(
        state,
        grid,
        geom,
        LinearParams(),
        dt=DT,
        steps=2,
        terms=ES_TERMS,
        divergence_knee_steps=2,
    )
    nonlinear_heat_flux_window(
        state, grid, geom, LinearParams(), dt=DT, steps=2, terms=ES_TERMS
    )
    assert not [w for w in recwarn if "divergence knee" in str(w.message)]


def test_shipped_optimization_example_stays_at_or_below_the_knee():
    """The QA example's window is the reason the guard exists; pin it."""

    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "optimization"
        / "QA_optimization.py"
    ).read_text()
    match = re.search(r"WINDOW_STEPS\s*=\s*([\d_]+)", source)
    assert match is not None
    assert int(match.group(1).replace("_", "")) <= DIVERGENCE_KNEE_STEPS
