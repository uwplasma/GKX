"""Small, executable physics gates for the optional adaptive eigensolver."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import solvax

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.geometry import SAlphaGeometry
from gkx.geometry.flux_tube import FluxTubeGeometryData, sample_flux_tube_geometry
from gkx.objectives.autodiff_validation import explicit_complex_operator_matrix
from gkx.objectives.core import (
    AdaptiveLinearEigensolverConfig,
    solver_objective_vector_from_geometry,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.runtime import (
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
)
from gkx.solvers.linear import adaptive_propagator_eigenpair
from gkx.workflows.runtime.toml import load_runtime_from_toml


_HAS_SOLVAX_EIGEN_API = all(
    callable(getattr(solvax, name, None))
    for name in (
        "adaptive_eigenpair",
        "eigenpair_reverse",
        "estimate_rk4_timestep",
        "propagator_eigenpairs",
    )
)
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _HAS_SOLVAX_EIGEN_API,
        reason="requires the SOLVAX differentiable eigenpair API",
    ),
]
_ROOT = Path(__file__).resolve().parents[2]
_PHYSICS_CASES = (
    ("ITG-adiabatic-electron", "axisymmetric/runtime_cyclone_quasilinear.toml", False),
    ("ETG-adiabatic-ion", "axisymmetric/runtime_etg.toml", False),
    ("TEM-finite-electron-mass", "axisymmetric/runtime_tem.toml", False),
    ("KBM-Apar", "axisymmetric/runtime_kbm.toml", False),
    ("KBM-Apar-Bpar", "axisymmetric/runtime_kbm.toml", True),
    ("Miller", "axisymmetric/runtime_cyclone_miller_quasilinear.toml", False),
    ("QHS", "non-axisymmetric/runtime_hsx_linear_quasilinear.toml", False),
    ("QI", "non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml", False),
)
_FAST_CONFIG = AdaptiveLinearEigensolverConfig(
    krylov_dim=16,
    restart_krylov_dim=8,
    chunk_horizon=10.0,
    stability_dimension=8,
    adjoint_krylov_dim=16,
)


def _use_cached_vmec_eik(runtime):
    """Use a generated reduced fixture when the VMEC backend is unavailable."""

    if runtime.geometry.model != "vmec":
        return runtime
    if runtime.geometry.vmec_file is None:
        pytest.skip("VMEC integration requires a source equilibrium")
    stem = Path(runtime.geometry.vmec_file).stem.removeprefix("wout_")
    matches = tuple((_ROOT / ".cache/gkx/vmec_eik").glob(f"{stem}_*.eik.nc"))
    if not matches:
        pytest.skip("VMEC integration needs its backend or a generated eik cache")
    fixture = min(matches, key=lambda path: path.stat().st_size)
    return replace(
        runtime,
        geometry=replace(
            runtime.geometry,
            model="imported-eik",
            geometry_file=str(fixture),
        ),
    )


@pytest.mark.parametrize(
    ("_name", "relative_path", "enable_bpar"),
    _PHYSICS_CASES,
    ids=[case[0] for case in _PHYSICS_CASES],
)
def test_adaptive_observables_match_dense_across_physics(
    _name: str,
    relative_path: str,
    enable_bpar: bool,
) -> None:
    """Cover branch, field, species, boundary, and geometry dispatch."""

    runtime, raw = load_runtime_from_toml(_ROOT / "examples/linear" / relative_path)
    runtime = replace(
        runtime,
        grid=replace(
            runtime.grid,
            Nx=1,
            Nz=8,
            ntheta=8 if runtime.grid.ntheta is not None else None,
            nperiod=1 if runtime.grid.nperiod is not None else None,
        ),
    )
    runtime = _use_cached_vmec_eik(runtime)
    if enable_bpar:
        runtime = replace(
            runtime,
            physics=replace(runtime.physics, use_bpar=True),
            terms=replace(runtime.terms, bpar=1.0),
        )
    runtime_geometry = build_runtime_geometry(runtime)
    full_grid = build_spectral_grid(runtime.grid)
    ky_index = int(
        np.argmin(np.abs(np.asarray(full_grid.ky) - float(raw["run"]["ky"])))
    )
    grid = select_ky_grid(full_grid, ky_index)
    geometry = (
        runtime_geometry
        if isinstance(runtime_geometry, FluxTubeGeometryData)
        else sample_flux_tube_geometry(runtime_geometry, grid.z)
    )
    params = build_runtime_linear_params(runtime, Nm=4, geom=runtime_geometry)
    terms = build_runtime_linear_terms(runtime)
    adaptive_config = (
        AdaptiveLinearEigensolverConfig() if _name in {"QHS", "QI"} else _FAST_CONFIG
    )

    def objective(eigensolver: str) -> jax.Array:
        return solver_objective_vector_from_geometry(
            geometry,
            spectral_grid=grid,
            n_laguerre=2,
            n_hermite=4,
            params_linear=params,
            terms=terms,
            eigensolver=eigensolver,
            adaptive_config=adaptive_config,
        )

    dense = objective("dense")
    adaptive = objective("adaptive-propagator")
    np.testing.assert_allclose(adaptive, dense, rtol=1.0e-8, atol=1.0e-9)


def test_biorthogonal_continuation_crosses_real_growth_ordering() -> None:
    """Track a Cyclone ITG mode after it becomes the third-fastest branch."""

    scan = (12.0, 13.0, 13.25, 14.0)
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geometry = SAlphaGeometry.from_config(cfg.geometry)
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    shape = (2, 3, 1, 1, 8)
    phase = jnp.arange(np.prod(shape), dtype=jnp.float64) + 1.0
    broadband = jnp.exp(1j * phase * 0.6180339887498948).reshape(shape)
    right = left = None
    ranks: list[int] = []

    for parameter in scan:
        params = LinearParams(
            R_over_LTi=parameter,
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )
        cache = build_linear_cache(grid, geometry, params, Nl=2, Nm=3)

        def apply(state: jax.Array) -> jax.Array:
            return linear_rhs_cached(
                state,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0]

        solution = adaptive_propagator_eigenpair(
            broadband if right is None else right,
            cache,
            params,
            terms=terms,
            krylov_dim=24,
            restart_krylov_dim=12,
            candidate_count=4,
            tol=1.0e-8,
            continuation_vector=right,
            continuation_covector=left,
            continuation_overlap_floor=0.9,
            continuation_spectral_gap_floor=0.05,
        )
        assert solution.converged and solution.continuation_passed

        matrix = np.asarray(explicit_complex_operator_matrix(apply, shape))
        dense_values, dense_vectors = np.linalg.eig(matrix)
        dense_vectors /= np.linalg.norm(dense_vectors, axis=0)
        if left is None:
            dense_index = int(np.argmax(dense_values.real))
        else:
            left_flat = np.asarray(left).reshape(-1)
            scores = np.abs(left_flat.conj() @ dense_vectors)
            dense_index = int(np.argmax(scores))
        rank = int(
            np.flatnonzero(np.argsort(dense_values.real)[::-1] == dense_index)[0]
        )
        ranks.append(rank)
        np.testing.assert_allclose(
            solution.eigenvalue,
            dense_values[dense_index],
            rtol=1.0e-8,
        )

        transpose = jax.linear_transpose(apply, solution.eigenvector)

        @jax.jit
        def adjoint(vector: jax.Array) -> jax.Array:
            return jnp.conj(transpose(jnp.conj(vector))[0])

        left_modes = solvax.propagator_eigenpairs(
            adjoint,
            broadband,
            dt=solution.filter_dt,
            steps=solution.filter_steps,
            krylov_dim=24,
            candidates=4,
            tol=1.0e-8,
        )
        distances = np.where(
            np.asarray(left_modes.converged),
            np.abs(
                np.asarray(left_modes.eigenvalues)
                - np.conj(complex(np.asarray(solution.eigenvalue)))
            ),
            np.inf,
        )
        left = left_modes.eigenvectors[int(np.argmin(distances))]
        right = solution.eigenvector

    assert ranks == [0, 0, 1, 2]
