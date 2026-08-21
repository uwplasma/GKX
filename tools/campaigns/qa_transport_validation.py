#!/usr/bin/env python3
"""Run restartable, matched baseline/candidate QA transport ensembles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.solvers.nonlinear.diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
)
import vmex as vj
from vmex import optimize as opt
from vmex.core import turbulence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = ROOT / "examples" / "optimization"


@dataclass(frozen=True)
class Case:
    nx: int = 16
    ny: int = 16
    nz: int = 24
    nl: int = 4
    nm: int = 8
    dt: float = 0.05
    tmax: float = 1500.0
    window_start: float = 1100.0


CASES = {
    "nominal": Case(),
    "dt04": Case(dt=0.04),
    "dt025": Case(dt=0.025),
    "perp12": Case(nx=12, ny=12),
    "perp20": Case(nx=20, ny=20),
    "perp24": Case(nx=24, ny=24),
    "perp24long": Case(nx=24, ny=24, tmax=2500.0, window_start=1900.0),
    "z16": Case(nz=16),
    "z32": Case(nz=32),
    "v36": Case(nl=3, nm=6),
    "v612": Case(nl=6, nm=12),
}


def solve_input(path: Path):
    """Re-solve one accepted vacuum boundary at the campaign resolution."""

    inp = vj.VmecInput.from_file(path)
    inp = replace(
        inp,
        am=np.zeros_like(inp.am),
        pres_scale=0.0,
        ns_array=np.asarray([101]),
        ftol_array=np.asarray([1.0e-10]),
        niter_array=np.asarray([15_000]),
    )
    return opt.solve_equilibrium(inp)


def initial_state(case: Case, seed: int) -> jnp.ndarray:
    """Fresh Hermitian multimode perturbation; reuse only within a matched pair."""

    state = jnp.zeros(
        (1, case.nl, case.nm, case.ny, case.nx, case.nz), dtype=jnp.complex64
    )
    modes = ((1, 0), (1, 1), (2, -1), (2, 2), (3, -2))
    key = jax.random.PRNGKey(seed)
    values = 1.0e-3 * (
        jax.random.normal(key, (len(modes), case.nz), dtype=jnp.float32)
        + 1j
        * jax.random.normal(
            jax.random.fold_in(key, 1),
            (len(modes), case.nz),
            dtype=jnp.float32,
        )
    )
    for value, (ky, kx) in zip(values, modes):
        state = state.at[0, 0, 0, ky, kx].set(value)
        state = state.at[0, 0, 0, -ky, -kx].set(jnp.conj(value))
    return state


def run_trace(
    *,
    case_name: str,
    case: Case,
    design: str,
    seed: int,
    equilibrium,
    output_dir: Path,
) -> None:
    output = output_dir / f"{case_name}_{design}_seed{seed:03d}.npz"
    if output.exists():
        print(f"skip {output.name}", flush=True)
        return

    grid = gkx.build_spectral_grid(
        gkx.GridConfig(Nx=case.nx, Ny=case.ny, Nz=case.nz, Lx=62.8, Ly=62.8)
    )
    geometry = turbulence.flux_tube_geometry(
        equilibrium.state,
        equilibrium.runtime,
        s_index=64,
        alpha=0.0,
        ntheta=case.nz,
    )
    params = gkx.LinearParams(
        tprim=3.0,
        fprim=1.0,
        nu=0.01,
        kpar_scale=geometry.gradpar_value,
        nu_hermite=1.0,
        nu_laguerre=2.0,
        nu_hyper=0.0,
        p_hyper_m=float(min(20, max(case.nm // 2, 1))),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        D_hyper=0.05,
        p_hyper_kperp=2.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.125,
    )
    terms = gkx.TermConfig(
        collisions=1.0,
        hypercollisions=1.0,
        hyperdiffusion=1.0,
        end_damping=1.0,
        nonlinear=1.0,
        apar=0.0,
        bpar=0.0,
    )
    stride = round(1.0 / case.dt)
    started = perf_counter()
    _, diagnostics, state, _ = integrate_nonlinear_explicit_diagnostics_state(
        initial_state(case, seed),
        grid,
        geometry,
        params,
        case.dt,
        round(case.tmax / case.dt),
        method="rk3",
        terms=terms,
        checkpoint=False,
        sample_stride=stride,
        diagnostics_stride=stride,
        resolved_diagnostics=False,
    )
    jax.block_until_ready(state)
    elapsed = perf_counter() - started
    time = np.asarray(diagnostics.t, dtype=np.float64)
    heat_flux = np.asarray(diagnostics.heat_flux_t, dtype=np.float64)
    np.savez_compressed(
        output,
        time=time,
        heat_flux=heat_flux,
        elapsed_seconds=np.asarray(elapsed),
    )
    late = heat_flux[time >= case.window_start]
    print(
        f"done {output.name}: {elapsed:.1f} s, late mean={late.mean():.8e}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()
    if args.seed_stop <= args.seed_start:
        parser.error("--seed-stop must exceed --seed-start")

    inputs = {
        "baseline": args.input_dir / "input.qa_transport_baseline",
        "candidate": args.input_dir / "input.qa_transport_candidate",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    equilibria = {name: solve_input(path) for name, path in inputs.items()}
    for seed in range(args.seed_start, args.seed_stop):
        for design in ("baseline", "candidate"):
            run_trace(
                case_name=args.case,
                case=CASES[args.case],
                design=design,
                seed=seed,
                equilibrium=equilibria[design],
                output_dir=args.output_dir,
            )


if __name__ == "__main__":
    main()
