"""Q29: the two coverage holes #242 and #253 left, closed and measured.

#242 (Q15) gave the runtime structured solver status and made it fail closed,
and recorded what it did not reach: the IMEX *diagnostics* route, the sheared
IMEX route, and the saved summary files. #253 (Q23) projected a supplied state
onto the linked chain cover at every entry point a user reaches by name, and
recorded what it did not reach: the raw ``integrate_nonlinear`` and
``integrate_nonlinear_cached`` drivers.

This script measures both, on the linked nonlinear Cyclone grid #253 used
(Nx8/Ny8/Nz16, Nl4/Nm8, jtwist 1, rate .1) and on the periodic no-op arm:

1. the supplied state's off-chain content, and what the raw drivers integrate;
2. one nonlinear right-hand side, chain rows, with and without off-chain
   content, and the amplitude scan behind it;
3. an N-step raw-driver run's final state and heat flux, and the two raw doors
   agreeing bitwise;
4. the no-op arms: an on-cover state, a periodic deck, and the optimized-HLO
   fingerprint of the compiled driver on both, plus the same fingerprint at two
   step counts so the linked deck's delta is shown to be constant in ``steps``
   -- that is, outside the scan body;
5. the IMEX diagnostics route and the sheared IMEX route: a starved inner budget
   is visible through ``return_solve_stats=True`` and refused by the shared host
   gate, while a converged budget passes; and the HLO instruction count of each
   scan before, after without stats requested, and after with them;
6. the saved summary files: the solver-status keys a runtime linear implicit run
   and an IMEX nonlinear run now write to ``*.summary.json``;
7. the linear pilot's certified eigenpair and explicit-time fit, which must not
   move.

Run once per worktree; ``before`` = a detached checkout of ``origin/main``,
``after`` = this branch. The two logs are compared line for line. Sections 5 and
6 print ``unavailable on this arm`` where the ``before`` arm has no such channel;
that absence is the finding.
"""

import hashlib
import inspect
import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.artifacts.io import write_runtime_linear_artifacts
from gkx.config import GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.operators.fluxes import heat_flux_total
from gkx.operators.moments import fieldline_quadrature_weights
from gkx.runtime import (
    _build_initial_condition,
    _runtime_linear_dispatch_deps,
    apply_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.solvers_linear_implicit import require_converged_implicit_solves
from gkx.solvers_nonlinear_diagnostic_integration import (
    _imex_nonlinear_diagnostics_deps,
    integrate_nonlinear_imex_diagnostics,
)
from gkx.solvers_nonlinear_imex_diagnostics import (
    _build_imex_scan_context,
    _imex_option_bundle,
    run_imex_diagnostic_scan,
)
from gkx.solvers_nonlinear_state_integration import (
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_sheared,
    integrate_nonlinear_sheared_transport,
    nonlinear_rhs_cached,
)
from gkx.terms.assembly import compute_fields_cached
from gkx.workflows.runtime.toml import load_runtime_from_toml

ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
REPO = Path(gkx.__file__).resolve().parents[2]
print(f"arm={ARM} gkx={gkx.__file__}", flush=True)

NL, NM = 4, 8
DT, STEPS = 1.0e-3, 8
deps = _runtime_linear_dispatch_deps().full_deps
base_cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")

HAS_SCAN_STATS = (
    "return_solve_stats" in inspect.signature(run_imex_diagnostic_scan).parameters
)
HAS_SHEARED_STATS = (
    "return_solve_stats" in inspect.signature(integrate_nonlinear_sheared).parameters
)
UNAVAILABLE = "unavailable on this arm"


def cover_from_gather(gather, *, ny: int, nx: int) -> np.ndarray:
    """Chain cover from a cache's flat gather mask, conjugate mirrors included.

    Spelled out here rather than imported, so the same file runs on both arms.
    """

    mask = np.reshape(np.asarray(gather, dtype=bool), (nx, ny)).T
    if ny > 1:
        rows = np.any(mask, axis=1)
        mirror_rows = np.mod(-np.arange(ny), ny)
        fill = ~rows & rows[mirror_rows] & (np.arange(ny) != 0)
        mirrored = mask[mirror_rows][:, np.mod(-np.arange(nx), nx)]
        mask = mask | (fill[:, None] & mirrored)
    return mask


def build_deck(boundary: str, *, nx: int = 8, ny: int = 8, nz: int = 16, nl: int = NL):
    """Return the nonlinear deck, its grid/geometry/params/cache and its cover."""

    cfg = replace(
        base_cfg,
        grid=GridConfig(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            ntheta=16,
            nperiod=1,
            boundary=boundary,
            jtwist=1 if boundary == "linked" else None,
        ),
        time=replace(base_cfg.time, damp_ends_rate=0.1),
        physics=replace(base_cfg.physics, linear=False, nonlinear=True),
        terms=replace(base_cfg.terms, nonlinear=1.0),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    params = build_runtime_linear_params(cfg, Nm=NM, geom=geom)
    terms = build_runtime_term_config(cfg)
    cache = deps.build_linear_cache(grid, geom, params, nl, NM)
    ny_g = int(np.asarray(grid.ky).size)
    nx_g = int(np.asarray(grid.kx).size)
    if bool(getattr(cache, "linked_use_gather", False)) and not bool(
        getattr(cache, "linked_full_cover", False)
    ):
        cover = cover_from_gather(cache.linked_gather_mask, ny=ny_g, nx=nx_g)
    else:
        cover = np.ones((ny_g, nx_g), dtype=bool)
    return cfg, grid, geom, params, terms, cache, cover


def hlo_fingerprint(lowered_text: str) -> str:
    """Instruction count and hash of optimized HLO, call-stack metadata stripped.

    A module's text also carries the Python call stack that produced it: a
    ``metadata={...}`` field per instruction and, at the top, string and
    ``FileLocations`` tables of file names, function names and *line numbers*.
    Those move whenever a source file is edited, which would make every arm
    differ for reasons that are not the graph. Keeping only the instruction
    lines and dropping their metadata leaves the graph itself.
    """

    kept = []
    for line in lowered_text.splitlines():
        if " = " not in line:
            continue
        line = re.sub(r"metadata=\{[^}]*\}", "", line)
        line = re.sub(r'source_file="[^"]*"|source_line=\d+', "", line)
        kept.append(line.rstrip())
    body = "\n".join(kept)
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    return f"{len(kept)} instructions, sha256 {digest}"


_COMPUTATION_HEAD = re.compile(r"^(ENTRY )?%[^ ]+ \(.*\) -> .* \{$")


def hlo_structure(lowered_text: str) -> str:
    """The same fingerprint, split into the entry and the scan's loop body.

    An instruction count alone cannot say *where* work was added: a per-step
    addition inside the loop body and a one-off addition before the loop are
    both a fixed number of instructions in a module whose trip count is a
    constant. Splitting the module by computation can. The loop body is the
    computation XLA names ``region_*`` for ``lax.scan``'s body; everything the
    driver does before the scan lands in ``ENTRY``.
    """

    computations: dict[str, list[str]] = {}
    current: str | None = None
    for line in lowered_text.splitlines():
        if _COMPUTATION_HEAD.match(line):
            current = line.split("(")[0].strip()
            computations[current] = []
            continue
        if line == "}":
            current = None
            continue
        if current is not None and " = " in line:
            stripped = re.sub(r"metadata=\{[^}]*\}", "", line)
            stripped = re.sub(r'source_file="[^"]*"|source_line=\d+', "", stripped)
            computations[current].append(stripped.rstrip())

    def fingerprint(lines: list[str]) -> str:
        """Count and a *name-normalized* hash of one computation.

        A raw text hash of a single computation is not comparable across arms:
        XLA numbers instructions module-wide, so adding anything to the entry
        renumbers ``%fusion.37`` into ``%fusion.38`` inside an untouched loop
        body. Rewriting every ``%name`` to its index of first appearance within
        the computation removes exactly that and nothing else -- operand order,
        opcodes, shapes and layouts are all still hashed -- so two computations
        that hash the same really are the same graph.
        """

        names: dict[str, int] = {}

        def canonical(match: re.Match[str]) -> str:
            token = match.group(0)
            return f"%v{names.setdefault(token, len(names))}"

        normalized = [re.sub(r"%[\w.$-]+", canonical, line) for line in lines]
        body = "\n".join(normalized)
        return (
            f"{len(lines)} instructions, normalized sha256 "
            + (hashlib.sha256(body.encode()).hexdigest()[:16])
        )

    entry = next(
        (v for k, v in computations.items() if k.startswith("ENTRY")),
        [],
    )
    bodies = {k: v for k, v in computations.items() if ".region_" in k}
    loop = max(bodies.values(), key=len) if bodies else []
    return (
        f"module {hlo_fingerprint(lowered_text)}; "
        f"ENTRY {fingerprint(entry)}; "
        f"scan body {fingerprint(loop)}"
    )


def checksum(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array)).tobytes()
    ).hexdigest()[:16]


# ---- 0. the deck, its cover and the supplied state -------------------------

cfg, grid, geom, params, terms, cache, cover = build_deck("linked")
ny_g = int(np.asarray(grid.ky).size)
nx_g = int(np.asarray(grid.kx).size)
nz_g = int(np.asarray(grid.z).size)
shape = (1, NL, NM, ny_g, nx_g, nz_g)
off = np.broadcast_to(~cover[:, :, None], shape)
_vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

rng = np.random.default_rng(1923)
supplied = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
supplied = (supplied / np.max(np.abs(supplied))).astype(np.complex64)
on_cover = np.where(off, 0.0, supplied).astype(np.complex64)

print(
    f"deck: ny={ny_g} nx={nx_g} nz={nz_g}; cover {int(cover.sum())}/{cover.size} "
    f"(ky, kx) modes; chain kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(np.any(cover, axis=0)))}; "
    f"off-chain kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(~np.any(cover, axis=0)))}",
    flush=True,
)
print(
    f"state {shape} n={int(np.prod(shape))} off-chain unknowns={int(off.sum())} "
    f"({off.mean():.4f} of the state)",
    flush=True,
)
print(
    f"supplied state: max|G|={np.max(np.abs(supplied)):.6e} "
    f"off-chain max={np.max(np.abs(supplied[off])):.6e}",
    flush=True,
)

# ---- 1. one nonlinear right-hand side --------------------------------------

dG_on_cover = np.asarray(
    nonlinear_rhs_cached(jnp.asarray(on_cover), cache, params, terms)[0]
)
dG_supplied = np.asarray(
    nonlinear_rhs_cached(jnp.asarray(supplied), cache, params, terms)[0]
)
chain_scale = np.max(np.abs(dG_on_cover[~off]))
rhs_rel = float(np.max(np.abs(dG_supplied[~off] - dG_on_cover[~off])) / chain_scale)
print(
    "one nonlinear RHS: chain rows change "
    f"{rhs_rel:.6e} relative when the off-chain rows are present "
    f"(chain |dG| max {chain_scale:.6e})",
    flush=True,
)
for amp in (1.0, 1.0e-2, 1.0e-4, 1.414214e-10):
    a = np.complex64(amp)
    d0 = np.asarray(
        nonlinear_rhs_cached(jnp.asarray(on_cover * a), cache, params, terms)[0]
    )
    d1 = np.asarray(
        nonlinear_rhs_cached(jnp.asarray(supplied * a), cache, params, terms)[0]
    )
    print(
        f"    amplitude max|G|={amp:.6e}: chain-row relative change="
        f"{float(np.max(np.abs(d1[~off] - d0[~off])) / np.max(np.abs(d0[~off]))):.6e}",
        flush=True,
    )

# ---- 2. the raw drivers from a supplied state ------------------------------


def raw_cached(state, *, steps: int = STEPS, deck=None):
    """``integrate_nonlinear_cached``: final state and its heat flux."""

    _cfg, _grid, _geom, _params, _terms, _cache, _cover = deck
    G_final, fields_t = integrate_nonlinear_cached(
        jnp.asarray(state),
        _cache,
        _params,
        DT,
        steps,
        method="rk3",
        terms=_terms,
    )
    phi = fields_t.phi[-1]
    zero = jnp.zeros_like(phi)
    apar = zero if fields_t.apar is None else fields_t.apar[-1]
    bpar = zero if fields_t.bpar is None else fields_t.bpar[-1]
    _vf, ff = fieldline_quadrature_weights(_geom, _grid)
    heat = heat_flux_total(G_final, phi, apar, bpar, _cache, _grid, _params, ff)
    return np.asarray(G_final), float(np.asarray(heat))


linked_deck = (cfg, grid, geom, params, terms, cache, cover)
G_sup, q_sup = raw_cached(supplied, deck=linked_deck)
G_cov, q_cov = raw_cached(on_cover, deck=linked_deck)
scale = np.max(np.abs(G_cov))
print(
    f"integrate_nonlinear_cached ({STEPS} rk3 steps, dt={DT}) from the supplied "
    "state vs the on-cover state: final-state max relative difference="
    f"{float(np.max(np.abs(G_sup - G_cov)) / scale):.6e}; "
    f"final-state off-chain max={np.max(np.abs(G_sup[off])):.6e}; "
    f"heat flux={q_sup:.12e} vs {q_cov:.12e}, relative difference="
    f"{abs(q_sup - q_cov) / abs(q_cov):.6e}",
    flush=True,
)

G_grid_door, _fields_grid_door = integrate_nonlinear(
    jnp.asarray(supplied),
    grid,
    geom,
    params,
    DT,
    STEPS,
    method="rk3",
    cache=cache,
    terms=terms,
)
print(
    "integrate_nonlinear from the same supplied state is bitwise equal to "
    f"integrate_nonlinear_cached={bool(np.array_equal(np.asarray(G_grid_door), G_sup))}",
    flush=True,
)

# ---- 3. the no-op arms and the compiled graphs -----------------------------

ic = np.asarray(
    _build_initial_condition(
        grid, geom, cfg, ky_index=1, kx_index=0, Nl=NL, Nm=NM, nspecies=1
    )
)
G_default, q_default = raw_cached(ic, deck=linked_deck)
print(
    "no-op arm, linked deck from the runtime's own on-cover initial condition: "
    f"final-state checksum={checksum(G_default)} heat flux={q_default:.12e}",
    flush=True,
)


def driver_graph(state, *, steps: int, deck):
    _cfg, _grid, _geom, _params, _terms, _cache, _cover = deck

    def run(G):
        return integrate_nonlinear_cached(
            G, _cache, _params, DT, steps, method="rk3", terms=_terms
        )

    return jax.jit(run).lower(jnp.asarray(state)).compile().as_text()


for steps in (8, 16):
    text = driver_graph(on_cover, steps=steps, deck=linked_deck)
    print(
        f"linked deck, jit(integrate_nonlinear_cached), steps={steps}: "
        f"optimized HLO = {hlo_structure(text)}",
        flush=True,
    )

pcfg, pgrid, pgeom, pparams, pterms, pcache, pcover = build_deck("periodic")
periodic_deck = (pcfg, pgrid, pgeom, pparams, pterms, pcache, pcover)
pshape = (
    1,
    NL,
    NM,
    int(np.asarray(pgrid.ky).size),
    int(np.asarray(pgrid.kx).size),
    int(np.asarray(pgrid.z).size),
)
prng = np.random.default_rng(2319)
pstate = (prng.normal(size=pshape) + 1j * prng.normal(size=pshape)).astype(np.complex64)
pG, pq = raw_cached(pstate, deck=periodic_deck)
print(
    f"no-op arm, periodic deck: cover is every mode={bool(pcover.all())}; "
    f"final-state checksum={checksum(pG)} heat flux={pq:.12e}",
    flush=True,
)
for steps in (8, 16):
    text = driver_graph(pstate, steps=steps, deck=periodic_deck)
    print(
        f"periodic deck, jit(integrate_nonlinear_cached), steps={steps}: "
        f"optimized HLO = {hlo_structure(text)}",
        flush=True,
    )

# ---- 4. the IMEX diagnostics route -----------------------------------------
#
# A small deck: the status question is about the channel, not the grid. The
# starved arm uses one GMRES iteration against a tolerance it cannot reach.

scfg, sgrid, sgeom, sparams, sterms, scache, scover = build_deck(
    "linked", nx=4, ny=4, nz=8, nl=2
)
small_deck = (scfg, sgrid, sgeom, sparams, sterms, scache, scover)
sshape = (
    1,
    2,
    NM,
    int(np.asarray(sgrid.ky).size),
    int(np.asarray(sgrid.kx).size),
    int(np.asarray(sgrid.z).size),
)
srng = np.random.default_rng(704)
sstate = (srng.normal(size=sshape) + 1j * srng.normal(size=sshape)).astype(np.complex64)
sstate = (sstate / np.max(np.abs(sstate)) * 1.0e-3).astype(np.complex64)

STARVED = {"implicit_tol": 1.0e-14, "implicit_maxiter": 1, "implicit_restart": 1}
GENEROUS = {"implicit_tol": 1.0e-8, "implicit_maxiter": 200, "implicit_restart": 20}
IMEX_STEPS = 4


def imex_diagnostics_stats(budget):
    if not HAS_SCAN_STATS:
        return None
    _t, _diag, stats = integrate_nonlinear_imex_diagnostics(
        jnp.asarray(sstate),
        sgrid,
        sgeom,
        sparams,
        DT,
        IMEX_STEPS,
        terms=sterms,
        return_solve_stats=True,
        **budget,
    )
    return stats


def report_stats(label, stats):
    if stats is None:
        print(f"{label}: {UNAVAILABLE}", flush=True)
        return
    try:
        summary = require_converged_implicit_solves(stats, label=label)
        verdict = "host gate accepts"
    except RuntimeError as exc:
        summary = None
        verdict = f"host gate refuses: {type(exc).__name__}"
    solves = int(np.asarray(stats.solves))
    unconverged = int(np.asarray(stats.unconverged_solves))
    residual = float(np.asarray(stats.max_relative_residual))
    iterations = int(np.asarray(stats.max_iterations))
    converged = "n/a" if summary is None else summary.converged
    print(
        f"{label}: solves={solves} unconverged={unconverged} "
        f"max_relative_residual={residual:.6e} max_iterations={iterations} "
        f"converged={converged}; {verdict}",
        flush=True,
    )


report_stats("IMEX diagnostics, starved budget", imex_diagnostics_stats(STARVED))
report_stats("IMEX diagnostics, generous budget", imex_diagnostics_stats(GENEROUS))


def imex_diagnostics_graphs():
    """Optimized HLO of the IMEX diagnostic scan, with and without stats."""

    options = _imex_option_bundle(
        cache=scache,
        terms=sterms,
        collision_split=False,
        implicit_preconditioner=None,
        compressed_real_fft=True,
        use_dealias_mask=False,
        z_index=None,
        fixed_mode_ky_index=None,
        fixed_mode_kx_index=None,
        external_phi=None,
        laguerre_mode="grid",
        implicit_iters=3,
        implicit_relax=0.7,
        implicit_tol=GENEROUS["implicit_tol"],
        implicit_maxiter=GENEROUS["implicit_maxiter"],
        implicit_restart=GENEROUS["implicit_restart"],
        omega_ky_index=None,
        omega_kx_index=None,
        flux_scale=1.0,
        wphi_scale=1.0,
        method="imex",
        steps=IMEX_STEPS,
        checkpoint=False,
        sample_stride=1,
        diagnostics_stride=1,
        show_progress=False,
        collision_scheme="implicit",
    )
    context = _build_imex_scan_context(
        jnp.asarray(sstate),
        sgrid,
        sgeom,
        sparams,
        DT,
        deps=_imex_nonlinear_diagnostics_deps(),
        preparation=options.preparation,
        runtime=options.runtime,
        diagnostics=options.diagnostics,
        scan=options.scan,
    )
    prepared = context.prepared
    fields0 = compute_fields_cached(
        prepared.G0, prepared.cache, sparams, terms=prepared.term_cfg
    )
    diag0 = context.compute_diag_from_state(
        prepared.G0, fields0, prepared.G0, fields0, prepared.dt_val
    )
    carry = (
        prepared.G0,
        prepared.G0,
        fields0,
        diag0,
        jnp.asarray(0.0, dtype=prepared.real_dtype),
    )
    if HAS_SCAN_STATS:
        from gkx.solvers_linear_implicit import _empty_implicit_solve_stats

        carry = (*carry, _empty_implicit_solve_stats(prepared.state_dtype))

    def plain(c):
        return run_imex_diagnostic_scan(
            context.step, c, steps=IMEX_STEPS, checkpoint=False
        )

    texts = {"plain": jax.jit(plain).lower(carry).compile().as_text()}
    if HAS_SCAN_STATS:

        def with_stats(c):
            return run_imex_diagnostic_scan(
                context.step,
                c,
                steps=IMEX_STEPS,
                checkpoint=False,
                return_solve_stats=True,
            )

        texts["with_stats"] = jax.jit(with_stats).lower(carry).compile().as_text()
    return texts


for name, text in imex_diagnostics_graphs().items():
    print(
        f"IMEX diagnostics scan HLO ({name}): {hlo_fingerprint(text)}",
        flush=True,
    )

# ---- 5. the sheared IMEX route ---------------------------------------------

SHEAR = 0.05


def sheared_stats(budget):
    if not HAS_SHEARED_STATS:
        return None, None
    _state, _fields, stats = integrate_nonlinear_sheared(
        jnp.asarray(sstate),
        sgrid,
        sgeom,
        sparams,
        DT,
        IMEX_STEPS,
        shear_rate=SHEAR,
        method="imex",
        cache=scache,
        terms=sterms,
        return_solve_stats=True,
        **budget,
    )
    trace = integrate_nonlinear_sheared_transport(
        jnp.asarray(sstate),
        sgrid,
        sgeom,
        sparams,
        DT,
        IMEX_STEPS,
        shear_rate=SHEAR,
        method="imex",
        cache=scache,
        terms=sterms,
        return_solve_stats=True,
        **budget,
    )
    return stats, trace.solve_stats


for label, budget in (("starved", STARVED), ("generous", GENEROUS)):
    direct, transport = sheared_stats(budget)
    report_stats(f"sheared IMEX, {label} budget", direct)
    report_stats(f"sheared IMEX transport, {label} budget", transport)


def sheared_graph(method: str, *, with_stats: bool):
    kwargs = dict(
        shear_rate=SHEAR,
        method=method,
        cache=scache,
        terms=sterms,
    )
    if method == "imex":
        kwargs.update(GENEROUS)
    if with_stats:
        kwargs["return_solve_stats"] = True

    def run(G):
        return integrate_nonlinear_sheared(
            G, sgrid, sgeom, sparams, DT, IMEX_STEPS, **kwargs
        )

    return jax.jit(run).lower(jnp.asarray(sstate)).compile().as_text()


print(
    f"sheared IMEX scan HLO (plain): {hlo_fingerprint(sheared_graph('imex', with_stats=False))}",
    flush=True,
)
if HAS_SHEARED_STATS:
    print(
        "sheared IMEX scan HLO (with_stats): "
        f"{hlo_fingerprint(sheared_graph('imex', with_stats=True))}",
        flush=True,
    )
print(
    "sheared rk2 scan HLO (control, no implicit solve): "
    f"{hlo_fingerprint(sheared_graph('rk2', with_stats=False))}",
    flush=True,
)

# ---- 6. the saved summary files --------------------------------------------

from gkx.runtime import run_runtime_linear  # noqa: E402

pilot = replace(
    base_cfg,
    grid=replace(base_cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(base_cfg.time, damp_ends_rate=0.1),
)
STATUS_KEYS = (
    "eigen_route",
    "eigen_residual",
    "eigen_tolerance",
    "eigen_certified",
    "eigen_inner_converged",
    "implicit_converged",
    "implicit_max_relative_residual",
    "implicit_max_iterations",
    "implicit_unconverged_solves",
)


def saved_status(label, result):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pilot.summary.json"
        write_runtime_linear_artifacts(out, result)
        saved = json.loads(out.read_text(encoding="utf-8"))
    present = [k for k in STATUS_KEYS if k in saved]
    print(
        f"saved linear summary ({label}): solver-status keys present="
        f"{present if present else UNAVAILABLE}",
        flush=True,
    )
    for key in present:
        value = saved[key]
        shown = f"{value:.6e}" if isinstance(value, float) else value
        print(f"    {key}={shown}", flush=True)


saved_status(
    "krylov",
    run_runtime_linear(pilot, ky_target=0.3, Nl=NL, Nm=NM, solver="krylov"),
)
# The implicit route is the runtime's own time integrator, not the
# CFL-controlled explicit one, so this arm asks for "time" rather than
# "explicit_time": that is the route #242 made fail closed, and the one whose
# status a saved file had no way to report.
implicit_pilot = replace(
    pilot, time=replace(pilot.time, method="implicit", t_max=1.0, dt=0.02)
)
saved_status(
    "implicit time run",
    run_runtime_linear(implicit_pilot, ky_target=0.3, Nl=NL, Nm=NM, solver="time"),
)

# ---- 7. the linear pilot must not move -------------------------------------

from gkx.workflows.linear import _prepare_linear_runtime_context  # noqa: E402

pilot_ctx = _prepare_linear_runtime_context(
    pilot,
    deps=deps,
    ky_target=0.3,
    n_laguerre=NL,
    n_hermite=NM,
    solver="krylov",
    fit_signal="auto",
    return_state=True,
    initial_state=None,
    status_callback=None,
)
pilot_ic = np.asarray(pilot_ctx.initial_state)
pilot_off = np.zeros(pilot_ic.shape, dtype=bool)
pilot_off[..., [3, 4, 5], :] = True
prng2 = np.random.default_rng(1918)
pilot_noise = (
    prng2.normal(size=pilot_ic.shape) + 1j * prng2.normal(size=pilot_ic.shape)
).astype(np.complex64)
pilot_noise = pilot_noise * np.complex64(
    np.max(np.abs(pilot_ic)) / np.max(np.abs(pilot_noise))
)
pilot_supplied = np.where(pilot_off, pilot_noise, pilot_ic).astype(np.complex64)
for solver, tag in (("krylov", "certified eigenpair"), ("explicit_time", "time fit")):
    res = run_runtime_linear(
        pilot,
        ky_target=0.3,
        Nl=NL,
        Nm=NM,
        solver=solver,
        initial_state=pilot_supplied,
    )
    status = getattr(res, "eigen_status", None)
    extra = (
        ""
        if status is None
        else (
            f" residual={float(status.residual):.6e} "
            f"certified={bool(status.certified)} route={status.route}"
        )
    )
    print(
        f"linear pilot {tag} ({solver}) from a supplied state: "
        f"gamma={float(res.gamma):.18e} omega={float(res.omega):.18e}{extra}",
        flush=True,
    )
