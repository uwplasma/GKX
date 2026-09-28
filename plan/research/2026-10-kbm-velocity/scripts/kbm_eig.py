#!/usr/bin/env python3
"""GKX KBM eigenpair on examples/06_electromagnetic/case_full.toml with deck variants.

    python kbm_eig.py <S|M> <ky> <Nl> <Nm> <variant> <out.npz> [shift_re shift_im]
S = examples/06_electromagnetic/case_full.toml (s-alpha); M = kbm_miller.toml next to this script.
The optional shift seeds the KBM shift-invert Krylov route (lambda = gamma - i omega).
variant: '+'-joined subset of {base, nohyper, coll, bpar, rk4}.
  damp<r>: [time] damp_ends_rate = r;  np<k>: nperiod = k, Nz = 32 (2k - 1)
  rk4:     [time] method = "rk4" (time mode only)
  nohyper: [physics] hypercollisions = false and [terms] hypercollisions = 0
  coll:    species nu = 0.01 (ion) / 0.6 (electron, sqrt(mi/me) scaling)
  bpar:    use_bpar = true, [terms] bpar = 1
Prints "RESULT {json}" and saves the eigenvector with its Hermite/Laguerre spectra.
"""

import json
from dataclasses import replace
import os
import sys
import tempfile
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import gkx  # noqa: E402
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

geom, ky, nl, nm, variant, out = (
    sys.argv[1],
    float(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
    sys.argv[6],
)
deck = (
    Path(gkx.__file__).parents[2] / "examples/06_electromagnetic/case_full.toml"
    if geom == "S"
    else Path(__file__).with_name("kbm_miller.toml")
)
kcfg = None
if os.environ.get("KBM_SPARSE"):
    import gkx.runtime as _rt

    import sparse_eig

    _rt.dominant_eigenpair = sparse_eig.dominant_eigenpair
if len(sys.argv) > 8:
    from gkx.benchmarking_shared import KBM_KRYLOV_DEFAULT

    shift = complex(float(sys.argv[7]), float(sys.argv[8]))
    kcfg = replace(
        KBM_KRYLOV_DEFAULT,
        shift=shift,
        shift_maxiter=200,
        shift_restart=40,
        krylov_dim=24,
    )
text = deck.read_text()


def sub(a, b, count=1):
    global text
    assert text.count(a) == count, (a, text.count(a))
    text = text.replace(a, b)


for v in variant.split("+"):
    if v == "nohyper":
        sub("\nhypercollisions = true\n", "\nhypercollisions = false\n")
        sub("\nhypercollisions = 1.0\n", "\nhypercollisions = 0.0\n")
    elif v == "coll":
        sub("nu = 0.0\n", "nu = __NU__\n", 2)
        text = text.replace("nu = __NU__\n", "nu = 0.01\n", 1).replace(
            "nu = __NU__\n", "nu = 0.6\n", 1
        )
    elif v == "rk4":
        sub('method = "imex2"', 'method = "rk4"')
    elif v.startswith(
        "damp"
    ):  # damp<rate>: explicit end-damping rate (time route default is amp/dt)
        sub("[time]\n", f"[time]\ndamp_ends_rate = {float(v[4:])}\n")
    elif v.startswith(
        "np"
    ):  # np<k>: nperiod k, Nz = ntheta (2k - 1) (Miller deck only)
        k = int(v[2:])
        sub("Nz = 96\n", f"Nz = {32 * (2 * k - 1)}\n")
        sub("nperiod = 2\n", f"nperiod = {k}\n")
    elif v == "bpar":
        sub("use_bpar = false\n", "use_bpar = true\n")
        sub("\nbpar = 0.0\n", "\nbpar = 1.0\n")
    else:
        assert v == "base", v
with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
    f.write(text)
cfg, _ = load_runtime_from_toml(Path(f.name))
t0 = time.perf_counter()
if os.environ.get("KBM_TIME", "").startswith(
    "adapt"
):  # "adapt,t_max": rk4 at the adaptive CFL step
    ttmax = float(os.environ["KBM_TIME"].split(",")[1])
    cfg = replace(
        cfg,
        time=replace(
            cfg.time, method="rk4", fixed_dt=False, t_max=ttmax, sample_stride=20
        ),
    )
    res = run_runtime_linear(
        cfg, ky_target=ky, Nl=nl, Nm=nm, solver="time", return_state=True
    )
elif os.environ.get(
    "KBM_TIME"
):  # "dt,t_max": GX-style time integration with the runtime fit
    tdt, ttmax = (float(s) for s in os.environ["KBM_TIME"].split(","))
    res = run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=nl,
        Nm=nm,
        solver="time",
        dt=tdt,
        steps=2 * int(round(ttmax / tdt / 2)),
        return_state=True,
    )
else:
    res = run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=nl,
        Nm=nm,
        solver="krylov",
        return_state=True,
        krylov_cfg=kcfg,
    )
wall = time.perf_counter() - t0
st = np.asarray(res.state)
# state layout: (Ns, Nl, Nm, ...) -> spectra per species
axes = tuple(range(3, st.ndim))
p = (np.abs(st) ** 2).sum(axis=axes)
herm = p.sum(axis=1)  # (Ns, Nm)
lag = p.sum(axis=2)  # (Ns, Nl)
herm_n, lag_n = herm / herm.sum(1, keepdims=True), lag / lag.sum(1, keepdims=True)
status = res.eigen_status
resid = None
if status is not None:
    resid = getattr(status, "residual", None) or getattr(
        status, "relative_residual", None
    )
row = dict(
    time=os.environ.get("KBM_TIME"),
    fit_settled=getattr(res, "fit_settled", None),
    fit_r2=getattr(res, "fit_r2", None),
    geometry=geom,
    ky=res.ky,
    Nl=nl,
    Nm=nm,
    variant=variant,
    gamma=res.gamma,
    omega=res.omega,
    residual=None if resid is None else float(resid),
    wall_s=round(wall, 1),
    state_shape=list(st.shape),
    sparse=None if not os.environ.get("KBM_SPARSE") else sparse_eig.INFO,
    herm_tail=[float(h[-max(1, nm // 8) :].sum()) for h in herm_n],
    lag_tail=[float(h[-max(1, nl // 8) :].sum()) for h in lag_n],
)
np.savez_compressed(
    out,
    state=st,
    herm=herm_n,
    lag=lag_n,
    **{k: v for k, v in row.items() if isinstance(v, float)},
)
print("RESULT", json.dumps(row), flush=True)
