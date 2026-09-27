#!/usr/bin/env python3
"""GKX certified (adaptive Krylov) eigenpair on the GX-matched Cyclone decks.

    python gkx_cert.py S|M <ky> <Nl> <Nm>
S = examples/01_linear_tokamak/case_full.toml of the checkout on PYTHONPATH/ installed; M = cyclone_miller_linear.toml here.
Prints one line "RESULT {json}".
"""

import json
import sys
import time
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
import gkx  # noqa: E402
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

geom, ky, nl, nm = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
deck = (
    (Path(gkx.__file__).parents[2] / "examples/01_linear_tokamak/case_full.toml")
    if geom == "S"
    else Path(__file__).with_name("cyclone_miller_linear.toml")
)
cfg, _ = load_runtime_from_toml(deck)
t0 = time.perf_counter()
res = run_runtime_linear(cfg, ky_target=ky, Nl=nl, Nm=nm, solver="krylov")
print(
    "RESULT",
    json.dumps(
        dict(
            geometry=geom,
            deck=deck.name,
            Nl=nl,
            Nm=nm,
            ky=res.ky,
            gamma=res.gamma,
            omega=res.omega,
            wall_s=round(time.perf_counter() - t0, 1),
        )
    ),
    flush=True,
)
