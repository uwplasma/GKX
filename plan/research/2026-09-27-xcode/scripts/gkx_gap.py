#!/usr/bin/env python3
"""GKX side of the term-knockout gap study: circular Miller, ky = 0.30, certified eigenpair.

    python gkx_gap.py <variant> <Nl> <Nm>   (variant: base|nodrift|nomirror|nodrift_nomirror)
Prints one line "RESULT {json}".
"""
import json, sys, time
from dataclasses import replace
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
from gkx.runtime import run_runtime_linear
from gkx.workflows.runtime.toml import load_runtime_from_toml

var, nl, nm = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg, _ = load_runtime_from_toml(Path(__file__).with_name("cyclone_miller_linear.toml"))
kw = {}
if "nodrift" in var:
    kw.update(curvature=0.0, gradb=0.0)
if "nomirror" in var:
    kw["mirror"] = 0.0
cfg = replace(cfg, terms=replace(cfg.terms, **kw))
t0 = time.perf_counter()
res = run_runtime_linear(cfg, ky_target=0.3, Nl=nl, Nm=nm, solver="krylov")
print("RESULT", json.dumps(dict(variant=var, Nl=nl, Nm=nm, ky=res.ky, gamma=res.gamma, omega=res.omega,
      wall_s=round(time.perf_counter() - t0, 1))), flush=True)
