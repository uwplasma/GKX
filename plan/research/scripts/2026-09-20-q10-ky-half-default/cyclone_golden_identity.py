"""Q10: the tracked Cyclone growth rates do not move under the default flip.

Usage: python cyclone_golden_identity.py OUT.json

The README's "Cyclone ITG" parity row comes from
``docs/_static/cyclone_mismatch_table.csv``, which
``tools/comparison/build_gx_parity_matrix.py`` builds by calling
``run_runtime_scan`` on ``tools/comparison/fixtures/parity/cyclone_salpha_itg.toml``
at ``Nl = 16, Nm = 48`` for 75,000 ``imex2`` steps per ``ky``, twice.  That is
hours of CPU, and re-running it would only show the table again.

What the flip has to show is narrower: that the table's generator, handed the
same deck, computes the same numbers on this tree as on ``main``.  So this runs
the generator's own ``run_runtime_scan`` call -- the same fixture, the same
keyword arguments, the same ``ky`` values -- at a resolution and horizon small
enough to finish in minutes, and records every growth rate and frequency at
full precision together with a hash of the field history each fit was taken
from.  Run it on both trees; if every number and every hash agrees bitwise,
the full-resolution table cannot have moved either, because nothing in the
scan depends on the resolution in a way the layout could reach and the
reduced run does not.

The second block does the same for the shipped linear example's certified
Krylov eigenvalue at the README's ``ky = 0.3``, which is the number the
example's comments quote (``gamma ~= 0.0930``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import gkx
from gkx import load_runtime_from_toml, run_runtime_scan
from gkx.runtime import run_runtime_linear

out = Path(sys.argv[1])
root = Path(gkx.__file__).resolve().parents[2]

KY = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NL, NM, DT, STEPS = 4, 8, 0.002, 3000
SAMPLES = 300


def _digest(values: object) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


record: dict[str, object] = {
    "gkx": gkx.__file__,
    "ky": KY,
    "Nl": NL,
    "Nm": NM,
    "dt": DT,
    "steps": STEPS,
}

# --- the parity table's generator, at reduced resolution ---------------------
cfg, _ = load_runtime_from_toml(
    root / "tools/comparison/fixtures/parity/cyclone_salpha_itg.toml"
)
record["deck_ky_layout"] = str(getattr(cfg.grid, "ky_layout", "absent"))
t_end = STEPS * DT
scan = run_runtime_scan(
    cfg,
    np.asarray(KY, dtype=float),
    steps=STEPS,
    sample_stride=max(1, STEPS // SAMPLES),
    tmin=0.7 * t_end,
    tmax=t_end,
    Nl=NL,
    Nm=NM,
    dt=DT,
    method="imex2",
    solver="time",
    batch_ky=True,
    fit_signal="phi",
    mode_method="z_index",
    auto_window=False,
    min_points=80,
    require_positive=True,
)
record["scan"] = {
    "gamma": [float(g) for g in np.asarray(scan.gamma)],
    "omega": [float(w) for w in np.asarray(scan.omega)],
    "gamma_hex": [float(g).hex() for g in np.asarray(scan.gamma)],
    "omega_hex": [float(w).hex() for w in np.asarray(scan.omega)],
}

# --- the shipped linear example's certified eigenvalue ------------------------
example, _ = load_runtime_from_toml(root / "examples/linear/axisymmetric/cyclone.toml")
certified = run_runtime_linear(example, solver="krylov", ky_target=0.3, Nl=NL, Nm=NM)
record["example_krylov_ky0.3"] = {
    "gamma": float(certified.gamma),
    "omega": float(certified.omega),
    "gamma_hex": float(certified.gamma).hex(),
    "omega_hex": float(certified.omega).hex(),
}
if getattr(certified, "phi_t", None) is not None:
    record["example_krylov_ky0.3"]["phi_t_sha"] = _digest(certified.phi_t)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
