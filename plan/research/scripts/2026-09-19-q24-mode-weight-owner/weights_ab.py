"""Q24 A/B: every shipped ``ky`` mode weight, for whichever gkx is on PYTHONPATH.

Usage: python weights_ab.py OUT.json        (run once per tree)
       python weights_ab.py --diff A.json B.json

The weight rule is the only thing queue row Q24 changes, so the proof that no
shipped number moves is that the weight arrays themselves are bit-identical
everywhere the changed row is not reached.  The ladder covers ``Ny`` 2..17 --
even and odd, so both sides of "Nyc does not determine Ny" are exercised --
against ``Nx`` in {1, 2, 3, 4, 8}, both layouts, and both the dealiased and the
undealiased weight.  Arrays are compared as raw bytes, not to a tolerance.
"""

from __future__ import annotations

import json
import sys

import numpy as np

NY_LADDER = range(2, 18)
NX_LADDER = (1, 2, 3, 4, 8)


def dump(path: str) -> None:
    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid, select_real_fft_ky_grid
    from gkx.core_ky_layout import half_ky_values
    from gkx.operators.moments import _hermitian_mode_weight, _transport_mode_weight

    out: dict[str, list[str]] = {}
    for ny in NY_LADDER:
        for nx in NX_LADDER:
            cfg = GridConfig(Nx=nx, Ny=ny, Nz=2, Lx=6.0, Ly=6.0)
            full = build_spectral_grid(cfg)
            half = select_real_fft_ky_grid(full, half_ky_values(full.ky))
            for layout, grid in (("full", full), ("half", half)):
                for dealias in (False, True):
                    for fn in (_hermitian_mode_weight, _transport_mode_weight):
                        key = (
                            f"{fn.__name__}/{layout}/ny{ny}/nx{nx}"
                            f"/dealias{int(dealias)}"
                        )
                        arr = np.asarray(fn(grid, use_dealias=dealias))
                        out[key] = [str(arr.dtype), arr.tobytes().hex()]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=0, sort_keys=True)
    print(f"{len(out)} cases -> {path}")


def diff(left_path: str, right_path: str) -> None:
    with open(left_path, encoding="utf-8") as handle:
        left = json.load(handle)
    with open(right_path, encoding="utf-8") as handle:
        right = json.load(handle)
    assert set(left) == set(right), "the two trees produced different case sets"
    moved = sorted(key for key in left if left[key] != right[key])
    print(f"cases {len(left)}  bitwise {len(left) - len(moved)}  moved {len(moved)}")
    for key in moved:
        print(f"  {key}")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        dump(sys.argv[1])
