#!/usr/bin/env python3
"""GS2 collisional TEM scan (Dannert & Jenko 2005 case) at ky rho_s = 0.3.

    python tem_gs2_coll.py write <root>    write <root>/gs2/<case>/<case>.in, print the queue
    python tem_gs2_coll.py fit <root>      CSV of final-window gamma/omega (GKX units)

Reuses the collisionless input of plan/research/2026-10-06-val-ke/scripts/tem_gs2.py
(rung t1) and switches on GS2's default model operator (pitch-angle scattering
and energy diffusion with momentum and energy restoring terms; electron-ion
scattering through zeff = 1). vnewk_e is the scan variable; the ion rate is the
physically consistent vnewk_i = vnewk_e (m_e/m_i)^(1/2) (T_e/T_i)^(3/2).
GKX takes nu = sqrt(2) vnewk (docs/conventions.rst).
"""

import math
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "2026-10-06-val-ke" / "scripts")
)
import tem_gs2  # noqa: E402

VNEWK_E = (0.05, 0.2, 0.8)
RATIO = math.sqrt(1.0 / 1836.0) * 3.0**1.5  # vnewk_i / vnewk_e


def gs2_input(name, vnewk_e):
    text = tem_gs2.gs2_input(name, 0.3, tem_gs2.RUNGS["t1"])
    text = text.replace('collision_model = "none"', 'collision_model = "default"')
    head, ion, electron = (
        text.split("vnewk = 0.0", 2)[0],
        *text.split("vnewk = 0.0")[1:],
    )
    return (
        head
        + f"vnewk = {vnewk_e * RATIO:.10e}"
        + ion
        + f"vnewk = {vnewk_e:.10e}"
        + electron
    )


if __name__ == "__main__":
    if sys.argv[1] == "write":
        for v in VNEWK_E:
            name = f"TEM_ky0.30_t1_vnewk{v:g}"
            d = Path(sys.argv[2]) / "gs2" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.in").write_text(gs2_input(name, v))
            print(
                f"gs2 {name} nu_gkx={math.sqrt(2) * v * RATIO:.6e},{math.sqrt(2) * v:.6e}"
            )
    else:
        tem_gs2.fit(sys.argv[2])
