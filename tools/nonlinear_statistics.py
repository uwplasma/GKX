"""Nonlinear heat-flux statistics: run a deck variant and report mean +- SEM.

    python tools/nonlinear_statistics.py run  DECK OUT --tprim 2.49 --Nl 4 --Nm 8 --t-max 1000
    python tools/nonlinear_statistics.py stats OUT/*.npz [--t-min T]

Protocol: the transient is discarded where the trace first reaches its running
median of the second half (or at --t-min); the retained window is block
averaged. The SEM uses the Sokal integrated autocorrelation time shared with
the runtime stop policy, ``sem = std / sqrt(n dt / (2 tau))``. A window is
accepted only if it spans >= 20 tau and its two halves agree within 2 SEM.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from gkx.diagnostics.analysis import sokal_autocorrelation_time


def window_stats(t, q, t_min=None):
    t, q = np.asarray(t, float), np.asarray(q, float)
    if t_min is None:  # first crossing of the median of the later half
        ref = np.median(q[q.size // 2 :])
        idx = np.nonzero(q >= ref)[0]
        t_min = float(t[idx[0]]) if idx.size else float(t[0])
        t_min = max(t_min, 0.25 * float(t[-1]))
    sel = t >= t_min
    tw, qw = t[sel], q[sel]
    dt = float(np.median(np.diff(tw)))
    tau, cut, rho = sokal_autocorrelation_time(qw, dt)
    n_eff = min(qw.size, qw.size * dt / (2 * tau)) if tau > 0 else qw.size
    sem = float(np.std(qw, ddof=1) / np.sqrt(max(n_eff, 1.0)))
    h = qw.size // 2
    halves = []
    for part in (qw[:h], qw[h:]):
        tp, _, _ = sokal_autocorrelation_time(part, dt)
        ne = min(part.size, part.size * dt / (2 * tp)) if tp > 0 else part.size
        halves.append(
            (float(part.mean()), float(np.std(part, ddof=1) / np.sqrt(max(ne, 1.0))))
        )
    stationary = abs(halves[0][0] - halves[1][0]) <= 2 * np.hypot(
        halves[0][1], halves[1][1]
    )
    span = float(tw[-1] - tw[0])
    return dict(
        t_min=float(tw[0]),
        t_max=float(tw[-1]),
        n=int(qw.size),
        mean=float(qw.mean()),
        sem=sem,
        tau=float(tau),
        span_in_tau=span / tau if tau > 0 else float("inf"),
        stationary=bool(stationary),
        accepted=bool(stationary and span >= 20 * tau),
    )


def _run(a):
    import gkx

    txt = Path(a.deck).read_text()
    sub = {
        "tprim": a.tprim,
        "Nl": a.Nl,
        "Nm": a.Nm,
        "t_max": a.t_max,
        "dt": a.dt,
        "method": f'"{a.method}"' if a.method else None,
    }
    if a.dt:  # fixed step: the CFL controller costs ~5x per step on GPU
        txt = re.sub(r"^fixed_dt\s*=.*$", "fixed_dt = true", txt, flags=re.M)
    for k, v in sub.items():
        if v is not None:
            txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {v}", txt, flags=re.M)
    txt = re.sub(r"^steps\s*=.*\n", "", txt, flags=re.M)
    txt = re.sub(r"^run_to\s*=.*\n", "", txt, flags=re.M)
    txt = txt.replace("[time]\n", '[time]\nrun_to = "t_max"\n', 1)
    if a.nu is not None:  # ion-ion collision frequency, in the species table
        txt = re.sub(r"^nu\s*=.*\n", "", txt, flags=re.M)
        txt = txt.replace("kinetic = true\n", f"kinetic = true\nnu = {a.nu}\n", 1)
    for extra in a.set:
        sec, key, val = re.match(r"(\w+)\.(\w+)=(.*)", extra).groups()
        if re.search(rf"^{key}\s*=", txt, flags=re.M):
            txt = re.sub(rf"^{key}\s*=.*$", f"{key} = {val}", txt, count=1, flags=re.M)
        else:
            txt = txt.replace(f"[{sec}]\n", f"[{sec}]\n{key} = {val}\n", 1)
    if a.ke:  # kinetic electrons at m_i/m_e = 3670, A_par at beta = 1e-4
        ion = re.search(
            r"\[\[species\]\].*?kinetic = true\n(nu = .*\n)?", txt, flags=re.S
        ).group(0)
        el = ion.replace('"ion"', '"electron"').replace("charge = 1.0", "charge = -1.0")
        el = el.replace("mass = 1.0", "mass = 0.00027248")
        txt = txt.replace(ion, ion + "\n" + el, 1)
        for k, v in (
            ("electrostatic", "false"),
            ("electromagnetic", "true"),
            ("adiabatic_electrons", "false"),
            ("beta", "1.0e-4"),
        ):
            txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {v}", txt, flags=re.M)
        txt = txt.replace(
            "[physics]\n", "[physics]\nuse_apar = true\nuse_bpar = false\n", 1
        )
        txt = txt.replace("[terms]\n", "[terms]\napar = 1.0\nbpar = 0.0\n", 1)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deck = Path(f"{out}.toml")
    deck.write_text(txt)
    case = gkx.load(deck)
    res = gkx.solve(
        case,
        ky_target=case.run.ky,
        Nl=case.run.Nl,
        Nm=case.run.Nm,
        steps=round(a.t_max / a.dt) if a.dt else None,
        show_progress=False,
    )
    d = res.diagnostics
    np.savez(
        f"{out}.npz",
        t=np.asarray(d.t),
        q=np.asarray(d.heat_flux_t),
        qs=np.asarray(d.heat_flux_species_t),
        wg=np.asarray(d.Wg_t),
    )


def _stats(a):
    for f in a.files:
        z = np.load(f)
        s = window_stats(z["t"], z["q"], a.t_min)
        print(
            f"RESULT {Path(f).stem}: Q={s['mean']:.3f} +- {s['sem']:.3f} "
            f"window=[{s['t_min']:.0f},{s['t_max']:.0f}] tau={s['tau']:.1f} "
            f"span/tau={s['span_in_tau']:.1f} stationary={s['stationary']} accepted={s['accepted']}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("deck")
    r.add_argument("out")
    r.add_argument("--tprim", type=float)
    r.add_argument("--Nl", type=int)
    r.add_argument("--Nm", type=int)
    r.add_argument("--t-max", dest="t_max", type=float)
    r.add_argument("--dt", type=float)
    r.add_argument("--method")
    r.add_argument("--ke", action="store_true", help="add kinetic electrons + A_par")
    r.add_argument("--nu", type=float)
    r.add_argument("--set", nargs="*", default=[], help="section.key=value overrides")
    s = sp.add_parser("stats")
    s.add_argument("files", nargs="+")
    s.add_argument("--t-min", type=float)
    a = p.parse_args()
    _run(a) if a.cmd == "run" else _stats(a)
