# ruff: noqa: E402
"""V1: re-read existing GX Nl24 / Nl32 outputs (read-only; no simulation).

Inputs (office): nl24.out.nc (Nl24, this deck) and full96.out.nc (Nl32), plus big.nc Phi snapshots.
Prints W(l), W(m) time evolution, tail fractions, P(Nl-1, m<=2), gamma(t) windows, |phi|^2(theta),
and the complex overlap of the final phi(theta) between the two runs.
"""

import netCDF4 as nc
import numpy as np

runs = {
    "Nl24": (
        "/home/rjorge/gkx-nl24-discriminator-20260912.vvmgDD/nl24.out.nc",
        "/home/rjorge/gkx-nl24-discriminator-20260912.vvmgDD/nl24.big.nc",
    ),
    "Nl32": (
        "/home/rjorge/gx-nyquist-resolution-20260905.Ut2U6L/full96.out.nc",
        "/home/rjorge/gx-nyquist-resolution-20260905.Ut2U6L/full96.big.nc",
    ),
    "Nl32_Nz192": (
        "/home/rjorge/gx-nyquist-resolution-20260905.Ut2U6L/full192.out.nc",
        None,
    ),
}
phis = {}
for name, (out, big) in runs.items():
    d = nc.Dataset(out)
    t = np.asarray(d.groups["Grids"]["time"][:])
    theta = np.asarray(d.groups["Grids"]["theta"][:])
    diag = d.groups["Diagnostics"]
    W = np.asarray(diag["Wg_lmst"][:])[:, 0]  # (t, m, l)
    phi2 = np.asarray(diag["Phi2_t"][:])
    phi2z = np.asarray(diag["Phi2_zt"][:])
    Nm, Nl = W.shape[1], W.shape[2]
    print(
        f"\n=== {name}: Nt={len(t)} T={t[-1]:.1f} Nm={Nm} Nl={Nl} Ntheta={len(theta)} theta in [{theta.min():.2f},{theta.max():.2f}]"
    )
    # gamma(t) from Phi2: 0.5 dlnPhi2/dt
    lnp = np.log(phi2)
    for a, b in ((50, 100), (100, 200), (200, 300), (150, 300), (210, 300)):
        m = (t >= a) & (t <= b)
        if m.sum() > 5:
            g = 0.5 * np.polyfit(t[m], lnp[m], 1)[0]
            print(f"  gamma fit [{a},{b}] = {g:.6f}")
    gi = 0.5 * np.gradient(lnp, t)
    print(
        "  instantaneous gamma at t=100/150/200/250/300:",
        " ".join(
            f"{gi[np.argmin(np.abs(t - tt))]:.5f}" for tt in (100, 150, 200, 250, 300)
        ),
    )
    # Laguerre and Hermite spectra normalized at selected times
    Wl = W.sum(axis=1)  # (t, l)
    Wm = W.sum(axis=2)  # (t, m)
    for tt in (50, 100, 200, 300):
        i = np.argmin(np.abs(t - tt))
        wl = Wl[i] / Wl[i].sum()
        wm = Wm[i] / Wm[i].sum()
        q = Nl // 4
        print(
            f"  t={t[i]:5.1f}: W(l) upper-quarter frac={wl[-q:].sum():.4f}  W(l)[-4:]={np.array2string(wl[-4:], precision=4)}"
            f"  argmax_l(l>=Nl/2)={Nl // 2 + int(np.argmax(wl[Nl // 2 :]))}  W(m) upper-quarter frac={wm[-Nm // 4 :].sum():.4f}"
        )
    i = -1
    wl = Wl[i] / Wl[i].sum()
    print(
        "  final W(l)/sum, all l:", np.array2string(wl, precision=2, max_line_width=250)
    )
    top_low_m = W[i, :3, Nl - 1].sum() / W[i].sum()
    corner = W[i, Nm - 1, Nl - 1] / W[i, 0, 0]
    print(
        f"  final P(l=Nl-1, m<=2)/total = {top_low_m:.3e}; corner/00 = {corner:.3e}; P(l=Nl-1,all m)/total = {W[i, :, Nl - 1].sum() / W[i].sum():.3e}"
    )
    # hump growth in time: upper-quarter fraction time series
    q = Nl // 4
    frac = Wl[:, -q:].sum(axis=1) / Wl.sum(axis=1)
    print(
        "  upper-quarter Laguerre fraction at t=20/50/100/150/200/250/300:",
        " ".join(
            f"{frac[np.argmin(np.abs(t - tt))]:.4f}"
            for tt in (20, 50, 100, 150, 200, 250, 300)
        ),
    )
    # z structure
    pz = phi2z[-1] / phi2z[-1].max()
    print(
        "  final |phi|^2(theta)/max at theta=0,±pi,±2pi,±3pi:",
        " ".join(
            f"{pz[np.argmin(np.abs(theta - th))]:.2e}"
            for th in (0, np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 3 * np.pi, -3 * np.pi)
        ),
    )
    # W(l) split by |theta| is not available (Wg_zst is summed over l,m); report Wg_zst/max
    wz = np.asarray(diag["Wg_zst"][-1, 0])
    wz = wz / wz.max()
    print(
        "  final Wg(theta)/max at same thetas:",
        " ".join(
            f"{wz[np.argmin(np.abs(theta - th))]:.2e}"
            for th in (0, np.pi, -np.pi, 2 * np.pi, -2 * np.pi, 3 * np.pi, -3 * np.pi)
        ),
    )
    if big:
        b = nc.Dataset(big)
        P = np.asarray(b.groups["Diagnostics"]["Phi"][:])  # (t, ky, kx, theta, ri)
        tb = np.asarray(b.groups["Grids"]["time"][:])
        ph = P[-1, :, 0, :, 0] + 1j * P[-1, :, 0, :, 1]  # (ky, theta)
        kyi = int(np.argmax(np.abs(ph).sum(axis=1)))
        phis[name] = (tb[-1], ph[kyi] / np.linalg.norm(ph[kyi]), theta)
        print(
            f"  big.nc: {P.shape[0]} snapshots, last t={tb[-1]:.1f}, active ky index {kyi}"
        )
if "Nl24" in phis and "Nl32" in phis:
    (t1, p1, th1), (t2, p2, th2) = phis["Nl24"], phis["Nl32"]
    ov = abs(np.vdot(p1, p2))
    print(
        f"\nOverlap |<phi24|phi32>| at t={t1:.0f}/{t2:.0f}: {ov:.6f}   (parity check: |<phi(theta)|phi(-theta)>| Nl24={abs(np.vdot(p1, p1[::-1])):.4f} Nl32={abs(np.vdot(p2, p2[::-1])):.4f})"
    )
