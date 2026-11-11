"""Saturation detector for nonlinear heat-flux traces, in physical time.

The trace is resampled onto a uniform grid of physical time, so the verdict does
not depend on the step size, the diagnostic cadence or the mass ratio. After a
minimum run time ``t_min`` it cuts the start-up transient with MSER, estimates
the integrated autocorrelation time with Sokal's automatic window, takes the
standard error from batch means and tests the window for drift. Each species
and the total must pass. See ``docs/numerics.rst`` for the algorithm and
references.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np


@dataclass(frozen=True)
class SaturationStopConfig:
    """Thresholds of the saturation detector (times in a/v_t)."""

    t_min: float = 800.0  # no "saturated" verdict before this run time
    rel_sem: float = 0.10  # batch-means SEM over the mean
    drift_rel: float = 0.10  # tolerated change of the mean across the window
    drift_z: float = 2.0  # significance of the OLS trend of the block means
    min_span: float = 250.0  # shortest retained window
    k_min: int = 10  # independent blocks needed to quote an error


def _finite(v: float) -> float | None:
    """JSON-safe value: ``None`` for an unresolved (infinite) error."""
    return float(v) if math.isfinite(v) else None


def _uniform_means(t: np.ndarray, q: np.ndarray, h: float) -> np.ndarray:
    """Bin means of the piecewise-linear signal on a grid of width ``h``."""
    c = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(t))])
    e = t[0] + h * np.arange(int((t[-1] - t[0]) // h) + 1)
    return np.diff(np.interp(e, t, c)) / h


def _tau(x: np.ndarray, h: float, c: float = 5.0) -> float:
    """Integrated autocorrelation time, Sokal window ``M >= c tau``."""
    x = x - x.mean()
    n = x.size
    if n < 8 or not x.any():
        return h
    f = np.fft.rfft(x, 2 * n)
    rho = np.fft.irfft(f * np.conj(f))[:n]
    tau = 0.5 + np.cumsum(rho[1:] / rho[0])
    k = np.nonzero(np.arange(1, n) >= c * tau)[0]
    return h * max(float(tau[k[0]] if k.size else tau[-1]), 0.5)


def _mser(x: np.ndarray, max_frac: float = 0.35, n_batch: int = 100) -> int:
    """MSER truncation point (bins) from at most ``n_batch`` block means."""
    g = max(x.size // n_batch, 1)
    m = x.size // g
    b = x[: m * g].reshape(m, g).mean(1)
    cost = [
        np.sum((b[d:] - b[d:].mean()) ** 2) / (m - d) ** 2
        for d in range(int(m * max_frac) + 1)
    ]
    return int(np.argmin(cost)) * g


def _channel(
    w: np.ndarray, h: float, scale_all: float, cfg: SaturationStopConfig
) -> dict[str, Any]:
    """Mean, batch-means SEM and drift of one channel over the retained window."""
    tau = _tau(w, h)
    blk = max(math.ceil(4 * tau / h), math.ceil(w.size / 40))  # blocks >= 4 tau
    k = w.size // blk
    m = float(w.mean())
    scale = max(abs(m), 0.2 * scale_all)  # safe when species cancel
    r: dict[str, Any] = dict(
        mean=m, tau=tau, K=k, scale=scale, sem=math.inf, drift_z=math.inf
    )
    if k >= cfg.k_min:
        b = w[w.size - k * blk :].reshape(k, blk).mean(1)
        i = np.arange(k) - (k - 1) / 2
        slope = (i @ b) / (i @ i)
        resid = np.sum((b - b.mean() - slope * i) ** 2) / (k - 2)
        r.update(
            sem=float(b.std(ddof=1) / math.sqrt(k)),
            drift_rel=abs(slope) * (k - 1) / scale,
            drift_z=abs(slope) / math.sqrt(max(resid, 1e-300) / (i @ i)),
        )
    else:  # correlation unresolved: outer thirds, flagged only at twice the tolerance
        a, _, c = (p.mean() for p in np.array_split(w, 3))
        r.update(drift_rel=abs(c - a) / scale / 2)
    r["rel_sem"] = r["sem"] / scale
    return r


def assess_saturation(
    t: Any,
    q: Any,
    cfg: SaturationStopConfig | None = None,
    *,
    t_min: float | None = None,
    _confirm: bool = True,
) -> dict[str, Any]:
    """Classify a heat-flux trace ``q(t)``, shape ``(n,)`` or ``(n, species)``.

    ``status`` is one of ``too_short``, ``growing``, ``quiescent``, ``drifting``,
    ``intermittent``, ``unconfirmed`` or ``saturated``. Species columns are
    gated individually together with their sum.
    """
    cfg = cfg or SaturationStopConfig()
    t = np.asarray(t, float)
    q = np.asarray(q, float)
    q = q[:, None] if q.ndim == 1 else q
    out: dict[str, Any] = dict(status="too_short", saturated=False, t_end=float(t[-1]))
    if t.size < 2:
        return out
    t_min = cfg.t_min if t_min is None else t_min
    span_all = float(t[-1] - t[0])
    h = min(
        max(float(np.median(np.diff(t))), span_all / 2048, 1e-9),
        max(0.5, span_all / 2048),
    )
    chans = list(q.T) + ([q.sum(1)] if q.shape[1] > 1 else [])
    xs = [_uniform_means(t, c, h) for c in chans]
    n = xs[0].size
    if span_all < t_min or n < 64:
        return out
    tot = np.abs(xs[-1])
    qm = [tot[i * (n // 4) : (i + 1) * (n // 4)].mean() for i in range(4)]
    if qm[3] > 1.5 * qm[2] and qm[2] > 1.2 * qm[1] and qm[1] > qm[0] > 0:
        return {**out, "status": "growing"}
    if qm[3] <= 1e-2 * tot.max():  # decayed (or never left zero): linearly stable
        return {**out, "status": "quiescent"}
    lead = int(np.argmax([abs(x.mean()) for x in xs[: q.shape[1]]]))
    ref = (
        xs[-1]
        if abs(xs[-1].mean()) > 0.2 * sum(abs(x.mean()) for x in xs[: q.shape[1]])
        else xs[lead]
    )
    s0 = _mser(ref)
    span = (n - s0) * h
    out.update(t_start=float(t[0] + s0 * h), span=span)
    scale_all = sum(abs(x[s0:].mean()) for x in xs[: q.shape[1]])
    res = [_channel(x[s0:], h, scale_all, cfg) for x in xs]
    tot_r = res[-1]
    out.update(
        window_tmin=out["t_start"],
        window_tmax=out["t_end"],
        mean=tot_r["mean"],
        sem=_finite(tot_r["sem"]),
        rel_sem=_finite(tot_r["rel_sem"]),
        tau_ac=tot_r["tau"],
    )
    gated = [
        r
        for r in res
        if r is tot_r or abs(r["mean"]) > 2 * r["sem"] or r["scale"] > 0.05 * scale_all
    ]
    if span < cfg.min_span:
        return out
    if any(
        r["drift_rel"] > cfg.drift_rel and r["drift_z"] > cfg.drift_z for r in gated
    ):
        return {**out, "status": "drifting"}
    if (
        span < 30 * max(r["tau"] for r in res)
        or any(r["K"] < cfg.k_min for r in res)
        or any(r["rel_sem"] > cfg.rel_sem for r in gated)
    ):  # correlation too long to resolve the mean within the window
        return {**out, "status": "intermittent"}
    if _confirm:  # the verdict must already have held 20% of the window earlier
        j = np.searchsorted(t, t[0] + int(n - 0.2 * (n - s0)) * h)
        prev = assess_saturation(t[:j], q[:j], cfg, t_min=0.0, _confirm=False)
        if prev["status"] != "saturated" or abs(
            prev["mean"] - out["mean"]
        ) > 2 * math.hypot(prev["sem"], out["sem"]):
            return {**out, "status": "unconfirmed"}
    return {**out, "status": "saturated", "saturated": True}


def make_saturation_check(
    cfg: SaturationStopConfig, t_cap: float
) -> Callable[[np.ndarray, np.ndarray, bool], dict[str, Any]]:
    """Stop check for the chunk loop, looking every ``max(25, 5%)`` of the elapsed time.

    At the cap (``final``) an unsaturated verdict is reported as ``cap_reached:<status>``.
    """
    state: dict[str, Any] = dict(next=0.0, last=None)

    def check(t: np.ndarray, q: np.ndarray, final: bool) -> dict[str, Any]:
        if not final and state["last"] and t[-1] < state["next"]:
            return state["last"]
        d = assess_saturation(t, q, cfg, t_min=min(cfg.t_min, 0.95 * t_cap))
        state["next"] = max(
            t[-1] + max(25.0, 0.05 * t[-1]), t[0] + min(cfg.t_min, 0.95 * t_cap)
        )
        if final and not d["saturated"]:
            d["status"] = f"cap_reached:{d['status']}"
        d["stop"] = d["saturated"]
        state["last"] = d
        return d

    return check


__all__ = ["SaturationStopConfig", "assess_saturation", "make_saturation_check"]
