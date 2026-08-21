"""A geometry-only predictor of nonlinear ITG transport, calibrated on matched data.

Why geometry rather than growth rates. On the tracked RBC(1,1) landscape every
quasilinear proxy is anticorrelated with the nonlinear flux (Spearman -0.707 for
the best of them; see ``build_quasilinear_skill_audit.py``). Those proxies are
built from the linear growth rate at a *fixed* gradient, which measures drive and
says nothing about how far the configuration sits above its own instability
threshold, nor about how strongly zonal flows will saturate it.

The critical-gradient literature identifies what actually sets the ITG threshold,
and all of it is geometry: field-line curvature, parallel connection length,
local magnetic shear, and flux-surface expansion --

* Roberg-Clark, Plunk & Xanthopoulos, *Phys. Rev. Research* **5**, L032030
  (2023), arXiv:2301.06773;
* Roberg-Clark, Plunk & Xanthopoulos, arXiv:2210.16030 (quasi-helical);
* Roberg-Clark et al., arXiv:2506.22166 (quasi-isodynamic);
* Xanthopoulos et al., *Phys. Rev. Lett.* **125**, 265001 (2020), zonal-flow
  control of stellarator ITG transport.

Directly measuring a critical gradient in GKX turned out to be impractical here:
the growth rate is still shifting 33% between (N_l, N_m) = (8, 10) and (12, 16)
at ~96 s per evaluation, and it stays positive down to zero temperature gradient,
so there is no clean zero crossing to bisect. The geometric quantities that
*control* the threshold are free by comparison -- one flux-tube mapping, no
gyrokinetic solve -- and they are differentiable, so they can drive an optimizer.

This script extracts those quantities along the field line for every point on the
landscape and measures how well each predicts the replicated nonlinear heat flux.
Nothing is fitted before the correlations are reported, and a leave-one-out score
is included so an apparent multi-feature win cannot come from overfitting 24
points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_LANDSCAPE = Path("docs/_static/vmec_boundary_transport_landscape_rbc11_full.json")


def landscape_points() -> list[dict]:
    data = json.loads(_LANDSCAPE.read_text())
    rows = {round(r["coefficient_value"], 10): r for r in data["rows"]}
    points = []
    for entry in data["nonlinear_ensemble_points"]:
        if not entry.get("passed"):
            continue
        row = rows.get(round(entry["coefficient_value"], 10))
        if row is None:
            continue
        points.append(
            {
                "fraction": row["relative_fraction"],
                "coefficient": entry["coefficient_value"],
                "flux": entry["mean"],
                "flux_sem": entry["sem"],
            }
        )
    points.sort(key=lambda p: p["fraction"])
    return points


def geometric_features(geometry: dict) -> dict[str, float]:
    """Field-line quantities the critical-gradient literature identifies.

    ``geometry`` is the mapping returned by ``vmex.core.turbulence`` with the
    standard flux-tube fields on a common theta grid.
    """

    theta = np.asarray(geometry["theta"], dtype=float)
    bmag = np.asarray(geometry["bmag"], dtype=float)
    gradpar = float(np.mean(np.abs(np.asarray(geometry["gradpar"], dtype=float))))
    gds2 = np.asarray(geometry["gds2"], dtype=float)
    gds21 = np.asarray(geometry["gds21"], dtype=float)
    gds22 = np.asarray(geometry["gds22"], dtype=float)
    cvdrift = np.asarray(geometry["cvdrift"], dtype=float)
    gbdrift = np.asarray(geometry["gbdrift"], dtype=float)

    # Weight by time spent on the field line: dl/B, the standard bounce measure.
    weight = 1.0 / np.maximum(bmag, 1e-30)
    weight = weight / weight.sum()

    # "Bad" curvature is the destabilizing sign; averaging the signed drift
    # cancels good against bad and hides the drive.
    bad = np.clip(cvdrift, 0.0, None)

    # Local magnetic shear along the tube. gds21/gds22 is the standard
    # integrated-local-shear combination.
    local_shear = gds21 / np.maximum(gds22, 1e-30)

    return {
        # Drive geometry: how much bad curvature, and how strong.
        "bad_curvature_mean": float(np.sum(bad * weight)),
        "bad_curvature_max": float(bad.max()),
        "bad_curvature_fraction": float(np.sum((cvdrift > 0) * weight)),
        # Connection length: 1/gradpar sets the parallel transit, and longer
        # connection means more time in bad curvature.
        "connection_length": float(1.0 / max(gradpar, 1e-30)),
        # Local shear: stabilizes toroidal ITG by limiting the ballooning
        # envelope's extent along the tube.
        "local_shear_rms": float(np.sqrt(np.sum(local_shear**2 * weight))),
        "local_shear_range": float(local_shear.max() - local_shear.min()),
        # Flux expansion: gds2 is |grad alpha|^2, which sets k_perp for a given
        # binormal mode number -- larger means the same mode has finer real-space
        # scale and weaker transport.
        "flux_expansion_mean": float(np.sum(gds2 * weight)),
        "flux_expansion_min": float(gds2.min()),
        # Field strength variation, which sets trapping.
        "bmag_variation": float(bmag.max() / max(bmag.min(), 1e-30)),
        # Combined drive-over-stabilization group. This is the shape the
        # critical-gradient papers argue for: bad curvature drives, local shear
        # and flux expansion resist.
        "curvature_over_shear": float(
            np.sum(bad * weight) / max(np.sqrt(np.sum(local_shear**2 * weight)), 1e-30)
        ),
        "curvature_over_expansion": float(
            np.sum(bad * weight) / max(np.sum(gds2 * weight), 1e-30)
        ),
        "gbdrift_bad_mean": float(np.sum(np.clip(gbdrift, 0.0, None) * weight)),
        "theta_extent": float(theta.max() - theta.min()),
    }


def leave_one_out_r2(feature: np.ndarray, flux: np.ndarray) -> float:
    """Leave-one-out R^2 for a single-feature log-linear fit.

    Reported alongside the raw correlation because 24 points is few enough that
    an in-sample fit can look good for the wrong reason.
    """

    logq = np.log(flux)
    predictions = np.empty_like(logq)
    for index in range(feature.size):
        mask = np.ones(feature.size, dtype=bool)
        mask[index] = False
        slope, intercept = np.polyfit(feature[mask], logq[mask], 1)
        predictions[index] = slope * feature[index] + intercept
    residual = np.sum((logq - predictions) ** 2)
    total = np.sum((logq - logq.mean()) ** 2)
    return float(1.0 - residual / max(total, 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/geometric_saturation_predictor.json"),
    )
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    from tools.artifacts.build_zonal_flow_saturation_model import write_input

    args.workdir.mkdir(parents=True, exist_ok=True)
    points = landscape_points()
    print(f"landscape points: {len(points)}", flush=True)

    for point in points:
        label = f"f{point['fraction']:+.2f}".replace(".", "p")
        deck = write_input(
            args.template, point["coefficient"], args.workdir / f"input.{label}"
        )
        equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(deck))
        mapping = turb.gk_fieldline_geometry(
            equilibrium.state,
            equilibrium.runtime,
            s_index=args.s_index,
            alpha=0.0,
            ntheta=args.ntheta,
        )
        point["features"] = geometric_features(mapping)
        print(f"  f={point['fraction']:+.2f}  Q_nl={point['flux']:.2f}", flush=True)

    flux = np.array([p["flux"] for p in points])
    names = list(points[0]["features"])
    scores = {}
    for name in names:
        values = np.array([p["features"][name] for p in points])
        if np.allclose(values, values[0]):
            continue
        scores[name] = {
            "pearson": float(pearsonr(values, flux)[0]),
            "spearman": float(spearmanr(values, flux)[0]),
            "loo_r2": leave_one_out_r2(values, flux),
        }

    print(f"\n{'geometric feature':<28}{'Pearson':>9}{'Spearman':>10}{'LOO R^2':>10}")
    for name, s in sorted(scores.items(), key=lambda kv: -abs(kv[1]["spearman"])):
        print(f"{name:<28}{s['pearson']:>9.3f}{s['spearman']:>10.3f}{s['loo_r2']:>10.3f}")

    # Collinearity diagnostic. On a ONE-parameter boundary scan every geometric
    # quantity is a function of the same scan coordinate, so features are
    # collinear by construction: here corr(log L_c, log S_rms) = -0.94 with a
    # condition number near 270. Single-feature signs are therefore
    # interpretable and multi-feature EXPONENTS are not -- a multivariate fit
    # will happily return a shear exponent whose sign contradicts the
    # single-feature correlation. Separating the terms needs a scan over more
    # than one boundary coefficient.
    matrix = np.column_stack(
        [np.log(np.abs(np.array([p["features"][n] for p in points])) + 1e-30) for n in scores]
        + [np.ones(len(points))]
    )
    collinearity = {
        "condition_number": float(np.linalg.cond(matrix)),
        "note": (
            "one-parameter scan: features are collinear by construction; "
            "single-feature signs are interpretable, multivariate exponents are not"
        ),
    }

    summary = {"points": points, "scores": scores, "collinearity": collinearity}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
