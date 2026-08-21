"""Campaign: calibrate a geometric transport predictor across devices and shapes.

Design follows from what the single-coefficient RBC(1,1) study could and could
not establish. There, geometric features predicted the replicated nonlinear flux
well (connection length: Spearman +0.76, leave-one-out R^2 0.53, 8% median
error, no gyrokinetic solve) with every sign matching the critical-gradient
mechanism. But on a ONE-parameter scan all geometric quantities are functions of
the same coordinate -- corr(log L_c, log S_rms) = -0.94, condition number 267 --
so individual exponents are not identifiable and no multi-feature law can be
calibrated.

Two axes break that degeneracy:

1. **Multiple boundary coefficients** per device. Perturbing RBC(0,1), RBC(1,1),
   ZBS(0,1) and ZBS(1,1) moves connection length, local shear, curvature and
   flux expansion along different directions.
2. **Multiple devices** (QA, QH, QI, and tokamaks). Measured across the VMEX
   portfolio these features already span a genuine 2D region rather than a line:
   L_c ~ 9 occurs with S_rms = 220 (Nuhrenberg-Zille QHS) and with S_rms = 2.8
   (li383).

Together they give the conditioning a multi-feature fit needs, and holding out
whole devices tests whether a calibrated law transfers rather than memorizes.

Nonlinear fidelity. The tracked RBC(1,1) ground truth used n64:64:64:40:40 grids
with t = 1100-1500 windows, which is hours per point and not affordable across a
portfolio. This campaign uses a reduced but *internally consistent* setup, and
overlaps deliberately with the high-fidelity landscape so the reduction can be
checked rather than assumed. Correlation studies need consistency across
configurations, not converged absolute flux -- but that is a claim about ranking
only, and the overlap points are what license it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

# Boundary coefficients perturbed per device. Chosen to move the geometric
# features along different directions: (0,1) is dominated by elongation/shaping,
# (1,1) by triangularity-like deformation, and the ZBS pair breaks up-down
# symmetry of the response.
_COEFFICIENTS = ("RBC(0,1)", "RBC(1,1)", "ZBS(0,1)", "ZBS(1,1)")

#: Relative perturbations applied to each coefficient.
_FRACTIONS = (-0.4, -0.2, 0.0, 0.2, 0.4)

#: Device portfolio spanning QA / QH / QI and axisymmetric references.
_DEVICES = (
    "nfp2_QI",
    "nfp1_QI",
    "nfp4_QH_warm_start",
    "NuhrenbergZille_1988_QHS",
    "LandremanPaul2021_QA_lowres",
    "LandremanPaul2021_QH_reactorScale_lowres",
    "li383_low_res",
    "circular_tokamak",
    "DSHAPE",
    "ITERModel",
)

# A near-zero mean gradpar means the flux-tube mapping degenerated rather than
# the device having an enormous connection length; nfp2_QA returns 2.2e10 and
# must not enter a fit as a legitimate extreme.
_MIN_GRADPAR = 1.0e-6
_MAX_CONNECTION_LENGTH = 1.0e4


#: Number token as VMEC decks write it: 1.0, -4.1E-03, 7.7e-01, 3.510
_NUMBER = r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?"


def coefficient_pattern(name: str) -> re.Pattern[str]:
    """Match one boundary coefficient regardless of deck formatting.

    The shipped decks use at least three layouts -- ``RBC(   0,   0) = 1.0e+00,``
    with padded indices, ``RBC( 0,1) =  1.000`` with a single pad, and
    ``RBC(1,0) =   1.3782E+00`` unpadded -- and every one of them puts RBC and
    ZBS on the SAME line. So the index whitespace has to be flexible and the
    value cannot be anchored to end-of-line.
    """

    match = re.fullmatch(r"([A-Z]+)\((\d+),(\d+)\)", name)
    if match is None:
        raise ValueError(f"unrecognized coefficient name: {name!r}")
    prefix, first, second = match.groups()
    return re.compile(
        rf"({re.escape(prefix)}\(\s*{first}\s*,\s*{second}\s*\)\s*=\s*)({_NUMBER})"
    )


def perturb(text: str, coefficient: str, fraction: float) -> tuple[str, float] | None:
    """Scale one boundary coefficient by ``1 + fraction``.

    Returns ``None`` when the deck does not carry that coefficient, so a device
    is skipped for that axis rather than silently perturbed elsewhere.
    """

    pattern = coefficient_pattern(coefficient)
    match = pattern.search(text)
    if match is None:
        return None
    baseline = float(match.group(2).replace("D", "e").replace("d", "e"))
    if baseline == 0.0:
        return None
    value = baseline * (1.0 + fraction)
    return pattern.sub(lambda m: f"{m.group(1)}{value:.16E}", text, count=1), value


def sanity_check(features: dict[str, float]) -> str | None:
    """Reject degenerate mappings instead of letting them anchor a fit."""

    if not np.isfinite(list(features.values())).all():
        return "non-finite feature"
    if features["connection_length"] > _MAX_CONNECTION_LENGTH:
        return f"connection length {features['connection_length']:.3g} implies gradpar ~ 0"
    if features["local_shear_rms"] == 0.0:
        return "local shear identically zero"
    return None


def build_plan(devices: tuple[str, ...], data_dir: Path) -> list[dict]:
    """Enumerate the (device, coefficient, fraction) grid actually realizable."""

    plan = []
    for device in devices:
        deck = data_dir / f"input.{device}"
        if not deck.is_file():
            print(f"  skip {device}: no input deck")
            continue
        text = deck.read_text()
        for coefficient in _COEFFICIENTS:
            if coefficient_pattern(coefficient).search(text) is None:
                continue
            for fraction in _FRACTIONS:
                plan.append(
                    {
                        "device": device,
                        "coefficient": coefficient,
                        "fraction": fraction,
                        "deck": str(deck),
                    }
                )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--devices", type=str, default=",".join(_DEVICES))
    parser.add_argument("--s-index", type=int, default=7)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/geometric_transport_campaign_features.json"),
    )
    args = parser.parse_args()

    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    from tools.artifacts.build_geometric_saturation_predictor import geometric_features

    data_dir = Path(vj.__file__).resolve().parents[1] / "examples/data"
    devices = tuple(d.strip() for d in args.devices.split(",") if d.strip())
    plan = build_plan(devices, data_dir)
    print(f"planned configurations: {len(plan)}", flush=True)

    args.workdir.mkdir(parents=True, exist_ok=True)
    records, rejected = [], []
    for index, item in enumerate(plan):
        label = (
            f"{item['device']}__{item['coefficient'].replace('(','').replace(')','').replace(',','')}"
            f"__f{item['fraction']:+.2f}".replace(".", "p")
        )
        text = Path(item["deck"]).read_text()
        perturbed = perturb(text, item["coefficient"], item["fraction"])
        if perturbed is None:
            continue
        deck_text, value = perturbed
        deck_path = args.workdir / f"input.{label}"
        deck_path.write_text(deck_text)

        try:
            equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(deck_path))
            mapping = turb.gk_fieldline_geometry(
                equilibrium.state,
                equilibrium.runtime,
                s_index=args.s_index,
                alpha=0.0,
                ntheta=args.ntheta,
            )
            features = geometric_features(mapping)
        except Exception as err:
            rejected.append({**item, "label": label, "reason": f"{type(err).__name__}: {err}"[:160]})
            print(f"  [{index + 1}/{len(plan)}] {label}: FAILED", flush=True)
            continue

        problem = sanity_check(features)
        if problem:
            rejected.append({**item, "label": label, "reason": problem})
            print(f"  [{index + 1}/{len(plan)}] {label}: rejected -- {problem}", flush=True)
            continue

        records.append(
            {
                **item,
                "label": label,
                "coefficient_value": value,
                "input_path": str(deck_path),
                "features": features,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  [{index + 1}/{len(plan)}] accepted {len(records)}", flush=True)

    summary = {
        "kind": "geometric_transport_campaign_features",
        "claim_level": "geometry_only_features_pending_nonlinear_ground_truth",
        "devices": list(devices),
        "coefficients": list(_COEFFICIENTS),
        "fractions": list(_FRACTIONS),
        "n_accepted": len(records),
        "n_rejected": len(rejected),
        "records": records,
        "rejected": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\naccepted {len(records)}, rejected {len(rejected)}")
    if records:
        matrix = np.column_stack(
            [
                np.log(np.abs([r["features"][n] for r in records]) + 1e-30)
                for n in ("connection_length", "local_shear_rms",
                          "bad_curvature_mean", "flux_expansion_mean")
            ]
            + [np.ones(len(records))]
        )
        print(f"design condition number: {np.linalg.cond(matrix):.1f}"
              "   (RBC(1,1)-only scan was 267)")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
