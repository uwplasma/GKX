#!/usr/bin/env python3
"""Fast local release-readiness checks for CI, packaging, and docs wiring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable

# The repo-hygiene CI job runs the release gates straight from a checkout with
# no install, so reach the repository's single TOML shim through ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gkx.utils import tomlcompat as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_CI_SNIPPETS = (
    "wide-coverage-shards",
    "coverage-wide-shard-manifest.json",
    "--require-shard-data",
    "--coverage-xml coverage-wide.xml",
    "--enforce-package-coverage",
    "codecov/codecov-action",
    "tools/release/check_parallel_scaling_artifacts.py",
    "tools/release/check_package_architecture_manifest.py",
    "tools/release/check_parallel_scaling_artifacts.py --performance-manifest-only",
    "tools/release/check_quasilinear_promotion_guardrails.py",
    "tools/release/check_vmec_boozer_gates.py differentiability-claim",
    "tools/release/check_release_readiness.py technical-status",
    "tools/release/check_release_readiness.py",
    "rm -rf build dist",
)
REQUIRED_CODECOV_SNIPPETS = (
    "after_n_builds: 2",
    "wait_for_ci: true",
    "flags:",
    "- wide-package",
)
REQUIRED_RELEASE_SNIPPETS = (
    "name: Release",
    "gh-action-pypi-publish",
    "tools/release/check_release_readiness.py version",
    "tools/release/check_repository_size_manifest.py",
    "tools/release/check_repository_size_manifest.py release-artifacts",
    "tools/release/check_package_architecture_manifest.py",
    "tools/release/check_parallel_scaling_artifacts.py --performance-manifest-only",
    "tools/release/check_parallel_scaling_artifacts.py",
    "tools/release/check_quasilinear_promotion_guardrails.py",
    "tools/release/check_vmec_boozer_gates.py differentiability-claim",
    "tools/release/check_release_readiness.py technical-status",
    "tools/release/check_release_readiness.py",
    "rm -rf build dist",
)
REQUIRED_README_SNIPPETS = (
    "pip install gkx",
    "gkx",
    "MIT",
)
REQUIRED_STATIC_ARTIFACTS = (
    "benchmarks/references/gkx_1_7_release_contract.json",
    "docs/_static/runtime_memory_benchmark.png",
    "docs/_static/runtime_memory_summary_ship_refresh.json",
    "docs/_static/runtime_memory_results_ship_refresh.csv",
    "docs/_static/validation_gate_index.json",
    "docs/_static/validation_coverage_manifest_summary.json",
    "docs/_static/quasilinear_promotion_guardrails.json",
    "docs/_static/vmec_boozer_differentiability_claim_guard.json",
    "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
    "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
    "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
    "docs/_static/technical_release_status.json",
    "docs/_static/w7x_tem_extension_status.json",
    "docs/_static/independent_ky_scan_scaling_large.json",
    "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
    "docs/_static/parallelization_completion_status.json",
    "docs/_static/nonlinear_sharding_strong_scaling_large.json",
    "docs/_static/nonlinear_domain_parallel_identity_gate.json",
    "docs/_static/nonlinear_spectral_communication_identity_gate.json",
    "docs/_static/vmec_boundary_transport_landscape_admission.json",
    "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
    "docs/_static/nonlinear_campaign_admission_report.json",
    "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
)
TECHNICAL_COMPLETION_TARGET = 0.98
TECHNICAL_STATUS_ARTIFACT = "docs/_static/technical_release_status.json"
RELEASE_CONTRACT_ARTIFACT = "benchmarks/references/gkx_1_7_release_contract.json"
OPTIMIZATION_STATUS_ARTIFACT = RELEASE_CONTRACT_ARTIFACT
REQUIRED_OPTIMIZATION_STATUS_FLAGS = {
    "qa_baseline_gate_passed": True,
    "quasilinear_model_selection_passed": False,
    "simple_quasilinear_absolute_flux_promoted": False,
    "long_window_nonlinear_audit_passed": True,
    "nonlinear_prelaunch_policy_ready": True,
    "nonlinear_campaign_admission_ready": True,
    "negative_reference_blocks_weak_margin": True,
}
REQUIRED_OPTIMIZATION_CLAIM_EVIDENCE_LEVEL = "scoped_matched_replicated_nonlinear_audit"
REQUIRED_OPTIMIZATION_CLAIM_BLOCKERS = (
    "quasilinear_model_selection_not_promoted",
    "simple_quasilinear_absolute_flux_not_promoted",
)
REQUIRED_PRELAUNCH_GATE_ROWS = (
    {
        "label": "replicated landscape admission",
        "path": "docs/_static/vmec_boundary_transport_landscape_admission.json",
        "expected_raw_passed": True,
        "required_blocker": None,
        "min_sample_count": 12.0,
    },
    {
        "label": "selected reduced prelaunch",
        "path": "docs/_static/vmec_boundary_transport_prelaunch_gate.json",
        "expected_raw_passed": True,
        "required_blocker": None,
        "min_sample_count": 18.0,
    },
    {
        "label": "weak reduced-margin reference",
        "path": "docs/_static/strict_qa_top12_edge_prelaunch_gate.json",
        "expected_raw_passed": False,
        "required_blocker": "insufficient_reduced_margin_for_nonlinear_audit",
        "min_sample_count": 18.0,
    },
    {
        "label": "next nonlinear campaign admission",
        "path": "docs/_static/nonlinear_campaign_admission_report.json",
        "expected_raw_passed": True,
        "required_blocker": None,
        "min_sample_count": 18.0,
    },
)


class ReleaseReadinessError(RuntimeError):
    """Raised when a release-readiness contract is not satisfied."""


SOURCE_VERSION = REPO_ROOT / "src" / "gkx" / "_version.py"
PYPROJECT = REPO_ROOT / "pyproject.toml"
VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$")


class ReleaseVersionError(ValueError):
    """Raised when release metadata is internally inconsistent."""


@dataclass(frozen=True)
class EvidenceCheck:
    """One checkable release-readiness item."""

    name: str
    path: str
    snippet: str | None = None
    required: bool = True


def _read_evidence(root: Path, path: str) -> str | None:
    target = root / path
    if not target.exists():
        return None
    if target.suffix.lower() in {".png", ".pdf", ".nc"}:
        return "<binary>"
    return target.read_text(encoding="utf-8", errors="replace")


def _evaluate_check(root: Path, check: EvidenceCheck) -> dict[str, Any]:
    text = _read_evidence(root, check.path)
    exists = text is not None
    snippet_ok = True if check.snippet is None else bool(text and check.snippet in text)
    passed = bool(exists and snippet_ok)
    return {
        "name": check.name,
        "path": check.path,
        "required": check.required,
        "exists": exists,
        "snippet": check.snippet,
        "passed": passed,
    }


LANES: dict[str, tuple[EvidenceCheck, ...]] = {
    "ci_coverage_release": (
        EvidenceCheck(
            "wide coverage matrix", ".github/workflows/ci.yml", "wide-coverage-shards"
        ),
        EvidenceCheck(
            "wide shard manifest",
            ".github/workflows/ci.yml",
            "coverage-wide-shard-manifest.json",
        ),
        EvidenceCheck("95 percent gate", ".github/workflows/ci.yml", "--fail-under 95"),
        EvidenceCheck(
            "measured manifest coverage audit",
            ".github/workflows/ci.yml",
            "--coverage-xml coverage-wide.xml",
        ),
        EvidenceCheck(
            "codecov upload", ".github/workflows/ci.yml", "codecov/codecov-action"
        ),
        EvidenceCheck(
            "release readiness check",
            ".github/workflows/ci.yml",
            "tools/release/check_release_readiness.py",
        ),
    ),
    "parallelization_release_surface": (
        EvidenceCheck(
            "parallelization policy docs",
            "docs/parallelization.rst",
            "Production-ready parallelism",
        ),
        EvidenceCheck(
            "runtime parallel input docs", "docs/inputs.rst", 'strategy = "batch"'
        ),
        EvidenceCheck(
            "independent ky scaling artifact",
            "docs/_static/independent_ky_scan_scaling_large.json",
            "not a nonlinear domain-decomposition",
        ),
        EvidenceCheck(
            "quasilinear UQ scaling artifact",
            "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
            "not a promoted absolute nonlinear heat-flux predictor",
        ),
        EvidenceCheck(
            "parallelization completion status",
            "docs/_static/parallelization_completion_status.json",
            "production_completion_percent",
        ),
        EvidenceCheck(
            "nonlinear sharding scoped diagnostic",
            "docs/_static/nonlinear_sharding_strong_scaling_large.json",
            "not a production speedup claim",
        ),
        EvidenceCheck(
            "nonlinear domain identity gate",
            "docs/_static/nonlinear_domain_parallel_identity_gate.json",
            "no production routing or speedup claim",
        ),
        EvidenceCheck(
            "nonlinear spectral communication gate",
            "docs/_static/nonlinear_spectral_communication_identity_gate.json",
            "no production routing or speedup claim",
        ),
        EvidenceCheck(
            "parallel artifact checker",
            "tools/release/check_parallel_scaling_artifacts.py",
            "FAMILIES",
        ),
    ),
    "refactor_modularity": (
        EvidenceCheck(
            "package architecture manifest",
            "tools/package_architecture_manifest.toml",
            "allowed_root_prefix_modules",
        ),
        EvidenceCheck(
            "package architecture checker",
            "tools/release/check_package_architecture_manifest.py",
            "root-level prefix modules",
        ),
        EvidenceCheck(
            "operators package facade",
            "src/gkx/operators/__init__.py",
            "hermite_streaming",
        ),
        EvidenceCheck(
            "linear operator package",
            "src/gkx/operators/linear/__init__.py",
            "build_linear_cache",
        ),
        EvidenceCheck(
            "linear solver package",
            "src/gkx/solvers/linear/__init__.py",
            "KrylovConfig",
        ),
        EvidenceCheck(
            "nonlinear operator package",
            "src/gkx/operators/nonlinear/__init__.py",
            "nonlinear_rhs_cached_impl",
        ),
        EvidenceCheck(
            "nonlinear solver package",
            "src/gkx/solvers/nonlinear/__init__.py",
            "solve_imex_step",
        ),
        EvidenceCheck(
            "runtime scan orchestration module",
            "src/gkx/workflows/runtime/orchestration_scan.py",
            "RuntimeScanDeps",
        ),
        EvidenceCheck(
            "runtime policy module",
            "src/gkx/workflows/runtime/policies.py",
            "RuntimeIndependentParallelPlan",
        ),
        EvidenceCheck(
            "linear cache builder", "src/gkx/operators/linear/cache_builder.py"
        ),
        EvidenceCheck(
            "linear moments module", "src/gkx/operators/linear/moments.py"
        ),
        EvidenceCheck(
            "linear params module", "src/gkx/operators/linear/params.py"
        ),
        EvidenceCheck(
            "linear parallel module", "src/gkx/solvers/linear/parallel.py"
        ),
        EvidenceCheck(
            "nonlinear helper module", "src/gkx/operators/nonlinear/policies.py"
        ),
        EvidenceCheck(
            "benchmark policy module",
            "src/gkx/benchmarking/shared.py",
            "CYCLONE_KRYLOV_DEFAULT",
        ),
        EvidenceCheck(
            "diagnostic moment kernels", "src/gkx/diagnostics/moments.py"
        ),
        EvidenceCheck(
            "coverage manifest",
            "tools/validation_coverage_manifest.toml",
            "gkx.workflows.runtime.orchestration_scan",
        ),
    ),
    "docs_release_hygiene": (
        EvidenceCheck("readme install", "README.md", "pip install gkx"),
        EvidenceCheck("readme executable", "README.md", "gkx"),
        EvidenceCheck("MIT license in README", "README.md", "MIT"),
        EvidenceCheck("release scope ledger", "docs/release_scope.rst", "Claim scope"),
        EvidenceCheck("examples docs", "docs/examples.rst", "parallelization"),
        EvidenceCheck(
            "release workflow",
            ".github/workflows/release.yml",
            "gh-action-pypi-publish",
        ),
    ),
    "performance_artifacts": (
        EvidenceCheck(
            "runtime memory figure", "docs/_static/runtime_memory_benchmark.png"
        ),
        EvidenceCheck("runtime memory manifest", "tools/runtime_memory_manifest.toml"),
        EvidenceCheck(
            "performance manifest",
            "tools/performance_optimization_manifest.toml",
            "parallel_scaling",
        ),
        EvidenceCheck(
            "nonlinear RHS profile", "docs/_static/nonlinear_rhs_profile.json"
        ),
        EvidenceCheck(
            "full nonlinear RHS trace",
            "docs/_static/full_nonlinear_rhs_trace_summary.json",
        ),
        EvidenceCheck("performance docs", "docs/performance.rst", "No speedup claim"),
    ),
    "scientific_gate_guardrails": (
        EvidenceCheck(
            "validation gate index", "docs/_static/validation_gate_index.json"
        ),
        EvidenceCheck(
            "quasilinear guardrails",
            "docs/_static/quasilinear_promotion_guardrails.json",
        ),
        EvidenceCheck(
            "VMEC/Boozer differentiability claim guard",
            "docs/_static/vmec_boozer_differentiability_claim_guard.json",
            "not_full_nonlinear_transport_optimization",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer frequency gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer quasilinear gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck(
            "finite-beta VMEC/Boozer reduced nonlinear-window gradient gate",
            "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
            "shaped_tokamak_pressure",
        ),
        EvidenceCheck("frozen 1.7 release contract", RELEASE_CONTRACT_ARTIFACT),
        EvidenceCheck(
            "stellarator optimization docs",
            "docs/stellarator_optimization.rst",
            "finite-difference",
        ),
        EvidenceCheck("quasilinear docs", "docs/quasilinear.rst", "absolute-flux"),
    ),
}


def build_technical_release_status(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return technical release completion status from tracked evidence."""

    root = root.resolve()
    lanes: dict[str, Any] = {}
    scores: list[float] = []
    failed_required: list[str] = []
    for lane, checks in LANES.items():
        evaluated = [_evaluate_check(root, check) for check in checks]
        required = [item for item in evaluated if item["required"]]
        passed_required = [item for item in required if item["passed"]]
        score = 100.0 * len(passed_required) / max(len(required), 1)
        scores.append(score)
        lane_failed = [item["name"] for item in required if not item["passed"]]
        failed_required.extend(f"{lane}: {name}" for name in lane_failed)
        lanes[lane] = {
            "completion_percent": score,
            "passed_required": len(passed_required),
            "required": len(required),
            "failed_required": lane_failed,
            "checks": evaluated,
        }
    overall = sum(scores) / max(len(scores), 1)
    return {
        "kind": "gkx_technical_release_status",
        "root": str(root),
        "technical_release_completion_percent": overall,
        "target_percent": 98.0,
        "passed": overall >= 98.0 and not failed_required,
        "failed_required": failed_required,
        "lanes": lanes,
        "scope": (
            "Technical release readiness for CI, refactor, docs/release hygiene, "
            "parallelization artifacts, performance artifacts, and guardrails. "
            "Deferred manuscript physics lanes remain scoped separately."
        ),
    }


def read_project_version(root: Path = REPO_ROOT) -> str:
    """Return the PEP 621 project version from ``pyproject.toml``."""

    path = root / "pyproject.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        version = payload["project"]["version"]
    except KeyError as exc:
        raise ReleaseVersionError("pyproject.toml is missing project.version") from exc
    if not isinstance(version, str) or not version.strip():
        raise ReleaseVersionError(
            "pyproject.toml project.version must be a non-empty string"
        )
    return version.strip()


def read_source_version(root: Path = REPO_ROOT) -> str:
    """Return ``gkx.__version__`` without importing the package."""

    path = root / "src" / "gkx" / "_version.py"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VERSION_RE.match(line.strip())
        if match:
            return match.group(1)
    raise ReleaseVersionError(f"{path.relative_to(root)} does not define __version__")


def normalize_tag(tag: str | None) -> str | None:
    """Normalize GitHub tag strings such as ``refs/tags/v1.2.3``."""

    if tag is None:
        return None
    tag = tag.strip()
    if not tag:
        return None
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    return tag


def default_tag_from_github_env() -> str | None:
    """Return the GitHub ref name only for tag-triggered workflows."""

    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return None
    return os.environ.get("GITHUB_REF_NAME")


def fetch_pypi_versions(package: str) -> set[str]:
    """Return released versions for ``package`` from the public PyPI JSON API.

    PyPI answers 404 for a project that has never published a release, which is
    the expected state for a first publish. Treat that as an empty release set
    so the duplicate-version check stays meaningful instead of failing the
    debut release.
    """

    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310 - fixed PyPI HTTPS endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise ReleaseVersionError(f"PyPI response for {package!r} is missing releases")
    return {str(version) for version in releases}


def validate_release_version(
    *,
    root: Path = REPO_ROOT,
    tag: str | None = None,
    require_tag: bool = False,
    package: str = "gkx",
    pypi_versions: Iterable[str] | None = None,
) -> dict[str, object]:
    """Validate package version, source version, optional tag, and PyPI uniqueness."""

    root = root.resolve()
    project_version = read_project_version(root)
    source_version = read_source_version(root)
    if source_version != project_version:
        raise ReleaseVersionError(
            f"src/gkx/_version.py has {source_version!r}, "
            f"but pyproject.toml has {project_version!r}"
        )

    normalized_tag = normalize_tag(tag)
    if require_tag and normalized_tag is None:
        raise ReleaseVersionError("release publishing requires a tag like v1.2.3")
    if normalized_tag is not None:
        expected = f"v{project_version}"
        if normalized_tag != expected:
            raise ReleaseVersionError(
                f"release tag {normalized_tag!r} does not match project version {project_version!r}; "
                f"expected {expected!r}"
            )

    duplicate_on_pypi = False
    if pypi_versions is not None:
        duplicate_on_pypi = project_version in {
            str(version) for version in pypi_versions
        }
        if duplicate_on_pypi:
            raise ReleaseVersionError(
                f"{package} {project_version} already exists on PyPI; bump the version before publishing"
            )

    return {
        "package": package,
        "project_version": project_version,
        "source_version": source_version,
        "tag": normalized_tag,
        "require_tag": require_tag,
        "checked_pypi": pypi_versions is not None,
        "duplicate_on_pypi": duplicate_on_pypi,
    }


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReleaseReadinessError(f"missing required file: {path}") from exc


def _missing_snippets(text: str, snippets: tuple[str, ...]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def _project_metadata(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    project = pyproject.get("project", {})
    scripts = project.get("scripts", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "scripts": sorted(scripts),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise ReleaseReadinessError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseReadinessError(f"invalid JSON in required file: {path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseReadinessError(
            f"required JSON file must contain an object: {path}"
        )
    return payload


def _status_counts(lanes: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        status = lane.get("status")
        if isinstance(status, str):
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _is_deferred_lane(lane: dict[str, Any]) -> bool:
    status = str(lane.get("status", "")).strip().lower()
    claim_level = str(lane.get("claim_level", "")).strip().lower()
    return status == "deferred" or claim_level.startswith("deferred")


def _recomputed_active_summary(lanes: list[Any]) -> dict[str, Any]:
    active = [
        lane for lane in lanes if isinstance(lane, dict) and not _is_deferred_lane(lane)
    ]
    closed = [
        lane
        for lane in active
        if str(lane.get("status", "")).strip().lower() == "closed"
    ]
    incomplete = [
        {
            "lane": lane.get("lane"),
            "status": lane.get("status"),
            "claim_level": lane.get("claim_level"),
        }
        for lane in active
        if str(lane.get("status", "")).strip().lower() != "closed"
    ]
    return {
        "active_fraction_closed": len(closed) / max(len(active), 1),
        "n_active": len(active),
        "n_closed": len(closed),
        "n_incomplete": len(incomplete),
        "incomplete_lanes": incomplete,
    }


def _technical_release_status(root: Path) -> dict[str, Any]:
    payload = _read_json(root / TECHNICAL_STATUS_ARTIFACT)
    completion_percent = payload.get("technical_release_completion_percent")
    if not isinstance(completion_percent, (int, float)):
        raise ReleaseReadinessError(
            f"{TECHNICAL_STATUS_ARTIFACT} missing numeric "
            "technical_release_completion_percent"
        )
    failed_required = payload.get("failed_required", [])
    if not isinstance(failed_required, list):
        raise ReleaseReadinessError(
            f"{TECHNICAL_STATUS_ARTIFACT} failed_required must be a list"
        )
    target_percent = 100.0 * TECHNICAL_COMPLETION_TARGET
    passed = (
        bool(payload.get("passed"))
        and float(completion_percent) >= target_percent
        and not failed_required
    )
    return {
        "source": TECHNICAL_STATUS_ARTIFACT,
        "completion_percent": float(completion_percent),
        "target_percent": target_percent,
        "failed_required": failed_required,
        "passed": passed,
    }


def _prelaunch_gate_failures(prelaunch_gates: list[Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    rows_by_label: dict[str, dict[str, Any]] = {}
    duplicate_labels: set[str] = set()
    for row in prelaunch_gates:
        if not isinstance(row, dict):
            failures.append(
                {"label": None, "reason": "prelaunch row must be an object"}
            )
            continue
        label = row.get("label")
        if not isinstance(label, str) or not label:
            failures.append({"label": label, "reason": "prelaunch row missing label"})
            continue
        if label in rows_by_label:
            duplicate_labels.add(label)
            continue
        rows_by_label[label] = row

    for label in sorted(duplicate_labels):
        failures.append({"label": label, "reason": "duplicate prelaunch gate label"})

    for contract in REQUIRED_PRELAUNCH_GATE_ROWS:
        label = str(contract["label"])
        row = rows_by_label.get(label)
        if row is None:
            failures.append(
                {"label": label, "reason": "missing required prelaunch gate"}
            )
            continue
        if row.get("passed") is not True:
            failures.append(
                {
                    "label": label,
                    "reason": "normalized prelaunch row must pass",
                    "passed": row.get("passed"),
                }
            )
        if row.get("path") != contract["path"]:
            failures.append(
                {
                    "label": label,
                    "reason": "prelaunch gate path mismatch",
                    "expected": contract["path"],
                    "observed": row.get("path"),
                }
            )
        expected_raw = bool(contract["expected_raw_passed"])
        if row.get("expected_raw_passed") is not expected_raw:
            failures.append(
                {
                    "label": label,
                    "reason": "expected_raw_passed mismatch",
                    "expected": expected_raw,
                    "observed": row.get("expected_raw_passed"),
                }
            )
        if row.get("raw_passed") is not expected_raw:
            failures.append(
                {
                    "label": label,
                    "reason": "raw_passed mismatch",
                    "expected": expected_raw,
                    "observed": row.get("raw_passed"),
                }
            )
        blockers = row.get("blockers")
        if not isinstance(blockers, list) or not all(
            isinstance(item, str) for item in blockers
        ):
            failures.append(
                {
                    "label": label,
                    "reason": "blockers must be a list of strings",
                    "blockers": blockers,
                }
            )
            blockers = []
        required_blocker = contract["required_blocker"]
        if required_blocker is None and blockers:
            failures.append(
                {
                    "label": label,
                    "reason": "passing positive gate must not carry blockers",
                    "blockers": blockers,
                }
            )
        if required_blocker is not None and required_blocker not in blockers:
            failures.append(
                {
                    "label": label,
                    "reason": "negative reference missing required blocker",
                    "required_blocker": required_blocker,
                    "blockers": blockers,
                }
            )
        sample_count = row.get("sample_count")
        if not isinstance(sample_count, (int, float)) or float(sample_count) < float(
            contract["min_sample_count"]
        ):
            failures.append(
                {
                    "label": label,
                    "reason": "prelaunch sample count below contract",
                    "expected_min": contract["min_sample_count"],
                    "observed": sample_count,
                }
            )
    return failures


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_fingerprint_number(value: int | float) -> int | float | str:
    if isinstance(value, int):
        return value
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _numeric_array_payload(value: object) -> object | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _normalized_fingerprint_number(value)
    if not isinstance(value, list):
        return None
    rows = [_numeric_array_payload(item) for item in value]
    if any(row is None for row in rows):
        return None
    return rows


def _numeric_json_fingerprints(payload: object) -> dict[str, object]:
    scalars: dict[str, int | float | str] = {}
    arrays: dict[str, object] = {}

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            numeric = _numeric_array_payload(value)
            if numeric is not None:
                arrays[path or "$root"] = numeric
                return
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        scalars[path or "$root"] = _normalized_fingerprint_number(value)

    walk(payload, "")
    return {
        "scalar_count": len(scalars),
        "scalar_sha256": _canonical_json_sha256(scalars),
        "array_count": len(arrays),
        "array_sha256": _canonical_json_sha256(arrays),
    }


def build_frozen_output_fingerprint(
    root: Path, relative_path: str
) -> dict[str, object]:
    path = root / relative_path
    if not path.is_file():
        raise ReleaseReadinessError(f"frozen output missing: {relative_path}")
    report: dict[str, object] = {
        "path": relative_path,
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if path.suffix == ".json":
        report.update(_numeric_json_fingerprints(_read_json(path)))
    return report


def _frozen_output_fingerprint_summary(
    root: Path, release_contract: dict[str, Any]
) -> dict[str, object]:
    frozen = release_contract.get("frozen_output_fingerprints")
    baseline = release_contract.get("baseline")
    requires_fingerprints = isinstance(baseline, dict) and bool(baseline.get("git_tag"))
    if not isinstance(frozen, dict):
        if requires_fingerprints:
            raise ReleaseReadinessError(
                f"{RELEASE_CONTRACT_ARTIFACT} missing frozen_output_fingerprints"
            )
        return {"count": 0, "passed": True, "rows": []}
    rows = frozen.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} has no frozen output fingerprint entries"
        )
    checked: list[dict[str, object]] = []
    failures: list[str] = []
    for index, expected in enumerate(rows):
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            failures.append(f"entry {index}: invalid path")
            continue
        observed = build_frozen_output_fingerprint(root, str(expected["path"]))
        fields = ["content_sha256"]
        if str(expected["path"]).endswith(".json"):
            # Textual provenance can change during a rename; the frozen contract
            # protects numerical values and array layout rather than JSON formatting.
            fields = ["scalar_count", "scalar_sha256", "array_count", "array_sha256"]
        mismatches = [
            field for field in fields if observed.get(field) != expected.get(field)
        ]
        if mismatches:
            failures.append(f"{expected['path']}: {', '.join(mismatches)}")
        checked.append(
            {
                "label": expected.get("label"),
                **observed,
                "passed": not mismatches,
            }
        )
    if failures:
        raise ReleaseReadinessError(
            "frozen numerical output fingerprints changed: " + "; ".join(failures)
        )
    return {"count": len(checked), "passed": True, "rows": checked}


def _lane_status_summary(root: Path) -> dict[str, Any]:
    payload = _read_json(root / RELEASE_CONTRACT_ARTIFACT)
    if payload.get("kind") != "gkx_1_7_frozen_release_contract":
        raise ReleaseReadinessError(f"{RELEASE_CONTRACT_ARTIFACT} has an invalid kind")
    lanes = payload.get("release_lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} missing release_lanes"
        )
    public_api = payload.get("public_api")
    if not isinstance(public_api, dict) or not isinstance(
        public_api.get("exports"), list
    ):
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} missing public_api.exports"
        )
    exports = public_api["exports"]
    export_names = [row.get("name") for row in exports if isinstance(row, dict)]
    if public_api.get("count") != len(exports) or len(set(export_names)) != len(
        exports
    ):
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} public API count/names are inconsistent"
        )
    performance = payload.get("performance")
    if not isinstance(performance, dict) or not isinstance(
        performance.get("rows"), list
    ):
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} missing performance.rows"
        )
    performance_rows = performance["rows"]
    if performance.get("row_count") != len(performance_rows):
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} performance row count is inconsistent"
        )
    refresh = performance.get("representative_refresh")
    if not isinstance(refresh, dict) or not isinstance(refresh.get("path"), str):
        raise ReleaseReadinessError(
            f"{RELEASE_CONTRACT_ARTIFACT} missing performance.representative_refresh"
        )
    refresh_path = str(refresh["path"])
    refresh_payload = _read_json(root / refresh_path)
    if refresh_payload.get("kind") != "gkx_representative_performance_refresh":
        raise ReleaseReadinessError(f"{refresh_path} has an invalid kind")
    workloads = refresh_payload.get("workloads")
    summary = refresh_payload.get("summary")
    if (
        not isinstance(workloads, list)
        or len(workloads) < 2
        or not isinstance(summary, dict)
    ):
        raise ReleaseReadinessError(
            f"{refresh_path} must contain at least two workloads and a summary"
        )
    cpu_rows = [row.get("cpu") for row in workloads if isinstance(row, dict)]
    gpu_rows = [row.get("gpu") for row in workloads if isinstance(row, dict)]
    cpu_admitted = sum(
        isinstance(row, dict) and row.get("admitted") is True for row in cpu_rows
    )
    gpu_admitted = sum(
        isinstance(row, dict) and row.get("admitted") is True for row in gpu_rows
    )
    gpu_blocked = sum(
        isinstance(row, dict)
        and row.get("admitted") is False
        and bool(row.get("blocker"))
        for row in gpu_rows
    )
    observed_refresh = {
        "correctness_passed": all(
            isinstance(row, dict)
            and isinstance(row.get("correctness"), dict)
            and row["correctness"].get("cpu_finite") is True
            and row["correctness"].get("gpu_finite") is True
            for row in workloads
        ),
        "cpu_rows_admitted": cpu_admitted,
        "gpu_rows_admitted": gpu_admitted,
        "gpu_rows_blocked": gpu_blocked,
        "performance_claim_updated": bool(summary.get("performance_claim_updated")),
    }
    for field, value in observed_refresh.items():
        if summary.get(field) != value or refresh.get(field) != value:
            raise ReleaseReadinessError(
                f"{refresh_path} has inconsistent representative-refresh field {field}"
            )
    if cpu_admitted < 2 or not observed_refresh["correctness_passed"]:
        raise ReleaseReadinessError(
            f"{refresh_path} does not admit the bounded CPU correctness/performance subset"
        )
    if observed_refresh["performance_claim_updated"] and gpu_admitted == 0:
        raise ReleaseReadinessError(
            f"{refresh_path} cannot update a performance claim without an admitted GPU row"
        )
    fingerprint_summary = _frozen_output_fingerprint_summary(root, payload)
    recomputed = _recomputed_active_summary(lanes)
    active_fraction_closed = float(recomputed["active_fraction_closed"])
    release_scoped_incomplete = int(recomputed["n_incomplete"])
    target_passed = (
        active_fraction_closed >= TECHNICAL_COMPLETION_TARGET
        and release_scoped_incomplete == 0
    )
    return {
        "target_fraction": TECHNICAL_COMPLETION_TARGET,
        "source": f"{RELEASE_CONTRACT_ARTIFACT}:release_lanes",
        "active_fraction_closed": active_fraction_closed,
        "release_scoped_open_or_blocked": release_scoped_incomplete,
        "release_scoped_incomplete": release_scoped_incomplete,
        "recomputed_active_summary": recomputed,
        "passed": target_passed,
        "status_artifacts": {
            RELEASE_CONTRACT_ARTIFACT: {
                "kind": payload.get("kind"),
                "status_counts": _status_counts(lanes),
                "public_api_count": len(exports),
                "performance_row_count": len(performance_rows),
                "representative_performance_refresh": observed_refresh,
                "frozen_output_fingerprint_count": fingerprint_summary["count"],
                "lanes": lanes,
            }
        },
        "frozen_outputs": fingerprint_summary,
    }


def _optimization_status_summary(root: Path) -> dict[str, Any]:
    payload = _read_json(root / OPTIMIZATION_STATUS_ARTIFACT)
    policy = payload.get("optimization_policy")
    if not isinstance(policy, dict):
        raise ReleaseReadinessError(
            f"{OPTIMIZATION_STATUS_ARTIFACT} missing optimization_policy object"
        )
    summary = policy.get("summary")
    if not isinstance(summary, dict):
        raise ReleaseReadinessError(
            f"{OPTIMIZATION_STATUS_ARTIFACT} optimization_policy missing summary object"
        )
    prelaunch_gates = policy.get("prelaunch_gates")
    if not isinstance(prelaunch_gates, list):
        raise ReleaseReadinessError(
            f"{OPTIMIZATION_STATUS_ARTIFACT} optimization_policy missing prelaunch_gates list"
        )

    failed_flags = []
    for key, expected in REQUIRED_OPTIMIZATION_STATUS_FLAGS.items():
        observed = summary.get(key)
        if not isinstance(observed, bool) or observed is not expected:
            failed_flags.append(
                {
                    "key": key,
                    "expected": expected,
                    "observed": observed,
                }
            )
    claim_evidence_level = summary.get("claim_evidence_level")
    if claim_evidence_level != REQUIRED_OPTIMIZATION_CLAIM_EVIDENCE_LEVEL:
        failed_flags.append(
            {
                "key": "claim_evidence_level",
                "expected": REQUIRED_OPTIMIZATION_CLAIM_EVIDENCE_LEVEL,
                "observed": claim_evidence_level,
            }
        )
    claim_blockers = summary.get("claim_promotion_blockers")
    if not isinstance(claim_blockers, list):
        failed_flags.append(
            {
                "key": "claim_promotion_blockers",
                "expected": list(REQUIRED_OPTIMIZATION_CLAIM_BLOCKERS),
                "observed": claim_blockers,
            }
        )
    else:
        missing_claim_blockers = [
            blocker
            for blocker in REQUIRED_OPTIMIZATION_CLAIM_BLOCKERS
            if blocker not in claim_blockers
        ]
        if missing_claim_blockers:
            failed_flags.append(
                {
                    "key": "claim_promotion_blockers",
                    "expected": list(REQUIRED_OPTIMIZATION_CLAIM_BLOCKERS),
                    "observed": claim_blockers,
                    "missing": missing_claim_blockers,
                }
            )
    failed_prelaunch_rows = [
        {
            "label": row.get("label"),
            "passed": row.get("passed"),
            "raw_passed": row.get("raw_passed"),
            "blockers": row.get("blockers"),
        }
        for row in prelaunch_gates
        if isinstance(row, dict) and not bool(row.get("passed", False))
    ]
    failed_prelaunch_contracts = _prelaunch_gate_failures(prelaunch_gates)
    passed = (
        not failed_flags
        and not failed_prelaunch_rows
        and not failed_prelaunch_contracts
        and len(prelaunch_gates) >= len(REQUIRED_PRELAUNCH_GATE_ROWS)
    )
    return {
        "source": OPTIMIZATION_STATUS_ARTIFACT,
        "kind": payload.get("kind"),
        "passed": passed,
        "required_flags": REQUIRED_OPTIMIZATION_STATUS_FLAGS,
        "required_prelaunch_gate_rows": list(REQUIRED_PRELAUNCH_GATE_ROWS),
        "failed_flags": failed_flags,
        "failed_prelaunch_rows": failed_prelaunch_rows,
        "failed_prelaunch_contracts": failed_prelaunch_contracts,
        "prelaunch_gate_count": len(prelaunch_gates),
        "summary": {
            key: summary.get(key)
            for key in sorted(
                set(REQUIRED_OPTIMIZATION_STATUS_FLAGS)
                | {
                    "direct_scalar_transport_blocked",
                    "claim_evidence_level",
                    "claim_promotion_blockers",
                    "projected_transport_improved",
                    "positive_prelaunch_gate_passed",
                    "landscape_admission_passed",
                }
            )
        },
    }


def check_release_readiness(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready release-readiness report or raise on failure."""

    root = root.resolve()
    failures: list[str] = []

    version_report = validate_release_version(root=root)
    project = _project_metadata(root)
    if project["name"] != "gkx":
        failures.append("pyproject project.name must be 'gkx'")
    expected_scripts = {"gkx", "gkx"}
    missing_scripts = sorted(expected_scripts - set(project["scripts"]))
    if missing_scripts:
        failures.append(f"missing executable entry points: {missing_scripts}")

    ci_text = _read(root / ".github" / "workflows" / "ci.yml")
    missing_ci = _missing_snippets(ci_text, REQUIRED_CI_SNIPPETS)
    if missing_ci:
        failures.append(f"ci.yml missing release checks: {missing_ci}")

    codecov_text = _read(root / "codecov.yml")
    missing_codecov = _missing_snippets(codecov_text, REQUIRED_CODECOV_SNIPPETS)
    if missing_codecov:
        failures.append(
            f"codecov.yml missing wide-coverage status policy: {missing_codecov}"
        )

    release_text = _read(root / ".github" / "workflows" / "release.yml")
    missing_release = _missing_snippets(release_text, REQUIRED_RELEASE_SNIPPETS)
    if missing_release:
        failures.append(
            f"release.yml missing publish/version checks: {missing_release}"
        )

    readme_text = _read(root / "README.md")
    missing_readme = _missing_snippets(readme_text, REQUIRED_README_SNIPPETS)
    if missing_readme:
        failures.append(f"README missing release-user snippets: {missing_readme}")

    missing_artifacts = [
        path for path in REQUIRED_STATIC_ARTIFACTS if not (root / path).exists()
    ]
    if missing_artifacts:
        failures.append(
            f"missing required docs/static release artifacts: {missing_artifacts}"
        )

    try:
        technical_status = _technical_release_status(root)
        if not technical_status["passed"]:
            failures.append(
                "technical release status below target: "
                f"{technical_status['completion_percent']:.3f} < "
                f"{technical_status['target_percent']:.3f} or failed checks "
                f"= {technical_status['failed_required']}"
            )
    except ReleaseReadinessError as exc:
        technical_status = {
            "source": TECHNICAL_STATUS_ARTIFACT,
            "completion_percent": 0.0,
            "target_percent": 100.0 * TECHNICAL_COMPLETION_TARGET,
            "failed_required": [],
            "passed": False,
            "error": str(exc),
        }
        failures.append(str(exc))

    try:
        lane_status = _lane_status_summary(root)
        if not lane_status["passed"]:
            failures.append(
                "release-scoped technical completion below target: "
                f"{lane_status['active_fraction_closed']:.3f} < "
                f"{lane_status['target_fraction']:.3f} or incomplete release lanes "
                f"= {lane_status['release_scoped_incomplete']}"
            )
    except ReleaseReadinessError as exc:
        lane_status = {
            "target_fraction": TECHNICAL_COMPLETION_TARGET,
            "source": f"{RELEASE_CONTRACT_ARTIFACT}:release_lanes",
            "active_fraction_closed": 0.0,
            "release_scoped_open_or_blocked": None,
            "release_scoped_incomplete": None,
            "passed": False,
            "status_artifacts": {},
            "error": str(exc),
        }
        failures.append(str(exc))

    try:
        optimization_status = _optimization_status_summary(root)
        if not optimization_status["passed"]:
            failures.append(
                "optimization status prelaunch/claim-boundary flags failed: "
                f"flags={optimization_status['failed_flags']} "
                f"prelaunch_rows={optimization_status['failed_prelaunch_rows']} "
                f"prelaunch_contracts={optimization_status['failed_prelaunch_contracts']}"
            )
    except ReleaseReadinessError as exc:
        optimization_status = {
            "source": OPTIMIZATION_STATUS_ARTIFACT,
            "passed": False,
            "error": str(exc),
        }
        failures.append(str(exc))

    report = {
        "kind": "gkx_release_readiness",
        "root": str(root),
        "project": project,
        "version": version_report,
        "release_target": {
            "technical_completion_fraction": TECHNICAL_COMPLETION_TARGET,
            "status_source": lane_status["source"],
        },
        "technical_status": technical_status,
        "lane_status": lane_status,
        "optimization_status": optimization_status,
        "required_ci_snippets": list(REQUIRED_CI_SNIPPETS),
        "required_codecov_snippets": list(REQUIRED_CODECOV_SNIPPETS),
        "required_release_snippets": list(REQUIRED_RELEASE_SNIPPETS),
        "required_readme_snippets": list(REQUIRED_README_SNIPPETS),
        "required_static_artifacts": list(REQUIRED_STATIC_ARTIFACTS),
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise ReleaseReadinessError("; ".join(failures))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser


def build_version_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate release-version consistency before publishing artifacts."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--tag",
        default=default_tag_from_github_env(),
        help="Release tag to validate.",
    )
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="Fail unless --tag is a v-prefixed release tag.",
    )
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="Fail if this version already exists on PyPI.",
    )
    parser.add_argument(
        "--package",
        default="gkx",
        help="PyPI package name for duplicate checks.",
    )
    return parser


def main_version(argv: list[str] | None = None) -> int:
    args = build_version_parser().parse_args(argv)
    pypi_versions = fetch_pypi_versions(args.package) if args.check_pypi else None
    try:
        report = validate_release_version(
            root=args.root,
            tag=args.tag,
            require_tag=bool(args.require_tag),
            package=str(args.package),
            pypi_versions=pypi_versions,
        )
    except ReleaseVersionError as exc:
        raise SystemExit(f"release-version check failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_technical_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the tracked technical-release completion status artifact."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--out-json", type=Path, default=REPO_ROOT / TECHNICAL_STATUS_ARTIFACT
    )
    parser.add_argument("--fail-under", type=float, default=98.0)
    return parser


def main_technical_status(argv: list[str] | None = None) -> int:
    args = build_technical_status_parser().parse_args(argv)
    report = build_technical_release_status(args.root)
    report["target_percent"] = float(args.fail_under)
    report["passed"] = (
        float(report["technical_release_completion_percent"]) >= float(args.fail_under)
        and not report["failed_required"]
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json is None:
        print(payload, end="")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    if not report["passed"]:
        print(payload, end="")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "version":
        return main_version(tokens[1:])
    if tokens and tokens[0] == "technical-status":
        return main_technical_status(tokens[1:])

    args = build_parser().parse_args(tokens)
    report = check_release_readiness(root=args.root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out_json is None:
        print(payload, end="")
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
