#!/usr/bin/env python3
"""Validate package architecture and produce a migration inventory.

The checker prevents new root-level prefix modules from appearing while the
codebase migrates toward domain packages. Existing prefix modules are allowed
only when they are explicitly listed as temporary migration scaffolding.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

# The repo-hygiene CI job runs the release gates straight from a checkout with
# no install, so reach the repository's single TOML shim through ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gkx.utils import tomlcompat as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tools" / "package_architecture_manifest.toml"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "src" / "gkx"


def _as_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _as_nonempty_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return _as_string_list(value, field)


def _as_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings")
        result.append(item.strip())
    duplicates = sorted({item for item in result if result.count(item) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate entries: {duplicates}")
    return result


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _as_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _repo_path(raw: str) -> Path:
    return (REPO_ROOT / raw).resolve()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load an architecture TOML manifest."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse as a TOML table")
    return data


def _root_module_path(module: str, source_root: Path) -> Path:
    if not module.startswith("gkx."):
        raise ValueError(f"root-prefix module must start with gkx.: {module}")
    remainder = module.removeprefix("gkx.")
    if "." in remainder:
        raise ValueError(
            f"root-prefix allowlist entries must be root modules: {module}"
        )
    return source_root / f"{remainder}.py"


def _package_path(package: str, source_root: Path) -> Path:
    if not package.startswith("gkx."):
        raise ValueError(f"required package must start with gkx.: {package}")
    remainder = package.removeprefix("gkx.")
    return source_root.joinpath(*remainder.split(".")) / "__init__.py"


def _count_matching_files(path: Path, *, pattern: str, recursive: bool) -> int:
    """Count files for topology gates, treating deleted targets as zero."""

    if not path.exists():
        return 0
    iterator = path.rglob(pattern) if recursive else path.glob(pattern)
    return sum(1 for item in iterator if item.is_file())


def _source_line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _validate_complexity_policy(
    data: dict[str, Any],
    *,
    source_root: Path,
    require_targets: bool = False,
) -> list[dict[str, Any]]:
    policy = data.get("complexity_policy")
    if policy is None:
        return []
    if not isinstance(policy, dict):
        raise ValueError("complexity_policy must be a TOML table")
    mode = _as_nonempty_string(policy.get("mode"), "complexity_policy.mode")
    if mode != "no_regression_until_target":
        raise ValueError("complexity_policy.mode must be 'no_regression_until_target'")
    _as_nonempty_string(policy.get("description"), "complexity_policy.description")
    default_limit = _as_nonnegative_int(
        policy.get("default_max_lines"), "complexity_policy.default_max_lines"
    )
    facade_limit = _as_nonnegative_int(
        policy.get("public_facade_max_lines"),
        "complexity_policy.public_facade_max_lines",
    )
    public_facades = set(
        _as_string_list(
            policy.get("public_facades"), "complexity_policy.public_facades"
        )
    )
    raw_exceptions = policy.get("exceptions")
    if not isinstance(raw_exceptions, list):
        raise ValueError("complexity_policy.exceptions must be a list")

    exceptions: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw_exceptions):
        if not isinstance(entry, dict):
            raise ValueError(f"complexity_policy.exceptions[{index}] must be a table")
        prefix = f"complexity_policy.exceptions[{index}]"
        path = _as_nonempty_string(entry.get("path"), f"{prefix}.path")
        if path in exceptions:
            raise ValueError(
                f"complexity_policy.exceptions contains duplicate path: {path}"
            )
        baseline = _as_nonnegative_int(
            entry.get("baseline_lines"), f"{prefix}.baseline_lines"
        )
        target = _as_nonnegative_int(
            entry.get("target_lines"), f"{prefix}.target_lines"
        )
        if target > baseline:
            raise ValueError(
                f"{path}: complexity target {target} cannot exceed baseline {baseline}"
            )
        exceptions[path] = {
            "baseline": baseline,
            "target": target,
            "reason": _as_nonempty_string(entry.get("reason"), f"{prefix}.reason"),
        }

    source_files = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    existing = {path.relative_to(source_root).as_posix(): path for path in source_files}
    stale = sorted(set(exceptions) - set(existing))
    if stale:
        raise ValueError(f"complexity exceptions reference missing modules: {stale}")

    rows: list[dict[str, Any]] = []
    unowned: list[str] = []
    for relative, path in existing.items():
        lines = _source_line_count(path)
        limit = facade_limit if relative in public_facades else default_limit
        exception = exceptions.get(relative)
        if lines > limit and exception is None:
            unowned.append(f"{relative} ({lines}>{limit})")
            continue
        if exception is None:
            continue
        baseline = int(exception["baseline"])
        target = int(exception["target"])
        if lines > baseline:
            raise ValueError(
                f"{relative}: complexity regressed to {lines} lines, above baseline {baseline}"
            )
        if require_targets and lines > target:
            raise ValueError(
                f"{relative}: complexity target not met; {lines} lines exceeds {target}"
            )
        rows.append(
            {
                "path": relative,
                "lines": lines,
                "limit": limit,
                "baseline": baseline,
                "target": target,
                "remaining_to_target": max(0, lines - target),
                "target_met": lines <= target,
                "reason": str(exception["reason"]),
            }
        )
    if unowned:
        raise ValueError(
            "source modules exceed complexity budgets without reviewed exceptions: "
            + ", ".join(unowned)
        )
    return rows


def _validate_topology_policy(
    data: dict[str, Any],
    *,
    require_targets: bool = False,
) -> list[dict[str, Any]]:
    topology = data.get("topology_policy")
    if topology is None:
        return []
    if not isinstance(topology, dict):
        raise ValueError("topology_policy must be a TOML table")
    mode = _as_nonempty_string(topology.get("mode"), "topology_policy.mode")
    if mode != "no_regression_until_target":
        raise ValueError("topology_policy.mode must be 'no_regression_until_target'")
    _as_nonempty_string(
        topology.get("description"),
        "topology_policy.description",
    )
    counts = topology.get("counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("topology_policy.counts must be a non-empty list")

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(counts):
        if not isinstance(entry, dict):
            raise ValueError(f"topology_policy.counts[{index}] must be a table")
        prefix = f"topology_policy.counts[{index}]"
        name = _as_nonempty_string(entry.get("name"), f"{prefix}.name")
        raw_path = _as_nonempty_string(entry.get("path"), f"{prefix}.path")
        pattern = _as_nonempty_string(entry.get("pattern"), f"{prefix}.pattern")
        recursive = _as_bool(entry.get("recursive"), f"{prefix}.recursive")
        baseline = _as_nonnegative_int(entry.get("baseline"), f"{prefix}.baseline")
        target = _as_nonnegative_int(entry.get("target"), f"{prefix}.target")
        if target > baseline:
            raise ValueError(
                f"{name}: topology target {target} cannot exceed baseline {baseline}"
            )
        count = _count_matching_files(
            _repo_path(raw_path),
            pattern=pattern,
            recursive=recursive,
        )
        if count > baseline:
            raise ValueError(
                f"{name}: topology count regressed to {count}, above baseline {baseline}"
            )
        if require_targets and count > target:
            raise ValueError(
                f"{name}: topology target not met; count {count} exceeds target {target}"
            )
        rows.append(
            {
                "name": name,
                "path": raw_path,
                "pattern": pattern,
                "recursive": recursive,
                "count": count,
                "baseline": baseline,
                "target": target,
                "remaining_to_target": max(0, count - target),
                "target_met": count <= target,
            }
        )
    return rows


def _validate_line_budget_policy(
    data: dict[str, Any],
    *,
    require_targets: bool = False,
) -> list[dict[str, Any]]:
    """Validate aggregate Python-line budgets without hiding moved code."""

    policy = data.get("line_budget_policy")
    if policy is None:
        return []
    if not isinstance(policy, dict):
        raise ValueError("line_budget_policy must be a TOML table")
    mode = _as_nonempty_string(policy.get("mode"), "line_budget_policy.mode")
    if mode != "no_regression_until_target":
        raise ValueError(
            "line_budget_policy.mode must be 'no_regression_until_target'"
        )
    _as_nonempty_string(
        policy.get("description"), "line_budget_policy.description"
    )
    counts = policy.get("counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("line_budget_policy.counts must be a non-empty list")

    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(counts):
        if not isinstance(entry, dict):
            raise ValueError(f"line_budget_policy.counts[{index}] must be a table")
        prefix = f"line_budget_policy.counts[{index}]"
        name = _as_nonempty_string(entry.get("name"), f"{prefix}.name")
        raw_path = _as_nonempty_string(entry.get("path"), f"{prefix}.path")
        pattern = _as_nonempty_string(entry.get("pattern"), f"{prefix}.pattern")
        recursive = _as_bool(entry.get("recursive"), f"{prefix}.recursive")
        baseline = _as_nonnegative_int(entry.get("baseline"), f"{prefix}.baseline")
        target = _as_nonnegative_int(entry.get("target"), f"{prefix}.target")
        if target > baseline:
            raise ValueError(
                f"{name}: line-budget target {target} cannot exceed baseline {baseline}"
            )
        root = _repo_path(raw_path)
        if root.exists():
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            paths = sorted(path for path in iterator if path.is_file())
            lines = sum(_source_line_count(path) for path in paths)
        else:
            paths = []
            lines = 0
        if lines > baseline:
            raise ValueError(
                f"{name}: line count regressed to {lines}, above baseline {baseline}"
            )
        if require_targets and lines > target:
            raise ValueError(
                f"{name}: line-budget target not met; {lines} exceeds {target}"
            )
        rows.append(
            {
                "name": name,
                "path": raw_path,
                "pattern": pattern,
                "recursive": recursive,
                "files": len(paths),
                "lines": lines,
                "baseline": baseline,
                "target": target,
                "remaining_to_target": max(0, lines - target),
                "target_met": lines <= target,
            }
        )
    return rows


def validate_architecture_policy(
    data: dict[str, Any],
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    check_paths: bool = True,
    require_topology_targets: bool = False,
    require_complexity_targets: bool = False,
    require_line_targets: bool = False,
) -> dict[str, Any]:
    """Validate architecture-policy content and return a compact summary."""

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest must contain [metadata]")
    if metadata.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must be 1")
    for field in ("title", "layout_authority", "status"):
        _as_nonempty_string(metadata.get(field), f"metadata.{field}")
    if check_paths:
        authority = _repo_path(str(metadata["layout_authority"]))
        if not authority.is_file():
            raise ValueError(f"layout authority does not exist: {authority}")

    root_policy = data.get("root_prefix_policy")
    if not isinstance(root_policy, dict):
        raise ValueError("manifest must contain [root_prefix_policy]")
    blocked_prefixes = _as_nonempty_list(
        root_policy.get("blocked_prefixes"),
        "root_prefix_policy.blocked_prefixes",
    )
    allowed_modules = _as_string_list(
        root_policy.get("allowed_root_prefix_modules"),
        "root_prefix_policy.allowed_root_prefix_modules",
    )
    allowed_set = set(allowed_modules)

    root_modules: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = f"gkx.{path.stem}"
        if path.stem.startswith(tuple(blocked_prefixes)):
            root_modules.append(module)

    new_modules = sorted(set(root_modules) - allowed_set)
    stale_allowlist = sorted(allowed_set - set(root_modules))
    if new_modules:
        raise ValueError(
            "root-level prefix modules must be moved into domain packages or "
            f"listed as temporary facades: {new_modules}"
        )
    if stale_allowlist:
        raise ValueError(
            "root-prefix allowlist contains modules that no longer exist; "
            f"remove them from the manifest: {stale_allowlist}"
        )
    if check_paths:
        for module in allowed_modules:
            path = _root_module_path(module, source_root)
            if not path.is_file():
                raise ValueError(
                    f"allowed root-prefix module source does not exist: {module}"
                )

    package_policy = data.get("package_policy")
    if not isinstance(package_policy, dict):
        raise ValueError("manifest must contain [package_policy]")
    required_packages = _as_nonempty_list(
        package_policy.get("required_domain_packages"),
        "package_policy.required_domain_packages",
    )
    required_docs = _as_nonempty_list(
        package_policy.get("required_docs"),
        "package_policy.required_docs",
    )
    if check_paths:
        for package in required_packages:
            path = _package_path(package, source_root)
            if not path.is_file():
                raise ValueError(
                    f"required domain package is missing __init__.py: {package}"
                )
        for doc in required_docs:
            path = _repo_path(doc)
            if not path.is_file():
                raise ValueError(f"required architecture doc does not exist: {doc}")

    topology_counts = _validate_topology_policy(
        data,
        require_targets=require_topology_targets,
    )
    complexity_exceptions = _validate_complexity_policy(
        data,
        source_root=source_root,
        require_targets=require_complexity_targets,
    )
    line_budget_counts = _validate_line_budget_policy(
        data,
        require_targets=require_line_targets,
    )

    return {
        "layout_authority": str(metadata["layout_authority"]),
        "n_blocked_prefixes": len(blocked_prefixes),
        "n_allowed_root_prefix_modules": len(allowed_modules),
        "n_current_root_prefix_modules": len(root_modules),
        "n_required_domain_packages": len(required_packages),
        "required_domain_packages": required_packages,
        "n_topology_counts": len(topology_counts),
        "topology_counts": topology_counts,
        "topology_targets_met": all(row["target_met"] for row in topology_counts),
        "n_complexity_exceptions": len(complexity_exceptions),
        "complexity_exceptions": complexity_exceptions,
        "complexity_targets_met": all(
            row["target_met"] for row in complexity_exceptions
        ),
        "n_line_budget_counts": len(line_budget_counts),
        "line_budget_counts": line_budget_counts,
        "line_budget_targets_met": all(
            row["target_met"] for row in line_budget_counts
        ),
        "status": str(metadata["status"]),
    }


ROOT_MARKERS = ("pyproject.toml", ".git")
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".rst",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".csv",
    ".txt",
}


@dataclass(frozen=True)
class InventoryRow:
    path: str
    area: str
    role: str
    action: str
    suffix: str
    bytes: int
    lines: int | None
    notes: str


def _repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in (current, *current.parents):
        if all((parent / marker).exists() for marker in ROOT_MARKERS):
            return parent
    raise RuntimeError(f"could not find repository root from {start}")


def _git_ls_files(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return [root / line for line in proc.stdout.splitlines() if line]


def _line_count(path: Path) -> int | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        return sum(1 for _ in path.open("rb"))
    except OSError:
        return None


def _area(rel: Path) -> str:
    if len(rel.parts) == 1:
        return "root"
    if rel.parts[0] == "src" and len(rel.parts) > 2 and rel.parts[1] == "gkx":
        return (
            "src/gkx"
            if len(rel.parts) == 3
            else f"src/gkx/{rel.parts[2]}"
        )
    if rel.parts[0] in {"tests", "tools", "examples", "benchmarks", "docs"}:
        return rel.parts[0] if len(rel.parts) == 1 else f"{rel.parts[0]}/{rel.parts[1]}"
    return rel.parts[0]


def _role_and_action(rel: Path) -> tuple[str, str, str]:
    parts = rel.parts
    name = rel.name
    stem = rel.stem
    path = rel.as_posix()

    if parts[0] == "src" and len(parts) > 2 and parts[1] == "gkx":
        domain = parts[2] if len(parts) > 3 else "root"
        targets = {
            "artifacts": "gkx.io",
            "benchmarking": "benchmarks",
            "core": "gkx.model or gkx.numerics",
            "diagnostics": "gkx.diagnostics",
            "geometry": "gkx.geometry",
            "objectives": "gkx.optimize",
            "operators": "gkx.model",
            "parallel": "gkx.numerics",
            "solvers": "gkx.solve or gkx.numerics",
            "terms": "gkx.model",
            "utils": "nearest consuming domain",
            "workflows": "gkx.solve or gkx.cli",
            "root": "gkx public API or nearest domain",
        }
        return (
            "promoted library code",
            "merge-to-2.0-domain",
            f"target={targets.get(domain, 'nearest coherent gkx domain')}; preserve promoted physics and JAX contracts",
        )

    if parts[0] == "tests":
        if len(parts) == 2:
            return (
                "flat test",
                "move-or-merge",
                "root-level tests should move into domain folders",
            )
        return (
            "organized test",
            "merge-by-domain",
            "target=tests/{unit,integration,physics,release}; retain detection power and coverage",
        )

    if parts[0] == "tools":
        if name == "README.md":
            return "tool documentation", "keep", "documents maintenance-tool ownership"
        if len(parts) == 2 and rel.suffix == ".py":
            if stem.startswith(("probe_", "debug_")):
                return (
                    "probe/debug tool",
                    "delete-or-move-out-of-main",
                    "not a maintained release entry point",
                )
            if stem.startswith(("compare_", "generate_gx_")):
                return (
                    "comparison utility",
                    "move",
                    "belongs under tools/comparison if still current",
                )
            if stem.startswith(("profile_", "benchmark_")):
                return (
                    "profiling/performance tool",
                    "move",
                    "belongs under tools/profiling or benchmarks",
                )
            if stem.startswith(
                ("build_", "plot_", "make_", "digitize_", "derive_", "compress_")
            ):
                return (
                    "artifact builder",
                    "move-or-merge",
                    "belongs under tools/artifacts if output is referenced",
                )
            if stem.startswith(("check_", "audit_", "run_tests", "run_wide_coverage")):
                return "release gate", "move", "belongs under tools/release"
            if stem.startswith(
                (
                    "write_",
                    "run_",
                    "postprocess_",
                    "prepare_",
                    "finalize_",
                    "import_",
                    "rank_",
                    "design_",
                )
            ):
                return (
                    "campaign helper",
                    "move-or-delete",
                    "keep only if documented active campaign",
                )
            return "flat maintenance tool", "classify", "needs owner review"
        if len(parts) > 2:
            folder = parts[1]
            role = {
                "release": "release gate",
                "artifacts": "artifact builder",
                "profiling": "profiling/performance tool",
                "comparison": "comparison utility",
                "campaigns": "campaign helper",
            }.get(folder, "organized maintenance tool")
            return role, "keep-or-merge", f"owned by tools/{folder}"
        return "tool support file", "keep-or-review", "non-python tool asset"

    if parts[0] == "examples":
        return (
            "public example",
            "keep-or-scope",
            "must be runnable or explicitly marked long/manual",
        )

    if parts[0] == "benchmarks":
        if parts[1:2] == ("results",):
            return "benchmark result index", "keep-small", "no raw solver outputs"
        return "benchmark driver", "keep-small", "root benchmark layer is canonical"

    if parts[0] == "docs":
        if len(parts) > 1 and parts[1] == "_static":
            return (
                "docs artifact",
                "keep-if-referenced",
                "delete stale generated companions",
            )
        return "documentation", "keep-current", "must match promoted claims and layout"

    if parts[0] in {".github", "pyproject.toml", "uv.lock"} or path in {
        "pyproject.toml",
        "uv.lock",
    }:
        return "project infrastructure", "keep", "release/build/CI infrastructure"

    return "repository support", "keep-or-review", "classify during consolidation"


def build_inventory(root: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for path in _git_ls_files(root):
        if not path.exists():
            # Let the inventory run during refactors before deletions are staged.
            # Once the deletion is committed, git ls-files will no longer report it.
            continue
        rel = path.relative_to(root)
        role, action, notes = _role_and_action(rel)
        rows.append(
            InventoryRow(
                path=rel.as_posix(),
                area=_area(rel),
                role=role,
                action=action,
                suffix=path.suffix,
                bytes=path.stat().st_size,
                lines=_line_count(path),
                notes=notes,
            )
        )
    return rows


def _write_json(rows: Iterable[InventoryRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: Iterable[InventoryRow], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(rows[0]).keys())
            if rows
            else list(InventoryRow.__annotations__),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summary(rows: Iterable[InventoryRow]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = summary.setdefault(row.action, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += row.bytes
    return dict(sorted(summary.items()))


def build_inventory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root or any path inside it.",
    )
    parser.add_argument("--json-out", type=Path, help="Write full inventory JSON.")
    parser.add_argument("--csv-out", type=Path, help="Write full inventory CSV.")
    parser.add_argument(
        "--summary-json-out", type=Path, help="Write action summary JSON."
    )
    return parser


def main_inventory(argv: list[str] | None = None) -> int:
    args = build_inventory_parser().parse_args(argv)
    root = _repo_root(args.root)
    rows = build_inventory(root)
    if args.json_out:
        _write_json(rows, args.json_out)
    if args.csv_out:
        _write_csv(rows, args.csv_out)
    summary = _summary(rows)
    if args.summary_json_out:
        args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"tracked_files": len(rows), "actions": summary}, indent=2, sort_keys=True
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--require-topology-targets",
        action="store_true",
        help="Fail unless all topology counts are already at their final targets.",
    )
    parser.add_argument(
        "--require-complexity-targets",
        action="store_true",
        help="Fail unless all reviewed oversized modules meet their target budgets.",
    )
    parser.add_argument(
        "--require-line-targets",
        action="store_true",
        help="Fail unless all aggregate Python-line budgets meet final targets.",
    )
    return parser


def main_architecture(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_architecture_policy(
        load_manifest(args.manifest),
        source_root=args.source_root,
        check_paths=True,
        require_topology_targets=args.require_topology_targets,
        require_complexity_targets=args.require_complexity_targets,
        require_line_targets=args.require_line_targets,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.out_json}")
    else:
        print(payload, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(argv if argv is not None else sys.argv[1:])
    if tokens and tokens[0] == "inventory":
        return main_inventory(tokens[1:])
    return main_architecture(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
