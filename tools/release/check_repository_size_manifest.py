#!/usr/bin/env python3
"""Check tracked repository size against the artifact-hygiene manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

# The repo-hygiene CI job runs the release gates straight from a checkout with
# no install, so reach the repository's single TOML shim through ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gkx.utils import tomlcompat as tomllib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "tools/repository_size_manifest.toml"
DEFAULT_RELEASE_ARTIFACT_MANIFEST = ROOT / "tools/release_artifact_manifest.toml"
DEFAULT_RELEASE_MANIFEST = DEFAULT_RELEASE_ARTIFACT_MANIFEST
DEFAULT_STATIC_DIR = ROOT / "docs" / "_static"
VALID_RELEASE_ARTIFACT_ACTIONS = {
    "keep_in_repo",
    "keep_preview_in_repo",
    "move_to_release",
    "regenerate_on_demand",
}
LOCAL_ARTIFACT_ROOTS = (
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "docs/_build",
    "tools_out",
)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _tracked_paths(root: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True
    )
    raw = proc.stdout.split(b"\0")
    return [Path(item.decode()) for item in raw if item]


def _category(path: Path) -> str:
    parts = path.parts
    if not parts:
        return "."
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "_static":
        return "docs/_static"
    if len(parts) >= 2 and parts[0] == "docs" and parts[1] == "_build":
        return "docs/_build"
    return parts[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_repository_size_report(
    root: str | Path = ROOT, *, top_n: int = 30
) -> dict[str, Any]:
    """Return a JSON-ready size report for tracked files and local artifact roots."""

    root_path = Path(root).resolve()
    rows = []
    by_category: dict[str, int] = defaultdict(int)
    for rel in _tracked_paths(root_path):
        path = root_path / rel
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rows.append({"path": str(rel), "bytes": int(size), "category": _category(rel)})
        by_category[_category(rel)] += int(size)

    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    local_roots = []
    for rel in LOCAL_ARTIFACT_ROOTS:
        path = root_path / rel
        local_roots.append(
            {"path": rel, "bytes": int(_directory_size(path)), "exists": path.exists()}
        )

    return {
        "kind": "repository_size_audit",
        "root": str(root_path),
        "tracked_total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "tracked_file_count": len(rows),
        "tracked_by_category": dict(
            sorted(by_category.items(), key=lambda item: item[1], reverse=True)
        ),
        "largest_tracked_files": rows[:top_n],
        "local_artifact_roots": local_roots,
        "notes": (
            "This report is non-destructive. Move large tracked validation artifacts to "
            "GitHub Releases or another artifact store before considering a coordinated "
            "history rewrite."
        ),
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    policy = data.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path} must contain a [policy] table")
    for key in ("max_tracked_total_bytes", "max_unlisted_tracked_file_bytes"):
        value = policy.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} policy.{key} must be a positive integer")
    allowed = data.get("allowed_large_files", [])
    if not isinstance(allowed, list):
        raise ValueError(f"{path} allowed_large_files must be an array of tables")
    paths: set[str] = set()
    for idx, item in enumerate(allowed):
        if not isinstance(item, dict):
            raise ValueError(f"{path} allowed_large_files[{idx}] must be a table")
        item_path = item.get("path")
        max_bytes = item.get("max_bytes")
        reason = item.get("reason")
        if not isinstance(item_path, str) or not item_path:
            raise ValueError(
                f"{path} allowed_large_files[{idx}].path must be a non-empty string"
            )
        if item_path in paths:
            raise ValueError(
                f"{path} lists duplicate allowed_large_files path {item_path!r}"
            )
        paths.add(item_path)
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError(
                f"{path} allowed_large_files[{idx}].max_bytes must be a positive integer"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                f"{path} allowed_large_files[{idx}].reason must be a non-empty string"
            )
    return data


def _load_release_artifact_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    policy = data.get("policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path} must contain a [policy] table")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{path} must contain at least one [[artifacts]] table")
    seen: set[str] = set()
    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            raise ValueError(f"{path} artifacts[{idx}] must be a table")
        for key in (
            "path",
            "size_bytes",
            "sha256",
            "action",
            "reason",
            "replay_command",
        ):
            if key not in item:
                raise ValueError(f"{path} artifacts[{idx}] is missing {key!r}")
        artifact_path = item["path"]
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"{path} artifacts[{idx}].path must be a non-empty string")
        if artifact_path in seen:
            raise ValueError(f"{path} lists duplicate artifact path {artifact_path!r}")
        seen.add(artifact_path)
        if not isinstance(item["size_bytes"], int) or int(item["size_bytes"]) <= 0:
            raise ValueError(
                f"{path} artifacts[{idx}].size_bytes must be a positive integer"
            )
        if not isinstance(item["sha256"], str) or len(item["sha256"]) != 64:
            raise ValueError(
                f"{path} artifacts[{idx}].sha256 must be a hex sha256 string"
            )
        if item["action"] not in VALID_RELEASE_ARTIFACT_ACTIONS:
            raise ValueError(
                f"{path} artifacts[{idx}].action must be one of {sorted(VALID_RELEASE_ARTIFACT_ACTIONS)}"
            )
        if item["action"] in {"move_to_release", "keep_preview_in_repo"}:
            for key in ("release_asset_name", "preview_strategy"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(
                        f"{path} artifacts[{idx}] {item['action']} entries need {key!r}"
                    )
        if item["action"] == "regenerate_on_demand":
            replay_command = item.get("replay_command")
            if not isinstance(replay_command, str) or not replay_command.strip():
                raise ValueError(
                    f"{path} artifacts[{idx}] regenerate_on_demand entries need "
                    "a non-empty 'replay_command'"
                )
    return data


def check_repository_size_manifest(
    *,
    root: str | Path = ROOT,
    manifest: str | Path = DEFAULT_MANIFEST,
    top_n: int = 50,
) -> dict[str, Any]:
    """Return a JSON-ready pass/fail report for tracked repository size policy."""

    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    data = _load_manifest(manifest_path)
    policy = data["policy"]
    allowed = {item["path"]: item for item in data.get("allowed_large_files", [])}
    audit = build_repository_size_report(root_path, top_n=max(top_n, len(allowed) + 10))

    failures: list[str] = []
    if int(audit["tracked_total_bytes"]) > int(policy["max_tracked_total_bytes"]):
        failures.append(
            "tracked_total_bytes={actual} exceeds max_tracked_total_bytes={limit}".format(
                actual=audit["tracked_total_bytes"],
                limit=policy["max_tracked_total_bytes"],
            )
        )

    tracked = {row["path"]: int(row["bytes"]) for row in audit["largest_tracked_files"]}
    if len(tracked) < int(audit["tracked_file_count"]):
        for row in build_repository_size_report(
            root_path, top_n=int(audit["tracked_file_count"])
        )["largest_tracked_files"]:
            tracked[row["path"]] = int(row["bytes"])

    for path, item in allowed.items():
        if path not in tracked:
            failures.append(f"allowed large file {path!r} is not tracked")
            continue
        if tracked[path] > int(item["max_bytes"]):
            failures.append(
                f"{path} has {tracked[path]} bytes, exceeding allowed max_bytes={item['max_bytes']}"
            )

    threshold = int(policy["max_unlisted_tracked_file_bytes"])
    unlisted_large = [
        {"path": path, "bytes": size}
        for path, size in sorted(
            tracked.items(), key=lambda item: item[1], reverse=True
        )
        if size > threshold and path not in allowed
    ]
    for item in unlisted_large:
        failures.append(
            "{path} has {bytes} bytes, exceeding max_unlisted_tracked_file_bytes={threshold} and is not whitelisted".format(
                path=item["path"],
                bytes=item["bytes"],
                threshold=threshold,
            )
        )

    return {
        "kind": "repository_size_manifest_check",
        "manifest": str(manifest_path),
        "passed": not failures,
        "failures": failures,
        "policy": policy,
        "allowed_large_files": list(allowed.values()),
        "unlisted_large_files": unlisted_large,
        "audit": audit,
    }


def check_release_artifact_manifest(
    *,
    root: str | Path = ROOT,
    manifest: str | Path = DEFAULT_RELEASE_ARTIFACT_MANIFEST,
) -> dict[str, Any]:
    """Return a JSON-ready validation report for the release-artifact manifest."""

    root_path = Path(root).resolve()
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root_path / manifest_path
    data = _load_release_artifact_manifest(manifest_path)
    rows = []
    failures = []
    total_move_bytes = 0
    total_keep_bytes = 0
    for item in data["artifacts"]:
        rel = Path(item["path"])
        path = root_path / rel
        exists = path.exists()
        actual_size = path.stat().st_size if exists else None
        actual_sha = _sha256(path) if exists and path.is_file() else None
        action = str(item["action"])
        if action == "move_to_release":
            total_move_bytes += int(item["size_bytes"])
        if action == "keep_in_repo" and actual_size is not None:
            total_keep_bytes += int(actual_size)
        row = {
            "path": item["path"],
            "action": action,
            "exists": exists,
            "expected_size_bytes": item["size_bytes"],
            "actual_size_bytes": actual_size,
            "expected_sha256": item["sha256"],
            "actual_sha256": actual_sha,
            "release_asset_name": item.get("release_asset_name"),
            "preview_strategy": item.get("preview_strategy"),
            "release_tag": item.get("release_tag"),
            "release_url": item.get("release_url"),
            "replay_command": item.get("replay_command"),
        }
        rows.append(row)
        if not exists:
            if action == "regenerate_on_demand":
                continue
            release_url = item.get("release_url")
            release_tag = item.get("release_tag")
            if (
                action == "move_to_release"
                and isinstance(release_url, str)
                and release_url.strip()
            ):
                if not isinstance(release_tag, str) or not release_tag.strip():
                    failures.append(
                        f"{item['path']} moved to release but is missing release_tag"
                    )
                continue
            failures.append(f"{item['path']} does not exist")
            continue
        if actual_size != int(item["size_bytes"]):
            failures.append(
                f"{item['path']} size {actual_size} != manifest size {item['size_bytes']}"
            )
        if actual_sha != item["sha256"]:
            failures.append(
                f"{item['path']} sha256 {actual_sha} != manifest sha256 {item['sha256']}"
            )

    return {
        "kind": "release_artifact_manifest_check",
        "manifest": str(manifest_path),
        "passed": not failures,
        "failures": failures,
        "policy": data["policy"],
        "artifact_count": len(rows),
        "move_to_release_bytes": total_move_bytes,
        "keep_in_repo_bytes": total_keep_bytes,
        "artifacts": rows,
        "notes": (
            "This validates provenance for large tracked assets. Existing files are "
            "checked against size and sha256. move_to_release entries may be absent "
            "from Git only when the manifest records release_tag and release_url; "
            "regenerate_on_demand entries may be absent when their replay command is "
            "recorded."
        ),
    }


def _repo_relative(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def release_manifest_paths(
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> set[Path]:
    """Return repo-relative artifact paths pinned by the release manifest."""

    path = Path(manifest)
    if not path.exists():
        return set()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    paths: set[Path] = set()
    for item in data.get("artifacts", []):
        raw = item.get("path")
        if isinstance(raw, str) and raw:
            paths.add(Path(raw))
    return paths


def release_preview_targets(
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
) -> list[Path]:
    """Return repo-relative release-manifest PNG preview targets."""

    path = Path(manifest)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    targets: list[Path] = []
    for item in data.get("artifacts", []):
        rel = Path(str(item.get("path", "")))
        action = str(item.get("action", ""))
        preview_strategy = str(item.get("preview_strategy", ""))
        if rel.suffix.lower() != ".png":
            continue
        if action not in {"move_to_release", "keep_preview_in_repo"}:
            continue
        if "preview" in preview_strategy.lower() or action == "keep_preview_in_repo":
            targets.append(rel)
    return targets


def compress_png_preview(
    path: str | Path,
    *,
    max_width: int = 1800,
    colors: int = 192,
    dry_run: bool = False,
    restore_if_larger: bool = True,
) -> dict[str, Any]:
    """Compress one PNG preview and return size/checksum metadata."""

    from PIL import Image

    target = Path(path)
    before_size = target.stat().st_size
    before_sha = _sha256(target)
    image = Image.open(target).convert("RGBA")
    original_dimensions = image.size
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    rgb = background.convert("RGB")
    report: dict[str, Any] = {
        "path": str(target),
        "original_dimensions": original_dimensions,
        "preview_dimensions": image.size,
        "before_size_bytes": before_size,
        "before_sha256": before_sha,
        "dry_run": dry_run,
    }
    if dry_run:
        report.update(
            {
                "after_size_bytes": before_size,
                "after_sha256": before_sha,
                "saved_bytes": 0,
            }
        )
        return report
    original_bytes = target.read_bytes() if restore_if_larger else None
    palette = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    palette.save(target, optimize=True)
    after_size = target.stat().st_size
    if restore_if_larger and original_bytes is not None and after_size >= before_size:
        target.write_bytes(original_bytes)
        after_size = before_size
    after_sha = _sha256(target)
    report.update(
        {
            "after_size_bytes": after_size,
            "after_sha256": after_sha,
            "saved_bytes": before_size - after_size,
        }
    )
    return report


def compress_docs_previews(
    *,
    static_dir: str | Path = DEFAULT_STATIC_DIR,
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
    min_bytes: int = 300_000,
    max_width: int = 1800,
    colors: int = 192,
    dry_run: bool = False,
    include_manifest_paths: bool = False,
) -> list[dict[str, Any]]:
    """Compress large docs PNG previews, skipping release-manifest paths by default."""

    static = Path(static_dir)
    manifest_rel = (
        release_manifest_paths(manifest) if not include_manifest_paths else set()
    )
    reports: list[dict[str, Any]] = []
    for path in sorted(static.glob("*.png")):
        if path.stat().st_size < min_bytes:
            continue
        rel = _repo_relative(path)
        if rel in manifest_rel or path in manifest_rel:
            reports.append(
                {
                    "path": str(path),
                    "skipped": True,
                    "reason": "release_artifact_manifest",
                    "before_size_bytes": path.stat().st_size,
                    "after_size_bytes": path.stat().st_size,
                    "saved_bytes": 0,
                }
            )
            continue
        report = compress_png_preview(
            path,
            max_width=max_width,
            colors=colors,
            dry_run=dry_run,
            restore_if_larger=True,
        )
        report["skipped"] = False
        reports.append(report)
    return reports


def compress_release_previews(
    *,
    manifest: str | Path = DEFAULT_RELEASE_MANIFEST,
    max_width: int = 2200,
    colors: int = 192,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Compress PNG previews selected by the release artifact manifest."""

    reports: list[dict[str, Any]] = []
    for rel in release_preview_targets(manifest):
        path = rel if rel.is_absolute() else ROOT / rel
        report = compress_png_preview(
            path,
            max_width=max_width,
            colors=colors,
            dry_run=dry_run,
            restore_if_larger=False,
        )
        report["manifest_path"] = str(rel)
        report["skipped"] = False
        reports.append(report)
    return reports


def build_compress_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("docs", "release"),
        default="docs",
        help="docs scans docs/_static; release follows tools/release_artifact_manifest.toml.",
    )
    parser.add_argument(
        "--static-dir",
        default=str(DEFAULT_STATIC_DIR),
        help="docs/_static directory to scan in docs mode.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RELEASE_MANIFEST),
        help="Release artifact manifest used for skip or target selection.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=300_000,
        help="Docs mode only: only compress PNGs at or above this size.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Maximum preview width. Defaults to 1800 for docs and 2200 for release.",
    )
    parser.add_argument("--colors", type=int, default=192, help="Palette colors.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without modifying files."
    )
    parser.add_argument(
        "--include-release-manifest-paths",
        action="store_true",
        help="Docs mode only: also compress files pinned by the release manifest.",
    )
    return parser


def _print_reports(reports: list[dict[str, Any]], *, release: bool) -> None:
    total_saved = 0
    for report in reports:
        if report.get("skipped"):
            print(f"skip {report['path']}: {report['reason']}")
            continue
        total_saved += int(report["saved_bytes"])
        label = (
            report.get("manifest_path", report["path"]) if release else report["path"]
        )
        print(
            "{path}: {before} -> {after} bytes, saved {saved}".format(
                path=label,
                before=report["before_size_bytes"],
                after=report["after_size_bytes"],
                saved=report["saved_bytes"],
            )
        )
    print(f"total_saved={total_saved}")


def main_compress_previews(argv: list[str] | None = None) -> int:
    args = build_compress_parser().parse_args(argv)
    if args.mode == "release":
        reports = compress_release_previews(
            manifest=args.manifest,
            max_width=2200 if args.max_width is None else int(args.max_width),
            colors=int(args.colors),
            dry_run=bool(args.dry_run),
        )
        _print_reports(reports, release=True)
        return 0
    reports = compress_docs_previews(
        static_dir=args.static_dir,
        manifest=args.manifest,
        min_bytes=int(args.min_bytes),
        max_width=1800 if args.max_width is None else int(args.max_width),
        colors=int(args.colors),
        dry_run=bool(args.dry_run),
        include_manifest_paths=bool(args.include_release_manifest_paths),
    )
    _print_reports(reports, release=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Repository size manifest TOML.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def build_audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit tracked file sizes and local artifact roots without modifying the repo."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root to audit.")
    parser.add_argument(
        "--top", type=int, default=30, help="Number of tracked files to list."
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def build_release_artifact_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the release-artifact manifest for large tracked assets."
    )
    parser.add_argument("--root", default=str(ROOT), help="Repository root to check.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_RELEASE_ARTIFACT_MANIFEST),
        help="Release artifact manifest TOML.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path.")
    return parser


def main_audit(argv: list[str] | None = None) -> int:
    args = build_audit_parser().parse_args(argv)
    report = build_repository_size_report(args.root, top_n=args.top)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    return 0


def main_release_artifacts(argv: list[str] | None = None) -> int:
    args = build_release_artifact_parser().parse_args(argv)
    report = check_release_artifact_manifest(root=args.root, manifest=args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    if not report["passed"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "release artifact manifest check passed: artifacts={count} move_to_release_bytes={bytes}".format(
            count=report["artifact_count"],
            bytes=report["move_to_release_bytes"],
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "audit":
        return main_audit(tokens[1:])
    if tokens and tokens[0] == "release-artifacts":
        return main_release_artifacts(tokens[1:])
    if tokens and tokens[0] == "compress-previews":
        return main_compress_previews(tokens[1:])

    args = build_parser().parse_args(tokens)
    report = check_repository_size_manifest(root=args.root, manifest=args.manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {out}")
    else:
        print(text)
    if not report["passed"]:
        for failure in report["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        "repository size check passed: tracked_total_bytes={bytes}".format(
            bytes=report["audit"]["tracked_total_bytes"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
