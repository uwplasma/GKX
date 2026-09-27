"""ARCH-B inventory: import graph, cycles, single-consumer modules, dead names.

Standard library only. Run from a checkout root: ``python plan/research/2026-09-27-arch-b/inventory.py``.
Edges include function-local imports; ``from gkx.pkg import sub`` counts as an
edge to ``gkx.pkg.sub`` when ``sub`` is a module. Lazy registries
(``_EXPORT_TARGETS``) are read as edges from the registry module.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
SRC = ROOT / "src" / "gkx"


def modname(path: Path) -> str:
    name = path.relative_to(SRC.parent).as_posix()[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


MODULES = {modname(p): p for p in sorted(SRC.rglob("*.py"))}


def resolve(base: str, name: str | None) -> str | None:
    if name and f"{base}.{name}" in MODULES:
        return f"{base}.{name}"
    return base if base in MODULES else None


def edges() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for me, path in MODULES.items():
        tree = ast.parse(path.read_text())
        pkg = me if path.name == "__init__.py" else me.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    parts = parts[: len(parts) - (node.level - 1)]
                    mod = ".".join(parts + ([mod] if mod else []))
                if not mod.startswith("gkx"):
                    continue
                for alias in node.names:
                    target = resolve(mod, alias.name)
                    if target and target != me:
                        graph[me].add(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name.startswith("gkx")
                        and alias.name in MODULES
                        and alias.name != me
                    ):
                        graph[me].add(alias.name)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if (
                    re.fullmatch(r"gkx(\.\w+)+", node.value)
                    and node.value in MODULES
                    and node.value != me
                ):
                    graph[me].add(node.value)
    return graph


def sccs(graph: dict[str, set[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    out: list[list[str]] = []
    counter = [0]
    sys.setrecursionlimit(10000)

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph.get(v, ()):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in MODULES:
        if v not in index:
            visit(v)
    return out


def main() -> None:
    graph = edges()
    consumers: dict[str, set[str]] = defaultdict(set)
    for a, targets in graph.items():
        for b in targets:
            consumers[b].add(a)
    lines = {m: len(p.read_text().splitlines()) for m, p in MODULES.items()}
    single = sorted(
        (m, next(iter(consumers[m])), lines[m])
        for m in MODULES
        if len(consumers[m]) == 1 and not MODULES[m].name == "__init__.py"
    )
    zero = sorted(m for m in MODULES if not consumers[m])
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    corpus_files = [
        f
        for f in files
        if f.endswith((".py", ".toml", ".rst", ".md", ".yml", ".yaml", ".cfg", ".ini"))
        and not f.startswith("plan/")
    ]
    words: dict[str, int] = defaultdict(int)
    for f in corpus_files:
        try:
            text = (ROOT / f).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
            words[w] += 1
    dead = []
    for m, p in MODULES.items():
        tree = ast.parse(p.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                if name.startswith("__"):
                    continue
                body = ast.get_source_segment(p.read_text(), node) or ""
                own = len(re.findall(rf"\b{re.escape(name)}\b", body))
                if words[name] - own <= 0:
                    dead.append((m, name, node.end_lineno - node.lineno + 1))
    print(
        json.dumps(
            {
                "files": len(MODULES),
                "lines": sum(lines.values()),
                "cycles": sccs(graph),
                "single_consumer": single,
                "no_consumer": zero,
                "dead_definitions": sorted(dead),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
