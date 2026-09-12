#!/usr/bin/env python3
"""Summarize notebook size, execution state, and portability risks.

By default only notebooks tracked by Git are inspected. Use ``--all`` to include
ignored and untracked local research notebooks without modifying them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


DATA_REFERENCE_RE = re.compile(
    r"(?P<quote>['\"])(?P<path>[^'\"\n]+\.(?:npy|npz|h5|hdf5|txt|csv|yaml|yml))(?P=quote)",
    re.IGNORECASE,
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:/mnt/[a-z]/|/(?:home|Users|tmp|var|opt|data)/|[A-Za-z]:[/\\])",
    re.IGNORECASE,
)


@dataclass
class NotebookInfo:
    path: str
    kib: float
    cells: int
    code_cells: int
    executed_cells: int
    outputs: int
    data_references: list[str]
    has_absolute_paths: bool


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracked_notebooks(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "notebooks/*.ipynb", "notebooks/**/*.ipynb"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [root / line for line in result.stdout.splitlines() if line]


def inspect(path: Path, root: Path) -> NotebookInfo:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    source = "\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in code_cells
    )
    references = sorted({match.group("path") for match in DATA_REFERENCE_RE.finditer(source)})
    return NotebookInfo(
        path=path.relative_to(root).as_posix(),
        kib=round(path.stat().st_size / 1024, 1),
        cells=len(cells),
        code_cells=len(code_cells),
        executed_cells=sum(cell.get("execution_count") is not None for cell in code_cells),
        outputs=sum(len(cell.get("outputs", [])) for cell in code_cells),
        data_references=references,
        has_absolute_paths=bool(ABSOLUTE_PATH_RE.search(source)),
    )


def print_table(items: list[NotebookInfo]) -> None:
    header = f"{'notebook':64} {'KiB':>8} {'cells':>5} {'exec':>5} {'out':>5} {'abs':>4} {'data':>5}"
    print(header)
    print("-" * len(header))
    for item in items:
        print(
            f"{item.path[:64]:64} {item.kib:8.1f} {item.cells:5d} "
            f"{item.executed_cells:5d} {item.outputs:5d} "
            f"{('yes' if item.has_absolute_paths else 'no'):>4} "
            f"{len(item.data_references):5d}"
        )

    print()
    print(f"notebooks: {len(items)}")
    print(f"total size: {sum(item.kib for item in items) / 1024:.2f} MiB")
    print(f"embedded outputs: {sum(item.outputs for item in items)}")
    print(f"notebooks with absolute paths: {sum(item.has_absolute_paths for item in items)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="include ignored and untracked notebooks under notebooks/",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    root = repository_root()
    paths = sorted((root / "notebooks").rglob("*.ipynb")) if args.all else tracked_notebooks(root)
    items = [inspect(path, root) for path in paths]
    if args.json:
        print(json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False))
    else:
        print_table(items)


if __name__ == "__main__":
    main()
