#!/usr/bin/env python3
"""
Cleanup analysis script: cross-reference output/ with notebook references.
Generates lists of safe-to-delete paths.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"d:\rave-sim-main\rave-sim-main")
OUTPUT_DIR = ROOT / "output"
NOTEBOOKS_DIR = ROOT / "notebooks"


def find_all_output_dirs():
    """Find all timestamped output directories (depth 3: output/YYYY/MM/timestamp/)."""
    dirs = set()
    for year_dir in OUTPUT_DIR.iterdir():
        if not year_dir.is_dir() or not re.match(r"^\d{4}$", year_dir.name):
            continue
        for month_dir in year_dir.iterdir():
            if not month_dir.is_dir() or not re.match(r"^\d{2}$", month_dir.name):
                continue
            for ts_dir in month_dir.iterdir():
                if ts_dir.is_dir():
                    rel = ts_dir.relative_to(ROOT).as_posix()
                    dirs.add(rel)
    return dirs


def find_all_output_files():
    """Find all files directly in output/ root (not in year subdirs)."""
    files = []
    for item in OUTPUT_DIR.iterdir():
        if item.is_file():
            files.append(item)
    return files


def extract_notebook_references():
    """Extract all output/YYYY/MM/timestamp references from notebooks (.ipynb and .py)."""
    refs = set()
    pattern = re.compile(r'output/\d{4}/\d{2}/\d{8}_\d+')

    for nb_dir in NOTEBOOKS_DIR.rglob("*"):
        if nb_dir.is_file() and nb_dir.suffix in (".ipynb", ".py"):
            try:
                content = nb_dir.read_text(encoding="utf-8", errors="ignore")
                for match in pattern.finditer(content):
                    refs.add(match.group())
            except Exception:
                pass

    return refs


def find_ipynb_checkpoints():
    """Find all .ipynb_checkpoints directories."""
    return sorted(NOTEBOOKS_DIR.rglob(".ipynb_checkpoints"))


def find_debug_wave_files():
    """Find large debug_wave npy files."""
    return sorted(NOTEBOOKS_DIR.rglob("debug_wave/*.npy"))


def get_dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def format_size(size_bytes):
    """Format bytes to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def main():
    print("=" * 80)
    print("RAVE-SIM OUTPUT CLEANUP ANALYSIS")
    print("=" * 80)

    # -- Category 1: Root-level orphaned files --
    print("\n" + "-" * 80)
    print("[Category 1] output/ root-level orphaned files (not referenced by any notebook)")
    print("-" * 80)

    root_files = find_all_output_files()
    total_root_files = 0
    for f in sorted(root_files):
        size = f.stat().st_size
        total_root_files += size
        print(f"  {f.relative_to(ROOT).as_posix()}  ({format_size(size)})")
    print(f"  => Subtotal: {format_size(total_root_files)}")

    # -- Category 2: output1/ duplicate --
    print("\n" + "-" * 80)
    print("[Category 2] output/output1/ -- duplicate of root-level PNGs")
    print("-" * 80)
    output1 = OUTPUT_DIR / "output1"
    output1_size = 0
    if output1.exists():
        output1_size = get_dir_size(output1)
        print(f"  output/output1/  ({format_size(output1_size)})")
        for f in sorted(output1.rglob("*")):
            if f.is_file():
                print(f"    {f.name}  ({format_size(f.stat().st_size)})")
    else:
        print("  (does not exist)")

    # -- Category 3: Unreferenced timestamped output dirs --
    print("\n" + "-" * 80)
    print("[Category 3] Timestamped output directories NOT referenced by any notebook")
    print("-" * 80)

    print("  Scanning disk directories ...")
    disk_dirs = find_all_output_dirs()
    print(f"  On-disk dirs: {len(disk_dirs)}")

    print("  Extracting notebook references ...")
    notebook_refs = extract_notebook_references()
    print(f"  Notebook referenced dirs: {len(notebook_refs)}")

    unreferenced = disk_dirs - notebook_refs
    print(f"  Unreferenced dirs: {len(unreferenced)}")

    # Group by month and compute sizes
    by_month = defaultdict(list)
    total_unref_size = 0
    unreferenced_with_sizes = []

    print("\n  Computing sizes of unreferenced dirs (this may take a few minutes) ...")
    for i, d in enumerate(sorted(unreferenced)):
        full_path = ROOT / d
        sz = get_dir_size(full_path)
        total_unref_size += sz
        unreferenced_with_sizes.append((d, sz))

        parts = d.split("/")
        month_key = f"{parts[1]}/{parts[2]}"
        by_month[month_key].append((d, sz))

        if (i + 1) % 200 == 0:
            print(f"    Processed: {i + 1}/{len(unreferenced)} ...")

    print(f"\n  Unreferenced dirs total size: {format_size(total_unref_size)}")

    print("\n  By month:")
    for month in sorted(by_month.keys()):
        dirs_in_month = by_month[month]
        month_total = sum(sz for _, sz in dirs_in_month)
        print(f"    output/{month}/  -- {len(dirs_in_month)} dirs, {format_size(month_total)}")

    print("\n  Sample (first 10):")
    for d, sz in unreferenced_with_sizes[:10]:
        print(f"    {d}  ({format_size(sz)})")

    # -- Category 4: .ipynb_checkpoints --
    print("\n" + "-" * 80)
    print("[Category 4] Jupyter checkpoint directories (.ipynb_checkpoints)")
    print("-" * 80)
    checkpoints = find_ipynb_checkpoints()
    total_cp = 0
    for cp in checkpoints:
        sz = get_dir_size(cp)
        total_cp += sz
        print(f"  {cp.relative_to(ROOT).as_posix()}/  ({format_size(sz)})")
    print(f"  => Subtotal: {format_size(total_cp)}")

    # -- Category 5: debug_wave npy files --
    print("\n" + "-" * 80)
    print("[Category 5] debug_wave large .npy files")
    print("-" * 80)
    debug_files = find_debug_wave_files()
    total_debug = 0
    for f in debug_files:
        sz = f.stat().st_size
        total_debug += sz
        print(f"  {f.relative_to(ROOT).as_posix()}  ({format_size(sz)})")
    if not debug_files:
        print("  (none)")
    print(f"  => Subtotal: {format_size(total_debug)}")

    # -- Category 6: Empty directories (excluding build/output structure) --
    print("\n" + "-" * 80)
    print("[Category 6] Empty directories")
    print("-" * 80)
    empty_dirs = []
    EXCLUDE_PREFIXES = (
        "build", "build-", ".git", "__pycache__", ".venv", "node_modules",
        "CMakeFiles", "Testing", "incremental",
    )
    for item in ROOT.rglob("*"):
        if not item.is_dir():
            continue
        parts = item.relative_to(ROOT).parts
        if any(p.startswith(EXCLUDE_PREFIXES) for p in parts):
            continue
        try:
            if not any(item.iterdir()):
                rel = item.relative_to(ROOT).as_posix()
                if rel.startswith("output/") and rel.count("/") <= 2:
                    continue
                empty_dirs.append(rel)
        except (OSError, PermissionError):
            pass
    for d in sorted(empty_dirs):
        print(f"  {d}/")
    if not empty_dirs:
        print("  (none)")

    # -- Category 7: Misc orphaned files in notebooks --
    print("\n" + "-" * 80)
    print("[Category 7] Misc orphaned files in notebooks/")
    print("-" * 80)
    misc = [
        NOTEBOOKS_DIR / "新建 Microsoft PowerPoint 演示文稿.pptx",
        NOTEBOOKS_DIR / "Phase_shift_42_4.npy",
        NOTEBOOKS_DIR / "Error_measured.npy",
        NOTEBOOKS_DIR / "Sensitivity_measured.npy",
        NOTEBOOKS_DIR / "spectrum_70_spekpy_filtered_3mmAl.h5",
    ]
    for f in misc:
        if f.exists():
            sz = f.stat().st_size
            print(f"  {f.relative_to(ROOT).as_posix()}  ({format_size(sz)})")
        else:
            print(f"  {f.relative_to(ROOT).as_posix()}  (does not exist)")

    # -- Category 8: Old notebook dirs for archival --
    print("\n" + "-" * 80)
    print("[Category 8] Old notebook directories (archive candidates)")
    print("-" * 80)
    archive_candidates = [
        "notebooks/save_notebooks",
        "notebooks/test_notebooks",
        "notebooks/shockwavetest_251225",
        "notebooks/capsule_test_260508",
        "notebooks/new_test",
    ]
    for d in archive_candidates:
        p = ROOT / d
        if p.exists():
            sz = get_dir_size(p)
            print(f"  {d}/  ({format_size(sz)})")
        else:
            print(f"  {d}/  (does not exist)")

    # -- Summary --
    print("\n" + "=" * 80)
    print("CLEANUP SUMMARY")
    print("=" * 80)
    total_cleanup = total_root_files + output1_size + total_unref_size + total_cp + total_debug
    print(f"  Category 1 (orphan root files):      {format_size(total_root_files)}")
    print(f"  Category 2 (output1/ duplicate):     {format_size(output1_size)}")
    print(f"  Category 3 (unreferenced dirs):      {format_size(total_unref_size)}  ({len(unreferenced)} dirs)")
    print(f"  Category 4 (Jupyter checkpoints):    {format_size(total_cp)}  ({len(checkpoints)} dirs)")
    print(f"  Category 5 (debug_wave files):       {format_size(total_debug)}  ({len(debug_files)} files)")
    print(f"  Category 6 (empty dirs):             {len(empty_dirs)} dirs")
    print(f"  " + "-" * 60)
    print(f"  TOTAL safe to delete:                {format_size(total_cleanup)}")

    # -- Write deletion lists --
    print("\n" + "=" * 80)
    print("Generating cleanup list files...")
    print("=" * 80)

    # List 1: All safe-to-delete paths
    safe_list_path = ROOT / "cleanup_safe_to_delete.txt"
    with open(safe_list_path, "w", encoding="utf-8") as f:
        f.write("# RAVE-SIM Safe-to-Delete Manifest\n")
        f.write(f"# Generated: 2026-08-07\n")
        f.write(f"# Total reclaimable: {format_size(total_cleanup)}\n")
        f.write("# Each line is a path relative to project root\n")
        f.write("# Review each line before deleting.\n\n")

        f.write("## Category 1: Orphan root files\n")
        for item in sorted(root_files):
            f.write(f"{item.relative_to(ROOT).as_posix()}\n")

        f.write("\n## Category 2: output1/ duplicate\n")
        f.write("output/output1\n")

        f.write("\n## Category 3: Unreferenced output directories\n")
        for d, _ in sorted(unreferenced_with_sizes):
            f.write(f"{d}\n")

        f.write("\n## Category 4: Jupyter checkpoints\n")
        for cp in sorted(checkpoints):
            f.write(f"{cp.relative_to(ROOT).as_posix()}\n")

        f.write("\n## Category 5: debug_wave files\n")
        for df in sorted(debug_files):
            f.write(f"{df.relative_to(ROOT).as_posix()}\n")

        f.write("\n## Category 6: Empty directories\n")
        for d in sorted(empty_dirs):
            f.write(f"{d}\n")

    print(f"  [OK] Full manifest: {safe_list_path}")

    # List 2: Category 3 only (biggest impact, for careful review)
    unref_list_path = ROOT / "cleanup_unreferenced_dirs.txt"
    with open(unref_list_path, "w", encoding="utf-8") as f:
        f.write("# Unreferenced output directories (Category 3)\n")
        f.write(f"# {len(unreferenced)} dirs, {format_size(total_unref_size)}\n\n")
        for d, sz in sorted(unreferenced_with_sizes):
            f.write(f"{d}  # {format_size(sz)}\n")

    print(f"  [OK] Unreferenced dirs: {unref_list_path}")

    # List 3: Quick-win items (low risk, small sizes)
    quick_win_path = ROOT / "cleanup_quick_wins.txt"
    quick_total = total_root_files + output1_size + total_cp + total_debug
    with open(quick_win_path, "w", encoding="utf-8") as f:
        f.write("# Quick-win cleanup (low risk, no review needed)\n")
        f.write(f"# Total reclaimable: {format_size(quick_total)}\n\n")

        f.write("## Orphan root files\n")
        for item in sorted(root_files):
            f.write(f"{item.relative_to(ROOT).as_posix()}\n")

        f.write("\n## output1/ duplicate\n")
        f.write("output/output1\n")

        f.write("\n## Jupyter checkpoints\n")
        for cp in sorted(checkpoints):
            f.write(f"{cp.relative_to(ROOT).as_posix()}\n")

        f.write("\n## debug_wave files\n")
        for df in sorted(debug_files):
            f.write(f"{df.relative_to(ROOT).as_posix()}\n")

        f.write("\n## Empty directories\n")
        for d in sorted(empty_dirs):
            f.write(f"{d}\n")

    print(f"  [OK] Quick wins: {quick_win_path}")
    print("\nDone! Please review the generated .txt files before executing deletion.")


if __name__ == "__main__":
    main()
