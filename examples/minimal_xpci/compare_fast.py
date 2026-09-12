#!/usr/bin/env python3
"""Run the same small 2D sample case with big-wave and fast-wave."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FASTWAVE = ROOT / "fast-wave" / "build-Release" / "fastwave"
sys.path.insert(0, str(ROOT / "big-wave"))

import config  # noqa: E402
import multisim  # noqa: E402


def sphere_grid(size: int = 18, radius: int = 6) -> np.ndarray:
    z, y, x = np.ogrid[:size, :size, :size]
    center = size // 2
    grid = np.zeros((size, size, size), dtype=np.uint32)
    grid[(x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius**2] = 1
    return grid


def simulation_config(grid_path: Path) -> dict:
    nx = ny = 1024
    dx = 3.5e-8
    # The detector stays compact while the wider numerical field acts as the
    # guard band required by the boundary-reflection check.
    detector_size = 2.0e-5
    return {
        "dtype": "c8",
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "sim_params": {
            "N": nx * ny,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dx,
            "z_detector": 0.06,
            "detector_size": detector_size,
            "detector_size_x": detector_size,
            "detector_size_y": detector_size,
            "detector_pixel_size_x": 2 * dx,
            "detector_pixel_size_y": 2 * dx,
            "chunk_size": nx * 16,
        },
        "runtime": {
            "big_wave": {
                "memory_budget_gb": 1.0,
                "chunk_size": nx * 16,
                "fft2_backend": "scipy_in_memory",
                "detector_integrator": "area_v1",
            }
        },
        "multisource": {
            "type": "points",
            "nr_source_points": 1,
            "energy_range": [8_000.0, 8_000.0],
            "x_range": [0.0, 0.0],
            "y_range": [0.0, 0.0],
            "z": 0.0,
            "seed": 42,
        },
        "elements": [
            {
                "type": "sample",
                "z_start": 0.01,
                "pixel_size_x": 2.0e-7,
                "pixel_size_y": 2.0e-7,
                "pixel_size_z": 2.0e-7,
                "grid_path": str(grid_path),
                "materials": [["C", 2.0]],
                "x_positions": [0.0],
                "y_positions": [0.0],
            }
        ],
    }


def relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm((actual - expected).ravel())
        / max(np.linalg.norm(expected.ravel()), np.finfo(float).eps)
    )


def main() -> None:
    if not FASTWAVE.is_file():
        raise FileNotFoundError(f"fast-wave binary not found: {FASTWAVE}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = ROOT / "output" / "minimal_xpci" / f"fast_compare_{stamp}"
    output.mkdir(parents=True)
    grid_path = output / "carbon_sphere.npy"
    np.save(grid_path, sphere_grid())

    prepared = multisim.setup_simulation(
        simulation_config(grid_path), output, output / "prepared"
    )
    big_dir = output / "big_wave"
    fast_dir = output / "fast_wave"
    shutil.copytree(prepared, big_dir)
    shutil.copytree(prepared, fast_dir)

    started = time.perf_counter()
    multisim.run_single_simulation(big_dir, 0, output / "scratch")
    big_seconds = time.perf_counter() - started

    started = time.perf_counter()
    completed = subprocess.run(
        [str(FASTWAVE), str(fast_dir), "-s", "0"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    fast_seconds = time.perf_counter() - started
    (output / "fastwave_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "fastwave_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"fast-wave exited with {completed.returncode}:\n"
            f"{completed.stderr or completed.stdout}"
        )

    big = np.load(big_dir / "00000000" / "detected.npy").squeeze()
    fast = np.load(fast_dir / "00000000" / "detected.npy").squeeze()
    if big.shape != fast.shape:
        raise AssertionError(f"shape mismatch: big-wave {big.shape}, fast-wave {fast.shape}")

    error = relative_l2(fast, big)
    peak_relative = float(np.max(np.abs(fast - big)) / np.max(np.abs(big)))
    np.savez_compressed(
        output / "comparison.npz",
        big_wave=big,
        fast_wave=fast,
        difference=fast - big,
        relative_l2=error,
        peak_relative_error=peak_relative,
    )

    import matplotlib

    # Render deterministically without requiring an X server.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = np.mean(big)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    panels = (big / scale, fast / scale, (fast - big) / scale)
    titles = ("big-wave", "fast-wave", "(fast - big) / mean(big)")
    for axis, image, title in zip(axes, panels, titles):
        view = axis.imshow(image, origin="lower", cmap="inferno")
        axis.set_title(title)
        figure.colorbar(view, ax=axis, fraction=0.046)
    figure.savefig(output / "comparison.png", dpi=140)
    plt.close(figure)

    print(f"shape: {big.shape}")
    print(f"big-wave time: {big_seconds:.3f} s")
    print(f"fast-wave wall time: {fast_seconds:.3f} s")
    print(f"relative L2 error: {error:.9e}")
    print(f"peak relative error: {peak_relative:.9e}")
    print(f"results: {output}")
    print("PASS" if error < 2.0e-3 else "FAIL")
    if error >= 2.0e-3:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
