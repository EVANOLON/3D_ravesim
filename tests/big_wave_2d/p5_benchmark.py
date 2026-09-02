"""Fresh-process P5 end-to-end 2D staging benchmark.

This intentionally exercises PointSource -> OOC FFT2/IFFT2 -> optional local
operator -> propagation -> streaming detector -> atomic result save.  The
parent samples Linux RSS, VmSwap and process I/O while the worker runs.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import bfpy
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

import config  # noqa: E402,F401
import multisim  # noqa: E402
from optical_element import Material, Sample  # noqa: E402
from plasma_sample import PlasmaSample  # noqa: E402
from propagation import SimParams, cleanup_detector_output  # noqa: E402
from source import PointSource  # noqa: E402
from vector import DiskVector  # noqa: E402
import wavesim  # noqa: E402


def proc_status(pid: int) -> dict[str, int]:
    result = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSwap:")):
                result[line.split(":", 1)[0]] = int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return result


def proc_io(pid: int) -> dict[str, int]:
    result = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def system_swap_used() -> int:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("SwapTotal:", "SwapFree:")):
            values[line.split(":", 1)[0]] = int(line.split()[1]) * 1024
    return values.get("SwapTotal", 0) - values.get("SwapFree", 0)


def make_element(mode: str, nx: int, ny: int, dx: float, z_start: float):
    if mode == "vacuum":
        return [], []
    grid_nx = grid_ny = 258
    # Keep the physical object identical across 4096/8192/16384. Changing the
    # object extent with N would make an apparent "convergence" comparison
    # scientifically meaningless even though detector coordinates match.
    object_span_x = 4e-4
    object_span_y = 4e-4
    pixel_x = object_span_x / (grid_nx - 2)
    pixel_y = object_span_y / (grid_ny - 2)
    if mode == "sample":
        material = Material("P5-benchmark", 1.0)
        element = Sample(
            z_start=z_start,
            pixel_size_x=pixel_x,
            pixel_size_y=pixel_y,
            pixel_size_z=1e-6,
            grid=np.ones((1, grid_ny, grid_nx), dtype=np.uint32),
            materials=[material],
            x_positions=np.array([0.0]),
            y_positions=np.array([0.0]),
        )
        return [element], [(material, 1.2e-7 + 2.4e-9j)]
    shape = (4, grid_ny, grid_nx)
    ne = np.full(shape, 2e19, dtype=np.float32)
    ni = np.full(shape, 1e19, dtype=np.float32)
    te = np.full(shape, 200.0, dtype=np.float32)
    zstar = np.full(shape, 2.0, dtype=np.float32)
    element = PlasmaSample(
        z_start=z_start,
        pixel_size_x=pixel_x,
        pixel_size_y=pixel_y,
        pixel_size_z=1e-6,
        ne_grid=ne,
        ni_grid=ni,
        te_grid=te,
        zstar_grid=zstar,
        Z=6,
        x_positions=np.array([0.0]),
        y_positions=np.array([0.0]),
    )
    return [element], []


def run_worker(args, root: Path) -> None:
    dtype = np.dtype(np.complex64)
    nx, ny = args.nx, args.ny
    n = nx * ny
    vectors = root / "vectors"
    output_dir = root / "output"
    vectors.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    bfpy.generate_header_c8(vectors / "u.npy", n)
    bfpy.generate_header_c8(vectors / "spectrum.npy", n)

    events = []
    started = time.perf_counter()

    def progress(stage, completed, total):
        if completed == total:
            events.append([stage, round(time.perf_counter() - started, 6)])

    budget_bytes = args.memory_budget_mib * 1024**2
    common = dict(
        scratchfile=vectors / "scratch.npy",
        len=n,
        dtype=dtype,
        fft2_backend="bfpy_ooc",
        fft2_memory_budget_bytes=budget_bytes,
        fft2_progress_cb=progress,
    )
    u = DiskVector(file=vectors / "u.npy", **common)
    U = DiskVector(file=vectors / "spectrum.npy", **common)

    dx = args.simulation_fov / nx if args.simulation_fov > 0 else 2e-7
    wavelength = 1e-9
    detector_size = 4e-4
    detector_pixel = 8e-7
    p = SimParams(
        N=n,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dx,
        z_detector=10.0,
        detector_size=detector_size,
        detector_size_x=detector_size,
        detector_size_y=detector_size,
        detector_pixel_size_x=detector_pixel,
        detector_pixel_size_y=detector_pixel,
        wl=wavelength,
        chunk_size=nx * args.tile_rows,
        memory_budget_gb=args.memory_budget_mib / 1024.0,
        fft2_backend="bfpy_ooc",
        detector_integrator=args.integrator,
    )
    p.detector_output_dir = str(output_dir)
    p.detector_progress_cb = progress
    elements, table = make_element(args.mode, nx, ny, dx, 5.0)
    cutoff_frequency = (
        args.cutoff_frequency if args.cutoff_frequency > 0 else 0.2 / dx
    )
    cutoff_angle = math.asin(cutoff_frequency * wavelength)
    detector_outputs = wavesim.run_simulation(
        params=p,
        source=PointSource(x=1e-6, y=-1.4e-6, z=0.0),
        elements=elements,
        cutoff_angles=[cutoff_angle] * (len(elements) + 1),
        u=u,
        U=U,
        deltabeta_table=table,
        sub_dir=output_dir,
        vectors_dir=vectors,
        save_final_u_vectors=False,
    )
    multisim._save_detector_outputs(output_dir / "detected.npy", detector_outputs)
    for output in detector_outputs:
        cleanup_detector_output(output)
    detected = np.load(output_dir / "detected.npy", mmap_mode="r")
    image = detected[0]
    fingerprint_width = min(256, image.shape[1])
    fingerprint_start = (image.shape[1] - fingerprint_width) // 2
    centerline = image[
        image.shape[0] // 2,
        fingerprint_start:fingerprint_start + fingerprint_width,
    ]
    result = {
        "mode": args.mode,
        "nx": nx,
        "ny": ny,
        "dtype": "c8",
        "integrator": args.integrator,
        "memory_budget_mib": args.memory_budget_mib,
        "tile_rows": args.tile_rows,
        "dx": dx,
        "simulation_fov": nx * dx,
        "cutoff_frequency": cutoff_frequency,
        "wall_seconds_worker": round(time.perf_counter() - started, 6),
        "input_field_mib": round((n * dtype.itemsize) / 1024**2, 3),
        "detected_shape": list(detected.shape),
        "detected_energy": float(np.sum(image)),
        "detected_peak": float(np.max(image)),
        "centerline_256": [float(item) for item in centerline],
        "centerline_start_pixel": fingerprint_start,
        "completed_events": events,
        "detector_metadata": p.detector_last_metadata,
        "scratch_clean": not (vectors / "scratch.npy").exists(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def run_parent(args) -> dict:
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p5-stage-", dir=work_root) as directory:
        root = Path(directory)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-directory", str(root),
            "--mode", args.mode,
            "--nx", str(args.nx),
            "--ny", str(args.ny),
            "--memory-budget-mib", str(args.memory_budget_mib),
            "--tile-rows", str(args.tile_rows),
            "--integrator", args.integrator,
            "--simulation-fov", str(args.simulation_fov),
            "--cutoff-frequency", str(args.cutoff_frequency),
        ]
        swap_before = system_swap_used()
        started = time.perf_counter()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        peak_rss = peak_swap = 0
        last_io = {}
        while process.poll() is None:
            status = proc_status(process.pid)
            peak_rss = max(peak_rss, status.get("VmRSS", 0))
            peak_swap = max(peak_swap, status.get("VmSwap", 0))
            current_io = proc_io(process.pid)
            if current_io:
                last_io = current_io
            time.sleep(0.05)
        stdout, stderr = process.communicate()
        wall = time.perf_counter() - started
        if process.returncode != 0:
            return {
                "status": "failed",
                "returncode": process.returncode,
                "stderr": stderr[-4000:],
                "stdout": stdout[-4000:],
                "wall_seconds": round(wall, 6),
                "peak_rss_mib": round(peak_rss / 1024**2, 3),
                "peak_vm_swap_mib": round(peak_swap / 1024**2, 3),
            }
        worker = json.loads(stdout.strip().splitlines()[-1])
        worker.update(
            {
                "status": "passed",
                "wall_seconds_parent": round(wall, 6),
                "peak_rss_mib": round(peak_rss / 1024**2, 3),
                "peak_vm_swap_mib": round(peak_swap / 1024**2, 3),
                "system_swap_delta_mib": round((system_swap_used() - swap_before) / 1024**2, 3),
                "io_read_mib": round(last_io.get("read_bytes", 0) / 1024**2, 3),
                "io_write_mib": round(last_io.get("write_bytes", 0) / 1024**2, 3),
                "worker_stderr_tail": stderr[-2000:],
            }
        )
        return worker


def compare_results(paths: list[Path]) -> dict:
    """Compare staged grids only when their physical detector domain matches."""
    if len(paths) < 2:
        raise ValueError("at least two result files are required for convergence")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    results.sort(key=lambda item: (item["nx"], item["ny"]))
    reference = results[0]
    exact_keys = (
        "mode", "simulation_fov", "cutoff_frequency", "integrator",
        "detected_shape", "centerline_start_pixel",
    )
    for result in results:
        if result.get("status") != "passed":
            raise ValueError("all convergence inputs must have status=passed")
        for key in exact_keys:
            if result.get(key) != reference.get(key):
                raise ValueError(
                    f"cannot compare different physical configurations: {key} "
                    f"{reference.get(key)!r} != {result.get(key)!r}"
                )
        ref_geometry = reference["detector_metadata"]["effective_geometry"]
        geometry = result["detector_metadata"]["effective_geometry"]
        for key in ("detector_pixel_size_x", "detector_pixel_size_y", "current_z"):
            if geometry.get(key) != ref_geometry.get(key):
                raise ValueError(f"cannot compare different detector geometry: {key}")
        if len(result["centerline_256"]) != len(reference["centerline_256"]):
            raise ValueError("centerline fingerprints must use the same detector coordinates")

    comparisons = []
    for coarse, fine in zip(results, results[1:]):
        coarse_line = np.asarray(coarse["centerline_256"], dtype=np.float64)
        fine_line = np.asarray(fine["centerline_256"], dtype=np.float64)
        denominator = max(np.linalg.norm(fine_line), np.finfo(float).eps)
        comparisons.append(
            {
                "coarse": [coarse["ny"], coarse["nx"]],
                "fine": [fine["ny"], fine["nx"]],
                "centerline_relative_l2": float(
                    np.linalg.norm(fine_line - coarse_line) / denominator
                ),
                "energy_relative_difference": float(
                    abs(fine["detected_energy"] - coarse["detected_energy"])
                    / max(abs(fine["detected_energy"]), np.finfo(float).eps)
                ),
                "peak_relative_difference": float(
                    abs(fine["detected_peak"] - coarse["detected_peak"])
                    / max(abs(fine["detected_peak"]), np.finfo(float).eps)
                ),
            }
        )
    return {
        "status": "passed",
        "mode": reference["mode"],
        "simulation_fov": reference["simulation_fov"],
        "cutoff_frequency": reference["cutoff_frequency"],
        "integrator": reference["integrator"],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vacuum", "sample", "plasma"), default="vacuum")
    parser.add_argument("--nx", type=int, default=4096)
    parser.add_argument("--ny", type=int, default=4096)
    parser.add_argument("--memory-budget-mib", type=int, default=512)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--integrator", choices=("legacy_fastwave", "area_v1"), default="legacy_fastwave")
    parser.add_argument(
        "--simulation-fov", type=float, default=0.0,
        help="fixed physical simulation width/height; zero keeps dx=2e-7",
    )
    parser.add_argument(
        "--cutoff-frequency", type=float, default=0.0,
        help="fixed physical cutoff for convergence; zero derives 0.2/dx",
    )
    parser.add_argument("--work-root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--compare-results", nargs="+", type=Path, default=None,
        help="compare two or more benchmark JSON files on an identical physical domain",
    )
    parser.add_argument("--worker-directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compare_results is not None:
        result = compare_results(args.compare_results)
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return
    if args.nx <= 0 or args.ny <= 0 or args.nx * args.ny <= 0:
        raise ValueError("nx and ny must be positive")
    if args.worker_directory is not None:
        run_worker(args, args.worker_directory)
        return
    result = run_parent(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
