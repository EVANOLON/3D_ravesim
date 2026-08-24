"""Fresh-process RSS and wall-time probe for the P3 streaming detector."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

import config  # noqa: E402,F401  # establish the legacy module import order
import multisim  # noqa: E402
import propagation  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import DiskVector  # noqa: E402


def current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS is unavailable")


def run_probe(args, root: Path) -> None:
    dtype = np.complex64 if args.dtype == "c8" else np.complex128
    infile = root / "input.npy"
    events = []
    samples = []
    process_samples = []
    sampler_stop = threading.Event()

    def sample_process():
        while not sampler_stop.wait(0.005):
            process_samples.append(current_rss_mib())

    def progress(stage, completed, total):
        events.append((stage, completed, total))
        samples.append(current_rss_mib())

    params = SimParams(
        N=args.nx * args.ny,
        nx=args.nx,
        ny=args.ny,
        dx=1.0,
        dy=1.0,
        z_detector=10000.0,
        detector_size=float(min(args.nx, args.ny)),
        detector_size_x=float(args.nx),
        detector_size_y=float(args.ny),
        detector_pixel_size_x=1.0,
        detector_pixel_size_y=1.0,
        wl=1e-10,
        chunk_size=args.nx * args.tile_rows,
        memory_budget_gb=args.memory_budget_mib / 1024.0,
        detector_integrator=args.integrator,
    )
    params.detector_output_dir = str(root)
    params.detector_progress_cb = progress
    vector = DiskVector(
        infile,
        root / "unused-scratch.npy",
        params.N,
        dtype,
    )
    sampler = threading.Thread(target=sample_process, daemon=True)
    sampler.start()
    started = time.perf_counter()
    output = propagation.square_and_downsample_2d(vector, params, params.z_detector)
    detector_seconds = time.perf_counter() - started
    rss_after_detector = current_rss_mib()
    multisim._save_detector_outputs(root / "detected.npy", [output])
    rss_after_save = current_rss_mib()
    wall_seconds = time.perf_counter() - started
    sampler_stop.set()
    sampler.join()
    formal = np.load(root / "detected.npy", mmap_mode="r")
    sample = formal.reshape(-1)[:: max(1, formal.size // 4096)]
    result = {
        "nx": args.nx,
        "ny": args.ny,
        "dtype": args.dtype,
        "integrator": args.integrator,
        "memory_budget_mib": args.memory_budget_mib,
        "tile_rows": args.tile_rows,
        "callback_peak_rss_mib": round(max(samples), 3),
        "process_peak_rss_mib": round(max(process_samples), 3),
        "detector_seconds": round(detector_seconds, 6),
        "rss_after_detector_mib": round(rss_after_detector, 3),
        "rss_after_save_mib": round(rss_after_save, 3),
        "wall_seconds": round(wall_seconds, 6),
        "input_mib": round(infile.stat().st_size / 1024**2, 3),
        "output_mib": round(output.nbytes / 1024**2, 3),
        "output_storage": params.detector_last_metadata["output_storage"],
        "checksum_sample": round(float(sample.sum()), 9),
        "completed_stages": sorted(
            {stage for stage, completed, total in events if completed == total}
        ),
        "rss_by_stage_peak_mib": {
            stage: round(
                max(rss for event, rss in zip(events, samples) if event[0] == stage), 3
            )
            for stage in sorted({event[0] for event in events})
        },
    }
    propagation.cleanup_detector_output(output)
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=4096)
    parser.add_argument("--ny", type=int, default=4096)
    parser.add_argument("--dtype", choices=("c8", "c16"), default="c8")
    parser.add_argument(
        "--integrator", choices=("legacy_fastwave", "area_v1"), default="legacy_fastwave"
    )
    parser.add_argument("--memory-budget-mib", type=int, default=32)
    parser.add_argument("--tile-rows", type=int, default=32)
    parser.add_argument("--worker-directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    dtype = np.complex64 if args.dtype == "c8" else np.complex128
    if args.worker_directory is not None:
        run_probe(args, args.worker_directory)
        return

    with tempfile.TemporaryDirectory(prefix="big-wave-p3-") as directory:
        root = Path(directory)
        field = np.lib.format.open_memmap(
            root / "input.npy", mode="w+", dtype=dtype, shape=(args.ny * args.nx,)
        )
        x = np.arange(args.nx, dtype=np.float64)
        for y_start in range(0, args.ny, 32):
            rows = min(32, args.ny - y_start)
            y = np.arange(y_start, y_start + rows, dtype=np.float64)[:, None]
            block = (
                np.sin(x[None, :] * 0.013 + y * 0.017)
                + 1j * np.cos(x[None, :] * 0.019 - y * 0.011)
            ).astype(dtype)
            field[y_start * args.nx : (y_start + rows) * args.nx] = block.reshape(-1)
        field.flush()
        del field
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--nx",
                str(args.nx),
                "--ny",
                str(args.ny),
                "--dtype",
                args.dtype,
                "--integrator",
                args.integrator,
                "--memory-budget-mib",
                str(args.memory_budget_mib),
                "--tile-rows",
                str(args.tile_rows),
                "--worker-directory",
                str(root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
