"""Fresh-process RSS and wall-time probe for the P2 bfpy FFT2 backend."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

import bfpy
import numpy as np


def peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def current_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS is unavailable")


def run_probe(args, root: Path) -> None:
    dtype = np.complex64 if args.dtype == "c8" else np.complex128
    function = bfpy.fft2_c8 if args.dtype == "c8" else bfpy.fft2_c16
    infile = root / "input.npy"
    outfile = root / "output.npy"
    scratch = root / "scratch.npy"
    events = []
    pass_rss = []

    def progress(pass_name, completed, total):
        if completed == total:
            events.append((pass_name, completed, total))
            pass_rss.append((pass_name, current_rss_mib()))

    started = time.perf_counter()
    function(
        infile,
        outfile,
        scratch,
        args.nx,
        args.ny,
        progress,
        None,
        args.memory_budget_mib * 1024 * 1024,
    )
    wall_seconds = time.perf_counter() - started
    rss_mib = peak_rss_mib()
    output = np.load(outfile, mmap_mode="r")
    sample = output.reshape(-1)[::max(1, output.size // 4096)]
    checksum = [float(sample.real.sum()), float(sample.imag.sum())]
    print(json.dumps({
        "nx": args.nx,
        "ny": args.ny,
        "dtype": args.dtype,
        "memory_budget_mib": args.memory_budget_mib,
        "peak_rss_mib": round(rss_mib, 3),
        "callback_peak_rss_mib": round(max(value for _, value in pass_rss), 3),
        "rss_at_pass_completion_mib": [
            [name, round(value, 3)] for name, value in pass_rss
        ],
        "wall_seconds": round(wall_seconds, 6),
        "input_bytes": infile.stat().st_size,
        "output_bytes": outfile.stat().st_size,
        "checksum_sample": checksum,
        "completed_passes": [event[0] for event in events],
        "scratch_clean": not scratch.exists(),
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=2048)
    parser.add_argument("--ny", type=int, default=2048)
    parser.add_argument("--dtype", choices=("c8", "c16"), default="c8")
    parser.add_argument("--memory-budget-mib", type=int, default=32)
    parser.add_argument("--worker-directory", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    dtype = np.complex64 if args.dtype == "c8" else np.complex128
    if args.worker_directory is not None:
        run_probe(args, args.worker_directory)
        return

    with tempfile.TemporaryDirectory(prefix="bfpy-p2-") as directory:
        root = Path(directory)
        infile = root / "input.npy"
        field = np.lib.format.open_memmap(
            infile, mode="w+", dtype=dtype, shape=(args.ny, args.nx)
        )
        x = np.arange(args.nx, dtype=np.float64)
        for y_start in range(0, args.ny, 32):
            rows = min(32, args.ny - y_start)
            y = np.arange(y_start, y_start + rows, dtype=np.float64)[:, None]
            field[y_start:y_start + rows] = (
                np.sin(x[None, :] * 0.013 + y * 0.017)
                + 1j * np.cos(x[None, :] * 0.019 - y * 0.011)
            ).astype(dtype)
        field.flush()
        del field
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--nx", str(args.nx),
                "--ny", str(args.ny),
                "--dtype", args.dtype,
                "--memory-budget-mib", str(args.memory_budget_mib),
                "--worker-directory", str(root),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout.strip())


if __name__ == "__main__":
    main()
