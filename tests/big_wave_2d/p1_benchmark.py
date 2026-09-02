"""Fresh-process RSS/time probe for P1 row-tiled local operators."""

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

import config  # noqa: F401, E402
import optical_element  # noqa: E402
import plasma_sample  # noqa: E402
import propagation  # noqa: E402
from optical_element import Material, Sample  # noqa: E402
from plasma_sample import PlasmaSample  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import NumpyVector  # noqa: E402


def params(nx, ny, tile_rows):
    return SimParams(
        N=nx * ny,
        nx=nx,
        ny=ny,
        dx=1e-7,
        dy=1e-7,
        z_detector=0.2,
        detector_size=nx * 1e-7,
        detector_size_x=nx * 1e-7,
        detector_size_y=ny * 1e-7,
        detector_pixel_size_x=4e-7,
        detector_pixel_size_y=4e-7,
        wl=propagation.convert_energy_wavelength(8000.0),
        chunk_size=nx * tile_rows,
    )


def memmap(path, shape, dtype, value):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array[:] = value
    array.flush()
    del array
    return np.load(path, mmap_mode="r", allow_pickle=False)


def run(operator, nx, ny, tile_rows):
    simulation = params(nx, ny, tile_rows)
    wave = NumpyVector(np.ones(simulation.N, dtype=np.complex64))
    fourier = NumpyVector(np.zeros(simulation.N, dtype=np.complex64))
    start = time.perf_counter()

    if operator == "analytic":
        propagation.propagate_analytically_2d(wave, 0.1, 0.0, 0.0, simulation)
    else:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grid_shape = (1, ny + 1, nx + 1)
            if operator == "sample":
                grid = memmap(root / "grid.npy", grid_shape, np.uint32, 1)
                element = Sample(
                    z_start=0.1,
                    pixel_size_x=simulation.dx,
                    pixel_size_y=simulation.get_dy(),
                    pixel_size_z=1e-7,
                    grid=grid,
                    materials=[Material("W", 19.35)],
                    x_positions=np.array([0.0]),
                    y_positions=np.array([0.0]),
                )
                element.db_list = np.array(
                    [0.0 + 0.0j, 2.0e-6 + 1.0e-7j], dtype=np.complex128
                )
                old_propagate = optical_element.propagate_2d
                optical_element.propagate_2d = lambda *args, **kwargs: None
                try:
                    element._apply_2d(wave, fourier, simulation, 1e6, 0, None)
                finally:
                    optical_element.propagate_2d = old_propagate
            else:
                ne = memmap(root / "ne.npy", grid_shape, np.float32, 4e21)
                ni = memmap(root / "ni.npy", grid_shape, np.float32, 4e21 / 13.0)
                te = memmap(root / "te.npy", grid_shape, np.float32, 100.0)
                zstar = memmap(root / "zstar.npy", grid_shape, np.float32, 13.0)
                element = PlasmaSample(
                    z_start=0.1,
                    pixel_size_x=simulation.dx,
                    pixel_size_y=simulation.get_dy(),
                    pixel_size_z=1e-7,
                    ne_grid=ne,
                    ni_grid=ni,
                    te_grid=te,
                    zstar_grid=zstar,
                    Z=13,
                    x_positions=np.array([0.0]),
                    y_positions=np.array([0.0]),
                )
                old_propagate = propagation.propagate_2d
                propagation.propagate_2d = lambda *args, **kwargs: None
                try:
                    element._apply_2d(wave, fourier, simulation, 1e6, 0, None)
                finally:
                    propagation.propagate_2d = old_propagate

    elapsed = time.perf_counter() - start
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "operator": operator,
        "nx": nx,
        "ny": ny,
        "tile_rows": tile_rows,
        "dtype": "complex64",
        "elapsed_s": round(elapsed, 6),
        "peak_rss_mib": round(max_rss_kib / 1024.0, 3),
        "checksum": [float(np.real(wave.vec[0])), float(np.imag(wave.vec[0]))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", choices=("analytic", "sample", "plasma"), required=True)
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--tile-rows", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.operator, args.nx, args.ny, args.tile_rows), sort_keys=True))


if __name__ == "__main__":
    main()
