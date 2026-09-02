"""P1 row-tile, mmap and vectorized-plasma acceptance tests."""

import inspect
import math
from pathlib import Path
import sys
import tempfile
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
for path in (BIG_WAVE, ROOT / "rave_agent"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
import multisim  # noqa: E402
import optical_element  # noqa: E402
import plasma  # noqa: E402
import plasma_sample  # noqa: E402
import propagation  # noqa: E402
from optical_element import Material, Sample  # noqa: E402
from plasma_sample import PlasmaSample  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import NumpyVector  # noqa: E402


def make_params(nx, ny, tile_rows, dtype=np.complex64):
    return SimParams(
        N=nx * ny,
        nx=nx,
        ny=ny,
        dx=1e-7,
        dy=1.3e-7,
        z_detector=0.1,
        detector_size=nx * 1e-7,
        detector_size_x=nx * 1e-7,
        detector_size_y=ny * 1.3e-7,
        detector_pixel_size_x=4e-7,
        detector_pixel_size_y=5.2e-7,
        wl=propagation.convert_energy_wavelength(8000.0),
        chunk_size=nx * tile_rows,
    )


def sample_for_grid(grid):
    sample = Sample(
        z_start=0.01,
        pixel_size_x=1.1e-7,
        pixel_size_y=1.4e-7,
        pixel_size_z=2e-7,
        grid=grid,
        materials=[Material("W", 19.35)],
        x_positions=np.array([0.2e-7]),
        y_positions=np.array([-0.3e-7]),
    )
    sample.db_list = np.array([0.0 + 0.0j, 2.5e-6 + 1.1e-7j], dtype=np.complex128)
    return sample


def plasma_for_grids(ne, ni, te, zs):
    return PlasmaSample(
        z_start=0.01,
        pixel_size_x=1.1e-7,
        pixel_size_y=1.4e-7,
        pixel_size_z=2e-7,
        ne_grid=ne,
        ni_grid=ni,
        te_grid=te,
        zstar_grid=zs,
        Z=13,
        x_positions=np.array([0.2e-7]),
        y_positions=np.array([-0.3e-7]),
    )


class TestP1PropagationTiles(unittest.TestCase):
    def test_row_alignment_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "complete rows"):
            propagation.require_row_aligned_chunk_size(17, 16, 8)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            propagation.require_row_aligned_chunk_size(16 * 9, 16, 8)

    def test_analytical_source_is_tile_rows_invariant_for_c8_and_c16(self):
        for dtype in (np.complex64, np.complex128):
            results = []
            for rows in (1, 3, 8):
                params = make_params(32, 24, rows, dtype)
                vector = NumpyVector(np.zeros(params.N, dtype=dtype))
                propagation.propagate_analytically_2d(
                    vector, 0.02, 0.3e-6, -0.2e-6, params
                )
                self.assertEqual(vector.vec.dtype, dtype)
                results.append(vector.vec.copy())
            for result in results[1:]:
                np.testing.assert_array_equal(result, results[0])

    def test_cutoff_and_propagation_are_tile_rows_invariant(self):
        nx, ny = 32, 24
        rng = np.random.default_rng(42)
        initial = (
            rng.normal(size=nx * ny) + 1j * rng.normal(size=nx * ny)
        ).astype(np.complex64)

        cutoff_results = []
        propagation_results = []
        for rows in (1, 4, 8):
            params = make_params(nx, ny, rows)
            cutoff = NumpyVector(initial.copy())
            propagation.apply_frequency_cutoff_2d(
                cutoff, 2.0e6, params.dx, params.get_dy(), nx, ny, params.chunk_size
            )
            cutoff_results.append(cutoff.vec.copy())

            u = NumpyVector(initial.copy())
            U = NumpyVector(np.zeros_like(initial))
            propagation.propagate_2d(
                u,
                U,
                params.dx,
                params.get_dy(),
                params.wl,
                0.003,
                params.chunk_size,
                2.0e6,
                nx,
                ny,
            )
            propagation_results.append(u.vec.copy())

        for result in cutoff_results[1:]:
            np.testing.assert_array_equal(result, cutoff_results[0])
        for result in propagation_results[1:]:
            np.testing.assert_allclose(result, propagation_results[0], rtol=0, atol=2e-7)

    def test_operator_source_contains_no_flat_ix_iy_chunk_expansion(self):
        for function in (
            propagation.propagate_analytically_2d,
            propagation.propagate_2d,
            propagation.apply_frequency_cutoff_2d,
        ):
            source = inspect.getsource(function)
            self.assertNotIn("flat = idx + np.arange", source)

    def test_analytical_temporary_memory_scales_with_tile_rows(self):
        def peak_for(tile_rows):
            params = make_params(1024, 1024, tile_rows)
            vector = NumpyVector(np.zeros(params.N, dtype=np.complex64))
            tracemalloc.start()
            propagation.propagate_analytically_2d(vector, 0.2, 0.0, 0.0, params)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        small = peak_for(8)
        large = peak_for(128)
        print(f"P1 analytic tile peaks: rows=8 {small / 2**20:.2f} MiB, rows=128 {large / 2**20:.2f} MiB")
        self.assertLess(small, 8 * 2**20)
        self.assertGreater(large, small * 3)


class TestP1SampleAndMmap(unittest.TestCase):
    def test_setup_records_p1_algorithm_metadata(self):
        dct = {
            "dtype": "c8",
            "use_disk_vector": True,
            "save_final_u_vectors": False,
            "sim_params": {
                "N": 64 * 64,
                "nx": 64,
                "ny": 64,
                "dx": 1e-6,
                "dy": 1e-6,
                "z_detector": 0.1,
                "detector_size": 32e-6,
                "detector_size_x": 32e-6,
                "detector_size_y": 32e-6,
                "detector_pixel_size_x": 4e-6,
                "detector_pixel_size_y": 4e-6,
                "chunk_size": "auto",
            },
            "runtime": {"big_wave": {"memory_budget_gb": 1.0, "chunk_size": "auto"}},
            "multisource": {
                "type": "points",
                "nr_source_points": 1,
                "energy_range": [1000.0, 1000.0],
                "x_range": [0.0, 0.0],
                "z": 0.0,
                "seed": 1,
            },
            "elements": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(dct, root, root / "runs")
            computed = config.load(sim_dir / "computed.yaml")
        self.assertEqual(computed["provenance"]["algorithm_version"], "big-wave-2d-p4-v1")
        self.assertEqual(computed["algorithms"]["local_operators_2d"]["version"], "row_tile_v1")
        self.assertEqual(computed["algorithms"]["plasma_deltabeta"]["version"], "vectorized_tile_v1")
        self.assertEqual(
            computed["runtime"]["big_wave"]["resolved"]["tile_rows"], 64
        )

    def test_config_opens_sample_and_plasma_grids_as_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            material = np.lib.format.open_memmap(
                root / "material.npy", mode="w+", dtype=np.uint32, shape=(1, 4, 4)
            )
            material[:] = 0
            del material
            for name, value in (("ne", 1e20), ("ni", 1e19), ("te", 100.0), ("zs", 13.0)):
                array = np.lib.format.open_memmap(
                    root / f"{name}.npy", mode="w+", dtype=np.float32, shape=(1, 4, 4)
                )
                array[:] = value
                del array

            sample = config.parse_optical_element(
                {
                    "type": "sample",
                    "z_start": 0.1,
                    "pixel_size_x": 1e-7,
                    "pixel_size_y": 1e-7,
                    "pixel_size_z": 1e-7,
                    "grid_path": "material.npy",
                    "materials": [["W", 19.35]],
                    "x_positions": [0.0],
                    "y_positions": [0.0],
                },
                root,
            )
            plasma_element = config.parse_optical_element(
                {
                    "type": "plasma_sample",
                    "z_start": 0.1,
                    "pixel_size_x": 1e-7,
                    "pixel_size_y": 1e-7,
                    "pixel_size_z": 1e-7,
                    "ne_grid_path": "ne.npy",
                    "ni_grid_path": "ni.npy",
                    "te_grid_path": "te.npy",
                    "zstar_grid_path": "zs.npy",
                    "Z": 13,
                    "x_positions": [0.0],
                    "y_positions": [0.0],
                },
                root,
            )
            self.assertIsInstance(sample.grid, np.memmap)
            self.assertFalse(sample.grid.flags.writeable)
            for grid in (
                plasma_element.ne_grid,
                plasma_element.ni_grid,
                plasma_element.te_grid,
                plasma_element.zstar_grid,
            ):
                self.assertIsInstance(grid, np.memmap)
                self.assertFalse(grid.flags.writeable)

    def test_sample_is_tile_rows_invariant_and_matches_full_reference(self):
        nx, ny = 24, 16
        grid = np.zeros((1, 13, 19), dtype=np.uint32)
        grid[:, 2:11, 3:16] = 1
        outputs = []
        with mock.patch.object(optical_element, "propagate_2d", return_value=None):
            for rows in (1, 4, 8):
                params = make_params(nx, ny, rows)
                sample = sample_for_grid(grid)
                vector = NumpyVector(np.ones(params.N, dtype=np.complex64))
                U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
                sample._apply_2d(vector, U, params, 1e6, 0, None)
                outputs.append(vector.vec.copy())
        for output in outputs[1:]:
            np.testing.assert_array_equal(output, outputs[0])

        # Independent full-field reference reproducing fast-wave's bilinear
        # mapping and vacuum-outside-grid behavior.
        params = make_params(nx, ny, ny)
        sample = sample_for_grid(grid)
        iy, ix = np.indices((ny, nx))
        x_idx = (
            (ix - nx / 2.0) * params.dx
            + sample.x_positions[0]
            + grid.shape[2] * sample.pixel_size_x * 0.5
        ) / sample.pixel_size_x
        y_idx = (
            (iy - ny / 2.0) * params.get_dy()
            + sample.y_positions[0]
            + grid.shape[1] * sample.pixel_size_y * 0.5
        ) / sample.pixel_size_y
        xf_raw = np.floor(x_idx).astype(np.int64)
        yf_raw = np.floor(y_idx).astype(np.int64)
        xf = np.clip(xf_raw, 0, grid.shape[2] - 2)
        yf = np.clip(yf_raw, 0, grid.shape[1] - 2)
        xfrac = x_idx - xf_raw
        yfrac = y_idx - yf_raw
        row_db = sample.db_list[grid[0]]
        interp = (
            (row_db[yf, xf] * (1 - xfrac) + row_db[yf, xf + 1] * xfrac) * (1 - yfrac)
            + (row_db[yf + 1, xf] * (1 - xfrac) + row_db[yf + 1, xf + 1] * xfrac) * yfrac
        )
        inside = (
            (x_idx >= 0) & (x_idx < grid.shape[2] - 1)
            & (y_idx >= 0) & (y_idx < grid.shape[1] - 1)
        )
        interp[~inside] = 0
        expected = optical_element.material_factor(
            interp, sample.pixel_size_z, params.wl
        ).astype(np.complex64).reshape(-1)
        np.testing.assert_allclose(outputs[0], expected, rtol=2e-6, atol=2e-7)


class TestP1PlasmaVectorization(unittest.TestCase):
    def test_vectorized_plasma_matches_scalar_formula(self):
        rng = np.random.default_rng(7)
        ne = rng.uniform(0, 5e21, size=(7, 9))
        ni = ne / 13.0
        te = rng.uniform(20, 500, size=ne.shape)
        zs = np.full(ne.shape, 13.0)
        energy = 8000.0
        delta, beta, atlen = plasma.plasma_delta_beta_grid(ne, ni, te, zs, 13, energy)
        expected = np.empty(ne.shape + (3,), dtype=np.float64)
        for index in np.ndindex(ne.shape):
            expected[index] = plasma.plasma_delta_beta(
                float(ne[index]),
                float(ni[index]),
                float(te[index]),
                float(zs[index]),
                13,
                energy,
            )
        np.testing.assert_allclose(delta, expected[..., 0], rtol=2e-15, atol=0)
        np.testing.assert_allclose(beta, expected[..., 1], rtol=2e-15, atol=0)
        np.testing.assert_allclose(atlen, expected[..., 2], rtol=2e-15, atol=0)

    def test_plasma_sample_is_tile_rows_invariant(self):
        sim_nx, sim_ny = 20, 16
        shape = (1, 13, 17)
        y, x = np.indices(shape[1:])
        ne2 = (1e21 + (x + y) * 2e19).astype(np.float32)
        ne = ne2[None, ...]
        ni = (ne / 13.0).astype(np.float32)
        te = np.full(shape, 120.0, dtype=np.float32)
        zs = np.full(shape, 13.0, dtype=np.float32)
        outputs = []
        with mock.patch.object(plasma_sample, "plasma_delta_beta", side_effect=AssertionError("scalar path used"), create=True), \
             mock.patch("propagation.propagate_2d", return_value=None):
            # _apply_2d imports propagate_2d locally; patch the defining module.
            for rows in (1, 4, 8):
                params = make_params(sim_nx, sim_ny, rows)
                element = plasma_for_grids(ne, ni, te, zs)
                vector = NumpyVector(np.ones(params.N, dtype=np.complex64))
                U = NumpyVector(np.zeros(params.N, dtype=np.complex64))
                element._apply_2d(vector, U, params, 1e6, 0, None)
                outputs.append(vector.vec.copy())
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], rtol=0, atol=2e-7)

    def test_vectorized_slice_is_faster_than_scalar_loop(self):
        shape = (96, 96)
        ne = np.full(shape, 4e21, dtype=np.float64)
        ni = ne / 13.0
        te = np.full(shape, 100.0)
        zs = np.full(shape, 13.0)
        start = time.perf_counter()
        plasma.plasma_delta_beta_grid(ne, ni, te, zs, 13, 8000.0)
        vector_time = time.perf_counter() - start

        start = time.perf_counter()
        for index in np.ndindex(shape):
            plasma.plasma_delta_beta(
                ne[index], ni[index], te[index], zs[index], 13, 8000.0
            )
        scalar_time = time.perf_counter() - start
        print(
            f"P1 plasma slice 96x96: vector={vector_time:.4f}s, "
            f"scalar={scalar_time:.4f}s, speedup={scalar_time / vector_time:.1f}x"
        )
        self.assertLess(vector_time, scalar_time * 0.5)


if __name__ == "__main__":
    unittest.main()
