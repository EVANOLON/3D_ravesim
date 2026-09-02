"""P3 streaming 2D detector acceptance tests."""

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
RAVE_AGENT = ROOT / "rave_agent"
for path in (BIG_WAVE, RAVE_AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
import multisim  # noqa: E402
import propagation  # noqa: E402
import validate_sim  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import NumpyVector  # noqa: E402


def detector_params(
    nx=11,
    ny=7,
    dx=0.7,
    dy=1.1,
    detector_size_x=5.2,
    detector_size_y=5.1,
    detector_pixel_size_x=1.3,
    detector_pixel_size_y=1.7,
    integrator="legacy_fastwave",
    memory_budget_gb=0.0,
):
    return SimParams(
        N=nx * ny,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        z_detector=8.0,
        detector_size=detector_size_x,
        detector_size_x=detector_size_x,
        detector_size_y=detector_size_y,
        detector_pixel_size_x=detector_pixel_size_x,
        detector_pixel_size_y=detector_pixel_size_y,
        wl=1e-10,
        chunk_size=2 * nx,
        memory_budget_gb=memory_budget_gb,
        detector_integrator=integrator,
    )


def legacy_reference(field, params, current_z):
    nx, ny = params.nx, params.ny
    dx, dy = params.dx, params.get_dy()
    ds_x, ds_y = params.detector_pixel_size_x, params.detector_pixel_size_y
    out_x = int(params.get_detector_size_x() // ds_x)
    out_y = int(params.get_detector_size_y() // ds_y)
    result = np.zeros((out_y, out_x), dtype=np.float64)
    for q in range(out_y):
        y = (q - out_y // 2) * ds_y
        lo_y = max(math.trunc((y - ds_y / 2) / dy) + ny // 2, 0)
        hi_y = min(math.trunc((y + ds_y / 2) / dy) + ny // 2, ny)
        for p in range(out_x):
            x = (p - out_x // 2) * ds_x
            lo_x = max(math.trunc((x - ds_x / 2) / dx) + nx // 2, 0)
            hi_x = min(math.trunc((x + ds_x / 2) / dx) + nx // 2, nx)
            value = 0.0
            for j in range(lo_y, hi_y):
                for i in range(lo_x, hi_x):
                    sample = field[j, i]
                    value += sample.real * sample.real + sample.imag * sample.imag
            cosine = current_z / math.sqrt(x * x + y * y + current_z * current_z)
            result[q, p] = value * dx * dy * cosine
    return result


def area_reference(field, params, current_z):
    nx, ny = params.nx, params.ny
    dx, dy = params.dx, params.get_dy()
    ds_x, ds_y = params.detector_pixel_size_x, params.detector_pixel_size_y
    out_x = int(params.get_detector_size_x() // ds_x)
    out_y = int(params.get_detector_size_y() // ds_y)
    grid_left_x = (-(nx // 2) - 0.5) * dx
    grid_left_y = (-(ny // 2) - 0.5) * dy
    result = np.zeros((out_y, out_x), dtype=np.float64)
    for q in range(out_y):
        y = (q - out_y // 2) * ds_y
        detector_lo_y, detector_hi_y = y - ds_y / 2, y + ds_y / 2
        for p in range(out_x):
            x = (p - out_x // 2) * ds_x
            detector_lo_x, detector_hi_x = x - ds_x / 2, x + ds_x / 2
            value = 0.0
            for j in range(ny):
                cell_lo_y = grid_left_y + j * dy
                overlap_y = max(
                    0.0,
                    min(detector_hi_y, cell_lo_y + dy)
                    - max(detector_lo_y, cell_lo_y),
                )
                for i in range(nx):
                    cell_lo_x = grid_left_x + i * dx
                    overlap_x = max(
                        0.0,
                        min(detector_hi_x, cell_lo_x + dx)
                        - max(detector_lo_x, cell_lo_x),
                    )
                    sample = field[j, i]
                    value += (
                        (sample.real * sample.real + sample.imag * sample.imag)
                        * overlap_x
                        * overlap_y
                    )
            cosine = current_z / math.sqrt(x * x + y * y + current_z * current_z)
            result[q, p] = value * cosine
    return result


def p3_config(integrator="area_v1", nx=64, ny=32):
    return {
        "dtype": "c8",
        "use_disk_vector": True,
        "save_final_u_vectors": False,
        "sim_params": {
            "N": nx * ny,
            "nx": nx,
            "ny": ny,
            "dx": 1e-7,
            "dy": 1e-7,
            "z_detector": 0.1,
            "detector_size": min(nx, ny) * 0.5e-7,
            "detector_size_x": nx * 0.5e-7,
            "detector_size_y": ny * 0.5e-7,
            "detector_pixel_size_x": 4e-7,
            "detector_pixel_size_y": 4e-7,
            "chunk_size": "auto",
        },
        "runtime": {
            "big_wave": {
                "memory_budget_gb": 1.0,
                "chunk_size": "auto",
                "fft2_backend": "bfpy_ooc",
                "detector_integrator": integrator,
            }
        },
        "multisource": {
            "type": "points",
            "nr_source_points": 1,
            "energy_range": [8000.0, 8000.0],
            "x_range": [0.0, 0.0],
            "z": 0.0,
            "seed": 1,
        },
        "elements": [],
    }


class TestLegacyFastwaveStreaming(unittest.TestCase):
    def test_matches_direct_cuda_semantics_without_full_field_allocations(self):
        params = detector_params()
        rng = np.random.default_rng(19)
        field = rng.normal(size=(params.ny, params.nx)) + 1j * rng.normal(
            size=(params.ny, params.nx)
        )
        expected = legacy_reference(field, params, current_z=8.0)
        original_empty = np.empty

        def guarded_empty(shape, *args, **kwargs):
            if shape in (params.N, (params.ny + 1, params.nx + 1)):
                raise AssertionError(f"forbidden full-field detector allocation: {shape}")
            return original_empty(shape, *args, **kwargs)

        with mock.patch.object(propagation.np, "empty", side_effect=guarded_empty), self.assertWarns(
            RuntimeWarning
        ):
            actual = propagation.square_and_downsample_2d(
                NumpyVector(field.reshape(-1).astype(np.complex128)), params, 8.0
            )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
        self.assertEqual(params.detector_last_metadata["integrator"], "legacy_fastwave")
        self.assertTrue(params.detector_last_metadata["count_map"]["non_uniform"])

    def test_non_integer_ratio_and_zero_count_are_reported(self):
        params = detector_params(
            nx=4,
            ny=4,
            dx=1.0,
            dy=1.0,
            detector_size_x=4.0,
            detector_size_y=4.0,
            detector_pixel_size_x=0.5,
            detector_pixel_size_y=0.5,
        )
        field = np.ones((4, 4), dtype=np.complex64)
        with self.assertWarnsRegex(RuntimeWarning, "non-integer"):
            propagation.square_and_downsample_2d(NumpyVector(field.reshape(-1)), params, 3.0)
        metadata = params.detector_last_metadata
        self.assertGreater(metadata["count_map"]["zero_pixels"], 0)
        self.assertTrue(metadata["non_integer_pixel_ratio"]["x"])


class TestAreaV1Streaming(unittest.TestCase):
    def test_matches_brute_force_overlap_for_non_integer_ratios(self):
        params = detector_params(integrator="area_v1")
        rng = np.random.default_rng(23)
        field = rng.normal(size=(params.ny, params.nx)) + 1j * rng.normal(
            size=(params.ny, params.nx)
        )
        expected = area_reference(field, params, current_z=8.0)
        actual = propagation.square_and_downsample_2d(
            NumpyVector(field.reshape(-1).astype(np.complex128)), params, 8.0
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)

    def test_uniform_field_preserves_area_for_non_integer_ratio(self):
        params = detector_params(
            detector_size_x=3.9,
            detector_size_y=3.4,
            integrator="area_v1",
        )
        field = np.ones((params.ny, params.nx), dtype=np.complex64)
        actual = propagation.square_and_downsample_2d(
            NumpyVector(field.reshape(-1)), params, 8.0
        )
        out_y, out_x = actual.shape
        x = (np.arange(out_x) - out_x // 2) * params.detector_pixel_size_x
        y = (np.arange(out_y) - out_y // 2) * params.detector_pixel_size_y
        cosine = np.empty_like(actual)
        for q, y_value in enumerate(y):
            cosine[q] = 8.0 / np.sqrt(x * x + y_value * y_value + 64.0)
        np.testing.assert_allclose(
            actual / cosine,
            params.detector_pixel_size_x * params.detector_pixel_size_y,
            rtol=2e-7,
            atol=2e-7,
        )

    def test_oversized_output_uses_removable_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            params = detector_params(
                integrator="area_v1", memory_budget_gb=1e-9
            )
            params.detector_output_dir = directory
            field = np.ones((params.ny, params.nx), dtype=np.complex64)
            output = propagation.square_and_downsample_2d(
                NumpyVector(field.reshape(-1)), params, 8.0
            )
            self.assertIsInstance(output, np.memmap)
            filename = Path(output.filename)
            self.assertTrue(filename.exists())
            self.assertEqual(params.detector_last_metadata["output_storage"], "memmap")
            expected = np.asarray(output).copy()
            formal = Path(directory) / "detected.npy"
            multisim._save_detector_outputs(formal, [output])
            np.testing.assert_array_equal(np.load(formal)[0], expected)
            self.assertFalse((Path(directory) / "detected.npy.part").exists())
            propagation.cleanup_detector_output(output)
            self.assertFalse(filename.exists())


class TestP3RuntimeIntegration(unittest.TestCase):
    def test_fresnel_effective_detector_geometry_is_recorded_and_used(self):
        params = detector_params()
        params.use_fresnel_scaling = True
        params.configure_fresnel_detector(z_source=0.0, z_sample=2.0)
        field = np.ones((params.ny, params.nx), dtype=np.complex64)
        with self.assertWarns(RuntimeWarning):
            output = propagation.square_and_downsample_2d(
                NumpyVector(field.reshape(-1)), params, params.z_detector
            )
        metadata = params.detector_last_metadata
        geometry = metadata["effective_geometry"]
        self.assertEqual(output.shape, (2, 4))
        self.assertAlmostEqual(geometry["magnification"], 4.0)
        self.assertAlmostEqual(geometry["current_z"], 1.5)
        self.assertAlmostEqual(geometry["detector_pixel_size_x"], 1.3 / 4.0)
        self.assertTrue(geometry["fresnel_scaled"])
        self.assertAlmostEqual(
            metadata["pixel_to_grid_ratio"]["x"], (1.3 / 4.0) / params.dx
        )

    def test_progress_and_cancellation(self):
        params = detector_params(integrator="area_v1")
        field = np.ones((params.ny, params.nx), dtype=np.complex64)
        events = []
        params.detector_progress_cb = lambda *event: events.append(event)
        propagation.square_and_downsample_2d(NumpyVector(field.reshape(-1)), params, 8.0)
        self.assertIn(("detector_integrate", params.ny, params.ny), events)
        self.assertIn(("detector_geometry", 1, 1), events)

        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            return calls >= 3

        params.detector_cancel_token = cancel
        with self.assertRaises(InterruptedError):
            propagation.square_and_downsample_2d(
                NumpyVector(field.reshape(-1)), params, 8.0
            )

    def test_config_metadata_feasibility_and_end_to_end_area_v1(self):
        dct = p3_config("area_v1")
        resolved = config.resolve_sim_params(dct)
        self.assertEqual(resolved["detector_integrator"], "area_v1")
        bad = copy.deepcopy(dct)
        bad["runtime"]["big_wave"]["detector_integrator"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported detector_integrator"):
            config.resolve_sim_params(bad)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(dct, root, root / "runs")
            computed = config.load(sim_dir / "computed.yaml")
            detector_algorithm = computed["algorithms"]["detector_2d"]
            self.assertEqual(detector_algorithm["version"], "area_separable_stream_v1")
            self.assertEqual(computed["provenance"]["algorithm_version"], "big-wave-2d-p4-v1")
            validation = validate_sim.validate(sim_dir)
            self.assertTrue(validation["ok"], validation["checks"])

            scratch = root / "scratch"
            scratch.mkdir()
            events = []
            multisim.run_single_simulation(
                sim_dir,
                0,
                scratch,
                detector_progress_cb=lambda *event: events.append(event),
            )
            sub_dir = sim_dir / "00000000"
            detected = np.load(sub_dir / "detected.npy", mmap_mode="r")
            metadata = config.load(sub_dir / "detector_metadata.yaml")
            self.assertEqual(detected.shape, (1, 4, 8))
            self.assertEqual(metadata["integrator"], "area_v1")
            self.assertEqual(metadata["effective_geometry"]["dx"], 1e-7)
            self.assertTrue(events)
            self.assertFalse((sub_dir / "detected.npy.part").exists())


if __name__ == "__main__":
    unittest.main()
