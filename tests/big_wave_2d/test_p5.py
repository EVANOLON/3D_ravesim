"""P5 analytical-physics and cross-engine acceptance tests."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import fresnel


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

import config  # noqa: E402
import multisim  # noqa: E402
from optical_element import (  # noqa: E402
    Material,
    Sample,
    generate_deltabeta_table,
    material_factor,
)
from plasma_sample import PlasmaSample  # noqa: E402
from propagation import (  # noqa: E402
    SimParams,
    apply_frequency_cutoff_2d,
    convert_energy_wavelength,
    propagate_2d,
    propagate_analytically_2d,
)
from vector import NumpyVector  # noqa: E402
from tests.big_wave_2d.p5_benchmark import compare_results  # noqa: E402


FASTWAVE = ROOT / "fast-wave" / "build-Release" / "fastwave"


def relative_l2(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return float(
        np.linalg.norm((actual - expected).reshape(-1))
        / max(np.linalg.norm(expected.reshape(-1)), np.finfo(float).eps)
    )


def params(nx=64, ny=32, dx=1e-6, dy=1.5e-6, wl=1e-10, z=0.2):
    return SimParams(
        N=nx * ny, nx=nx, ny=ny, dx=dx, dy=dy, z_detector=z,
        detector_size=min(nx * dx, ny * dy),
        detector_size_x=nx * dx, detector_size_y=ny * dy,
        detector_pixel_size_x=dx, detector_pixel_size_y=dy,
        wl=wl, chunk_size=nx * 4, detector_integrator="area_v1",
    )


class TestP5AnalyticalPhysics(unittest.TestCase):
    def test_point_source_spherical_amplitude_phase_and_xy_offset(self):
        p = params(nx=40, ny=28, dx=0.7e-6, dy=1.1e-6, wl=1.3e-10, z=0.3)
        source_x, source_y, distance = 1.2e-6, -2.4e-6, 0.3
        u = NumpyVector(np.zeros(p.N, dtype=np.complex128))
        propagate_analytically_2d(u, distance, source_x, source_y, p)
        x = (np.arange(p.nx) - p.nx / 2) * p.dx - source_x
        y = (np.arange(p.ny) - p.ny / 2) * p.get_dy() - source_y
        radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2 + distance**2)
        actual = u.vec.reshape(p.ny, p.nx)
        expected_amplitude = 1.0 / radius
        self.assertLess(relative_l2(np.abs(actual), expected_amplitude), 2e-15)
        # Independent division and multiplication orders lose a few microradians
        # when reducing a ~1e10-radian carrier phase modulo 2*pi.
        expected_phase = np.exp(2j * np.pi * radius / p.wl) / radius
        circular_error = np.max(np.abs(np.angle(actual * np.conj(expected_phase))))
        self.assertLess(circular_error, 3e-6)
        peak = np.unravel_index(np.argmax(np.abs(u.vec.reshape(p.ny, p.nx))), (p.ny, p.nx))
        self.assertLessEqual(abs(peak[1] - (p.nx / 2 + source_x / p.dx)), 1.0)
        self.assertLessEqual(abs(peak[0] - (p.ny / 2 + source_y / p.get_dy())), 1.0)

    def test_plane_and_tilted_plane_wave_propagation(self):
        p = params(nx=64, ny=32, dx=1.0, dy=1.0, wl=0.02, z=0.7)
        x = np.arange(p.nx)[None, :]
        y = np.arange(p.ny)[:, None]
        for mx, my in ((0, 0), (5, -3)):
            field = np.exp(2j * np.pi * (mx * x / p.nx + my * y / p.ny))
            u = NumpyVector(field.astype(np.complex128).reshape(-1))
            U = NumpyVector(np.zeros(p.N, dtype=np.complex128))
            propagate_2d(
                u, U, p.dx, p.get_dy(), p.wl, p.z_detector,
                p.chunk_size, 20.0, p.nx, p.ny,
            )
            fx, fy = mx / (p.nx * p.dx), my / (p.ny * p.get_dy())
            transfer = np.exp(2j * np.pi * p.z_detector / p.wl) * np.exp(
                -1j * np.pi * p.wl * p.z_detector * (fx * fx + fy * fy)
            )
            self.assertLess(relative_l2(u.vec.reshape(p.ny, p.nx), field * transfer), 2e-14)

    def test_uniform_thin_layer_phase_and_beer_lambert(self):
        p = params(nx=32, ny=16, dx=1e-6, dy=1e-6, wl=1e-10, z=0.1)
        thickness = 2e-7
        delta_beta = 1.2e-7 + 2.4e-8j
        grid = np.ones((1, p.ny + 2, p.nx + 2), dtype=np.uint32)
        material = Material("P5-test", 1.0)
        sample = Sample(
            z_start=0.05, pixel_size_x=p.dx, pixel_size_z=thickness,
            grid=grid, materials=[material], x_positions=np.array([0.0]),
            pixel_size_y=p.get_dy(), y_positions=np.array([0.0]),
        )
        sample.store_deltabetas([(material, delta_beta)])
        u = NumpyVector(np.ones(p.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(p.N, dtype=np.complex128))
        sample.apply(u, U, p, cutoff_freq=1.0 / p.dx, stepping_iteration=0, history=None)
        expected = material_factor(delta_beta, thickness, p.wl) * np.exp(
            2j * np.pi * thickness / p.wl
        )
        circular_error = np.max(np.abs(np.angle(u.vec * np.conj(expected))))
        measured_intensity = float(np.mean(np.abs(u.vec) ** 2))
        beer_lambert = float(np.exp(-4 * np.pi * thickness * delta_beta.imag / p.wl))
        self.assertLess(circular_error, 2e-12)
        self.assertAlmostEqual(measured_intensity, beer_lambert, delta=2e-13)

    def test_circular_cutoff_covers_axis_and_diagonal(self):
        nx = ny = 64
        cutoff = 10 / nx
        spectrum = np.zeros((ny, nx), dtype=np.complex128)
        spectrum[0, 14] = 1.0       # axis: inside sqrt(2) * cutoff
        spectrum[0, 15] = 2.0       # axis: outside
        spectrum[10, 10] = 3.0      # diagonal: exactly at the documented boundary
        spectrum[10, 11] = 4.0      # diagonal: outside
        vector = NumpyVector(spectrum.reshape(-1))
        apply_frequency_cutoff_2d(vector, cutoff, 1.0, 1.0, nx, ny, 4 * nx)
        actual = vector.vec.reshape(ny, nx)
        self.assertEqual(actual[0, 14], 1.0)
        self.assertEqual(actual[0, 15], 0.0)
        self.assertEqual(actual[10, 10], 3.0)
        self.assertEqual(actual[10, 11], 0.0)

    def test_rectangular_aperture_matches_fresnel_integral(self):
        p = params(nx=256, ny=256, dx=1e-6, dy=1e-6, wl=1e-10, z=0.4)
        aperture_pixels = 16
        aperture = np.zeros((p.ny, p.nx), dtype=np.complex128)
        y0, x0 = p.ny // 2, p.nx // 2
        aperture[
            y0 - aperture_pixels // 2 : y0 + aperture_pixels // 2,
            x0 - aperture_pixels // 2 : x0 + aperture_pixels // 2,
        ] = 1.0
        u = NumpyVector(aperture.reshape(-1))
        U = NumpyVector(np.zeros(p.N, dtype=np.complex128))
        propagate_2d(
            u, U, p.dx, p.get_dy(), p.wl, p.z_detector,
            p.chunk_size, 1.0 / p.dx, p.nx, p.ny,
        )
        coordinate = (np.arange(p.nx) - p.nx // 2) * p.dx
        width = aperture_pixels * p.dx
        scale = np.sqrt(2.0 / (p.wl * p.z_detector))
        # The even, cell-centred discrete aperture occupies [-8.5, 7.5] um.
        aperture_center = -0.5 * p.dx
        lower = scale * (aperture_center - width / 2 - coordinate)
        upper = scale * (aperture_center + width / 2 - coordinate)
        s0, c0 = fresnel(lower)
        s1, c1 = fresnel(upper)
        one_d = (c1 - c0) + 1j * (s1 - s0)
        expected = np.abs(one_d[:, None] * one_d[None, :]) ** 2
        actual = np.abs(u.vec.reshape(p.ny, p.nx)) ** 2
        expected /= expected.max()
        actual /= actual.max()
        central = np.s_[48:-48, 48:-48]
        self.assertLess(relative_l2(actual[central], expected[central]), 0.12)

    def test_offset_thin_tungsten_matches_explicit_transmission(self):
        nx = ny = 64
        energy = 8000.0
        wl = convert_energy_wavelength(energy)
        p = params(nx=nx, ny=ny, dx=1e-6, dy=1e-6, wl=wl, z=0.1)
        p.chunk_size = nx * 8
        thickness = 2e-7
        margin = 12
        grid = np.zeros((1, ny + margin, nx + margin), dtype=np.uint32)
        grid[0, 20:56, 27:45] = 1
        tungsten = Material("W", 19.3)
        table = generate_deltabeta_table([tungsten], energy)
        delta_beta = table[0][1]
        sample = Sample(
            z_start=0.05,
            pixel_size_x=p.dx,
            pixel_size_y=p.get_dy(),
            pixel_size_z=thickness,
            grid=grid,
            materials=[tungsten],
            x_positions=np.array([2 * p.dx]),
            y_positions=np.array([-3 * p.get_dy()]),
        )
        sample.store_deltabetas(table)
        actual = NumpyVector(np.ones(p.N, dtype=np.complex128))
        scratch = NumpyVector(np.zeros(p.N, dtype=np.complex128))
        cutoff = 0.4 / p.dx
        sample.apply(actual, scratch, p, cutoff, 0, None)

        # Sample coordinates select exact integer cells here: base margin/2,
        # plus the configured x/y offsets.
        mask = grid[0, 3:3 + ny, 8:8 + nx]
        expected_field = material_factor(delta_beta * (mask == 1), thickness, wl)
        expected = NumpyVector(expected_field.astype(np.complex128).reshape(-1))
        expected_scratch = NumpyVector(np.zeros(p.N, dtype=np.complex128))
        propagate_2d(
            expected, expected_scratch, p.dx, p.get_dy(), p.wl, thickness,
            p.chunk_size, cutoff, p.nx, p.ny,
        )
        self.assertLess(relative_l2(actual.vec, expected.vec), 2e-13)

    def test_hollow_capsule_is_xy_symmetric_and_not_solid(self):
        nx = ny = 72
        energy = 8000.0
        wl = convert_energy_wavelength(energy)
        p = params(nx=nx, ny=ny, dx=0.5e-6, dy=0.5e-6, wl=wl, z=0.1)
        p.chunk_size = nx * 8
        nz, margin = 7, 8
        grid_shape = (nz, ny + margin, nx + margin)
        z = (np.arange(nz) - (nz - 1) / 2.0)[:, None, None]
        y = (np.arange(grid_shape[1]) - grid_shape[1] / 2.0)[None, :, None]
        x = (np.arange(grid_shape[2]) - grid_shape[2] / 2.0)[None, None, :]
        radius = np.sqrt(x * x + y * y + z * z)
        shell_grid = ((radius <= 18.0) & (radius >= 11.0)).astype(np.uint32)
        solid_grid = (radius <= 18.0).astype(np.uint32)
        carbon = Material("C", 2.0)
        table = generate_deltabeta_table([carbon], energy)

        def simulate(grid):
            sample = Sample(
                z_start=0.05,
                pixel_size_x=p.dx,
                pixel_size_y=p.get_dy(),
                pixel_size_z=0.25e-6,
                grid=grid,
                materials=[carbon],
                x_positions=np.array([0.0]),
                y_positions=np.array([0.0]),
            )
            sample.store_deltabetas(table)
            wave = NumpyVector(np.ones(p.N, dtype=np.complex128))
            spectrum = NumpyVector(np.zeros(p.N, dtype=np.complex128))
            sample.apply(wave, spectrum, p, 0.4 / p.dx, 0, None)
            return wave.vec.reshape(ny, nx)

        shell = simulate(shell_grid)
        solid = simulate(solid_grid)
        intensity = np.abs(shell) ** 2
        self.assertLess(
            relative_l2(intensity[ny // 2, :], intensity[:, nx // 2]),
            2e-13,
        )
        self.assertGreater(relative_l2(shell, solid), 1e-3)

    def test_plasma_1d_section_matches_2d_centerline(self):
        nx, ny, nz = 64, 16, 2
        dx = 1e-6
        wl = convert_energy_wavelength(8000.0)
        grid_nx, grid_ny = nx + 4, ny + 4

        def plasma(shape, pixel_size_y=0.0, y_positions=None):
            return PlasmaSample(
                z_start=0.05,
                pixel_size_x=dx,
                pixel_size_y=pixel_size_y,
                pixel_size_z=0.2e-6,
                ne_grid=np.full(shape, 1e24, dtype=np.float64),
                ni_grid=np.full(shape, 1e24, dtype=np.float64),
                te_grid=np.full(shape, 100.0, dtype=np.float64),
                zstar_grid=np.ones(shape, dtype=np.float64),
                Z=1,
                x_positions=np.array([0.0]),
                y_positions=y_positions,
            )

        p1 = SimParams(
            N=nx, nx=nx, ny=1, dx=dx, dy=0.0, z_detector=0.1,
            detector_size=nx * dx, detector_pixel_size_x=dx,
            detector_pixel_size_y=dx, wl=wl, chunk_size=16,
        )
        p2 = params(nx=nx, ny=ny, dx=dx, dy=dx, wl=wl, z=0.1)
        p2.chunk_size = nx * 4
        one_d = plasma((nz, grid_nx))
        two_d = plasma(
            (nz, grid_ny, grid_nx), pixel_size_y=dx,
            y_positions=np.array([0.0]),
        )
        u1 = NumpyVector(np.ones(nx, dtype=np.complex128))
        U1 = NumpyVector(np.zeros(nx, dtype=np.complex128))
        u2 = NumpyVector(np.ones(nx * ny, dtype=np.complex128))
        U2 = NumpyVector(np.zeros(nx * ny, dtype=np.complex128))
        cutoff = 0.4 / dx
        one_d.apply(u1, U1, p1, cutoff, 0, None)
        two_d.apply(u2, U2, p2, cutoff, 0, None)
        centerline = u2.vec.reshape(ny, nx)[ny // 2]
        self.assertLess(relative_l2(centerline, u1.vec), 2e-12)


def cross_engine_config():
    nx = ny = 64
    return {
        "dtype": "c8",
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "sim_params": {
            "N": nx * ny, "nx": nx, "ny": ny, "dx": 1e-6, "dy": 1e-6,
            "z_detector": 0.1, "detector_size": 32e-6,
            "detector_size_x": 32e-6, "detector_size_y": 32e-6,
            "detector_pixel_size_x": 2e-6, "detector_pixel_size_y": 2e-6,
            "chunk_size": "auto",
        },
        "runtime": {"big_wave": {
            "memory_budget_gb": 1.0, "chunk_size": "auto",
            "fft2_backend": "scipy_in_memory", "detector_integrator": "legacy_fastwave",
        }},
        "multisource": {
            "type": "points", "nr_source_points": 1,
            "energy_range": [1000.0, 1000.0],
            "x_range": [0.25e-6, 0.25e-6], "y_range": [-0.4e-6, -0.4e-6],
            "z": 0.0, "seed": 3,
        },
        "elements": [],
    }


class TestP5CrossEngine(unittest.TestCase):
    @unittest.skipUnless(FASTWAVE.is_file(), "fast-wave binary is not built")
    def test_big_wave_matches_fast_wave_for_offset_point_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = multisim.setup_simulation(cross_engine_config(), root, root / "runs")
            big_dir, fast_dir = root / "big", root / "fast"
            shutil.copytree(prepared, big_dir)
            shutil.copytree(prepared, fast_dir)
            multisim.run_single_simulation(big_dir, 0, root / "scratch")
            result = subprocess.run(
                [str(FASTWAVE), str(fast_dir), "-s", "0"],
                check=False, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                self.fail(f"fast-wave failed: {result.stderr or result.stdout}")
            big = np.load(big_dir / "00000000" / "detected.npy")
            fast = np.load(fast_dir / "00000000" / "detected.npy")
            self.assertEqual(big.shape, fast.shape)
            error = relative_l2(big, fast)
            print(f"big-wave/fast-wave relative L2: {error:.9g}")
            self.assertLess(error, 2e-4)


class TestP5ConvergenceEvidence(unittest.TestCase):
    def test_comparator_requires_identical_physical_domain(self):
        geometry = {
            "effective_geometry": {
                "detector_pixel_size_x": 1.0,
                "detector_pixel_size_y": 1.0,
                "current_z": 2.0,
            }
        }
        coarse = {
            "status": "passed", "mode": "sample", "nx": 4, "ny": 4,
            "simulation_fov": 8.0, "cutoff_frequency": 3.0,
            "integrator": "area_v1", "detected_shape": [1, 2, 2],
            "centerline_256": [1.0, 2.0], "detected_energy": 4.0,
            "detected_peak": 2.0, "detector_metadata": geometry,
        }
        fine = copy.deepcopy(coarse)
        fine.update({
            "nx": 8, "ny": 8, "centerline_256": [1.0, 2.2],
            "detected_energy": 4.1, "detected_peak": 2.2,
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coarse_path, fine_path = root / "coarse.json", root / "fine.json"
            coarse_path.write_text(json.dumps(coarse), encoding="utf-8")
            fine_path.write_text(json.dumps(fine), encoding="utf-8")
            comparison = compare_results([fine_path, coarse_path])
            self.assertEqual(comparison["comparisons"][0]["coarse"], [4, 4])
            self.assertGreater(
                comparison["comparisons"][0]["centerline_relative_l2"], 0.0
            )
            fine["simulation_fov"] = 16.0
            fine_path.write_text(json.dumps(fine), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "physical configurations"):
                compare_results([coarse_path, fine_path])


if __name__ == "__main__":
    unittest.main()
