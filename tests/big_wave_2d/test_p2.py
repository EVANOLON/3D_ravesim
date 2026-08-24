"""P2 transactional out-of-core FFT2 acceptance tests."""

from collections import namedtuple
import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import bfpy
import numpy as np
from scipy import fft as spfft


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
import vector  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import DiskVector, NumpyVector  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def fft_function(dtype, inverse=False):
    if dtype == np.dtype(np.complex64):
        return bfpy.ifft2_c8 if inverse else bfpy.fft2_c8
    return bfpy.ifft2_c16 if inverse else bfpy.fft2_c16


def random_field(ny, nx, dtype, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))).astype(dtype)


def p2_config(nx=64, ny=32):
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
                "detector_integrator": "legacy_fastwave",
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


class TestBfpyOocNumerics(unittest.TestCase):
    def test_api_is_exported(self):
        for name in ("fft2_c8", "ifft2_c8", "fft2_c16", "ifft2_c16"):
            self.assertTrue(callable(getattr(bfpy, name)))

    def test_c8_c16_square_and_non_square_forward_inverse(self):
        for dtype in (np.dtype(np.complex64), np.dtype(np.complex128)):
            for ny, nx in ((8, 8), (8, 16), (16, 8), (7, 11)):
                with self.subTest(dtype=dtype, shape=(ny, nx)), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    infile = root / "input.npy"
                    spectrum = root / "spectrum.npy"
                    restored = root / "restored.npy"
                    scratch = root / "scratch.npy"
                    field = random_field(ny, nx, dtype)
                    np.save(infile, field.reshape(-1))
                    original_digest = digest(infile)

                    events = []
                    fft_function(dtype)(
                        infile, spectrum, scratch, nx, ny,
                        lambda *event: events.append(event), None, 4096,
                    )
                    actual = np.load(spectrum).reshape(ny, nx)
                    tolerance = 3e-5 if dtype == np.dtype(np.complex64) else 2e-12
                    np.testing.assert_allclose(
                        actual, spfft.fft2(field), rtol=tolerance, atol=tolerance
                    )
                    self.assertEqual(digest(infile), original_digest)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertIn("transpose_to_output", {event[0] for event in events})
                    self.assertEqual(events[-1][0], "complete")

                    fft_function(dtype, inverse=True)(
                        spectrum, restored, scratch, nx, ny, None, None, 4096
                    )
                    np.testing.assert_allclose(
                        np.load(restored).reshape(ny, nx),
                        field,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                    self.assertFalse(scratch.exists())

    def test_two_dimensional_npy_shape_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = random_field(4, 8, np.dtype(np.complex64))
            np.save(root / "input.npy", field)
            bfpy.fft2_c8(
                root / "input.npy", root / "output.npy", root / "scratch.npy",
                8, 4, None, None, 1024,
            )
            self.assertEqual(np.load(root / "output.npy").shape, (4, 8))

    def test_path_conflict_bad_dtype_shape_and_length_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = random_field(4, 8, np.dtype(np.complex64))
            infile = root / "input.npy"
            np.save(infile, field.reshape(-1))
            before = digest(infile)
            with self.assertRaisesRegex(ValueError, "path conflict"):
                bfpy.fft2_c8(infile, infile, root / "scratch.npy", 8, 4)
            self.assertEqual(digest(infile), before)
            hardlink = root / "input-hardlink.npy"
            os.link(infile, hardlink)
            with self.assertRaisesRegex(ValueError, "path conflict"):
                bfpy.fft2_c8(infile, root / "out.npy", hardlink, 8, 4)
            symlink = root / "input-symlink.npy"
            symlink.symlink_to(infile)
            with self.assertRaisesRegex(ValueError, "path conflict"):
                bfpy.fft2_c8(infile, root / "out.npy", symlink, 8, 4)
            with self.assertRaisesRegex(ValueError, "dtype mismatch"):
                bfpy.fft2_c16(infile, root / "out.npy", root / "scratch.npy", 8, 4)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                bfpy.fft2_c8(infile, root / "out.npy", root / "scratch.npy", 16, 4)

            truncated = root / "truncated.npy"
            truncated.write_bytes(infile.read_bytes()[:-3])
            with self.assertRaisesRegex(ValueError, "file length mismatch"):
                bfpy.fft2_c8(truncated, root / "out.npy", root / "scratch.npy", 8, 4)


class TestBfpyOocRecovery(unittest.TestCase):
    def test_cooperative_cancel_preserves_recoverable_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = random_field(32, 64, np.dtype(np.complex64))
            infile = root / "input.npy"
            outfile = root / "output.npy"
            scratch = root / "scratch.npy"
            np.save(infile, field.reshape(-1))
            outfile.write_bytes(b"previous-valid-output-sentinel")
            previous_output = outfile.read_bytes()
            token = threading.Event()

            def progress(pass_name, completed, total):
                if pass_name == "row_fft" and completed > 0:
                    token.set()

            with self.assertRaises(InterruptedError):
                bfpy.fft2_c8(
                    infile, outfile, scratch, 64, 32, progress, token, 4096
                )
            self.assertEqual(outfile.read_bytes(), previous_output)
            self.assertTrue(Path(str(scratch) + ".fft2.manifest").exists() or Path(str(outfile) + ".part").exists())

            token.clear()
            bfpy.fft2_c8(infile, outfile, scratch, 64, 32, None, token, 4096)
            np.testing.assert_allclose(
                np.load(outfile).reshape(32, 64),
                spfft.fft2(field),
                rtol=3e-5,
                atol=3e-5,
            )
            self.assertFalse(Path(str(scratch) + ".fft2.manifest").exists())
            self.assertFalse(Path(str(scratch) + ".column.part").exists())
            self.assertFalse(Path(str(outfile) + ".part").exists())

    def test_corrupt_completed_artifact_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = random_field(16, 32, np.dtype(np.complex64), seed=19)
            infile = root / "input.npy"
            outfile = root / "output.npy"
            scratch = root / "scratch.npy"
            np.save(infile, field.reshape(-1))
            token = threading.Event()

            def progress(pass_name, completed, total):
                if pass_name == "column_fft" and completed > 0:
                    token.set()

            with self.assertRaises(InterruptedError):
                bfpy.fft2_c8(
                    infile, outfile, scratch, 32, 16, progress, token, 2048
                )
            self.assertTrue(scratch.exists())
            scratch.write_bytes(b"corrupt")
            token.clear()
            bfpy.fft2_c8(infile, outfile, scratch, 32, 16, None, token, 2048)
            np.testing.assert_allclose(
                np.load(outfile).reshape(16, 32),
                spfft.fft2(field),
                rtol=3e-5,
                atol=3e-5,
            )

    def test_process_kill_during_every_pass_recovers(self):
        worker = Path(__file__).with_name("p2_kill_worker.py")
        for target_pass in (
            "row_fft",
            "transpose_to_scratch",
            "column_fft",
            "transpose_to_output",
        ):
            with self.subTest(pass_name=target_pass), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                infile = root / "input.npy"
                outfile = root / "output.npy"
                scratch = root / "scratch.npy"
                field = random_field(32, 64, np.dtype(np.complex64), seed=11)
                np.save(infile, field.reshape(-1))
                before = digest(infile)
                process = subprocess.run(
                    [
                        sys.executable,
                        str(worker),
                        str(infile),
                        str(outfile),
                        str(scratch),
                        "64",
                        "32",
                        "c8",
                        target_pass,
                        "4096",
                    ],
                    check=False,
                )
                self.assertEqual(process.returncode, 90)
                self.assertEqual(digest(infile), before)

                bfpy.fft2_c8(infile, outfile, scratch, 64, 32, None, None, 4096)
                np.testing.assert_allclose(
                    np.load(outfile).reshape(32, 64),
                    spfft.fft2(field),
                    rtol=3e-5,
                    atol=3e-5,
                )
                self.assertEqual(digest(infile), before)


class TestBigWaveP2Integration(unittest.TestCase):
    def test_small_diskvector_simulation_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(p2_config(), root, root / "runs")
            scratch_dir = root / "scratch"
            scratch_dir.mkdir()
            progress = []
            multisim.run_single_simulation(
                sim_dir,
                0,
                scratch_dir,
                fft2_progress_cb=lambda pass_name, completed, total: progress.append(
                    pass_name
                ) if completed == total else None,
            )
            detected = np.load(multisim.get_sub_dir(sim_dir, 0) / "detected.npy")
            self.assertEqual(detected.ndim, 3)
            self.assertEqual(detected.shape[0], 1)
            self.assertTrue(np.all(np.isfinite(detected)))
            self.assertGreater(float(detected.max()), 0.0)
            self.assertIn("row_fft", progress)
            self.assertIn("complete", progress)

    def test_diskvector_fft2_never_calls_numpy_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            field = random_field(8, 16, np.dtype(np.complex64))
            np.save(root / "input.npy", field.reshape(-1))
            source = DiskVector(
                root / "input.npy", root / "scratch.npy", field.size,
                np.dtype(np.complex64), fft2_memory_budget_bytes=4096,
            )
            destination = DiskVector(
                root / "output.npy", root / "scratch.npy", field.size,
                np.dtype(np.complex64), fft2_memory_budget_bytes=4096,
            )
            with mock.patch.object(vector.np, "load", side_effect=AssertionError("whole-field load")):
                source.fft2(destination, 16, 8)
            np.testing.assert_allclose(
                np.load(destination.file).reshape(8, 16),
                spfft.fft2(field),
                rtol=3e-5,
                atol=3e-5,
            )

    def test_diskvector_propagation_matches_numpyvector(self):
        nx, ny = 32, 16
        dtype = np.dtype(np.complex64)
        field = random_field(ny, nx, dtype)
        params = SimParams(
            N=nx * ny,
            nx=nx,
            ny=ny,
            dx=2e-7,
            dy=2.5e-7,
            z_detector=0.1,
            detector_size=nx * 2e-7,
            detector_size_x=nx * 2e-7,
            detector_size_y=ny * 2.5e-7,
            detector_pixel_size_x=8e-7,
            detector_pixel_size_y=1e-6,
            wl=propagation.convert_energy_wavelength(8000.0),
            chunk_size=nx * 4,
            fft2_backend="bfpy_ooc",
        )
        cutoff = 1e7
        reference_u = NumpyVector(field.reshape(-1).copy())
        reference_U = NumpyVector(np.zeros(field.size, dtype=dtype))
        propagation.propagate_2d(
            reference_u, reference_U, params.dx, params.get_dy(), params.wl,
            0.002, params.chunk_size, cutoff, nx, ny,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "u.npy", field.reshape(-1))
            disk_u = DiskVector(
                root / "u.npy", root / "scratch.npy", field.size, dtype,
                fft2_memory_budget_bytes=8192,
            )
            disk_U = DiskVector(
                root / "spectrum.npy", root / "scratch.npy", field.size, dtype,
                fft2_memory_budget_bytes=8192,
            )
            propagation.propagate_2d(
                disk_u, disk_U, params.dx, params.get_dy(), params.wl,
                0.002, params.chunk_size, cutoff, nx, ny,
            )
            np.testing.assert_allclose(
                np.load(disk_u.file), reference_u.vec, rtol=7e-5, atol=7e-5
            )

    def test_config_metadata_and_feasibility_enable_ooc(self):
        dct = p2_config()
        resolved = config.resolve_sim_params(dct)
        self.assertEqual(resolved["fft2_backend"], "bfpy_ooc")
        automatic = copy.deepcopy(dct)
        del automatic["runtime"]["big_wave"]["fft2_backend"]
        self.assertEqual(
            config.resolve_sim_params(automatic)["fft2_backend"], "bfpy_ooc"
        )
        without_disk = copy.deepcopy(dct)
        without_disk["use_disk_vector"] = False
        with self.assertRaisesRegex(ValueError, "requires use_disk_vector"):
            config.resolve_sim_params(without_disk)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(dct, root, root / "runs")
            computed = config.load(sim_dir / "computed.yaml")
        self.assertEqual(computed["provenance"]["algorithm_version"], "big-wave-2d-p4-v1")
        self.assertEqual(computed["algorithms"]["fft2"]["version"], "bfpy_ooc_transactional_v1")
        self.assertTrue(computed["algorithms"]["fft2"]["restart_manifest"])

        points = 16384 * 16384
        validation = {
            "ok": True,
            "checks": [],
            "is_2d": True,
            "N": points,
            "nx": 16384,
            "ny": 16384,
            "dtype": "c8",
            "use_disk_vector": True,
            "grid_bytes": 0,
            "detector_pixels": 4096 * 4096,
            "chunk_size": 16384 * 128,
            "memory_budget_gb": 64.0,
            "fft2_backend": "bfpy_ooc",
        }
        DiskUsage = namedtuple("DiskUsage", "total used free")
        with mock.patch.object(validate_sim, "validate", return_value=validation), mock.patch.object(
            validate_sim, "host_memory", return_value=(256 * validate_sim.GIB, 240 * validate_sim.GIB, 0, 0)
        ), mock.patch.object(
            validate_sim.shutil,
            "disk_usage",
            return_value=DiskUsage(1000 * validate_sim.GIB, 0, 1000 * validate_sim.GIB),
        ):
            feasibility = validate_sim.feasibility(Path("unused"), "big-wave")
        self.assertTrue(feasibility["backend"]["ok"])
        self.assertNotIn("fft2_backend_not_out_of_core", feasibility["blocked_by"])


if __name__ == "__main__":
    unittest.main()
