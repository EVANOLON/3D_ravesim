"""P0 safety-contract tests for big-wave 2D (stdlib unittest only)."""

from collections import namedtuple
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
from propagation import SimParams  # noqa: E402
import validate_sim  # noqa: E402


def base_config(nx=64, ny=64, budget=1.0):
    return {
        "dtype": "c8",
        "use_disk_vector": True,
        "save_final_u_vectors": False,
        "sim_params": {
            "N": nx * ny,
            "nx": nx,
            "ny": ny,
            "dx": 1e-6,
            "dy": 1e-6,
            "z_detector": 0.1,
            "detector_size": min(nx, ny) * 0.5e-6,
            "detector_size_x": nx * 0.5e-6,
            "detector_size_y": ny * 0.5e-6,
            "detector_pixel_size_x": 4e-6,
            "detector_pixel_size_y": 4e-6,
            "chunk_size": "auto",
        },
        "runtime": {
            "big_wave": {
                "memory_budget_gb": budget,
                "chunk_size": "auto",
                "fft2_backend": "scipy_in_memory",
                "detector_integrator": "legacy_fastwave",
            }
        },
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


def write_prepared_sim(path, dct):
    path.mkdir()
    resolved = copy.deepcopy(dct)
    resolved["sim_params"] = config.resolve_sim_params(resolved)
    config.save(path / "config.yaml", resolved)
    params = resolved["sim_params"]
    runtime = {
        key: params[key]
        for key in (
            "memory_budget_gb",
            "chunk_size",
            "fft2_backend",
            "detector_integrator",
        )
    }
    config.save(
        path / "computed.yaml",
        {
            "cutoff_angles": [0.0],
            "max_x": 0.0,
            "energy_range": [0.0, 0.0],
            "source_points": [],
            "runtime": {"big_wave": {"resolved": runtime}},
        },
    )


class TestP0Configuration(unittest.TestCase):
    def test_auto_chunk_is_conservative_and_row_aligned(self):
        dct = base_config(16384, 16384, budget=6.0)
        del dct["dtype"]
        resolved = config.resolve_sim_params(dct)
        self.assertEqual(config.parse_dtype(dct).itemsize, 16)
        self.assertEqual(resolved["chunk_size"] % 16384, 0)
        self.assertLessEqual(resolved["chunk_size"], 256 * 16384)
        self.assertLess(resolved["chunk_size"], resolved["N"])

    def test_runtime_values_override_legacy_sim_params(self):
        dct = base_config()
        dct["sim_params"]["chunk_size"] = 64
        dct["runtime"]["big_wave"]["chunk_size"] = 128
        dct["runtime"]["big_wave"]["memory_budget_gb"] = 2.5
        resolved = config.resolve_sim_params(dct)
        self.assertEqual(resolved["chunk_size"], 128)
        self.assertEqual(resolved["memory_budget_gb"], 2.5)

    def test_invalid_chunks_rejected(self):
        for chunk in (0, -1, 1.5, "bad", True):
            with self.subTest(chunk=chunk):
                dct = base_config()
                dct["runtime"]["big_wave"]["chunk_size"] = chunk
                with self.assertRaisesRegex(ValueError, "chunk_size"):
                    config.resolve_sim_params(dct)

    def test_invalid_memory_budgets_rejected(self):
        for budget in (0, -1, float("inf"), "bad"):
            with self.subTest(budget=budget):
                dct = base_config()
                dct["runtime"]["big_wave"]["memory_budget_gb"] = budget
                with self.assertRaisesRegex(ValueError, "memory_budget_gb"):
                    config.resolve_sim_params(dct)

    def test_p2_backend_requires_disk_vector(self):
        dct = base_config()
        dct["runtime"]["big_wave"]["fft2_backend"] = "bfpy_ooc"
        self.assertEqual(config.resolve_sim_params(dct)["fft2_backend"], "bfpy_ooc")
        dct["use_disk_vector"] = False
        with self.assertRaisesRegex(ValueError, "requires use_disk_vector"):
            config.resolve_sim_params(dct)

    def test_debug_wavefield_flag_must_be_boolean(self):
        dct = base_config()
        dct["runtime"]["big_wave"]["save_debug_wavefields"] = "false"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            config.resolve_sim_params(dct)

    def test_inconsistent_2d_shape_is_rejected(self):
        dct = base_config()
        dct["sim_params"]["N"] += 1
        with self.assertRaisesRegex(ValueError, r"must equal nx \* ny"):
            config.resolve_sim_params(dct)

    def test_non_power_of_two_shape_is_rejected_at_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "power of two"):
                multisim.setup_simulation(base_config(60, 64), root, root / "runs")

    def test_simparams_rejects_nonpositive_dimensions(self):
        with self.assertRaisesRegex(ValueError, "N must be positive"):
            SimParams(
                N=0,
                dx=1.0,
                z_detector=1.0,
                detector_size=1.0,
                detector_pixel_size_x=1.0,
                detector_pixel_size_y=1.0,
                wl=1.0,
                chunk_size=1,
            )

    def test_engine_runtime_gate_blocks_large_in_memory_fft2(self):
        params = SimParams(
            N=16384 * 16384,
            nx=16384,
            ny=16384,
            dx=1e-7,
            dy=1e-7,
            z_detector=1.0,
            detector_size=1e-3,
            detector_pixel_size_x=1e-6,
            detector_pixel_size_y=1e-6,
            wl=1e-10,
            chunk_size=16384 * 128,
            fft2_backend="scipy_in_memory",
        )
        with self.assertRaisesRegex(RuntimeError, "materializes the complete 2D field"):
            multisim.check_backend_runtime_safety(params)

    def test_setup_persists_request_resolution_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(base_config(), root, root / "runs")
            written = config.load(sim_dir / "config.yaml")
            computed = config.load(sim_dir / "computed.yaml")
            self.assertIsInstance(written["sim_params"]["chunk_size"], int)
            runtime = computed["runtime"]["big_wave"]
            self.assertEqual(runtime["requested"]["chunk_size"], "auto")
            self.assertEqual(
                runtime["resolved"]["chunk_size"],
                written["sim_params"]["chunk_size"],
            )
            self.assertFalse(runtime["resolved"]["save_debug_wavefields"])
            self.assertEqual(
                computed["algorithms"]["frequency_cutoff_2d"]["version"],
                "circular_scalar_v1",
            )
            self.assertEqual(
                computed["provenance"]["algorithm_version"],
                "big-wave-2d-p4-v1",
            )
            self.assertTrue(computed["provenance"]["git_commit"])


class TestP0Feasibility(unittest.TestCase):
    def test_array_header_is_read_via_memmap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.npy"
            np.save(path, np.zeros((2, 4, 4), dtype=np.uint32))
            original = validate_sim.np.load
            with mock.patch.object(validate_sim.np, "load", wraps=original) as loader:
                info = validate_sim._array_info(path)
            self.assertEqual(info["shape"], [2, 4, 4])
            self.assertEqual(loader.call_args.kwargs["mmap_mode"], "r")

    def test_memory_budget_is_a_hard_gate(self):
        DiskUsage = namedtuple("DiskUsage", "total used free")
        with tempfile.TemporaryDirectory() as directory:
            sim_dir = Path(directory) / "sim"
            write_prepared_sim(sim_dir, base_config(budget=0.1))
            with mock.patch.object(
                validate_sim,
                "host_memory",
                return_value=(64 * validate_sim.GIB, 60 * validate_sim.GIB, 0, 0),
            ), mock.patch.object(
                validate_sim.shutil,
                "disk_usage",
                return_value=DiskUsage(1000 * validate_sim.GIB, 0, 1000 * validate_sim.GIB),
            ):
                result = validate_sim.feasibility(sim_dir, "big-wave")
        self.assertTrue(result["ok"])
        self.assertFalse(result["feasible"])
        self.assertIn("ram", result["blocked_by"])
        self.assertAlmostEqual(result["ram"]["memory_budget_gb"], 0.1)

    def test_disk_headroom_is_a_hard_gate(self):
        DiskUsage = namedtuple("DiskUsage", "total used free")
        with tempfile.TemporaryDirectory() as directory:
            sim_dir = Path(directory) / "sim"
            write_prepared_sim(sim_dir, base_config(budget=2.0))
            with mock.patch.object(
                validate_sim,
                "host_memory",
                return_value=(64 * validate_sim.GIB, 60 * validate_sim.GIB, 0, 0),
            ), mock.patch.object(
                validate_sim.shutil,
                "disk_usage",
                return_value=DiskUsage(validate_sim.GIB, validate_sim.GIB - 1, 1),
            ):
                result = validate_sim.feasibility(sim_dir, "big-wave")
        self.assertFalse(result["feasible"])
        self.assertIn("disk", result["blocked_by"])

    def test_invalid_cutoff_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sim_dir = Path(directory) / "sim"
            write_prepared_sim(sim_dir, base_config())
            computed = config.load(sim_dir / "computed.yaml")
            computed["cutoff_angles"] = [math.pi]
            config.save(sim_dir / "computed.yaml", computed)
            result = validate_sim.validate(sim_dir)
        self.assertFalse(result["ok"])
        self.assertIn("z_layout", [item["name"] for item in result["checks"] if not item["ok"]])

    def test_large_2d_scipy_diskvector_mismatch_is_blocked(self):
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
            "fft2_backend": "scipy_in_memory",
        }
        DiskUsage = namedtuple("DiskUsage", "total used free")
        with mock.patch.object(validate_sim, "validate", return_value=validation), \
             mock.patch.object(
                 validate_sim,
                 "host_memory",
                 return_value=(256 * validate_sim.GIB, 240 * validate_sim.GIB, 0, 0),
             ), \
             mock.patch.object(
                 validate_sim.shutil,
                 "disk_usage",
                 return_value=DiskUsage(1000 * validate_sim.GIB, 0, 1000 * validate_sim.GIB),
             ):
            result = validate_sim.feasibility(Path("unused"), "big-wave")
        self.assertFalse(result["feasible"])
        self.assertIn("fft2_backend_vector_mismatch", result["blocked_by"])

    def test_failure_classification_and_feasibility_exit_code(self):
        self.assertEqual(validate_sim.classify_returncode(137), "oom_or_sigkill")
        self.assertEqual(
            validate_sim.classify_returncode(1, "MemoryError"),
            "python_memory_error",
        )
        self.assertEqual(
            validate_sim.classify_returncode(1, "No space left on device"),
            "disk_full",
        )
        self.assertEqual(
            validate_sim.result_exit_code({"ok": True, "feasible": False}, True),
            2,
        )
        self.assertEqual(
            validate_sim.result_exit_code({"ok": True, "feasible": True}, True),
            0,
        )


if __name__ == "__main__":
    unittest.main()
