"""P4 multi-source, History v2 and runtime-checkpoint acceptance tests."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = ROOT / "big-wave"
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

from checkpoint import RunCheckpoint  # noqa: E402
import config  # noqa: E402
from history import History, load_history, migrate_legacy_history  # noqa: E402
import multisim  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import NumpyVector  # noqa: E402
import wavesim  # noqa: E402


def p4_config(nr_sources=3):
    nx, ny = 32, 16
    return {
        "dtype": "c8",
        "use_disk_vector": False,
        "save_final_u_vectors": False,
        "sim_params": {
            "N": nx * ny,
            "nx": nx,
            "ny": ny,
            "dx": 1e-6,
            "dy": 1.5e-6,
            "z_detector": 0.1,
            "detector_size": 8e-6,
            "detector_size_x": 16e-6,
            "detector_size_y": 12e-6,
            "detector_pixel_size_x": 2e-6,
            "detector_pixel_size_y": 3e-6,
            "chunk_size": "auto",
        },
        "runtime": {
            "big_wave": {
                "memory_budget_gb": 1.0,
                "chunk_size": "auto",
                "fft2_backend": "scipy_in_memory",
                "detector_integrator": "area_v1",
            }
        },
        "multisource": {
            "type": "points",
            "nr_source_points": nr_sources,
            "energy_range": [1000.0, 1000.0],
            "x_range": [-1e-7, 1e-7],
            "y_range": [-2e-7, 3e-7],
            "z": 0.0,
            "seed": 17,
        },
        "elements": [],
    }


class TestP4MultiSourceY(unittest.TestCase):
    def test_y_is_persisted_per_source_and_in_computed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(p4_config(), root, root / "runs")
            computed = config.load(sim_dir / "computed.yaml")
            ys = []
            for index, point in enumerate(computed["source_points"]):
                subconfig = config.load(sim_dir / f"{index:08d}" / "subconfig.yaml")
                self.assertIn("y", subconfig["source"])
                self.assertEqual(float(point["y"]), float(subconfig["source"]["y"]))
                ys.append(float(point["y"]))
            self.assertTrue(any(value != 0.0 for value in ys))
            self.assertEqual(computed["provenance"]["algorithm_version"], "big-wave-2d-p4-v1")

    def test_missing_y_range_remains_backward_compatible(self):
        dct = p4_config(1)
        del dct["multisource"]["y_range"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sim_dir = multisim.setup_simulation(dct, root, root / "runs")
            subconfig = config.load(sim_dir / "00000000" / "subconfig.yaml")
            self.assertEqual(float(subconfig["source"]["y"]), 0.0)

    def test_invalid_y_range_is_rejected(self):
        for value in ([1.0], [1.0, -1.0], [0.0, float("inf")]):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                dct = p4_config(1)
                dct["multisource"]["y_range"] = value
                root = Path(directory)
                with self.assertRaisesRegex(ValueError, "y_range"):
                    multisim.setup_simulation(dct, root, root / "runs")


class TestP4History(unittest.TestCase):
    def test_canonical_axis_order_for_small_in_memory_history(self):
        history = History()
        history.push(np.full((2, 3), 1.0), 0.1)
        history.push(np.full((2, 3), 2.0), 0.2)
        values = history.get_history()
        self.assertEqual(values.shape, (2, 2, 3))
        np.testing.assert_array_equal(values[:, 0, 0], [1.0, 2.0])
        self.assertEqual(history.metadata()["axis_order"], ["z", "y", "x"])

    def test_streaming_roi_downsample_and_versioned_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.h5"
            history = History(
                path, memory_budget_bytes=0, downsample=(2, 2), roi=(1, 5, 1, 7)
            )
            frame = np.arange(48, dtype=np.float64).reshape(6, 8)
            history.push(frame, 0.1)
            history.push(frame + 100, 0.2)
            self.assertTrue(history.is_streaming)
            self.assertEqual(history.metadata()["shape"], [2, 2, 3])
            history.finalize()
            history.close()
            loaded = load_history(path)
            expected = np.stack([frame[1:5:2, 1:7:2], (frame + 100)[1:5:2, 1:7:2]])
            np.testing.assert_array_equal(loaded.values, expected)
            self.assertEqual(loaded.metadata["axis_order"], ["z", "y", "x"])
            self.assertTrue(loaded.metadata["complete"])

    def test_legacy_axis_migration_does_not_overwrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = np.arange(24).reshape(2, 3, 4)
            np.save(root / "history.npy", legacy)
            np.save(root / "history_z.npy", [0.1, 0.2, 0.3, 0.4])
            loaded = load_history(root)
            self.assertEqual(loaded.values.shape, (4, 2, 3))
            target = migrate_legacy_history(root)
            self.assertEqual(target.name, "history_v2.npy")
            np.testing.assert_array_equal(np.load(root / "history.npy"), legacy)
            migrated = load_history(target)
            np.testing.assert_array_equal(migrated.values, np.transpose(legacy, (2, 0, 1)))

    def test_max_frames_and_cancel_are_hard_gates(self):
        history = History(max_frames=1)
        history.push(np.ones((2, 2)), 0.1)
        with self.assertRaisesRegex(RuntimeError, "maximum frame"):
            history.push(np.ones((2, 2)), 0.2)
        cancelled = History(cancel_token=True)
        with self.assertRaises(InterruptedError):
            cancelled.push(np.ones((2, 2)), 0.1)

    def test_run_writes_centered_coordinates_and_streamed_canonical_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dct = p4_config(1)
            sim_dir = multisim.setup_simulation(dct, root, root / "runs")
            multisim.run_single_simulation(
                sim_dir,
                0,
                root / "scratch",
                history_dz=0.025,
                history_options={"storage": "hdf5", "downsample": [2, 2]},
            )
            subdir = sim_dir / "00000000"
            loaded = load_history(subdir)
            self.assertEqual(loaded.values.shape, (3, 2, 4))
            np.testing.assert_allclose(
                loaded.x, np.array([-8.0, -4.0, 0.0, 4.0]) * 1e-6
            )
            np.testing.assert_allclose(loaded.y, np.array([-6.0, 0.0]) * 1e-6)
            metadata = config.load(subdir / "history_metadata.yaml")
            self.assertEqual(list(metadata["axis_order"]), ["z", "y", "x"])
            self.assertEqual(metadata["storage"], "hdf5")


class DummySource:
    z = 0.0

    def __init__(self, forbid=False):
        self.forbid = forbid

    def propagate_to(self, z_out, params, cutoff_freq, u, U, history):
        if self.forbid:
            raise AssertionError("source must not run while resuming")
        u.vec[:] = 0.0
        U.vec[:] = 0.0


class DummySliceElement:
    z_start = 1.0
    x_positions = np.array([0.0])

    def __init__(self, fail_after_first=False):
        self.fail_after_first = fail_after_first

    def store_deltabetas(self, table):
        pass

    def get_thickness(self):
        return 3.0

    def apply(self, u, U, params, cutoff_freq, stepping_iteration, history):
        start = int(getattr(self, "_checkpoint_start_slice", 0))
        callback = getattr(self, "_checkpoint_callback", None)
        for index in range(start, 3):
            u.vec += 1.0
            U.vec[:] = 2.0 * u.vec
            if callback is not None:
                callback(index + 1, self.z_start + index + 1)
            if self.fail_after_first:
                raise RuntimeError("simulated process interruption")


class TestP4Checkpoint(unittest.TestCase):
    def _params(self):
        return SimParams(
            N=16, nx=4, ny=4, dx=1.0, dy=1.0, z_detector=4.0,
            detector_size=4.0, detector_size_x=4.0, detector_size_y=4.0,
            detector_pixel_size_x=1.0, detector_pixel_size_y=1.0,
            wl=1e-10, chunk_size=8, detector_integrator="area_v1",
        )

    def test_checksum_manifest_restores_both_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            fingerprint = {"N": 8, "dtype": np.dtype(np.complex64).str}
            checkpoint = RunCheckpoint(Path(directory), fingerprint)
            u = NumpyVector(np.arange(8, dtype=np.complex64))
            U = NumpyVector((np.arange(8) * 3j).astype(np.complex64))
            checkpoint.save(
                u, U, current_z=2.5, element_index=1, slice_index=4,
                u_fourier_valid=True, stage="element_slice",
            )
            u.vec[:] = 0
            U.vec[:] = 0
            state = checkpoint.load(u, U)
            np.testing.assert_array_equal(u.vec, np.arange(8, dtype=np.complex64))
            np.testing.assert_array_equal(U.vec, (np.arange(8) * 3j).astype(np.complex64))
            self.assertEqual((state.element_index, state.slice_index), (1, 4))
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.assertNotEqual(
                manifest["vectors"]["u"]["file"].lower(),
                manifest["vectors"]["U"]["file"].lower(),
            )
            vector_path = Path(directory) / manifest["vectors"]["u"]["file"]
            with vector_path.open("r+b") as handle:
                handle.seek(-1, 2)
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "checksum"):
                checkpoint.load(u, U)

    def test_slice_checkpoint_resumes_without_replaying_source_or_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = RunCheckpoint(
                root / "checkpoint", {"N": 16, "dtype": np.dtype(np.complex64).str}
            )
            params = self._params()
            u = NumpyVector(np.zeros(16, dtype=np.complex64))
            U = NumpyVector(np.zeros(16, dtype=np.complex64))
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                wavesim.run_simulation(
                    params, DummySource(), [DummySliceElement(True)], [0.0, 0.0],
                    u, U, [], root, root, False, checkpoint=checkpoint,
                )
            restored_u = NumpyVector(np.zeros(16, dtype=np.complex64))
            restored_U = NumpyVector(np.zeros(16, dtype=np.complex64))
            outputs = wavesim.run_simulation(
                params, DummySource(forbid=True), [DummySliceElement(False)], [0.0, 0.0],
                restored_u, restored_U, [], root, root, False,
                checkpoint=checkpoint, resume_checkpoint=True,
            )
            np.testing.assert_array_equal(restored_u.vec, np.full(16, 3, dtype=np.complex64))
            self.assertEqual(len(outputs), 1)
            with h5py.File(root / "unrelated.h5", "w") as handle:
                handle["ok"] = [1]

    def test_runtime_checkpoint_rejects_phase_snapshot_configuration(self):
        element = DummySliceElement()
        element.x_positions = np.array([0.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = RunCheckpoint(root / "checkpoint", {"N": 16, "dtype": "<c8"})
            with self.assertRaisesRegex(ValueError, "phase stepping"):
                wavesim.run_simulation(
                    self._params(), DummySource(), [element], [0.0, 0.0],
                    NumpyVector(np.zeros(16, dtype=np.complex64)),
                    NumpyVector(np.zeros(16, dtype=np.complex64)),
                    [], root, root, False, checkpoint=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
