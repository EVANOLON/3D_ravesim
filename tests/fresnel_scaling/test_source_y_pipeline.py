"""Prepared 2D source coordinates: validation, provenance and actual propagation."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "nist_lookup", ROOT / "big-wave"):
    sys.path.insert(0, str(path))

import config
import multisim

FASTWAVE = Path(os.environ.get("FASTWAVE_BINARY", ROOT / "fast-wave/build-Release/fastwave"))


def case(mode="off", x=0.0, y=1e-6):
    return {
        "dtype": "c8", "use_disk_vector": False, "save_final_u_vectors": False,
        "sim_params": {
            "is2d": True, "N": 128 * 128, "nx": 128, "ny": 128,
            "dx": 1e-6, "dy": 1e-6, "z_detector": 0.4,
            "detector_size": 32e-6, "detector_size_y": 32e-6,
            "detector_pixel_size_x": 2e-6, "detector_pixel_size_y": 2e-6,
            "chunk_size": 128 * 16,
            "use_fresnel_scaling": mode != "off", "use_cone_beam_bpm": mode == "cb",
        },
        "runtime": {"big_wave": {
            "memory_budget_gb": 1.0, "chunk_size": 128 * 16,
            "fft2_backend": "scipy_in_memory", "detector_integrator": "area_v1",
        }},
        "multisource": {
            "type": "points", "nr_source_points": 1, "seed": 17,
            "energy_range": [1000.0, 1000.0], "z": 0.0,
            "x_range": [x, x], "y_range": [y, y],
        },
        "elements": [{
            "type": "sample", "z_start": 0.2,
            "pixel_size_x": 1e-6, "pixel_size_y": 1e-6, "pixel_size_z": 1e-6,
            "grid_path": "sample.npy", "materials": [["C", 1.5]],
            "x_positions": [0.0], "y_positions": [0.0],
        }],
    }


def prepare(root, dct):
    # A square specimen makes exchanging x and y an exact geometry symmetry.
    np.save(root / "sample.npy", np.ones((4, 12, 12), dtype=np.uint32))
    return multisim.setup_simulation(dct, root, root / "runs")


class TestSourceYMetadata(unittest.TestCase):
    def test_explicit_y_and_units(self):
        result = multisim.point_source_geometry_2d(
            {"x": 1e-6, "y": -2e-6, "z": 0.1}, {"y_range": [-3e-6, 3e-6]})
        self.assertEqual(result["y"], -2e-6)
        self.assertEqual(result["units"], "m")
        self.assertEqual(result["y_origin"], "explicit")

    def test_legacy_zero_is_explicitly_reported(self):
        with self.assertLogs("big-wave", level="WARNING"):
            result = multisim.point_source_geometry_2d({"x": 0, "z": 0}, {})
        self.assertEqual(result["y"], 0)
        self.assertEqual(result["y_origin"], "legacy_default_zero")

    def test_missing_y_with_requested_extent_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regenerate"):
            multisim.point_source_geometry_2d({"x": 0, "z": 0}, {"y_range": [-1e-6, 1e-6]})

    def test_nonfinite_coordinates_are_rejected(self):
        for axis in ("x", "y", "z"):
            point = {"x": 0, "y": 0, "z": 0}
            point[axis] = float("nan")
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "finite"):
                multisim.point_source_geometry_2d(point, {})


class TestSourceYPropagation(unittest.TestCase):
    def test_fresnel_modes_record_effective_sampling_instead_of_raw_nyquist(self):
        for mode in ("thin", "cb"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                dct = case(mode, x=1e-6, y=0)
                dct["sim_params"]["dy"] = 8e-6
                dct["sim_params"]["detector_size_y"] = 4e-6
                run = prepare(Path(directory), dct)
                sampling = config.load(run / "computed.yaml")["fresnel_sampling"]
                self.assertFalse(sampling["raw_nyquist_check_applies"])
                self.assertEqual(
                    sampling["mode"], "fresnel_similarity_effective_frame_v1"
                )
                self.assertLessEqual(
                    sampling["detector_fov_to_wavefront_fov"][0], 1.0
                )
                self.assertLessEqual(
                    sampling["detector_fov_to_wavefront_fov"][1], 1.0
                )

    def test_y_changes_image_and_obeys_xy_symmetry_in_all_modes(self):
        for mode in ("off", "thin", "cb"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                images = {}
                for label, x, y in (("zero", 0, 0), ("x", 1e-6, 0), ("y", 0, 1e-6)):
                    run = prepare(root, case(mode, x, y))
                    multisim.run_single_simulation(run, 0, root / (label + "_scratch"))
                    sub = run / "00000000"
                    images[label] = np.load(sub / "detected.npy").squeeze()
                    metadata = config.load(sub / "source_geometry.yaml")
                    self.assertEqual(metadata["y"], y)
                    self.assertEqual(metadata["y_origin"], "explicit")
                    self.assertEqual(config.load(sub / "detector_metadata.yaml")["source_geometry"], metadata)
                    computed = config.load(run / "computed.yaml")
                    self.assertEqual(computed["source_points"][0]["y"], y)
                    self.assertEqual(computed["source_sampling"]["y_mean"], y)
                    self.assertEqual(computed["source_sampling"]["y_sigma"], 0)
                delta = np.linalg.norm(images["y"] - images["zero"]) / np.linalg.norm(images["zero"])
                self.assertGreater(delta, 1e-5)
                error = np.linalg.norm(images["y"] - images["x"].T) / np.linalg.norm(images["y"])
                self.assertLess(error, 2e-5)

    def test_old_prepared_run_fails_before_loading_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = prepare(root, case("cb"))
            path = run / "00000000/subconfig.yaml"
            sub = config.load(path)
            del sub["source"]["y"]
            config.save(path, sub)
            (run / "sample.npy").unlink()  # prove validation precedes material loading
            with self.assertRaisesRegex(ValueError, "source.y is missing"):
                multisim.run_single_simulation(run, 0, root / "scratch")

    @unittest.skipUnless(FASTWAVE.is_file(), "fastwave binary not available")
    def test_gpu_y_metadata_and_cpu_agreement(self):
        for mode in ("off", "thin", "cb"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = prepare(root, case(mode, x=-0.5e-6, y=1e-6))
                gpu = root / "gpu"
                shutil.copytree(run, gpu)
                multisim.run_single_simulation(run, 0, root / "scratch")
                proc = subprocess.run([str(FASTWAVE), str(gpu), "-s", "0"],
                                      capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIn("2D point source:", proc.stdout + proc.stderr)
                self.assertEqual(config.load(run / "00000000/source_geometry.yaml"),
                                 config.load(gpu / "00000000/source_geometry.yaml"))
                a = np.load(run / "00000000/detected.npy")
                b = np.load(gpu / "00000000/detected.npy")
                error = np.linalg.norm(a - b) / np.linalg.norm(a)
                print(f"source-y {mode}: CPU/GPU relative L2={error:.6g}")
                self.assertLess(error, 2e-4)
                path = gpu / "00000000/subconfig.yaml"
                sub = config.load(path)
                del sub["source"]["y"]
                config.save(path, sub)
                proc = subprocess.run([str(FASTWAVE), str(gpu), "-s", "0"],
                                      capture_output=True, text=True, timeout=60)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("source.y is missing", proc.stdout + proc.stderr)

                # Legacy on-axis-y sources remain executable, with provenance.
                path = gpu / "config.yaml"
                dct = config.load(path)
                del dct["multisource"]["y_range"]
                config.save(path, dct)
                proc = subprocess.run([str(FASTWAVE), str(gpu), "-s", "0"],
                                      capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                metadata = config.load(gpu / "00000000/source_geometry.yaml")
                self.assertEqual(metadata["y"], 0)
                self.assertEqual(metadata["y_origin"], "legacy_default_zero")
                self.assertIn("legacy y=0", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
