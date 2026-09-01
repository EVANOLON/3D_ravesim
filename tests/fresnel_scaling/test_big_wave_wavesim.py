import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BIG_WAVE = REPO_ROOT / "big-wave"
NIST_LOOKUP = REPO_ROOT / "nist_lookup"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NIST_LOOKUP) not in sys.path:
    sys.path.insert(0, str(NIST_LOOKUP))
if str(BIG_WAVE) not in sys.path:
    sys.path.insert(0, str(BIG_WAVE))

import config  # noqa: E402,F401
import wavesim  # noqa: E402
from propagation import SimParams  # noqa: E402
from vector import NumpyVector  # noqa: E402


class PlaneSource:
    z = 0.0

    def propagate_to(self, z_out, params, cutoff_freq, u, U, history):
        u.vec[:] = 1.0
        u.fft2(U, params.nx, params.ny)


class EmptyElement:
    z_start = 1.0
    x_positions = [0.0]

    def store_deltabetas(self, table):
        pass

    def get_thickness(self):
        return 0.0

    def apply(self, u, U, params, cutoff_freq, stepping_iteration, history):
        u.fft2(U, params.nx, params.ny)


class ThickEmptyElement(EmptyElement):
    def get_thickness(self):
        return 0.5


class TestBigWaveFresnelPropagation(unittest.TestCase):
    def test_final_leg_uses_effective_distance(self):
        params = SimParams(
            N=8 * 8,
            nx=8,
            ny=8,
            dx=1.0e-6,
            dy=1.0e-6,
            z_detector=5.0,
            detector_size=8.0e-6,
            detector_size_x=8.0e-6,
            detector_size_y=8.0e-6,
            detector_pixel_size_x=1.0e-6,
            detector_pixel_size_y=1.0e-6,
            wl=1.0e-10,
            chunk_size=8 * 4,
            use_fresnel_scaling=True,
        )
        params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)
        u = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        propagated_distances = []

        def capture_propagation(
            u, U, sim_params, dz, cutoff_freq, current_z,
            skip_fft=False, history=None,
        ):
            propagated_distances.append(dz)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            wavesim, "propagate_with_history_2d", side_effect=capture_propagation
        ), mock.patch.object(
            wavesim, "square_and_downsample_2d", return_value=np.zeros((1, 1))
        ):
            root = Path(directory)
            wavesim.run_simulation(
                params=params,
                source=PlaneSource(),
                elements=[EmptyElement()],
                cutoff_angles=[0.01, 0.01],
                u=u,
                U=U,
                deltabeta_table=[],
                sub_dir=root,
                vectors_dir=root,
                save_final_u_vectors=False,
            )

        self.assertEqual(len(propagated_distances), 1)
        self.assertAlmostEqual(
            propagated_distances[0], params.fresnel_effective_z, places=15
        )
        self.assertNotAlmostEqual(propagated_distances[0], 4.0, places=6)

    def test_cb_bpm_final_leg_starts_at_sample_exit_scale(self):
        params = SimParams(
            N=8 * 8,
            nx=8,
            ny=8,
            dx=1.0e-6,
            dy=1.0e-6,
            z_detector=5.0,
            detector_size=8.0e-6,
            detector_size_x=8.0e-6,
            detector_size_y=8.0e-6,
            detector_pixel_size_x=1.0e-6,
            detector_pixel_size_y=1.0e-6,
            wl=1.0e-10,
            chunk_size=8 * 4,
            use_cone_beam_bpm=True,
        )
        params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)
        u = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        propagated_distances = []

        def capture_propagation(
            u, U, sim_params, dz, cutoff_freq, current_z,
            skip_fft=False, history=None,
        ):
            propagated_distances.append(dz)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            wavesim, "propagate_with_history_2d", side_effect=capture_propagation
        ), mock.patch.object(
            wavesim, "square_and_downsample_2d", return_value=np.zeros((1, 1))
        ):
            root = Path(directory)
            wavesim.run_simulation(
                params=params,
                source=PlaneSource(),
                elements=[ThickEmptyElement()],
                cutoff_angles=[0.01, 0.01],
                u=u,
                U=U,
                deltabeta_table=[],
                sub_dir=root,
                vectors_dir=root,
                save_final_u_vectors=False,
            )

        self.assertEqual(len(propagated_distances), 1)
        self.assertAlmostEqual(propagated_distances[0], 3.5 / (1.5 * 5.0))


if __name__ == "__main__":
    unittest.main()
