import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nist_lookup"))
sys.path.insert(0, str(ROOT / "big-wave"))

import config  # noqa: F401,E402 - establish the big-wave import order
import optical_element
from propagation import SimParams
from vector import NumpyVector


class ConeBeamSampleTests(unittest.TestCase):
    def test_material_coordinates_and_slice_distance_follow_local_scale(self):
        grid = np.broadcast_to(
            np.arange(5, dtype=np.uint32)[None, None, :], (1, 5, 5)
        ).copy()
        sample = optical_element.Sample(
            z_start=1.0,
            pixel_size_x=1.0,
            pixel_size_y=1.0,
            pixel_size_z=0.2,
            grid=grid,
            materials=[optical_element.Material("test", 1.0)] * 4,
            x_positions=np.array([0.0]),
            y_positions=np.array([0.0]),
        )
        sample.db_list = np.arange(5, dtype=np.complex128)
        params = SimParams(
            N=16,
            nx=4,
            ny=4,
            dx=1.0,
            dy=1.0,
            z_detector=3.0,
            detector_size=4.0,
            detector_size_x=4.0,
            detector_size_y=4.0,
            detector_pixel_size_x=1.0,
            detector_pixel_size_y=1.0,
            wl=1.0,
            chunk_size=8,
            use_cone_beam_bpm=True,
        )
        params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)
        u = NumpyVector(np.ones(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        propagation_distances = []

        def capture_propagation(*args):
            propagation_distances.append(args[5])

        with mock.patch.object(
            optical_element, "material_factor", side_effect=lambda db, dz, wl: 1.0 + db
        ), mock.patch.object(
            optical_element, "propagate_2d", side_effect=capture_propagation
        ):
            sample._apply_2d(u, U, params, 0.0, 0, None)

        # At the slice midpoint m=1.1. The central y row therefore samples
        # x indices [0.3, 1.4, 2.5, 3.6], rather than the unscaled values.
        np.testing.assert_allclose(
            u.vec.reshape(4, 4)[2].real, 1.0 + np.array([0.3, 1.4, 2.5, 3.6])
        )
        self.assertEqual(len(propagation_distances), 1)
        self.assertAlmostEqual(propagation_distances[0], 0.2 / 1.2)


if __name__ == "__main__":
    unittest.main()
