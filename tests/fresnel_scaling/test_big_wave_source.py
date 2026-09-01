import sys
import unittest
from pathlib import Path

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

import config  # noqa: E402,F401  # establish the repository's import order
from propagation import SimParams  # noqa: E402
from source import PointSource  # noqa: E402
from vector import NumpyVector  # noqa: E402


def fresnel_params() -> SimParams:
    params = SimParams(
        N=16 * 8,
        nx=16,
        ny=8,
        dx=2.0e-6,
        dy=3.0e-6,
        z_detector=5.0,
        detector_size=1.0e-3,
        detector_size_x=1.0e-3,
        detector_size_y=1.0e-3,
        detector_pixel_size_x=10.0e-6,
        detector_pixel_size_y=10.0e-6,
        wl=1.0e-10,
        chunk_size=16 * 4,
        use_fresnel_scaling=True,
    )
    params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)
    return params


class TestBigWaveFresnelSource(unittest.TestCase):
    def test_centered_source_becomes_normalized_plane_wave(self):
        params = fresnel_params()
        u = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))

        PointSource(x=0.0, y=0.0, z=0.0).propagate_to(
            1.0, params, 1.0 / params.dx, u, U, None
        )

        np.testing.assert_allclose(u.vec, 1.0 + 0.0j, rtol=0.0, atol=1e-13)

    def test_off_axis_source_leaves_expected_linear_phase(self):
        params = fresnel_params()
        source = PointSource(x=25.0e-6, y=-30.0e-6, z=0.0)
        u = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))

        source.propagate_to(1.0, params, 1.0 / params.dx, u, U, None)

        iy, ix = np.indices((params.ny, params.nx))
        x = (ix - params.nx / 2.0) * params.dx
        y = (iy - params.ny / 2.0) * params.get_dy()
        expected = np.exp(
            -2j * np.pi * (x * source.x + y * source.y) / params.wl
        )
        np.testing.assert_allclose(
            u.vec.reshape(params.ny, params.nx), expected, rtol=0.0, atol=2e-12
        )

    def test_requires_configured_effective_geometry(self):
        params = fresnel_params()
        params.fresnel_effective_z = 0.0
        u = NumpyVector(np.zeros(params.N, dtype=np.complex128))
        U = NumpyVector(np.zeros(params.N, dtype=np.complex128))

        with self.assertRaisesRegex(ValueError, "has not been configured"):
            PointSource(x=0.0, y=0.0, z=0.0).propagate_to(
                1.0, params, 1.0 / params.dx, u, U, None
            )


if __name__ == "__main__":
    unittest.main()
