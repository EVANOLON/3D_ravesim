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

import config  # noqa: E402,F401
from multisim import compute_fresnel_cutoff_geometry  # noqa: E402
from propagation import SimParams  # noqa: E402


def params() -> SimParams:
    return SimParams(
        N=16 * 8,
        ny=8,
        dx=1.0e-6,
        dy=1.0e-6,
        z_detector=4.8,
        detector_size=0.014,
        detector_size_x=0.014,
        detector_size_y=0.010,
        detector_pixel_size_x=2.0e-6,
        detector_pixel_size_y=2.0e-6,
        wl=1.0e-10,
        chunk_size=16 * 4,
        use_fresnel_scaling=True,
    )


class TestFresnelCutoffGeometry(unittest.TestCase):
    def test_centered_source_uses_sample_to_detector_angle(self):
        angle_x, angle_y, magnification, effective_z = (
            compute_fresnel_cutoff_geometry(
                params(), 0.0, 0.5, (0.0, 0.0), (0.0, 0.0)
            )
        )
        self.assertAlmostEqual(magnification, 9.6, places=14)
        self.assertAlmostEqual(effective_z, 0.5 * 4.3 / 4.8, places=14)
        self.assertAlmostEqual(angle_x, np.arctan(0.007 / 4.3), places=14)
        self.assertAlmostEqual(angle_y, np.arctan(0.005 / 4.3), places=14)

    def test_off_axis_source_tilt_expands_bandwidth(self):
        centered_x, _, _, _ = compute_fresnel_cutoff_geometry(
            params(), 0.0, 0.5, (0.0, 0.0), (0.0, 0.0)
        )
        off_axis_x, _, _, _ = compute_fresnel_cutoff_geometry(
            params(), 0.0, 0.5, (-50.0e-6, 25.0e-6), (0.0, 0.0)
        )
        self.assertGreater(off_axis_x, centered_x)
        self.assertAlmostEqual(
            off_axis_x, np.arctan(0.007 / 4.3 + 50.0e-6 / 0.5), places=14
        )


if __name__ == "__main__":
    unittest.main()
