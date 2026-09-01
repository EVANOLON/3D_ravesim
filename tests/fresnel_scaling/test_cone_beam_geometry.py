import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nist_lookup"))
sys.path.insert(0, str(ROOT / "big-wave"))

from config import parse_sim_params
from multisim import validate_fresnel_mode
from propagation import SimParams


def make_params(**overrides):
    values = dict(
        N=16,
        dx=1e-6,
        z_detector=3.0,
        detector_size=16e-6,
        detector_pixel_size_x=1e-6,
        detector_pixel_size_y=1e-6,
        wl=1e-10,
        chunk_size=16,
    )
    values.update(overrides)
    return SimParams(**values)


class ConeBeamGeometryTests(unittest.TestCase):
    def test_cb_bpm_enables_fresnel_and_maps_coordinates(self):
        params = make_params(use_cone_beam_bpm=True)
        params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)

        self.assertTrue(params.use_fresnel_scaling)
        self.assertAlmostEqual(params.fresnel_magnification, 3.0)
        self.assertAlmostEqual(params.fresnel_transverse_scale(1.5), 1.5)
        self.assertAlmostEqual(params.effective_slice_dz(1.0, 0.5), 1.0 / 3.0)
        self.assertAlmostEqual(params.effective_final_dz(1.5), 1.0 / 3.0)

    def test_thin_fresnel_keeps_single_effective_detector_leg(self):
        params = make_params(use_fresnel_scaling=True)
        params.configure_fresnel_detector(z_source=0.0, z_sample=1.0)

        self.assertEqual(params.fresnel_transverse_scale(2.0), 1.0)
        self.assertEqual(params.effective_slice_dz(1.0, 0.5), 0.5)
        self.assertAlmostEqual(params.effective_final_dz(1.25), 2.0 / 3.0)

    def test_config_cb_bpm_boolean_is_strict_and_implies_fresnel(self):
        base = dict(
            N=16,
            dx=1e-6,
            z_detector=3.0,
            detector_size=16e-6,
            detector_pixel_size_x=1e-6,
            detector_pixel_size_y=1e-6,
            chunk_size=16,
            use_cone_beam_bpm="true",
        )
        parsed = parse_sim_params(base)
        self.assertTrue(parsed.use_cone_beam_bpm)
        self.assertTrue(parsed.use_fresnel_scaling)

        base["use_cone_beam_bpm"] = "sometimes"
        with self.assertRaises(ValueError):
            parse_sim_params(base)

    def test_mode_validation_separates_thin_and_cb_bpm_scope(self):
        Sample = type("Sample", (), {})
        Grating = type("Grating", (), {})

        thin = make_params(ny=4, use_fresnel_scaling=True)
        validate_fresnel_mode(thin, [Sample()], "points")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_fresnel_mode(thin, [Sample(), Sample()], "points")

        cb_bpm = make_params(ny=4, use_cone_beam_bpm=True)
        validate_fresnel_mode(cb_bpm, [Sample(), Sample()], "points")
        with self.assertRaisesRegex(ValueError, "Sample/PlasmaSample"):
            validate_fresnel_mode(cb_bpm, [Grating()], "points")
        with self.assertRaisesRegex(ValueError, "point sources"):
            validate_fresnel_mode(cb_bpm, [Sample()], "vectors")


if __name__ == "__main__":
    unittest.main()
