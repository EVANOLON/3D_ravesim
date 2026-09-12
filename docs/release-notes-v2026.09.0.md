# RAVE-SIM v2026.09.0

This release consolidates the extended RAVE-SIM implementation for
two-dimensional X-ray wave propagation, three-dimensional voxel samples,
plasma samples, cone-beam/Fresnel geometry, and detector imaging.

## Highlights

- Added CUDA-accelerated 2D transverse wave-field propagation and 2D FFT paths.
- Added 3D voxelized `Sample` and 2D `PlasmaSample` propagation.
- Added cone-beam BPM and Fresnel effective-geometry support across both engines.
- Added transactional, restartable out-of-core 2D FFT support to `big-fourier`.
- Added memory-bounded 2D operators, detector integration, History v2, and
  runtime checkpoints to `big-wave`.
- Made the area-weighted `area_v1` detector integrator the `big-wave` default;
  the current `fast-wave` CUDA detector uses the same overlap-area semantics.
- Retained `legacy_fastwave` only for reproducing historical truncation-based
  detector outputs.
- Added validation, feasibility, plotting, and Multi1D++ bridge tooling under
  `rave_agent/` and `bridge/`.
- Added the data-free `examples/minimal_xpci/` supported quick start.

## Detector correction

The historical detector kernel used integer-truncated grid-point counts. For
non-integer detector-pixel/grid ratios this could create a periodic count-map
ripple and biased border pixels. Both production engines now use area-weighted
overlap integration. The validator rejects explicit `legacy_fastwave` use with
non-integer ratios, and the historical correction utility requires an explicit
`--legacy-output` confirmation.

## Validation performed

- `tests.big_wave_2d.test_p2`, `test_p3`, and `test_p4`: 33/33 passed.
- Minimal in-memory XPCI example: passed, producing a 248 x 248 detector image.
- `big-wave`/`fast-wave` CUDA comparison: relative L2 error
  `1.504239483e-05` (acceptance threshold `2e-3`).
- A known P0 failure was reproduced on the pre-change baseline and is not
  attributed to the detector-integrator migration; a new aggregate full-suite
  pass count is therefore not claimed here.

## Reproducibility and notebooks

Supported release entry points are kept under `examples/`. The notebooks are
research and legacy workflow records; many require external data, local paths,
or a CUDA build. Paper-scale datasets and generated wave fields are not bundled
with this source release.

## Compatibility notes

- Existing configurations that omit `detector_integrator` now resolve to
  `area_v1`.
- Set `legacy_fastwave` explicitly only when reproducing historical `big-wave`
  output and only with a compatible integer detector-pixel/grid ratio.
- `nist_lookup/` remains a separately licensed bundled component under GNU GPL
  v3; the main RAVE-SIM source is BSD-3-Clause.
