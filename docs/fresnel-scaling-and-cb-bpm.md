# Fresnel scaling and cone-beam BPM

The 2D engines expose two explicit point-source acceleration modes under
`sim_params`:

```yaml
sim_params:
  use_fresnel_scaling: true
  use_cone_beam_bpm: false
```

`use_fresnel_scaling: true` is the thin-object mode. It replaces the spherical
source by its equivalent tilted plane wave at the first element, propagates one
effective detector leg, and evaluates the detector at pixel sizes divided by
the geometric magnification. Use it only for one Sample or PlasmaSample whose
internal propagation is negligible for the required accuracy.

For a thick, multislice, or multiple-element object, enable CB-BPM:

```yaml
sim_params:
  use_cone_beam_bpm: true
```

CB-BPM automatically enables Fresnel scaling. It keeps a fixed transverse
reference grid while sampling each material slice at the local cone-beam scale
`m(z) = (z-z_source)/(z_reference-z_source)`. A physical propagation interval
from `z0` to `z1` is mapped to
`(z1-z0)/(m(z0)*m(z1))`. Material transmission still uses the physical slice
thickness. This is the similarity-coordinate form of cone-beam multislice BPM;
it avoids interpolating the complete wavefield after every slice.

Both modes currently require a 2D point source, source < first element <
detector, and Sample/PlasmaSample elements. Thin mode requires exactly one
element. Existing configurations remain unchanged because both switches
default to `false`.

For prepared simulations, `computed.yaml` records `fresnel_mode` and the full
algorithm geometry under `fresnel_geometry`, including the source and
reference planes, magnification, effective detector distance, coordinate
sampling rule, and propagation-step mapping.

## Two-dimensional point-source positions

Both 2D engines use `source.x`, `source.y`, and `source.z` in metres. Direct
propagation shifts the spherical source in both transverse directions; thin
Fresnel and CB-BPM illumination retain both offsets in the linear phase ramp.
For example, a fixed source at y = 1 micrometre is prepared with:

```yaml
multisource:
  type: points
  x_range: [0.0, 0.0]
  y_range: [1.0e-6, 1.0e-6]
```

These are source-sampling fields to merge into a complete configuration.
For each axis the existing Gaussian sampler interprets `[a, b]` as mean
`(a+b)/2` and standard deviation `(b-a)/2`, **not** hard bounds or FWHM.
The Gaussian FWHM is `2*sqrt(2*ln(2))*sigma`. Omitting `y_range` retains y=0.
`computed.yaml` records this convention in `source_sampling` and the actual
realizations in `source_points`; each `subconfig.yaml` records explicit y.

The prepared per-source coordinates are authoritative during execution.
If a 2D subconfig lacks `source.y` but the top-level `y_range` requests a
nonzero position or spread, both engines reject the run before loading the
material grids. Regenerate the simulation directory rather than silently
running a line source or modifying already-computed outputs. A legacy run
with no nonzero y range remains supported with a warning and y=0.

Both engines log the effective x/y/z and write `source_geometry.yaml` beside
`detected.npy`, including units and whether y was explicit or a legacy zero.
The Python detector metadata also embeds this information. Rebuild fastwave
after updating C++ code; old binaries and old prepared directories do not
gain this validation or provenance automatically.

The current 2D kernels use a shared circular frequency cutoff, not independent
x/y cutoffs. Direct-mode preparation now encloses both detector dimensions and
source extents when deriving that scalar and its ray-footprint estimate;
previously only the x cutoff reached the propagator. In direct mode, both axes'
Nyquist checks use the actual shared scalar. `cutoff_angles_y` is diagnostic,
not a second kernel input. Rectangular direct-mode configurations with coarse y
sampling may therefore require a finer `dy` or a larger `ny`.

Thin Fresnel and CB-BPM modes instead validate the magnified effective frame:
the effective detector must fit the wavefront field of view, and the derived
sampling values are recorded in `computed.yaml` under `fresnel_sampling`. The
physical aperture angle is retained there for audit only; applying it as a raw
grid Nyquist frequency would reject valid similarity-coordinate layouts.
Regenerate prepared directories to obtain the corrected metadata and checks;
do not mix old and new results.

Regression: `tests/fresnel_scaling/test_source_y_pipeline.py` checks y-only
image changes, x/y symmetry, source provenance, stale prepared-run rejection,
and CPU/GPU agreement in direct, thin-Fresnel, and CB-BPM modes. GPU tests use
`FASTWAVE_BINARY` when set, otherwise `fast-wave/build-Release/fastwave`.
