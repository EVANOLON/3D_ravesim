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
