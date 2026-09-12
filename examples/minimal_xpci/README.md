# Minimal 2D XPCI example

This example propagates an 8 keV plane wave through a small voxelized
carbon sphere, compares the detector intensity with free space, and saves the
result under the repository's ignored `output/` directory. It generates all
inputs in memory and does not require CUDA or external research data.

From the repository root:

```bash
python examples/minimal_xpci/run.py
```

Expected behavior:

- the run completes in a few seconds on a typical workstation;
- `output/minimal_xpci/result.npz` contains sample, reference, difference, and
  relative-contrast detector arrays;
- `output/minimal_xpci/preview.png` is written when Matplotlib is available;
  rendering uses a headless backend and does not require `DISPLAY` or an X server;
- the command prints `PASS` after checking that the output is finite, 2D, and
  measurably different from free-space propagation.

Use `--output-dir PATH` to choose another output location.

## CUDA cross-check

If `fast-wave/build-Release/fastwave` is already built, run the same compact
point-source/carbon-sphere configuration through both engines:

```bash
python examples/minimal_xpci/compare_fast.py
```

The command saves both detector arrays, a difference image, and the fast-wave
logs under `output/minimal_xpci/fast_compare_<timestamp>/`. It passes when the
detector arrays have the same shape and their relative L2 difference is below
`2e-3`. The current fast-wave CUDA detector and big-wave's `area_v1` integrator
use the same area-weighted detector semantics.
