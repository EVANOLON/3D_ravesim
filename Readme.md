# RAVE-SIM: Really big/fast wAVE SIMulation

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.543500-blue)](https://doi.org/10.1364/OE.543500)
![GitHub last commit](https://img.shields.io/github/last-commit/EVANOLON/3D_ravesim)
![License](https://img.shields.io/github/license/EVANOLON/3D_ravesim)

RAVE-SIM is an X-ray wave propagation simulation framework originally developed at ETH Zurich and described in the 2025 **Optics Express** article *Simulation Framework for X-ray Grating Interferometry Optimization* ([DOI: 10.1364/OE.543500](https://doi.org/10.1364/OE.543500)). It simulates coherent X-rays traveling from a point source through optical elements (gratings and samples) and free-space propagation to a detector. This repository extends the original framework with CUDA-accelerated 2D wave-field propagation, 3D voxelized samples, 2D area detectors, and plasma-sample modeling.

## Extensions in This Repository

The following capabilities were added and integrated by **[Sijie Fan (EVANOLON)](https://github.com/EVANOLON)**:

- Extended `fast-wave` from one-dimensional wave-field simulation to two-dimensional transverse wave fields using CUDA-accelerated 2D FFT propagation
- Added support for three-dimensional voxelized samples and two-dimensional area detectors
- Added grid-generation and simulation workflows for three-dimensional spherical shells, non-spherical shells, and capsule targets
- Implemented laser-produced-plasma optical elements (`PlasmaSample`), including free-electron dispersion, bound-electron contributions, and inverse-bremsstrahlung absorption
- Added Python and C++/CUDA support for two-dimensional `PlasmaSample` simulations
- Added validation tests and example notebooks for 2D propagation, 3D samples, plasma samples, and detector outputs

In this documentation, **3D simulation** refers to propagating a two-dimensional transverse wave field through a three-dimensional sample volume. It does not imply a full three-dimensional electromagnetic-field solver.

## Features

- **Two simulation engines:**
  - **big-wave** (Python + Rust) — Out-of-core computation using disk-backed wave fields, supporting arbitrarily large 1D simulations
  - **fast-wave** (C++ / CUDA) — GPU-accelerated simulation supporting both 1D and 2D wave fields
- **1D simulation** — Mature and fully tested for line-grating interferometers (Talbot-Lau, Talbot)
- **2D simulation** — Supports 3D samples and 2D area detectors (verified in fast-wave)
- **Plasma samples** — Laser-produced plasma optical elements with free-electron dispersion, bound-electron (Chantler) contributions, and Kramers inverse-bremsstrahlung absorption
- **Out-of-core FFT** — Rust-based four-step FFT algorithm for memory-efficient large transforms
- **Config-driven** — YAML-based configuration shared across engines
- **Multi-source simulation** — Parallel simulation of multiple source points with spectral sampling

## Repository Structure

```
.
├── big-fourier/          # Rust out-of-core FFT library + Python bindings
│   └── src/              #   Rust source (bfft, chunkmat, npy)
├── big-wave/             # Python wave simulation engine
│   ├── wavesim.py        #   Core simulation logic
│   ├── propagation.py    #   Wave propagation (Fresnel diffraction)
│   ├── optical_element.py #   Gratings, samples, etc.
│   ├── plasma_sample.py  #   Laser-produced plasma optical element
│   ├── plasma.py         #   Plasma δ/β physics (free + bound + Kramers)
│   ├── multisim.py       #   Multi-source simulation orchestration
│   ├── vector.py         #   NumpyVector / DiskVector abstraction
│   └── config.py         #   YAML configuration parser
├── fast-wave/            # C++/CUDA GPU simulation engine
│   ├── fwcuda/           #   CUDA kernels (kernels.cu)
│   ├── src/              #   C++ simulation logic (simulation.cpp, fft.cpp)
│   ├── include/          #   C++ headers
│   ├── test/             #   CUDA unit tests (fwtest.cu)
│   └── CMakeLists.txt    #   CMake build configuration
├── grid/                 # Sample grid generation utilities
├── nist_lookup/          # Material properties database (δ, β lookup)
├── examples/             # Small supported examples
├── notebooks/            # Jupyter research examples and legacy workflows
├── post-processing/      # Analysis scripts
├── CMakeLists.txt        # Convenience entry point forwarding to fast-wave/
├── Cargo.toml            # Rust workspace
└── requirements.txt      # Python dependencies
```

## Physics Background

Wave propagation in free space is modeled by **Fresnel diffraction (paraxial approximation)**:

```
u(x, z+dz) = IFFT[ FFT[u(x,z)] × H(f) ]
H(f) = exp(+2π·i·dz/λ) × exp(-π·i·λ·dz·f²)
```

When passing through matter, the wave interacts with the material as:

```
u_out = u_in × exp(-2π·i·t/λ · (δ + iβ)⁺)
     = u_in × exp(-2π·i·δ·t/λ) × exp(-2π·β·t/λ)
```

where δ is the refractive index decrement (phase shift) and β is the absorption index, obtained from the NIST database. The `⁺` denotes complex conjugation of the δ + iβ value.

The spatial phase convention is `exp(+i·k·z)` (positive spatial phase), consistent across both big-wave and fast-wave engines. The point-source spherical wave is initialized as `u(x) = exp(+2π·i·r/λ) / r` in 2D (or `/√r` in 1D), where `r = √(x² + z²)`.

## Quick Start

### Python environment
```bash
pip install -r requirements.txt
```

### Supported minimal example

Run the data-free 2D XPCI example from the repository root:

```bash
python examples/minimal_xpci/run.py
```

This is the recommended first run. It uses the Python `big-wave` engine in
memory, generates its sample data, validates the detector result, and writes
ignored artifacts under `output/minimal_xpci/`.

### fast-wave CUDA build

```bash
cmake -S fast-wave -B fast-wave/build-Release -DCMAKE_BUILD_TYPE=Release
cmake --build fast-wave/build-Release
```

After building, compare the supported example across both engines:

```bash
python examples/minimal_xpci/compare_fast.py
```

Lower-level production runs consume a prepared simulation directory containing
`config.yaml`, `computed.yaml`, and per-source `subconfig.yaml` files. The
`rave_agent` and `bridge` workflows prepare and validate those directories;
`big-wave/main.py` is a source-level development demonstration, not a
configuration-file command-line entry point.

### Notebooks
For a quick, data-free 2D XPCI run, start with:
```bash
python examples/minimal_xpci/run.py
```

Additional research and legacy notebooks are catalogued in
[`notebooks/README.md`](notebooks/README.md). They are not all supported release
entry points and some require external data or a CUDA build. Launch Jupyter with:
```bash
jupyter lab notebooks/
```

## Citation

If you use this extended software in your research, cite the software release
described in [`CITATION.cff`](CITATION.cff) and the original framework article:

```bibtex
@article{Spindler2025Simulation,
  author={Spindler, Simon and Pereira, Alexandre and Sommer, Pascal and Rawlik, M. and Romano, L. and Stampanoni, Marco},
  title={Simulation Framework for X-ray Grating Interferometry Optimization},
  journal={Optics Express},
  volume={33},
  number={1},
  pages={1345},
  doi={10.1364/OE.543500},
  year={2025},
  publisher={Optica Publishing Group}
}
```

## License

The main RAVE-SIM code is distributed under the BSD 3-Clause License. Copyright
(c) 2024, ETH Zurich.

See the [LICENSE](LICENSE) file for details. Bundled third-party components keep
their own licenses; in particular, `nist_lookup/` is distributed under its
included GNU GPL v3 license.

## Original Project and Related Publications

- Framework paper: [Optics Express, DOI: 10.1364/OE.543500](https://doi.org/10.1364/OE.543500)
- The original framework contains code developed for Pascal Sommer's master's thesis and Alexandre Vieira Pereira's and Simon Spindler's PhD theses at ETH Zurich.
- The extensions listed above are specific to this repository and are not claimed as part of the original publication.
