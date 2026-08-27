# RAVE-SIM: Really big/fast wAVE SIMulation

[![DOI](https://img.shields.io/badge/DOI-10.1364%2FOE.543500-blue)](https://doi.org/10.1364/OE.543500)
![GitHub last commit](https://img.shields.io/github/last-commit/EVANOLON/3D_ravesim)
![License](https://img.shields.io/github/license/EVANOLON/3D_ravesim)

RAVE-SIM is an X-ray wave propagation simulation framework originally developed at ETH Zurich, published in **Optics Express** ([DOI: 10.1364/OE.543500](https://doi.org/10.1364/OE.543500)). It simulates coherent X-rays traveling from a point source through optical elements (gratings, samples) and free-space propagation to a detector.
3D calculation mode was added to it. The fast-wave 3D mode has been verified.

## Features

- **Two simulation engines:**
  - **big-wave** (Python + Rust) — Out-of-core computation using disk-backed wave fields, supporting arbitrarily large 1D simulations
  - **fast-wave** (C++ / CUDA) — GPU-accelerated simulation supporting both 1D and 2D wave fields
- **1D simulation** — Mature and fully tested for line-grating interferometers (Talbot-Lau, Talbot)
- **2D simulation** — In development, supports 3D samples and 2D area detectors
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
├── notebooks/            # Jupyter example notebooks
├── post-processing/      # Analysis scripts
├── CMakeLists.txt        # fast-wave build (root symlink)
├── Cargo.toml            # Rust workspace
└── requirements.txt      # Python dependencies
```

## Physics Background

Wave propagation in free space is modeled by **Fresnel diffraction (paraxial approximation)**:

```
u(x, z+dz) = IFFT[ FFT[u(x,z)] × H(f) ]
H(f) = exp(-2π·i·dz/λ) × exp(π·i·λ·dz·f²)
```

When passing through matter, the wave interacts with the material as:

```
u_out = u_in × exp(2π·i·t/λ · (δ + iβ))
```

where δ is the refractive index decrement (phase shift) and β is the absorption index, obtained from the NIST database.

## Quick Start

### Python environment
```bash
pip install -r requirements.txt
```

### big-wave (Python)
```bash
cd big-wave
python main.py config.yaml
```

### fast-wave (CUDA GPU)
```bash
cd fast-wave
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/fastwave <config_dir> --source_idx 0
```

### Multi-source simulation
```bash
cd big-wave
python multisim.py config.yaml
```

### Notebooks
Example notebooks are available in the `notebooks/` directory. Launch Jupyter:
```bash
jupyter lab notebooks/
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{Ravesim2024,
  title={RAVE-SIM: Really big/fast wAVE SIMulation},
  journal={Optics Express},
  doi={10.1364/OE.543500},
  year={2024},
  author={Sommer, Pascal and Vieira Pereira, Alexandre and Spindler, Simon and others},
  publisher={Optica Publishing Group}
}
```

## License

Copyright (c) 2024, ETH Zurich. All rights reserved.

See the [LICENSE](LICENSE) file for details.

## Related Publications

- Framework paper: [Optics Express, DOI: 10.1364/OE.543500](https://doi.org/10.1364/OE.543500)
- Contains code for Pascal Sommer's master thesis as well as Alexandre Vieira Pereira's and Simon Spindler's PhD theses at ETH Zurich.
