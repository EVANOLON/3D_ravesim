# PlasmaSample 2D Upgrade Design Report

## 1. Current Status

### What works now (1D)
| Component | Status | File |
|-----------|--------|------|
| `plasma_delta_beta()` physics | ✅ Complete | `big-wave/plasma.py` |
| `PlasmaSample.apply()` 1D | ✅ Working | `big-wave/plasma_sample.py:82-147` |
| Python config parsing | ❌ Missing — `config.py` has no `"plasma_sample"` type | `big-wave/config.py` |
| C++ support | ❌ Not implemented | — |

### What the 2D framework already supports
| Capability | Python (big-wave) | C++ (fast-wave 3d variant) |
|------------|-------------------|---------------------------|
| `propagate_2d()` | ✅ `propagation.py:230` | ✅ `propagate_2d` CUDA kernel |
| `square_and_downsample_2d()` | ✅ `propagation.py:505` | ✅ `square_and_downsample_2d` |
| `Sample._apply_2d()` (bilinear) | ✅ `optical_element.py:502` | ✅ `apply_sample_factors_2d_kernel` |
| Per-pixel deltabeta compute | ❌ PlasmaSample-specific | ❌ PlasmaSample-specific |

---

## 2. Python-side Changes (`big-wave/plasma_sample.py`)

### 2.1 Dataclass: Add 2D fields

```python
@dataclass
class PlasmaSample:
    z_start: float
    pixel_size_x: float
    pixel_size_z: float
    ne_grid: np.ndarray          # shape (nz, nx) for 1D, (nz, ny, nx) for 2D
    ni_grid: np.ndarray
    te_grid: np.ndarray
    zstar_grid: np.ndarray
    Z: int
    x_positions: np.ndarray
    # ── NEW 2D fields ──
    pixel_size_y: float = 0.0
    y_positions: Optional[np.ndarray] = None
```

### 2.2 `check_valid()` — Accept 3D grids

```python
def check_valid(self) -> None:
    shape = self.ne_grid.shape
    ndim = len(shape)
    assert ndim in (2, 3), f"PlasmaSample grids must be 2D (z,x) or 3D (z,y,x), got shape {shape}"
    assert shape[0] > 0
    for name, g in [("ni", self.ni_grid), ("te", self.te_grid), ("zstar", self.zstar_grid)]:
        assert g.shape == shape, f"{name}_grid shape mismatch: {g.shape} != {shape}"
    assert self.z_start > 0
    assert self.pixel_size_x > 0
    if ndim == 3:
        assert self.pixel_size_y > 0
```

### 2.3 `apply()` — Route to 1D or 2D

```python
def apply(self, u, U, sim_params, cutoff_freq, stepping_iteration, history):
    if sim_params.is_2d and self.ne_grid.ndim == 3:
        self._apply_2d(u, U, sim_params, cutoff_freq, stepping_iteration, history)
    else:
        self._apply_1d(u, U, sim_params, cutoff_freq, stepping_iteration, history)
```

### 2.4 New `_apply_2d()` method

**Algorithm** (mirrors `Sample._apply_2d` at `optical_element.py:502`):

```
for each z-slice rowidx:
    1. (optional) push history via square_and_downsample_2d()

    2. Build 2D deltabeta grid (ny × nx):
       for each (iy, ix):
           ne = ne_grid[rowidx, iy, ix]
           if ne <= 0 → deltabeta = 0 + 0j
           else → plasma_delta_beta(ne, ni, Te, Z*, Z, energy)
                  → deltabeta[iy, ix] = delta + 1j * beta

    3. Wavefront modifier (bilinear interpolation):
       for each chunk → map flat index → (ix, iy) → physical (x, y)
         → map to sample grid (x_idx, y_idx)
         → bilinear interpolate deltabeta from 4 nearest neighbours
         → chunk *= material_factor(deltabeta, pixel_size_z, wl)

    4. propagate_2d(u, U, dx, dy, wl, pixel_size_z, chunk_size, cutoff_freq, nx, ny)
```

**Kernel implementation notes:**
- The inner double loop over `(ny, nx)` for per-pixel `plasma_delta_beta()` can be vectorised with numpy: compute ne, ni, Te, Z* arrays, then call `plasma_delta_beta` element-wise.
- For performance, consider caching `plasma_delta_beta` results when multiple z-slices have identical ne, ni, Te, Z* (e.g. steady-state plasma).

### 2.5 Configuration parsing (`big-wave/config.py`)

Add a `"plasma_sample"` branch to `parse_optical_element()`:

```python
elif dct["type"] == "plasma_sample":
    ne = np.load(config_dir / dct["ne_grid_path"])
    ni = np.load(config_dir / dct["ni_grid_path"])
    te = np.load(config_dir / dct["te_grid_path"])
    zs = np.load(config_dir / dct["zstar_grid_path"])
    return PlasmaSample(
        z_start=float(dct["z_start"]),
        pixel_size_x=float(dct["pixel_size_x"]),
        pixel_size_z=float(dct["pixel_size_z"]),
        pixel_size_y=float(dct.get("pixel_size_y", 0.0)),
        ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs,
        Z=int(dct["Z"]),
        x_positions=np.array(dct["x_positions"]),
        y_positions=np.array(dct.get("y_positions", [0.0])),
    )
```

---

## 3. C++-side Changes (`fast-wave - 260428 3d版本/`)

### 3.1 New struct: `PlasmaSample` (`include/optical_element.hpp`)

```cpp
struct PlasmaSample : public OpticalElement {
    // Plasma grids (loaded from .npy files)
    std::vector<float> ne_grid;          // linearised: (nz * ny * nx)
    std::vector<float> ni_grid;          // ion density
    std::vector<float> te_grid;          // electron temperature
    std::vector<float> zstar_grid;       // average ionisation
    int Z;                               // atomic number

    // Pre-computed deltabeta (set at parse time to avoid GPU compute)
    std::vector<Complex<double>> deltabeta_grid;

    std::size_t x_len, y_len, z_len;
    double pixel_size_x, pixel_size_y, pixel_size_z;

    double total_thickness() const override { return pixel_size_z * z_len; }
    std::size_t nr_history_entries() const override { return z_len; }
};
```

### 3.2 New CUDA kernel: `apply_plasma_sample_factors_2d_kernel` (`fwcuda/kernels.cu`)

```cuda
template <typename S>
__global__ void apply_plasma_sample_factors_2d_kernel(
    DevComplex<S> *d_u, SimParams params, double dz,
    float *d_ne, float *d_ni, float *d_te, float *d_zstar,
    DevComplex<double> *d_deltabeta,       // pre-computed or computed on-the-fly
    int Z,
    double pixel_size_x, double pixel_size_y,
    std::size_t x_len, std::size_t y_len,
    int z_slice_index, double x_position, double y_position)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    const int idx = iy * nx + ix;

    // Physical coordinates (centered, with phase step offset)
    const double x = (ix - nx / 2.0) * dx + x_position + pixel_size_x * x_len * 0.5;
    const double y = (iy - ny / 2.0) * dy + y_position + pixel_size_y * y_len * 0.5;

    // Map to sample grid → clamp + bilinear interpolation on ne, ni, Te, Z*
    // → compute delta, beta via plasma_delta_beta()
    // → apply material_factor
}
```

**Two options for the kernel:**

| Option | Pros | Cons |
|--------|------|------|
| **A: Pre-compute CPU-side** — parse time: loop over all pixels, call `plasma_delta_beta()`, upload `deltabeta_grid` to GPU | Simple kernel (exactly like `apply_sample_factors_2d_kernel`), no GPU physics code | Needs full grid in CPU RAM; grid changes per z-slice |
| **B: On-the-fly GPU** — kernel calls `plasma_delta_beta` device function | No pre-compute pass; handles dynamic grids | Need to port `plasma_delta_beta` physics to CUDA (`sqrt`, `exp`, constants) |

**Recommendation**: Start with **Option A** (pre-compute on CPU) for correctness, then add Option B for performance if needed. The `PreciseSample::deltabeta_grid` pattern already implements Option A for density-dependent cold materials.

### 3.3 New parser: `parse_plasma_sample()` (`src/config_parsing.cpp`)

```cpp
[[nodiscard]] PlasmaSample parse_plasma_sample(
    const YAML::Node &node, const fs::path &sim_dir)
{
    // Load 4 .npy grid files
    auto ne_arr = npypp::LoadFull<float>(sim_dir / node["ne_grid_path"].as<std::string>());
    auto ni_arr = npypp::LoadFull<float>(sim_dir / node["ni_grid_path"].as<std::string>());
    auto te_arr = npypp::LoadFull<float>(sim_dir / node["te_grid_path"].as<std::string>());
    auto zs_arr = npypp::LoadFull<float>(sim_dir / node["zstar_grid_path"].as<std::string>());

    // Validate shapes match
    // ...

    // Pre-compute deltabeta_grid (Option A)
    const double energy = ...;  // from subconfig
    std::vector<Complex<double>> db_grid(ne_arr.data.size());
    for (std::size_t i = 0; i < ne_arr.data.size(); ++i) {
        double d, b, _;
        plasma_delta_beta_host(ne_arr.data[i], ni_arr.data[i],
                               te_arr.data[i], zs_arr.data[i],
                               Z, energy, &d, &b, &_);
        db_grid[i] = Complex<double>{d, b};
    }

    // Return populated struct
}
```

**Note**: `plasma_delta_beta_host()` is a pure-C++ host function implementing the same physics as `big-wave/plasma.py`. It needs `sqrt`, `exp`, `pow`, `pi`, physical constants. No SQLite/nist_lookup dependency needed (bound-electron contribution uses the Chantler-scaled fraction already available in the Python code; for C++ this can be simplified to the free-electron term + an optional lookup).

### 3.4 Schedule dispatch (`src/simulation.cpp`)

In `run_simulation_inner_2d()`, add case:

```cpp
case OpticalElementType::PlasmaSample: {
    PlasmaSample* ps = reinterpret_cast<PlasmaSample*>(el);
    apply_plasma_sample_2d<S>(*ps, d_u, d_U, config.sim_params, fft,
                              cutoff_freq_x, cutoff_freq_y, phase_step);
    break;
}
```

### 3.5 New `apply_plasma_sample_2d()` function

```cpp
template <typename S>
void apply_plasma_sample_2d(PlasmaSample ps, DevComplex<S> *d_u, ...) {
    const double dz = ps.pixel_size_z;
    // Upload 4 plasma grids + deltabeta_grid to GPU
    // Loop over z-slices:
    for (std::size_t i = 0; i < ps.z_len; ++i) {
        apply_plasma_sample_factors_2d_kernel<S>(d_u, params, dz,
            d_ne, d_ni, d_te, d_zstar, d_deltabeta, ps.Z,
            ps.pixel_size_x, ps.pixel_size_y,
            ps.x_len, ps.y_len, i,
            ps.x_positions[phase_step], ps.y_positions[phase_step]);
        propagate_2d<S>(params, fft, dz, ...);
    }
}
```

---

## 4. Config format (YAML)

A typical `plasma_sample` element in `config.yaml`:

```yaml
elements:
  - type: plasma_sample
    z_start: 0.01
    pixel_size_x: 2e-7
    pixel_size_y: 2e-7
    pixel_size_z: 1e-6
    ne_grid_path: plasma/ne_grid.npy      # shape (nz, ny, nx) for 2D
    ni_grid_path: plasma/ni_grid.npy
    te_grid_path: plasma/te_grid.npy
    zstar_grid_path: plasma/zstar_grid.npy
    Z: 13
    x_positions: [0.0]
    y_positions: [0.0]
```

---

## 5. Implementation order

| Phase | What | Files | Depends on |
|-------|------|-------|-------|
| **P1** | Python: dataclass + _apply_2d | `plasma_sample.py` | — |
| **P2** | Python: config parsing | `config.py` | P1 |
| **P3** | C++: plasma_delta_beta host function | new `plasma.cpp` in 3d variant | — |
| **P4** | C++: struct + parser | `optical_element.hpp`, `config_parsing.cpp` | P3 |
| **P5** | C++: CUDA kernel | `kernels.cu` | P4 |
| **P6** | C++: schedule integration | `simulation.cpp` | P4, P5 |
| **P7** | Test | notebook, C++ build + run | P2, P6 |

---

## 6. Verification tests

| # | Test | Method | Expected |
|---|------|--------|----------|
| 1 | Python 1D backward compat | `ny=1, ne_grid.ndim=2` → `_apply_1d` | Same result as before |
| 2 | Python 2D self-consistency | 3D grid, run `_apply_2d` with uniform plasma | Phase shift ∝ ne·dz |
| 3 | Python 2D vs 1D regression | `ny=1, ne_grid.ndim=3` shapes → `_apply_2d` | Same as 1D path |
| 4 | Python cross-check vs Sample | Plasma with Z*=0 → neutral atom | Matches `xray_delta_beta` |
| 5 | C++ host physics | Unit test `plasma_delta_beta_host()` | Matches Python values |
| 6 | C++ GPU full pipeline | `fastwave -s 0 ...` on 2D plasma config | `detected.npy` shape `(1, ny_pix, nx_pix)` |

---

## 7. Risk assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Plasma delta/beta GPU kernel complex | High — CUDA debug is slow | Start with CPU pre-compute (Option A) |
| nist_lookup not available in C++ | Medium — bound-electron term missing | Implement simplified: free-electron only for initial release |
| Grid data transfer CPU→GPU large | Medium — 4× float grids per z-slice | Upload all grids once, use pre-computed deltabeta |
| Pipeline integration regression | Low | Keep 1D path untouched, only add new 2D branch |
