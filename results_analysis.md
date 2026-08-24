# RAVE-SIM Thin-Sample Validation Report

> **Date**: 2026-07-09
> **Engine**: RAVE-SIM (fast-wave CUDA C++), NVIDIA GeForce RTX 5070 Ti Laptop GPU
> **Config**: is2d, N=16384×16384, dx=85nm, z_detector=4.0m, detector 0.85mm@200nm

---

## 1. Overview

This report validates the RAVE-SIM simulation engine against fundamental X-ray physics using thin square plates:

| Validation | Material | Thickness | Physics Law | Criterion |
|---|---|---|---|---|
| Absorption | W (19.35 g/cm³) | 1 / 2 / 5 μm | Beer-Lambert: I/I₀ = exp(-μ·t) | μ deviation < 5% |
| Phase | C (1.5 g/cm³) | 10 / 20 μm | Phase shift: Δφ = -2π·δ·t/λ | Slope deviation < 10% |

---

## 2. Absorption Validation: W Plate (Beer-Lambert Law)

### 2.1 Setup

- W square plate 200 μm × 200 μm, thickness 1 / 2 / 5 μm
- 10 keV monochromatic, point source, single source point
- I₀ measured from empty-field reference; I measured from sample center ROI
- Theory: I/I₀ = exp(-μ·t), μ = 4π·β/λ
- β_W from NIST database (Chantler scattering factors)

### 2.2 Results

| Thickness (μm) | I/I₀ (sim) | I/I₀ (theory) | Ratio |
|---|---|---|---|
| 1 | 0.8372 | 0.8346 | 1.0032 |
| 2 | 0.7027 | 0.6966 | 1.0089 |
| 5 | 0.4098 | 0.4049 | 1.0120 |

### 2.3 Attenuation Coefficient

| Parameter | Value |
|---|---|
| β_W (10 keV) | 1.7839×10⁻⁶ |
| λ | 1.2398×10⁻¹⁰ m |
| μ_theory | 1.8080×10⁵ m⁻¹ (= 1808 cm⁻¹) |
| μ_fitted ± 1σ | (1.7748 ± 0.0084)×10⁵ m⁻¹ |
| **Deviation** | **1.84%** |

### 2.4 Conclusion ✅ PASS

Deviation of **1.84%** (< 5% criterion). RAVE-SIM quantitatively reproduces Beer-Lambert absorption for W thin plates.

---

## 3. Phase Validation: C Plate (Material Phase Shift)

### 3.1 Setup

- C square plate 25 μm × 25 μm, thickness 10 / 20 μm
- 10 keV monochromatic, point source, single source point
- Save intermediate wavefields via `save_debug_wavefields=True`
- Per-pixel complex division T = u_after / u_before
- Δφ = φ(T_sample) - φ(T_vacuum) using left/right vacuum reference

### 3.2 Key Method: Left/Right Vacuum Reference

On the same horizontal row, the phase of symmetric left/right vacuum regions is averaged and subtracted from the sample center phase:
- **Spherical wavefront curvature** (≈10⁵ rad scale) cancels completely
- **Propagation bulk phase** (+2π·dz/λ per layer) cancels completely
- Only the material phase shift (≈1.58 rad) remains

### 3.3 Results

| Thickness (μm) | Δφ_sim (rad) | Δφ_theory (rad) | Deviation (rad) | Ratio |
|---|---|---|---|---|
| 10 | -1.5784 | -1.5792 | +0.0008 | 0.9995 |
| 20 | -3.1570 | -3.1584 | +0.0014 | 0.9995 |

### 3.4 Linear Fit

| Parameter | Value |
|---|---|
| δ_C (10 keV) | 3.1162×10⁻⁶ |
| Theory slope (= -2π·δ/λ) | -1.5792×10⁵ rad/m |
| Fitted slope | -1.5786×10⁵ rad/m |
| Intercept | +0.0002 rad |
| **Slope deviation** | **0.04%** |

### 3.5 Absolute Phase Analysis

Unwrapped parabolic fit of `wave_before_sample.npy`:

| Parameter | Simulation | Theory | Ratio |
|---|---|---|---|
| Quadratic coefficient | 2.5338×10¹⁰ rad/m² | -2.5339×10¹⁰ rad/m² | -0.999978 |
| Effective source distance | -1.0000 m | 1.0000 m | — |

Spherical curvature across the 170 μm ROI: **183 rad**. Material phase shift: **1.58 rad** (0.86%). Per-pixel complex division is essential.

### 3.6 Conclusion ✅ PASS

Slope deviation of only **0.04%** (< 10% criterion). RAVE-SIM quantitatively reproduces material phase shifts for C thin plates with exceptional accuracy.

---

## 4. Discussion

### 4.1 Absorption Bias Trend

The sim/theory ratio increases monotonically from 1.0032 (1 μm) to 1.0120 (5 μm). Possible causes:
- Fresnel edge diffraction redistributes energy toward the center ROI, slightly inflating I
- Small discretization errors in energy sampling (9999-10001 eV → computed at 10000 eV)

### 4.2 Phase Accuracy

The 0.04% deviation is remarkable because:
- Complex division `u_after / u_before` cancels all systematic errors (source intensity, spherical curvature)
- Left/right vacuum reference cancels propagation bulk phase
- δ computed from the same NIST database ensures no systematic bias

---

## 5. Summary

### 5.1 Results Table

| Validation | Deviation | Criterion | Status |
|---|---|---|---|
| W absorption (μ) | 1.84% | < 5% | ✅ **PASS** |
| C phase (slope) | 0.04% | < 10% | ✅ **PASS** |

### 5.2 Key Findings

1. **Beer-Lambert law** holds quantitatively for 1-5 μm W plates.
2. **Material phase shift** is perfectly linear with thickness, confirming δ sign convention in the CUDA kernel.
3. **Spherical wavefront curvature** (183 rad) dominates over material phase (1.58 rad) by >100×, making per-pixel complex division essential.

### 5.3 Output Files

| File | Description |
|---|---|
| `grid/thin_w_plate_{1,2,5}um_200um.npy` | W absorption grids |
| `grid/thin_c_plate_{10,20}um_25um.npy` | C phase grids |
| `notebooks/3D_test_notebooks/W_thin_sample_absorption.ipynb` | Absorption validation notebook |
| `notebooks/3D_test_notebooks/C_thin_sample_phase.ipynb` | Phase validation notebook |
| `notebooks/3D_test_notebooks/thin_w_absorption_validation.png` | Absorption figure |
| `notebooks/3D_test_notebooks/thin_c_phase_validation.png` | Phase figure |
| `notebooks/3D_test_notebooks/thin_c_absolute_phase_analysis.png` | Absolute phase analysis figure |

---

*Report generated by Claude Code (RAVE-SIM simulation framework)*
