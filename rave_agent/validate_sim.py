#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAVE-SIM configuration validator and backend-aware feasibility gate."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

RAVE_ROOT = Path(__file__).resolve().parent.parent
BIG_WAVE = RAVE_ROOT / "big-wave"
NIST = RAVE_ROOT / "nist_lookup"
for path in (BIG_WAVE, NIST):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

# Config is the safe import entry; importing propagation first exposes a latent
# circular import in older entry points in this repository.
import config as rave_config  # noqa: E402
from propagation import (  # noqa: E402
    convert_energy_wavelength,
    grid_density_check,
    grid_density_check_2d,
)

GIB = 1024 ** 3
MIB = 1024 ** 2
P0_MAX_SCIPY_FFT2_POINTS = 64 * 1024 * 1024
P0_LOCAL_BYTES_PER_POINT = 256
P0_IO_THROUGHPUT_BYTES_PER_SEC = 1000 * 1000 ** 2
NVIDIA_SMI_CANDIDATES = [
    "/usr/lib/wsl/lib/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
]
GRID_PATH_KEYS = {
    "sample": ("grid_path",),
    "precise_sample": ("material_grid_path", "density_grid_path"),
    "plasma_sample": (
        "ne_grid_path",
        "ni_grid_path",
        "te_grid_path",
        "zstar_grid_path",
    ),
}


def is_pow2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": str(detail)}


def load_sim(sim_dir):
    """Load lightweight configuration only; never materialize sample arrays."""
    sim_dir = Path(sim_dir)
    dct = rave_config.load(sim_dir / "config.yaml")
    params = rave_config.parse_sim_params(dct["sim_params"])
    computed = rave_config.load(sim_dir / "computed.yaml")
    return dct, params, computed, list(dct.get("elements", []))


def _array_info(path):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    info = {
        "path": str(path),
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
    }
    del array
    return info


def grid_inventory(sim_dir, elements):
    """Read only NPY headers and return a de-duplicated input-grid inventory."""
    seen = set()
    items = []
    errors = []
    for element in elements:
        for key in GRID_PATH_KEYS.get(str(element.get("type", "")), ()):
            if key not in element:
                errors.append(f"{element.get('type', 'element')}.{key} is missing")
                continue
            path = (Path(sim_dir) / str(element[key])).resolve()
            if path in seen:
                continue
            seen.add(path)
            try:
                items.append(_array_info(path))
            except Exception as exc:
                errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return items, errors


def _grid_depth(sim_dir, element, key):
    info = _array_info(Path(sim_dir) / str(element[key]))
    if not info["shape"]:
        raise ValueError(f"grid {element[key]} must have at least one dimension")
    return int(info["shape"][0])


def element_thickness(sim_dir, element):
    kind = str(element.get("type", ""))
    if kind in ("grating", "env_grating"):
        return float(element.get("thickness", 0.0)) + float(
            element.get("substrate_thickness", 0.0)
        )
    if kind == "sample":
        return float(element["pixel_size_z"]) * _grid_depth(sim_dir, element, "grid_path")
    if kind == "precise_sample":
        return float(element["pixel_size_z"]) * _grid_depth(
            sim_dir, element, "material_grid_path"
        )
    if kind == "plasma_sample":
        return float(element["pixel_size_z"]) * _grid_depth(sim_dir, element, "ne_grid_path")
    if kind == "save_and_exit":
        return 0.0
    raise ValueError(f"unknown optical element type: {kind!r}")


def validate_z_layout(sim_dir, params, z_source, elements, cutoff_angles):
    if len(cutoff_angles) != len(elements) + 1:
        raise ValueError(
            f"expected {len(elements) + 1} cutoff angles, got {len(cutoff_angles)}"
        )
    tolerance = 1e-8
    current_z = float(z_source)
    phase_steps = []
    for index, element in enumerate(elements):
        z_start = float(element["z_start"])
        if current_z > z_start + tolerance:
            raise ValueError(f"element {index} overlaps the preceding element")
        steps = len(element.get("x_positions", [0.0]))
        if steps > 1:
            phase_steps.append(steps)
        kind = str(element.get("type", ""))
        if kind == "save_and_exit":
            expected_z = (
                params.z_detector
                if index == len(elements) - 1
                else float(elements[index + 1]["z_start"])
            )
            if z_start != expected_z:
                raise ValueError("save_and_exit must share z with detector/next element")
        current_z = z_start + element_thickness(sim_dir, element)
    if current_z > params.z_detector + tolerance:
        raise ValueError("detector must be after the last optical element")
    if len(set(phase_steps)) > 1:
        raise ValueError(f"inconsistent phase-step counts: {phase_steps}")
    if any(a > b for a, b in zip(cutoff_angles, cutoff_angles[1:])):
        raise ValueError("cutoff angles must be increasing")
    if any(not 0 <= angle <= math.pi / 2 for angle in cutoff_angles):
        raise ValueError("cutoff angles must be between 0 and pi/2")
    return phase_steps


def _resolved_runtime(dct, params, computed):
    computed_runtime = computed.get("runtime", {}).get("big_wave", {})
    resolved = computed_runtime.get("resolved", computed_runtime)
    runtime = rave_config.get_runtime_big_wave(dct)
    sim = dct.get("sim_params", {})
    save_debug_wavefields = resolved.get(
        "save_debug_wavefields",
        runtime.get(
            "save_debug_wavefields",
            sim.get("save_debug_wavefields", False),
        ),
    )
    if not isinstance(save_debug_wavefields, bool):
        raise ValueError("save_debug_wavefields must be a boolean")
    return {
        "memory_budget_gb": float(
            resolved.get(
                "memory_budget_gb",
                runtime.get(
                    "memory_budget_gb",
                    sim.get("memory_budget_gb", rave_config.DEFAULT_MEMORY_BUDGET_GB),
                ),
            )
        ),
        "chunk_size": int(resolved.get("chunk_size", params.chunk_size)),
        "fft2_backend": str(resolved.get("fft2_backend", params.fft2_backend)),
        "detector_integrator": str(
            resolved.get("detector_integrator", params.detector_integrator)
        ),
        "save_debug_wavefields": save_debug_wavefields,
    }


def validate(sim_dir):
    checks = []
    meta = {"sim_dir": str(sim_dir)}
    try:
        dct, params, computed, elements = load_sim(sim_dir)
    except Exception as exc:
        checks.append(check("config_parse", False, f"{type(exc).__name__}: {exc}"))
        return {"ok": False, "checks": checks, **meta}

    checks.append(check("config_parse", True, "YAML + sim_params parsed"))
    is_2d = bool(params.is_2d)
    points = int(params.N)
    nx = int(params.nx if is_2d else points)
    ny = int(params.ny if is_2d else 1)
    dx = float(params.dx)
    dy = float(params.get_dy())

    if is_2d:
        checks.append(check("shape", nx * ny == points, f"N={points}, nx*ny={nx * ny}"))
        checks.append(
            check("power_of_two", is_pow2(nx) and is_pow2(ny), f"nx={nx}, ny={ny}")
        )
    else:
        checks.append(check("power_of_two", is_pow2(points), f"N={points}"))

    if bool(getattr(params, "use_fresnel_scaling", False)):
        # Fresnel scaling maps the physical detector to an effective sample-plane
        # detector (detector_size / M); the wavefront FOV need not cover the
        # physical detector, matching multisim.setup_simulation.
        fov_ok = True
        fov_detail = "Fresnel scaling: FOV check skipped (effective detector = detector_size/M)"
    elif is_2d:
        fov_ok = nx * dx >= params.get_detector_size_x() and ny * dy >= params.get_detector_size_y()
        fov_detail = (
            f"nx*dx={nx * dx:.3e} vs det_x={params.get_detector_size_x():.3e}, "
            f"ny*dy={ny * dy:.3e} vs det_y={params.get_detector_size_y():.3e}"
        )
    else:
        fov_ok = points * dx >= params.detector_size
        fov_detail = f"N*dx={points * dx:.3e} vs detector_size={params.detector_size:.3e}"
    checks.append(check("fov", fov_ok, fov_detail))

    inventory, grid_errors = grid_inventory(sim_dir, elements)
    checks.append(
        check(
            "input_grids",
            not grid_errors,
            "headers readable" if not grid_errors else "; ".join(grid_errors),
        )
    )

    multisource = dct.get("multisource", {})
    z_source = float(multisource.get("z", 0.0))
    energy_range = tuple(map(float, computed.get("energy_range", [0.0, 0.0])))
    cutoff_angles = list(map(float, computed.get("cutoff_angles", [])))
    cutoff_angles_y = list(map(float, computed.get("cutoff_angles_y", [])))
    try:
        phase_steps = validate_z_layout(sim_dir, params, z_source, elements, cutoff_angles)
        checks.append(check("z_layout", True, "z order, overlap and phase steps valid"))
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        phase_steps = []
        checks.append(check("z_layout", False, str(exc)))

    try:
        runtime = _resolved_runtime(dct, params, computed)
        checks.append(check("runtime_config", True, "big-wave runtime values resolved"))
    except (TypeError, ValueError) as exc:
        checks.append(check("runtime_config", False, str(exc)))
        runtime = {
            "memory_budget_gb": rave_config.DEFAULT_MEMORY_BUDGET_GB,
            "chunk_size": int(params.chunk_size),
            "fft2_backend": params.fft2_backend,
            "detector_integrator": params.detector_integrator,
            "save_debug_wavefields": False,
        }
    chunk_size = runtime["chunk_size"]
    chunk_ok = 0 < chunk_size <= points
    if is_2d:
        chunk_ok = chunk_ok and chunk_size % nx == 0
    checks.append(
        check("chunk_size", chunk_ok, f"chunk_size={chunk_size}, points={points}, nx={nx}")
    )
    detector_integrator = str(runtime["detector_integrator"])
    detector_integrator_ok = (
        not is_2d or detector_integrator in rave_config.DETECTOR_INTEGRATORS
    )
    checks.append(
        check(
            "detector_integrator",
            detector_integrator_ok,
            (
                f"detector_integrator={detector_integrator}"
                if detector_integrator_ok
                else f"unsupported detector_integrator={detector_integrator!r}"
            ),
        )
    )

    # ``legacy_fastwave`` reproduces the historical pre-area-weighted index map.
    # A non-integer detector-pixel/grid ratio loses fractional overlap and can
    # produce non-uniform counts. ``area_v1`` exactly integrates overlap for the
    # piecewise-constant grid discretization; detector/domain coverage remains a
    # separate concern for both paths.
    if is_2d:
        sampling = computed.get("fresnel_sampling")
        ratio = None
        ratio_source = "raw physical detector pixel / grid spacing"
        if isinstance(sampling, dict):
            try:
                ratio = [float(v) for v in sampling["detector_pixels_per_grid_point"]]
                ratio_source = "fresnel_sampling.detector_pixels_per_grid_point"
            except (KeyError, TypeError, ValueError):
                ratio = None
        if ratio is None:
            try:
                ratio = [
                    float(params.detector_pixel_size_x) / dx,
                    float(params.detector_pixel_size_y) / dy,
                ]
            except (TypeError, ValueError, ZeroDivisionError):
                ratio = None
        if ratio is None:
            checks.append(
                check(
                    "detector_integrator_pixel_ratio",
                    True,
                    "detector pixel/grid ratio unavailable; not checked",
                )
            )
        else:
            tol = 1e-9
            non_integer = any(
                abs(value - round(value)) > tol * max(1.0, abs(value)) for value in ratio
            )
            ratio_text = f"x={ratio[0]:.6g}, y={ratio[1]:.6g}"
            if not non_integer:
                checks.append(
                    check(
                        "detector_integrator_pixel_ratio",
                        True,
                        f"{ratio_text} integer; fractional-ratio check passed for "
                        f"{detector_integrator} ({ratio_source}); detector/domain "
                        "coverage is checked separately",
                    )
                )
            elif detector_integrator == "legacy_fastwave":
                checks.append(
                    check(
                        "detector_integrator_pixel_ratio",
                        False,
                        "legacy_fastwave with a non-integer detector pixel/grid ratio "
                        f"({ratio_text}; {ratio_source}) loses fractional overlap and can "
                        "produce non-uniform counts. Use detector_integrator: area_v1 "
                        "(the default) for this geometry.",
                    )
                )
            else:
                checks.append(
                    check(
                        "detector_integrator_pixel_ratio",
                        True,
                        f"{ratio_text} non-integer; {detector_integrator} integrates the "
                        "exact overlap for the piecewise-constant grid discretization "
                        f"({ratio_source})",
                    )
                )

    try:
        if energy_range[1] > 0:
            wavelength = convert_energy_wavelength(energy_range[1])
            first_z = float(elements[0]["z_start"]) if elements else params.z_detector
            dz = max(first_z - z_source, 1e-12)
            x_range = multisource.get("x_range", [0.0, 0.0])
            x_source = max(abs(float(x_range[0])), abs(float(x_range[1])))
            use_fresnel = bool(getattr(params, "use_fresnel_scaling", False))
            if is_2d:
                y_range = multisource.get("y_range", [0.0, 0.0])
                y_source = max(abs(float(y_range[0])), abs(float(y_range[1])))
                # Fresnel scaling initialises a plane wave (no spherical
                # source->sample propagation), so the source->sample grid
                # density check does not apply.
                if not use_fresnel:
                    grid_density_check_2d(
                        dz, x_source, nx, dx, y_source, ny, dy, wavelength
                    )
                    for angle in cutoff_angles:
                        max_frequency = math.sqrt(2.0) * math.sin(angle) / wavelength
                        if max_frequency > 0.5 / dx:
                            raise ValueError("2D circular cutoff exceeds x-axis Nyquist")
                    # rectangular detector: the y cutoff angle is derived from the
                    # y detector size, not the x size.
                    for angle in (cutoff_angles_y or cutoff_angles):
                        max_frequency = math.sqrt(2.0) * math.sin(angle) / wavelength
                        if max_frequency > 0.5 / dy:
                            raise ValueError("2D circular cutoff exceeds y-axis Nyquist")
                else:
                    # Fresnel-similarity / cone-beam modes propagate the
                    # *magnified* field: the aperture phase rides as a carrier
                    # and the detector is downsampled with
                    # detector_pixel_size / M.  The raw-grid Nyquist comparison
                    # of the physical aperture angle is therefore not the
                    # applicable criterion (it rejects geometrically
                    # well-resolved layouts); the effective-frame criteria are
                    # validated here instead, using the values recorded by the
                    # generator in computed.yaml["fresnel_sampling"].
                    fs = computed.get("fresnel_sampling")
                    if not isinstance(fs, dict):
                        raise ValueError(
                            "Fresnel-scaled 2D run has no computed.yaml fresnel_sampling "
                            "block; regenerate the simulation directory with the "
                            "current generator so the effective-frame sampling "
                            "criteria can be checked"
                        )
                    fov = [float(v) for v in fs.get("wavefront_fov", [0.0, 0.0])]
                    det = [float(v) for v in fs.get("detector_size_effective", [0.0, 0.0])]
                    px = [
                        float(v)
                        for v in fs.get("detector_pixel_effective", [dx, dy])
                    ]
                    if fov[0] < det[0] or fov[1] < det[1]:
                        raise ValueError(
                            "effective detector does not fit the wavefront FOV "
                            f"(detector/M={det} m vs FOV={fov} m)"
                        )
                    if px[0] < dx or px[1] < dy:
                        raise ValueError(
                            "effective detector pixel is smaller than the wavefront grid "
                            f"spacing (pixel/M={px} m vs dx,dy={dx},{dy} m)"
                        )
                    meta["fresnel_sampling"] = fs
            else:
                if not use_fresnel:
                    grid_density_check(dz, x_source, points, dx, wavelength)
            checks.append(check("nyquist", True, f"sampling valid at E={energy_range[1]:.0f} eV"))
        else:
            checks.append(check("nyquist", True, "no energy info; skipped"))
    except (AssertionError, ValueError) as exc:
        checks.append(check("nyquist", False, str(exc)))

    out_x = max(1, int(params.get_detector_size_x() // params.detector_pixel_size_x))
    out_y = (
        max(1, int(params.get_detector_size_y() // params.detector_pixel_size_y))
        if is_2d
        else 1
    )
    meta.update(
        {
            "is_2d": is_2d,
            "N": points,
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "detector_size": params.detector_size,
            "detector_pixels": out_x * out_y,
            "detector_output_shape": [out_y, out_x],
            "energy_range": list(energy_range),
            "z_source": z_source,
            "nr_elements": len(elements),
            "elements": [str(element.get("type", "unknown")) for element in elements],
            "element_phase_steps": phase_steps,
            "nr_source_points": int(multisource.get("nr_source_points", 1)),
            "dtype": str(dct.get("dtype", "c16")).lower(),
            "use_disk_vector": bool(dct.get("use_disk_vector", False)),
            "grid_bytes": sum(item["bytes"] for item in inventory),
            "grid_inventory": inventory,
            **runtime,
            "engine_hint": "big-wave" if bool(dct.get("use_disk_vector", False)) else "either",
        }
    )
    return {"ok": all(item["ok"] for item in checks), "checks": checks, **meta}


def nvidia_smi(query):
    for smi in NVIDIA_SMI_CANDIDATES:
        if not os.path.exists(smi):
            continue
        try:
            result = subprocess.run(
                [smi, "--query-gpu=" + query, "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    return None


def gpu_memory():
    raw = nvidia_smi("memory.free,memory.total")
    if not raw:
        return None, None
    try:
        parts = [part.strip() for part in raw.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def host_memory():
    try:
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                parts = value.strip().split()
                if parts:
                    info[key.strip()] = int(parts[0]) * 1024
        return (
            info.get("MemTotal"),
            info.get("MemAvailable"),
            info.get("SwapTotal"),
            info.get("SwapFree"),
        )
    except Exception:
        return None, None, None, None


def classify_returncode(returncode, stderr=""):
    """Classify process failures using both status and diagnostic text."""
    diagnostic = str(stderr).lower()
    if "memoryerror" in diagnostic or "cannot allocate memory" in diagnostic:
        return "python_memory_error"
    if "no space left on device" in diagnostic or "enospc" in diagnostic:
        return "disk_full"
    if returncode is None:
        return "not_finished"
    if returncode == 0:
        return "ok"
    if returncode in (137, -9) or "killed" in diagnostic:
        return "oom_or_sigkill"
    if returncode in (143, -15):
        return "sigterm"
    if returncode == 1:
        return "generic_error"
    return f"exit_{returncode}"


def _estimated_fft2_calls(sim_dir, elements):
    calls = 2 + 2 * (len(elements) + 1)
    for element in elements:
        kind = str(element.get("type", ""))
        if kind in ("grating", "env_grating"):
            calls += 2 * max(1, int(element.get("nr_steps", 1)))
        elif kind == "sample":
            calls += 2 * _grid_depth(sim_dir, element, "grid_path")
        elif kind == "precise_sample":
            calls += 2 * _grid_depth(sim_dir, element, "material_grid_path")
        elif kind == "plasma_sample":
            calls += 2 * _grid_depth(sim_dir, element, "ne_grid_path")
    return calls


def feasibility(sim_dir, engine="fast-wave"):
    validation = validate(sim_dir)
    result = dict(validation)
    if not validation["ok"]:
        result["feasible"] = False
        result["blocked_by"] = [item["name"] for item in validation["checks"] if not item["ok"]]
        return result

    is_2d = validation["is_2d"]
    points = validation["nx"] * validation["ny"] if is_2d else validation["N"]
    nx = int(validation.get("nx", points))
    itemsize = 8 if validation.get("dtype", "c16") == "c8" else 16
    wave_bytes = points * itemsize
    use_disk = bool(validation.get("use_disk_vector", False))
    grid_bytes = int(validation.get("grid_bytes", 0))
    detector_pixels = int(validation.get("detector_pixels", points))
    blocked_by = []

    if engine == "fast-wave":
        vram_est_bytes = wave_bytes * 3
        free_mib, total_mib = gpu_memory()
        free_gb = None if free_mib is None else free_mib / 1024.0
        gpu_ok = free_gb is not None and vram_est_bytes / GIB <= free_gb * 0.85
        result["gpu"] = {
            "est_vram_gb": round(vram_est_bytes / GIB, 3),
            "free_gb": None if free_gb is None else round(free_gb, 2),
            "total_gb": None if total_mib is None else round(total_mib / 1024.0, 2),
            "ok": gpu_ok,
            "detail": "15% VRAM headroom required" if free_gb is not None else "nvidia-smi unavailable",
        }
        if not gpu_ok:
            blocked_by.append("gpu_vram")
    else:
        result["gpu"] = {"ok": None, "note": "big-wave does not use the GPU gate"}

    disk_components = {
        "input_grids": grid_bytes,
        "input_wave": wave_bytes if use_disk else 0,
        "fourier_wave": wave_bytes if use_disk else 0,
        "fft_scratch": 2 * wave_bytes if use_disk else 0,
        "transactional_output": wave_bytes if use_disk else 0,
        "detector_output": detector_pixels * 8,
        "history_checkpoint_debug": 0,
    }
    disk_required = sum(disk_components.values())
    try:
        disk_free = shutil.disk_usage(sim_dir).free
        disk_ok = disk_free >= disk_required * 1.2
    except Exception:
        disk_free = None
        disk_ok = False
    result["disk"] = {
        "components_gb": {key: round(value / GIB, 3) for key, value in disk_components.items()},
        "est_disk_gb": round(disk_required / GIB, 3),
        "required_with_20pct_headroom_gb": round(disk_required * 1.2 / GIB, 3),
        "free_disk_gb": None if disk_free is None else round(disk_free / GIB, 2),
        "ok": disk_ok,
        "unestimated_optional": ["history", "checkpoint", "debug snapshots"],
    }
    if not disk_ok:
        blocked_by.append("disk")

    if engine == "big-wave":
        backend = str(validation.get("fft2_backend", "scipy_in_memory"))
        if backend == "bfpy_ooc":
            backend_ok = use_disk
            backend_detail = (
                "transactional out-of-core row FFT / transpose / column FFT backend"
                if backend_ok
                else "bfpy_ooc requires use_disk_vector=true"
            )
            if not backend_ok:
                blocked_by.append("fft2_backend_requires_disk_vector")
        elif backend == "scipy_in_memory":
            backend_ok = not use_disk
            backend_detail = (
                "SciPy in-memory FFT2 for NumpyVector"
                if backend_ok
                else "DiskVector requires bfpy_ooc; whole-field SciPy loading was removed"
            )
            if not backend_ok:
                blocked_by.append("fft2_backend_vector_mismatch")
        else:
            backend_ok = False
            backend_detail = f"backend {backend!r} is not implemented by the current engine"
            blocked_by.append("fft2_backend_unavailable")
        if backend_ok and backend == "scipy_in_memory" and is_2d and points > P0_MAX_SCIPY_FFT2_POINTS:
            backend_ok = False
            backend_detail = (
                "scipy_in_memory FFT2 materializes the full field; "
                f"use bfpy_ooc above {P0_MAX_SCIPY_FFT2_POINTS} points"
            )
            blocked_by.append("fft2_backend_not_out_of_core")
        result["backend"] = {
            "name": backend,
            "ok": backend_ok,
            "detail": backend_detail,
            "max_2d_points": (
                None if backend == "bfpy_ooc" else P0_MAX_SCIPY_FFT2_POINTS
            ),
        }

        overhead = 512 * MIB
        chunk = min(points, int(validation.get("chunk_size", points)))
        budget_bytes = float(validation["memory_budget_gb"]) * GIB
        if backend == "bfpy_ooc":
            fft_stage_name = "bfpy_ooc_fft2"
            fft_stage_bytes = overhead + int(budget_bytes * 0.5)
        else:
            fft_stage_name = "scipy_fft2"
            fft_stage_bytes = grid_bytes + overhead + 5 * wave_bytes
        detector_output_bytes = detector_pixels * 8
        if detector_output_bytes > budget_bytes:
            # P3 spills the output to a memmap and drops completed pages.
            detector_resident_bytes = min(
                detector_output_bytes,
                max(64 * MIB, min(int(budget_bytes * 0.25), 512 * MIB)),
            )
            detector_storage = "memmap"
        else:
            detector_resident_bytes = detector_output_bytes
            detector_storage = "memory"
        detector_stage_name = (
            "legacy_detector_streaming"
            if validation.get(
                "detector_integrator", rave_config.DEFAULT_DETECTOR_INTEGRATOR
            )
            == "legacy_fastwave"
            else "area_detector_streaming"
        )
        detector_shape = validation.get("detector_output_shape", [1, detector_pixels])
        detector_axis_work_bytes = 4 * max(map(int, detector_shape)) * 8
        stage_bytes = {
            "point_source": overhead + chunk * (96 + itemsize),
            "sample_or_plasma": grid_bytes + overhead + chunk * (P0_LOCAL_BYTES_PER_POINT + 4 * itemsize),
            fft_stage_name: fft_stage_bytes,
            # P3 keeps only a row tile, 1D prefix/work arrays and the detector
            # output. Oversized detector outputs are backed by a memmap.
            detector_stage_name: (
                overhead
                + chunk * (itemsize + 8)
                + detector_resident_bytes
                + (nx + 1) * 8
                + detector_axis_work_bytes
            ),
            # History retains detector frames, while a NumpyVector snapshot copies
            # a full field.  The number of frames is runtime-controlled and is
            # therefore reported separately as an unbounded multiplier.
            "history_one_frame": overhead + detector_pixels * 8,
            "snapshot": overhead + (0 if use_disk else wave_bytes),
        }
        ram_est_bytes = max(stage_bytes.values())
        mem_total, mem_available, swap_total, swap_free = host_memory()
        host_limit = None if mem_available is None else mem_available * 0.8
        allowed_bytes = budget_bytes if host_limit is None else min(budget_bytes, host_limit)
        ram_ok = ram_est_bytes <= allowed_bytes
        result["ram"] = {
            "stages_gb": {key: round(value / GIB, 3) for key, value in stage_bytes.items()},
            "est_ram_gb": round(ram_est_bytes / GIB, 3),
            "memory_budget_gb": round(budget_bytes / GIB, 3),
            "allowed_gb": round(allowed_bytes / GIB, 3),
            "mem_total_gb": None if mem_total is None else round(mem_total / GIB, 2),
            "mem_avail_gb": None if mem_available is None else round(mem_available / GIB, 2),
            "swap_total_gb": None if swap_total is None else round(swap_total / GIB, 2),
            "swap_free_gb": None if swap_free is None else round(swap_free / GIB, 2),
            "ok": ram_ok,
            "detail": "must fit configured budget and 80% of available physical RAM",
            "detector_output_storage": detector_storage,
            "unestimated_multipliers": [
                "history frame count (depends on history_dz)",
                "simultaneously retained debug snapshots",
            ],
        }
        if not ram_ok:
            blocked_by.append("ram")
    else:
        result["backend"] = {"name": "fast-wave", "ok": True}
        result["ram"] = {"ok": None, "note": "RAM gate applies to big-wave"}

    try:
        _, _, _, elements = load_sim(sim_dir)
        fft2_calls = _estimated_fft2_calls(sim_dir, elements) if is_2d else 0
    except Exception:
        fft2_calls = 0
    io_bytes = fft2_calls * 2 * wave_bytes if use_disk else 0
    seconds_per_fft = 2 * wave_bytes / P0_IO_THROUGHPUT_BYTES_PER_SEC if use_disk else 0.0
    result["io"] = {
        "estimated_fft2_calls": fft2_calls,
        "bytes_per_fft_read_write": 2 * wave_bytes if use_disk else 0,
        "io_seconds_per_fft": round(seconds_per_fft, 3),
        "estimated_read_write_gb": round(io_bytes / GIB, 3),
        "assumed_throughput_mb_s": 1000,
        "lower_bound_minutes": round(io_bytes / P0_IO_THROUGHPUT_BYTES_PER_SEC / 60, 2),
        "note": "I/O lower bound only; CPU FFT and optical-operator time are not calibrated",
    }
    result["est_time_min"] = result["io"]["lower_bound_minutes"]
    result["estimate_kind"] = "io_lower_bound_not_runtime_prediction"
    result["exit_code_classification"] = {
        "137/-9": "oom_or_sigkill",
        "MemoryError": "python_memory_error",
        "ENOSPC": "disk_full",
    }
    result["blocked_by"] = list(dict.fromkeys(blocked_by))
    result["feasible"] = not result["blocked_by"]
    return result


def result_exit_code(result, feasibility_mode=False):
    passed = bool(result.get("ok", False))
    if feasibility_mode:
        passed = passed and bool(result.get("feasible", False))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", metavar="SIM_DIR")
    parser.add_argument("--feasibility", metavar="SIM_DIR")
    parser.add_argument("--engine", default="fast-wave", choices=["fast-wave", "big-wave"])
    args = parser.parse_args()
    if args.validate:
        output = validate(args.validate)
    elif args.feasibility:
        output = feasibility(args.feasibility, args.engine)
    else:
        parser.error("one of --validate / --feasibility is required")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    sys.exit(result_exit_code(output, feasibility_mode=bool(args.feasibility)))


if __name__ == "__main__":
    main()
