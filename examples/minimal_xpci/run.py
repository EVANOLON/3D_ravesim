#!/usr/bin/env python3
"""Run a small, data-free 2D X-ray phase-contrast imaging example."""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np


def install_bfpy_fallback() -> None:
    """Allow the in-memory example to run before the optional Rust module is built."""
    try:
        __import__("bfpy")
        return
    except ImportError:
        pass

    module = types.ModuleType("bfpy")

    class ChunkedEditor:
        def __init__(self, *_args, **_kwargs):
            pass

        def position(self):
            return 0

        def read_chunk_c16(self, buffer):
            return len(buffer)

        def read_chunk_c8(self, buffer):
            return len(buffer)

        def write_chunk_c16(self, _buffer):
            pass

        def write_chunk_c8(self, _buffer):
            pass

        def advance(self, _count):
            pass

        def __len__(self):
            return 0

        def dtype(self):
            return "c16"

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("disk-backed FFT requires the optional bfpy extension")

    module.ChunkedEditor = ChunkedEditor
    module.generate_header_c16 = unavailable
    module.generate_header_c8 = unavailable
    module.fft_c16 = unavailable
    module.fft_c8 = unavailable
    module.ifft_c16 = unavailable
    module.ifft_c8 = unavailable
    sys.modules["bfpy"] = module


def sphere_grid(nx: int, ny: int, nz: int, radius: int) -> np.ndarray:
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    mask = (
        (xx - nx // 2) ** 2
        + (yy - ny // 2) ** 2
        + (zz - nz // 2) ** 2
        <= radius**2
    )
    grid = np.zeros((nz, ny, nx), dtype=np.uint32)
    grid[mask] = 1
    return grid


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "minimal_xpci",
        help="directory for result.npz and preview.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "big-wave"))
    install_bfpy_fallback()

    # config must be imported first because the original big-wave modules have
    # a circular import that relies on this established initialization order.
    import config  # noqa: F401
    from optical_element import Material, Sample, generate_deltabeta_table
    from propagation import (
        SimParams,
        convert_energy_wavelength,
        propagate_2d,
        square_and_downsample_2d,
    )
    from vector import NumpyVector

    started = time.perf_counter()
    nx = ny = 512
    count = nx * ny
    wavelength = convert_energy_wavelength(8_000.0)
    sample_z = 0.01
    detector_z = 0.06
    # A 51.2 um field gives the compact object enough zero-padding for the
    # centimetre-scale Fresnel propagation without periodic wrap-around.
    dx = 1.0e-7
    cutoff_frequency = 0.4 / dx

    params = SimParams(
        N=count,
        dx=dx,
        dy=dx,
        ny=ny,
        z_detector=detector_z,
        # Keep the detector slightly inside the numerical grid so boundary
        # cells have complete area coverage.
        detector_size=(nx - 16) * dx,
        detector_size_x=(nx - 16) * dx,
        detector_size_y=(ny - 16) * dx,
        detector_pixel_size_x=2 * dx,
        detector_pixel_size_y=2 * dx,
        wl=wavelength,
        chunk_size=8192,
        detector_integrator="area_v1",
    )

    sample = Sample(
        z_start=sample_z,
        pixel_size_x=2.0e-7,
        pixel_size_y=2.0e-7,
        pixel_size_z=2.0e-7,
        grid=sphere_grid(24, 24, 24, radius=9),
        materials=[Material("C", 2.0)],
        x_positions=np.array([0.0]),
        y_positions=np.array([0.0]),
    )
    sample.check_valid()
    sample.store_deltabetas(generate_deltabeta_table(sample.materials, 8_000.0))

    # Use a unit plane wave so the example isolates sample propagation and
    # detector integration. Point-source cone-beam geometry is covered by the
    # dedicated Fresnel-scaling tests.
    wave = NumpyVector(np.ones(count, dtype=np.complex64))
    spectrum = NumpyVector(np.zeros(count, dtype=np.complex64))
    sample.apply(wave, spectrum, params, cutoff_frequency, 0, None)
    remaining_distance = detector_z - (sample_z + sample.get_thickness())
    propagate_2d(
        wave,
        spectrum,
        params.dx,
        params.get_dy(),
        params.wl,
        remaining_distance,
        params.chunk_size,
        cutoff_frequency,
        params.nx,
        params.ny,
    )
    detected = square_and_downsample_2d(wave, params, detector_z)

    reference_wave = NumpyVector(np.ones(count, dtype=np.complex64))
    reference_spectrum = NumpyVector(np.zeros(count, dtype=np.complex64))
    propagate_2d(
        reference_wave,
        reference_spectrum,
        params.dx,
        params.get_dy(),
        params.wl,
        detector_z - sample_z,
        params.chunk_size,
        cutoff_frequency,
        params.nx,
        params.ny,
    )
    reference = square_and_downsample_2d(reference_wave, params, detector_z)
    difference = detected - reference
    relative_contrast = np.divide(
        difference,
        reference,
        out=np.zeros_like(difference),
        where=reference > np.finfo(reference.dtype).tiny,
    )

    assert detected.ndim == 2 and detected.shape == reference.shape
    assert np.isfinite(detected).all() and np.isfinite(reference).all()
    assert np.max(np.abs(difference)) > 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "result.npz",
        sample=detected,
        reference=reference,
        difference=difference,
        relative_contrast=relative_contrast,
    )

    try:
        import matplotlib

        # This release example must also render on servers and CI workers that
        # inherit DISPLAY but do not have a reachable X server.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
        for axis, image, title in zip(
            axes,
            (
                detected / np.mean(reference),
                reference / np.mean(reference),
                relative_contrast,
            ),
            ("Carbon sphere / mean reference", "Free space / mean", "Relative contrast"),
        ):
            view = axis.imshow(image, origin="lower", cmap="inferno")
            axis.set_title(title)
            figure.colorbar(view, ax=axis, fraction=0.046)
        figure.savefig(output_dir / "preview.png", dpi=140)
        plt.close(figure)
    except ImportError:
        print("Matplotlib not installed; skipped preview.png")

    elapsed = time.perf_counter() - started
    print(f"detector shape: {detected.shape}")
    print(f"maximum contrast difference: {np.max(np.abs(difference)):.6e}")
    print(f"results: {output_dir}")
    print(f"PASS ({elapsed:.2f} s)")


if __name__ == "__main__":
    main()
