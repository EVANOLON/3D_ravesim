  # Copyright (c) 2024, ETH Zurich

import logging
from dataclasses import dataclass, field
import math
import mmap
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Tuple
import warnings
import weakref
import numpy as np

from util import detector_x_vector
from vector import Vector

logger = logging.getLogger("big-wave")

DETECTOR_INTEGRATOR_VERSIONS = {
    "legacy_fastwave": "legacy_fastwave_stream_v1",
    "area_v1": "area_separable_stream_v1",
}


@dataclass
class SimParams:
    """Global parameters for the simulation"""

    N: int
    dx: float
    z_detector: float
    detector_size: float
    detector_pixel_size_x: float
    detector_pixel_size_y: float
    wl: float
    chunk_size: int

    # 2D simulation parameters (ny=1 and dy=0 => 1D mode, fully backward compatible)
    ny: int = 1
    dy: float = 0.0
    detector_size_x: float = 0.0
    detector_size_y: float = 0.0
    # Optional explicit nx.  When zero, nx is derived from N // ny.
    nx: int = 0
    # Runtime / provenance fields (not physics); kept on SimParams for plumbing.
    memory_budget_gb: float = 0.0
    fft2_backend: str = "scipy_in_memory"
    detector_integrator: str = "legacy_fastwave"
    use_fresnel_scaling: bool = False
    fresnel_magnification: float = field(default=1.0, init=False)
    fresnel_effective_z: float = field(default=0.0, init=False)
    # Runtime-only P3 detector controls. They are deliberately not physics
    # configuration and may be set by the orchestration layer after parsing.
    detector_output_dir: str | None = field(default=None, repr=False)
    detector_progress_cb: Callable[[str, int, int], Any] | None = field(
        default=None, repr=False
    )
    detector_cancel_token: Any | None = field(default=None, repr=False)
    detector_last_metadata: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}")
        if self.ny <= 0:
            raise ValueError(f"ny must be positive, got {self.ny}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.ny > 1:
            if self.nx == 0:
                self.nx = self.N // self.ny
            if self.nx * self.ny != self.N:
                raise ValueError(
                    f"N ({self.N}) must equal nx * ny ({self.nx} * {self.ny} = {self.nx * self.ny})"
                )
            logger.info(
                f"2D mode: nx={self.nx}, ny={self.ny}, "
                f"dx={self.dx:.3e}, dy={self.get_dy():.3e}, "
                f"detector=({self.get_detector_size_x():.3e} x {self.get_detector_size_y():.3e}) m"
            )
        else:
            self.nx = self.N
            logger.debug(
                f"1D mode: N={self.N}, dx={self.dx:.3e}, detector={self.detector_size:.3e} m"
            )
        if self.detector_integrator not in DETECTOR_INTEGRATOR_VERSIONS:
            raise ValueError(
                f"unsupported detector_integrator={self.detector_integrator!r}; "
                f"expected one of {sorted(DETECTOR_INTEGRATOR_VERSIONS)!r}"
            )

    @property
    def is_2d(self) -> bool:
        return self.ny > 1

    def get_dy(self) -> float:
        return self.dy if self.dy != 0.0 else self.dx

    def get_detector_size_x(self) -> float:
        return self.detector_size_x if self.detector_size_x != 0.0 else self.detector_size

    def get_detector_size_y(self) -> float:
        return self.detector_size_y if self.detector_size_y != 0.0 else self.get_dy() * self.ny

    def configure_fresnel_detector(self, z_source: float, z_sample: float) -> None:
        """Resolve fast-wave-compatible effective detector geometry."""
        if not self.use_fresnel_scaling:
            self.fresnel_magnification = 1.0
            self.fresnel_effective_z = 0.0
            return
        z_source_to_sample = z_sample - z_source
        z_sample_to_detector = self.z_detector - z_sample
        if z_source_to_sample <= 0 or z_sample_to_detector <= 0:
            raise ValueError(
                "Fresnel scaling requires z_source < z_sample < z_detector"
            )
        self.fresnel_effective_z = (
            z_source_to_sample
            * z_sample_to_detector
            / (z_source_to_sample + z_sample_to_detector)
        )
        self.fresnel_magnification = (
            z_source_to_sample + z_sample_to_detector
        ) / z_source_to_sample

    def effective_detector_geometry(
        self, current_z: float
    ) -> tuple[float, float, float, float]:
        """Return effective (pixel_x, pixel_y, z, magnification)."""
        at_final_detector = math.isclose(
            current_z,
            self.z_detector,
            rel_tol=1e-12,
            abs_tol=max(1e-15, abs(self.z_detector) * 1e-12),
        )
        if self.use_fresnel_scaling and at_final_detector:
            if self.fresnel_effective_z <= 0 or self.fresnel_magnification <= 0:
                raise ValueError("Fresnel detector geometry has not been configured")
            return (
                self.detector_pixel_size_x / self.fresnel_magnification,
                self.detector_pixel_size_y / self.fresnel_magnification,
                self.fresnel_effective_z,
                self.fresnel_magnification,
            )
        return (
            self.detector_pixel_size_x,
            self.detector_pixel_size_y,
            current_z,
            1.0,
        )


h = 6.62607004 * 10 ** (-34)  # planck constant in mˆ2 kg / s
c_0 = 299792458  # speed of light in m / s
eV_to_joule = 1.602176634 * 10 ** (-19)


def grid_density_check(dz: float, x_source: float, N: int, dx: float, wl: float):
    """
    Check if dx is sufficiently small so that the waves after the initial analytical
    propagation don't violate the Nyquist condition.
    """

    def dist_x(idx: int) -> float:
        return np.sqrt(((N / 2 - idx) * dx + np.abs(x_source)) ** 2 + dz**2)

    nyquist_satisfied = np.abs(dist_x(0) - dist_x(1)) <= wl / 2
    logger.debug(
        f"grid_density_check: dz={dz:.3e}, x_source={x_source:.3e}, "
        f"N={N}, dx={dx:.3e}, wl={wl:.3e} -> {'OK' if nyquist_satisfied else 'FAIL'}"
    )

    if not nyquist_satisfied:
        raise ValueError(
            "Nyquist not satisfied, dx too large. In `propagation.py` there is a util function to compute the maximal possible dx."
        )


def grid_density_check_2d(dz: float, x_source: float, Nx: int, dx: float,
                          y_source: float, Ny: int, dy: float, wl: float):
    """
    Perform Nyquist grid density check for both x and y directions (2D mode).
    """
    logger.info(
        f"2D Nyquist check: dz={dz:.3e}, "
        f"x: Nx={Nx}, dx={dx:.3e}, x_source={x_source:.3e} | "
        f"y: Ny={Ny}, dy={dy:.3e}, y_source={y_source:.3e}"
    )
    grid_density_check(dz, x_source, Nx, dx, wl)
    grid_density_check(dz, y_source, Ny, dy, wl)


def max_dx(dz: float, x_source: float, N: int, wl: float) -> float:
    """
    Calculate the maximal possible dx such that the waves after the initial analytical
    propagation don't violate the Nyquist condition.

    We use binary search here because inverting this analytically is a pain.
    """
    min: float = 0
    max: float = 0.1

    while (max - min) / max > 0.01:
        dx = (max + min) / 2
        try:
            grid_density_check(dz, x_source, N, dx, wl)
            min = dx
        except:
            max = dx

    return min


def convert_energy_wavelength(energy_or_wavelength: float) -> float:
    """Given an energy in eV, calculate the wavelength in metres or vice versa"""
    return h * c_0 / (energy_or_wavelength * eV_to_joule)


def require_row_aligned_chunk_size(chunk_size: int, nx: int, ny: int) -> int:
    """Validate the P1 2D tile contract and return the number of tile rows."""
    if nx <= 0 or ny <= 0:
        raise ValueError(f"nx and ny must be positive, got nx={nx}, ny={ny}")
    if chunk_size <= 0 or chunk_size % nx != 0:
        raise ValueError(
            f"2D chunk_size must contain complete rows: chunk_size={chunk_size}, nx={nx}"
        )
    if chunk_size > nx * ny:
        raise ValueError(
            f"2D chunk_size must not exceed field size: {chunk_size} > {nx * ny}"
        )
    return chunk_size // nx


def _tile_view(idx: int, chunk: np.ndarray, nx: int, ny: int) -> tuple[np.ndarray, int]:
    """Return a (tile_rows, nx) view and its first y row without index grids."""
    if idx % nx != 0 or len(chunk) % nx != 0:
        raise ValueError(
            f"2D vector callback is not row aligned: idx={idx}, len={len(chunk)}, nx={nx}"
        )
    y_start = idx // nx
    tile_rows = len(chunk) // nx
    if y_start + tile_rows > ny:
        raise ValueError("2D vector callback extends beyond ny")
    return chunk.reshape(tile_rows, nx), y_start


def _frequency_squared(n: int, spacing: float) -> np.ndarray:
    """Squared FFT frequencies with one O(n) allocation."""
    return np.square(np.fft.fftfreq(n) / spacing)


def apply_frequency_cutoff_2d(
    vec: Vector, freq: float, dx: float, dy: float,
    nx: int, ny: int, chunk_size: int
) -> None:
    """
    2D circular frequency cutoff. Zero out frequency components outside a circular
    region of radius `freq` in the 2D frequency plane.

    This is the 2D equivalent of apply_frequency_cutoff for 1D (which uses a
    symmetric interval). For 2D we use a circular mask, matching fast-wave.
    """
    # Use Euclidean norm of (freq_x, freq_y) as the cutoff radius, matching
    # fast-wave's propagate_convolve_step_2d_kernel which checks
    #   kx² + ky² ≤ cutoff_freq_x² + cutoff_freq_y².
    # For a square detector freq_x == freq_y == freq, giving threshold 2·freq².
    freq_sq = 2.0 * freq ** 2

    require_row_aligned_chunk_size(chunk_size, nx, ny)
    if len(vec) != nx * ny:
        raise ValueError(f"2D vector length {len(vec)} does not match nx*ny={nx * ny}")
    kx2 = _frequency_squared(nx, dx)
    ky2 = _frequency_squared(ny, dy)

    def modifier(idx: int, chunk: np.ndarray) -> None:
        tile, y_start = _tile_view(idx, chunk, nx, ny)
        mask = ky2[y_start:y_start + tile.shape[0], None] + kx2[None, :] > freq_sq
        tile[mask] = 0.0

    vec.modify_chunked(chunk_size, modifier)


def apply_frequency_cutoff(
    vec: Vector, freq: float, dx: float, chunk_size: int
) -> None:
    """Remove waves that do not pass through the detector. Modify the vector in place."""

    # the idea here is that instead of generating fftfreq/dx and then
    # comparing it to `freq` pointwise, we note that fftfreq first goes
    # from zero to 0.5 and then -0.5 to 0. Therefore we just have to remove
    # a part centered on the middle, the outer sections remain as they are.
    frac = freq * dx
    n = len(vec)
    removal_start_idx = math.ceil(n * frac)
    if removal_start_idx > n / 2:
        # nothing to do
        return

    removal_end_idx = n - removal_start_idx + 1

    def modifier(idx: int, chunk: np.ndarray) -> None:
        start = max(removal_start_idx - idx, 0)
        end = max(removal_end_idx - idx, 0)
        chunk[start:end] = 0.0

    vec.modify_chunked(chunk_size, modifier)


def propagate(
    u: Vector,
    U: Vector,
    dx: float,
    wl: float,  # wavelength
    dz: float,  # z distance to propagate
    chunk_size: int,  # number of vector entries to load into memory at once
    cutoff_frequency: float,
    skip_fft: bool = False,
) -> None:
    """
    Modify the provided vector to propagate it through empty space in the z direction using the Fresnel
    approximation.

    The `U` argument doesn't have to be a valid vector at the beginning, it will be overwritten before
    anything is read from it. After this function returns it will contain the fourier transform of the
    propagated vector.

    However if `skip_fft` is True, then the `U` argument should be the fourier transform of `u` and then
    we can skip the fft operation.
    """

    n = len(u)

    def propagate_chunk(idx: int, chunk: np.ndarray) -> None:
        fx = fftfreq_chunk(n, idx, chunk_size) / dx
        chunk *= np.exp(1j * 2 * np.pi / wl * dz) * np.exp(
            -1j * np.pi * wl * dz * (fx**2)
        )

    if not skip_fft:
        u.fft(U)

    U.modify_chunked(
        chunk_size,
        propagate_chunk,
    )
    apply_frequency_cutoff(U, cutoff_frequency, dx, chunk_size)
    U.ifft(u)


def propagate_2d(
    u: Vector,
    U: Vector,
    dx: float,
    dy: float,
    wl: float,
    dz: float,
    chunk_size: int,
    cutoff_frequency: float,
    nx: int,
    ny: int,
    skip_fft: bool = False,
) -> None:
    """
    2D Fresnel propagation through empty space in the z direction.

    The wave field is stored in row-major flat order (ny rows of nx columns).
    The 2D FFT is decomposed into row-wise + column-wise 1D FFTs via the
    Vector protocol's fft2/ifft2 methods.

    The `U` argument doesn't have to be a valid vector at the beginning,
    it will be overwritten before anything is read from it. After this function
    returns it will contain the fourier transform of the propagated vector.

    However if `skip_fft` is True, then the `U` argument should be the fourier
    transform of `u` and then we can skip the fft operation.
    """

    require_row_aligned_chunk_size(chunk_size, nx, ny)
    if len(u) != nx * ny or len(U) != nx * ny:
        raise ValueError("2D u/U vector lengths must equal nx*ny")
    kx2 = _frequency_squared(nx, dx)
    ky2 = _frequency_squared(ny, dy)
    global_phase = np.exp(1j * 2 * np.pi / wl * dz)
    phase_scale = -1j * np.pi * wl * dz

    def propagate_chunk(idx: int, chunk: np.ndarray) -> None:
        tile, y_start = _tile_view(idx, chunk, nx, ny)
        k_sq = ky2[y_start:y_start + tile.shape[0], None] + kx2[None, :]
        tile *= global_phase * np.exp(phase_scale * k_sq)

    if not skip_fft:
        u.fft2(U, nx, ny)

    U.modify_chunked(chunk_size, propagate_chunk)
    apply_frequency_cutoff_2d(U, cutoff_frequency, dx, dy, nx, ny, chunk_size)
    U.ifft2(u, nx, ny)


def propagate_analytically(u: Vector, z: float, x_source: float, params: SimParams):
    def analytical(idx: int, chunk: np.ndarray):
        x = (
            idx + np.arange(len(chunk), dtype=np.float64) - params.N / 2
        ) * params.dx - x_source
        r = np.sqrt(x**2 + z**2)
        # we use the sqrt of r instead of r so that the probability, i.e. wave function squared, decreases linearly with
        # distance instead of quadratically. This is because we are effectively in 2d and not 3d.
        chunk[:] = np.exp(r * (2j * np.pi / params.wl)) / np.sqrt(r)

    u.write_chunked(params.chunk_size, analytical)


def propagate_analytically_2d(
    u: Vector, z: float, x_source: float, y_source: float, params: SimParams
):
    """
    2D spherical wave initialization from a point source at (x_source, y_source).

    Amplitude is 1/r (spherical wave for 3D space), matching fast-wave's
    ``propagate_analytically_2d_kernel``. This differs from the 1D version
    which uses 1/sqrt(r) (cylindrical wave for 2D simulation).
    """
    nx = params.nx
    ny = params.ny
    require_row_aligned_chunk_size(params.chunk_size, nx, ny)
    if len(u) != nx * ny:
        raise ValueError(f"2D vector length {len(u)} does not match nx*ny={nx * ny}")
    x = (np.arange(nx, dtype=np.float64) - nx / 2.0) * params.dx - x_source
    y = (np.arange(ny, dtype=np.float64) - ny / 2.0) * params.get_dy() - y_source
    x2 = np.square(x)
    y2 = np.square(y)
    wave_number = 2j * np.pi / params.wl

    def analytical(idx: int, chunk: np.ndarray):
        tile, y_start = _tile_view(idx, chunk, nx, ny)
        r = np.sqrt(y2[y_start:y_start + tile.shape[0], None] + x2[None, :] + z**2)
        tile[:] = np.exp(r * wave_number) / r

    u.write_chunked(params.chunk_size, analytical)


def fftfreq_1d(n: int, i: int) -> float:
    """
    Single-value version of np.fft.fftfreq.

    Returns the frequency bin value for index i in a 1D FFT of length n.
    """
    N = (n - 1) // 2 + 1
    if i < N:
        return i / n
    else:
        return (i - n) / n


def fftfreq_chunk(n: int, start: int, chunk_size: int) -> np.ndarray:
    """
    Generate a chunk of the fftfreq array as computed by np.fft.fftfreq.

    Adapted from https://github.com/numpy/numpy/blob/v1.24.0/numpy/fft/helper.py#L123-L169

    This leads to some ugly calculations, but we have full unit test coverage so it's fine :D
    """

    N = (n - 1) // 2 + 1

    # ph, nh = positive/negative half, the first half of the fftfreq array contains positive values
    start_ph = start
    end_ph = min(max(start_ph, N), start + chunk_size)
    start_nh = max(-(n // 2), start - n)
    end_nh = min(0, max(start + chunk_size - n, start_nh))

    ph = np.arange(start_ph, end_ph, dtype=int)
    nh = np.arange(start_nh, end_nh, dtype=int)

    fac = 1.0 / n
    return np.concatenate((ph, nh)) * fac


def compute_cutoff_angles(
    detector_size: float,
    dx: float,
    energy_range: Tuple[float, float],
    z_source: float,
    z_distances: list[float],
    element_heights: list[float],
    z_detector: float,
    max_x: float,
) -> Tuple[list[float], list[float]]:
    """
    Calculate the list of angles to which we limit the simulation that ensures that all interesting rays are kept
    but tries to remove as much of the excess rays as possible. In earlier z ranges we can keep a narrow angle,
    for later z values the angle will have to open up.

    Parameters
    ----------
    detector_size : float
        Width of the detector in metres
    dx : float
        Spacing between points in the simulation.
    energy_range : Tuple[float, float]
        A tuple of the minimum and maximum photon energy in eV
    z_distances : list[float]
        List of z distances from the source where gratings and samples are located. Note that
        this function doesn't consider the thickness of these optical elements.
    element_heights : list[float]
        List of the z-heights of all of the optical elements in the simulation
    z_detector : float
        Distance to the detector.
    max_x : float
        Maximal absolute source x position from which we want to consider rays

    Returns
    -------
    (cutoff angles, max x at each of the optical element starts and at z_detector) : Tuple[list[float], list[float]]
    """

    assert len(z_distances) == len(element_heights)
    assert energy_range[0] <= energy_range[1]

    wl_e_min = convert_energy_wavelength(energy_range[0])
    wl_e_max = convert_energy_wavelength(energy_range[1])

    direct_source_to_detector_angle = np.arctan(
        (detector_size / 2 + max_x) / z_detector
    )
    if 2 / (np.sin(direct_source_to_detector_angle) / wl_e_max) < dx:
        deg = direct_source_to_detector_angle * 180.0 / math.pi
        threshold = 2 / (np.sin(direct_source_to_detector_angle) / wl_e_max)
        raise ValueError(
            f"Grid sampled to coarsly, highest energy waves won't cover the full detector! Angle {deg} degrees requires dx <= {threshold}."
        )

    # When figuring out the largest angles we have to account for, it makes sense to first figure out how large the angles can even get
    # from a frequency standpoint. It doesn't make sense to allow for huge angles that will never be filled in the frequency spectrum
    # anyway. Low energies correspond to large angles, so that's where we are the most constrained.
    e_min_gradient: float
    if wl_e_min * 2 / dx < 1:
        e_min_angle = np.arcsin(wl_e_min * 2 / dx)
        e_min_gradient = np.tan(e_min_angle)
    else:
        # The angle is 90 degrees, no restriction gained
        e_min_gradient = float(np.inf)

    current_max_x = max_x
    current_z = z_source
    max_gradients: list[float] = []
    max_x_list: list[float] = []
    for next_z in z_distances + [z_detector]:
        if current_z == z_detector:
            raise ValueError(
                "An element in the list has the same z coordinate as the detector! This is not allowed."
            )

        max_gradients.append(
            min(
                (detector_size / 2 + current_max_x) / (z_detector - current_z),
                e_min_gradient,
            )
        )
        current_max_x += max_gradients[-1] * (next_z - current_z)
        max_x_list.append(float(current_max_x))
        current_z = next_z

    assert current_max_x >= detector_size / 2

    # When the detector is placed right after the last element without any gap in
    # between, we want to not increase the cutoff angle any further for this last
    # step, since we're not going to perform any further propagation there.
    skip_last_propagate = (
        len(z_distances) > 0
        and np.abs(z_detector - (z_distances[-1] + element_heights[-1])) < 1e-8
    )
    if skip_last_propagate:
        max_gradients[-1] = max_gradients[-2]
        max_x_list[-1] = max_x_list[-2] + max_gradients[-1] * element_heights[-1]

    return ([float(np.arctan(g)) for g in max_gradients], max_x_list)


def square_and_downsample(
    u: Vector, sim_params: SimParams, current_z: float
) -> np.ndarray:
    """
    Compute the detector output from `u` by cropping to the detector size, computing the intensity, and downscaling.
    """

    outsize = int(sim_params.detector_size // sim_params.detector_pixel_size_x)
    assert outsize <= sim_params.N

    out = np.zeros(outsize, dtype=np.float64)
    n = len(u)
    assert n >= outsize

    # Weird hack: np.histogram doesn't use half-open intervals for every bin, instead the upper range
    # end is actually inclusive, which just seems weird. We get around this by adding an additional
    # bin at the upper end, which we then discard. See the [:-1] below.
    detector_range = (
        -(outsize / 2) * sim_params.detector_pixel_size_x,
        (outsize / 2 + 1) * sim_params.detector_pixel_size_x,
    )
    nr_bins = outsize + 1

    def f(idx: int, chunk: np.ndarray) -> None:
        positions = (idx + np.arange(len(chunk)) - sim_params.N / 2) * sim_params.dx
        abs_chunk = chunk.real * chunk.real + chunk.imag * chunk.imag

        out[:] += np.histogram(
            positions, bins=nr_bins, range=detector_range, weights=abs_chunk
        )[0][:-1]

    u.read_chunked(sim_params.chunk_size, f)

    x = detector_x_vector(sim_params.detector_size, sim_params.detector_pixel_size_x)
    angles = np.arctan(x / current_z)
    return (
        out
        * np.cos(angles)
        * sim_params.dx
        * sim_params.detector_pixel_size_y
        / np.sqrt(x**2 + current_z**2)
    )


def _detector_cancelled(token: Any | None) -> bool:
    """Support the same callable/event-style cancellation tokens as P2."""
    if token is None:
        return False
    if callable(token):
        return bool(token())
    for name in ("is_cancelled", "is_set", "cancelled"):
        if hasattr(token, name):
            value = getattr(token, name)
            return bool(value() if callable(value) else value)
    return False


def _detector_progress(
    sim_params: SimParams, stage: str, completed: int, total: int
) -> None:
    if _detector_cancelled(sim_params.detector_cancel_token):
        raise InterruptedError(f"2D detector cancelled during {stage}")
    if sim_params.detector_progress_cb is not None:
        sim_params.detector_progress_cb(stage, int(completed), int(total))


def _safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _allocate_detector_output(
    sim_params: SimParams, shape: tuple[int, int]
) -> tuple[np.ndarray, str]:
    """Allocate the detector image, spilling it to a temporary NPY memmap if needed."""
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float64).itemsize
    budget = (
        int(sim_params.memory_budget_gb * 1024**3)
        if sim_params.memory_budget_gb > 0
        else None
    )
    if budget is None or nbytes <= budget:
        return np.zeros(shape, dtype=np.float64), "memory"

    directory = Path(sim_params.detector_output_dir or tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(
        prefix="big-wave-detector-", suffix=".npy", dir=directory
    )
    os.close(fd)
    try:
        out = np.lib.format.open_memmap(
            filename, mode="w+", dtype=np.float64, shape=shape
        )
    except Exception:
        _safe_unlink(filename)
        raise
    # A newly truncated NPY data region is logically zero. Touching every page
    # here would transiently resident-set the entire oversized output.
    # ndarray subclasses accept attributes; retain the finalizer on the object
    # so direct unit-test callers do not leak the temporary file.
    out._rave_detector_temp_path = filename  # type: ignore[attr-defined]
    out._rave_detector_finalizer = weakref.finalize(  # type: ignore[attr-defined]
        out, _safe_unlink, filename
    )
    return out, "memmap"


def _flush_detector_output(out: np.ndarray, drop_pages: bool = False) -> None:
    if not isinstance(out, np.memmap):
        return
    mapping = getattr(out, "_mmap", None)
    if mapping is None or getattr(mapping, "closed", False):
        return
    out.flush()
    if (
        drop_pages
        and mapping is not None
        and hasattr(mapping, "madvise")
        and hasattr(mmap, "MADV_DONTNEED")
    ):
        mapping.madvise(mmap.MADV_DONTNEED)


def cleanup_detector_output(out: np.ndarray) -> None:
    """Release and remove a P3 temporary memmap after its formal result is saved."""
    if not isinstance(out, np.memmap):
        return
    _flush_detector_output(out)
    mapping = getattr(out, "_mmap", None)
    if mapping is not None and not getattr(mapping, "closed", False):
        mapping.close()
    finalizer = getattr(out, "_rave_detector_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


def _detector_geometry(
    sim_params: SimParams, current_z: float
) -> tuple[
    int,
    int,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    nx = sim_params.nx
    ny = sim_params.ny
    dx = sim_params.dx
    dy = sim_params.get_dy()
    physical_ds_px = sim_params.detector_pixel_size_x
    physical_ds_py = sim_params.detector_pixel_size_y
    ds_px, ds_py, detector_z, magnification = (
        sim_params.effective_detector_geometry(current_z)
    )
    if min(nx, ny) <= 0 or min(dx, dy, ds_px, ds_py) <= 0:
        raise ValueError("2D detector grid and pixel spacings must be positive")
    if nx * ny != sim_params.N:
        raise ValueError("2D detector shape does not match SimParams.N")
    # fast-wave fixes the output shape from physical detector pixels, then uses
    # magnification-scaled pixels for sampling and the obliquity factor.
    outsize_x = int(sim_params.get_detector_size_x() // physical_ds_px)
    outsize_y = int(sim_params.get_detector_size_y() // physical_ds_py)
    if outsize_x <= 0 or outsize_y <= 0:
        raise ValueError(
            f"2D detector output must be non-empty, got ({outsize_y}, {outsize_x})"
        )
    x_det = (np.arange(outsize_x, dtype=np.float64) - outsize_x // 2) * ds_px
    y_det = (np.arange(outsize_y, dtype=np.float64) - outsize_y // 2) * ds_py
    if not math.isfinite(detector_z) or detector_z == 0:
        raise ValueError(
            f"2D detector effective current_z must be finite and non-zero, got {detector_z}"
        )
    return (
        outsize_x,
        outsize_y,
        dx,
        dy,
        ds_px,
        ds_py,
        x_det,
        y_det,
        detector_z,
        magnification,
    )


def _legacy_index_ranges(
    out_n: int, grid_n: int, ds: float, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """CUDA-compatible truncation-toward-zero detector index ranges."""
    pixel = np.arange(out_n, dtype=np.float64)
    detector_position = (pixel - out_n // 2) * ds
    lo = (
        np.trunc((detector_position - ds * 0.5) / spacing).astype(np.int64)
        + grid_n // 2
    )
    hi = (
        np.trunc((detector_position + ds * 0.5) / spacing).astype(np.int64)
        + grid_n // 2
    )
    return np.clip(lo, 0, grid_n), np.clip(hi, 0, grid_n)


def _legacy_fastwave_stream(
    u: Vector, sim_params: SimParams, current_z: float
) -> np.ndarray:
    (
        outsize_x,
        outsize_y,
        dx,
        dy,
        ds_px,
        ds_py,
        x_det,
        y_det,
        detector_z,
        magnification,
    ) = _detector_geometry(sim_params, current_z)
    nx, ny = sim_params.nx, sim_params.ny
    require_row_aligned_chunk_size(sim_params.chunk_size, nx, ny)
    if len(u) != nx * ny:
        raise ValueError(f"2D detector vector length {len(u)} != nx*ny={nx * ny}")

    lo_x, hi_x = _legacy_index_ranges(outsize_x, nx, ds_px, dx)
    lo_y, hi_y = _legacy_index_ranges(outsize_y, ny, ds_py, dy)
    count_x = hi_x - lo_x
    count_y = hi_y - lo_y
    zero_count = outsize_x * outsize_y - int(np.count_nonzero(count_x)) * int(
        np.count_nonzero(count_y)
    )
    ratio_x = ds_px / dx
    ratio_y = ds_py / dy
    non_integer_x = not math.isclose(ratio_x, round(ratio_x), rel_tol=1e-12, abs_tol=1e-12)
    non_integer_y = not math.isclose(ratio_y, round(ratio_y), rel_tol=1e-12, abs_tol=1e-12)
    if non_integer_x or non_integer_y:
        warnings.warn(
            "legacy_fastwave detector pixel/grid ratio is non-integer; "
            "the CUDA-compatible count map can be non-uniform",
            RuntimeWarning,
            stacklevel=3,
        )
    if zero_count:
        warnings.warn(
            f"legacy_fastwave detector has {zero_count} zero-count pixels",
            RuntimeWarning,
            stacklevel=3,
        )

    out, storage = _allocate_detector_output(sim_params, (outsize_y, outsize_x))
    prefix = np.empty(nx + 1, dtype=np.float64)
    prefix[0] = 0.0
    x_squared = np.square(x_det)
    z_squared = detector_z * detector_z
    cosine_cache: dict[int, np.ndarray] = {}
    _detector_progress(sim_params, "detector_integrate", 0, ny)

    def integrate_tile(idx: int, chunk: np.ndarray) -> None:
        tile, y_start = _tile_view(idx, chunk, nx, ny)
        for local_row, row in enumerate(tile):
            grid_y = y_start + local_row
            intensity = row.real * row.real + row.imag * row.imag
            np.cumsum(intensity, dtype=np.float64, out=prefix[1:])
            row_sums = prefix[hi_x] - prefix[lo_x]
            first_output_y = int(np.searchsorted(hi_y, grid_y, side="right"))
            end_output_y = int(np.searchsorted(lo_y, grid_y, side="right"))
            for output_y in range(first_output_y, end_output_y):
                cosine = cosine_cache.get(output_y)
                if cosine is None:
                    y = y_det[output_y]
                    cosine = detector_z / np.sqrt(x_squared + y * y + z_squared)
                    cosine_cache[output_y] = cosine
                out[output_y, :] += row_sums * (dx * dy) * cosine
            for output_y in tuple(cosine_cache):
                if hi_y[output_y] <= grid_y + 1:
                    del cosine_cache[output_y]
        _flush_detector_output(out, drop_pages=True)
        _detector_progress(
            sim_params, "detector_integrate", y_start + tile.shape[0], ny
        )

    u.read_chunked(sim_params.chunk_size, integrate_tile)
    _flush_detector_output(out, drop_pages=True)
    _detector_progress(sim_params, "detector_geometry", 0, 1)
    _detector_progress(sim_params, "detector_geometry", 1, 1)

    min_count = int(count_x.min()) * int(count_y.min())
    max_count = int(count_x.max()) * int(count_y.max())
    sim_params.detector_last_metadata = {
        "integrator": "legacy_fastwave",
        "version": DETECTOR_INTEGRATOR_VERSIONS["legacy_fastwave"],
        "streaming": True,
        "geometry_application": "fused_row_accumulation",
        "output_shape": [outsize_y, outsize_x],
        "output_storage": storage,
        "effective_geometry": {
            "dx": float(dx),
            "dy": float(dy),
            "detector_pixel_size_x": float(ds_px),
            "detector_pixel_size_y": float(ds_py),
            "current_z": float(detector_z),
            "magnification": float(magnification),
            "fresnel_scaled": bool(magnification != 1.0),
        },
        "pixel_to_grid_ratio": {"x": float(ratio_x), "y": float(ratio_y)},
        "non_integer_pixel_ratio": {"x": non_integer_x, "y": non_integer_y},
        "count_map": {
            "min": min_count,
            "max": max_count,
            "mean": float(np.mean(count_x) * np.mean(count_y)),
            "zero_pixels": zero_count,
            "non_uniform": min_count != max_count,
        },
    }
    return out


def _area_axis_plan(
    detector_positions: np.ndarray,
    detector_spacing: float,
    grid_n: int,
    grid_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute clipped piecewise-constant integration coordinates for one axis."""
    grid_left = (-(grid_n // 2) - 0.5) * grid_spacing
    grid_right = grid_left + grid_n * grid_spacing
    detector_lo = detector_positions - detector_spacing * 0.5
    detector_hi = detector_positions + detector_spacing * 0.5
    clipped_lo = np.clip(detector_lo, grid_left, grid_right)
    clipped_hi = np.clip(detector_hi, grid_left, grid_right)
    clipped_hi = np.maximum(clipped_hi, clipped_lo)

    scaled_lo = (clipped_lo - grid_left) / grid_spacing
    scaled_hi = (clipped_hi - grid_left) / grid_spacing
    index_lo = np.floor(scaled_lo).astype(np.int64)
    index_hi = np.floor(scaled_hi).astype(np.int64)
    fraction_lo = scaled_lo - index_lo
    fraction_hi = scaled_hi - index_hi
    # Floating roundoff at the right edge must map exactly to prefix[grid_n].
    at_right_lo = index_lo >= grid_n
    at_right_hi = index_hi >= grid_n
    index_lo[at_right_lo] = grid_n
    index_hi[at_right_hi] = grid_n
    fraction_lo[at_right_lo] = 0.0
    fraction_hi[at_right_hi] = 0.0
    return index_lo, fraction_lo, index_hi, fraction_hi, detector_lo, detector_hi


def _prefix_integral_at(
    intensity: np.ndarray,
    prefix: np.ndarray,
    indices: np.ndarray,
    fractions: np.ndarray,
    spacing: float,
) -> np.ndarray:
    result = prefix[indices] * spacing
    interior = indices < len(intensity)
    result[interior] += (
        fractions[interior] * spacing * intensity[indices[interior]]
    )
    return result


def _area_v1_stream(
    u: Vector, sim_params: SimParams, current_z: float
) -> np.ndarray:
    (
        outsize_x,
        outsize_y,
        dx,
        dy,
        ds_px,
        ds_py,
        x_det,
        y_det,
        detector_z,
        magnification,
    ) = _detector_geometry(sim_params, current_z)
    nx, ny = sim_params.nx, sim_params.ny
    require_row_aligned_chunk_size(sim_params.chunk_size, nx, ny)
    if len(u) != nx * ny:
        raise ValueError(f"2D detector vector length {len(u)} != nx*ny={nx * ny}")

    x_i0, x_f0, x_i1, x_f1, x_lo, x_hi = _area_axis_plan(
        x_det, ds_px, nx, dx
    )
    _, _, _, _, y_lo, y_hi = _area_axis_plan(y_det, ds_py, ny, dy)
    grid_y_left = (-(ny // 2) - 0.5) * dy
    prefix = np.empty(nx + 1, dtype=np.float64)
    prefix[0] = 0.0
    out, storage = _allocate_detector_output(sim_params, (outsize_y, outsize_x))
    x_squared = np.square(x_det)
    z_squared = detector_z * detector_z
    cosine_cache: dict[int, np.ndarray] = {}
    _detector_progress(sim_params, "detector_integrate", 0, ny)

    def integrate_tile(idx: int, chunk: np.ndarray) -> None:
        tile, y_start = _tile_view(idx, chunk, nx, ny)
        for local_row, row in enumerate(tile):
            grid_y = y_start + local_row
            cell_lo = grid_y_left + grid_y * dy
            cell_hi = cell_lo + dy
            first_output_y = int(np.searchsorted(y_hi, cell_lo, side="right"))
            end_output_y = int(np.searchsorted(y_lo, cell_hi, side="left"))
            if end_output_y <= first_output_y:
                continue
            intensity = row.real * row.real + row.imag * row.imag
            np.cumsum(intensity, dtype=np.float64, out=prefix[1:])
            x_integral = _prefix_integral_at(
                intensity, prefix, x_i1, x_f1, dx
            ) - _prefix_integral_at(intensity, prefix, x_i0, x_f0, dx)
            for output_y in range(first_output_y, end_output_y):
                overlap_y = min(cell_hi, y_hi[output_y]) - max(
                    cell_lo, y_lo[output_y]
                )
                if overlap_y > 0:
                    cosine = cosine_cache.get(output_y)
                    if cosine is None:
                        y = y_det[output_y]
                        cosine = detector_z / np.sqrt(
                            x_squared + y * y + z_squared
                        )
                        cosine_cache[output_y] = cosine
                    out[output_y, :] += x_integral * overlap_y * cosine
            for output_y in tuple(cosine_cache):
                if y_hi[output_y] <= cell_hi:
                    del cosine_cache[output_y]
        _flush_detector_output(out, drop_pages=True)
        _detector_progress(
            sim_params, "detector_integrate", y_start + tile.shape[0], ny
        )

    u.read_chunked(sim_params.chunk_size, integrate_tile)
    _flush_detector_output(out, drop_pages=True)
    _detector_progress(sim_params, "detector_geometry", 0, 1)
    _detector_progress(sim_params, "detector_geometry", 1, 1)

    grid_x_left = (-(nx // 2) - 0.5) * dx
    grid_x_right = grid_x_left + nx * dx
    grid_y_right = grid_y_left + ny * dy
    covered_x = np.maximum(
        0.0, np.minimum(x_hi, grid_x_right) - np.maximum(x_lo, grid_x_left)
    )
    covered_y = np.maximum(
        0.0, np.minimum(y_hi, grid_y_right) - np.maximum(y_lo, grid_y_left)
    )
    zero_coverage = outsize_x * outsize_y - int(np.count_nonzero(covered_x)) * int(
        np.count_nonzero(covered_y)
    )
    if zero_coverage:
        warnings.warn(
            f"area_v1 detector has {zero_coverage} zero-coverage pixels",
            RuntimeWarning,
            stacklevel=3,
        )
    sim_params.detector_last_metadata = {
        "integrator": "area_v1",
        "version": DETECTOR_INTEGRATOR_VERSIONS["area_v1"],
        "streaming": True,
        "geometry_application": "fused_row_accumulation",
        "output_shape": [outsize_y, outsize_x],
        "output_storage": storage,
        "effective_geometry": {
            "dx": float(dx),
            "dy": float(dy),
            "detector_pixel_size_x": float(ds_px),
            "detector_pixel_size_y": float(ds_py),
            "current_z": float(detector_z),
            "magnification": float(magnification),
            "fresnel_scaled": bool(magnification != 1.0),
        },
        "coverage_area": {
            "min": float(covered_x.min() * covered_y.min()),
            "max": float(covered_x.max() * covered_y.max()),
            "mean": float(np.mean(covered_x) * np.mean(covered_y)),
            "zero_pixels": zero_coverage,
        },
    }
    return out


def square_and_downsample_2d(
    u: Vector, sim_params: SimParams, current_z: float
) -> np.ndarray:
    """Stream a 2D field into the configured detector integration backend.

    ``legacy_fastwave`` preserves fast-wave's integer truncation/count-map
    semantics. ``area_v1`` integrates separable physical overlap areas. Both
    paths keep only a field-row tile plus the detector output in memory and
    apply the geometric cosine without an output-sized mesh grid.
    """
    if sim_params.detector_integrator == "legacy_fastwave":
        return _legacy_fastwave_stream(u, sim_params, current_z)
    if sim_params.detector_integrator == "area_v1":
        return _area_v1_stream(u, sim_params, current_z)
    raise ValueError(
        f"unsupported detector_integrator={sim_params.detector_integrator!r}"
    )
