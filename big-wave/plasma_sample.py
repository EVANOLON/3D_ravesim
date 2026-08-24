"""
PlasmaSample — an OpticalElement for X-ray wave propagation through
laser-produced plasma described by radiation-hydrodynamic output grids.

Mirrors precise_Sample in structure, but replaces material_index + density
with ne, ni, Te, Z* grids and evaluates plasma optics in vectorized tiles.

Can be imported independently of the rest of big-wave — all heavy imports
are deferred to apply().
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from plasma import plasma_delta_beta_grid

if TYPE_CHECKING:
    from history import History
    from optical_element import DeltabetaTable
    from propagation import SimParams
    from vector import Vector

logger = logging.getLogger("big-wave")


def _material_factor(deltabeta: np.ndarray, thickness: float, wl: float) -> np.ndarray:
    """exp(2*pi*i * thickness/wl * deltabeta) — inlined from optical_element."""
    return np.exp(-2j * np.pi * thickness / wl * np.conj(deltabeta))


@dataclass
class PlasmaSample:
    """
    A plasma region represented by electron/ion density, temperature, and
    ionization-state grids on a 2D (z x x) rectilinear mesh.

    Axis 0 = z (propagation direction), axis 1 = x (transverse direction).
    The sample is centred on the optical axis (middle column -> x = 0).

    Vacuum is represented by n_e == 0 (or equivalently n_i == 0), which
    yields delta = beta = 0.
    """

    z_start: float          # z-coordinate of the front face [m]
    pixel_size_x: float     # transverse pixel size [m] (x direction)
    pixel_size_z: float     # longitudinal pixel (layer) thickness [m]
    ne_grid: np.ndarray     # electron number density [cm^-3], shape (nz, nx) or (nz, ny, nx)
    ni_grid: np.ndarray     # ion number density [cm^-3]
    te_grid: np.ndarray     # electron temperature [eV]
    zstar_grid: np.ndarray  # average ionization state
    Z: int                  # atomic number of the plasma element
    x_positions: np.ndarray # phase-stepping offsets in x [m]
    # 2D support (defaults for backward compat)
    pixel_size_y: float = 0.0          # pixel size in y [m]; 0 = 1D mode
    y_positions: Optional[np.ndarray] = None  # phase-stepping offsets in y

    def check_valid(self) -> None:
        shape = self.ne_grid.shape
        ndim = len(shape)
        assert ndim in (2, 3), f"PlasmaSample grids must be 2D (z, x) or 3D (z, y, x), got shape {shape}"
        assert shape[0] > 0
        for name, g in [
            ("ni", self.ni_grid),
            ("te", self.te_grid),
            ("zstar", self.zstar_grid),
        ]:
            assert g.shape == shape, (
                f"{name}_grid shape mismatch: {g.shape} != {shape}"
            )
        assert self.z_start > 0
        assert self.pixel_size_x > 0
        assert self.pixel_size_z > 0
        if ndim == 3:
            assert self.pixel_size_y > 0, "3D grids require pixel_size_y > 0"
            assert shape[1] >= 2 and shape[2] >= 2, "3D grids require at least 2x2 transverse pixels"

    def get_thickness(self) -> float:
        return self.pixel_size_z * self.ne_grid.shape[0]

    def store_deltabetas(self, deltabeta_table) -> None:
        # delta/beta computed on-the-fly per pixel; nothing to pre-store.
        pass

    def apply(
        self,
        u,
        U,
        sim_params,
        cutoff_freq: float,
        stepping_iteration: int,
        history,
    ) -> None:
        if sim_params.is_2d and self.ne_grid.ndim == 3:
            self._apply_2d(u, U, sim_params, cutoff_freq, stepping_iteration, history)
        else:
            self._apply_1d(u, U, sim_params, cutoff_freq, stepping_iteration, history)

    def _apply_2d(self, u, U, sim_params, cutoff_freq, stepping_iteration, history):
        """2D plasma propagation using mmap-backed, vectorized row tiles."""
        from propagation import (
            _tile_view,
            convert_energy_wavelength,
            propagate_2d,
            require_row_aligned_chunk_size,
            square_and_downsample_2d,
        )

        energy = convert_energy_wavelength(sim_params.wl)
        nz, ny, nx = self.ne_grid.shape
        dy_s = self.pixel_size_y if self.pixel_size_y != 0.0 else self.pixel_size_x

        x_pos = self.x_positions[stepping_iteration]
        y_pos = (self.y_positions[stepping_iteration]
                 if self.y_positions is not None else 0.0)

        sim_nx = sim_params.nx
        sim_ny = sim_params.ny
        require_row_aligned_chunk_size(sim_params.chunk_size, sim_nx, sim_ny)

        x_idx = (
            (np.arange(sim_nx, dtype=np.float64) - sim_nx / 2.0) * sim_params.dx
            + x_pos
            + nx * self.pixel_size_x * 0.5
        ) / self.pixel_size_x
        x_floor_raw = np.floor(x_idx).astype(np.int64)
        x_frac = x_idx - x_floor_raw
        inside_x = (x_idx >= 0.0) & (x_idx < float(nx) - 1.0)
        x_floor = np.clip(x_floor_raw, 0, nx - 2)
        sim_y = (
            np.arange(sim_ny, dtype=np.float64) - sim_ny / 2.0
        ) * sim_params.get_dy()

        start_slice = int(getattr(self, "_checkpoint_start_slice", 0))
        if start_slice < 0 or start_slice > nz:
            raise ValueError(f"invalid PlasmaSample checkpoint slice {start_slice}/{nz}")
        checkpoint_callback = getattr(self, "_checkpoint_callback", None)

        for rowidx in range(start_slice, nz):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample_2d(u, sim_params, z), z,
                )

            def modifier(idx, chunk):
                tile, y_start = _tile_view(idx, chunk, sim_nx, sim_ny)
                rows = tile.shape[0]
                y_idx = (
                    sim_y[y_start:y_start + rows]
                    + y_pos
                    + ny * dy_s * 0.5
                ) / dy_s
                y_floor_raw = np.floor(y_idx).astype(np.int64)
                y_frac = y_idx - y_floor_raw
                inside_y = (y_idx >= 0.0) & (y_idx < float(ny) - 1.0)
                y_floor = np.clip(y_floor_raw, 0, ny - 2)
                x0 = x_floor[None, :]
                y0 = y_floor[:, None]

                def corner_deltabeta(y_offset, x_offset):
                    indices = (rowidx, y0 + y_offset, x0 + x_offset)
                    delta, beta, _ = plasma_delta_beta_grid(
                        self.ne_grid[indices],
                        self.ni_grid[indices],
                        self.te_grid[indices],
                        self.zstar_grid[indices],
                        self.Z,
                        energy,
                    )
                    return delta + 1j * beta

                db_00 = corner_deltabeta(0, 0)
                db_01 = corner_deltabeta(0, 1)
                db_10 = corner_deltabeta(1, 0)
                db_11 = corner_deltabeta(1, 1)
                db_y0 = db_00 * (1.0 - x_frac[None, :]) + db_01 * x_frac[None, :]
                db_y1 = db_10 * (1.0 - x_frac[None, :]) + db_11 * x_frac[None, :]
                interpolated_db = (
                    db_y0 * (1.0 - y_frac[:, None]) + db_y1 * y_frac[:, None]
                )
                interpolated_db[~(inside_y[:, None] & inside_x[None, :])] = 0.0
                tile *= _material_factor(
                    interpolated_db, self.pixel_size_z, sim_params.wl
                )

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate_2d(
                u, U,
                sim_params.dx, sim_params.get_dy(),
                sim_params.wl, self.pixel_size_z,
                sim_params.chunk_size, cutoff_freq,
                sim_nx, sim_ny,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    rowidx + 1, self.z_start + (rowidx + 1) * self.pixel_size_z
                )

    def _apply_1d(
        self,
        u,
        U,
        sim_params,
        cutoff_freq: float,
        stepping_iteration: int,
        history,
    ) -> None:
        # Lazy imports — only triggered at simulation runtime, not at module load.
        from propagation import (
            convert_energy_wavelength,
            propagate,
            square_and_downsample,
        )

        energy = convert_energy_wavelength(sim_params.wl)
        nz, nx = self.ne_grid.shape

        # x-coordinates of the sample column centres (centred on axis)
        row_x = (
            np.arange(nx) * self.pixel_size_x
            + self.x_positions[stepping_iteration]
            - nx * self.pixel_size_x * 0.5
        )

        for rowidx in range(nz):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

            delta, beta, _ = plasma_delta_beta_grid(
                self.ne_grid[rowidx, :],
                self.ni_grid[rowidx, :],
                self.te_grid[rowidx, :],
                self.zstar_grid[rowidx, :],
                self.Z,
                energy,
            )
            row_deltabeta = delta + 1j * beta

            def modifier(idx: int, chunk: np.ndarray) -> None:
                start = idx - sim_params.N / 2
                x_chunk = np.arange(start, start + len(chunk)) * sim_params.dx
                deltabeta = np.interp(
                    x_chunk, xp=row_x, fp=row_deltabeta, left=0.0, right=0.0
                )
                chunk *= _material_factor(deltabeta, self.pixel_size_z, sim_params.wl)

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.pixel_size_z,
                sim_params.chunk_size,
                cutoff_freq,
            )
