"""
PlasmaSample — an OpticalElement for X-ray wave propagation through
laser-produced plasma described by radiation-hydrodynamic output grids.

Mirrors precise_Sample in structure, but replaces material_index + density
with ne, ni, Te, Z* grids and calls plasma_delta_beta() per pixel.

Can be imported independently of the rest of big-wave — all heavy imports
are deferred to apply().
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from plasma import plasma_delta_beta

if TYPE_CHECKING:
    from history import History
    from optical_element import DeltabetaTable
    from propagation import SimParams
    from vector import Vector

logger = logging.getLogger("big-wave")


def _material_factor(deltabeta: np.ndarray, thickness: float, wl: float) -> np.ndarray:
    """exp(2*pi*i * thickness/wl * deltabeta) — inlined from optical_element."""
    return np.exp(2j * np.pi * thickness / wl * deltabeta)


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
        """2D plasma sample propagation with bilinear interpolation.

        The 2D deltabeta grid (ny × nx) is computed per z-slice by calling
        plasma_delta_beta() for each pixel, then bilinearly interpolated
        onto the simulation wavefront grid.  Mirrors Sample._apply_2d().
        """
        from propagation import (
            convert_energy_wavelength,
            propagate_2d,
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

        for rowidx in range(nz):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample_2d(u, sim_params, z), z,
                )

            # Build the 2D deltabeta grid for this z-slice: (ny, nx)
            ne_slice = self.ne_grid[rowidx, :, :]
            ni_slice = self.ni_grid[rowidx, :, :]
            te_slice = self.te_grid[rowidx, :, :]
            zs_slice = self.zstar_grid[rowidx, :, :]

            row_deltabeta = np.zeros((ny, nx), dtype=np.complex128)
            # Vectorise: find non-vacuum pixels
            mask = ne_slice > 0.0
            if mask.any():
                ne_m = ne_slice[mask].ravel()
                ni_m = ni_slice[mask].ravel()
                te_m = te_slice[mask].ravel()
                zs_m = zs_slice[mask].ravel()
                # Call plasma_delta_beta for each masked pixel
                d_list = []
                b_list = []
                for i in range(len(ne_m)):
                    d, b, _ = plasma_delta_beta(
                        float(ne_m[i]), float(ni_m[i]), float(te_m[i]),
                        float(zs_m[i]), self.Z, energy)
                    d_list.append(d)
                    b_list.append(b)
                row_deltabeta[mask] = np.array(d_list, dtype=np.float64) + \
                                      1j * np.array(b_list, dtype=np.float64)
            # vacuum pixels stay at 0+0j

            def modifier(idx, chunk):
                flat = idx + np.arange(len(chunk), dtype=np.int64)
                ix = flat % sim_nx
                iy = flat // sim_nx

                # Physical coordinates (centred, with phase step offset)
                x = (ix - sim_nx / 2.0) * sim_params.dx + x_pos + nx * self.pixel_size_x * 0.5
                y = (iy - sim_ny / 2.0) * sim_params.get_dy() + y_pos + ny * dy_s * 0.5

                # Map to sample grid coordinates
                x_idx = x / self.pixel_size_x
                y_idx = y / dy_s

                # Bilinear interpolation with edge clamping
                x_floor = np.floor(x_idx).astype(np.int64)
                y_floor = np.floor(y_idx).astype(np.int64)
                x_frac = x_idx - x_floor
                y_frac = y_idx - y_floor

                x_floor = np.clip(x_floor, 0, nx - 2)
                y_floor = np.clip(y_floor, 0, ny - 2)

                # Four corner values
                db_00 = row_deltabeta[y_floor, x_floor]
                db_01 = row_deltabeta[y_floor, x_floor + 1]
                db_10 = row_deltabeta[y_floor + 1, x_floor]
                db_11 = row_deltabeta[y_floor + 1, x_floor + 1]

                # Bilinear interpolation
                db_y0 = db_00 * (1.0 - x_frac) + db_01 * x_frac
                db_y1 = db_10 * (1.0 - x_frac) + db_11 * x_frac
                interpolated_db = db_y0 * (1.0 - y_frac) + db_y1 * y_frac

                chunk *= _material_factor(interpolated_db, self.pixel_size_z, sim_params.wl)

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate_2d(
                u, U,
                sim_params.dx, sim_params.get_dy(),
                sim_params.wl, self.pixel_size_z,
                sim_params.chunk_size, cutoff_freq,
                sim_nx, sim_ny,
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

            # Compute delta + i*beta for each column in this z-slice.
            row_deltabeta = np.zeros(nx, dtype=np.complex128)
            for colidx in range(nx):
                ne = float(self.ne_grid[rowidx, colidx])
                if ne <= 0.0:
                    row_deltabeta[colidx] = 0.0  # vacuum
                    continue
                ni = float(self.ni_grid[rowidx, colidx])
                Te = float(self.te_grid[rowidx, colidx])
                Zs = float(self.zstar_grid[rowidx, colidx])
                d, b, _ = plasma_delta_beta(ne, ni, Te, Zs, self.Z, energy)
                row_deltabeta[colidx] = complex(d, b)

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
