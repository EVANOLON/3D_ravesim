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
        """2D plasma sample propagation — not yet implemented (see docs/plasma_2d_upgrade_design.md)."""
        raise NotImplementedError(
            "PlasmaSample._apply_2d is not yet implemented. "
            "See docs/plasma_2d_upgrade_design.md for the upgrade design."
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
