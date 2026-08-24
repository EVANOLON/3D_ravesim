# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
import logging
from typing import Optional, Protocol, Tuple, TypeVar
from nist_lookup.xraydb_plugin import xray_delta_beta  # type: ignore
import numpy as np

from history import History
from propagation import (
    SimParams,
    square_and_downsample,
    square_and_downsample_2d,
    propagate,
    propagate_2d,
    convert_energy_wavelength,
)
from vector import Vector

logger = logging.getLogger("big-wave")


@dataclass
class Material:
    """Description of one material from which a grating or a sample is made"""

    mat: str
    density: float


# Each entry in the table is a tuple of the material and the corresponding
# deltabeta. Note that this table is only valid for the one wavelength for
# which it was computed.
DeltabetaTable = list[tuple[Material, np.complex128]]

ComplexOrNdarray = TypeVar("ComplexOrNdarray", np.complex128, np.ndarray)


def material_factor(
    deltabeta: ComplexOrNdarray, thickness: float, wl: float
) -> ComplexOrNdarray:
    return np.exp(-2j * np.pi * thickness / wl * np.conj(deltabeta))


def extract_from_deltabeta_table(
    material: Material, deltabeta_table: DeltabetaTable
) -> np.complex128:
    """
    Find the best match in the deltabeta_table for the given material. This table is precomputed and
    stored in `subconfig.yaml` when the project directory is generated. That way the `OpticalElement`
    classes don't have to call nist_lookup at runtime.
    """

    candidates = sorted(
        [
            (abs(tm[0].density - material.density), tm[1])
            for tm in deltabeta_table
            if tm[0].mat == material.mat
        ]
    )
    assert len(candidates) > 0, f"Material {material.mat} not found in deltabeta table"
    assert (
        candidates[0][0] < 1e-5
    ), f"Material {material.mat} found with density error of {candidates[0][0]}"

    return candidates[0][1]


class OpticalElement(Protocol):
    """An object that can be put between the source and the detector"""

    z_start: float
    x_positions: np.ndarray

    def get_thickness(self) -> float:
        """The z distance covered by the apply function"""
        ...

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        """
        Assumes that the vector is given at the start z of the
        object, and then iterates through the object from there on.

        The caller is responsible for keeping track of the new z position,
        i.e. z_start + thickness.

        This will modify the provided vectors, the output will be located in `u`.

        All implementations gurantee that `U` will contain the FFT of `u` afterwards.
        """
        ...

    def check_valid(self) -> None:
        """
        Assert that the given configuration makes sense. If not,
        the function should raise an exception.
        """
        ...

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        """
        The materials are specified as a name and a density, but for the calculations we
        need the delta and beta values.

        This method stores the delta and beta values for each material in `self`, using
        the table that we get from `subconfig.yaml` where all the required delta and beta
        values are stored.
        """


@dataclass
class Grating(OpticalElement):
    """Description of a grating. A higher duty cycle corresponds to using more of material A"""

    pitch: float
    dc: Tuple[float, float]  # duty cycle
    z_start: float
    thickness: float  # does not include the substrate
    nr_steps: int
    x_positions: np.ndarray
    substrate_thickness: float
    mat_a: Optional[Material]
    mat_b: Optional[Material]
    mat_substrate: Optional[Material]

    def get_thickness(self) -> float:
        return self.thickness + self.substrate_thickness

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        z_step = self.thickness / self.nr_steps

        # In every step we multiply by a vector containing only two different factors, one per
        # material. We therefore precompute these two factors and allocate the `factor` vector
        # which we will update for every step and every chunk and then multiply on to the chunks.
        factor_a = material_factor(self.db_a, z_step, sim_params.wl)
        factor_b = material_factor(self.db_b, z_step, sim_params.wl)

        factor = np.zeros(sim_params.chunk_size, dtype=u.dtype)

        for i in range(self.nr_steps):
            logger.debug(f"Grating step {i}/{self.nr_steps}")

            z_fraction = i / self.nr_steps

            def modifier(idx: int, chunk: np.ndarray) -> None:
                mask = generate_grating_chunk(
                    sim_params.dx,
                    idx - sim_params.N // 2,
                    len(chunk),
                    self,
                    z_fraction,
                    stepping_iteration,
                )
                np.place(factor[: len(chunk)], mask, factor_a)
                np.place(factor[: len(chunk)], 1.0 - mask, factor_b)
                chunk *= factor[: len(chunk)]

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                z_step,
                sim_params.chunk_size,
                cutoff_freq,
            )

            if history is not None:
                z = self.z_start + (i + 1) * z_step
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

        if self.substrate_thickness > 0:
            factor_substrate = material_factor(
                self.db_substrate, self.substrate_thickness, sim_params.wl
            )

            def modifier(idx: int, chunk: np.ndarray) -> None:
                chunk *= factor_substrate

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.substrate_thickness,
                sim_params.chunk_size,
                cutoff_freq,
            )

    def check_valid(self) -> None:
        assert self.substrate_thickness >= 0
        assert self.thickness > 0

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_a = np.complex128(0.0)
        if self.mat_a is not None:
            self.db_a = extract_from_deltabeta_table(self.mat_a, deltabeta_table)

        self.db_b = np.complex128(0.0)
        if self.mat_b is not None:
            self.db_b = extract_from_deltabeta_table(self.mat_b, deltabeta_table)

        self.db_substrate = np.complex128(0.0)
        if self.mat_substrate is not None:
            self.db_substrate = extract_from_deltabeta_table(
                self.mat_substrate, deltabeta_table
            )

@dataclass
class EnvGrating(OpticalElement):
    """Description of a grating. A higher duty cycle corresponds to using more of material A"""

    pitch0: float # small pitch
    pitch1: float # large pitch
    dc0: Tuple[float, float]  # duty cycle
    dc1: Tuple[float, float]  # duty cycle
    z_start: float
    thickness: float  # does not include the substrate
    nr_steps: int
    x_positions: np.ndarray
    substrate_thickness: float
    mat_a: Optional[Material]
    mat_b: Optional[Material]
    mat_substrate: Optional[Material]

    def get_thickness(self) -> float:
        return self.thickness + self.substrate_thickness

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        z_step = self.thickness / self.nr_steps

        # In every step we multiply by a vector containing only two different factors, one per
        # material. We therefore precompute these two factors and allocate the `factor` vector
        # which we will update for every step and every chunk and then multiply on to the chunks.
        factor_a = material_factor(self.db_a, z_step, sim_params.wl)
        factor_b = material_factor(self.db_b, z_step, sim_params.wl)

        factor = np.zeros(sim_params.chunk_size, dtype=u.dtype)

        for i in range(self.nr_steps):
            logger.debug(f"Grating step {i}/{self.nr_steps}")

            z_fraction = i / self.nr_steps

            def modifier(idx: int, chunk: np.ndarray) -> None:
                mask = generate_env_grating_chunk(
                    sim_params.dx,
                    idx - sim_params.N // 2,
                    len(chunk),
                    self,
                    z_fraction,
                    stepping_iteration,
                )
                np.place(factor[: len(chunk)], mask, factor_a)
                np.place(factor[: len(chunk)], 1.0 - mask, factor_b)
                chunk *= factor[: len(chunk)]

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                z_step,
                sim_params.chunk_size,
                cutoff_freq,
            )

            if history is not None:
                z = self.z_start + (i + 1) * z_step
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

        if self.substrate_thickness > 0:
            factor_substrate = material_factor(
                self.db_substrate, self.substrate_thickness, sim_params.wl
            )

            def modifier(idx: int, chunk: np.ndarray) -> None:
                chunk *= factor_substrate

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate(
                u,
                U,
                sim_params.dx,
                sim_params.wl,
                self.substrate_thickness,
                sim_params.chunk_size,
                cutoff_freq,
            )

    def check_valid(self) -> None:
        assert self.substrate_thickness >= 0
        assert self.thickness > 0

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_a = np.complex128(0.0)
        if self.mat_a is not None:
            self.db_a = extract_from_deltabeta_table(self.mat_a, deltabeta_table)

        self.db_b = np.complex128(0.0)
        if self.mat_b is not None:
            self.db_b = extract_from_deltabeta_table(self.mat_b, deltabeta_table)

        self.db_substrate = np.complex128(0.0)
        if self.mat_substrate is not None:
            self.db_substrate = extract_from_deltabeta_table(
                self.mat_substrate, deltabeta_table
            )


def generate_grating_chunk(
    dx: float,
    start: int,
    chunk_size: int,
    grating: Grating,
    z_fraction: float,
    stepping_iteration: int,
) -> np.ndarray:
    """Generate a chunk of the grating as a boolean mask array"""

    dc = (1 - z_fraction) * grating.dc[0] + z_fraction * grating.dc[1]
    x_start = start * dx + grating.x_positions[stepping_iteration]
    x_stop = x_start + chunk_size * dx

    x = np.linspace(x_start, x_stop, chunk_size, endpoint=False)
    phase = np.mod(x / grating.pitch + dc / 2, 1)
    return phase < dc

def generate_env_grating_chunk(
    dx: float,
    start: int,
    chunk_size: int,
    grating: EnvGrating,
    z_fraction: float,
    stepping_iteration: int,
) -> np.ndarray:
    """Generate a chunk of the grating as a boolean mask array"""

    dc0 = (1 - z_fraction) * grating.dc0[0] + z_fraction * grating.dc0[1]
    dc1 = (1 - z_fraction) * grating.dc1[0] + z_fraction * grating.dc1[1]
    x_start = start * dx + grating.x_positions[stepping_iteration]
    x_stop = x_start + chunk_size * dx

    x = np.linspace(x_start, x_stop, chunk_size, endpoint=False)
    phase0 = np.mod(x / grating.pitch0 + dc0 / 2, 1) 
    phase1 = np.mod(x / grating.pitch1, 1)
    return (phase0 < dc0) * (phase1 < dc1)


@dataclass
class Sample(OpticalElement):
    """
    A sample to be analyzed, represented as a pixel grid of material indices.

    Grid shapes:
    - 1D simulation: grid[z, x]          (2D array)
    - 2D simulation: grid[z, y, x]       (3D array)

    Axis 0 = z (propagation direction)
    Axis 1 = y (vertical, for 2D only)
    Axis 2 = x (horizontal)

    Note that the indices into the materials list are 1-based because 0 represents an empty space.

    Currently does not support rotating the grid. Assumes rectangular pixels.

    Samples are centered on the x/y axes, i.e. the middle entry is at x=0, y=0.
    """

    z_start: float
    pixel_size_x: float  # in metres
    pixel_size_z: float  # in metres
    grid: np.ndarray
    materials: list[Material]
    x_positions: np.ndarray
    # 2D optional fields (must come after required fields for dataclass ordering)
    pixel_size_y: float = 0.0  # in metres, 0 means use pixel_size_x
    y_positions: Optional[np.ndarray] = None

    def check_valid(self) -> None:
        shape = self.grid.shape
        if len(shape) == 2:
            assert shape[0] > 0 and shape[1] > 0
        elif len(shape) == 3:
            assert shape[0] > 0 and shape[1] >= 2 and shape[2] >= 2
        else:
            raise ValueError(f"Sample grid must be 2D or 3D, got shape {shape}")
        assert self.grid.dtype in (np.uint32, np.int32), f"grid dtype must be uint32 or int32, got {self.grid.dtype}"
        assert self.z_start > 0
        assert self.pixel_size_x > 0
        assert self.pixel_size_z > 0
        assert len(self.materials) >= np.max(self.grid)

        if self.y_positions is not None:
            assert len(self.y_positions) == len(self.x_positions)

    def get_thickness(self) -> float:
        return self.pixel_size_z * self.grid.shape[0]

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        self.db_list: np.ndarray = np.zeros(
            len(self.materials) + 1, dtype=np.complex128
        )
        for idx, material in enumerate(self.materials):
            self.db_list[idx + 1] = extract_from_deltabeta_table(
                material, deltabeta_table
            )

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        if sim_params.is_2d and len(self.grid.shape) == 3:
            self._apply_2d(u, U, sim_params, cutoff_freq, stepping_iteration, history)
        else:
            self._apply_1d(u, U, sim_params, cutoff_freq, stepping_iteration, history)

    def _apply_1d(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        rowlen = self.grid.shape[1]
        row_x = (
            np.arange(rowlen) * self.pixel_size_x
            + self.x_positions[stepping_iteration]
            - rowlen * self.pixel_size_x * 0.5
        )
        for rowidx in range(self.grid.shape[0]):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )

            # Calculate the vector of delta+ibeta values for this row without applying material_factor yet since
            # we want to interpolate the values in this space, instead of the space after taking the exp of this.
            row_deltabeta: np.ndarray = self.db_list[self.grid[rowidx, :]]

            def modifier(idx: int, chunk: np.ndarray) -> None:
                start = idx - sim_params.N / 2
                x_chunk = np.arange(start, start + len(chunk)) * sim_params.dx
                deltabeta = np.interp(
                    x_chunk, xp=row_x, fp=row_deltabeta, left=0.0, right=0.0
                )
                chunk *= material_factor(deltabeta, self.pixel_size_z, sim_params.wl)

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

    def _apply_2d(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        """Apply a 3D material grid using row-aligned 2D wavefront tiles."""
        from propagation import require_row_aligned_chunk_size, _tile_view

        z_len, y_len, x_len = self.grid.shape
        dy_s = self.pixel_size_y if self.pixel_size_y != 0.0 else self.pixel_size_x

        x_pos = self.x_positions[stepping_iteration]
        y_pos = (self.y_positions[stepping_iteration]
                 if self.y_positions is not None else 0.0)

        nx = sim_params.nx
        ny = sim_params.ny
        require_row_aligned_chunk_size(sim_params.chunk_size, nx, ny)

        # The x mapping is identical for every y tile and z slice.  Only O(nx)
        # coordinate/index arrays are retained.
        x_idx = (
            (np.arange(nx, dtype=np.float64) - nx / 2.0) * sim_params.dx
            + x_pos
            + x_len * self.pixel_size_x * 0.5
        ) / self.pixel_size_x
        x_floor_raw = np.floor(x_idx).astype(np.int64)
        x_frac = x_idx - x_floor_raw
        inside_x = (x_idx >= 0.0) & (x_idx < float(x_len) - 1.0)
        x_floor = np.clip(x_floor_raw, 0, x_len - 2)
        sim_y = (np.arange(ny, dtype=np.float64) - ny / 2.0) * sim_params.get_dy()

        start_slice = int(getattr(self, "_checkpoint_start_slice", 0))
        if start_slice < 0 or start_slice > z_len:
            raise ValueError(f"invalid Sample checkpoint slice {start_slice}/{z_len}")
        checkpoint_callback = getattr(self, "_checkpoint_callback", None)

        for rowidx in range(start_slice, z_len):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample_2d(u, sim_params, z), z,
                )

            def modifier(idx: int, chunk: np.ndarray) -> None:
                tile, y_start = _tile_view(idx, chunk, nx, ny)
                rows = tile.shape[0]
                y_idx = (
                    sim_y[y_start:y_start + rows]
                    + y_pos
                    + y_len * dy_s * 0.5
                ) / dy_s
                y_floor_raw = np.floor(y_idx).astype(np.int64)
                y_frac = y_idx - y_floor_raw
                inside_y = (y_idx >= 0.0) & (y_idx < float(y_len) - 1.0)
                y_floor = np.clip(y_floor_raw, 0, y_len - 2)

                # Advanced indexing reads only the required z/y tile from a
                # memmapped integer grid.  Complex128 exists only for this tile.
                x0 = x_floor[None, :]
                y0 = y_floor[:, None]
                db_00 = self.db_list[self.grid[rowidx, y0, x0]]
                db_01 = self.db_list[self.grid[rowidx, y0, x0 + 1]]
                db_10 = self.db_list[self.grid[rowidx, y0 + 1, x0]]
                db_11 = self.db_list[self.grid[rowidx, y0 + 1, x0 + 1]]

                db_y0 = db_00 * (1.0 - x_frac[None, :]) + db_01 * x_frac[None, :]
                db_y1 = db_10 * (1.0 - x_frac[None, :]) + db_11 * x_frac[None, :]
                interpolated_db = (
                    db_y0 * (1.0 - y_frac[:, None]) + db_y1 * y_frac[:, None]
                )
                interpolated_db[~(inside_y[:, None] & inside_x[None, :])] = 0.0
                tile *= material_factor(
                    interpolated_db, self.pixel_size_z, sim_params.wl
                )

            u.modify_chunked(sim_params.chunk_size, modifier)
            propagate_2d(
                u, U,
                sim_params.dx, sim_params.get_dy(),
                sim_params.wl, self.pixel_size_z,
                sim_params.chunk_size, cutoff_freq,
                nx, ny,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    rowidx + 1, self.z_start + (rowidx + 1) * self.pixel_size_z
                )

@dataclass
class precise_Sample(OpticalElement):
    """
    A sample to be analyzed, represented as separate material index and density grids.

    Axis 0 of the grids is the z position, axis 1 is the x position. For example, the
    following sample could represent a small section of a Si/Au grating with a Si substrate,
    where 0 represents holes in the grating.

    ```
    material_grid = [ [1, 2, 1, 0, 1, 2, 0],
                      [1, 1, 1, 1, 1, 1, 1] ]
    density_grid = [ [2.33, 19.3, 2.33, 0, 2.33, 19.3, 0],
                     [2.33, 2.33, 2.33, 2.33, 2.33, 2.33, 2.33] ]
    materials = [Si, Au]
    ```

    Note that the indices into the materials list are 1-based because 0 represents an empty space.

    Currently does not support rotating the grid. Assumes rectangular pixels

    Samples are centered on the x-axis, i.e. the middle entry is at x=0
    """

    z_start: float
    pixel_size_x: float  # in metres
    pixel_size_z: float  # in metres
    material_grid: np.ndarray  # 材料索引网格
    density_grid: np.ndarray   # 密度值网格 (g/cm³)
    materials: list[Material]
    x_positions: np.ndarray

    def check_valid(self) -> None:
        shape = self.material_grid.shape
        assert len(shape) == 2
        assert shape[0] > 0
        assert shape[1] > 0
        assert self.material_grid.dtype in (np.uint32, np.int32), f"material_grid dtype must be uint32 or int32, got {self.material_grid.dtype}"
        assert self.density_grid.shape == shape
        assert self.density_grid.dtype == np.float32
        assert self.z_start > 0
        assert self.pixel_size_x > 0
        assert self.pixel_size_z > 0
        assert len(self.materials) >= np.max(self.material_grid)

    def get_thickness(self) -> float:
        return self.pixel_size_z * self.material_grid.shape[0]

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        # 不再预先计算db_list，因为每个像素的密度可能不同
        # 我们将在apply方法中实时计算每个像素的deltabeta
        pass

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        rowlen = self.material_grid.shape[1]
        row_x = (
            np.arange(rowlen) * self.pixel_size_x
            + self.x_positions[stepping_iteration]
            - rowlen * self.pixel_size_x * 0.5
        )

        for rowidx in range(self.material_grid.shape[0]):
            if history is not None:
                z = self.z_start + rowidx * self.pixel_size_z
                history.push(
                    square_and_downsample(u, sim_params, z),
                    z,
                )
        
            # 为当前行的每个像素计算deltabeta
            row_deltabeta = np.zeros(rowlen, dtype=np.complex128)
            for colidx in range(rowlen):
                mat_index = self.material_grid[rowidx, colidx]
                if mat_index == 0:  # 真空
                    row_deltabeta[colidx] = 0.0
                else:
                    material = self.materials[mat_index - 1]
                    density = self.density_grid[rowidx, colidx]
                    # 实时计算deltabeta
                    # print(material.mat, density)
                    db = xray_delta_beta(material.mat, density, energy=convert_energy_wavelength(sim_params.wl))  # 将波长转换为能量(eV)
                    row_deltabeta[colidx] = db[0] + 1j * db[1]

            def modifier(idx: int, chunk: np.ndarray) -> None:
                start = idx - sim_params.N / 2
                x_chunk = np.arange(start, start + len(chunk)) * sim_params.dx
                deltabeta = np.interp(
                    x_chunk, xp=row_x, fp=row_deltabeta, left=0.0, right=0.0
                )
                chunk *= material_factor(deltabeta, self.pixel_size_z, sim_params.wl)

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

@dataclass
class SaveAndExit(OpticalElement):
    """
    This is not a real optical element, instead it's a marker object that tells the simulation
    to stop right there and save the current `u` vector as output. The saved `u` vector can be
    used as the input for a future simulation.
    """

    z_start: float
    x_positions: np.ndarray

    def get_thickness(self) -> float:
        return 0

    def apply(
        self,
        u: Vector,
        U: Vector,
        sim_params: SimParams,
        cutoff_freq: float,
        stepping_iteration: int,
        history: Optional[History],
    ) -> None:
        raise ValueError(
            "Trying to apply a SaveAndExit element, this is not allowed since it's only a marker for the setup"
        )

    def check_valid(self) -> None:
        pass

    def store_deltabetas(self, deltabeta_table: DeltabetaTable) -> None:
        pass


def collect_all_materials(
    optical_elements: list[OpticalElement],
) -> list[Material]:
    """
    Collect a deduplicated list of all materials present in the setup
    """

    materials = []

    def push(mat: Material):
        # insert into `materials` if it's not already present
        if mat not in materials:
            materials.append(mat)

    for el in optical_elements:
        if isinstance(el, Grating):
            if el.mat_a is not None:
                push(el.mat_a)
            if el.mat_b is not None:
                push(el.mat_b)
            if el.mat_substrate is not None:
                push(el.mat_substrate)
        if isinstance(el, EnvGrating):
            if el.mat_a is not None:
                push(el.mat_a)
            if el.mat_b is not None:
                push(el.mat_b)
            if el.mat_substrate is not None:
                push(el.mat_substrate)
        elif isinstance(el, Sample):
            for mat in el.materials:
                push(mat)
        elif isinstance(el, precise_Sample):  
            for mat in el.materials:  
                push(mat)
    return materials


def generate_deltabeta_table(
    materials: list[Material],
    energy: float,
) -> DeltabetaTable:
    """
    Precompute the delta and beta values for the current energy using the nist_lookup
    library and store them in a table.
    """

    table = []

    for m in materials:
        db = xray_delta_beta(m.mat, m.density, energy)
        table.append((m, (db[0] + 1j * db[1])))

    return table
