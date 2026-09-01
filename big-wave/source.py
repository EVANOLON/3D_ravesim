# Copyright (c) 2024, ETH Zurich

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

from history import History, generate_analytic_history, generate_analytic_history_2d, propagate_with_history, propagate_with_history_2d
from propagation import (
    SimParams,
    apply_frequency_cutoff,
    apply_frequency_cutoff_2d,
    grid_density_check,
    grid_density_check_2d,
    propagate_analytically,
    propagate_analytically_2d,
)
from vector import Vector


class Source(Protocol):
    """
    A source of waves located at distance z
    """

    z: float

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        """
        Propagate waves to `z_out`. The output vector `u` is modified in place. `U` is
        not guaranteed to contain the fft of `u` afterwards.
        """
        ...


@dataclass
class PointSource(Source):
    x: float
    z: float
    y: float = 0.0

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        dz = z_out - self.z

        if sim_params.is_2d:
            if sim_params.use_fresnel_scaling:
                if sim_params.fresnel_effective_z <= 0 or sim_params.fresnel_magnification <= 0:
                    raise ValueError("Fresnel detector geometry has not been configured")
                if dz <= 0:
                    raise ValueError("Fresnel source-to-sample distance must be positive")
                if history is not None:
                    raise ValueError(
                        "physical-z source history is not supported with Fresnel scaling"
                    )

                # Fresnel scaling removes the incident spherical quadratic
                # phase.  A centred point source becomes a constant field;
                # an off-axis source leaves the linear phase ramp below.
                # The 1/dz amplitude matches the central amplitude of the
                # original 3D spherical-wave initializer.
                wave_number = 2.0 * np.pi / sim_params.wl

                def initialize_fresnel_plane(idx: int, chunk: np.ndarray) -> None:
                    flat = idx + np.arange(len(chunk), dtype=np.int64)
                    ix = flat % sim_params.nx
                    iy = flat // sim_params.nx
                    x = (ix - sim_params.nx / 2.0) * sim_params.dx
                    y = (iy - sim_params.ny / 2.0) * sim_params.get_dy()
                    phase = -wave_number * (x * self.x + y * self.y) / dz
                    chunk[:] = np.exp(1j * phase) / dz

                u.write_chunked(sim_params.chunk_size, initialize_fresnel_plane)
                u.fft2(U, sim_params.nx, sim_params.ny)
                apply_frequency_cutoff_2d(
                    U, cutoff_freq, sim_params.dx, sim_params.get_dy(),
                    sim_params.nx, sim_params.ny, sim_params.chunk_size,
                )
                U.ifft2(u, sim_params.nx, sim_params.ny)
                return

            grid_density_check_2d(
                dz, self.x, sim_params.nx, sim_params.dx,
                self.y, sim_params.ny, sim_params.get_dy(), sim_params.wl,
            )
            propagate_analytically_2d(u, dz, self.x, self.y, sim_params)
            u.fft2(U, sim_params.nx, sim_params.ny)
            apply_frequency_cutoff_2d(
                U, cutoff_freq, sim_params.dx, sim_params.get_dy(),
                sim_params.nx, sim_params.ny, sim_params.chunk_size,
            )
            U.ifft2(u, sim_params.nx, sim_params.ny)

            if history is not None:
                zs = np.arange(0, z_out, history[1])[1:]
                generate_analytic_history_2d(
                    history[0], zs, self.x, self.y, sim_params, cutoff_freq,
                )
        else:
            grid_density_check(dz, self.x, sim_params.N, sim_params.dx, sim_params.wl)
            propagate_analytically(u, dz, self.x, sim_params)

            # Discussion with Alex 2023-07-21: not sure if we want to apply frequency cutoff
            # after analytical propagation or not. Can still comment this out later.
            u.fft(U)
            apply_frequency_cutoff(U, cutoff_freq, sim_params.dx, sim_params.chunk_size)
            U.ifft(u)

            if history is not None:
                # don't generate a history entry at z=0 to avoid division by zero shennanigans
                zs = np.arange(0, z_out, history[1])[1:]
                generate_analytic_history(history[0], zs, self.x, sim_params, cutoff_freq)


@dataclass
class VectorSource(Source):
    input: Vector
    z: float

    def propagate_to(
        self,
        z_out: float,
        sim_params: SimParams,
        cutoff_freq: float,
        u: Vector,
        U: Vector,
        history: Optional[Tuple[History, float]],
    ) -> None:
        assert type(self.input) == type(u)

        dz = z_out - self.z
        if dz == 0:
            self.input.copy_to(u)
            return

        self.input.fft(U)
        propagate_with_history(
            u,
            U,
            sim_params,
            dz,
            cutoff_freq,
            self.z,
            skip_fft=True,
            history=history,
        )
