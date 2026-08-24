# Copyright (c) 2024, ETH Zurich

import functools
import logging
import math
from pathlib import Path
import numpy as np
from typing import Any, Callable, Optional
from ruamel.yaml import comments as yc, YAML

import optical_element
import plasma_sample as plasma_element
import propagation
import source
import vector

logger = logging.getLogger("big-wave")


# The ruamel.yaml type here is useful for preserving the comments in yaml configurations. Other than
# that we use it in the same way as a normal python dict.
DictType = yc.CommentedMap | dict[str, Any]

DEFAULT_MEMORY_BUDGET_GB = 6.0
DEFAULT_FFT2_BACKEND = "scipy_in_memory"
OOC_FFT2_BACKEND = "bfpy_ooc"
FFT2_BACKENDS = {DEFAULT_FFT2_BACKEND, OOC_FFT2_BACKEND}
DEFAULT_DETECTOR_INTEGRATOR = "legacy_fastwave"
DETECTOR_INTEGRATORS = set(propagation.DETECTOR_INTEGRATOR_VERSIONS)
DEFAULT_SAVE_DEBUG_WAVEFIELDS = False
MAX_AUTO_TILE_ROWS_2D = 256
MAX_SCIPY_FFT2_POINTS = 64 * 1024 * 1024
# Sample/Plasma interpolation creates several int64, float64 and complex128
# temporaries per wavefront point.  Use a conservative aggregate instead of
# sizing a tile from the wave dtype alone.
LOCAL_OPERATOR_BYTES_PER_POINT = 256
FFT2_MEMORY_BUDGET_FRACTION = 0.5


def get_float(dct: DictType | list, path: list) -> float:
    pathname = ".".join(map(str, path))
    try:
        val = functools.reduce(lambda d, k: d[k], path, dct)
    except KeyError:
        raise KeyError(f"path '{pathname}' does not exist")

    assert isinstance(val, (float, int)), f"entry '{pathname}' must be a float"
    return float(val)


def get_int(dct: DictType | list, path: list) -> int:
    pathname = ".".join(map(str, path))
    try:
        val = functools.reduce(lambda d, k: d[k], path, dct)
    except KeyError:
        raise KeyError(f"path '{pathname}' does not exist")

    assert isinstance(val, int), f"entry '{pathname}' must be an int"
    return int(val)


def get_bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def parse_sim_params(dct: DictType) -> propagation.SimParams:
    return propagation.SimParams(
        N=get_int(dct, ["N"]),
        dx=get_float(dct, ["dx"]),
        z_detector=get_float(dct, ["z_detector"]),
        detector_size=float(dct.get("detector_size", get_float(dct, ["detector_size_x"]) if "detector_size_x" in dct else 0)),
        detector_pixel_size_x=get_float(dct, ["detector_pixel_size_x"]),
        detector_pixel_size_y=get_float(dct, ["detector_pixel_size_y"]),
        wl=0.0,  # the wavelength might differ for individual sources so we don't parse that here yet
        chunk_size=get_int(dct, ["chunk_size"]),
        # 2D optional fields
        ny=int(dct.get("ny", 1)),
        dy=float(dct.get("dy", 0.0)),
        detector_size_x=float(dct.get("detector_size_x", 0.0)),
        detector_size_y=float(dct.get("detector_size_y", 0.0)),
        nx=int(dct.get("nx", 0)),
        memory_budget_gb=float(dct.get("memory_budget_gb", 0.0)),
        fft2_backend=str(dct.get("fft2_backend", DEFAULT_FFT2_BACKEND)),
        detector_integrator=str(dct.get("detector_integrator", DEFAULT_DETECTOR_INTEGRATOR)),
        use_fresnel_scaling=get_bool_value(
            dct.get("use_fresnel_scaling", False), "use_fresnel_scaling"
        ),
    )


def get_runtime_big_wave(dct: DictType) -> dict:
    """Return the runtime.big_wave settings dict (possibly empty)."""
    runtime = dct.get("runtime", {})
    if not isinstance(runtime, dict):
        return {}
    bw = runtime.get("big_wave", {})
    return bw if isinstance(bw, dict) else {}


def resolve_chunk_size(
    sim_dct: DictType,
    runtime: dict,
    dtype: np.dtype = np.dtype(np.complex128),
) -> int:
    """Resolve a possibly-auto chunk_size into an integer for engine config.

    The generated engine config must never contain the string ``auto``.
    For 2D, chunk_size is always resolved to whole rows (tile_rows * nx).
    """
    raw = runtime.get("chunk_size", sim_dct.get("chunk_size", "auto"))
    budget_raw = runtime.get(
        "memory_budget_gb",
        sim_dct.get("memory_budget_gb", DEFAULT_MEMORY_BUDGET_GB),
    )
    try:
        budget_gb = float(budget_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"memory_budget_gb must be a positive number, got {budget_raw!r}") from exc
    if not math.isfinite(budget_gb) or budget_gb <= 0:
        raise ValueError(f"memory_budget_gb must be a positive finite number, got {budget_raw!r}")

    n = int(sim_dct.get("N", 0))
    ny = int(sim_dct.get("ny", 1))
    nx = int(sim_dct.get("nx", 0))
    if n <= 0:
        raise ValueError(f"N must be a positive integer, got {n}")
    if ny <= 0:
        raise ValueError(f"ny must be a positive integer, got {ny}")
    if ny > 1:
        if nx <= 0:
            if n % ny != 0:
                raise ValueError(f"N ({n}) must be divisible by ny ({ny})")
            nx = n // ny
        if nx * ny != n:
            raise ValueError(f"N ({n}) must equal nx * ny ({nx} * {ny})")

    is_auto = isinstance(raw, str) and raw.strip().lower() == "auto"
    if not is_auto and (not isinstance(raw, int) or isinstance(raw, bool)):
        raise ValueError(f"chunk_size must be a positive integer or 'auto', got {raw!r}")

    if is_auto:
        itemsize = int(np.dtype(dtype).itemsize)
        target_bytes = budget_gb * (1024 ** 3) * 0.25
        if ny > 1:
            bytes_per_point = LOCAL_OPERATOR_BYTES_PER_POINT + 4 * itemsize
            bytes_per_row = max(1, nx * bytes_per_point)
            tile_rows = max(1, int(target_bytes // bytes_per_row))
            tile_rows = min(tile_rows, ny, MAX_AUTO_TILE_ROWS_2D)
            chunk_size = tile_rows * nx
        else:
            # Preserve a generous 1D chunk while keeping it bounded by N.
            bytes_per_point = max(32, 4 * itemsize)
            chunk_size = max(1, int(target_bytes // bytes_per_point))
            chunk_size = min(chunk_size, n)
    else:
        chunk_size = int(raw)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if ny > 1:
            # All 2D chunks operate on complete rows.  Round down so an explicit
            # limit is never exceeded; a sub-row request becomes one row.
            tile_rows = max(1, chunk_size // nx)
            chunk_size = min(tile_rows, ny) * nx
        else:
            chunk_size = min(chunk_size, n)
    return chunk_size


def estimate_chunk_peak_gb(
    chunk_size: int, nx: int, memory_budget_gb: float, itemsize: int
) -> float:
    """Conservative peak working-set estimate (GB) for one 2D chunk plus the OOC FFT budget.

    Accounts for the chunk itself replicated across operator buffers, the
    bounded out-of-core FFT2 budget, and a flat allowance for sample deltabeta
    rows / detector intermediates.
    """
    chunk_buffers = 4  # u chunk, U chunk, operator temporaries
    fft_budget_gb = memory_budget_gb * FFT2_MEMORY_BUDGET_FRACTION
    flat_gb = 0.125
    return chunk_size * itemsize * chunk_buffers / (1024**3) + fft_budget_gb + flat_gb


def check_chunk_memory_safety(sim_params, itemsize: int = 16) -> None:
    """Hard pre-launch memory gate (P0): reject chunk/budget combos whose
    estimated peak working set exceeds ``memory_budget_gb``.

    Fails in seconds instead of being OOM-killed minutes into the run.
    """
    if not sim_params.is_2d or sim_params.memory_budget_gb <= 0:
        return
    nx = int(sim_params.nx)
    peak_gb = estimate_chunk_peak_gb(
        int(sim_params.chunk_size), nx, float(sim_params.memory_budget_gb), int(itemsize)
    )
    if peak_gb > float(sim_params.memory_budget_gb):
        fft_budget_gb = float(sim_params.memory_budget_gb) * FFT2_MEMORY_BUDGET_FRACTION
        flat_gb = 0.125
        max_tile_rows = max(
            1,
            int(
                (float(sim_params.memory_budget_gb) - fft_budget_gb - flat_gb)
                / max(1, nx * int(itemsize) * 4)
            ),
        )
        raise RuntimeError(
            "P0 memory gate: chunk_size="
            f"{sim_params.chunk_size} peak working set ~{peak_gb:.2f} GB exceeds "
            f"memory_budget_gb={sim_params.memory_budget_gb}; "
            f"set chunk_size: auto or tile_rows <= {max_tile_rows}"
        )


def resolve_sim_params(dct: DictType) -> DictType:
    """Return a copy of sim_params with runtime.big_wave values resolved.

    This is the function used by setup_simulation before writing config.yaml.
    It never leaves ``chunk_size: auto`` in the generated engine configuration.
    """
    import copy
    sim = copy.deepcopy(dct.get("sim_params", {}))
    runtime = get_runtime_big_wave(dct)
    dtype_name = str(dct.get("dtype", "c16")).lower()
    if dtype_name not in ("c8", "c16"):
        raise ValueError(f"dtype must be 'c8' or 'c16', got {dtype_name!r}")
    dtype = np.dtype(np.complex64 if dtype_name == "c8" else np.complex128)

    memory_budget = runtime.get(
        "memory_budget_gb",
        sim.get("memory_budget_gb", DEFAULT_MEMORY_BUDGET_GB),
    )
    try:
        memory_budget = float(memory_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory_budget_gb must be a positive number") from exc
    if not math.isfinite(memory_budget) or memory_budget <= 0:
        raise ValueError(f"memory_budget_gb must be positive and finite, got {memory_budget!r}")

    is_2d = int(sim.get("ny", 1)) > 1
    default_fft2_backend = (
        OOC_FFT2_BACKEND
        if is_2d and bool(dct.get("use_disk_vector", False))
        else DEFAULT_FFT2_BACKEND
    )
    fft2_backend = str(runtime.get(
        "fft2_backend", sim.get("fft2_backend", default_fft2_backend)
    ))
    if fft2_backend not in FFT2_BACKENDS:
        raise ValueError(
            f"unsupported fft2_backend={fft2_backend!r}; "
            f"expected one of {sorted(FFT2_BACKENDS)!r}"
        )
    if fft2_backend == OOC_FFT2_BACKEND and not bool(dct.get("use_disk_vector", False)):
        raise ValueError("fft2_backend='bfpy_ooc' requires use_disk_vector=true")
    detector_integrator = str(runtime.get(
        "detector_integrator",
        sim.get("detector_integrator", DEFAULT_DETECTOR_INTEGRATOR),
    ))
    if detector_integrator not in DETECTOR_INTEGRATORS:
        raise ValueError(
            f"unsupported detector_integrator={detector_integrator!r}; "
            f"expected one of {sorted(DETECTOR_INTEGRATORS)!r}"
        )
    save_debug_wavefields = runtime.get(
        "save_debug_wavefields",
        sim.get("save_debug_wavefields", DEFAULT_SAVE_DEBUG_WAVEFIELDS),
    )
    if not isinstance(save_debug_wavefields, bool):
        raise ValueError("save_debug_wavefields must be a boolean")

    sim["memory_budget_gb"] = memory_budget
    sim["fft2_backend"] = fft2_backend
    sim["detector_integrator"] = detector_integrator
    sim["chunk_size"] = resolve_chunk_size(sim, runtime, dtype)
    ny = int(sim.get("ny", 1))
    n = int(sim.get("N", 0))
    if ny > 1 and "nx" not in sim and n > 0:
        sim["nx"] = n // ny
    if "nx" in sim and n > 0 and int(sim["nx"]) * ny != n:
        raise ValueError(f"N ({n}) must equal nx * ny ({sim['nx']} * {ny})")

    # Hard memory gate: reject oversized chunks at setup time (P0).
    if ny > 1 and n > 0:
        nx_resolved = int(sim.get("nx", n // ny))
        peak_gb = estimate_chunk_peak_gb(
            int(sim["chunk_size"]), nx_resolved, memory_budget, dtype.itemsize
        )
        if peak_gb > memory_budget:
            fft_budget_gb = memory_budget * FFT2_MEMORY_BUDGET_FRACTION
            flat_gb = 0.125
            max_tile_rows = max(
                1,
                int(
                    (memory_budget - fft_budget_gb - flat_gb)
                    / max(1, nx_resolved * dtype.itemsize * 4)
                ),
            )
            raise ValueError(
                "P0 memory gate: chunk_size="
                f"{sim['chunk_size']} peak working set ~{peak_gb:.2f} GB exceeds "
                f"memory_budget_gb={memory_budget}; "
                f"set chunk_size: auto or tile_rows <= {max_tile_rows}"
            )
    return sim


def parse_material(input: list) -> optical_element.Material:
    assert isinstance(input[0], str)
    return optical_element.Material(input[0], get_float(input, [1]))

def parse_optional_material(
    dct: DictType, key: str
) -> Optional[optical_element.Material]:
    if not key in dct:
        return None

    input = dct[key]
    if input is None:
        return None
    else:
        return parse_material(input)


def parse_optical_element(
    dct: DictType, config_dir: Path
) -> optical_element.OpticalElement:
    """
    Parse a single OpticalElement from a config dict.

    Parameters
    ----------
    dct : DictType
        Only the part of the config dict corresponding to this element. Should not be called with the whole config dict.
    config_dir : Path
        Directory from where the relative lookup to the grid arrays should happen.
    """

    if dct["type"] == "grating":
        return optical_element.Grating(
            pitch=get_float(dct, ["pitch"]),
            dc=(get_float(dct, ["dc", 0]), get_float(dct, ["dc", 1])),
            z_start=get_float(dct, ["z_start"]),
            thickness=get_float(dct, ["thickness"]),
            nr_steps=get_int(dct, ["nr_steps"]),
            x_positions=np.array(dct["x_positions"]),
            substrate_thickness=get_float(dct, ["substrate_thickness"]),
            mat_a=parse_optional_material(dct, "mat_a"),
            mat_b=parse_optional_material(dct, "mat_b"),
            mat_substrate=parse_optional_material(dct, "mat_substrate"),
        )
    if dct["type"] == "env_grating":
        return optical_element.EnvGrating(
            pitch0=get_float(dct, ["pitch0"]),
            pitch1=get_float(dct, ["pitch1"]),
            dc0=(get_float(dct, ["dc0", 0]), get_float(dct, ["dc0", 1])),
            dc1=(get_float(dct, ["dc1", 0]), get_float(dct, ["dc1", 1])),
            z_start=get_float(dct, ["z_start"]),
            thickness=get_float(dct, ["thickness"]),
            nr_steps=get_int(dct, ["nr_steps"]),
            x_positions=np.array(dct["x_positions"]),
            substrate_thickness=get_float(dct, ["substrate_thickness"]),
            mat_a=parse_optional_material(dct, "mat_a"),
            mat_b=parse_optional_material(dct, "mat_b"),
            mat_substrate=parse_optional_material(dct, "mat_substrate"),
        )
    elif dct["type"] == "sample":
        sample_materials = [parse_material(m) for m in dct["materials"]]
        grid = np.load(
            config_dir / dct["grid_path"], mmap_mode="r", allow_pickle=False
        )
        assert grid.dtype in (np.uint32, np.int32), f"grid dtype must be uint32 or int32, got {grid.dtype}"

        return optical_element.Sample(
            z_start=get_float(dct, ["z_start"]),
            pixel_size_x=get_float(dct, ["pixel_size_x"]),
            pixel_size_z=get_float(dct, ["pixel_size_z"]),
            grid=grid,
            materials=sample_materials,
            x_positions=np.array(dct["x_positions"]),
            pixel_size_y=float(dct.get("pixel_size_y", 0.0)),
            y_positions=np.array(dct.get("y_positions", [0.0])),
        )
    elif dct["type"] == "precise_sample":
        sample_materials = [parse_material(m) for m in dct["materials"]]
        material_grid = np.load(
            config_dir / dct["material_grid_path"], mmap_mode="r", allow_pickle=False
        )
        density_grid = np.load(
            config_dir / dct["density_grid_path"], mmap_mode="r", allow_pickle=False
        )
        assert material_grid.dtype in (np.uint32, np.int32), f"material_grid dtype must be uint32 or int32, got {material_grid.dtype}"
        assert density_grid.dtype == np.float32

        return optical_element.precise_Sample(
            z_start=get_float(dct, ["z_start"]),
            pixel_size_x=get_float(dct, ["pixel_size_x"]),
            pixel_size_z=get_float(dct, ["pixel_size_z"]),
            material_grid=material_grid,
            density_grid=density_grid,
            materials=sample_materials,
            x_positions=np.array(dct["x_positions"]),
        )
    elif dct["type"] == "save_and_exit":
        return optical_element.SaveAndExit(
            z_start=get_float(dct, ["z_start"]), x_positions=np.array([0])
        )
    elif dct["type"] == "plasma_sample":
        ne = np.load(config_dir / dct["ne_grid_path"], mmap_mode="r", allow_pickle=False)
        ni = np.load(config_dir / dct["ni_grid_path"], mmap_mode="r", allow_pickle=False)
        te = np.load(config_dir / dct["te_grid_path"], mmap_mode="r", allow_pickle=False)
        zs = np.load(config_dir / dct["zstar_grid_path"], mmap_mode="r", allow_pickle=False)
        return plasma_element.PlasmaSample(
            z_start=float(dct["z_start"]),
            pixel_size_x=float(dct["pixel_size_x"]),
            pixel_size_z=float(dct["pixel_size_z"]),
            pixel_size_y=float(dct.get("pixel_size_y", 0.0)),
            ne_grid=ne, ni_grid=ni, te_grid=te, zstar_grid=zs,
            Z=int(dct["Z"]),
            x_positions=np.array(dct["x_positions"]),
            y_positions=np.array(dct.get("y_positions", [0.0])),
        )
    else:
        raise ValueError(f'Unknown optical element type: {dct["type"]}.')


def parse_source(
    dct: DictType,
    use_disk_vector: bool,
    scratchfile: Path,
    N: int,
    dtype: np.dtype,
    fft2_backend: str = OOC_FFT2_BACKEND,
    fft2_memory_budget_bytes: int = 256 * 1024 * 1024,
    fft2_progress_cb: Callable[[str, int, int], Any] | None = None,
    fft2_cancel_token: Any | None = None,
) -> source.Source:
    if dct["type"] == "point":
        return source.PointSource(
            x=get_float(dct, ["x"]),
            z=get_float(dct, ["z"]),
            y=float(dct.get("y", 0.0)),
        )
    elif dct["type"] == "vector":
        path = Path(dct["input_path"])
        v: vector.Vector
        if use_disk_vector:
            v = vector.DiskVector(
                file=path,
                scratchfile=scratchfile,
                len=N,
                dtype=dtype,
                fft2_backend=fft2_backend,
                fft2_memory_budget_bytes=fft2_memory_budget_bytes,
                fft2_progress_cb=fft2_progress_cb,
                fft2_cancel_token=fft2_cancel_token,
            )
        else:
            nparr = np.load(path)
            if nparr.dtype != dtype:
                logger.warn(
                    f"Dtype {nparr.dtype} of loaded source vector {path} does not match dtype {dtype}. Converting."
                )
                nparr = nparr.astype(dtype)
            v = vector.NumpyVector(nparr)

        return source.VectorSource(
            input=v,
            z=get_float(dct, ["z"]),
        )
    else:
        raise ValueError(f'Unknown source type: {dct["type"]}.')


def parse_dtype(dct: DictType) -> np.dtype:
    """
    Given the full config dict, parse the dtype. Defaults to c16 if not specified.
    """

    if "dtype" not in dct:
        return np.dtype(np.complex128)
    val = dct["dtype"]

    if not isinstance(val, str):
        raise ValueError(f"Expected dtype to be a string, got {val}")

    if val == "c8":
        return np.dtype(np.complex64)
    elif val == "c16":
        return np.dtype(np.complex128)
    else:
        raise ValueError(f"Unknown dtype: {val}. Has to be either c8 or c16")


def load(path: Path) -> DictType:
    yaml = YAML(typ="rt")
    return yaml.load(path)


def save(path: Path, dct: DictType) -> None:
    with open(path, "w") as f:
        yaml = YAML(typ="rt")
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(dct, f)
