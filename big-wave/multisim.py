# Copyright (c) 2024, ETH Zurich

from datetime import datetime
import logging
import copy
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Callable, Optional, Tuple
import bfpy
import numpy as np
import h5py  # type: ignore

from checkpoint import RunCheckpoint
import config
from history import History
from optical_element import (
    DeltabetaTable,
    OpticalElement,
    SaveAndExit,
    collect_all_materials,
    generate_deltabeta_table,
)
from propagation import (
    DETECTOR_INTEGRATOR_VERSIONS,
    SimParams,
    cleanup_detector_output,
    compute_cutoff_angles,
    convert_energy_wavelength,
    grid_density_check,
)
from util import detector_x_vector, get_sub_dir, setup_logger
from vector import DiskVector, NumpyVector, Vector
import wavesim

logger = logging.getLogger("big-wave")


def _git_commit() -> str:
    """Return the source revision used to create a simulation, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_backend_runtime_safety(
    sim_params: SimParams, use_disk_vector: Optional[bool] = None
) -> None:
    """Reject backend/vector combinations that would bypass the P2 OOC path."""
    if sim_params.is_2d and use_disk_vector is not None:
        if sim_params.fft2_backend == config.OOC_FFT2_BACKEND and not use_disk_vector:
            raise RuntimeError("bfpy_ooc FFT2 requires use_disk_vector=true")
        if sim_params.fft2_backend == config.DEFAULT_FFT2_BACKEND and use_disk_vector:
            raise RuntimeError(
                "DiskVector no longer materializes scipy_in_memory FFT2; "
                "set runtime.big_wave.fft2_backend=bfpy_ooc"
            )
    if (
        sim_params.is_2d
        and sim_params.fft2_backend == config.DEFAULT_FFT2_BACKEND
        and sim_params.N > config.MAX_SCIPY_FFT2_POINTS
    ):
        raise RuntimeError(
            "scipy_in_memory FFT2 materializes the complete 2D "
            f"field ({sim_params.N} points); limit is "
            f"{config.MAX_SCIPY_FFT2_POINTS}. Select bfpy_ooc with a DiskVector."
        )


def setup_sim_dir(save_dir: Path) -> Path:
    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    full = now.strftime("%Y%m%d_%H%M%S%f")

    path = save_dir / year / month / full

    os.makedirs(path)
    return path


def _save_detector_outputs(path: Path, outputs: list[np.ndarray]) -> None:
    """Write phase outputs without materializing an additional 3D array."""
    if not outputs:
        raise ValueError("simulation produced no detector outputs")
    shape = tuple(outputs[0].shape)
    if any(tuple(output.shape) != shape for output in outputs):
        raise ValueError("all detector phase outputs must have the same shape")
    dtype = np.result_type(*(output.dtype for output in outputs))
    part_path = path.with_name(path.name + ".part")
    try:
        target = np.lib.format.open_memmap(
            part_path, mode="w+", dtype=dtype, shape=(len(outputs), *shape)
        )
        target_data_offset = int(target.offset)
        target.flush()
        target_mapping = getattr(target, "_mmap", None)
        if target_mapping is not None:
            target_mapping.close()
        del target
        if (
            len(outputs) == 1
            and isinstance(outputs[0], np.memmap)
            and hasattr(outputs[0], "_rave_detector_temp_path")
            and int(outputs[0].offset) == target_data_offset
        ):
            output = outputs[0]
            output_mapping = getattr(output, "_mmap", None)
            if output_mapping is not None and not output_mapping.closed:
                output_mapping.close()
            source_path = Path(output.filename)
            if source_path.stat().st_dev == part_path.parent.stat().st_dev:
                with part_path.open("rb") as header_file:
                    target_header = header_file.read(target_data_offset)
                with source_path.open("r+b", buffering=0) as source_file:
                    source_file.seek(0)
                    source_file.write(target_header)
                    source_file.flush()
                    os.fsync(source_file.fileno())
                part_path.unlink()
                os.replace(source_path, path)
                return
        output_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        block_bytes = 8 * 1024**2
        with part_path.open("r+b", buffering=0) as target_file:
            for index, output in enumerate(outputs):
                target_file.seek(target_data_offset + index * output_bytes)
                if isinstance(output, np.memmap) and output.dtype == dtype:
                    remaining = output_bytes
                    output_mapping = getattr(output, "_mmap", None)
                    if output_mapping is not None and not output_mapping.closed:
                        output_mapping.close()
                    with Path(output.filename).open("rb", buffering=0) as source_file:
                        source_offset = int(output.offset)
                        if hasattr(os, "sendfile"):
                            while remaining:
                                sent = os.sendfile(
                                    target_file.fileno(),
                                    source_file.fileno(),
                                    source_offset,
                                    remaining,
                                )
                                if sent <= 0:
                                    raise OSError(
                                        "unexpected EOF in temporary detector output"
                                    )
                                source_offset += sent
                                remaining -= sent
                        else:
                            source_file.seek(source_offset)
                            while remaining:
                                data = source_file.read(min(block_bytes, remaining))
                                if not data:
                                    raise OSError(
                                        "unexpected EOF in temporary detector output"
                                    )
                                target_file.write(data)
                                remaining -= len(data)
                else:
                    flat_output = np.asarray(output).reshape(-1)
                    block_items = max(1, block_bytes // np.dtype(dtype).itemsize)
                    for start in range(0, flat_output.size, block_items):
                        block = np.asarray(
                            flat_output[start : start + block_items], dtype=dtype
                        )
                        target_file.write(block.tobytes(order="C"))
            target_file.flush()
            os.fsync(target_file.fileno())
        os.replace(part_path, path)
    finally:
        if part_path.exists():
            part_path.unlink()


def _resolve_history_options(
    options: Optional[dict], sim_params: SimParams, resume_checkpoint: bool
) -> dict:
    """Validate runtime-only History storage, ROI and downsampling controls."""
    raw = copy.deepcopy(options or {})
    unknown = set(raw) - {"storage", "memory_budget_mb", "max_frames", "downsample", "roi"}
    if unknown:
        raise ValueError(f"unknown history options: {sorted(unknown)}")
    storage = str(raw.get("storage", "auto")).lower()
    if storage not in {"auto", "memory", "hdf5"}:
        raise ValueError("history storage must be auto, memory, or hdf5")
    if resume_checkpoint and storage == "memory":
        raise ValueError("checkpoint resume with History requires auto or hdf5 storage")
    memory_budget_mb = float(
        raw.get("memory_budget_mb", min(256.0, max(16.0, sim_params.memory_budget_gb * 102.4)))
    )
    if not np.isfinite(memory_budget_mb) or memory_budget_mb < 0:
        raise ValueError("history memory_budget_mb must be finite and non-negative")
    max_frames = raw.get("max_frames")
    if max_frames is not None and (not isinstance(max_frames, int) or max_frames <= 0):
        raise ValueError("history max_frames must be a positive integer")
    downsample = raw.get("downsample", [1, 1])
    if isinstance(downsample, int):
        downsample = [downsample, downsample]
    if len(downsample) != 2 or any(not isinstance(item, int) or item <= 0 for item in downsample):
        raise ValueError("history downsample must contain two positive integers")
    roi = raw.get("roi")
    if roi is not None:
        if len(roi) != 4 or any(not isinstance(item, int) for item in roi):
            raise ValueError("history roi must be [y_start, y_stop, x_start, x_stop]")
        roi = tuple(roi)
    return {
        "storage": "hdf5" if resume_checkpoint else storage,
        "memory_budget_bytes": int(memory_budget_mb * 1024**2),
        "max_frames": max_frames,
        "downsample": tuple(downsample),
        "roi": roi,
    }


def run_single_simulation(
    sim_dir: Path,
    source_idx: int,
    scratch_dir: Path,
    save_keypoints_path: Optional[Path] = None,
    history_dz: Optional[float] = None,
    fft2_progress_cb: Callable[[str, int, int], Any] | None = None,
    fft2_cancel_token: Any | None = None,
    detector_progress_cb: Callable[[str, int, int], Any] | None = None,
    detector_cancel_token: Any | None = None,
    history_options: Optional[dict] = None,
    history_progress_cb: Callable[[str, int, int], Any] | None = None,
    history_cancel_token: Any | None = None,
    checkpoint_dir: Optional[Path] = None,
    resume_checkpoint: bool = False,
    checkpoint_progress_cb: Callable[[str, int, int], Any] | None = None,
    checkpoint_cancel_token: Any | None = None,
):
    setup_logger()
    logger.info(
        f"Running single simulation for source {source_idx} in simulation {sim_dir}"
    )
    start = time.perf_counter()

    if not isinstance(sim_dir, Path):
        sim_dir = Path(sim_dir)
    if not isinstance(scratch_dir, Path):
        scratch_dir = Path(scratch_dir)
    if save_keypoints_path is not None and not isinstance(save_keypoints_path, Path):
        save_keypoints_path = Path(save_keypoints_path)
    if checkpoint_dir is not None and not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    dct = config.load(sim_dir / "config.yaml")
    computed = config.load(sim_dir / "computed.yaml")
    angles = list(map(float, computed["cutoff_angles"]))

    use_disk_vector = dct["use_disk_vector"]
    save_final_u_vectors = dct["save_final_u_vectors"]
    sim_params_dct = copy.deepcopy(dct["sim_params"])
    if (
        use_disk_vector
        and int(sim_params_dct.get("ny", 1)) > 1
        and "fft2_backend" not in sim_params_dct
    ):
        # Backward compatibility for prepared simulations created before P2.
        sim_params_dct["fft2_backend"] = config.OOC_FFT2_BACKEND
    sim_params = config.parse_sim_params(sim_params_dct)
    check_backend_runtime_safety(sim_params, use_disk_vector)
    dtype = config.parse_dtype(dct)
    config.check_chunk_memory_safety(sim_params, dtype.itemsize)
    elements = [config.parse_optical_element(el, sim_dir) for el in dct["elements"]]

    sub_dir = get_sub_dir(sim_dir, source_idx)
    sub_dct = config.load(sub_dir / "subconfig.yaml")
    if sim_params.use_fresnel_scaling:
        if not elements:
            raise ValueError("Fresnel scaling requires at least one optical element")
        z_source = float(sub_dct["source"].get("z", 0.0))
        sim_params.configure_fresnel_detector(z_source, float(elements[0].z_start))
        logger.info(
            "P3 Fresnel detector geometry: M=%.6g, z_eff=%.6g m",
            sim_params.fresnel_magnification,
            sim_params.fresnel_effective_z,
        )

    vectors_dir = scratch_dir / "vectors"
    scratchfile = vectors_dir / "scratch.npy"
    sim_params.detector_output_dir = str(sub_dir)
    sim_params.detector_progress_cb = detector_progress_cb
    sim_params.detector_cancel_token = detector_cancel_token

    fft2_memory_budget_bytes = max(
        1,
        int(
            sim_params.memory_budget_gb
            * (1024 ** 3)
            * config.FFT2_MEMORY_BUDGET_FRACTION
        ),
    )
    source = config.parse_source(
        sub_dct["source"],
        use_disk_vector,
        scratchfile,
        sim_params.N,
        dtype,
        sim_params.fft2_backend,
        fft2_memory_budget_bytes,
        fft2_progress_cb,
        fft2_cancel_token,
    )

    energy = float(sub_dct["energy"])
    deltabeta_table: DeltabetaTable = [
        (config.parse_material(entry[0]), entry[1][0] + entry[1][1] * 1j)
        for entry in sub_dct["deltabeta_table"]
    ]

    u: Vector
    U: Vector
    if use_disk_vector:
        vectors_dir.mkdir(exist_ok=resume_checkpoint)
        if dtype == np.complex128:
            bfpy.generate_header_c16(vectors_dir / "u.npy", sim_params.N)
            bfpy.generate_header_c16(vectors_dir / "spectrum.npy", sim_params.N)
        else:
            bfpy.generate_header_c8(vectors_dir / "u.npy", sim_params.N)
            bfpy.generate_header_c8(vectors_dir / "spectrum.npy", sim_params.N)

        u = DiskVector(
            vectors_dir / "u.npy",
            scratchfile,
            sim_params.N,
            dtype,
            fft2_backend=sim_params.fft2_backend,
            fft2_memory_budget_bytes=fft2_memory_budget_bytes,
            fft2_progress_cb=fft2_progress_cb,
            fft2_cancel_token=fft2_cancel_token,
        )
        U = DiskVector(
            vectors_dir / "spectrum.npy",
            scratchfile,
            sim_params.N,
            dtype,
            fft2_backend=sim_params.fft2_backend,
            fft2_memory_budget_bytes=fft2_memory_budget_bytes,
            fft2_progress_cb=fft2_progress_cb,
            fft2_cancel_token=fft2_cancel_token,
        )

    else:
        u = NumpyVector(np.zeros(sim_params.N, dtype=dtype))
        U = NumpyVector(np.zeros(sim_params.N, dtype=dtype))

    sim_params.wl = convert_energy_wavelength(energy)

    history: Optional[Tuple[History, float]] = None
    if history_dz is not None:
        configured_history = config.get_runtime_big_wave(dct).get("history", {})
        merged_history = copy.deepcopy(configured_history)
        if history_options is not None:
            merged_history.update(history_options)
        resolved_history = _resolve_history_options(
            merged_history, sim_params, resume_checkpoint
        )
        storage = resolved_history.pop("storage")
        storage_path = sub_dir / "history.h5" if sim_params.is_2d and storage != "memory" else None
        memory_budget_bytes = resolved_history.pop("memory_budget_bytes")
        if storage == "hdf5":
            memory_budget_bytes = 0
        history = (
            History(
                storage_path=storage_path,
                memory_budget_bytes=memory_budget_bytes,
                progress_cb=history_progress_cb,
                cancel_token=history_cancel_token,
                resume=resume_checkpoint,
                **resolved_history,
            ),
            history_dz,
        )

    checkpoint: Optional[RunCheckpoint] = None
    if checkpoint_dir is not None:
        checkpoint = RunCheckpoint(
            checkpoint_dir / f"{source_idx:08d}",
            {
                "N": sim_params.N,
                "nx": sim_params.nx,
                "ny": sim_params.ny,
                "dtype": dtype.str,
                "wavelength": sim_params.wl,
                "z_detector": sim_params.z_detector,
                "source_index": source_idx,
            },
            progress_cb=checkpoint_progress_cb,
            cancel_token=checkpoint_cancel_token,
        )

    detector_outputs = wavesim.run_simulation(
        params=sim_params,
        source=source,
        elements=elements,
        cutoff_angles=angles,
        u=u,
        U=U,
        deltabeta_table=deltabeta_table,
        sub_dir=sub_dir,
        vectors_dir=vectors_dir,
        save_final_u_vectors=save_final_u_vectors,
        history=history,
        save_keypoints_path=save_keypoints_path,
        checkpoint=checkpoint,
        resume_checkpoint=resume_checkpoint,
    )

    if history is not None:
        hist = history[0]
        hist.finalize()
        if not hist.is_streaming:
            np.save(sub_dir / "history.npy", hist.get_history())
        np.save(sub_dir / "history_z.npy", hist.get_z())
        if sim_params.is_2d:
            out_x = int(sim_params.get_detector_size_x() // sim_params.detector_pixel_size_x)
            out_y = int(sim_params.get_detector_size_y() // sim_params.detector_pixel_size_y)
            x = (np.arange(out_x) - out_x // 2) * sim_params.detector_pixel_size_x
            y = (np.arange(out_y) - out_y // 2) * sim_params.detector_pixel_size_y
            if hist.roi is not None:
                y0, y1, x0, x1 = hist.roi
                x, y = x[x0:x1], y[y0:y1]
            sy, sx = hist.downsample
            x, y = x[::sx], y[::sy]
            np.save(sub_dir / "history_x.npy", x)
            np.save(sub_dir / "history_y.npy", y)
        else:
            np.save(
                sub_dir / "history_x.npy",
                detector_x_vector(
                    sim_params.detector_size, sim_params.detector_pixel_size_x
                ),
            )
        metadata = hist.metadata()
        metadata["coordinate_definition"] = "(index - floor(count/2)) * detector_pixel_size"
        config.save(sub_dir / "history_metadata.yaml", metadata)
        hist.close()

    try:
        _save_detector_outputs(sub_dir / "detected.npy", detector_outputs)
        if sim_params.is_2d:
            detector_metadata = copy.deepcopy(sim_params.detector_last_metadata)
            detector_metadata["phase_steps"] = len(detector_outputs)
            config.save(sub_dir / "detector_metadata.yaml", detector_metadata)
    finally:
        for detector_output in detector_outputs:
            cleanup_detector_output(detector_output)

    if checkpoint is not None:
        checkpoint.mark_complete()

    shutil.rmtree(vectors_dir, ignore_errors=True)

    end = time.perf_counter()
    logger.info(
        f"Finished running simulation for source {source_idx} in {end - start} seconds"
    )


def deltabeta_table_to_tuples(
    table: DeltabetaTable,
) -> list[tuple[tuple[str, float], tuple[float, float]]]:
    """
    We can't save the Material and complex types directly in yaml, so we first simplify them to tuples.
    """

    return [
        ((entry[0].mat, entry[0].density), (float(entry[1].real), float(entry[1].imag)))
        for entry in table
    ]


def reduce_simulation_setup_for_save_and_exit(
    dct: config.DictType, max_x_list: list[float]
) -> float:
    """
    If there is a save_and_exit element present, we remove all the further elements after it
    and move the z_detector to the z coordinate of the save_and_exit element.

    Returns the maximal absolute coordinate at which light will occur at the (potentially updated) z_detector.
    """

    reduced_elements: list[config.DictType] = []
    max_x = max_x_list[-1]
    for i, el in enumerate(dct["elements"]):
        if el["type"] == "save_and_exit":
            max_x = max_x_list[i]
            dct["sim_params"]["z_detector"] = el["z_start"]
            if dct["save_final_u_vectors"]:
                logger.warn(
                    "save_final_u_vectors was activated on a config that contains a save_and_exit element. Note that the vectors will be saved, but at the save_and_exit position instead of the originally specified z_detector."
                )
            dct["save_final_u_vectors"] = True
            dct["elements"] = reduced_elements
            break
        else:
            reduced_elements.append(el)

    return max_x


def check_simulation_inputs(
    params: SimParams,
    z_source: float,
    elements: list[OpticalElement],
    cutoff_angles: list[float],
):
    """Check if the inputs to the simulation are valid"""

    assert len(cutoff_angles) == len(elements) + 1

    # Accept small z-overlaps to prevent problems in cases where the overlap is just
    # due to float-decimal conversions.
    z_tolerance = 1e-8

    if len(elements) > 0:
        assert z_source <= elements[0].z_start, "The source must not be at a greater z-coordinate than the first optical element"

    nr_phase_steps: Optional[int] = None
    current_z: float = z_source
    for i, el in enumerate(elements):
        steps = len(el.x_positions)
        if steps > 1:
            assert (
                nr_phase_steps is None or nr_phase_steps == steps
            ), "If multiple elements have phase stepping enabled, then all elements must have the same number of steps."
            nr_phase_steps = steps

        el.check_valid()
        assert current_z <= el.z_start + z_tolerance, "Overlapping optical elements are not supported"
        current_z = el.z_start + el.get_thickness()

        if isinstance(el, SaveAndExit):
            if i == len(elements) - 1:
                assert (
                    params.z_detector == el.z_start
                ), "A save_and_exit element at the end must have the same z coordinate as the detector"
            else:
                assert (
                    elements[i + 1].z_start == el.z_start
                ), "A save_and_exit element must have the same z coordinate as the next optical element"

    assert current_z <= params.z_detector + z_tolerance, "Detector must be after the last optical element"

    for i in range(len(cutoff_angles) - 1):
        assert cutoff_angles[i] <= cutoff_angles[i + 1], "cutoff angles must be increasing"

    for a in cutoff_angles:
        assert 0 <= a <= np.pi / 2, "cutoff angles must be between 0 and pi/2"


def load_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        return np.array(f["energy"]), np.array(f["pdf"])


def generate_energies_from_spectrum(
    spectrum_energy: np.ndarray,
    spectrum_intensities: np.ndarray,
    rs: np.random.RandomState,
    energy_range: Tuple[float, float],
    nr_points: int,
) -> np.ndarray:
    """
    Use the given spectrum, cropped to energy_range, to generate a set of energies

    This assumes that the spectrum is given in equidistant energies and fills in the spaces between two energies as a uniform distribution
    """

    mask = np.logical_and(
        spectrum_energy >= energy_range[0], spectrum_energy < energy_range[1]
    )
    se = spectrum_energy[mask]
    si = spectrum_intensities[mask]

    if len(se) < 2:
        raise ValueError(
            "Not enough energy points of the spectrum lie within the given range"
        )

    bases = rs.choice(se, size=nr_points, replace=True, p=si / np.sum(si))

    spacing = se[1] - se[0]
    offsets = rs.rand(nr_points) * spacing  # type: ignore [attr-defined]

    return bases + offsets


def setup_simulation(dct: config.DictType, config_dir: Path, save_dir: Path) -> Path:
    """
    Create a directory for this simulation and set up all the files within it that
    are necessary so that the simulations for the individual sources can run afterwards.

    For an explanation of the directory structure see the `big-wave` Readme.

    Parameters
    ----------
    dct : dict
        The simulation configuration
    config_dir : Path
        Directory from where the relative lookup to the grid arrays and spectrum should happen.
        The grid array files will be copied over to the generated simulation directory.
    save_dir : Path
        Directory where simulation directories should be created.

    Returns the path to the created simulation directory.
    """

    setup_logger()
    logger.info("Setting up simulation")

    dct = copy.deepcopy(dct)

    # Keep the user-facing request separate from the numeric configuration that
    # is handed to the engine.  This makes auto-resolution auditable without
    # leaving strings such as ``auto`` in config.yaml.
    requested_runtime = copy.deepcopy(config.get_runtime_big_wave(dct))
    requested_sim_params = copy.deepcopy(dct.get("sim_params", {}))
    default_requested_fft2_backend = (
        config.OOC_FFT2_BACKEND
        if bool(dct.get("use_disk_vector", False))
        and int(requested_sim_params.get("ny", 1)) > 1
        else config.DEFAULT_FFT2_BACKEND
    )
    requested_big_wave = {
        "memory_budget_gb": requested_runtime.get(
            "memory_budget_gb",
            requested_sim_params.get(
                "memory_budget_gb", config.DEFAULT_MEMORY_BUDGET_GB
            ),
        ),
        "chunk_size": requested_runtime.get(
            "chunk_size", requested_sim_params.get("chunk_size", "auto")
        ),
        "fft2_backend": requested_runtime.get(
            "fft2_backend",
            requested_sim_params.get(
                "fft2_backend", default_requested_fft2_backend
            ),
        ),
        "detector_integrator": requested_runtime.get(
            "detector_integrator",
            requested_sim_params.get(
                "detector_integrator", config.DEFAULT_DETECTOR_INTEGRATOR
            ),
        ),
        "save_debug_wavefields": requested_runtime.get(
            "save_debug_wavefields",
            requested_sim_params.get(
                "save_debug_wavefields", config.DEFAULT_SAVE_DEBUG_WAVEFIELDS
            ),
        ),
    }

    if not isinstance(config_dir, Path):
        config_dir = Path(config_dir)
    if not isinstance(save_dir, Path):
        save_dir = Path(save_dir)

    # Resolve runtime.big_wave (memory_budget_gb, chunk_size: auto, etc.) before
    # any engine-facing parsing.  The generated config must only contain numeric
    # chunk_size and explicit nx for 2D.
    dct["sim_params"] = config.resolve_sim_params(dct)
    sim_params = config.parse_sim_params(dct["sim_params"])

    # --- 新增：处理 2D 配置时的 x 方向点数 ---
    Nx = sim_params.nx

    if sim_params.is_2d:
        if sim_params.nx * sim_params.ny != sim_params.N:
            raise ValueError(
                f"N ({sim_params.N}) must equal nx * ny "
                f"({sim_params.nx} * {sim_params.ny} = {sim_params.nx * sim_params.ny})"
            )
        if not ((sim_params.nx & (sim_params.nx - 1)) == 0 and sim_params.nx > 0):
            raise ValueError(f"R1 requires nx to be a power of two, got {sim_params.nx}")
        if not ((sim_params.ny & (sim_params.ny - 1)) == 0 and sim_params.ny > 0):
            raise ValueError(f"R1 requires ny to be a power of two, got {sim_params.ny}")
        # Avoid accidentally loading the whole 2D field for large configurations.
        # 64M points is a conservative cutoff: below that, a whole c8 field is
        # only ~512 MiB and auto-derived chunk_size may legitimately equal N.
        if sim_params.N > config.MAX_SCIPY_FFT2_POINTS and sim_params.chunk_size >= sim_params.N:
            raise ValueError(
                f"2D large config must not use chunk_size >= N ({sim_params.chunk_size} >= {sim_params.N})"
            )

# -----------------------------------------
    elements = [config.parse_optical_element(el, config_dir) for el in dct["elements"]]

    source_points: list[config.DictType]
    sub_dcts: list[config.DictType] = []
    energy_range: Tuple[float, float]
    max_x: float
    z_source: float

    # Check if Fresnel scaling is enabled (skip FOV assertions + Nyquist checks)
    use_fresnel = sim_params.use_fresnel_scaling
    if use_fresnel and not elements:
        raise ValueError("Fresnel scaling requires at least one optical element")

    if not use_fresnel:
        assert (
            sim_params.N * sim_params.dx >= sim_params.detector_size
        ), f"Detector is bigger than simulation space ({sim_params.detector_size} > {sim_params.N * sim_params.dx})."

        if sim_params.is_2d:
            assert sim_params.nx * sim_params.dx >= sim_params.get_detector_size_x(), \
                f"2D detector x-size exceeds simulation x-extent"
            assert sim_params.ny * sim_params.get_dy() >= sim_params.get_detector_size_y(), \
                f"2D detector y-size exceeds simulation y-extent"
    else:
        logger.info("Fresnel scaling enabled, simulation FOV at sample plane does not need to "
                     f"cover physical detector ({sim_params.N * sim_params.dx*1e3:.2f}mm vs {sim_params.detector_size*1e3:.1f}mm)")

    materials = collect_all_materials(elements)

    multisource = dct["multisource"]
    if multisource["type"] == "points":
        nr_source_points = int(multisource["nr_source_points"])
        energy_range = (
            float(multisource["energy_range"][0]),
            float(multisource["energy_range"][1]),
        )
        x_range = (
            float(multisource["x_range"][0]),
            float(multisource["x_range"][1]),
        )
        y_values = multisource.get("y_range", [0.0, 0.0])
        if not isinstance(y_values, (list, tuple)) or len(y_values) != 2:
            raise ValueError("multisource.y_range must contain exactly two values")
        y_range = (float(y_values[0]), float(y_values[1]))
        if not all(np.isfinite(value) for value in y_range) or y_range[0] > y_range[1]:
            raise ValueError("multisource.y_range must be finite and increasing")
        if not sim_params.is_2d and y_range != (0.0, 0.0):
            raise ValueError("multisource.y_range is only supported for 2D simulations")
        z_source = float(multisource["z"])
        seed = int(multisource["seed"])
        if seed == -1:
            seed = random.randint(0, 1000000)

        first_z = sim_params.z_detector
        if len(elements) > 0:
            first_z = elements[0].z_start

        # Use the highest possible energy for the grid density check since that one
        # has the shortest wavelength and is thus more likely to run into Nyquist
        # issues.
        wl_e_max = convert_energy_wavelength(energy_range[1])
        if not use_fresnel:
            grid_density_check(
                first_z - z_source, x_range[0], Nx, sim_params.dx, wl_e_max
            )
            grid_density_check(
                first_z - z_source, x_range[1], Nx, sim_params.dx, wl_e_max
            )
        else:
            logger.info("Fresnel scaling enabled, skipping source→sample Nyquist check")

        if sim_params.is_2d and not use_fresnel:
            from propagation import grid_density_check_2d
            for x_bound in x_range:
                for y_bound in y_range:
                    grid_density_check_2d(
                        first_z - z_source, x_bound, sim_params.nx, sim_params.dx,
                        y_bound, sim_params.ny, sim_params.get_dy(), wl_e_max,
                    )

        rs = np.random.RandomState(seed)
        source_rs = np.random.RandomState(seed)
        energies: np.ndarray
        if "spectrum" in multisource:
            spectrum_energy, spectrum_intensities = load_spectrum(
                config_dir / multisource["spectrum"]
            )
            energies = generate_energies_from_spectrum(
                spectrum_energy,
                spectrum_intensities,
                rs,
                energy_range,
                nr_source_points,
            )
        else:
            energies = (
                rs.rand(nr_source_points) * (energy_range[1] - energy_range[0])  # type: ignore [attr-defined]
                + energy_range[0]
            )
        sources = (
            source_rs.normal(loc=x_range[0]/2+x_range[1]/2,scale=(x_range[1]-x_range[0])/2,size=nr_source_points)  # type: ignore [attr-defined]
        )
        y_sources = source_rs.normal(
            loc=(y_range[0] + y_range[1]) / 2,
            scale=(y_range[1] - y_range[0]) / 2,
            size=nr_source_points,
        )
        del rs

        sub_dcts = [
            {
                "source": {
                    "type": "point", "x": float(sources[i]),
                    "y": float(y_sources[i]), "z": z_source,
                },
                "energy": float(energies[i]),
                "deltabeta_table": deltabeta_table_to_tuples(
                    generate_deltabeta_table(materials, energies[i])
                ),
            }
            for i in range(nr_source_points)
        ]

        source_points = [
            {
                "x": float(sources[i]),
                "y": float(y_sources[i]),
                "z": z_source,
                "energy": float(energies[i]),
            }
            for i in range(nr_source_points)
        ]

        max_x = max(abs(x_range[0]), abs(x_range[1]))

    elif multisource["type"] == "vectors":
        base_sim_dir = Path(multisource["base_sim_dir"])
        input_u_index = int(multisource["input_u_index"])

        base_dct = config.load(base_sim_dir / "config.yaml")
        base_sim_params = config.parse_sim_params(base_dct["sim_params"])
        z_source = base_sim_params.z_detector

        assert (
            base_sim_params.N == sim_params.N and base_sim_params.dx == sim_params.dx
        ), "simulation N and dx must match those of the base simulation in {base_sim_dir}"

        assert base_dct[
            "save_final_u_vectors"
        ], "The base simulation must have save_final_u_vectors set to true or have a save_and_exit element"

        base_computed = config.load(base_sim_dir / "computed.yaml")
        energy_range = (
            float(base_computed["energy_range"][0]),
            float(base_computed["energy_range"][1]),
        )
        max_x = float(base_computed["max_x"])

        # get all subdirectories whose name consists of eight digits.
        # This assumes that the base simulation has already been configured, but doesn't
        # require that it has already been executed.
        nr_source_points = len(
            list(base_sim_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/"))
        )

        for i in range(nr_source_points):
            base_sub_dir = get_sub_dir(base_sim_dir, i)

            upath = base_sub_dir / f"u_{input_u_index:04}.npy"
            if not upath.is_file():
                logger.warning(
                    f"Cannot find u vector {upath} for source {i:08}. This probably means that the base simulation did not run yet "
                    + "for that source point. Make sure that the u vectors of the base simulation are available by the time this "
                    + "simulation is executed."
                )

            base_sub_dct = config.load(base_sub_dir / "subconfig.yaml")
            energy = float(base_sub_dct["energy"])

            sub_dcts.append(
                {
                    "source": {
                        "type": "vector",
                        "input_path": str(upath),
                        "z": z_source,
                    },
                    "energy": energy,
                    "deltabeta_table": deltabeta_table_to_tuples(
                        generate_deltabeta_table(materials, energy)
                    ),
                }
            )

        source_points = base_computed["source_points"]

    else:
        raise ValueError(f"Unknown multisource type {multisource['type']}.")

    z_distances = [el.z_start for el in elements]
    element_heights = [el.get_thickness() for el in elements]
    if use_fresnel:
        # Fresnel scaling: use a simple geometric cutoff based on the
        # sample->detector geometry (not the source->detector full angle).
        # The angular bandwidth at the sample plane is much narrower.
        sample_half_size = max(abs(x_range[0]), abs(x_range[1]))
        cutoff_angle = float(np.arctan(
            (sim_params.detector_size / 2 + sample_half_size) / sim_params.z_detector
        ))
        angles = [cutoff_angle] * (len(elements) + 1)
        max_x_list = [sim_params.detector_size / 2] * (len(elements) + 1)
        logger.info(f"Fresnel mode: simple cutoff angle={cutoff_angle:.4f} rad")
    else:
        angles, max_x_list = compute_cutoff_angles(
            detector_size=sim_params.detector_size,
            dx=sim_params.dx,
            energy_range=energy_range,
            z_source=z_source,
            z_distances=z_distances,
            element_heights=element_heights,
            z_detector=sim_params.z_detector,
            max_x=max_x,
        )

    if sim_params.is_2d:
        # R1: the actual 2D circular cutoff uses kx^2 + ky^2 <= 2*f^2 when a
        # single scalar angle is used for both axes.  The maximum component
        # frequency is therefore sqrt(2)*f, which must stay below Nyquist.
        wl_cut = convert_energy_wavelength(energy_range[1])
        for a in angles:
            f = np.sin(a) / wl_cut
            if np.sqrt(2.0) * f > 0.5 / sim_params.dx:
                raise ValueError(
                    f"2D x-Nyquist cutoff exceeded: angle={a:.6f} rad, "
                    f"max_freq={np.sqrt(2.0) * f:.6e}, nyquist={0.5 / sim_params.dx:.6e}"
                )
            if np.sqrt(2.0) * f > 0.5 / sim_params.get_dy():
                raise ValueError(
                    f"2D y-Nyquist cutoff exceeded: angle={a:.6f} rad, "
                    f"max_freq={np.sqrt(2.0) * f:.6e}, nyquist={0.5 / sim_params.get_dy():.6e}"
                )

    # Maximal absolute x coordinate at z_detector where rays should appear according to the cutoff angles
    max_x = max_x_list[-1]
    if not use_fresnel:
        # size of the simulation such that no rays should reflect at all when using the computed cutoff angles.
        required_Nx = int(np.ceil(max_x / sim_params.dx) * 2)
        #3d adjustment start
        assert (
            Nx >= required_Nx
        ), f"Reflections at the boundary might occur: Nx={Nx}, required_Nx={required_Nx}, max_x={max_x}, dx={sim_params.dx}"
        #3d adjustment end


    # assert (
    #     sim_params.N >= required_N
    # ), f"Reflections at the boundary might occur: N={sim_params.N}, required_N={required_N}"

    grid_files = []
    for el in dct["elements"]:
        if el["type"] == "sample":
            grid_files.append(config_dir / el["grid_path"])
            el["grid_path"] = os.path.basename(el["grid_path"])
        elif el["type"] == "precise_sample":
            grid_files.append(config_dir / el["material_grid_path"])
            grid_files.append(config_dir / el["density_grid_path"])
            el["material_grid_path"] = os.path.basename(el["material_grid_path"])
            el["density_grid_path"] = os.path.basename(el["density_grid_path"])
        elif el["type"] == "plasma_sample":
            for key in ["ne_grid_path", "ni_grid_path", "te_grid_path", "zstar_grid_path"]:
                grid_files.append(config_dir / el[key])
                el[key] = os.path.basename(el[key])

    max_x = reduce_simulation_setup_for_save_and_exit(dct, max_x_list)

    check_simulation_inputs(sim_params, z_source, elements, angles)

    sim_dir = setup_sim_dir(save_dir)
    for gf in grid_files:
        shutil.copyfile(gf, sim_dir / os.path.basename(gf))

    config.save(sim_dir / "config.yaml", dct)

    config.save(
        sim_dir / "computed.yaml",
        {
            "cutoff_angles": angles,
            "max_x": max_x,
            "energy_range": energy_range,
            "source_points": source_points,
            "runtime": {
                "big_wave": {
                    "requested": requested_big_wave,
                    "resolved": {
                        "memory_budget_gb": sim_params.memory_budget_gb,
                        "chunk_size": sim_params.chunk_size,
                        "tile_rows": (
                            sim_params.chunk_size // sim_params.nx
                            if sim_params.is_2d else None
                        ),
                        "fft2_backend": sim_params.fft2_backend,
                        "detector_integrator": sim_params.detector_integrator,
                        "save_debug_wavefields": requested_big_wave["save_debug_wavefields"],
                    },
                    # Retain the flat fields for readers created before P0.
                    "memory_budget_gb": sim_params.memory_budget_gb,
                    "chunk_size": sim_params.chunk_size,
                    "tile_rows": (
                        sim_params.chunk_size // sim_params.nx
                        if sim_params.is_2d else None
                    ),
                    "fft2_backend": sim_params.fft2_backend,
                    "detector_integrator": sim_params.detector_integrator,
                    "save_debug_wavefields": requested_big_wave["save_debug_wavefields"],
                }
            },
            "algorithms": {
                "frequency_cutoff_2d": {
                    "version": "circular_scalar_v1",
                    "criterion": "kx^2 + ky^2 <= 2 * f^2",
                    "axis_max_frequency_factor": float(np.sqrt(2.0)),
                },
                "local_operators_2d": {
                    "version": "row_tile_v1",
                    "tile_rows": sim_params.chunk_size // sim_params.nx,
                    "coordinate_storage": "axis_only",
                },
                "sample_grid_reader": {
                    "version": "npy_memmap_tile_v1",
                    "mmap_mode": "r",
                },
                "plasma_deltabeta": {
                    "version": "vectorized_tile_v1",
                    "chantler_cache_key": "element_and_energy",
                },
                "fft2": {
                    "version": (
                        "bfpy_ooc_transactional_v1"
                        if sim_params.fft2_backend == config.OOC_FFT2_BACKEND
                        else "scipy_in_memory_v1"
                    ),
                    "backend": sim_params.fft2_backend,
                    "memory_budget_fraction": config.FFT2_MEMORY_BUDGET_FRACTION,
                    "passes": [
                        "row_fft",
                        "transpose_to_scratch",
                        "column_fft",
                        "transpose_to_output",
                    ],
                    "restart_manifest": (
                        sim_params.fft2_backend == config.OOC_FFT2_BACKEND
                    ),
                },
                "detector_2d": {
                    "integrator": sim_params.detector_integrator,
                    "version": DETECTOR_INTEGRATOR_VERSIONS[
                        sim_params.detector_integrator
                    ],
                    "streaming": True,
                    "row_prefix_only": True,
                    "geometry_application": "fused_row_accumulation",
                    "fresnel_effective_geometry": sim_params.use_fresnel_scaling,
                    "per_source_metadata": "detector_metadata.yaml",
                },
                "history_2d": {
                    "version": "history_v2",
                    "axis_order": ["z", "y", "x"],
                    "streaming_format": "hdf5",
                    "legacy_reader": "y_x_z_to_z_y_x",
                },
                "run_checkpoint": {
                    "version": "transactional_manifest_v1",
                    "vectors": ["u", "U"],
                    "granularity": ["element", "sample_slice", "plasma_slice"],
                    "phase_snapshot_separate": True,
                },
            } if sim_params.is_2d else {},
            "provenance": {
                "algorithm_version": "big-wave-2d-p4-v1",
                "git_commit": _git_commit(),
            },
        },
    )

    for i in range(len(sub_dcts)):
        sub_dir = get_sub_dir(sim_dir, i)
        os.makedirs(sub_dir)
        config.save(sub_dir / "subconfig.yaml", sub_dcts[i])

    logger.info(f"Finished setting up simulation in {sim_dir}")
    return sim_dir
