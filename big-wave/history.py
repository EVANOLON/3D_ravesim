# Copyright (c) 2024, ETH Zurich

"""Versioned and optionally streamed simulation histories.

The legacy big-wave 2D writer stacked frames on the last axis and therefore
produced ``(y, x, z)``. Format version 2 uses ``(z, y, x)`` so that it agrees
with fast-wave and can be appended one frame at a time.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import h5py  # type: ignore
import numpy as np

from propagation import (
    SimParams,
    cleanup_detector_output,
    propagate,
    propagate_2d,
    square_and_downsample,
    square_and_downsample_2d,
)
from vector import Vector


HISTORY_FORMAT_VERSION = 2
HISTORY_AXIS_ORDER_1D = ("x", "z")
HISTORY_AXIS_ORDER_2D = ("z", "y", "x")


def _cancelled(token: Any | None) -> bool:
    if token is None:
        return False
    if isinstance(token, bool):
        return token
    for name in ("is_cancelled", "is_set", "cancelled"):
        value = getattr(token, name, None)
        if value is not None:
            return bool(value() if callable(value) else value)
    return False


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(value)
    if len(pair) != 2 or any(not isinstance(item, int) or item <= 0 for item in pair):
        raise ValueError(f"{name} must contain two positive integers")
    return int(pair[0]), int(pair[1])


class History:
    """Collect 1D history in memory or stream canonical 2D frames to HDF5.

    ``memory_budget_bytes=0`` forces streaming on the first 2D frame. A
    positive budget starts in memory and spills accumulated frames before
    appending more frames.
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        memory_budget_bytes: int | None = None,
        max_frames: int | None = None,
        downsample: int | tuple[int, int] = (1, 1),
        roi: tuple[int, int, int, int] | None = None,
        progress_cb: Callable[[str, int, int], Any] | None = None,
        cancel_token: Any | None = None,
        resume: bool = False,
    ) -> None:
        if memory_budget_bytes is not None and memory_budget_bytes < 0:
            raise ValueError("history memory_budget_bytes must be non-negative")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("history max_frames must be positive")
        if roi is not None:
            if len(roi) != 4 or any(not isinstance(item, int) for item in roi):
                raise ValueError("history roi must be (y_start, y_stop, x_start, x_stop)")
            if roi[0] < 0 or roi[2] < 0 or roi[1] <= roi[0] or roi[3] <= roi[2]:
                raise ValueError("history roi bounds must be non-negative and increasing")

        self.entries: list[np.ndarray] = []
        self.zs: list[float] = []
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.memory_budget_bytes = memory_budget_bytes
        self.max_frames = max_frames
        self.downsample = _pair(downsample, "history downsample")
        self.roi = roi
        self.progress_cb = progress_cb
        self.cancel_token = cancel_token
        self.frame_ndim: int | None = None
        self.frame_shape: tuple[int, ...] | None = None
        self._h5: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None
        self._z_dataset: h5py.Dataset | None = None

        if resume and self.storage_path is not None and self.storage_path.exists():
            self._open_existing_stream()

    def _open_existing_stream(self) -> None:
        assert self.storage_path is not None
        self._h5 = h5py.File(self.storage_path, "r+")
        if int(self._h5.attrs.get("format_version", 0)) != HISTORY_FORMAT_VERSION:
            self.close()
            raise ValueError("cannot resume an unversioned or legacy history file")
        self._dataset = self._h5["history"]
        self._z_dataset = self._h5["z"]
        self.frame_ndim = len(self._dataset.shape) - 1
        self.frame_shape = tuple(int(item) for item in self._dataset.shape[1:])
        self.zs = [float(item) for item in self._z_dataset[:]]

    @property
    def is_streaming(self) -> bool:
        return self._dataset is not None

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        array = np.asanyarray(frame)
        if array.ndim not in (1, 2):
            raise ValueError(f"history frame must be 1D or 2D, got shape {array.shape}")
        if array.ndim == 2:
            if self.roi is not None:
                y0, y1, x0, x1 = self.roi
                if y1 > array.shape[0] or x1 > array.shape[1]:
                    raise ValueError(f"history roi {self.roi} exceeds frame shape {array.shape}")
                array = array[y0:y1, x0:x1]
            sy, sx = self.downsample
            array = array[::sy, ::sx]
        return array

    def _ensure_stream(self, frame: np.ndarray) -> None:
        if self._dataset is not None:
            return
        if self.storage_path is None:
            raise MemoryError("history exceeded memory budget without storage_path")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = self.storage_path.with_name(self.storage_path.name + ".part")
        if part_path.exists():
            part_path.unlink()
        self._h5 = h5py.File(part_path, "w")
        self._h5.attrs["format_version"] = HISTORY_FORMAT_VERSION
        self._h5.attrs["axis_order"] = "z,y,x"
        self._h5.attrs["complete"] = False
        chunks = (1, *tuple(max(1, min(256, int(item))) for item in frame.shape))
        self._dataset = self._h5.create_dataset(
            "history", shape=(0, *frame.shape), maxshape=(None, *frame.shape),
            chunks=chunks, dtype=frame.dtype,
        )
        self._z_dataset = self._h5.create_dataset(
            "z", shape=(0,), maxshape=(None,), chunks=(256,), dtype=np.float64,
        )
        old_entries, old_zs = self.entries, self.zs.copy()
        self.entries, self.zs = [], []
        for old_frame, old_z in zip(old_entries, old_zs):
            self._append_stream(old_frame, old_z)
        self._h5.flush()
        self._h5.close()
        part_path.replace(self.storage_path)
        self._h5 = h5py.File(self.storage_path, "r+")
        self._dataset = self._h5["history"]
        self._z_dataset = self._h5["z"]

    def _append_stream(self, frame: np.ndarray, z: float) -> None:
        assert self._dataset is not None and self._z_dataset is not None
        index = int(self._dataset.shape[0])
        self._dataset.resize(index + 1, axis=0)
        self._dataset[index] = frame
        self._z_dataset.resize(index + 1, axis=0)
        self._z_dataset[index] = z
        self.zs.append(float(z))
        assert self._h5 is not None
        self._h5.flush()

    def push(self, downscaled: np.ndarray, z: float) -> None:
        """Add a frame, applying ROI/downsampling before memory accounting."""
        if _cancelled(self.cancel_token):
            cleanup_detector_output(downscaled)
            raise InterruptedError("history recording cancelled")
        if self.max_frames is not None and len(self) >= self.max_frames:
            cleanup_detector_output(downscaled)
            raise RuntimeError(f"history maximum frame count {self.max_frames} exceeded")
        try:
            frame = self._prepare_frame(downscaled)
            if self.frame_ndim is None:
                self.frame_ndim = frame.ndim
                self.frame_shape = tuple(frame.shape)
            elif frame.ndim != self.frame_ndim or tuple(frame.shape) != self.frame_shape:
                raise ValueError(f"history frame shape changed from {self.frame_shape} to {frame.shape}")
            should_stream = frame.ndim == 2 and self.storage_path is not None and (
                self.memory_budget_bytes == 0
                or self.is_streaming
                or (
                    self.memory_budget_bytes is not None
                    and sum(item.nbytes for item in self.entries) + frame.nbytes
                    > self.memory_budget_bytes
                )
            )
            if should_stream:
                self._ensure_stream(frame)
                self._append_stream(frame, float(z))
            else:
                self.entries.append(np.array(frame, copy=True))
                self.zs.append(float(z))
        finally:
            cleanup_detector_output(downscaled)
        if self.progress_cb is not None:
            self.progress_cb("history_frame", len(self), self.max_frames or 0)

    def get_history(self) -> np.ndarray:
        """Return history in the established 1D or canonical 2D axis order."""
        if self.is_streaming:
            assert self._dataset is not None
            return self._dataset[:]
        if not self.entries:
            return np.empty((0, 0))
        if self.frame_ndim == 1:
            return np.stack(self.entries, axis=-1)
        return np.stack(self.entries, axis=0)

    def get_z(self) -> list[float]:
        return self.zs

    def metadata(self) -> dict[str, Any]:
        shape = (
            tuple(int(item) for item in self._dataset.shape)
            if self._dataset is not None
            else tuple(int(item) for item in self.get_history().shape)
        )
        return {
            "format_version": HISTORY_FORMAT_VERSION if self.frame_ndim == 2 else 1,
            "axis_order": list(HISTORY_AXIS_ORDER_2D if self.frame_ndim == 2 else HISTORY_AXIS_ORDER_1D),
            "storage": "hdf5" if self.is_streaming else "npy",
            "shape": list(shape),
            "frames": len(self),
            "downsample_yx": list(self.downsample),
            "roi_yxyx": list(self.roi) if self.roi is not None else None,
        }

    def finalize(self) -> None:
        if self._h5 is not None:
            self._h5.attrs["complete"] = True
            self._h5.flush()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
            self._dataset = None
            self._z_dataset = None

    def __len__(self) -> int:
        if self._dataset is not None:
            return int(self._dataset.shape[0])
        assert len(self.entries) == len(self.zs)
        return len(self.entries)


@dataclass
class HistoryData:
    values: np.ndarray
    z: np.ndarray
    x: np.ndarray | None
    y: np.ndarray | None
    metadata: dict[str, Any]


def load_history(path: Path | str) -> HistoryData:
    """Load v2 HDF5/NPY history and transpose legacy 2D NPY."""
    root = Path(path)
    directory = root if root.is_dir() else root.parent
    h5_path = root if root.suffix in (".h5", ".hdf5") else directory / "history.h5"
    x_path, y_path, z_path = directory / "history_x.npy", directory / "history_y.npy", directory / "history_z.npy"
    x = np.load(x_path, allow_pickle=False) if x_path.exists() else None
    y = np.load(y_path, allow_pickle=False) if y_path.exists() else None
    if h5_path.exists():
        with h5py.File(h5_path, "r") as handle:
            version = int(handle.attrs.get("format_version", 0))
            if version != HISTORY_FORMAT_VERSION:
                raise ValueError(f"unsupported history format_version {version}")
            values, z = handle["history"][:], handle["z"][:]
            metadata = {
                "format_version": version,
                "axis_order": str(handle.attrs.get("axis_order", "z,y,x")).split(","),
                "storage": "hdf5",
                "complete": bool(handle.attrs.get("complete", False)),
            }
        return HistoryData(values, z, x, y, metadata)
    npy_path = root if root.suffix == ".npy" else directory / "history.npy"
    values = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    z = np.load(z_path, allow_pickle=False) if z_path.exists() else np.arange(values.shape[-1])
    metadata_path = (
        root.with_name(f"{root.stem}_metadata.yaml")
        if root.is_file() and root.name != "history.npy"
        else directory / "history_metadata.yaml"
    )
    version, axis_order = 0, None
    if metadata_path.exists():
        from ruamel.yaml import YAML
        loaded = YAML(typ="safe").load(metadata_path) or {}
        version, axis_order = int(loaded.get("format_version", 0)), list(loaded.get("axis_order", []))
    if values.ndim == 3:
        if version == HISTORY_FORMAT_VERSION and axis_order == list(HISTORY_AXIS_ORDER_2D):
            canonical = np.asarray(values)
        elif values.shape[-1] == len(z):
            canonical = np.transpose(values, (2, 0, 1))
        else:
            raise ValueError("cannot infer legacy 2D history axis order")
        return HistoryData(
            canonical, np.asarray(z), x, y,
            {"format_version": HISTORY_FORMAT_VERSION, "axis_order": list(HISTORY_AXIS_ORDER_2D),
             "storage": "npy", "migrated_from": "legacy_y_x_z" if version != HISTORY_FORMAT_VERSION else None},
        )
    return HistoryData(
        np.asarray(values), np.asarray(z), x, y,
        {"format_version": version or 1, "axis_order": list(HISTORY_AXIS_ORDER_1D), "storage": "npy"},
    )


def migrate_legacy_history(path: Path | str, output_path: Path | str | None = None) -> Path:
    """Write a legacy 2D history as canonical v2 without overwriting it."""
    root = Path(path)
    directory = root if root.is_dir() else root.parent
    loaded = load_history(root)
    if loaded.values.ndim != 3:
        raise ValueError("only legacy 2D histories require migration")
    target = Path(output_path) if output_path is not None else directory / "history_v2.npy"
    np.save(target, loaded.values)
    from ruamel.yaml import YAML
    metadata_path = target.with_name(f"{target.stem}_metadata.yaml")
    with metadata_path.open("w") as handle:
        YAML().dump(
            {"format_version": HISTORY_FORMAT_VERSION, "axis_order": list(HISTORY_AXIS_ORDER_2D),
             "storage": "npy", "shape": list(loaded.values.shape)}, handle,
        )
    return target


def generate_analytic_history(history: History, zs: np.ndarray, x_source: float, params: SimParams, cutoff_freq: float):
    gradient = cutoff_freq * params.wl
    n = params.detector_size // params.detector_pixel_size_x
    x = (np.arange(n) - n // 2) * params.detector_pixel_size_x - x_source
    for z in zs:
        angle = np.arctan(x / z)
        r2 = x**2 + z**2
        intensity = params.detector_pixel_size_x * params.detector_pixel_size_y * np.cos(angle) / r2
        intensity[np.abs(x) > gradient * z] = 0
        history.push(intensity, z)


def generate_analytic_history_2d(
    history: History, zs: np.ndarray, x_source: float, y_source: float,
    params: SimParams, cutoff_freq: float,
):
    """Generate a 2D point-source history without full X/Y meshgrids."""
    gradient = cutoff_freq * params.wl
    nx_pix = int(params.get_detector_size_x() // params.detector_pixel_size_x)
    ny_pix = int(params.get_detector_size_y() // params.detector_pixel_size_y)
    x = (np.arange(nx_pix) - nx_pix // 2) * params.detector_pixel_size_x - x_source
    y = (np.arange(ny_pix) - ny_pix // 2) * params.detector_pixel_size_y - y_source
    x2 = x * x
    for z in zs:
        intensity = np.empty((ny_pix, nx_pix), dtype=np.float64)
        for row, y_value in enumerate(y):
            r2 = x2 + y_value * y_value + z * z
            intensity[row] = params.detector_pixel_size_x * params.detector_pixel_size_y * z / np.power(r2, 1.5)
            intensity[row, np.abs(x) > gradient * z] = 0
            if abs(y_value) > gradient * z:
                intensity[row] = 0
        history.push(intensity, z)


def _history_z_steps(dz: float, hist_dz: float) -> np.ndarray:
    zs = np.arange(0, dz, hist_dz)
    if len(zs) == 0:
        return np.array([0.0, dz])
    if dz - zs[-1] < hist_dz / 2.0 and len(zs) > 1:
        zs[-1] = dz
    else:
        zs = np.append(zs, [dz])
    return zs


def propagate_with_history(
    u: Vector, U: Vector, sim_params: SimParams, dz: float, cutoff_freq: float,
    current_z: float, skip_fft: bool = False,
    history: Optional[Tuple[History, float]] = None,
):
    if dz == 0:
        return
    if history is None:
        propagate(u, U, sim_params.dx, sim_params.wl, dz, sim_params.chunk_size, cutoff_freq, skip_fft)
        return
    hist_obj, hist_dz = history
    zs = _history_z_steps(dz, hist_dz)
    if not skip_fft:
        u.fft(U)
    U_start = U.copy()
    for i in range(1, len(zs)):
        dz_step = float(zs[i] - zs[0])
        propagate(u, U, sim_params.dx, sim_params.wl, dz_step, sim_params.chunk_size, cutoff_freq, True)
        z = float(zs[i] + current_z)
        hist_obj.push(square_and_downsample(u, sim_params, z), z)
        if i < len(zs) - 1:
            U_start.copy_to(U)
    U_start.drop()


def propagate_with_history_2d(
    u: Vector, U: Vector, sim_params: SimParams, dz: float, cutoff_freq: float,
    current_z: float, skip_fft: bool = False,
    history: Optional[Tuple[History, float]] = None,
):
    if dz == 0:
        return
    nx, ny = sim_params.nx, sim_params.ny
    if history is None:
        propagate_2d(
            u, U, sim_params.dx, sim_params.get_dy(), sim_params.wl, dz,
            sim_params.chunk_size, cutoff_freq, nx, ny, skip_fft,
        )
        return
    hist_obj, hist_dz = history
    zs = _history_z_steps(dz, hist_dz)
    if not skip_fft:
        u.fft2(U, nx, ny)
    U_start = U.copy()
    for i in range(1, len(zs)):
        dz_step = float(zs[i] - zs[0])
        propagate_2d(
            u, U, sim_params.dx, sim_params.get_dy(), sim_params.wl, dz_step,
            sim_params.chunk_size, cutoff_freq, nx, ny, True,
        )
        z = float(zs[i] + current_z)
        hist_obj.push(square_and_downsample_2d(u, sim_params, z), z)
        if i < len(zs) - 1:
            U_start.copy_to(U)
    U_start.drop()
