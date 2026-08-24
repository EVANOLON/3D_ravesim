# Copyright (c) 2024, ETH Zurich

"""Transactional run checkpoints, separate from phase-stepping snapshots."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable
import uuid

import numpy as np

from vector import DiskVector, NumpyVector, Vector


CHECKPOINT_FORMAT_VERSION = 1


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_vector(vector: Vector, path: Path) -> None:
    if isinstance(vector, NumpyVector):
        with path.open("wb") as handle:
            np.save(handle, vector.vec, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    elif isinstance(vector, DiskVector):
        shutil.copyfile(vector.file, path)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    else:
        raise TypeError(f"unsupported checkpoint vector type {type(vector).__name__}")


def _restore_vector(path: Path, vector: Vector, expected_length: int, expected_dtype: np.dtype) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (expected_length,):
        raise ValueError(f"checkpoint vector shape {array.shape} != {(expected_length,)}")
    if np.dtype(array.dtype) != np.dtype(expected_dtype):
        raise ValueError(f"checkpoint vector dtype {array.dtype} != {expected_dtype}")
    del array
    if isinstance(vector, NumpyVector):
        vector.vec = np.load(path, allow_pickle=False)
    elif isinstance(vector, DiskVector):
        shutil.copyfile(path, vector.file)
        vector.len = expected_length
        vector.typ = np.dtype(expected_dtype)
    else:
        raise TypeError(f"unsupported checkpoint vector type {type(vector).__name__}")


@dataclass(frozen=True)
class CheckpointState:
    current_z: float
    element_index: int
    slice_index: int
    phase_step: int
    u_fourier_valid: bool
    stage: str


class RunCheckpoint:
    """Persist and verify u/U plus the next unit of work for one source."""

    def __init__(
        self,
        directory: Path | str,
        fingerprint: dict[str, Any],
        progress_cb: Callable[[str, int, int], Any] | None = None,
        cancel_token: Any | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.progress_cb = progress_cb
        self.cancel_token = cancel_token
        self.manifest_path = self.directory / "manifest.json"

    @property
    def exists(self) -> bool:
        return self.manifest_path.exists()

    def _progress(self, stage: str, completed: int, total: int) -> None:
        if _cancelled(self.cancel_token):
            raise InterruptedError(f"checkpoint cancelled during {stage}")
        if self.progress_cb is not None:
            self.progress_cb(stage, completed, total)

    def save(
        self,
        u: Vector,
        U: Vector,
        *,
        current_z: float,
        element_index: int,
        slice_index: int,
        phase_step: int = 0,
        u_fourier_valid: bool,
        stage: str,
    ) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        u_name = f"u.{generation}.npy"
        U_name = f"spectrum.{generation}.npy"
        u_part = self.directory / f"{u_name}.part"
        U_part = self.directory / f"{U_name}.part"
        manifest_part = self.directory / "manifest.json.part"
        self._progress("checkpoint_u", 0, 3)
        _write_vector(u, u_part)
        self._progress("checkpoint_U", 1, 3)
        _write_vector(U, U_part)
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "status": "ready",
            "fingerprint": self.fingerprint,
            "state": {
                "current_z": float(current_z),
                "element_index": int(element_index),
                "slice_index": int(slice_index),
                "phase_step": int(phase_step),
                "u_fourier_valid": bool(u_fourier_valid),
                "stage": str(stage),
            },
            "vectors": {
                "u": {"file": u_name, "sha256": _sha256(u_part)},
                "U": {"file": U_name, "sha256": _sha256(U_part)},
            },
        }
        with manifest_part.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(u_part, self.directory / u_name)
        os.replace(U_part, self.directory / U_name)
        os.replace(manifest_part, self.manifest_path)
        for pattern in ("u.*.npy", "spectrum.*.npy"):
            for stale in self.directory.glob(pattern):
                if stale.name not in {u_name, U_name}:
                    stale.unlink()
        self._progress("checkpoint_commit", 3, 3)

    def load(self, u: Vector, U: Vector) -> CheckpointState:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("format_version", 0)) != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if manifest.get("status") != "ready":
            raise ValueError(f"checkpoint status is {manifest.get('status')!r}, not 'ready'")
        if manifest.get("fingerprint") != self.fingerprint:
            raise ValueError("checkpoint does not match this simulation/source fingerprint")
        expected_length = int(self.fingerprint["N"])
        expected_dtype = np.dtype(self.fingerprint["dtype"])
        for name in ("u", "U"):
            path = self.directory / manifest["vectors"][name]["file"]
            if not path.exists() or _sha256(path) != manifest["vectors"][name]["sha256"]:
                raise ValueError(f"checkpoint {name} checksum mismatch")
        self._progress("checkpoint_restore", 0, 2)
        _restore_vector(
            self.directory / manifest["vectors"]["u"]["file"],
            u, expected_length, expected_dtype,
        )
        _restore_vector(
            self.directory / manifest["vectors"]["U"]["file"],
            U, expected_length, expected_dtype,
        )
        self._progress("checkpoint_restore", 2, 2)
        state = manifest["state"]
        return CheckpointState(
            current_z=float(state["current_z"]),
            element_index=int(state["element_index"]),
            slice_index=int(state["slice_index"]),
            phase_step=int(state["phase_step"]),
            u_fourier_valid=bool(state["u_fourier_valid"]),
            stage=str(state["stage"]),
        )

    def mark_complete(self) -> None:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["status"] = "complete"
        part = self.directory / "manifest.json.part"
        with part.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, self.manifest_path)
