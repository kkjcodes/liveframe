"""Portable, pickle-free golden tensors for cross-runtime numerical parity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

FIXTURE_SCHEMA = "liveframe.golden-fixture.v1"
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


class FixtureError(ValueError):
    """Raised when a golden fixture is unsafe, incomplete, or corrupted."""


@dataclass(frozen=True, slots=True)
class GoldenTensor:
    name: str
    role: str
    file: str
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    sha256: str
    scheduler_transition: str | None = None

    @property
    def expected_bytes(self) -> int:
        try:
            item_size = _DTYPE_BYTES[self.dtype]
        except KeyError as exc:
            raise FixtureError(f"unsupported golden tensor dtype: {self.dtype}") from exc
        return math.prod(self.shape) * item_size

    def validate(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise FixtureError(f"unsafe tensor name: {self.name!r}")
        if not self.role:
            raise FixtureError("tensor role must not be empty")
        if Path(self.file).name != self.file or not _NAME.fullmatch(self.file):
            raise FixtureError(f"unsafe tensor file: {self.file!r}")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise FixtureError("tensor dimensions must be positive")
        if self.byte_order not in {"little", "not-applicable"}:
            raise FixtureError("portable fixtures must be little endian")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.sha256):
            raise FixtureError("tensor SHA-256 is malformed")
        _ = self.expected_bytes
        if self.role == "scheduler_noise" and not self.scheduler_transition:
            raise FixtureError("scheduler noise must identify its exact transition")


@dataclass(frozen=True, slots=True)
class GoldenFixture:
    fixture_id: str
    source_runtime: str
    source_revision: str
    model_artifact_id: str
    tensors: tuple[GoldenTensor, ...]
    metadata: Mapping[str, Any]
    root: Path
    schema: str = FIXTURE_SCHEMA

    def verify(self) -> None:
        if not _NAME.fullmatch(self.fixture_id):
            raise FixtureError(f"unsafe fixture ID: {self.fixture_id!r}")
        if not self.source_runtime or not self.source_revision or not self.model_artifact_id:
            raise FixtureError("fixture provenance is incomplete")
        if not self.tensors:
            raise FixtureError("fixture must contain at least one tensor")
        names: set[str] = set()
        transitions: set[str] = set()
        initial_noise = 0
        for tensor in self.tensors:
            tensor.validate()
            if tensor.name in names:
                raise FixtureError(f"duplicate tensor name: {tensor.name}")
            names.add(tensor.name)
            if tensor.role == "initial_noise":
                initial_noise += 1
            if tensor.scheduler_transition:
                if tensor.scheduler_transition in transitions:
                    raise FixtureError(
                        f"duplicate scheduler transition: {tensor.scheduler_transition}"
                    )
                transitions.add(tensor.scheduler_transition)
            path = self.root / tensor.file
            try:
                size = path.stat().st_size
            except OSError as exc:
                raise FixtureError(f"missing golden tensor {path}: {exc}") from exc
            if size != tensor.expected_bytes:
                raise FixtureError(
                    f"golden tensor {tensor.name} byte size mismatch: "
                    f"expected {tensor.expected_bytes}, found {size}"
                )
            if _sha256(path) != tensor.sha256:
                raise FixtureError(f"golden tensor {tensor.name} checksum mismatch")
        if initial_noise != 1:
            raise FixtureError("fixture requires exactly one initial_noise tensor")


def write_fixture(
    root: Path,
    *,
    fixture_id: str,
    source_runtime: str,
    source_revision: str,
    model_artifact_id: str,
    tensors: Iterable[tuple[str, str, bytes, tuple[int, ...], str, str | None]],
    metadata: Mapping[str, Any] | None = None,
) -> GoldenFixture:
    root.mkdir(parents=True, exist_ok=True)
    entries: list[GoldenTensor] = []
    for name, role, payload, shape, dtype, transition in tensors:
        file = f"{name}.bin"
        entry = GoldenTensor(
            name=name,
            role=role,
            file=file,
            shape=shape,
            dtype=dtype,
            byte_order="not-applicable" if _DTYPE_BYTES.get(dtype) == 1 else "little",
            sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
            scheduler_transition=transition,
        )
        entry.validate()
        if len(payload) != entry.expected_bytes:
            raise FixtureError(
                f"payload {name} byte size mismatch: expected {entry.expected_bytes}, "
                f"found {len(payload)}"
            )
        _atomic_bytes(root / file, payload)
        entries.append(entry)
    manifest = {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": fixture_id,
        "source_runtime": source_runtime,
        "source_revision": source_revision,
        "model_artifact_id": model_artifact_id,
        "metadata": dict(metadata or {}),
        "tensors": [asdict(entry) for entry in entries],
    }
    _atomic_bytes(
        root / "fixture.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return load_fixture(root)


def load_fixture(root: Path) -> GoldenFixture:
    path = root / "fixture.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"cannot load fixture manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != FIXTURE_SCHEMA:
        raise FixtureError("unsupported golden fixture schema")
    rows = raw.get("tensors")
    if not isinstance(rows, list):
        raise FixtureError("fixture tensors must be an array")
    tensors = tuple(_tensor_from_json(row) for row in rows)
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise FixtureError("fixture metadata must be an object")
    fixture = GoldenFixture(
        fixture_id=_string(raw, "fixture_id"),
        source_runtime=_string(raw, "source_runtime"),
        source_revision=_string(raw, "source_revision"),
        model_artifact_id=_string(raw, "model_artifact_id"),
        tensors=tensors,
        metadata=metadata,
        root=root,
    )
    fixture.verify()
    return fixture


def _tensor_from_json(value: Any) -> GoldenTensor:
    if not isinstance(value, dict):
        raise FixtureError("tensor entry must be an object")
    shape = value.get("shape")
    invalid_shape = not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in shape
    )
    if invalid_shape:
        raise FixtureError("tensor shape must be an integer array")
    dimensions = cast(list[int], shape)
    transition = value.get("scheduler_transition")
    if transition is not None and not isinstance(transition, str):
        raise FixtureError("scheduler_transition must be null or a string")
    return GoldenTensor(
        name=_string(value, "name"),
        role=_string(value, "role"),
        file=_string(value, "file"),
        shape=tuple(dimensions),
        dtype=_string(value, "dtype"),
        byte_order=_string(value, "byte_order"),
        sha256=_string(value, "sha256"),
        scheduler_transition=transition,
    )


def _string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise FixtureError(f"{key} must be a non-empty string")
    return result


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
