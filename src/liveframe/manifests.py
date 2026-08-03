"""Strict upstream artifact manifest validation and local checksum binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a required artifact is mutable, unresolved, or mismatched."""


@dataclass(frozen=True, slots=True)
class Artifact:
    artifact_id: str
    kind: str
    repository: str
    revision: str | None
    revision_state: str
    local_sha256: str | None
    raw: Mapping[str, Any]

    @property
    def resolved(self) -> bool:
        return bool(self.revision) and "unresolved" not in self.revision_state


@dataclass(frozen=True, slots=True)
class UpstreamManifest:
    schema: str
    status: str
    artifacts: tuple[Artifact, ...]
    source: Path

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        return tuple(item.artifact_id for item in self.artifacts if not item.resolved)

    def require_resolved(self) -> None:
        if self.status not in {"pinned", "frozen"} or self.unresolved_ids:
            unresolved = ", ".join(self.unresolved_ids) or "manifest status is not pinned"
            raise ManifestError(f"upstream manifest is not execution-ready: {unresolved}")

    def artifact(self, artifact_id: str) -> Artifact:
        matches = [item for item in self.artifacts if item.artifact_id == artifact_id]
        if len(matches) != 1:
            raise ManifestError(f"expected exactly one artifact named {artifact_id!r}")
        return matches[0]


def load_upstream_manifest(path: Path) -> UpstreamManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load upstream manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != "liveframe.upstream-manifest.v1":
        raise ManifestError("unsupported upstream manifest schema")
    rows = raw.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("upstream manifest requires a non-empty artifacts array")
    artifacts: list[Artifact] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ManifestError("artifact entry must be an object")
        artifact_id = _required_string(row, "id")
        if artifact_id in seen:
            raise ManifestError(f"duplicate artifact ID: {artifact_id}")
        seen.add(artifact_id)
        artifacts.append(
            Artifact(
                artifact_id=artifact_id,
                kind=_required_string(row, "kind"),
                repository=_required_string(row, "repository"),
                revision=_optional_string(row, "revision"),
                revision_state=_required_string(row, "revision_state"),
                local_sha256=_optional_string(row, "local_sha256"),
                raw=row,
            )
        )
    return UpstreamManifest(
        schema=str(raw["schema"]),
        status=_required_string(raw, "status"),
        artifacts=tuple(artifacts),
        source=path,
    )


def verify_local_file(artifact: Artifact, path: Path) -> str:
    if not artifact.local_sha256:
        raise ManifestError(f"artifact {artifact.artifact_id} has no frozen local SHA-256")
    actual = file_sha256(path)
    if actual != artifact.local_sha256:
        raise ManifestError(
            f"artifact {artifact.artifact_id} checksum mismatch: "
            f"expected {artifact.local_sha256}, computed {actual}"
        )
    return actual


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestError(f"cannot hash {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ManifestError(f"{key} must be a non-empty string")
    return result


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result:
        raise ManifestError(f"{key} must be null or a non-empty string")
    return result
