"""Generated artifact manifest and atomic publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from hephaestus.changes import fingerprint_project
from hephaestus.mobile_robot import design_generation_id
from hephaestus.model import EngineeringProject
from hephaestus.persistence import PROJECT_FILENAME, ProjectPersistenceError, load_project

ARTIFACT_ROOT_NAME = "generated"
MANIFEST_FILENAME = "hephaestus.artifacts.json"
MANIFEST_SCHEMA = "hephaestus.artifacts.v1"
GENERATOR_VERSION = "hephaestus.artifacts.publisher.v1"


class ArtifactError(RuntimeError):
    """Raised when generated artifacts cannot be published or inspected safely."""


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    role: str
    owning_ids: tuple[str, ...]
    units: str
    coordinate_convention: str
    sha256: str
    dependencies: tuple[str, ...] = ()
    status: str = "fresh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "owning_ids": list(self.owning_ids),
            "units": self.units,
            "coordinate_convention": self.coordinate_convention,
            "sha256": self.sha256,
            "dependencies": list(self.dependencies),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    project_revision: int
    project_fingerprint: str
    generation_id: str
    generated_at: str
    generator_versions: Mapping[str, str]
    backend_versions: Mapping[str, str | None]
    artifacts: tuple[ArtifactRecord, ...]
    schema: str = MANIFEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "project_revision": self.project_revision,
            "project_fingerprint": self.project_fingerprint,
            "generation_id": self.generation_id,
            "generated_at": self.generated_at,
            "generator_versions": dict(sorted(self.generator_versions.items())),
            "backend_versions": dict(sorted(self.backend_versions.items())),
            "artifacts": [record.to_dict() for record in self.artifacts],
        }


@dataclass(slots=True)
class ArtifactStager:
    project_dir: Path
    project: EngineeringProject
    backend_versions: Mapping[str, str | None] = field(default_factory=dict)
    root: Path = field(init=False)
    staging: Path = field(init=False)
    _tmpdir_obj: tempfile.TemporaryDirectory[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.project_dir = self.project_dir.resolve()
        self.root = self.project_dir / ARTIFACT_ROOT_NAME
        _reject_symlink_ancestors(self.project_dir)
        _reject_symlink_ancestors(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._tmpdir_obj = tempfile.TemporaryDirectory(
            prefix=".hephaestus-generated-", dir=str(self.root)
        )
        self.staging = Path(self._tmpdir_obj.name)

    def path(self, relative: str) -> Path:
        safe = validate_relative_artifact_path(relative)
        path = self.staging / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, relative: str, content: str) -> Path:
        path = self.path(relative)
        path.write_text(content, encoding="utf-8")
        return path

    def copy_file(self, source: Path, relative: str) -> Path:
        target = self.path(relative)
        shutil.copy2(source, target)
        return target

    def publish(
        self,
        records: Iterable[ArtifactRecord],
        *,
        generator_versions: Mapping[str, str],
    ) -> ArtifactManifest:
        records_tuple = tuple(sorted(records, key=lambda item: item.path))
        _validate_records(self.staging, records_tuple)
        manifest = ArtifactManifest(
            project_revision=self.project.revision,
            project_fingerprint=fingerprint_project(self.project),
            generation_id=design_generation_id(self.project),
            generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            generator_versions={
                "hephaestus.artifacts": GENERATOR_VERSION,
                **dict(generator_versions),
            },
            backend_versions=self.backend_versions,
            artifacts=records_tuple,
        )
        manifest_payload = dumps_manifest(manifest)
        self.write_text(MANIFEST_FILENAME, manifest_payload)
        for child in self.root.iterdir():
            if child.name == self.staging.name:
                continue
            if child.name == MANIFEST_FILENAME:
                continue
            if child.is_symlink():
                raise ArtifactError(f"refusing to replace generated symlink: {child.name}")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        staged_children = sorted(self.staging.iterdir(), key=lambda item: item.name)
        for child in staged_children:
            if child.name == MANIFEST_FILENAME:
                continue
            os.replace(child, self.root / child.name)
        os.replace(self.staging / MANIFEST_FILENAME, self.root / MANIFEST_FILENAME)
        _fsync_directory(self.root)
        self.cleanup()
        return manifest

    def cleanup(self) -> None:
        self._tmpdir_obj.cleanup()


def artifact_root(project_path: Path) -> Path:
    project_file = (
        project_path if project_path.name == PROJECT_FILENAME else project_path / PROJECT_FILENAME
    )
    return project_file.parent / ARTIFACT_ROOT_NAME


def load_manifest(project_path: Path) -> tuple[ArtifactManifest | None, str]:
    project = load_project(project_path)
    manifest_path = artifact_root(project_path) / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None, "missing"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"could not load artifact manifest: {exc}") from exc
    manifest = manifest_from_dict(raw)
    status = "fresh" if manifest.project_fingerprint == fingerprint_project(project) else "stale"
    if status == "stale":
        manifest = _with_status(manifest, "stale")
    _validate_manifest_paths(artifact_root(project_path), manifest)
    return manifest, status


def manifest_from_dict(data: Mapping[str, Any]) -> ArtifactManifest:
    if data.get("schema") != MANIFEST_SCHEMA:
        raise ArtifactError("unsupported artifact manifest schema")
    artifacts = tuple(
        ArtifactRecord(
            path=str(item["path"]),
            role=str(item["role"]),
            owning_ids=tuple(str(value) for value in item.get("owning_ids", ())),
            units=str(item["units"]),
            coordinate_convention=str(item["coordinate_convention"]),
            sha256=str(item["sha256"]),
            dependencies=tuple(str(value) for value in item.get("dependencies", ())),
            status=str(item.get("status", "fresh")),
        )
        for item in data.get("artifacts", ())
    )
    return ArtifactManifest(
        project_revision=int(data["project_revision"]),
        project_fingerprint=str(data["project_fingerprint"]),
        generation_id=str(data["generation_id"]),
        generated_at=str(data["generated_at"]),
        generator_versions={str(k): str(v) for k, v in data.get("generator_versions", {}).items()},
        backend_versions={
            str(k): (None if v is None else str(v))
            for k, v in data.get("backend_versions", {}).items()
        },
        artifacts=artifacts,
        schema=str(data["schema"]),
    )


def dumps_manifest(manifest: ArtifactManifest) -> str:
    return json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"


def record_for_file(
    root: Path,
    relative: str,
    *,
    role: str,
    owning_ids: Iterable[str],
    units: str = "SI",
    coordinate_convention: str = "right-handed X-forward Y-left Z-up",
    dependencies: Iterable[str] = (),
) -> ArtifactRecord:
    safe = validate_relative_artifact_path(relative)
    path = root / safe
    if not path.exists() or not path.is_file():
        raise ArtifactError(f"artifact is missing: {relative}")
    return ArtifactRecord(
        path=safe,
        role=role,
        owning_ids=tuple(sorted(owning_ids)),
        units=units,
        coordinate_convention=coordinate_convention,
        sha256=sha256_file(path),
        dependencies=tuple(sorted(validate_relative_artifact_path(item) for item in dependencies)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_relative_artifact_path(relative: str) -> str:
    if not isinstance(relative, str) or not relative.strip():
        raise ArtifactError("artifact path must be a non-empty relative path")
    if "\\" in relative:
        raise ArtifactError(f"unsafe artifact path separator: {relative}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ArtifactError(f"unsafe artifact path: {relative}")
    if any(not _SAFE_PATH_PART.fullmatch(part) for part in pure.parts):
        raise ArtifactError(f"unsafe artifact path component: {relative}")
    return pure.as_posix()


def sanitized_name(value: str, *, default: str = "hephaestus_robot") -> str:
    candidate = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    if not candidate:
        candidate = default
    if not candidate[0].isalpha():
        candidate = f"robot_{candidate}"
    return candidate[:48]


def inspect_artifacts(project_path: Path) -> dict[str, Any]:
    manifest, status = load_manifest(project_path)
    if manifest is None:
        return {
            "schema": "hephaestus.artifacts.inspect.v1",
            "status": "missing",
            "artifact_count": 0,
            "artifacts": [],
        }
    by_role: dict[str, int] = {}
    for artifact in manifest.artifacts:
        by_role[artifact.role] = by_role.get(artifact.role, 0) + 1
    return {
        "schema": "hephaestus.artifacts.inspect.v1",
        "status": status,
        "project_revision": manifest.project_revision,
        "project_fingerprint": manifest.project_fingerprint,
        "generation_id": manifest.generation_id,
        "backend_versions": dict(manifest.backend_versions),
        "artifact_count": len(manifest.artifacts),
        "roles": dict(sorted(by_role.items())),
        "important_paths": [artifact.path for artifact in manifest.artifacts[:12]],
        "artifacts": [artifact.to_dict() for artifact in manifest.artifacts],
    }


def _validate_records(root: Path, records: tuple[ArtifactRecord, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        relative = validate_relative_artifact_path(record.path)
        if relative in seen:
            raise ArtifactError(f"duplicate artifact record: {relative}")
        seen.add(relative)
        path = root / relative
        if not path.exists():
            raise ArtifactError(f"manifest would point to missing artifact: {relative}")
        if sha256_file(path) != record.sha256:
            raise ArtifactError(f"artifact hash mismatch before publication: {relative}")


def _validate_manifest_paths(root: Path, manifest: ArtifactManifest) -> None:
    for record in manifest.artifacts:
        relative = validate_relative_artifact_path(record.path)
        path = root / relative
        if not path.exists():
            raise ArtifactError(f"manifest points to missing artifact: {relative}")
        if not path.is_file():
            raise ArtifactError(f"manifest points to non-file artifact: {relative}")
        if sha256_file(path) != record.sha256:
            raise ArtifactError(f"manifest artifact hash mismatch: {relative}")


def _with_status(manifest: ArtifactManifest, status: str) -> ArtifactManifest:
    return ArtifactManifest(
        manifest.project_revision,
        manifest.project_fingerprint,
        manifest.generation_id,
        manifest.generated_at,
        manifest.generator_versions,
        manifest.backend_versions,
        tuple(replace_record_status(record, status) for record in manifest.artifacts),
        manifest.schema,
    )


def replace_record_status(record: ArtifactRecord, status: str) -> ArtifactRecord:
    return ArtifactRecord(
        record.path,
        record.role,
        record.owning_ids,
        record.units,
        record.coordinate_convention,
        record.sha256,
        record.dependencies,
        status,
    )


def _reject_symlink_ancestors(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ProjectPersistenceError(f"refusing generated artifact symlink path: {current}")
        if not current.exists():
            break


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


_SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9._-]+$")
