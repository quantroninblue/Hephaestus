"""Deterministic project persistence."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from hephaestus.model import EngineeringProject, ModelError, UnsupportedSchemaVersionError
from hephaestus.validation import validate_project

PROJECT_FILENAME = "hephaestus.project.json"


class ProjectPersistenceError(RuntimeError):
    """Raised for clean project load/save errors."""


def resolve_project_file(path: Path) -> Path:
    """Resolve a project directory or canonical project filename to the project file."""

    return path if path.name == PROJECT_FILENAME else path / PROJECT_FILENAME


def canonical_project_json(project: EngineeringProject) -> str:
    return json.dumps(project.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"


def save_project(
    project: EngineeringProject,
    path: Path,
    *,
    overwrite: bool = True,
    validate: bool = True,
) -> Path:
    project_file = resolve_project_file(path)
    project_dir = project_file.parent

    if path.name != PROJECT_FILENAME and path.exists() and not path.is_dir():
        raise ProjectPersistenceError(f"project path is not a directory: {path}")
    _reject_symlink_ancestors(project_dir, "write")
    if project_dir.exists() and project_dir.is_symlink():
        raise ProjectPersistenceError(
            f"refusing to write through project directory symlink: {project_dir}"
        )
    if project_file.is_symlink():
        raise ProjectPersistenceError(f"refusing to overwrite project file symlink: {project_file}")
    if project_file.exists() and project_file.is_dir():
        raise ProjectPersistenceError(f"project file path is a directory: {project_file}")
    if project_file.exists() and not overwrite:
        raise ProjectPersistenceError(f"project file already exists: {project_file}")

    if validate:
        report = validate_project(project)
        if not report.is_valid:
            codes = ", ".join(finding.code for finding in report.findings)
            raise ProjectPersistenceError(f"refusing to save invalid project: {codes}")

    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        payload = canonical_project_json(project)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{PROJECT_FILENAME}.", suffix=".tmp", dir=str(project_dir), text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, project_file)
            _fsync_directory(project_dir)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)
            raise
    except OSError as exc:
        raise ProjectPersistenceError(f"could not write project file: {project_file}") from exc
    except ValueError as exc:
        raise ProjectPersistenceError(f"could not serialize project file: {exc}") from exc
    return project_file


def load_project(path: Path) -> EngineeringProject:
    project_file = resolve_project_file(path)
    _reject_symlink_ancestors(project_file.parent, "read")
    if project_file.is_symlink():
        raise ProjectPersistenceError(f"refusing to read project file symlink: {project_file}")
    try:
        raw = project_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProjectPersistenceError(f"could not read project file: {project_file}") from exc
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectPersistenceError(f"malformed project JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProjectPersistenceError("project document must be a JSON object")
    try:
        return EngineeringProject.from_dict(data)
    except UnsupportedSchemaVersionError:
        raise
    except ModelError as exc:
        raise ProjectPersistenceError(str(exc)) from exc


def _reject_symlink_ancestors(path: Path, operation: str) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ProjectPersistenceError(
                f"refusing to {operation} through project path symlink: {current}"
            )
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
