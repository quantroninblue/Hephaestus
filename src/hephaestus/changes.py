"""Typed project changes and atomic application."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from hephaestus.index import ProjectIndex
from hephaestus.model import (
    Connection,
    EngineeringProject,
    Entity,
    EntityKind,
    ModelError,
    Provenance,
    ProvenanceSource,
    Reference,
    Requirement,
    _to_data,
    assert_identifier,
)
from hephaestus.persistence import canonical_project_json, load_project, save_project
from hephaestus.validation import ValidationReport, ValidationSeverity, validate_project


class ChangeError(RuntimeError):
    """Raised when a change set cannot be applied atomically."""


class ChangeSetLoadError(ValueError):
    """Raised when untrusted ChangeSet JSON is malformed."""


class OperationKind(StrEnum):
    ADD_REQUIREMENT = "add_requirement"
    ADD_ENTITY = "add_entity"
    UPDATE_ENTITY = "update_entity"
    REMOVE_ENTITY = "remove_entity"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    CONNECT = "add_connection"
    DISCONNECT = "remove_connection"


@dataclass(frozen=True, slots=True)
class AddRequirementOperation:
    operation: Literal[OperationKind.ADD_REQUIREMENT]
    requirement: Requirement

    def __post_init__(self) -> None:
        if self.operation != OperationKind.ADD_REQUIREMENT:
            raise ChangeError("AddRequirementOperation must use operation add_requirement")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        existing = ProjectIndex.build(project).resolve(
            Reference("requirement", self.requirement.id)
        )
        if existing is not None:
            raise ChangeError(f"requirement already exists: {self.requirement.id}")
        return replace(project, requirements=(*project.requirements, self.requirement))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "requirement": _to_data(self.requirement),
        }


@dataclass(frozen=True, slots=True)
class AddEntityOperation:
    operation: Literal[OperationKind.ADD_ENTITY]
    entity: Entity

    def __post_init__(self) -> None:
        if self.operation != OperationKind.ADD_ENTITY:
            raise ChangeError("AddEntityOperation must use operation 'add_entity'")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        if ProjectIndex.build(project).resolve(Reference("entity", self.entity.id)) is not None:
            raise ChangeError(f"entity already exists: {self.entity.id}")
        return replace(project, entities=(*project.entities, self.entity))

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "entity": _to_data(self.entity)}


@dataclass(frozen=True, slots=True)
class UpdateEntityOperation:
    operation: Literal[OperationKind.UPDATE_ENTITY]
    entity_id: str
    name: str | None = None
    kind: EntityKind | None = None

    def __post_init__(self) -> None:
        if self.operation != OperationKind.UPDATE_ENTITY:
            raise ChangeError("UpdateEntityOperation must use operation 'update_entity'")
        if self.name is None and self.kind is None:
            raise ChangeError("update_entity requires at least one supported field")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        updated: list[Entity] = []
        found = False
        for entity in project.entities:
            if entity.id == self.entity_id:
                found = True
                updated.append(
                    replace(
                        entity,
                        name=self.name if self.name is not None else entity.name,
                        kind=self.kind if self.kind is not None else entity.kind,
                    )
                )
            else:
                updated.append(entity)
        if not found:
            raise ChangeError(f"entity does not exist: {self.entity_id}")
        return replace(project, entities=tuple(updated))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"operation": self.operation.value, "entity_id": self.entity_id}
        if self.name is not None:
            data["name"] = self.name
        if self.kind is not None:
            data["kind"] = self.kind.value
        return data


@dataclass(frozen=True, slots=True)
class RemoveEntityOperation:
    operation: Literal[OperationKind.REMOVE_ENTITY]
    entity_id: str

    def __post_init__(self) -> None:
        if self.operation != OperationKind.REMOVE_ENTITY:
            raise ChangeError("RemoveEntityOperation must use operation 'remove_entity'")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        index = ProjectIndex.build(project)
        if index.resolve(Reference("entity", self.entity_id)) is None:
            raise ChangeError(f"entity does not exist: {self.entity_id}")
        refs = index.removal_blockers("entity", self.entity_id)
        if refs:
            joined_refs = ", ".join(refs)
            raise ChangeError(f"entity {self.entity_id} is still referenced by {joined_refs}")
        return replace(
            project,
            entities=tuple(e for e in project.entities if e.id != self.entity_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "entity_id": self.entity_id}


@dataclass(frozen=True, slots=True)
class AddConnectionOperation:
    operation: Literal[OperationKind.ADD_CONNECTION]
    connection: Connection

    def __post_init__(self) -> None:
        if self.operation != OperationKind.ADD_CONNECTION:
            raise ChangeError("AddConnectionOperation must use operation 'add_connection'")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        existing = ProjectIndex.build(project).resolve(Reference("connection", self.connection.id))
        if existing is not None:
            raise ChangeError(f"connection already exists: {self.connection.id}")
        return replace(project, connections=(*project.connections, self.connection))

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "connection": _to_data(self.connection)}


@dataclass(frozen=True, slots=True)
class RemoveConnectionOperation:
    operation: Literal[OperationKind.REMOVE_CONNECTION]
    connection_id: str

    def __post_init__(self) -> None:
        if self.operation != OperationKind.REMOVE_CONNECTION:
            raise ChangeError("RemoveConnectionOperation must use operation 'remove_connection'")

    def apply(self, project: EngineeringProject) -> EngineeringProject:
        if ProjectIndex.build(project).resolve(Reference("connection", self.connection_id)) is None:
            raise ChangeError(f"connection does not exist: {self.connection_id}")
        return replace(
            project,
            connections=tuple(c for c in project.connections if c.id != self.connection_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "connection_id": self.connection_id}


ConnectOperation = AddConnectionOperation
DisconnectOperation = RemoveConnectionOperation
ChangeOperation = (
    AddRequirementOperation
    | AddEntityOperation
    | UpdateEntityOperation
    | RemoveEntityOperation
    | AddConnectionOperation
    | RemoveConnectionOperation
)


@dataclass(frozen=True, slots=True)
class ChangeSet:
    id: str
    expected_revision: int
    summary: str
    provenance: Provenance
    operations: tuple[ChangeOperation, ...]

    def __post_init__(self) -> None:
        assert_identifier(self.id, "change set id")
        if self.expected_revision < 0:
            raise ChangeError("expected_revision must be non-negative")
        if not self.summary.strip():
            raise ChangeError("ChangeSet summary must not be empty")
        if not self.operations:
            raise ChangeError("ChangeSet requires at least one operation")

    def apply(self, project: EngineeringProject, *, dry_run: bool = False) -> ChangeResult:
        if project.revision != self.expected_revision:
            raise ChangeError(
                "revision precondition failed: "
                f"expected {self.expected_revision}, got {project.revision}"
            )
        before = fingerprint_project(project)
        draft = copy.deepcopy(project)
        for index, operation in enumerate(self.operations):
            try:
                draft = operation.apply(draft)
            except Exception as exc:
                raise ChangeError(f"operation {index} failed: {exc}") from exc
        draft = replace(draft, revision=project.revision + 1)
        report = validate_project(draft)
        if not dry_run and not report.is_valid:
            codes = ", ".join(
                finding.code
                for finding in report.findings
                if finding.severity in {ValidationSeverity.ERROR, ValidationSeverity.FATAL}
            )
            raise ChangeError(f"final validation failed after all operations: {codes}")
        after = fingerprint_project(draft)
        return ChangeResult(
            project=draft,
            before_fingerprint=before,
            after_fingerprint=after,
            summary=self.summary,
            validation=report,
            dry_run=dry_run,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "expected_revision": self.expected_revision,
            "summary": self.summary,
            "provenance": self.provenance.to_dict(),
            "operations": [operation.to_dict() for operation in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangeSet:
        _require_keys(
            data,
            {"id", "expected_revision", "summary", "provenance", "operations"},
            "ChangeSet",
        )
        _reject_unknown(
            data,
            {"id", "expected_revision", "summary", "provenance", "operations"},
            "ChangeSet",
        )
        if not isinstance(data["operations"], list):
            raise ChangeSetLoadError("ChangeSet operations must be an array")
        try:
            operations = tuple(_operation_from_dict(item) for item in data["operations"])
            return cls(
                id=_strict_str(data["id"], "id"),
                expected_revision=_strict_int(data["expected_revision"], "expected_revision"),
                summary=_strict_str(data["summary"], "summary"),
                provenance=_provenance_from_dict(data["provenance"]),
                operations=operations,
            )
        except (ModelError, ChangeError, KeyError, TypeError, ValueError) as exc:
            raise ChangeSetLoadError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class ChangeResult:
    project: EngineeringProject
    before_fingerprint: str
    after_fingerprint: str
    summary: str
    validation: ValidationReport
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "dry_run": self.dry_run,
            "revision": self.project.revision,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "validation": self.validation.to_dict(),
        }


def fingerprint_project(project: EngineeringProject) -> str:
    return hashlib.sha256(canonical_project_json(project).encode("utf-8")).hexdigest()


def canonical_changeset_json(changeset: ChangeSet) -> str:
    return json.dumps(changeset.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"


def load_changeset(path: Path) -> ChangeSet:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ChangeSetLoadError(f"could not read ChangeSet file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ChangeSetLoadError(f"malformed ChangeSet JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ChangeSetLoadError("ChangeSet document must be a JSON object")
    return ChangeSet.from_dict(data)


def apply_changeset_file(
    project_path: Path, changeset_path: Path, *, dry_run: bool = False
) -> ChangeResult:
    project = load_project(project_path)
    changeset = load_changeset(changeset_path)
    result = changeset.apply(project, dry_run=dry_run)
    if not dry_run:
        current = load_project(project_path)
        if fingerprint_project(current) != result.before_fingerprint:
            raise ChangeError("project file changed after ChangeSet evaluation")
        save_project(result.project, project_path, overwrite=True)
    return result


def _operation_from_dict(data: Any) -> ChangeOperation:
    if not isinstance(data, dict):
        raise ChangeSetLoadError("ChangeSet operation must be an object")
    operation = data.get("operation")
    if operation == OperationKind.ADD_REQUIREMENT.value:
        _require_keys(data, {"operation", "requirement"}, "add_requirement operation")
        _reject_unknown(data, {"operation", "requirement"}, "add_requirement operation")
        return AddRequirementOperation(
            OperationKind.ADD_REQUIREMENT, _requirement_from_dict(data["requirement"])
        )
    if operation == OperationKind.ADD_ENTITY.value:
        _require_keys(data, {"operation", "entity"}, "add_entity operation")
        _reject_unknown(data, {"operation", "entity"}, "add_entity operation")
        return AddEntityOperation(OperationKind.ADD_ENTITY, _entity_from_dict(data["entity"]))
    if operation == OperationKind.UPDATE_ENTITY.value:
        _require_keys(data, {"operation", "entity_id"}, "update_entity operation")
        _reject_unknown(data, {"operation", "entity_id", "name", "kind"}, "update_entity operation")
        kind = EntityKind(data["kind"]) if "kind" in data else None
        return UpdateEntityOperation(
            OperationKind.UPDATE_ENTITY,
            _strict_str(data["entity_id"], "entity_id"),
            _strict_str(data["name"], "name") if "name" in data else None,
            kind,
        )
    if operation == OperationKind.REMOVE_ENTITY.value:
        _require_keys(data, {"operation", "entity_id"}, "remove_entity operation")
        _reject_unknown(data, {"operation", "entity_id"}, "remove_entity operation")
        return RemoveEntityOperation(
            OperationKind.REMOVE_ENTITY,
            _strict_str(data["entity_id"], "entity_id"),
        )
    if operation == OperationKind.ADD_CONNECTION.value:
        _require_keys(data, {"operation", "connection"}, "add_connection operation")
        _reject_unknown(data, {"operation", "connection"}, "add_connection operation")
        return AddConnectionOperation(
            OperationKind.ADD_CONNECTION, _connection_from_dict(data["connection"])
        )
    if operation == OperationKind.REMOVE_CONNECTION.value:
        _require_keys(data, {"operation", "connection_id"}, "remove_connection operation")
        _reject_unknown(data, {"operation", "connection_id"}, "remove_connection operation")
        return RemoveConnectionOperation(
            OperationKind.REMOVE_CONNECTION,
            _strict_str(data["connection_id"], "connection_id"),
        )
    raise ChangeSetLoadError(f"unknown ChangeSet operation: {operation!r}")


def _requirement_from_dict(data: Any) -> Requirement:
    if not isinstance(data, dict):
        raise ChangeSetLoadError("requirement payload must be an object")
    project_data = EngineeringProject.empty(name="ChangeSet requirement decode").to_dict()
    project_data["requirements"] = [data]
    return EngineeringProject.from_dict(project_data).requirements[0]


def _entity_from_dict(data: Any) -> Entity:
    if not isinstance(data, dict):
        raise ChangeSetLoadError("entity payload must be an object")
    _require_keys(data, {"id", "name"}, "entity payload")
    allowed = {"id", "name", "kind", "parent_entity", "frame_ref", "extensions"}
    _reject_unknown(data, allowed, "entity payload")
    return Entity(
        id=_strict_str(data["id"], "entity.id"),
        name=_strict_str(data["name"], "entity.name"),
        kind=EntityKind(data.get("kind", EntityKind.CUSTOM.value)),
        parent_entity=(
            Reference.from_dict(data["parent_entity"]) if "parent_entity" in data else None
        ),
        frame_ref=Reference.from_dict(data["frame_ref"]) if "frame_ref" in data else None,
        extensions=_json_object(data.get("extensions", {}), "entity.extensions"),
    )


def _connection_from_dict(data: Any) -> Connection:
    if not isinstance(data, dict):
        raise ChangeSetLoadError("connection payload must be an object")
    _require_keys(data, {"id", "kind", "source", "target"}, "connection payload")
    _reject_unknown(data, {"id", "kind", "source", "target", "metadata"}, "connection payload")
    return Connection(
        id=_strict_str(data["id"], "connection.id"),
        kind=_strict_str(data["kind"], "connection.kind"),
        source=Reference.from_dict(data["source"]),
        target=Reference.from_dict(data["target"]),
        metadata=_json_object(data.get("metadata", {}), "connection.metadata"),
    )


def _require_keys(data: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ChangeSetLoadError(f"missing required fields for {label}: {', '.join(missing)}")


def _reject_unknown(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ChangeSetLoadError(f"unknown fields for {label}: {', '.join(unknown)}")


def _provenance_from_dict(data: Any) -> Provenance:
    if not isinstance(data, dict):
        raise ChangeSetLoadError("provenance payload must be an object")
    _require_keys(data, {"source"}, "provenance payload")
    _reject_unknown(data, {"source", "detail", "reference"}, "provenance payload")
    return Provenance(
        source=ProvenanceSource(_strict_str(data["source"], "provenance.source")),
        detail=_strict_str(data.get("detail", ""), "provenance.detail"),
        reference=(
            _strict_str(data["reference"], "provenance.reference")
            if data.get("reference") is not None
            else None
        ),
    )


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChangeSetLoadError(f"{label} must be an object")
    try:
        json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ChangeSetLoadError(f"{label} must contain only JSON values") from exc
    return value


def _strict_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ChangeSetLoadError(f"{label} must be a string")
    return value


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChangeSetLoadError(f"{label} must be an integer")
    return value
