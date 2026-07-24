"""Deterministic JSON Schema generation for engineering boundary types."""

from __future__ import annotations

import json
from dataclasses import MISSING, fields, is_dataclass
from enum import StrEnum
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints

from hephaestus.changes import (
    AddConnectionOperation,
    AddEntityOperation,
    AddRequirementOperation,
    ChangeSet,
    OperationKind,
    RemoveConnectionOperation,
    RemoveEntityOperation,
    UpdateEntityOperation,
)
from hephaestus.model import EngineeringProject, Quantity

_OPERATION_TYPES = {
    OperationKind.ADD_REQUIREMENT.value: AddRequirementOperation,
    OperationKind.ADD_ENTITY.value: AddEntityOperation,
    OperationKind.UPDATE_ENTITY.value: UpdateEntityOperation,
    OperationKind.REMOVE_ENTITY.value: RemoveEntityOperation,
    OperationKind.ADD_CONNECTION.value: AddConnectionOperation,
    OperationKind.REMOVE_CONNECTION.value: RemoveConnectionOperation,
}


def engineering_project_schema() -> dict[str, Any]:
    """Return the JSON Schema for an ``EngineeringProject`` document."""

    return _document_schema("EngineeringProject", _schema_for_dataclass(EngineeringProject))


def changeset_schema() -> dict[str, Any]:
    """Return the JSON Schema for a serialized ``ChangeSet`` document."""

    schema = _schema_for_dataclass(ChangeSet)
    schema["properties"]["operations"] = {
        "type": "array",
        "items": {"oneOf": [operation_schema(name) for name in sorted(_OPERATION_TYPES)]},
    }
    return _document_schema("ChangeSet", schema)


def operation_schema(operation: str) -> dict[str, Any]:
    """Return the JSON Schema for one supported ChangeSet operation."""

    try:
        operation_type = _OPERATION_TYPES[operation]
    except KeyError as exc:
        raise ValueError(f"unknown operation schema: {operation}") from exc
    schema = _schema_for_dataclass(operation_type)
    schema.setdefault("required", [])
    schema["properties"]["operation"] = {"const": operation}
    if "operation" not in schema["required"]:
        schema["required"].append("operation")
        schema["required"].sort()
    return schema


def schema_json(schema: dict[str, Any]) -> str:
    """Serialize a schema deterministically."""

    return json.dumps(schema, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _document_schema(title: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        **schema,
    }


def _schema_for_dataclass(cls: type[Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = get_type_hints(cls)
    for item in fields(cls):
        if not item.init:
            continue
        annotation = hints.get(item.name, item.type)
        properties[item.name] = _schema_for_type(annotation)
        if item.default is MISSING and item.default_factory is MISSING:
            required.append(item.name)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {key: properties[key] for key in sorted(properties)},
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def _schema_for_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if annotation is Any:
        return _json_value_schema()
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}
    if annotation is Quantity:
        return _schema_for_dataclass(Quantity)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return {"type": "string", "enum": sorted(member.value for member in annotation)}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _schema_for_dataclass(annotation)
    if origin is Literal:
        literals = [value.value if isinstance(value, StrEnum) else value for value in args]
        return {"enum": sorted(literals, key=str)}
    if origin is tuple:
        item_schema = _schema_for_type(args[0]) if args else _json_value_schema()
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": item_schema}
        return {
            "type": "array",
            "prefixItems": [_schema_for_type(arg) for arg in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    if origin is list:
        item_schema = _schema_for_type(args[0]) if args else _json_value_schema()
        return {"type": "array", "items": item_schema}
    if origin is dict:
        value_schema = _schema_for_type(args[1]) if len(args) == 2 else _json_value_schema()
        return {"type": "object", "additionalProperties": value_schema}
    if origin in {UnionType} or str(origin) == "typing.Union":
        return {"anyOf": [_schema_for_type(arg) for arg in args]}
    return _json_value_schema()


def _json_value_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {}},
            {"type": "object", "additionalProperties": {}},
        ]
    }
