"""Bounded EngineeringProject tools."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.changes import (
    AddRequirementOperation,
    ChangeError,
    ChangeResult,
    ChangeSet,
    ChangeSetLoadError,
    OperationKind,
    fingerprint_project,
)
from hephaestus.design import (
    DesignCompilation,
    DesignError,
    WarehouseInspectionRobotIntent,
    compile_warehouse_inspection_robot,
)
from hephaestus.index import ProjectIndex
from hephaestus.mobile_robot import (
    CompiledMobileRobot,
    MobileRobotIntent,
    compile_mobile_robot_design,
)
from hephaestus.model import (
    ACCELERATION,
    DIMENSIONLESS,
    LENGTH,
    MASS,
    SPEED,
    TIME,
    EngineeringProject,
    Entity,
    ModelError,
    Provenance,
    ProvenanceSource,
    Quantity,
    QuantityRange,
    Reference,
    Requirement,
    RequirementKind,
    RequirementPriority,
    RequirementStatus,
    VerificationMethod,
    _to_data,
    coerce_value,
    q,
)
from hephaestus.persistence import load_project, save_project
from hephaestus.schema import changeset_schema
from hephaestus.tools.registry import (
    EmptyArgs,
    PermissionClass,
    RegisteredTool,
    ToolContext,
    ToolRegistry,
)
from hephaestus.validation import validate_project


class LimitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class EntityArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    entity_id: str


class ReferencesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: str
    id: str


class ChangeSetArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    changeset: dict[str, Any]


RequirementKindValue = Literal[
    "payload",
    "dimensions",
    "mass",
    "speed",
    "runtime",
    "range",
    "acceleration",
    "safety_factor",
]
ConstraintValue = Literal["at_least", "at_most", "exactly"]
PriorityValue = Literal["low", "medium", "high", "critical"]
VerificationValue = Literal[
    "inspection", "analysis", "test", "simulation", "demonstration", "unspecified"
]


class AddRequirementArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    statement: str = Field(min_length=1, max_length=500)
    quantity_value: float = Field(ge=0.0, allow_inf_nan=False)
    quantity_unit: str = Field(min_length=1, max_length=40)
    constraint: ConstraintValue
    kind: RequirementKindValue = "payload"
    priority: PriorityValue = RequirementPriority.MEDIUM.value
    verification_method: VerificationValue = VerificationMethod.UNSPECIFIED.value


class WarehouseInspectionRobotArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    robot_family: str = Field(default="differential_drive", min_length=1, max_length=40)
    robot_name: str = Field(min_length=1, max_length=120)
    mission: str = Field(min_length=1, max_length=500)
    warehouse_environment: str = Field(min_length=1, max_length=300)
    payload_kg: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    max_speed_mps: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    chassis_length_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    chassis_width_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    chassis_height_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    wheel_diameter_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    wheel_width_m: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    has_lidar: bool | None = None
    has_imu: bool | None = None


class CreateMobileRobotArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    robot_name: str = Field(min_length=1, max_length=80)
    mission_summary: str = Field(min_length=1, max_length=500)
    robot_family: str = Field(default="differential_drive", min_length=1, max_length=40)
    payload_kg: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    max_speed_mps: float | None = Field(default=None, gt=0.0, le=3.0, allow_inf_nan=False)
    chassis_length_m: float | None = Field(default=None, gt=0.0, le=2.0, allow_inf_nan=False)
    chassis_width_m: float | None = Field(default=None, gt=0.0, le=1.5, allow_inf_nan=False)
    chassis_height_m: float | None = Field(default=None, gt=0.0, le=0.8, allow_inf_nan=False)
    wheel_radius_m: float | None = Field(default=None, gt=0.0, le=0.3, allow_inf_nan=False)
    include_lidar: bool = True
    include_imu: bool = True
    indoor_warehouse: bool = True
    explicitly_supersedes_payload: bool = False


def _add_requirement_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "statement": {
                "type": "string",
                "description": "Human-readable engineering requirement statement.",
            },
            "quantity_value": {
                "type": "number",
                "description": "Finite non-negative numeric requirement value.",
                "minimum": 0,
            },
            "quantity_unit": {
                "type": "string",
                "description": "Physical unit for quantity_value, such as kg, m, s, or m/s.",
            },
            "constraint": {
                "type": "string",
                "enum": ["at_least", "at_most", "exactly"],
                "description": "How the quantity constrains the requirement.",
            },
            "kind": {
                "type": "string",
                "enum": [
                    "payload",
                    "dimensions",
                    "mass",
                    "speed",
                    "runtime",
                    "range",
                    "acceleration",
                    "safety_factor",
                ],
                "description": "Engineering requirement category.",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Requirement priority.",
            },
            "verification_method": {
                "type": "string",
                "enum": [
                    "inspection",
                    "analysis",
                    "test",
                    "simulation",
                    "demonstration",
                    "unspecified",
                ],
                "description": "How the requirement should be verified when known.",
            },
        },
        "required": ["statement", "quantity_value", "quantity_unit", "constraint"],
    }


def _warehouse_inspection_robot_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "robot_family": {
                "type": "string",
                "enum": ["differential_drive", "tracked", "legged", "manipulator"],
                "description": "Robot family. Only differential_drive is currently supported.",
            },
            "robot_name": {"type": "string", "description": "Human-readable robot name."},
            "mission": {
                "type": "string",
                "description": "Warehouse inspection mission the robot should perform.",
            },
            "warehouse_environment": {
                "type": "string",
                "description": "Indoor warehouse environment description.",
            },
            "payload_kg": {
                "type": "number",
                "minimum": 0,
                "description": "Payload capacity in kilograms.",
            },
            "max_speed_mps": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Maximum operating speed in meters per second.",
            },
            "chassis_length_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional chassis length in meters.",
            },
            "chassis_width_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional chassis width in meters.",
            },
            "chassis_height_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional chassis height in meters.",
            },
            "wheel_diameter_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional drive wheel diameter in meters.",
            },
            "wheel_width_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Optional drive wheel width in meters.",
            },
            "has_lidar": {
                "type": "boolean",
                "description": "Whether to include a 2D warehouse safety lidar.",
            },
            "has_imu": {
                "type": "boolean",
                "description": "Whether to include a base IMU.",
            },
        },
        "required": ["robot_name", "mission", "warehouse_environment"],
    }


def _create_mobile_robot_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "robot_name": {"type": "string", "description": "Short robot name."},
            "mission_summary": {
                "type": "string",
                "description": "Natural-language warehouse inspection mission summary.",
            },
            "robot_family": {
                "type": "string",
                "enum": ["differential_drive", "tracked", "legged", "manipulator"],
                "description": "Robot family. Only differential_drive is currently supported.",
            },
            "payload_kg": {"type": "number", "minimum": 0},
            "max_speed_mps": {"type": "number", "exclusiveMinimum": 0, "maximum": 3.0},
            "chassis_length_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 2.0},
            "chassis_width_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 1.5},
            "chassis_height_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.8},
            "wheel_radius_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.3},
            "include_lidar": {"type": "boolean"},
            "include_imu": {"type": "boolean"},
            "indoor_warehouse": {"type": "boolean"},
            "explicitly_supersedes_payload": {"type": "boolean"},
        },
        "required": ["robot_name", "mission_summary"],
    }


def _changeset_tool_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"changeset": changeset_schema()},
        "required": ["changeset"],
    }


def default_project_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (
        RegisteredTool(
            "project.get_summary",
            "Return bounded project metadata and category counts.",
            PermissionClass.READ_ONLY,
            EmptyArgs,
            _get_summary,
        ),
        RegisteredTool(
            "project.list_entities",
            "List project entities with pagination.",
            PermissionClass.READ_ONLY,
            LimitArgs,
            _list_entities,
        ),
        RegisteredTool(
            "project.get_entity",
            "Inspect one entity by stable ID.",
            PermissionClass.READ_ONLY,
            EntityArgs,
            _get_entity,
        ),
        RegisteredTool(
            "project.get_references",
            "Return incoming and outgoing references for one project item.",
            PermissionClass.READ_ONLY,
            ReferencesArgs,
            _get_references,
        ),
        RegisteredTool(
            "project.validate",
            "Run deterministic project validation.",
            PermissionClass.READ_ONLY,
            EmptyArgs,
            _validate,
        ),
        RegisteredTool(
            "project.preview_changeset",
            "Dry-run a serialized ChangeSet without writing.",
            PermissionClass.READ_ONLY,
            ChangeSetArgs,
            _preview_changeset,
            input_schema=_changeset_tool_input_schema(),
            model_facing=False,
        ),
        RegisteredTool(
            "project.apply_changeset",
            "Apply a serialized ChangeSet after application-controlled confirmation.",
            PermissionClass.PROJECT_MUTATION,
            ChangeSetArgs,
            _apply_changeset,
            input_schema=_changeset_tool_input_schema(),
            model_facing=False,
        ),
        RegisteredTool(
            "project_add_requirement",
            "Add one engineering requirement after application-controlled "
            "preview and confirmation.",
            PermissionClass.PROJECT_MUTATION,
            AddRequirementArgs,
            _apply_add_requirement,
            input_schema=_add_requirement_input_schema(),
            preview_handler=_preview_add_requirement,
            operation_family="requirement",
        ),
        RegisteredTool(
            "project_create_mobile_robot",
            "Create one differential-drive warehouse inspection mobile robot from "
            "a natural-language design intent after application-controlled preview "
            "and confirmation.",
            PermissionClass.PROJECT_MUTATION,
            CreateMobileRobotArgs,
            _apply_create_mobile_robot,
            input_schema=_create_mobile_robot_input_schema(),
            preview_handler=_preview_create_mobile_robot,
            operation_family="mobile_robot_design",
        ),
        RegisteredTool(
            "project_design_warehouse_inspection_robot",
            "Design one differential-drive warehouse inspection robot after "
            "application-controlled preview and confirmation.",
            PermissionClass.PROJECT_MUTATION,
            WarehouseInspectionRobotArgs,
            _apply_warehouse_inspection_robot,
            input_schema=_warehouse_inspection_robot_input_schema(),
            model_facing=False,
            preview_handler=_preview_warehouse_inspection_robot,
            operation_family="legacy_mobile_robot_design",
        ),
    ):
        registry.register(tool)
    return registry


def _project(context: ToolContext):
    return load_project(context.require_project_path())


def _get_summary(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    project = _project(context)
    return {
        "name": project.metadata.name,
        "schema_version": project.schema_version,
        "revision": project.revision,
        "counts": project.collection_counts(),
        "requirements": len(project.requirements),
        "assumptions": len(project.assumptions),
    }


def _list_entities(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = args if isinstance(args, LimitArgs) else LimitArgs.model_validate(args)
    entities = _project(context).entities
    selected = entities[parsed.offset : parsed.offset + parsed.limit]
    return {
        "items": [_entity_summary(entity) for entity in selected],
        "offset": parsed.offset,
        "limit": parsed.limit,
        "total": len(entities),
        "truncated": parsed.offset + parsed.limit < len(entities),
    }


def _get_entity(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = args if isinstance(args, EntityArgs) else EntityArgs.model_validate(args)
    for entity in _project(context).entities:
        if entity.id == parsed.entity_id:
            return _to_data(entity)
    raise ValueError(f"entity not found: {parsed.entity_id}")


def _get_references(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = args if isinstance(args, ReferencesArgs) else ReferencesArgs.model_validate(args)
    project = _project(context)
    index = ProjectIndex.build(project)
    ref = Reference(parsed.kind, parsed.id)
    item = index.resolve(ref)
    if item is None:
        raise ValueError(f"item not found: {parsed.kind} {parsed.id}")
    incoming = [edge.source_path for edge in index.incoming_references(parsed.kind, parsed.id)]
    outgoing = [
        {"path": edge.source_path, "target": edge.target.to_dict()}
        for edge in index.outgoing_references(item.path)
    ]
    return {"item": item.path, "incoming": incoming, "outgoing": outgoing}


def _validate(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    return validate_project(_project(context)).to_dict()


def _preview_changeset(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = args if isinstance(args, ChangeSetArgs) else ChangeSetArgs.model_validate(args)
    changeset = _changeset(parsed.changeset)
    project = _project(context)
    result = changeset.apply(project, dry_run=True)
    return result.to_dict()


def _preview_add_requirement(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args if isinstance(args, AddRequirementArgs) else AddRequirementArgs.model_validate(args)
    )
    project = _project(context)
    changeset = _add_requirement_changeset(parsed, project)
    result = changeset.apply(project, dry_run=True)
    return _add_requirement_result(changeset, result.to_dict(), parsed)


def _preview_create_mobile_robot(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args
        if isinstance(args, CreateMobileRobotArgs)
        else CreateMobileRobotArgs.model_validate(args)
    )
    project = _project(context)
    compiled = _compile_create_mobile_robot(project, parsed, dry_run=True)
    return _mobile_robot_result(compiled, dry_run=True)


def _preview_warehouse_inspection_robot(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args
        if isinstance(args, WarehouseInspectionRobotArgs)
        else WarehouseInspectionRobotArgs.model_validate(args)
    )
    project = _project(context)
    compilation = _compile_warehouse_inspection_robot(project, parsed)
    return _design_compilation_result(compilation, dry_run=True)


def _apply_changeset(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = args if isinstance(args, ChangeSetArgs) else ChangeSetArgs.model_validate(args)
    project_path, project = _load_project_for_mutation(context)
    changeset = _changeset(parsed.changeset)
    result = changeset.apply(project, dry_run=False)
    _save_change_result_if_current(result, project_path)
    return result.to_dict()


def _apply_add_requirement(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args if isinstance(args, AddRequirementArgs) else AddRequirementArgs.model_validate(args)
    )
    project_path, project = _load_project_for_mutation(context)
    changeset = _add_requirement_changeset(parsed, project)
    result = changeset.apply(project, dry_run=False)
    _save_change_result_if_current(result, project_path)
    return _add_requirement_result(changeset, result.to_dict(), parsed)


def _apply_create_mobile_robot(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args
        if isinstance(args, CreateMobileRobotArgs)
        else CreateMobileRobotArgs.model_validate(args)
    )
    project_path, project = _load_project_for_mutation(context)
    compiled = _compile_create_mobile_robot(project, parsed, dry_run=False)
    _save_mobile_robot_if_current(compiled, project_path)
    return _mobile_robot_result(compiled, dry_run=False)


def _apply_warehouse_inspection_robot(args: BaseModel, context: ToolContext) -> dict[str, Any]:
    parsed = (
        args
        if isinstance(args, WarehouseInspectionRobotArgs)
        else WarehouseInspectionRobotArgs.model_validate(args)
    )
    project_path, project = _load_project_for_mutation(context)
    compilation = _compile_warehouse_inspection_robot(project, parsed)
    _save_compiled_design_if_current(compilation, project_path)
    return _design_compilation_result(compilation, dry_run=False)


def _load_project_for_mutation(context: ToolContext) -> tuple[Path, EngineeringProject]:
    project_path = context.require_project_path()
    project = load_project(project_path)
    if (
        context.expected_fingerprint is not None
        and fingerprint_project(project) != context.expected_fingerprint
    ):
        raise ValueError("active project changed after preview")
    return project_path, project


def _save_change_result_if_current(result: ChangeResult, project_path: Path) -> None:
    current = load_project(project_path)
    if fingerprint_project(current) != result.before_fingerprint:
        raise ValueError("active project changed after ChangeSet evaluation")
    save_project(result.project, project_path, overwrite=True)


def _save_mobile_robot_if_current(result: CompiledMobileRobot, project_path: Path) -> None:
    current = load_project(project_path)
    if fingerprint_project(current) != result.before_fingerprint:
        raise ValueError("active project changed after design evaluation")
    if not result.validation.is_valid:
        codes = ", ".join(finding.code for finding in result.validation.findings)
        raise ValueError(f"compiled mobile robot design is invalid: {codes}")
    save_project(result.project, project_path, overwrite=True)


def _save_compiled_design_if_current(result: DesignCompilation, project_path: Path) -> None:
    current = load_project(project_path)
    if fingerprint_project(current) != result.before_fingerprint:
        raise ValueError("active project changed after design evaluation")
    if not result.validation.is_valid:
        codes = ", ".join(finding.code for finding in result.validation.findings)
        raise ValueError(f"compiled design is invalid: {codes}")
    save_project(result.project, project_path, overwrite=True)


def _changeset(data: dict[str, Any]) -> ChangeSet:
    try:
        return ChangeSet.from_dict(data)
    except (ChangeSetLoadError, ChangeError) as exc:
        raise ValueError(str(exc)) from exc


def _compile_create_mobile_robot(
    project: EngineeringProject, args: CreateMobileRobotArgs, *, dry_run: bool
) -> CompiledMobileRobot:
    if args.robot_family != "differential_drive":
        raise ValueError(
            f"unsupported robot family: {args.robot_family}; supported family is differential_drive"
        )
    intent = MobileRobotIntent(
        robot_name=args.robot_name,
        mission_summary=args.mission_summary,
        robot_family="differential_drive",
        payload_kg=args.payload_kg,
        max_speed_mps=args.max_speed_mps,
        chassis_length_m=args.chassis_length_m,
        chassis_width_m=args.chassis_width_m,
        chassis_height_m=args.chassis_height_m,
        wheel_radius_m=args.wheel_radius_m,
        include_lidar=args.include_lidar,
        include_imu=args.include_imu,
        indoor_warehouse=args.indoor_warehouse,
        explicitly_supersedes_payload=args.explicitly_supersedes_payload,
    )
    return compile_mobile_robot_design(project, intent, dry_run=dry_run)


def _mobile_robot_result(compiled: CompiledMobileRobot, *, dry_run: bool) -> dict[str, Any]:
    result = compiled.to_result(dry_run=dry_run).to_dict()
    result["operation"] = "create_mobile_robot"
    result["design_preview"] = compiled.preview
    result["human_preview"] = {
        "title": compiled.preview["summary"],
        "statement": compiled.preview["robot"]["mission"],
        "current_revision": compiled.preview["current_revision"],
        "proposed_revision": compiled.preview["proposed_revision"],
        "validation": "passed" if compiled.validation.is_valid else "has findings",
    }
    return result


def _compile_warehouse_inspection_robot(
    project: EngineeringProject, args: WarehouseInspectionRobotArgs
) -> DesignCompilation:
    try:
        return compile_warehouse_inspection_robot(
            project,
            WarehouseInspectionRobotIntent(
                robot_family=args.robot_family,
                robot_name=args.robot_name,
                mission=args.mission,
                warehouse_environment=args.warehouse_environment,
                payload_kg=args.payload_kg,
                max_speed_mps=args.max_speed_mps,
                chassis_length_m=args.chassis_length_m,
                chassis_width_m=args.chassis_width_m,
                chassis_height_m=args.chassis_height_m,
                wheel_diameter_m=args.wheel_diameter_m,
                wheel_width_m=args.wheel_width_m,
                has_lidar=args.has_lidar,
                has_imu=args.has_imu,
            ),
        )
    except DesignError as exc:
        raise ValueError(str(exc)) from exc


def _design_compilation_result(compilation: DesignCompilation, *, dry_run: bool) -> dict[str, Any]:
    return compilation.to_dict(dry_run=dry_run)


def _add_requirement_changeset(args: AddRequirementArgs, project: EngineeringProject) -> ChangeSet:
    detail = "AI engineering operation: add requirement"
    try:
        requirement = Requirement(
            id=_next_requirement_id(project, args.statement),
            kind=RequirementKind(args.kind),
            statement=args.statement.strip(),
            priority=RequirementPriority(args.priority),
            verification_method=VerificationMethod(args.verification_method),
            status=RequirementStatus.PROPOSED,
            value=_requirement_value_from_args(args),
            provenance=Provenance(ProvenanceSource.GENERATED, detail),
        )
        return ChangeSet(
            _add_requirement_change_id(requirement.id),
            project.revision,
            f"Add requirement: {requirement.statement}",
            Provenance(ProvenanceSource.GENERATED, detail),
            (AddRequirementOperation(OperationKind.ADD_REQUIREMENT, requirement),),
        )
    except (ChangeError, ModelError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def _add_requirement_result(
    changeset: ChangeSet, result: dict[str, Any], args: AddRequirementArgs
) -> dict[str, Any]:
    operation = changeset.operations[0]
    if not isinstance(operation, AddRequirementOperation):
        return result
    requirement = operation.requirement
    result["operation"] = "add_requirement"
    result["requirement"] = {
        "id": requirement.id,
        "statement": requirement.statement,
        "kind": requirement.kind.value,
        "priority": requirement.priority.value,
        "constraint": args.constraint,
        "value": _to_data(requirement.value),
    }
    validation = result.get("validation")
    validation_status = "unknown"
    if isinstance(validation, dict):
        validation_status = "passed" if validation.get("status") == "valid" else "has findings"
    result["human_preview"] = {
        "title": "Add requirement",
        "statement": requirement.statement,
        "current_revision": changeset.expected_revision,
        "proposed_revision": result.get("revision"),
        "validation": validation_status,
    }
    return result


def _requirement_value(value: dict[str, Any] | str | None) -> Quantity | QuantityRange | str | None:
    if value is None or isinstance(value, str):
        return value
    coerced = coerce_value(value)
    if isinstance(coerced, Quantity | QuantityRange):
        return coerced
    raise ValueError("requirement value must be a string, quantity, quantity range, or null")


def _requirement_value_from_args(args: AddRequirementArgs) -> Quantity | QuantityRange:
    dimension = _requirement_dimension(args.kind)
    quantity = q(args.quantity_value, args.quantity_unit.strip(), dimension)
    if args.constraint == "exactly":
        return quantity
    if args.constraint == "at_least":
        return QuantityRange(minimum=quantity)
    if args.constraint == "at_most":
        return QuantityRange(maximum=quantity)
    raise ValueError(f"unsupported requirement constraint: {args.constraint}")


def _requirement_dimension(kind: str):
    mapping = {
        RequirementKind.PAYLOAD.value: MASS,
        RequirementKind.MASS.value: MASS,
        RequirementKind.DIMENSIONS.value: LENGTH,
        RequirementKind.RANGE.value: LENGTH,
        RequirementKind.SPEED.value: SPEED,
        RequirementKind.ACCELERATION.value: ACCELERATION,
        RequirementKind.RUNTIME.value: TIME,
        RequirementKind.SAFETY_FACTOR.value: DIMENSIONLESS,
    }
    if kind not in mapping:
        raise ValueError(f"quantity-bearing requirement kind is not supported yet: {kind}")
    return mapping[kind]


def _next_requirement_id(project: EngineeringProject, statement: str) -> str:
    existing = {requirement.id for requirement in project.requirements}
    slug = re.sub(r"[^a-z0-9_]+", "_", statement.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_") or "requirement"
    base = f"requirement:{slug[:70]}"
    candidate = base
    index = 2
    while candidate in existing:
        suffix = f"_{index}"
        candidate = f"{base[: 96 - len(suffix)]}{suffix}"
        index += 1
    return candidate


def _add_requirement_change_id(requirement_id: str) -> str:
    digest = hashlib.sha256(requirement_id.encode("utf-8")).hexdigest()[:16]
    return f"change:add_requirement_{digest}"


def _entity_summary(entity: Entity) -> dict[str, Any]:
    return {
        "id": entity.id,
        "name": entity.name,
        "kind": entity.kind.value,
        "frame_ref": entity.frame_ref.to_dict() if entity.frame_ref else None,
    }
