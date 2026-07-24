"""Experimental bounded concept robot specification and compiler."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hephaestus.changes import fingerprint_project
from hephaestus.model import (
    Actuator,
    ActuatorKind,
    Assumption,
    Body,
    EngineeringDecision,
    EngineeringProject,
    Entity,
    EntityKind,
    FieldOfView,
    Frame,
    Geometry,
    GeometryBinding,
    GeometryKind,
    GeometryRole,
    Joint,
    JointKind,
    MassProperties,
    MassProvenance,
    Material,
    NoiseModel,
    Pose,
    Provenance,
    ProvenanceSource,
    QuantityRange,
    Reference,
    Requirement,
    RequirementKind,
    RequirementPriority,
    RequirementStatus,
    Sensor,
    SensorModality,
    SimulationScenario,
    SoftwareComponent,
    Transform,
    Vector3,
    VerificationMethod,
    q,
)
from hephaestus.validation import (
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    validate_project,
)

CONCEPT_EXTENSION_KEY = "hephaestus:concept_robot:v1"
CONCEPT_GENERATOR_VERSION = "hephaestus.concept_robot.compiler.v1"
CONCEPT_MARKER_KEY = "hephaestus:concept_robot:semantic_key"
MAX_BODIES = 20
MAX_JOINTS = 24
MAX_SENSORS = 8
MAX_ACTUATED_JOINTS = 12
MAX_DIMENSION_M = 5.0
MAX_MASS_KG = 250.0


class ConceptError(ValueError):
    """Raised when a concept robot spec cannot be compiled."""


class LocomotionMode(StrEnum):
    DIFFERENTIAL_DRIVE = "differential_drive"
    SKID_STEER = "skid_steer"
    STATIC = "static"


class PrimitiveType(StrEnum):
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"


class ConceptJointType(StrEnum):
    FIXED = "fixed"
    CONTINUOUS = "continuous"
    REVOLUTE = "revolute"


class ConceptSensorType(StrEnum):
    LIDAR_2D = "lidar_2d"
    IMU = "imu"


class ConceptWorld(StrEnum):
    ENGINEERING_LAB = "engineering_lab"
    WAREHOUSE = "warehouse"
    RAMP_COURSE = "ramp_course"
    ROUGH_TERRAIN = "rough_terrain"


class WheelRole(StrEnum):
    NONE = "none"
    LEFT_DRIVE = "left_drive"
    RIGHT_DRIVE = "right_drive"
    PASSIVE = "passive"


class FunctionalRole(StrEnum):
    CHASSIS = "chassis"
    DRIVE_WHEEL = "drive_wheel"
    PASSIVE_WHEEL = "passive_wheel"
    SENSOR_MOUNT = "sensor_mount"
    SENSOR_HOUSING = "sensor_housing"
    DECORATIVE = "decorative"
    ARM = "arm"
    PAYLOAD = "payload"
    OTHER = "other"


class ConceptPose(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    x_m: float = Field(allow_inf_nan=False)
    y_m: float = Field(allow_inf_nan=False)
    z_m: float = Field(allow_inf_nan=False)
    roll_rad: float = Field(default=0.0, allow_inf_nan=False)
    pitch_rad: float = Field(default=0.0, allow_inf_nan=False)
    yaw_rad: float = Field(default=0.0, allow_inf_nan=False)


class ConceptColor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    r: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    g: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    b: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    a: float = Field(default=1.0, ge=0.0, le=1.0, allow_inf_nan=False)

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)


class ConceptBodySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    semantic_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1, max_length=80)
    primitive_type: PrimitiveType
    size_x_m: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION_M, allow_inf_nan=False)
    size_y_m: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION_M, allow_inf_nan=False)
    size_z_m: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION_M, allow_inf_nan=False)
    radius_m: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION_M, allow_inf_nan=False)
    length_m: float = Field(default=0.0, ge=0.0, le=MAX_DIMENSION_M, allow_inf_nan=False)
    pose: ConceptPose
    mass_kg: float = Field(gt=0.0, le=MAX_MASS_KG, allow_inf_nan=False)
    visual_color: ConceptColor
    functional_role: FunctionalRole

    @model_validator(mode="after")
    def _valid_dimensions(self) -> ConceptBodySpec:
        if self.primitive_type is PrimitiveType.BOX:
            if self.size_x_m <= 0 or self.size_y_m <= 0 or self.size_z_m <= 0:
                raise ValueError("box bodies require positive size_x_m, size_y_m, size_z_m")
        elif self.primitive_type is PrimitiveType.CYLINDER:
            if self.radius_m <= 0 or self.length_m <= 0:
                raise ValueError("cylinder bodies require positive radius_m and length_m")
        elif self.primitive_type is PrimitiveType.SPHERE and self.radius_m <= 0:
            raise ValueError("sphere bodies require positive radius_m")
        return self


class ConceptJointLimit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    has_limits: bool = False
    lower_rad: float = Field(default=0.0, allow_inf_nan=False)
    upper_rad: float = Field(default=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _valid_limits(self) -> ConceptJointLimit:
        if self.has_limits and self.lower_rad >= self.upper_rad:
            raise ValueError("joint lower_rad must be smaller than upper_rad")
        return self


class ConceptJointSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    semantic_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    parent_body_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    child_body_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    joint_type: ConceptJointType
    origin: ConceptPose
    axis_x: float = Field(default=0.0, allow_inf_nan=False)
    axis_y: float = Field(default=0.0, allow_inf_nan=False)
    axis_z: float = Field(default=1.0, allow_inf_nan=False)
    limits: ConceptJointLimit = Field(default_factory=ConceptJointLimit)
    wheel_role: WheelRole = WheelRole.NONE

    @model_validator(mode="after")
    def _valid_axis(self) -> ConceptJointSpec:
        norm = math.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
        if self.joint_type is not ConceptJointType.FIXED and norm <= 0.0:
            raise ValueError("non-fixed joints require a nonzero axis")
        if self.parent_body_key == self.child_body_key:
            raise ValueError("joint parent and child must differ")
        return self

    def normalized_axis(self) -> tuple[float, float, float] | None:
        if self.joint_type is ConceptJointType.FIXED:
            return None
        norm = math.sqrt(self.axis_x**2 + self.axis_y**2 + self.axis_z**2)
        return (self.axis_x / norm, self.axis_y / norm, self.axis_z / norm)


class ConceptSensorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    semantic_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    sensor_type: ConceptSensorType
    parent_body_key: str = Field(min_length=2, max_length=48, pattern=r"^[a-z][a-z0-9_]*$")
    pose: ConceptPose
    update_rate_hz: float = Field(gt=0.0, le=500.0, allow_inf_nan=False)
    topic: str = Field(min_length=1, max_length=64)

    @field_validator("topic")
    @classmethod
    def _safe_topic(cls, value: str) -> str:
        if not re.fullmatch(r"/[a-z][a-z0-9_/]*", value):
            raise ValueError("sensor topic must be an absolute lowercase ROS topic")
        return value


class ConceptRobotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    spec_version: Literal["hephaestus.concept_robot.v1"]
    robot_name: str = Field(min_length=1, max_length=80)
    design_summary: str = Field(min_length=1, max_length=500)
    locomotion_mode: LocomotionMode
    bodies: tuple[ConceptBodySpec, ...] = Field(min_length=1, max_length=MAX_BODIES)
    joints: tuple[ConceptJointSpec, ...] = Field(default=(), max_length=MAX_JOINTS)
    sensors: tuple[ConceptSensorSpec, ...] = Field(default=(), max_length=MAX_SENSORS)
    world_template: ConceptWorld
    assumptions: tuple[str, ...] = Field(default=(), max_length=16)
    engineering_warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def _semantic_validation(self) -> ConceptRobotSpec:
        _reject_duplicate([body.semantic_key for body in self.bodies], "body semantic key")
        _reject_duplicate([joint.semantic_key for joint in self.joints], "joint semantic key")
        _reject_duplicate([sensor.semantic_key for sensor in self.sensors], "sensor semantic key")
        body_keys = {body.semantic_key for body in self.bodies}
        for joint in self.joints:
            if joint.parent_body_key not in body_keys or joint.child_body_key not in body_keys:
                raise ValueError("joint references an unknown body")
        for sensor in self.sensors:
            if sensor.parent_body_key not in body_keys:
                raise ValueError("sensor references an unknown body")
        children = [joint.child_body_key for joint in self.joints]
        _reject_duplicate(children, "joint child body")
        roots = body_keys - set(children)
        if len(roots) != 1:
            raise ValueError("concept requires exactly one root body")
        if _has_cycle(self.joints):
            raise ValueError("body tree must not contain cycles")
        actuated = [
            j for j in self.joints if j.wheel_role in {WheelRole.LEFT_DRIVE, WheelRole.RIGHT_DRIVE}
        ]
        if len(actuated) > MAX_ACTUATED_JOINTS:
            raise ValueError("too many actuated joints")
        if self.locomotion_mode in {LocomotionMode.DIFFERENTIAL_DRIVE, LocomotionMode.SKID_STEER}:
            if not any(j.wheel_role is WheelRole.LEFT_DRIVE for j in self.joints):
                raise ValueError("wheeled locomotion requires a left drive wheel joint")
            if not any(j.wheel_role is WheelRole.RIGHT_DRIVE for j in self.joints):
                raise ValueError("wheeled locomotion requires a right drive wheel joint")
        return self

    def root_body_key(self) -> str:
        children = {joint.child_body_key for joint in self.joints}
        return next(body.semantic_key for body in self.bodies if body.semantic_key not in children)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class CompiledConceptRobot:
    project: EngineeringProject
    spec: ConceptRobotSpec
    warnings: tuple[str, ...]
    validation: ValidationReport
    before_fingerprint: str
    after_fingerprint: str
    semantic_diff: tuple[str, ...] = ()

    def preview(self, *, current_revision: int) -> dict[str, Any]:
        return {
            "schema": "hephaestus.concept.preview.v1",
            "summary": f"Create concept robot {self.spec.robot_name}",
            "robot_name": self.spec.robot_name,
            "locomotion_mode": self.spec.locomotion_mode.value,
            "world_template": self.spec.world_template.value,
            "entity_counts": {
                "bodies": len(self.spec.bodies),
                "joints": len(self.spec.joints),
                "sensors": len(self.spec.sensors),
                "actuated_joints": len(_drive_joints(self.spec)),
            },
            "assumptions": list(self.spec.assumptions),
            "warnings": list(self.warnings),
            "semantic_diff": list(self.semantic_diff),
            "artifact_consequences": [
                "parametric CadQuery source",
                "STEP/STL primitive components",
                "URDF/Xacro robot description",
                "Gazebo SDF robot and deterministic world",
                "ROS 2 description, simulation and bringup packages",
                "ros2_control diff_drive_controller when drive wheels are present",
            ],
            "current_revision": current_revision,
            "proposed_revision": self.project.revision,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "validation": self.validation.to_dict(),
        }


def concept_robot_schema() -> dict[str, Any]:
    body = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "display_name": {"type": "string", "minLength": 1, "maxLength": 80},
            "primitive_type": {"type": "string", "enum": ["box", "cylinder", "sphere"]},
            "size_x_m": {"type": "number"},
            "size_y_m": {"type": "number"},
            "size_z_m": {"type": "number"},
            "radius_m": {"type": "number"},
            "length_m": {"type": "number"},
            "pose": _pose_schema(),
            "mass_kg": {"type": "number"},
            "visual_color": _color_schema(),
            "functional_role": {"type": "string", "enum": [item.value for item in FunctionalRole]},
        },
        "required": [
            "semantic_key",
            "display_name",
            "primitive_type",
            "size_x_m",
            "size_y_m",
            "size_z_m",
            "radius_m",
            "length_m",
            "pose",
            "mass_kg",
            "visual_color",
            "functional_role",
        ],
    }
    joint = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "parent_body_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "child_body_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "joint_type": {"type": "string", "enum": ["fixed", "continuous", "revolute"]},
            "origin": _pose_schema(),
            "axis_x": {"type": "number"},
            "axis_y": {"type": "number"},
            "axis_z": {"type": "number"},
            "limits": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "has_limits": {"type": "boolean"},
                    "lower_rad": {"type": "number"},
                    "upper_rad": {"type": "number"},
                },
                "required": ["has_limits", "lower_rad", "upper_rad"],
            },
            "wheel_role": {"type": "string", "enum": [item.value for item in WheelRole]},
        },
        "required": [
            "semantic_key",
            "parent_body_key",
            "child_body_key",
            "joint_type",
            "origin",
            "axis_x",
            "axis_y",
            "axis_z",
            "limits",
            "wheel_role",
        ],
    }
    sensor = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "sensor_type": {"type": "string", "enum": ["lidar_2d", "imu"]},
            "parent_body_key": {
                "type": "string",
                "minLength": 2,
                "maxLength": 48,
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "pose": _pose_schema(),
            "update_rate_hz": {"type": "number"},
            "topic": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
                "pattern": "^/[a-z][a-z0-9_/]*$",
            },
        },
        "required": [
            "semantic_key",
            "sensor_type",
            "parent_body_key",
            "pose",
            "update_rate_hz",
            "topic",
        ],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "spec_version": {"type": "string", "enum": ["hephaestus.concept_robot.v1"]},
            "robot_name": {"type": "string", "minLength": 1, "maxLength": 80},
            "design_summary": {"type": "string", "minLength": 1, "maxLength": 500},
            "locomotion_mode": {"type": "string", "enum": [item.value for item in LocomotionMode]},
            "bodies": {"type": "array", "items": body, "maxItems": MAX_BODIES},
            "joints": {"type": "array", "items": joint, "maxItems": MAX_JOINTS},
            "sensors": {"type": "array", "items": sensor, "maxItems": MAX_SENSORS},
            "world_template": {"type": "string", "enum": [item.value for item in ConceptWorld]},
            "assumptions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 16,
            },
            "engineering_warnings": {
                "type": "array",
                "items": {"type": "string", "maxLength": 240},
                "maxItems": 16,
            },
        },
        "required": [
            "spec_version",
            "robot_name",
            "design_summary",
            "locomotion_mode",
            "bodies",
            "joints",
            "sensors",
            "world_template",
            "assumptions",
            "engineering_warnings",
        ],
    }


def parse_concept_robot_spec(data: dict[str, Any]) -> ConceptRobotSpec:
    try:
        return ConceptRobotSpec.model_validate(data, strict=False)
    except ValidationError as exc:
        raise ConceptError("invalid concept robot specification") from exc


def compile_concept_robot_design(
    project: EngineeringProject,
    spec: ConceptRobotSpec,
    *,
    previous_spec: ConceptRobotSpec | None = None,
) -> CompiledConceptRobot:
    warnings = tuple(dict.fromkeys((*spec.engineering_warnings, *_engineering_warnings(spec))))
    base = _strip_existing_concept(project)
    slug = _slug(project.id)
    ids = semantic_id_map(spec, project_scope=slug)
    _reject_existing_ids(base, ids)
    provenance = Provenance(ProvenanceSource.GENERATED, "Milestone 3.1B concept robot compiler")
    canonical = _canonical_collections(spec, ids, provenance)
    extension = {
        "schema": "hephaestus.concept_robot.v1",
        "generator": CONCEPT_GENERATOR_VERSION,
        "source_spec": spec.to_payload(),
        "warnings": list(warnings),
        "ids": ids,
        "coordinate_convention": "right-handed; X forward, Y left, Z up; SI units",
    }
    draft = replace(
        base,
        metadata=replace(base.metadata, description=spec.design_summary.strip()),
        revision=project.revision + 1,
        requirements=(*base.requirements, *canonical["requirements"]),
        assumptions=(*base.assumptions, *canonical["assumptions"]),
        entities=(*base.entities, *canonical["entities"]),
        frames=(*base.frames, *canonical["frames"]),
        transforms=(*base.transforms, *canonical["transforms"]),
        materials=(*base.materials, *canonical["materials"]),
        geometries=(*base.geometries, *canonical["geometries"]),
        bodies=(*base.bodies, *canonical["bodies"]),
        joints=(*base.joints, *canonical["joints"]),
        actuators=(*base.actuators, *canonical["actuators"]),
        sensors=(*base.sensors, *canonical["sensors"]),
        software_components=(*base.software_components, *canonical["software_components"]),
        simulation_scenarios=(*base.simulation_scenarios, *canonical["simulation_scenarios"]),
        engineering_decisions=(*base.engineering_decisions, *canonical["decisions"]),
        extensions={**base.extensions, CONCEPT_EXTENSION_KEY: extension},
    )
    validation = ValidationReport(
        (*validate_project(draft).findings, *validate_concept_project(draft))
    )
    if not validation.is_valid:
        codes = ", ".join(finding.code for finding in validation.findings)
        raise ConceptError(f"compiled concept robot is invalid: {codes}")
    return CompiledConceptRobot(
        project=draft,
        spec=spec,
        warnings=warnings,
        validation=validation,
        before_fingerprint=fingerprint_project(project),
        after_fingerprint=fingerprint_project(draft),
        semantic_diff=_semantic_diff(previous_spec, spec),
    )


def concept_robot_spec(project: EngineeringProject) -> ConceptRobotSpec:
    extension = project.extensions.get(CONCEPT_EXTENSION_KEY)
    if not isinstance(extension, dict) or not isinstance(extension.get("source_spec"), dict):
        raise ConceptError("project does not contain a concept robot design")
    return parse_concept_robot_spec(extension["source_spec"])


def concept_extension(project: EngineeringProject) -> dict[str, Any] | None:
    value = project.extensions.get(CONCEPT_EXTENSION_KEY)
    return value if isinstance(value, dict) else None


def semantic_id_map(spec: ConceptRobotSpec, *, project_scope: str) -> dict[str, Any]:
    prefix = f"{project_scope}_concept"
    bodies: dict[str, dict[str, str]] = {}
    for body in spec.bodies:
        key = body.semantic_key
        bodies[key] = {
            "entity": f"entity:{prefix}_{key}",
            "body": f"body:{prefix}_{key}",
            "frame": f"frame:{prefix}_{key}",
            "geometry": f"geometry:{prefix}_{key}",
            "material": f"material:{prefix}_{key}",
        }
    joints = {j.semantic_key: f"joint:{prefix}_{j.semantic_key}" for j in spec.joints}
    sensors = {s.semantic_key: f"sensor:{prefix}_{s.semantic_key}" for s in spec.sensors}
    actuators = {j.semantic_key: f"actuator:{prefix}_{j.semantic_key}" for j in _drive_joints(spec)}
    return {
        "prefix": prefix,
        "robot_entity": f"entity:{prefix}_robot",
        "base_footprint": f"frame:{prefix}_base_footprint",
        "bodies": bodies,
        "joints": joints,
        "sensors": sensors,
        "actuators": actuators,
        "scenario": f"scenario:{prefix}_{spec.world_template.value}",
        "controller": f"software:{prefix}_diff_drive_controller",
        "bridge": f"software:{prefix}_ros_gz_bridge",
        "mission_requirement": f"requirement:{prefix}_mission",
        "passive_joint_decision": f"decision:{prefix}_passive_joints",
    }


def validate_concept_project(project: EngineeringProject) -> tuple[ValidationFinding, ...]:
    try:
        spec = concept_robot_spec(project)
    except ConceptError:
        return ()
    findings: list[ValidationFinding] = []
    for body in spec.bodies:
        try:
            primitive_inertia(body)
        except ConceptError as exc:
            findings.append(
                ValidationFinding(
                    "concept.inertia.invalid",
                    ValidationSeverity.ERROR,
                    str(exc),
                    f"extensions.{CONCEPT_EXTENSION_KEY}.source_spec.bodies.{body.semantic_key}",
                )
            )
    if spec.locomotion_mode is LocomotionMode.STATIC and _drive_joints(spec):
        findings.append(
            ValidationFinding(
                "concept.static.has_drive_joints",
                ValidationSeverity.ERROR,
                "Static concepts must not define drive wheel joints.",
            )
        )
    return tuple(findings)


def primitive_inertia(body: ConceptBodySpec) -> tuple[tuple[float, float, float], ...]:
    mass = body.mass_kg
    if body.primitive_type is PrimitiveType.BOX:
        x, y, z = body.size_x_m, body.size_y_m, body.size_z_m
        diag = (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )
    elif body.primitive_type is PrimitiveType.CYLINDER:
        r, length = body.radius_m, body.length_m
        diag = (
            mass * (3 * r * r + length * length) / 12.0,
            0.5 * mass * r * r,
            mass * (3 * r * r + length * length) / 12.0,
        )
    else:
        value = 0.4 * mass * body.radius_m * body.radius_m
        diag = (value, value, value)
    if any(not math.isfinite(v) or v <= 0.0 for v in diag):
        raise ConceptError("primitive inertia must have positive finite diagonal terms")
    return ((diag[0], 0.0, 0.0), (0.0, diag[1], 0.0), (0.0, 0.0, diag[2]))


def concept_backend_spec(project: EngineeringProject) -> dict[str, Any]:
    spec = concept_robot_spec(project)
    extension = concept_extension(project) or {}
    return {
        "spec": spec.to_payload(),
        "ids": extension.get("ids", {}),
        "warnings": extension.get("warnings", []),
    }


def _pose_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            name: {"type": "number"}
            for name in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")
        },
        "required": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"],
    }


def _color_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: {"type": "number"} for name in ("r", "g", "b", "a")},
        "required": ["r", "g", "b", "a"],
    }


def _canonical_collections(
    spec: ConceptRobotSpec, ids: dict[str, Any], provenance: Provenance
) -> dict[str, tuple[Any, ...]]:
    root = spec.root_body_key()
    robot_entity = ids["robot_entity"]
    base_footprint = ids["base_footprint"]
    body_ids = ids["bodies"]
    entities = [
        Entity(
            robot_entity,
            spec.robot_name,
            EntityKind.ASSEMBLY,
            frame_ref=_fr(base_footprint),
            extensions=_mark("robot"),
        )
    ]
    frames = [Frame(base_footprint, "base footprint", _er(robot_entity))]
    transforms = []
    materials = []
    geometries = []
    bodies = []
    joints = []
    actuators = []
    sensors = []
    for body in spec.bodies:
        bid = body_ids[body.semantic_key]
        entities.append(
            Entity(
                bid["entity"],
                body.display_name,
                EntityKind.BODY,
                _er(robot_entity),
                _fr(bid["frame"]),
                extensions=_mark(body.semantic_key),
            )
        )
        frames.append(Frame(bid["frame"], body.display_name, _er(bid["entity"])))
        parent_frame = (
            base_footprint
            if body.semantic_key == root
            else body_ids[_parent_for(spec, body.semantic_key)]["frame"]
        )
        transforms.append(
            _transform(
                f"transform:{ids['prefix']}_{body.semantic_key}",
                parent_frame,
                bid["frame"],
                body.pose,
                provenance,
            )
        )
        materials.append(
            Material(
                bid["material"],
                f"{body.display_name} material",
                provenance=provenance,
                extensions={
                    **_mark(body.semantic_key),
                    "color_rgba": list(body.visual_color.tuple()),
                },
            )
        )
        geometries.append(_geometry(body, bid["geometry"], bid["material"], provenance))
        bodies.append(
            Body(
                bid["body"],
                _er(bid["entity"]),
                _fr(bid["frame"]),
                (_gb(bid["geometry"]),),
                (_gb(bid["geometry"], GeometryRole.COLLISION),),
                _mass_properties(body),
                extensions={
                    **_mark(body.semantic_key),
                    "functional_role": body.functional_role.value,
                },
            )
        )
    for joint in spec.joints:
        kind = (
            JointKind.FIXED
            if joint.joint_type is ConceptJointType.FIXED
            else JointKind.CONTINUOUS
            if joint.joint_type is ConceptJointType.CONTINUOUS
            else JointKind.REVOLUTE
        )
        limits = {}
        if joint.limits.has_limits:
            limits = {
                "lower": q(joint.limits.lower_rad, "rad", "angle"),
                "upper": q(joint.limits.upper_rad, "rad", "angle"),
            }
        joints.append(
            Joint(
                ids["joints"][joint.semantic_key],
                joint.semantic_key,
                kind,
                _br(body_ids[joint.parent_body_key]["body"]),
                _br(body_ids[joint.child_body_key]["body"]),
                joint.normalized_axis(),
                limits,
                provenance,
                {**_mark(joint.semantic_key), "wheel_role": joint.wheel_role.value},
            )
        )
        if joint.wheel_role in {WheelRole.LEFT_DRIVE, WheelRole.RIGHT_DRIVE}:
            actuators.append(
                Actuator(
                    ids["actuators"][joint.semantic_key],
                    f"{joint.semantic_key} motor",
                    ActuatorKind.ROTARY,
                    _jr(ids["joints"][joint.semantic_key]),
                    _fr(body_ids[joint.child_body_key]["frame"]),
                    limits={"velocity": q(0.8, "m/s", "speed")},
                    extensions=_mark(joint.semantic_key),
                )
            )
    for sensor in spec.sensors:
        frame_id = f"frame:{ids['prefix']}_{sensor.semantic_key}"
        frames.append(
            Frame(frame_id, sensor.semantic_key, _er(body_ids[sensor.parent_body_key]["entity"]))
        )
        transforms.append(
            _transform(
                f"transform:{ids['prefix']}_{sensor.semantic_key}",
                body_ids[sensor.parent_body_key]["frame"],
                frame_id,
                sensor.pose,
                provenance,
            )
        )
        modality = (
            SensorModality.LIDAR_2D
            if sensor.sensor_type is ConceptSensorType.LIDAR_2D
            else SensorModality.IMU
        )
        kwargs: dict[str, Any] = {}
        if modality is SensorModality.LIDAR_2D:
            kwargs = {
                "field_of_view": FieldOfView(horizontal=q(6.283185307, "rad", "angle")),
                "range": QuantityRange(
                    minimum=q(0.05, "m", "length"), maximum=q(12.0, "m", "length")
                ),
            }
        sensors.append(
            Sensor(
                ids["sensors"][sensor.semantic_key],
                sensor.semantic_key,
                modality,
                _fr(frame_id),
                sensor.update_rate_hz,
                noise_model=NoiseModel("gaussian", {"stddev": 0.01}),
                extensions={
                    **_mark(sensor.semantic_key),
                    "topic": sensor.topic,
                    "frame_id": sensor.semantic_key,
                },
                **kwargs,
            )
        )
    requirements = (
        Requirement(
            ids["mission_requirement"],
            RequirementKind.USER_DEFINED,
            spec.design_summary,
            RequirementPriority.HIGH,
            VerificationMethod.SIMULATION,
            RequirementStatus.PROPOSED,
            "bounded concept robot",
            provenance,
            extensions=_mark("mission"),
        ),
    )
    assumptions = tuple(
        Assumption(f"assumption:{ids['prefix']}_{i}", item, provenance)
        for i, item in enumerate(spec.assumptions, start=1)
    )
    software = (
        SoftwareComponent(
            ids["controller"],
            "Differential drive controller",
            "ros2_control_controller",
            output_refs=tuple(_jr(ids["joints"][j.semantic_key]) for j in _drive_joints(spec)),
            timing={"update_rate": q(100.0, "Hz", "frequency")},
            extensions={"controller_type": "diff_drive_controller/DiffDriveController"},
        ),
        SoftwareComponent(
            ids["bridge"],
            "ROS/Gazebo bridge",
            "ros_gz_bridge",
            input_refs=tuple(_sr(ids["sensors"][s.semantic_key]) for s in spec.sensors),
            extensions={"topics": ["/clock", "/scan", "/imu"]},
        ),
    )
    scenarios = (
        SimulationScenario(
            ids["scenario"],
            f"{spec.world_template.value} concept smoke test",
            spec.design_summary,
            spec.world_template.value,
            q(20.0, "s", "time"),
            extensions={"world_template": spec.world_template.value, "update_rate_hz": 100.0},
        ),
    )
    decisions = (
        EngineeringDecision(
            ids["passive_joint_decision"],
            "Only drive wheel joints are actuated",
            (
                "Milestone 3.1B supports diff/skid wheel control; other concept joints "
                "are fixed or passive in simulation."
            ),
            provenance,
        ),
    )
    return {
        "requirements": requirements,
        "assumptions": assumptions,
        "entities": tuple(entities),
        "frames": tuple(frames),
        "transforms": tuple(transforms),
        "materials": tuple(materials),
        "geometries": tuple(geometries),
        "bodies": tuple(bodies),
        "joints": tuple(joints),
        "actuators": tuple(actuators),
        "sensors": tuple(sensors),
        "software_components": software,
        "simulation_scenarios": scenarios,
        "decisions": decisions,
    }


def _geometry(
    body: ConceptBodySpec, geometry_id: str, material_id: str, provenance: Provenance
) -> Geometry:
    kwargs = {
        "material_ref": _cr(material_id),
        "provenance": provenance,
        "extensions": {**_mark(body.semantic_key), "primitive_type": body.primitive_type.value},
    }
    if body.primitive_type is PrimitiveType.BOX:
        return Geometry.box(
            geometry_id,
            q(body.size_x_m, "m", "length"),
            q(body.size_y_m, "m", "length"),
            q(body.size_z_m, "m", "length"),
            **kwargs,
        )
    if body.primitive_type is PrimitiveType.CYLINDER:
        return Geometry(
            geometry_id,
            GeometryKind.CYLINDER,
            radius=q(body.radius_m, "m", "length"),
            length=q(body.length_m, "m", "length"),
            **kwargs,
        )
    return Geometry(
        geometry_id, GeometryKind.SPHERE, radius=q(body.radius_m, "m", "length"), **kwargs
    )


def _mass_properties(body: ConceptBodySpec) -> MassProperties:
    inertia = tuple(
        tuple(q(v, "kg*m^2", "inertia") for v in row) for row in primitive_inertia(body)
    )
    return MassProperties(
        q(body.mass_kg, "kg", "mass"),
        Vector3(q(0, "m", "length"), q(0, "m", "length"), q(0, "m", "length")),
        inertia,
        MassProvenance.EXPLICIT,
    )


def _engineering_warnings(spec: ConceptRobotSpec) -> tuple[str, ...]:
    warnings: list[str] = []
    bodies = {body.semantic_key: body for body in spec.bodies}
    drive_joints = _drive_joints(spec)
    if len(drive_joints) >= 2:
        ys = [abs(bodies[j.child_body_key].pose.y_m) for j in drive_joints]
        track = 2 * min(ys) if ys else 0.0
        root = bodies[spec.root_body_key()]
        width = root.size_y_m if root.primitive_type is PrimitiveType.BOX else root.radius_m * 2
        if track < max(0.35, width * 0.85):
            warnings.append(
                "Drive wheel track is narrow relative to the chassis; tipping risk is higher."
            )
    max_z = max(_body_top_z(body) for body in spec.bodies)
    root = bodies[spec.root_body_key()]
    base_width = root.size_y_m if root.primitive_type is PrimitiveType.BOX else root.radius_m * 2
    if max_z > base_width * 1.8:
        warnings.append("High center-mounted structures may make the robot likely to tip.")
    for joint in drive_joints:
        child = bodies[joint.child_body_key]
        if child.primitive_type is PrimitiveType.CYLINDER and child.radius_m < 0.04:
            warnings.append("Drive wheels are tiny for practical ground clearance.")
    passive = [
        j
        for j in spec.joints
        if j.joint_type is ConceptJointType.REVOLUTE and j.wheel_role is WheelRole.NONE
    ]
    if passive:
        warnings.append(
            "Non-drive revolute joints are represented but not actively controlled in this release."
        )
    return tuple(warnings)


def _semantic_diff(before: ConceptRobotSpec | None, after: ConceptRobotSpec) -> tuple[str, ...]:
    if before is None:
        return ()
    changes: list[str] = []
    before_bodies = {body.semantic_key: body for body in before.bodies}
    after_bodies = {body.semantic_key: body for body in after.bodies}
    for key in sorted(after_bodies.keys() - before_bodies.keys()):
        changes.append(f"added body {key}")
    for key in sorted(before_bodies.keys() - after_bodies.keys()):
        changes.append(f"removed body {key}")
    for key in sorted(before_bodies.keys() & after_bodies.keys()):
        if before_bodies[key].model_dump() != after_bodies[key].model_dump():
            changes.append(f"changed body {key}")
    if before.world_template != after.world_template:
        changes.append(
            f"world changed {before.world_template.value} -> {after.world_template.value}"
        )
    return tuple(changes)


def _strip_existing_concept(project: EngineeringProject) -> EngineeringProject:
    def keep(items):
        return tuple(
            item
            for item in items
            if not isinstance(getattr(item, "extensions", None), dict)
            or CONCEPT_MARKER_KEY not in item.extensions
        )

    prefix = _existing_prefix(project)

    def keep_id(items):
        if prefix is None:
            return items
        return tuple(
            item
            for item in items
            if not getattr(item, "id", "").startswith(
                (
                    f"transform:{prefix}_",
                    f"frame:{prefix}_",
                    f"scenario:{prefix}_",
                    f"software:{prefix}_",
                    f"requirement:{prefix}_",
                    f"assumption:{prefix}_",
                    f"decision:{prefix}_",
                )
            )
        )

    return replace(
        project,
        requirements=keep_id(project.requirements),
        assumptions=keep_id(project.assumptions),
        entities=keep(project.entities),
        frames=keep_id(project.frames),
        transforms=keep_id(project.transforms),
        materials=keep(project.materials),
        geometries=keep(project.geometries),
        bodies=keep(project.bodies),
        joints=keep(project.joints),
        actuators=keep(project.actuators),
        sensors=keep(project.sensors),
        software_components=keep_id(project.software_components),
        simulation_scenarios=keep_id(project.simulation_scenarios),
        engineering_decisions=keep_id(project.engineering_decisions),
        extensions={k: v for k, v in project.extensions.items() if k != CONCEPT_EXTENSION_KEY},
    )


def _existing_prefix(project: EngineeringProject) -> str | None:
    extension = concept_extension(project)
    ids = extension.get("ids", {}) if extension else {}
    prefix = ids.get("prefix") if isinstance(ids, dict) else None
    return str(prefix) if prefix else None


def _reject_existing_ids(project: EngineeringProject, ids: dict[str, Any]) -> None:
    generated = set(_walk_ids(ids))
    existing = set()
    for collection_name in EngineeringProject.COLLECTIONS:
        for item in getattr(project, collection_name):
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str):
                existing.add(item_id)
    collisions = sorted(generated & existing)
    if collisions:
        raise ConceptError(
            "concept robot would duplicate existing canonical IDs: " + ", ".join(collisions[:5])
        )


def _walk_ids(value: Any):
    if isinstance(value, str) and ":" in value:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_ids(item)


def _drive_joints(spec: ConceptRobotSpec) -> tuple[ConceptJointSpec, ...]:
    return tuple(
        joint
        for joint in spec.joints
        if joint.wheel_role in {WheelRole.LEFT_DRIVE, WheelRole.RIGHT_DRIVE}
    )


def _parent_for(spec: ConceptRobotSpec, child: str) -> str:
    for joint in spec.joints:
        if joint.child_body_key == child:
            return joint.parent_body_key
    return spec.root_body_key()


def _body_top_z(body: ConceptBodySpec) -> float:
    if body.primitive_type is PrimitiveType.BOX:
        return body.pose.z_m + body.size_z_m / 2
    if body.primitive_type is PrimitiveType.CYLINDER:
        return body.pose.z_m + max(body.radius_m, body.length_m / 2)
    return body.pose.z_m + body.radius_m


def _reject_duplicate(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _has_cycle(joints: tuple[ConceptJointSpec, ...]) -> bool:
    parents = {joint.child_body_key: joint.parent_body_key for joint in joints}
    for child in parents:
        seen = set()
        current = child
        while current in parents:
            if current in seen:
                return True
            seen.add(current)
            current = parents[current]
    return False


def _transform(
    id_value: str, parent: str, child: str, pose: ConceptPose, provenance: Provenance
) -> Transform:
    return Transform(
        id_value,
        _fr(parent),
        _fr(child),
        Pose(
            _fr(parent),
            Vector3(
                q(pose.x_m, "m", "length"), q(pose.y_m, "m", "length"), q(pose.z_m, "m", "length")
            ),
        ),
        provenance,
    )


def _mark(key: str) -> dict[str, str]:
    return {CONCEPT_MARKER_KEY: key}


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text[:48] or "project"


def _er(id_value: str) -> Reference:
    return Reference("entity", id_value)


def _fr(id_value: str) -> Reference:
    return Reference("frame", id_value)


def _br(id_value: str) -> Reference:
    return Reference("body", id_value)


def _jr(id_value: str) -> Reference:
    return Reference("joint", id_value)


def _sr(id_value: str) -> Reference:
    return Reference("sensor", id_value)


def _cr(id_value: str) -> Reference:
    return Reference("component", id_value)


def _gb(id_value: str, role: GeometryRole = GeometryRole.VISUAL) -> GeometryBinding:
    return GeometryBinding(_cr(id_value), role)
