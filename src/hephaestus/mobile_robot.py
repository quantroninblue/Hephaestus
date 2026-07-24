"""Narrow differential-drive mobile robot design compiler."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hephaestus.changes import ChangeResult, fingerprint_project
from hephaestus.model import (
    MASS,
    Actuator,
    ActuatorKind,
    Assumption,
    Body,
    CommunicationInterface,
    Connection,
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
    Quantity,
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
from hephaestus.persistence import save_project
from hephaestus.validation import (
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    validate_project,
)

DEFAULT_POLICY_VERSION = "hephaestus.diff_drive.defaults.v1"
DESIGN_EXTENSION_KEY = "hephaestus:mobile_robot:v1"
GENERATOR_VERSION = "hephaestus.mobile_robot.compiler.v1"


class DesignSource(StrEnum):
    USER = "user_provided"
    EXISTING_REQUIREMENT = "existing_structured_requirement"
    DEFAULT = "application_default"


@dataclass(frozen=True, slots=True)
class DesignValue:
    value: float
    unit: str
    dimension: str
    source: DesignSource
    source_id: str | None = None

    def quantity(self) -> Quantity:
        return q(self.value, self.unit, self.dimension)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "value": self.value,
            "unit": self.unit,
            "dimension": self.dimension,
            "source": self.source.value,
        }
        if self.source_id is not None:
            data["source_id"] = self.source_id
        return data


class MobileRobotIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    robot_name: str = Field(min_length=1, max_length=80)
    mission_summary: str = Field(min_length=1, max_length=500)
    robot_family: Literal["differential_drive"] = "differential_drive"
    payload_kg: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    max_speed_mps: float | None = Field(default=None, gt=0.0, le=3.0, allow_inf_nan=False)
    chassis_length_m: float | None = Field(default=None, gt=0.0, le=2.0, allow_inf_nan=False)
    chassis_width_m: float | None = Field(default=None, gt=0.0, le=1.5, allow_inf_nan=False)
    chassis_height_m: float | None = Field(default=None, gt=0.0, le=0.8, allow_inf_nan=False)
    wheel_radius_m: float | None = Field(default=None, gt=0.0, le=0.3, allow_inf_nan=False)
    wheel_width_m: float | None = Field(default=None, gt=0.0, le=0.2, allow_inf_nan=False)
    include_lidar: bool = True
    include_imu: bool = True
    indoor_warehouse: bool = True
    explicitly_supersedes_payload: bool = False


def design_intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "robot_name": {"type": "string", "description": "Short robot name."},
            "mission_summary": {
                "type": "string",
                "description": "Brief supported warehouse inspection mission summary.",
            },
            "robot_family": {
                "type": "string",
                "enum": ["differential_drive"],
                "description": "Only differential_drive is implemented.",
            },
            "payload_kg": {"type": "number", "minimum": 0},
            "max_speed_mps": {"type": "number", "exclusiveMinimum": 0, "maximum": 3.0},
            "chassis_length_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 2.0},
            "chassis_width_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 1.5},
            "chassis_height_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.8},
            "wheel_radius_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.3},
            "wheel_width_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.2},
            "include_lidar": {"type": "boolean"},
            "include_imu": {"type": "boolean"},
            "indoor_warehouse": {"type": "boolean"},
            "explicitly_supersedes_payload": {"type": "boolean"},
        },
        "required": ["robot_name", "mission_summary"],
    }


@dataclass(frozen=True, slots=True)
class CompiledMobileRobot:
    project: EngineeringProject
    preview: dict[str, Any]
    validation: ValidationReport
    before_fingerprint: str
    after_fingerprint: str

    def to_result(self, *, dry_run: bool) -> ChangeResult:
        return ChangeResult(
            project=self.project,
            before_fingerprint=self.before_fingerprint,
            after_fingerprint=self.after_fingerprint,
            summary=str(self.preview["summary"]),
            validation=self.validation,
            dry_run=dry_run,
        )


DEFAULTS: dict[str, DesignValue] = {
    "payload": DesignValue(10.0, "kg", "mass", DesignSource.DEFAULT),
    "max_speed": DesignValue(0.8, "m/s", "speed", DesignSource.DEFAULT),
    "chassis_length": DesignValue(0.72, "m", "length", DesignSource.DEFAULT),
    "chassis_width": DesignValue(0.48, "m", "length", DesignSource.DEFAULT),
    "chassis_height": DesignValue(0.22, "m", "length", DesignSource.DEFAULT),
    "chassis_mass": DesignValue(18.0, "kg", "mass", DesignSource.DEFAULT),
    "wheel_radius": DesignValue(0.09, "m", "length", DesignSource.DEFAULT),
    "wheel_width": DesignValue(0.045, "m", "length", DesignSource.DEFAULT),
    "wheel_mass": DesignValue(1.2, "kg", "mass", DesignSource.DEFAULT),
    "wheel_separation": DesignValue(0.54, "m", "length", DesignSource.DEFAULT),
    "ground_clearance": DesignValue(0.04, "m", "length", DesignSource.DEFAULT),
    "caster_radius": DesignValue(0.035, "m", "length", DesignSource.DEFAULT),
    "caster_mass": DesignValue(0.4, "kg", "mass", DesignSource.DEFAULT),
    "material_density": DesignValue(2700.0, "kg/m^3", "density", DesignSource.DEFAULT),
    "lidar_update_rate": DesignValue(10.0, "Hz", "frequency", DesignSource.DEFAULT),
    "imu_update_rate": DesignValue(50.0, "Hz", "frequency", DesignSource.DEFAULT),
    "simulation_update_rate": DesignValue(100.0, "Hz", "frequency", DesignSource.DEFAULT),
}


def parse_design_intent(data: dict[str, Any]) -> MobileRobotIntent:
    try:
        return MobileRobotIntent.model_validate(data)
    except ValidationError as exc:
        raise ValueError("invalid mobile robot design intent") from exc


def compile_mobile_robot_design(
    project: EngineeringProject,
    intent: MobileRobotIntent,
    *,
    dry_run: bool = False,
) -> CompiledMobileRobot:
    if project.extensions.get(DESIGN_EXTENSION_KEY):
        raise ValueError("project already contains a Milestone 3.0 mobile robot design")

    values = _resolve_values(project, intent)
    _validate_supported_intent(intent, values)
    slug = _slug(intent.robot_name)
    ids = _ids(slug)
    if values["payload"].source_id is not None:
        ids = {**ids, "payload_requirement": values["payload"].source_id}
    _reject_existing_ids(project, ids, reused_ids=_reused_ids(values))
    provenance = Provenance(ProvenanceSource.GENERATED, "Milestone 3.0 mobile robot compiler")

    requirements = (*_requirements(intent, values, ids, provenance),)
    assumptions = _assumptions(values, ids, provenance, intent)
    canonical = _canonical_collections(intent, values, ids, provenance)

    extension = _design_extension(intent, values, ids)
    draft = replace(
        project,
        metadata=replace(
            project.metadata,
            description=intent.mission_summary.strip(),
        ),
        revision=project.revision + 1,
        requirements=(*project.requirements, *requirements),
        assumptions=(*project.assumptions, *assumptions),
        entities=(*project.entities, *canonical["entities"]),
        frames=(*project.frames, *canonical["frames"]),
        transforms=(*project.transforms, *canonical["transforms"]),
        materials=(*project.materials, *canonical["materials"]),
        geometries=(*project.geometries, *canonical["geometries"]),
        bodies=(*project.bodies, *canonical["bodies"]),
        joints=(*project.joints, *canonical["joints"]),
        connections=(*project.connections, *canonical["connections"]),
        actuators=(*project.actuators, *canonical["actuators"]),
        sensors=(*project.sensors, *canonical["sensors"]),
        communication_interfaces=(
            *project.communication_interfaces,
            *canonical["communication_interfaces"],
        ),
        software_components=(*project.software_components, *canonical["software_components"]),
        simulation_scenarios=(*project.simulation_scenarios, *canonical["simulation_scenarios"]),
        engineering_decisions=(*project.engineering_decisions, *canonical["decisions"]),
        extensions={**project.extensions, DESIGN_EXTENSION_KEY: extension},
    )
    validation = validate_project(draft)
    design_findings = tuple(validate_mobile_robot_project(draft))
    if design_findings:
        validation = ValidationReport((*validation.findings, *design_findings))
    if not dry_run and not validation.is_valid:
        codes = ", ".join(finding.code for finding in validation.findings)
        raise ValueError(f"compiled mobile robot design is invalid: {codes}")

    before = fingerprint_project(project)
    after = fingerprint_project(draft)
    return CompiledMobileRobot(
        project=draft,
        preview=_preview(intent, values, draft, validation),
        validation=validation,
        before_fingerprint=before,
        after_fingerprint=after,
    )


def apply_mobile_robot_design(project_path: Path, intent: MobileRobotIntent) -> dict[str, Any]:
    from hephaestus.persistence import load_project

    project = load_project(project_path)
    compiled = compile_mobile_robot_design(project, intent, dry_run=False)
    current = load_project(project_path)
    if fingerprint_project(current) != compiled.before_fingerprint:
        raise ValueError("active project changed after design evaluation")
    save_project(compiled.project, project_path, overwrite=True)
    return _compiled_payload(compiled, dry_run=False)


def preview_mobile_robot_design(
    project: EngineeringProject, intent: MobileRobotIntent
) -> dict[str, Any]:
    return _compiled_payload(
        compile_mobile_robot_design(project, intent, dry_run=True), dry_run=True
    )


def mobile_robot_spec(project: EngineeringProject) -> dict[str, Any]:
    data = project.extensions.get(DESIGN_EXTENSION_KEY)
    if not isinstance(data, dict):
        raise ValueError("project does not contain a generated mobile robot design")
    return data


def validate_mobile_robot_project(project: EngineeringProject) -> tuple[ValidationFinding, ...]:
    try:
        spec = mobile_robot_spec(project)
    except ValueError:
        return ()
    findings: list[ValidationFinding] = []
    values = spec.get("values", {})
    ids = spec.get("ids", {})
    for name in (
        "chassis_length",
        "chassis_width",
        "chassis_height",
        "wheel_radius",
        "wheel_width",
        "wheel_separation",
        "ground_clearance",
    ):
        item = values.get(name)
        if not isinstance(item, dict) or float(item.get("value", 0.0)) <= 0.0:
            findings.append(
                ValidationFinding(
                    "mobile_robot.dimension.invalid",
                    ValidationSeverity.ERROR,
                    f"Mobile robot {name} must be positive.",
                    f"extensions.{DESIGN_EXTENSION_KEY}.values.{name}",
                )
            )
    wheel_radius = _spec_float(values, "wheel_radius")
    wheel_width = _spec_float(values, "wheel_width")
    wheel_separation = _spec_float(values, "wheel_separation")
    chassis_width = _spec_float(values, "chassis_width")
    ground_clearance = _spec_float(values, "ground_clearance")
    if wheel_radius is not None and wheel_radius < 0.04:
        findings.append(
            ValidationFinding(
                "mobile_robot.wheel_radius.implausible",
                ValidationSeverity.ERROR,
                "Wheel radius is too small for the warehouse inspection slice.",
            )
        )
    if wheel_width is not None and wheel_radius is not None and wheel_width > wheel_radius:
        findings.append(
            ValidationFinding(
                "mobile_robot.wheel_width.implausible",
                ValidationSeverity.ERROR,
                "Wheel width must not exceed wheel radius for this slice.",
            )
        )
    if (
        wheel_separation is not None
        and chassis_width is not None
        and wheel_separation <= chassis_width
    ):
        findings.append(
            ValidationFinding(
                "mobile_robot.wheel_separation.invalid",
                ValidationSeverity.ERROR,
                "Wheel separation must exceed chassis width so wheels sit outside the chassis.",
            )
        )
    if (
        ground_clearance is not None
        and wheel_radius is not None
        and ground_clearance >= wheel_radius
    ):
        findings.append(
            ValidationFinding(
                "mobile_robot.ground_clearance.invalid",
                ValidationSeverity.ERROR,
                "Ground clearance must be smaller than wheel radius.",
            )
        )
    required_ids = [
        ids.get("left_wheel_joint"),
        ids.get("right_wheel_joint"),
        ids.get("left_drive_actuator"),
        ids.get("right_drive_actuator"),
    ]
    if any(not item for item in required_ids):
        findings.append(
            ValidationFinding(
                "mobile_robot.canonical_ids.missing",
                ValidationSeverity.ERROR,
                "Compiled mobile robot design is missing required canonical IDs.",
            )
        )
    return tuple(findings)


def _resolve_values(
    project: EngineeringProject, intent: MobileRobotIntent
) -> dict[str, DesignValue]:
    values = dict(DEFAULTS)
    existing_payload = _structured_payload_requirement(project)
    if existing_payload is not None:
        existing_value, existing_requirement_id = existing_payload
        if intent.payload_kg is None or math.isclose(
            existing_value, intent.payload_kg, rel_tol=0.0, abs_tol=1e-9
        ):
            values["payload"] = DesignValue(
                existing_value,
                "kg",
                "mass",
                DesignSource.EXISTING_REQUIREMENT,
                existing_requirement_id,
            )
        elif not intent.explicitly_supersedes_payload:
            raise ValueError(
                "payload request contradicts an existing structured payload requirement; "
                "explicitly supersede it before changing the payload"
            )
        else:
            values["payload"] = DesignValue(intent.payload_kg, "kg", "mass", DesignSource.USER)
    elif intent.payload_kg is not None:
        values["payload"] = DesignValue(intent.payload_kg, "kg", "mass", DesignSource.USER)

    overrides = {
        "max_speed": (intent.max_speed_mps, "m/s", "speed"),
        "chassis_length": (intent.chassis_length_m, "m", "length"),
        "chassis_width": (intent.chassis_width_m, "m", "length"),
        "chassis_height": (intent.chassis_height_m, "m", "length"),
        "wheel_radius": (intent.wheel_radius_m, "m", "length"),
        "wheel_width": (intent.wheel_width_m, "m", "length"),
    }
    for key, (value, unit, dimension) in overrides.items():
        if value is not None:
            values[key] = DesignValue(value, unit, dimension, DesignSource.USER)

    minimum_wheel_separation = values["chassis_width"].value + values["wheel_width"].value + 0.01
    if values["wheel_separation"].value < minimum_wheel_separation:
        default = values["wheel_separation"]
        values["wheel_separation"] = DesignValue(
            minimum_wheel_separation,
            default.unit,
            default.dimension,
            DesignSource.DEFAULT,
        )
    return values


def _structured_payload_requirement(project: EngineeringProject) -> tuple[float, str] | None:
    payloads: list[tuple[float, str]] = []
    for requirement in project.requirements:
        if (
            requirement.kind is not RequirementKind.PAYLOAD
            or requirement.status is RequirementStatus.REJECTED
        ):
            continue
        value = requirement.value
        if isinstance(value, Quantity) and value.dimension == MASS:
            payloads.append((value.to("kg").value, requirement.id))
        elif isinstance(value, QuantityRange):
            quantity = value.minimum or value.target or value.maximum
            if quantity is not None and quantity.dimension == MASS:
                payloads.append((quantity.to("kg").value, requirement.id))
    if len({round(value, 9) for value, _requirement_id in payloads}) > 1:
        raise ValueError("existing structured payload requirements contradict each other")
    return payloads[0] if payloads else None


def _reused_ids(values: dict[str, DesignValue]) -> frozenset[str]:
    return frozenset(value.source_id for value in values.values() if value.source_id is not None)


def _reject_existing_ids(
    project: EngineeringProject, ids: dict[str, str], *, reused_ids: frozenset[str]
) -> None:
    generated_ids = {value for key, value in ids.items() if key != "prefix"} - reused_ids
    existing_ids: set[str] = set()
    for collection_name in EngineeringProject.COLLECTIONS:
        for item in getattr(project, collection_name):
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str):
                existing_ids.add(item_id)
    collisions = sorted(generated_ids & existing_ids)
    if collisions:
        preview = ", ".join(collisions[:5])
        if len(collisions) > 5:
            preview = f"{preview}, ..."
        raise ValueError("mobile robot design would duplicate existing canonical IDs: " + preview)


def _validate_supported_intent(intent: MobileRobotIntent, values: dict[str, DesignValue]) -> None:
    del values
    if intent.robot_family != "differential_drive":
        raise ValueError("Milestone 3.0 design supports only differential-drive mobile robots")
    if not intent.indoor_warehouse:
        raise ValueError("Milestone 3.0 design supports only indoor warehouse environments")
    if not intent.include_lidar or not intent.include_imu:
        raise ValueError("Milestone 3.0 design includes both 2D LiDAR and IMU")


def _requirements(
    intent: MobileRobotIntent,
    values: dict[str, DesignValue],
    ids: dict[str, str],
    provenance: Provenance,
) -> tuple[Requirement, ...]:
    payload = values["payload"]
    speed = values["max_speed"]
    requirements: list[Requirement] = [
        Requirement(
            ids["mission_requirement"],
            RequirementKind.USER_DEFINED,
            intent.mission_summary.strip(),
            RequirementPriority.HIGH,
            VerificationMethod.DEMONSTRATION,
            RequirementStatus.PROPOSED,
            "differential-drive warehouse inspection mission",
            provenance,
        )
    ]
    if payload.source_id is None:
        requirements.append(
            Requirement(
                ids["payload_requirement"],
                RequirementKind.PAYLOAD,
                f"Carry an inspection payload of at least {payload.value:g} {payload.unit}.",
                RequirementPriority.HIGH,
                VerificationMethod.TEST,
                RequirementStatus.PROPOSED,
                QuantityRange(minimum=payload.quantity()),
                provenance,
            )
        )
    requirements.append(
        Requirement(
            ids["speed_requirement"],
            RequirementKind.SPEED,
            f"Reach a maximum commanded speed of {speed.value:g} {speed.unit}.",
            RequirementPriority.MEDIUM,
            VerificationMethod.SIMULATION,
            RequirementStatus.PROPOSED,
            QuantityRange(maximum=speed.quantity()),
            provenance,
        )
    )
    return tuple(requirements)


def _assumptions(
    values: dict[str, DesignValue],
    ids: dict[str, str],
    provenance: Provenance,
    intent: MobileRobotIntent,
) -> tuple[Assumption, ...]:
    items: list[Assumption] = []
    for key, value in values.items():
        if value.source is DesignSource.DEFAULT:
            items.append(
                Assumption(
                    f"{ids['prefix']}:assumption:{key}",
                    f"{key.replace('_', ' ')} defaults to {value.value:g} {value.unit} "
                    f"under {DEFAULT_POLICY_VERSION}.",
                    provenance,
                )
            )
    items.append(
        Assumption(
            ids["environment_assumption"],
            "Simulation environment is a local indoor warehouse world with no remote assets.",
            provenance,
        )
    )
    if intent.include_lidar:
        items.append(
            Assumption(
                ids["lidar_assumption"],
                "2D LiDAR is mounted above the chassis on the base_link frame.",
                provenance,
            )
        )
    if intent.include_imu:
        items.append(
            Assumption(
                ids["imu_assumption"],
                "IMU is mounted near the chassis center of mass on the base_link frame.",
                provenance,
            )
        )
    return tuple(items)


def _canonical_collections(
    intent: MobileRobotIntent,
    values: dict[str, DesignValue],
    ids: dict[str, str],
    provenance: Provenance,
) -> dict[str, tuple[Any, ...]]:
    length = values["chassis_length"].quantity()
    width = values["chassis_width"].quantity()
    height = values["chassis_height"].quantity()
    wheel_radius = values["wheel_radius"].quantity()
    wheel_width = values["wheel_width"].quantity()
    caster_radius = values["caster_radius"].quantity()
    wheel_sep = values["wheel_separation"].value
    base_z = values["ground_clearance"].value + height.si_value / 2.0
    left_y = wheel_sep / 2.0
    right_y = -wheel_sep / 2.0
    wheel_x = -length.si_value * 0.22
    caster_x = length.si_value * 0.34

    entities = (
        Entity(
            ids["robot_entity"],
            intent.robot_name.strip(),
            EntityKind.ASSEMBLY,
            frame_ref=_fr(ids["base_footprint"]),
        ),
        Entity(
            ids["chassis_entity"],
            "Rigid chassis",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["base_link"]),
        ),
        Entity(
            ids["left_wheel_entity"],
            "Left driven wheel",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["left_wheel_frame"]),
        ),
        Entity(
            ids["right_wheel_entity"],
            "Right driven wheel",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["right_wheel_frame"]),
        ),
        Entity(
            ids["caster_entity"],
            "Passive caster support",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["caster_frame"]),
        ),
        Entity(
            ids["lidar_entity"],
            "2D LiDAR",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["lidar_frame"]),
        ),
        Entity(
            ids["imu_entity"],
            "IMU",
            EntityKind.BODY,
            _er(ids["robot_entity"]),
            _fr(ids["imu_frame"]),
        ),
    )
    frames = tuple(
        Frame(id_value, id_value.split(":", 1)[1].replace("_", " "))
        for id_value in (
            ids["base_footprint"],
            ids["base_link"],
            ids["left_wheel_frame"],
            ids["right_wheel_frame"],
            ids["caster_frame"],
            ids["lidar_frame"],
            ids["imu_frame"],
        )
    )
    transforms = (
        _transform(
            ids["tf_base"], ids["base_footprint"], ids["base_link"], 0.0, 0.0, base_z, provenance
        ),
        _transform(
            ids["tf_left_wheel"],
            ids["base_link"],
            ids["left_wheel_frame"],
            wheel_x,
            left_y,
            wheel_radius.si_value - base_z,
            provenance,
        ),
        _transform(
            ids["tf_right_wheel"],
            ids["base_link"],
            ids["right_wheel_frame"],
            wheel_x,
            right_y,
            wheel_radius.si_value - base_z,
            provenance,
        ),
        _transform(
            ids["tf_caster"],
            ids["base_link"],
            ids["caster_frame"],
            caster_x,
            0.0,
            caster_radius.si_value - base_z,
            provenance,
        ),
        _transform(
            ids["tf_lidar"],
            ids["base_link"],
            ids["lidar_frame"],
            length.si_value * 0.18,
            0.0,
            height.si_value / 2.0 + 0.08,
            provenance,
        ),
        _transform(ids["tf_imu"], ids["base_link"], ids["imu_frame"], 0.0, 0.0, 0.02, provenance),
    )
    material = Material(
        ids["material"],
        "Powder-coated aluminum surrogate",
        values["material_density"].quantity(),
        provenance=provenance,
        extensions={"hephaestus:material_policy": DEFAULT_POLICY_VERSION},
    )
    geometries = (
        Geometry.box(
            ids["chassis_geometry"],
            length,
            width,
            height,
            material_ref=_cr(ids["material"]),
            provenance=provenance,
        ),
        Geometry(
            ids["wheel_geometry"],
            GeometryKind.CYLINDER,
            radius=wheel_radius,
            length=wheel_width,
            material_ref=_cr(ids["material"]),
            provenance=provenance,
        ),
        Geometry(
            ids["caster_geometry"],
            GeometryKind.SPHERE,
            radius=caster_radius,
            material_ref=_cr(ids["material"]),
            provenance=provenance,
        ),
        Geometry(
            ids["lidar_geometry"],
            GeometryKind.CYLINDER,
            radius=q(0.045, "m", "length"),
            length=q(0.035, "m", "length"),
            material_ref=_cr(ids["material"]),
            provenance=provenance,
        ),
        Geometry.box(
            ids["imu_geometry"],
            q(0.05, "m", "length"),
            q(0.04, "m", "length"),
            q(0.02, "m", "length"),
            material_ref=_cr(ids["material"]),
            provenance=provenance,
        ),
    )
    bodies = (
        Body(
            ids["chassis_body"],
            _er(ids["chassis_entity"]),
            _fr(ids["base_link"]),
            (_gb(ids["chassis_geometry"]),),
            (_gb(ids["chassis_geometry"], GeometryRole.COLLISION),),
            _box_mass(
                values["chassis_mass"].quantity(),
                length.si_value,
                width.si_value,
                height.si_value,
            ),
        ),
        Body(
            ids["left_wheel_body"],
            _er(ids["left_wheel_entity"]),
            _fr(ids["left_wheel_frame"]),
            (_gb(ids["wheel_geometry"]),),
            (_gb(ids["wheel_geometry"], GeometryRole.COLLISION),),
            _cylinder_mass(
                values["wheel_mass"].quantity(), wheel_radius.si_value, wheel_width.si_value
            ),
        ),
        Body(
            ids["right_wheel_body"],
            _er(ids["right_wheel_entity"]),
            _fr(ids["right_wheel_frame"]),
            (_gb(ids["wheel_geometry"]),),
            (_gb(ids["wheel_geometry"], GeometryRole.COLLISION),),
            _cylinder_mass(
                values["wheel_mass"].quantity(), wheel_radius.si_value, wheel_width.si_value
            ),
        ),
        Body(
            ids["caster_body"],
            _er(ids["caster_entity"]),
            _fr(ids["caster_frame"]),
            (_gb(ids["caster_geometry"]),),
            (_gb(ids["caster_geometry"], GeometryRole.COLLISION),),
            _sphere_mass(values["caster_mass"].quantity(), caster_radius.si_value),
        ),
    )
    joints = (
        Joint(
            ids["left_wheel_joint"],
            "left_wheel_joint",
            JointKind.CONTINUOUS,
            _br(ids["chassis_body"]),
            _br(ids["left_wheel_body"]),
            axis=(0.0, 1.0, 0.0),
            provenance=provenance,
        ),
        Joint(
            ids["right_wheel_joint"],
            "right_wheel_joint",
            JointKind.CONTINUOUS,
            _br(ids["chassis_body"]),
            _br(ids["right_wheel_body"]),
            axis=(0.0, 1.0, 0.0),
            provenance=provenance,
        ),
        Joint(
            ids["caster_joint"],
            "caster_joint",
            JointKind.SPHERICAL,
            _br(ids["chassis_body"]),
            _br(ids["caster_body"]),
            provenance=provenance,
        ),
    )
    sensors = (
        Sensor(
            ids["lidar_sensor"],
            "2D LiDAR",
            SensorModality.LIDAR_2D,
            _fr(ids["lidar_frame"]),
            values["lidar_update_rate"].value,
            field_of_view=FieldOfView(horizontal=q(6.283185307, "rad", "angle")),
            range=QuantityRange(minimum=q(0.05, "m", "length"), maximum=q(12.0, "m", "length")),
            noise_model=NoiseModel("gaussian", {"stddev": 0.01}),
            extensions={"topic": "/scan", "frame_id": "lidar_link"},
        ),
        Sensor(
            ids["imu_sensor"],
            "IMU",
            SensorModality.IMU,
            _fr(ids["imu_frame"]),
            values["imu_update_rate"].value,
            noise_model=NoiseModel(
                "gaussian", {"angular_velocity_stddev": 0.0002, "linear_acceleration_stddev": 0.017}
            ),
            extensions={"topic": "/imu", "frame_id": "imu_link"},
        ),
    )
    actuators = (
        Actuator(
            ids["left_drive_actuator"],
            "Left drive motor",
            ActuatorKind.ROTARY,
            _jr(ids["left_wheel_joint"]),
            _fr(ids["left_wheel_frame"]),
            limits={"velocity": values["max_speed"].quantity()},
        ),
        Actuator(
            ids["right_drive_actuator"],
            "Right drive motor",
            ActuatorKind.ROTARY,
            _jr(ids["right_wheel_joint"]),
            _fr(ids["right_wheel_frame"]),
            limits={"velocity": values["max_speed"].quantity()},
        ),
    )
    interfaces = (
        CommunicationInterface(
            ids["scan_interface"],
            _sr(ids["lidar_sensor"]),
            name="/scan",
            update_rate_hz=values["lidar_update_rate"].value,
        ),
        CommunicationInterface(
            ids["imu_interface"],
            _sr(ids["imu_sensor"]),
            name="/imu",
            update_rate_hz=values["imu_update_rate"].value,
        ),
    )
    software = (
        SoftwareComponent(
            ids["diff_drive_controller"],
            "Differential drive controller",
            "ros2_control_controller",
            output_refs=(_jr(ids["left_wheel_joint"]), _jr(ids["right_wheel_joint"])),
            timing={"update_rate": values["simulation_update_rate"].quantity()},
            extensions={"controller_type": "diff_drive_controller/DiffDriveController"},
        ),
        SoftwareComponent(
            ids["bridge_component"],
            "ROS/Gazebo bridge",
            "ros_gz_bridge",
            input_refs=(_sr(ids["lidar_sensor"]), _sr(ids["imu_sensor"])),
            extensions={"topics": ["/clock", "/scan", "/imu"]},
        ),
    )
    scenarios = (
        SimulationScenario(
            ids["scenario"],
            "Warehouse inspection smoke test",
            "Headless Gazebo Harmonic warehouse validation",
            "indoor_warehouse",
            q(20.0, "s", "time"),
            extensions={
                "world": "warehouse_inspection.sdf",
                "update_rate_hz": values["simulation_update_rate"].value,
            },
        ),
    )
    decisions = (
        EngineeringDecision(
            ids["decision_family"],
            "Use differential-drive mobile base",
            "Milestone 3.0 intentionally supports one complete warehouse inspection slice "
            "rather than generic robot synthesis.",
            provenance,
        ),
        EngineeringDecision(
            ids["decision_collision"],
            "Use CAD-derived visual meshes and primitive collision geometry",
            "Primitive collisions remain dimensionally tied to canonical dimensions and are "
            "robust for Gazebo smoke tests.",
            provenance,
        ),
    )
    connections = (
        Connection(
            ids["caster_connection"],
            "passive_support",
            _br(ids["chassis_body"]),
            _br(ids["caster_body"]),
            {"joint": ids["caster_joint"]},
        ),
        Connection(
            ids["controller_left_connection"],
            "actuator_controls_joint",
            _ar(ids["left_drive_actuator"]),
            _jr(ids["left_wheel_joint"]),
        ),
        Connection(
            ids["controller_right_connection"],
            "actuator_controls_joint",
            _ar(ids["right_drive_actuator"]),
            _jr(ids["right_wheel_joint"]),
        ),
    )
    return {
        "entities": entities,
        "frames": frames,
        "transforms": transforms,
        "materials": (material,),
        "geometries": geometries,
        "bodies": bodies,
        "joints": joints,
        "connections": connections,
        "actuators": actuators,
        "sensors": sensors,
        "communication_interfaces": interfaces,
        "software_components": software,
        "simulation_scenarios": scenarios,
        "decisions": decisions,
    }


def _design_extension(
    intent: MobileRobotIntent, values: dict[str, DesignValue], ids: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema": "hephaestus.mobile_robot.v1",
        "supported_family": "differential_drive",
        "default_policy": DEFAULT_POLICY_VERSION,
        "generator": GENERATOR_VERSION,
        "robot_name": intent.robot_name.strip(),
        "mission_summary": intent.mission_summary.strip(),
        "values": {key: value.to_dict() for key, value in sorted(values.items())},
        "ids": {key: ids[key] for key in sorted(ids)},
        "coordinate_convention": "right-handed; X forward, Y left, Z up; quaternions w,x,y,z",
        "units": "SI metres kilograms seconds radians for canonical values",
    }


def _preview(
    intent: MobileRobotIntent,
    values: dict[str, DesignValue],
    project: EngineeringProject,
    validation: ValidationReport,
) -> dict[str, Any]:
    defaulted = [
        key for key, value in sorted(values.items()) if value.source is DesignSource.DEFAULT
    ]
    return {
        "summary": (
            f"Create differential-drive warehouse inspection robot {intent.robot_name.strip()}"
        ),
        "robot": {
            "name": intent.robot_name.strip(),
            "family": "differential_drive",
            "mission": intent.mission_summary.strip(),
            "dimensions_m": {
                "length": values["chassis_length"].value,
                "width": values["chassis_width"].value,
                "height": values["chassis_height"].value,
            },
            "payload_kg": values["payload"].value,
            "max_speed_mps": values["max_speed"].value,
            "wheel_radius_m": values["wheel_radius"].value,
            "wheel_separation_m": values["wheel_separation"].value,
            "sensors": ["2D LiDAR", "IMU"],
        },
        "value_sources": {key: value.source.value for key, value in sorted(values.items())},
        "assumptions": defaulted,
        "validation": validation.to_dict(),
        "current_revision": project.revision - 1,
        "proposed_revision": project.revision,
    }


def _compiled_payload(compiled: CompiledMobileRobot, *, dry_run: bool) -> dict[str, Any]:
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


def _ids(slug: str) -> dict[str, str]:
    prefix = f"robot:{slug}"
    raw = {
        "prefix": prefix,
        "robot_entity": f"entity:{slug}",
        "chassis_entity": f"entity:{slug}_chassis",
        "left_wheel_entity": f"entity:{slug}_left_wheel",
        "right_wheel_entity": f"entity:{slug}_right_wheel",
        "caster_entity": f"entity:{slug}_caster",
        "lidar_entity": f"entity:{slug}_lidar",
        "imu_entity": f"entity:{slug}_imu",
        "base_footprint": f"frame:{slug}_base_footprint",
        "base_link": f"frame:{slug}_base_link",
        "left_wheel_frame": f"frame:{slug}_left_wheel",
        "right_wheel_frame": f"frame:{slug}_right_wheel",
        "caster_frame": f"frame:{slug}_caster",
        "lidar_frame": f"frame:{slug}_lidar",
        "imu_frame": f"frame:{slug}_imu",
        "tf_base": f"component:{slug}_tf_base",
        "tf_left_wheel": f"component:{slug}_tf_left_wheel",
        "tf_right_wheel": f"component:{slug}_tf_right_wheel",
        "tf_caster": f"component:{slug}_tf_caster",
        "tf_lidar": f"component:{slug}_tf_lidar",
        "tf_imu": f"component:{slug}_tf_imu",
        "material": f"component:{slug}_aluminum",
        "chassis_geometry": f"component:{slug}_chassis_box",
        "wheel_geometry": f"component:{slug}_wheel_cylinder",
        "caster_geometry": f"component:{slug}_caster_sphere",
        "lidar_geometry": f"component:{slug}_lidar_cylinder",
        "imu_geometry": f"component:{slug}_imu_box",
        "chassis_body": f"body:{slug}_chassis",
        "left_wheel_body": f"body:{slug}_left_wheel",
        "right_wheel_body": f"body:{slug}_right_wheel",
        "caster_body": f"body:{slug}_caster",
        "left_wheel_joint": f"joint:{slug}_left_wheel_joint",
        "right_wheel_joint": f"joint:{slug}_right_wheel_joint",
        "caster_joint": f"joint:{slug}_caster_joint",
        "lidar_sensor": f"sensor:{slug}_lidar",
        "imu_sensor": f"sensor:{slug}_imu",
        "left_drive_actuator": f"actuator:{slug}_left_drive",
        "right_drive_actuator": f"actuator:{slug}_right_drive",
        "scan_interface": f"component:{slug}_scan_interface",
        "imu_interface": f"component:{slug}_imu_interface",
        "diff_drive_controller": f"software_component:{slug}_diff_drive_controller",
        "bridge_component": f"software_component:{slug}_ros_gz_bridge",
        "scenario": f"scenario:{slug}_warehouse",
        "mission_requirement": f"requirement:{slug}_mission",
        "payload_requirement": f"requirement:{slug}_payload",
        "speed_requirement": f"requirement:{slug}_speed",
        "environment_assumption": f"assumption:{slug}_environment",
        "lidar_assumption": f"assumption:{slug}_lidar",
        "imu_assumption": f"assumption:{slug}_imu",
        "decision_family": f"component:{slug}_decision_family",
        "decision_collision": f"component:{slug}_decision_collision",
        "caster_connection": f"connection:{slug}_caster_support",
        "controller_left_connection": f"connection:{slug}_left_drive_control",
        "controller_right_connection": f"connection:{slug}_right_drive_control",
    }
    return {key: _id(value) for key, value in raw.items()}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:48] or "warehouse_inspector"


def _id(value: str) -> str:
    value = re.sub(r"[^a-z0-9_:-]", "_", value.lower())
    value = re.sub(r"_+", "_", value)
    if ":" not in value:
        value = f"component:{value}"
    return value[:96]


def _fr(value: str) -> Reference:
    return Reference("frame", value)


def _er(value: str) -> Reference:
    return Reference("entity", value)


def _br(value: str) -> Reference:
    return Reference("body", value)


def _cr(value: str) -> Reference:
    return Reference("component", value)


def _jr(value: str) -> Reference:
    return Reference("joint", value)


def _sr(value: str) -> Reference:
    return Reference("sensor", value)


def _ar(value: str) -> Reference:
    return Reference("actuator", value)


def _gb(geometry_id: str, role: GeometryRole = GeometryRole.VISUAL) -> GeometryBinding:
    return GeometryBinding(_cr(geometry_id), role)


def _transform(
    transform_id: str,
    parent: str,
    child: str,
    x: float,
    y: float,
    z: float,
    provenance: Provenance,
) -> Transform:
    return Transform(
        transform_id,
        _fr(parent),
        _fr(child),
        Pose(
            _fr(parent),
            Vector3(q(x, "m", "length"), q(y, "m", "length"), q(z, "m", "length")),
        ),
        provenance,
    )


def _box_mass(mass: Quantity, x: float, y: float, z: float) -> MassProperties:
    m = mass.si_value
    return _mass_props(
        mass, (m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12)
    )


def _cylinder_mass(mass: Quantity, radius: float, length: float) -> MassProperties:
    m = mass.si_value
    i_axis = 0.5 * m * radius * radius
    i_other = (m * (3 * radius * radius + length * length)) / 12
    return _mass_props(mass, (i_other, i_axis, i_other))


def _sphere_mass(mass: Quantity, radius: float) -> MassProperties:
    inertia = 0.4 * mass.si_value * radius * radius
    return _mass_props(mass, (inertia, inertia, inertia))


def _mass_props(mass: Quantity, diag: tuple[float, float, float]) -> MassProperties:
    zero = q(0.0, "kg*m^2", "inertia")
    tensor = (
        (q(diag[0], "kg*m^2", "inertia"), zero, zero),
        (zero, q(diag[1], "kg*m^2", "inertia"), zero),
        (zero, zero, q(diag[2], "kg*m^2", "inertia")),
    )
    return MassProperties(
        mass,
        Vector3(q(0.0, "m", "length"), q(0.0, "m", "length"), q(0.0, "m", "length")),
        tensor,
        MassProvenance.DERIVED,
    )


def _spec_float(values: dict[str, Any], key: str) -> float | None:
    item = values.get(key)
    if isinstance(item, dict) and isinstance(item.get("value"), int | float):
        return float(item["value"])
    return None


def design_generation_id(project: EngineeringProject) -> str:
    seed = f"{project.revision}:{fingerprint_project(project)}"
    return "generation:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
