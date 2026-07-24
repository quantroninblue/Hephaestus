"""Typed design intent compilers for canonical engineering projects."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from hephaestus.changes import fingerprint_project
from hephaestus.model import (
    LENGTH,
    Actuator,
    ActuatorKind,
    Assumption,
    Body,
    BusKind,
    CommunicationBus,
    CommunicationInterface,
    ComputeDevice,
    ElectricalDevice,
    ElectricalPort,
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
    ModelError,
    Pose,
    PowerConnection,
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
    identity_pose,
    q,
    zero_vector,
)
from hephaestus.validation import ValidationReport, validate_project

DESIGN_EXTENSION_NS = "hephaestus.design"
SUPPORTED_ROBOT_FAMILY = "differential_drive"


class DesignError(ValueError):
    """Raised when a typed design intent cannot be compiled."""


class DesignValueSource(StrEnum):
    USER_PROVIDED = "user_provided"
    EXISTING_REQUIREMENT = "existing_requirement"
    DEFAULT_POLICY = "default_policy"
    ASSUMPTION = "assumption"


@dataclass(frozen=True, slots=True)
class DesignValue:
    value: float | bool | str
    source: DesignValueSource
    field: str
    requirement_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "value": self.value,
            "source": self.source.value,
            "field": self.field,
        }
        if self.requirement_id is not None:
            data["requirement_id"] = self.requirement_id
        return data


@dataclass(frozen=True, slots=True)
class WarehouseInspectionRobotIntent:
    robot_family: str
    robot_name: str
    mission: str
    warehouse_environment: str
    payload_kg: float | None = None
    max_speed_mps: float | None = None
    chassis_length_m: float | None = None
    chassis_width_m: float | None = None
    chassis_height_m: float | None = None
    wheel_diameter_m: float | None = None
    wheel_width_m: float | None = None
    has_lidar: bool | None = None
    has_imu: bool | None = None


@dataclass(frozen=True, slots=True)
class DesignCompilation:
    project: EngineeringProject
    before_fingerprint: str
    after_fingerprint: str
    validation: ValidationReport
    summary: str
    value_sources: dict[str, DesignValue]

    def to_dict(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "dry_run": dry_run,
            "revision": self.project.revision,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "validation": self.validation.to_dict(),
            "operation": "design_warehouse_inspection_robot",
            "design": {
                "robot_family": SUPPORTED_ROBOT_FAMILY,
                "robot_name": self.value_sources["robot_name"].value,
                "value_sources": {
                    key: value.to_dict() for key, value in sorted(self.value_sources.items())
                },
            },
            "human_preview": {
                "title": "Design warehouse inspection robot",
                "statement": self.summary,
                "current_revision": self.project.revision - 1,
                "proposed_revision": self.project.revision,
                "validation": "passed" if self.validation.is_valid else "has findings",
            },
        }


def compile_warehouse_inspection_robot(
    base_project: EngineeringProject, intent: WarehouseInspectionRobotIntent
) -> DesignCompilation:
    """Compile one differential-drive warehouse inspection robot into the project model."""

    values = _resolve_intent_values(base_project, intent)
    slug = _slug(values["robot_name"].value)
    prefix = f"{slug}_warehouse_inspection"
    generated_ids = _generated_ids(prefix, values)
    _reject_existing_ids(base_project, generated_ids)

    generated = _build_project_elements(prefix, values)
    project = replace(
        base_project,
        revision=base_project.revision + 1,
        requirements=(*base_project.requirements, *generated["requirements"]),
        assumptions=(*base_project.assumptions, *generated["assumptions"]),
        entities=(*base_project.entities, *generated["entities"]),
        frames=(*base_project.frames, *generated["frames"]),
        transforms=(*base_project.transforms, *generated["transforms"]),
        geometries=(*base_project.geometries, *generated["geometries"]),
        bodies=(*base_project.bodies, *generated["bodies"]),
        joints=(*base_project.joints, *generated["joints"]),
        actuators=(*base_project.actuators, *generated["actuators"]),
        sensors=(*base_project.sensors, *generated["sensors"]),
        electrical_devices=(*base_project.electrical_devices, *generated["electrical_devices"]),
        electrical_ports=(*base_project.electrical_ports, *generated["electrical_ports"]),
        power_connections=(*base_project.power_connections, *generated["power_connections"]),
        communication_buses=(*base_project.communication_buses, *generated["communication_buses"]),
        communication_interfaces=(
            *base_project.communication_interfaces,
            *generated["communication_interfaces"],
        ),
        compute_devices=(*base_project.compute_devices, *generated["compute_devices"]),
        software_components=(*base_project.software_components, *generated["software_components"]),
        simulation_scenarios=(
            *base_project.simulation_scenarios,
            *generated["simulation_scenarios"],
        ),
        engineering_decisions=(
            *base_project.engineering_decisions,
            *generated["engineering_decisions"],
        ),
        extensions={
            **base_project.extensions,
            DESIGN_EXTENSION_NS: {
                "last_operation": "design_warehouse_inspection_robot",
                "robot_family": SUPPORTED_ROBOT_FAMILY,
                "robot_name": values["robot_name"].value,
                "value_sources": {key: value.to_dict() for key, value in sorted(values.items())},
            },
        },
    )
    validation = validate_project(project)
    return DesignCompilation(
        project=project,
        before_fingerprint=fingerprint_project(base_project),
        after_fingerprint=fingerprint_project(project),
        validation=validation,
        summary=(
            "Design differential-drive warehouse inspection robot "
            f"{values['robot_name'].value!r} for {values['mission'].value}."
        ),
        value_sources=values,
    )


def _resolve_intent_values(
    project: EngineeringProject, intent: WarehouseInspectionRobotIntent
) -> dict[str, DesignValue]:
    if intent.robot_family.strip() != SUPPORTED_ROBOT_FAMILY:
        raise DesignError(
            "unsupported robot family: "
            f"{intent.robot_family}; supported family is {SUPPORTED_ROBOT_FAMILY}"
        )
    robot_name = _required_text(intent.robot_name, "robot_name")
    mission = _required_text(intent.mission, "mission")
    environment = _required_text(intent.warehouse_environment, "warehouse_environment")
    values = {
        "robot_name": DesignValue(robot_name, DesignValueSource.USER_PROVIDED, "robot_name"),
        "mission": DesignValue(mission, DesignValueSource.USER_PROVIDED, "mission"),
        "warehouse_environment": DesignValue(
            environment,
            DesignValueSource.USER_PROVIDED,
            "warehouse_environment",
        ),
        "payload_kg": _numeric_or_existing_requirement(
            project,
            intent.payload_kg,
            field="payload_kg",
            kind=RequirementKind.PAYLOAD,
            default=20.0,
            positive=False,
        ),
        "max_speed_mps": _numeric_or_existing_requirement(
            project,
            intent.max_speed_mps,
            field="max_speed_mps",
            kind=RequirementKind.SPEED,
            default=1.2,
            positive=True,
        ),
        "chassis_length_m": _defaulted_number(
            intent.chassis_length_m, "chassis_length_m", 0.8, positive=True
        ),
        "chassis_width_m": _defaulted_number(
            intent.chassis_width_m, "chassis_width_m", 0.55, positive=True
        ),
        "chassis_height_m": _defaulted_number(
            intent.chassis_height_m, "chassis_height_m", 0.28, positive=True
        ),
        "wheel_diameter_m": _defaulted_number(
            intent.wheel_diameter_m, "wheel_diameter_m", 0.18, positive=True
        ),
        "wheel_width_m": _defaulted_number(
            intent.wheel_width_m, "wheel_width_m", 0.06, positive=True
        ),
        "has_lidar": _defaulted_bool(intent.has_lidar, "has_lidar", True),
        "has_imu": _defaulted_bool(intent.has_imu, "has_imu", True),
    }
    _validate_relative_dimensions(values)
    return values


def _build_project_elements(
    prefix: str, values: dict[str, DesignValue]
) -> dict[str, tuple[Any, ...]]:
    robot_name = str(values["robot_name"].value)
    mission = str(values["mission"].value)
    environment = str(values["warehouse_environment"].value)
    payload_kg = float(values["payload_kg"].value)
    max_speed_mps = float(values["max_speed_mps"].value)
    length_m = float(values["chassis_length_m"].value)
    width_m = float(values["chassis_width_m"].value)
    height_m = float(values["chassis_height_m"].value)
    wheel_diameter_m = float(values["wheel_diameter_m"].value)
    wheel_width_m = float(values["wheel_width_m"].value)
    wheel_radius_m = wheel_diameter_m / 2.0
    has_lidar = bool(values["has_lidar"].value)
    has_imu = bool(values["has_imu"].value)

    generated = Provenance(ProvenanceSource.GENERATED, "warehouse inspection robot compiler")
    calculated = Provenance(ProvenanceSource.CALCULATED, "warehouse inspection robot compiler")
    user = Provenance(ProvenanceSource.USER, "warehouse inspection robot design intent")

    base_frame_id = f"frame:{prefix}_base_link"
    chassis_frame_id = f"frame:{prefix}_chassis"
    left_wheel_frame_id = f"frame:{prefix}_left_wheel"
    right_wheel_frame_id = f"frame:{prefix}_right_wheel"
    lidar_frame_id = f"frame:{prefix}_lidar"
    imu_frame_id = f"frame:{prefix}_imu"

    entities: tuple[Entity, ...] = (
        Entity(
            f"entity:{prefix}_robot",
            robot_name,
            EntityKind.ASSEMBLY,
            frame_ref=Reference("frame", base_frame_id),
            extensions=_extension("robot_name", values),
        ),
        Entity(
            f"entity:{prefix}_chassis",
            f"{robot_name} chassis",
            EntityKind.BODY,
            parent_entity=Reference("entity", f"entity:{prefix}_robot"),
            frame_ref=Reference("frame", chassis_frame_id),
            extensions=_extension("chassis_length_m", values),
        ),
        Entity(
            f"entity:{prefix}_left_wheel",
            "Left drive wheel",
            EntityKind.BODY,
            parent_entity=Reference("entity", f"entity:{prefix}_robot"),
            frame_ref=Reference("frame", left_wheel_frame_id),
            extensions=_extension("wheel_diameter_m", values),
        ),
        Entity(
            f"entity:{prefix}_right_wheel",
            "Right drive wheel",
            EntityKind.BODY,
            parent_entity=Reference("entity", f"entity:{prefix}_robot"),
            frame_ref=Reference("frame", right_wheel_frame_id),
            extensions=_extension("wheel_diameter_m", values),
        ),
    )

    sensor_entities: list[Entity] = []
    frames: list[Frame] = [
        Frame(f"frame:{prefix}_world", "Warehouse world"),
        Frame(base_frame_id, "Base link", Reference("entity", f"entity:{prefix}_robot")),
        Frame(chassis_frame_id, "Chassis frame", Reference("entity", f"entity:{prefix}_chassis")),
        Frame(
            left_wheel_frame_id,
            "Left wheel frame",
            Reference("entity", f"entity:{prefix}_left_wheel"),
        ),
        Frame(
            right_wheel_frame_id,
            "Right wheel frame",
            Reference("entity", f"entity:{prefix}_right_wheel"),
        ),
    ]
    if has_lidar:
        sensor_entities.append(
            Entity(
                f"entity:{prefix}_lidar",
                "2D safety lidar",
                EntityKind.ELECTRICAL,
                parent_entity=Reference("entity", f"entity:{prefix}_robot"),
                frame_ref=Reference("frame", lidar_frame_id),
                extensions=_extension("has_lidar", values),
            )
        )
        frames.append(
            Frame(lidar_frame_id, "Lidar frame", Reference("entity", f"entity:{prefix}_lidar"))
        )
    if has_imu:
        sensor_entities.append(
            Entity(
                f"entity:{prefix}_imu",
                "Base IMU",
                EntityKind.ELECTRICAL,
                parent_entity=Reference("entity", f"entity:{prefix}_robot"),
                frame_ref=Reference("frame", imu_frame_id),
                extensions=_extension("has_imu", values),
            )
        )
        frames.append(Frame(imu_frame_id, "IMU frame", Reference("entity", f"entity:{prefix}_imu")))

    geometries = (
        Geometry.box(
            f"geometry:{prefix}_chassis_box",
            q(length_m, "m", LENGTH),
            q(width_m, "m", LENGTH),
            q(height_m, "m", LENGTH),
            provenance=generated,
            extensions=_extension("chassis_length_m", values),
        ),
        Geometry(
            f"geometry:{prefix}_wheel_cylinder",
            GeometryKind.CYLINDER,
            radius=q(wheel_radius_m, "m", LENGTH),
            length=q(wheel_width_m, "m", LENGTH),
            provenance=generated,
            extensions=_extension("wheel_diameter_m", values),
        ),
    )

    chassis_mass = max(35.0, 18.0 + payload_kg * 1.5)
    wheel_mass = max(1.5, chassis_mass * 0.04)
    bodies = (
        Body(
            f"body:{prefix}_chassis",
            Reference("entity", f"entity:{prefix}_chassis"),
            Reference("frame", chassis_frame_id),
            visual_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_chassis_box"), GeometryRole.VISUAL
                ),
            ),
            collision_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_chassis_box"),
                    GeometryRole.COLLISION,
                ),
            ),
            mass_properties=_box_mass(chassis_mass, length_m, width_m, height_m),
            extensions=_extension("payload_kg", values),
        ),
        Body(
            f"body:{prefix}_left_wheel",
            Reference("entity", f"entity:{prefix}_left_wheel"),
            Reference("frame", left_wheel_frame_id),
            visual_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_wheel_cylinder"),
                    GeometryRole.VISUAL,
                ),
            ),
            collision_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_wheel_cylinder"),
                    GeometryRole.COLLISION,
                ),
            ),
            mass_properties=_wheel_mass(wheel_mass, wheel_radius_m, wheel_width_m),
        ),
        Body(
            f"body:{prefix}_right_wheel",
            Reference("entity", f"entity:{prefix}_right_wheel"),
            Reference("frame", right_wheel_frame_id),
            visual_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_wheel_cylinder"),
                    GeometryRole.VISUAL,
                ),
            ),
            collision_geometry=(
                GeometryBinding(
                    Reference("component", f"geometry:{prefix}_wheel_cylinder"),
                    GeometryRole.COLLISION,
                ),
            ),
            mass_properties=_wheel_mass(wheel_mass, wheel_radius_m, wheel_width_m),
        ),
    )

    transforms: list[Transform] = [
        Transform(
            f"transform:{prefix}_world_base",
            Reference("frame", f"frame:{prefix}_world"),
            Reference("frame", base_frame_id),
            identity_pose(f"frame:{prefix}_world"),
            user,
        ),
        Transform(
            f"transform:{prefix}_base_chassis",
            Reference("frame", base_frame_id),
            Reference("frame", chassis_frame_id),
            Pose(Reference("frame", base_frame_id), zero_vector()),
            calculated,
        ),
        Transform(
            f"transform:{prefix}_base_left_wheel",
            Reference("frame", base_frame_id),
            Reference("frame", left_wheel_frame_id),
            Pose(
                Reference("frame", base_frame_id),
                Vector3(
                    q(0.0, "m", LENGTH),
                    q(width_m / 2.0 + wheel_width_m / 2.0, "m", LENGTH),
                    q(-(height_m / 2.0 - wheel_radius_m), "m", LENGTH),
                ),
            ),
            calculated,
        ),
        Transform(
            f"transform:{prefix}_base_right_wheel",
            Reference("frame", base_frame_id),
            Reference("frame", right_wheel_frame_id),
            Pose(
                Reference("frame", base_frame_id),
                Vector3(
                    q(0.0, "m", LENGTH),
                    q(-(width_m / 2.0 + wheel_width_m / 2.0), "m", LENGTH),
                    q(-(height_m / 2.0 - wheel_radius_m), "m", LENGTH),
                ),
            ),
            calculated,
        ),
    ]
    if has_lidar:
        transforms.append(
            Transform(
                f"transform:{prefix}_base_lidar",
                Reference("frame", base_frame_id),
                Reference("frame", lidar_frame_id),
                Pose(
                    Reference("frame", base_frame_id),
                    Vector3(
                        q(length_m * 0.35, "m", LENGTH),
                        q(0.0, "m", LENGTH),
                        q(height_m / 2.0 + 0.08, "m", LENGTH),
                    ),
                ),
                calculated,
            )
        )
    if has_imu:
        transforms.append(
            Transform(
                f"transform:{prefix}_base_imu",
                Reference("frame", base_frame_id),
                Reference("frame", imu_frame_id),
                Pose(Reference("frame", base_frame_id), zero_vector()),
                calculated,
            )
        )

    left_joint = Joint(
        f"joint:{prefix}_left_wheel",
        "Left wheel continuous joint",
        JointKind.CONTINUOUS,
        Reference("body", f"body:{prefix}_chassis"),
        Reference("body", f"body:{prefix}_left_wheel"),
        axis=(0.0, 1.0, 0.0),
        provenance=generated,
    )
    right_joint = Joint(
        f"joint:{prefix}_right_wheel",
        "Right wheel continuous joint",
        JointKind.CONTINUOUS,
        Reference("body", f"body:{prefix}_chassis"),
        Reference("body", f"body:{prefix}_right_wheel"),
        axis=(0.0, 1.0, 0.0),
        provenance=generated,
    )
    actuators = (
        Actuator(
            f"actuator:{prefix}_left_drive",
            "Left differential-drive actuator",
            ActuatorKind.ROTARY,
            controlled_joint=Reference("joint", left_joint.id),
            mount_frame=Reference("frame", left_wheel_frame_id),
            limits={"vehicle_speed": q(max_speed_mps, "m/s", "speed")},
            extensions=_extension("max_speed_mps", values),
        ),
        Actuator(
            f"actuator:{prefix}_right_drive",
            "Right differential-drive actuator",
            ActuatorKind.ROTARY,
            controlled_joint=Reference("joint", right_joint.id),
            mount_frame=Reference("frame", right_wheel_frame_id),
            limits={"vehicle_speed": q(max_speed_mps, "m/s", "speed")},
            extensions=_extension("max_speed_mps", values),
        ),
    )

    electrical_devices = (
        ElectricalDevice(
            f"electrical:{prefix}_battery",
            "Battery",
            "battery",
            power_source=True,
            voltage_nominal=q(24.0, "V", "voltage"),
            power_limit=q(720.0, "W", "power"),
        ),
        ElectricalDevice(
            f"electrical:{prefix}_motor_driver",
            "Differential drive motor driver",
            "motor_driver",
            entity_ref=Reference("entity", f"entity:{prefix}_robot"),
        ),
        ElectricalDevice(
            f"electrical:{prefix}_compute",
            "Robot compute",
            "compute_board",
            entity_ref=Reference("entity", f"entity:{prefix}_robot"),
        ),
    )
    electrical_ports = (
        ElectricalPort(
            f"port:{prefix}_battery_out",
            Reference("electrical_device", f"electrical:{prefix}_battery"),
            "out",
            voltage_limit=q(24.0, "V", "voltage"),
            current_limit=q(30.0, "A", "current"),
        ),
        ElectricalPort(
            f"port:{prefix}_driver_power",
            Reference("electrical_device", f"electrical:{prefix}_motor_driver"),
            "power",
        ),
        ElectricalPort(
            f"port:{prefix}_compute_power",
            Reference("electrical_device", f"electrical:{prefix}_compute"),
            "power",
        ),
    )
    power_connections = (
        PowerConnection(
            f"power:{prefix}_battery_driver",
            Reference("electrical_port", f"port:{prefix}_battery_out"),
            Reference("electrical_port", f"port:{prefix}_driver_power"),
            voltage=q(24.0, "V", "voltage"),
        ),
        PowerConnection(
            f"power:{prefix}_battery_compute",
            Reference("electrical_port", f"port:{prefix}_battery_out"),
            Reference("electrical_port", f"port:{prefix}_compute_power"),
            voltage=q(24.0, "V", "voltage"),
        ),
    )
    bus = CommunicationBus(f"bus:{prefix}_ethernet", "Robot Ethernet", BusKind.ETHERNET)
    interfaces: list[CommunicationInterface] = [
        CommunicationInterface(
            f"interface:{prefix}_compute_eth",
            Reference("compute_device", f"compute:{prefix}_main"),
            Reference("bus", bus.id),
            "compute Ethernet",
            update_rate_hz=100.0,
        )
    ]
    sensors: list[Sensor] = []
    software_inputs: list[Reference] = []
    if has_lidar:
        interfaces.append(
            CommunicationInterface(
                f"interface:{prefix}_lidar_eth",
                Reference("sensor", f"sensor:{prefix}_lidar"),
                Reference("bus", bus.id),
                "lidar Ethernet",
                update_rate_hz=20.0,
            )
        )
        sensors.append(
            Sensor(
                f"sensor:{prefix}_lidar",
                "2D safety lidar",
                SensorModality.LIDAR_2D,
                Reference("frame", lidar_frame_id),
                update_rate_hz=20.0,
                field_of_view=FieldOfView(horizontal=q(270.0, "deg", "angle")),
                range=QuantityRange(minimum=q(0.1, "m", LENGTH), maximum=q(25.0, "m", LENGTH)),
                interface_refs=(Reference("component", f"interface:{prefix}_lidar_eth"),),
                extensions=_extension("has_lidar", values),
            )
        )
        software_inputs.append(Reference("sensor", f"sensor:{prefix}_lidar"))
    if has_imu:
        interfaces.append(
            CommunicationInterface(
                f"interface:{prefix}_imu_bus",
                Reference("sensor", f"sensor:{prefix}_imu"),
                Reference("bus", bus.id),
                "IMU data interface",
                update_rate_hz=200.0,
            )
        )
        sensors.append(
            Sensor(
                f"sensor:{prefix}_imu",
                "Base IMU",
                SensorModality.IMU,
                Reference("frame", imu_frame_id),
                update_rate_hz=200.0,
                interface_refs=(Reference("component", f"interface:{prefix}_imu_bus"),),
                extensions=_extension("has_imu", values),
            )
        )
        software_inputs.append(Reference("sensor", f"sensor:{prefix}_imu"))

    compute = ComputeDevice(
        f"compute:{prefix}_main",
        "Main robot compute",
        Reference("electrical_device", f"electrical:{prefix}_compute"),
        processor="arm64",
        interface_refs=(Reference("component", f"interface:{prefix}_compute_eth"),),
        firmware_reference="firmware/warehouse_inspection",
    )
    software = (
        SoftwareComponent(
            f"software:{prefix}_diff_drive_controller",
            "Differential drive controller",
            "motion_control",
            compute_ref=Reference("compute_device", compute.id),
            input_refs=tuple(software_inputs),
            output_refs=(
                Reference("actuator", actuators[0].id),
                Reference("actuator", actuators[1].id),
            ),
            timing={"period": q(0.02, "s", "time")},
        ),
        SoftwareComponent(
            f"software:{prefix}_inspection_mission",
            "Warehouse inspection mission",
            "mission",
            compute_ref=Reference("compute_device", compute.id),
            input_refs=tuple(software_inputs),
            timing={"period": q(0.1, "s", "time")},
            extensions=_extension("mission", values),
        ),
    )

    requirements = _requirements(prefix, values)
    assumptions = _assumptions(prefix, values, payload_kg, chassis_mass)
    decisions = (
        EngineeringDecision(
            f"decision:{prefix}_differential_drive",
            "Use differential drive for warehouse inspection.",
            "The accepted operation is limited to differential-drive mobile robots "
            "on indoor warehouse floors.",
            generated,
        ),
        EngineeringDecision(
            f"decision:{prefix}_default_policy",
            "Record defaults separately from user-provided design requirements.",
            "Missing physical dimensions and sensor flags are compiled from the "
            "application default policy and marked in namespaced extensions.",
            generated,
        ),
    )
    scenario = SimulationScenario(
        f"scenario:{prefix}_warehouse_inspection",
        "Warehouse inspection scenario",
        description=f"Execute mission: {mission}",
        environment=environment,
        duration=q(120.0, "s", "time"),
        initial_state_refs=(Reference("entity", f"entity:{prefix}_robot"),),
        validation_refs=tuple(
            Reference("requirement", requirement.id) for requirement in requirements
        ),
        extensions=_extension("warehouse_environment", values),
    )

    return {
        "requirements": requirements,
        "assumptions": assumptions,
        "entities": (*entities, *sensor_entities),
        "frames": tuple(frames),
        "transforms": tuple(transforms),
        "geometries": geometries,
        "bodies": bodies,
        "joints": (left_joint, right_joint),
        "actuators": actuators,
        "sensors": tuple(sensors),
        "electrical_devices": electrical_devices,
        "electrical_ports": electrical_ports,
        "power_connections": power_connections,
        "communication_buses": (bus,),
        "communication_interfaces": tuple(interfaces),
        "compute_devices": (compute,),
        "software_components": software,
        "simulation_scenarios": (scenario,),
        "engineering_decisions": decisions,
    }


def _requirements(prefix: str, values: dict[str, DesignValue]) -> tuple[Requirement, ...]:
    generated = Provenance(ProvenanceSource.GENERATED, "warehouse inspection robot compiler")
    accepted = RequirementStatus.ACCEPTED
    payload = float(values["payload_kg"].value)
    speed = float(values["max_speed_mps"].value)
    length = float(values["chassis_length_m"].value)
    width = float(values["chassis_width_m"].value)
    height = float(values["chassis_height_m"].value)
    wheel_diameter = float(values["wheel_diameter_m"].value)
    wheel_width = float(values["wheel_width_m"].value)
    return (
        Requirement(
            f"requirement:{prefix}_mission",
            RequirementKind.USER_DEFINED,
            f"Robot mission: {values['mission'].value}",
            RequirementPriority.HIGH,
            status=accepted,
            value=str(values["mission"].value),
            provenance=generated,
            extensions=_extension("mission", values),
        ),
        Requirement(
            f"requirement:{prefix}_payload",
            RequirementKind.PAYLOAD,
            f"Carry an inspection payload of at least {payload:g} kg.",
            RequirementPriority.HIGH,
            status=accepted,
            value=QuantityRange(minimum=q(payload, "kg", "mass")),
            provenance=generated,
            extensions=_extension("payload_kg", values),
        ),
        Requirement(
            f"requirement:{prefix}_max_speed",
            RequirementKind.SPEED,
            f"Limit warehouse operating speed to at most {speed:g} m/s.",
            RequirementPriority.HIGH,
            status=accepted,
            value=QuantityRange(maximum=q(speed, "m/s", "speed")),
            provenance=generated,
            extensions=_extension("max_speed_mps", values),
        ),
        Requirement(
            f"requirement:{prefix}_chassis_dimensions",
            RequirementKind.DIMENSIONS,
            f"Use chassis dimensions {length:g} m x {width:g} m x {height:g} m.",
            RequirementPriority.MEDIUM,
            status=accepted,
            value=f"{length:g} m x {width:g} m x {height:g} m",
            provenance=generated,
            extensions={
                DESIGN_EXTENSION_NS: {
                    "fields": {
                        "chassis_length_m": values["chassis_length_m"].to_dict(),
                        "chassis_width_m": values["chassis_width_m"].to_dict(),
                        "chassis_height_m": values["chassis_height_m"].to_dict(),
                    }
                }
            },
        ),
        Requirement(
            f"requirement:{prefix}_wheel_dimensions",
            RequirementKind.DIMENSIONS,
            f"Use drive wheels {wheel_diameter:g} m diameter x {wheel_width:g} m width.",
            RequirementPriority.MEDIUM,
            status=accepted,
            value=f"{wheel_diameter:g} m diameter x {wheel_width:g} m width",
            provenance=generated,
            extensions={
                DESIGN_EXTENSION_NS: {
                    "fields": {
                        "wheel_diameter_m": values["wheel_diameter_m"].to_dict(),
                        "wheel_width_m": values["wheel_width_m"].to_dict(),
                    }
                }
            },
        ),
        Requirement(
            f"requirement:{prefix}_environment",
            RequirementKind.ENVIRONMENT,
            f"Operate in warehouse environment: {values['warehouse_environment'].value}.",
            RequirementPriority.HIGH,
            status=accepted,
            value=str(values["warehouse_environment"].value),
            provenance=generated,
            extensions=_extension("warehouse_environment", values),
        ),
    )


def _assumptions(
    prefix: str, values: dict[str, DesignValue], payload_kg: float, chassis_mass_kg: float
) -> tuple[Assumption, ...]:
    generated = Provenance(ProvenanceSource.GENERATED, "warehouse inspection robot compiler")
    assumptions = [
        Assumption(
            f"assumption:{prefix}_flat_dry_floor",
            "Warehouse operation assumes a flat, dry indoor floor unless later "
            "requirements say otherwise.",
            generated,
        ),
        Assumption(
            f"assumption:{prefix}_payload_distribution",
            f"The {payload_kg:g} kg payload is centered over the chassis for "
            "first-pass mass properties.",
            generated,
        ),
        Assumption(
            f"assumption:{prefix}_estimated_chassis_mass",
            f"Chassis mass is estimated as {chassis_mass_kg:g} kg for canonical "
            "dynamics placeholders.",
            generated,
        ),
    ]
    for field, value in sorted(values.items()):
        if value.source in {DesignValueSource.DEFAULT_POLICY, DesignValueSource.ASSUMPTION}:
            assumptions.append(
                Assumption(
                    f"assumption:{prefix}_{field}",
                    f"{field} uses {value.source.value} value {value.value!r}.",
                    generated,
                )
            )
    return tuple(assumptions)


def _box_mass(mass_kg: float, length_m: float, width_m: float, height_m: float) -> MassProperties:
    mass = q(mass_kg, "kg", "mass")
    ixx = mass_kg * (width_m**2 + height_m**2) / 12.0
    iyy = mass_kg * (length_m**2 + height_m**2) / 12.0
    izz = mass_kg * (length_m**2 + width_m**2) / 12.0
    return MassProperties(
        mass, zero_vector(), _diagonal_inertia(ixx, iyy, izz), MassProvenance.DERIVED
    )


def _wheel_mass(mass_kg: float, radius_m: float, width_m: float) -> MassProperties:
    mass = q(mass_kg, "kg", "mass")
    i_axis = 0.5 * mass_kg * radius_m**2
    i_cross = (mass_kg * (3.0 * radius_m**2 + width_m**2)) / 12.0
    return MassProperties(
        mass,
        zero_vector(),
        _diagonal_inertia(i_cross, i_axis, i_cross),
        MassProvenance.DERIVED,
    )


def _diagonal_inertia(
    ixx: float, iyy: float, izz: float
) -> tuple[tuple[Quantity, Quantity, Quantity], ...]:
    zero = q(0.0, "kg*m^2", "inertia")
    return (
        (q(ixx, "kg*m^2", "inertia"), zero, zero),
        (zero, q(iyy, "kg*m^2", "inertia"), zero),
        (zero, zero, q(izz, "kg*m^2", "inertia")),
    )


def _numeric_or_existing_requirement(
    project: EngineeringProject,
    value: float | None,
    *,
    field: str,
    kind: RequirementKind,
    default: float,
    positive: bool,
) -> DesignValue:
    if value is not None:
        return _defaulted_number(value, field, default, positive=positive, user_supplied=True)
    existing = _requirement_quantity_value(project, kind)
    if existing is not None:
        return DesignValue(
            existing[0],
            DesignValueSource.EXISTING_REQUIREMENT,
            field,
            requirement_id=existing[1],
        )
    return _defaulted_number(None, field, default, positive=positive)


def _requirement_quantity_value(
    project: EngineeringProject, kind: RequirementKind
) -> tuple[float, str] | None:
    for requirement in project.requirements:
        if requirement.kind is not kind:
            continue
        value = requirement.value
        quantity: Quantity | None = None
        if isinstance(value, Quantity):
            quantity = value
        elif isinstance(value, QuantityRange):
            quantity = value.minimum or value.maximum or value.target
        if quantity is None:
            continue
        target_unit = "kg" if kind is RequirementKind.PAYLOAD else "m/s"
        try:
            return quantity.to(target_unit).value, requirement.id
        except ModelError:
            continue
    return None


def _defaulted_number(
    value: float | None,
    field: str,
    default: float,
    *,
    positive: bool,
    user_supplied: bool = False,
) -> DesignValue:
    selected = default if value is None else value
    if (
        not isinstance(selected, int | float)
        or isinstance(selected, bool)
        or not math.isfinite(selected)
    ):
        raise DesignError(f"{field} must be a finite number")
    numeric = float(selected)
    if positive and numeric <= 0.0:
        raise DesignError(f"{field} must be greater than zero")
    if not positive and numeric < 0.0:
        raise DesignError(f"{field} must be non-negative")
    source = (
        DesignValueSource.USER_PROVIDED
        if user_supplied or value is not None
        else DesignValueSource.DEFAULT_POLICY
    )
    return DesignValue(numeric, source, field)


def _defaulted_bool(value: bool | None, field: str, default: bool) -> DesignValue:
    if value is None:
        return DesignValue(default, DesignValueSource.DEFAULT_POLICY, field)
    if not isinstance(value, bool):
        raise DesignError(f"{field} must be true or false")
    return DesignValue(value, DesignValueSource.USER_PROVIDED, field)


def _validate_relative_dimensions(values: dict[str, DesignValue]) -> None:
    wheel_diameter = float(values["wheel_diameter_m"].value)
    chassis_height = float(values["chassis_height_m"].value)
    wheel_width = float(values["wheel_width_m"].value)
    chassis_width = float(values["chassis_width_m"].value)
    if wheel_diameter > chassis_height * 1.5:
        raise DesignError("wheel_diameter_m is implausibly large relative to chassis_height_m")
    if wheel_width > chassis_width * 0.4:
        raise DesignError("wheel_width_m is implausibly large relative to chassis_width_m")


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DesignError(f"{field} must not be empty")
    return " ".join(value.split())


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_") or "robot"
    if len(text) == 1:
        text = f"{text}_robot"
    return text[:48]


def _extension(field: str, values: dict[str, DesignValue]) -> dict[str, Any]:
    return {DESIGN_EXTENSION_NS: values[field].to_dict()}


def _generated_ids(prefix: str, values: dict[str, DesignValue]) -> set[str]:
    elements = _build_project_elements(prefix, values)
    ids: set[str] = set()
    for items in elements.values():
        for item in items:
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str):
                ids.add(item_id)
    return ids


def _reject_existing_ids(project: EngineeringProject, generated_ids: set[str]) -> None:
    existing = {
        item.id
        for collection_name in EngineeringProject.COLLECTIONS
        for item in getattr(project, collection_name)
    }
    conflicts = sorted(existing & generated_ids)
    if conflicts:
        digest = hashlib.sha256(",".join(conflicts).encode("utf-8")).hexdigest()[:8]
        raise DesignError(
            "warehouse inspection robot design already conflicts with existing project IDs "
            f"({len(conflicts)} conflicts, digest {digest})"
        )
