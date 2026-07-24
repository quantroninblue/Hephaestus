"""Deterministic engineering project validation."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hephaestus.index import ProjectIndex, iter_references_in_value
from hephaestus.model import (
    CAPACITY,
    CURRENT,
    ENERGY,
    INERTIA,
    POWER,
    SCHEMA_VERSION,
    VOLTAGE,
    EngineeringProject,
    GeometryKind,
    JointKind,
    Quantity,
    Reference,
)


class ValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    severity: ValidationSeverity
    message: str
    path: str | None = None
    evidence: Any | None = None
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "code": self.code,
                "severity": self.severity.value,
                "message": self.message,
                "path": self.path,
                "evidence": self.evidence,
                "remediation": self.remediation,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def error_count(self) -> int:
        return sum(
            f.severity in {ValidationSeverity.ERROR, ValidationSeverity.FATAL}
            for f in self.findings
        )

    @property
    def warning_count(self) -> int:
        return sum(f.severity is ValidationSeverity.WARNING for f in self.findings)

    @property
    def info_count(self) -> int:
        return sum(f.severity is ValidationSeverity.INFO for f in self.findings)

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "valid" if self.is_valid else "invalid",
            "summary": {
                "info": self.info_count,
                "warning": self.warning_count,
                "error": self.error_count,
                "total": len(self.findings),
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


Validator = Callable[[EngineeringProject, ProjectIndex], Iterable[ValidationFinding]]
_VALIDATORS: list[Validator] = []


def register_validator(validator: Validator) -> Validator:
    _VALIDATORS.append(validator)
    return validator


def validate_project(project: EngineeringProject) -> ValidationReport:
    index = ProjectIndex.build(project)
    findings: list[ValidationFinding] = []
    for validator in _VALIDATORS:
        findings.extend(validator(project, index))
    findings.sort(key=lambda item: (item.severity.value, item.code, item.path or "", item.message))
    if not findings:
        findings.append(
            ValidationFinding(
                code="project.valid",
                severity=ValidationSeverity.INFO,
                message="Project satisfies registered foundational validators.",
            )
        )
    return ValidationReport(tuple(findings))


def _ids(items: Iterable[Any]) -> set[str]:
    return {item.id for item in items}


def _quantity_positive(
    quantity: Quantity, path: str, code: str, label: str
) -> ValidationFinding | None:
    if quantity.si_value <= 0.0:
        return ValidationFinding(
            code, ValidationSeverity.ERROR, f"{label} must be positive.", path, quantity.to_dict()
        )
    return None


def _quantity_non_negative(
    quantity: Quantity, path: str, code: str, label: str
) -> ValidationFinding | None:
    if quantity.si_value < 0.0:
        return ValidationFinding(
            code,
            ValidationSeverity.ERROR,
            f"{label} must be non-negative.",
            path,
            quantity.to_dict(),
        )
    return None


@register_validator
def validate_schema(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    del index
    if project.schema_version != SCHEMA_VERSION:
        yield ValidationFinding(
            "project.schema.unsupported",
            ValidationSeverity.FATAL,
            "Project schema version is not supported.",
            "schema_version",
            project.schema_version,
            f"Use schema version {SCHEMA_VERSION}.",
        )


@register_validator
def validate_duplicate_ids(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    for collection_name in EngineeringProject.COLLECTIONS:
        seen: set[str] = set()
        for item_index, item in enumerate(getattr(project, collection_name)):
            if item.id in seen:
                yield ValidationFinding(
                    "id.duplicate",
                    ValidationSeverity.ERROR,
                    f"Duplicate id {item.id!r} in {collection_name}.",
                    f"{collection_name}[{item_index}].id",
                    item.id,
                    "Use stable identifiers that are unique within each collection.",
                )
            seen.add(item.id)

    for id_value, items in index.global_ids.items():
        if len(items) > 1:
            yield ValidationFinding(
                "id.global_duplicate",
                ValidationSeverity.ERROR,
                f"Identifier {id_value!r} appears in multiple engineering collections.",
                None,
                [item.path for item in items],
                "Use globally unique stable identifiers.",
            )

    for id_value, items in index.component_ambiguities().items():
        yield ValidationFinding(
            "reference.component_ambiguous",
            ValidationSeverity.ERROR,
            f"Component reference id {id_value!r} is ambiguous across component-like collections.",
            None,
            [item.path for item in items],
            "Rename one item so component references resolve to one object.",
        )


@register_validator
def validate_references(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    del project
    for root_path, edges in index.outgoing.items():
        for edge in edges:
            if index.resolve(edge.target) is None:
                yield ValidationFinding(
                    "reference.dangling",
                    ValidationSeverity.ERROR,
                    f"Reference points to missing {edge.target.kind} {edge.target.id!r}.",
                    edge.source_path,
                    edge.target.to_dict(),
                    "Create the referenced item or remove the reference.",
                )
        del root_path


@register_validator
def validate_field_specific_references(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    for item_index, geometry in enumerate(project.geometries):
        if (
            geometry.material_ref is not None
            and index.resolve_component_as(geometry.material_ref, {"materials"}) is None
        ):
            yield ValidationFinding(
                "geometry.material_ref.invalid",
                ValidationSeverity.ERROR,
                "Geometry material_ref must resolve to exactly one material.",
                f"geometries[{item_index}].material_ref",
                geometry.material_ref.to_dict(),
            )
    for body_index, body in enumerate(project.bodies):
        for binding_index, binding in enumerate(body.visual_geometry):
            if index.resolve_component_as(binding.geometry_ref, {"geometries"}) is None:
                yield ValidationFinding(
                    "body.visual_geometry.invalid",
                    ValidationSeverity.ERROR,
                    "Visual geometry binding must resolve to exactly one geometry.",
                    f"bodies[{body_index}].visual_geometry[{binding_index}].geometry_ref",
                    binding.geometry_ref.to_dict(),
                )
        for binding_index, binding in enumerate(body.collision_geometry):
            if index.resolve_component_as(binding.geometry_ref, {"geometries"}) is None:
                yield ValidationFinding(
                    "body.collision_geometry.invalid",
                    ValidationSeverity.ERROR,
                    "Collision geometry binding must resolve to exactly one geometry.",
                    f"bodies[{body_index}].collision_geometry[{binding_index}].geometry_ref",
                    binding.geometry_ref.to_dict(),
                )


def _iter_quantities(value: Any, path: str) -> Iterable[tuple[str, Quantity]]:
    if isinstance(value, Quantity):
        yield path, value
    elif isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from _iter_quantities(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_quantities(item, f"{path}.{key}")
    elif hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            if name == "COLLECTIONS":
                continue
            yield from _iter_quantities(getattr(value, name), f"{path}.{name}")


@register_validator
def validate_physical_values(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    del index
    for collection_name in EngineeringProject.COLLECTIONS:
        for item_index, item in enumerate(getattr(project, collection_name)):
            for path, quantity in _iter_quantities(item, f"{collection_name}[{item_index}]"):
                if not math.isfinite(quantity.value) or not math.isfinite(quantity.si_value):
                    yield ValidationFinding(
                        "quantity.non_finite",
                        ValidationSeverity.ERROR,
                        "Quantity values must be finite.",
                        path,
                        quantity.to_dict(),
                    )

    for item_index, geometry in enumerate(project.geometries):
        for dim_index, dimension in enumerate(geometry.dimensions):
            finding = _quantity_positive(
                dimension,
                f"geometries[{item_index}].dimensions[{dim_index}]",
                "geometry.dimension.non_positive",
                "Geometry dimension",
            )
            if finding:
                yield finding
        for name in ("radius", "length"):
            quantity = getattr(geometry, name)
            if quantity is not None:
                finding = _quantity_positive(
                    quantity,
                    f"geometries[{item_index}].{name}",
                    f"geometry.{name}.non_positive",
                    f"Geometry {name}",
                )
                if finding:
                    yield finding
        if geometry.kind == GeometryKind.MESH and geometry.mesh_format is None:
            yield ValidationFinding(
                "geometry.mesh.format_missing",
                ValidationSeverity.ERROR,
                "Mesh geometry must specify a mesh format.",
                f"geometries[{item_index}].mesh_format",
            )

    for body_index, body in enumerate(project.bodies):
        if body.mass_properties is None:
            continue
        mass = body.mass_properties.mass
        finding = _quantity_non_negative(
            mass,
            f"bodies[{body_index}].mass_properties.mass",
            "mass.negative",
            "Mass",
        )
        if finding:
            yield finding
        tensor = body.mass_properties.inertia_tensor
        yield from _validate_inertia_tensor(
            tensor, f"bodies[{body_index}].mass_properties.inertia_tensor"
        )

    for transform_index, transform in enumerate(project.transforms):
        orientation = transform.pose.orientation
        if not orientation.is_valid_rotation():
            yield ValidationFinding(
                "pose.quaternion.invalid",
                ValidationSeverity.ERROR,
                "Quaternion must be normalizable and nonzero.",
                f"transforms[{transform_index}].pose.orientation",
                orientation.to_dict(),
            )
        elif not orientation.is_normalized():
            yield ValidationFinding(
                "pose.quaternion.not_normalized",
                ValidationSeverity.WARNING,
                "Quaternion is normalizable but not normalized.",
                f"transforms[{transform_index}].pose.orientation",
                orientation.to_dict(),
                "Normalize the quaternion before backend export.",
            )

    for sensor_index, sensor in enumerate(project.sensors):
        if sensor.update_rate_hz is not None and (
            not math.isfinite(sensor.update_rate_hz) or sensor.update_rate_hz < 0.0
        ):
            yield ValidationFinding(
                "sensor.update_rate.invalid",
                ValidationSeverity.ERROR,
                "Sensor update rate must be finite and non-negative.",
                f"sensors[{sensor_index}].update_rate_hz",
                sensor.update_rate_hz,
            )
        if sensor.latency is not None:
            finding = _quantity_non_negative(
                sensor.latency,
                f"sensors[{sensor_index}].latency",
                "sensor.latency.negative",
                "Sensor latency",
            )
            if finding:
                yield finding
        if sensor.range is not None:
            for bound_name in ("minimum", "target", "maximum"):
                quantity = getattr(sensor.range, bound_name)
                if quantity is not None:
                    finding = _quantity_non_negative(
                        quantity,
                        f"sensors[{sensor_index}].range.{bound_name}",
                        "sensor.range.negative",
                        "Sensor range",
                    )
                    if finding:
                        yield finding


def _validate_inertia_tensor(
    tensor: tuple[tuple[Quantity, Quantity, Quantity], ...], path: str
) -> Iterable[ValidationFinding]:
    if len(tensor) != 3 or any(len(row) != 3 for row in tensor):
        yield ValidationFinding(
            "inertia.structure.invalid",
            ValidationSeverity.ERROR,
            "Inertia tensor must be 3x3.",
            path,
        )
        return
    values: list[list[float]] = []
    for row_index, row in enumerate(tensor):
        value_row: list[float] = []
        for col_index, quantity in enumerate(row):
            if quantity.dimension != INERTIA:
                yield ValidationFinding(
                    "inertia.unit.invalid",
                    ValidationSeverity.ERROR,
                    "Inertia tensor value has invalid units.",
                    f"{path}[{row_index}][{col_index}]",
                    quantity.to_dict(),
                )
            value_row.append(quantity.si_value)
        values.append(value_row)
    for index in range(3):
        if values[index][index] < 0.0:
            yield ValidationFinding(
                "inertia.diagonal.negative",
                ValidationSeverity.ERROR,
                "Inertia diagonal terms must be non-negative.",
                f"{path}[{index}][{index}]",
                values[index][index],
            )
    for row in range(3):
        for col in range(row + 1, 3):
            if not math.isclose(values[row][col], values[col][row], rel_tol=1e-9, abs_tol=1e-12):
                yield ValidationFinding(
                    "inertia.symmetry.invalid",
                    ValidationSeverity.ERROR,
                    "Inertia tensor must be symmetric.",
                    f"{path}[{row}][{col}]",
                    {"ij": values[row][col], "ji": values[col][row]},
                )
    ixx, iyy, izz = values[0][0], values[1][1], values[2][2]
    if ixx + iyy < izz or ixx + izz < iyy or iyy + izz < ixx:
        yield ValidationFinding(
            "inertia.principal_moments.implausible",
            ValidationSeverity.ERROR,
            "Principal inertia moments violate triangle inequalities.",
            path,
            {"ixx": ixx, "iyy": iyy, "izz": izz},
        )


@register_validator
def validate_mechanisms(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    del index
    body_ids = _ids(project.bodies)
    for joint_index, joint in enumerate(project.joints):
        if joint.parent_body.id == joint.child_body.id:
            yield ValidationFinding(
                "joint.self_connection",
                ValidationSeverity.ERROR,
                "Joint connects a body to itself.",
                f"joints[{joint_index}]",
                joint.id,
            )
        if joint.parent_body.id not in body_ids:
            yield ValidationFinding(
                "joint.parent_body.missing",
                ValidationSeverity.ERROR,
                "Joint references a missing parent body.",
                f"joints[{joint_index}].parent_body",
                joint.parent_body.to_dict(),
            )
        if joint.child_body.id not in body_ids:
            yield ValidationFinding(
                "joint.child_body.missing",
                ValidationSeverity.ERROR,
                "Joint references a missing child body.",
                f"joints[{joint_index}].child_body",
                joint.child_body.to_dict(),
            )
        if joint.kind in {JointKind.REVOLUTE, JointKind.CONTINUOUS, JointKind.PRISMATIC}:
            axis = joint.axis
            if axis is None or math.sqrt(sum(value * value for value in axis)) == 0.0:
                yield ValidationFinding(
                    "joint.axis.invalid",
                    ValidationSeverity.ERROR,
                    "Joint axis must be nonzero for axial joints.",
                    f"joints[{joint_index}].axis",
                    axis,
                )
            elif not math.isclose(
                math.sqrt(sum(value * value for value in axis)), 1.0, abs_tol=1e-9
            ):
                yield ValidationFinding(
                    "joint.axis.not_normalized",
                    ValidationSeverity.ERROR,
                    "Joint axis must be normalized.",
                    f"joints[{joint_index}].axis",
                    axis,
                )

    for connection_index, connection in enumerate(project.connections):
        if (
            connection.source.kind == connection.target.kind
            and connection.source.id == connection.target.id
        ):
            yield ValidationFinding(
                "connection.self_connection",
                ValidationSeverity.ERROR,
                "Connection links an item to itself.",
                f"connections[{connection_index}]",
                connection.id,
            )


@register_validator
def validate_frames(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    del index
    parents: dict[str, str] = {}
    for transform_index, transform in enumerate(project.transforms):
        child = transform.child_frame.id
        parent = transform.parent_frame.id
        if transform.pose.reference_frame.id != parent:
            yield ValidationFinding(
                "transform.pose_frame.mismatch",
                ValidationSeverity.ERROR,
                "Transform pose reference_frame must match parent_frame.",
                f"transforms[{transform_index}].pose.reference_frame",
                transform.pose.reference_frame.to_dict(),
            )
        if child in parents:
            yield ValidationFinding(
                "frame.multiple_parents",
                ValidationSeverity.ERROR,
                "Frame has more than one parent transform.",
                f"transforms[{transform_index}].child_frame",
                child,
            )
        parents[child] = parent

    for frame_id in parents:
        visited: set[str] = set()
        current = frame_id
        while current in parents:
            if current in visited:
                yield ValidationFinding(
                    "frame.cycle",
                    ValidationSeverity.ERROR,
                    "Frame transforms contain a cycle.",
                    "transforms",
                    current,
                    "Remove or invert one transform in the cycle.",
                )
                break
            visited.add(current)
            current = parents[current]


@register_validator
def validate_sensors_and_electrical(
    project: EngineeringProject, index: ProjectIndex
) -> Iterable[ValidationFinding]:
    frame_ids = _ids(project.frames)
    port_ids = _ids(project.electrical_ports)
    bus_ids = _ids(project.communication_buses)
    for sensor_index, sensor in enumerate(project.sensors):
        if sensor.mounting_frame.id not in frame_ids:
            yield ValidationFinding(
                "sensor.mounting_frame.missing",
                ValidationSeverity.ERROR,
                "Sensor mounting frame is missing.",
                f"sensors[{sensor_index}].mounting_frame",
                sensor.mounting_frame.to_dict(),
            )
        for ref_index, interface_ref in enumerate(sensor.interface_refs):
            if (
                interface_ref.kind != "component"
                or index.resolve_component_as(interface_ref, {"communication_interfaces"}) is None
            ):
                yield ValidationFinding(
                    "sensor.interface.invalid",
                    ValidationSeverity.ERROR,
                    "Sensor interface references must resolve to communication interfaces.",
                    f"sensors[{sensor_index}].interface_refs[{ref_index}]",
                    interface_ref.to_dict(),
                )

    device_ids = _ids(project.electrical_devices)
    for port_index, port in enumerate(project.electrical_ports):
        if port.device_ref.id not in device_ids:
            yield ValidationFinding(
                "electrical.port.device_missing",
                ValidationSeverity.ERROR,
                "Electrical port references a missing device.",
                f"electrical_ports[{port_index}].device_ref",
                port.device_ref.to_dict(),
            )
        for name, dimension in (("voltage_limit", VOLTAGE), ("current_limit", CURRENT)):
            quantity = getattr(port, name)
            if quantity is not None and quantity.dimension != dimension:
                yield ValidationFinding(
                    f"electrical.port.{name}.unit_invalid",
                    ValidationSeverity.ERROR,
                    f"Electrical port {name} has invalid units.",
                    f"electrical_ports[{port_index}].{name}",
                    quantity.to_dict(),
                )
            if quantity is not None:
                finding = _quantity_non_negative(
                    quantity,
                    f"electrical_ports[{port_index}].{name}",
                    f"electrical.port.{name}.negative",
                    f"Electrical port {name}",
                )
                if finding:
                    yield finding

    for device_index, device in enumerate(project.electrical_devices):
        if device.voltage_nominal is not None:
            if device.voltage_nominal.dimension != VOLTAGE:
                yield ValidationFinding(
                    "electrical.device.voltage.unit_invalid",
                    ValidationSeverity.ERROR,
                    "Electrical device nominal voltage has invalid units.",
                    f"electrical_devices[{device_index}].voltage_nominal",
                    device.voltage_nominal.to_dict(),
                )
            finding = _quantity_positive(
                device.voltage_nominal,
                f"electrical_devices[{device_index}].voltage_nominal",
                "electrical.device.voltage.non_positive",
                "Electrical device nominal voltage",
            )
            if finding:
                yield finding
        if device.power_limit is not None:
            if device.power_limit.dimension != POWER:
                yield ValidationFinding(
                    "electrical.device.power.unit_invalid",
                    ValidationSeverity.ERROR,
                    "Electrical device power limit has invalid units.",
                    f"electrical_devices[{device_index}].power_limit",
                    device.power_limit.to_dict(),
                )
            finding = _quantity_non_negative(
                device.power_limit,
                f"electrical_devices[{device_index}].power_limit",
                "electrical.device.power.negative",
                "Electrical device power limit",
            )
            if finding:
                yield finding
        capacity = device.extensions.get("battery:capacity")
        if (
            device.device_type == "battery"
            and isinstance(capacity, Quantity)
            and capacity.dimension not in {CAPACITY, ENERGY}
        ):
            yield ValidationFinding(
                "battery.capacity.unit_invalid",
                ValidationSeverity.ERROR,
                "Battery capacity must use capacity or energy units.",
                f"electrical_devices[{device_index}].extensions.battery:capacity",
                capacity.to_dict(),
            )

    for connection_index, connection in enumerate(project.power_connections):
        if connection.source_port.id == connection.target_port.id:
            yield ValidationFinding(
                "electrical.connection.self_connection",
                ValidationSeverity.ERROR,
                "Power connection cannot connect a port to itself.",
                f"power_connections[{connection_index}]",
                connection.id,
            )
        if connection.source_port.id not in port_ids:
            yield ValidationFinding(
                "electrical.connection.source_missing",
                ValidationSeverity.ERROR,
                "Power connection source port is missing.",
                f"power_connections[{connection_index}].source_port",
                connection.source_port.to_dict(),
            )
        if connection.target_port.id not in port_ids:
            yield ValidationFinding(
                "electrical.connection.target_missing",
                ValidationSeverity.ERROR,
                "Power connection target port is missing.",
                f"power_connections[{connection_index}].target_port",
                connection.target_port.to_dict(),
            )
        for name in ("voltage", "current_limit"):
            quantity = getattr(connection, name)
            if quantity is not None:
                finding = _quantity_non_negative(
                    quantity,
                    f"power_connections[{connection_index}].{name}",
                    f"electrical.connection.{name}.negative",
                    f"Power connection {name}",
                )
                if finding:
                    yield finding

    for interface_index, interface in enumerate(project.communication_interfaces):
        if interface.bus_ref is not None and interface.bus_ref.id not in bus_ids:
            yield ValidationFinding(
                "communication.bus.missing",
                ValidationSeverity.ERROR,
                "Communication interface references a missing bus.",
                f"communication_interfaces[{interface_index}].bus_ref",
                interface.bus_ref.to_dict(),
            )
        if interface.update_rate_hz is not None and (
            not math.isfinite(interface.update_rate_hz) or interface.update_rate_hz < 0.0
        ):
            yield ValidationFinding(
                "communication.update_rate.invalid",
                ValidationSeverity.ERROR,
                "Communication update rate must be finite and non-negative.",
                f"communication_interfaces[{interface_index}].update_rate_hz",
                interface.update_rate_hz,
            )


def _iter_references(project: EngineeringProject) -> Iterable[tuple[str, Reference]]:
    for collection_name in EngineeringProject.COLLECTIONS:
        for item_index, item in enumerate(getattr(project, collection_name)):
            yield from iter_references_in_value(item, f"{collection_name}[{item_index}]")
