"""Canonical engineering project model for Hephaestus.

Internal coordinate convention:
- right-handed coordinates
- X forward, Y left, Z up
- metres, kilograms, seconds, amperes, radians for canonical calculations
- quaternions are stored in ``w, x, y, z`` order
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import StrEnum
from types import UnionType
from typing import Any, ClassVar, Literal, Self, get_args, get_origin, get_type_hints

from pint import UnitRegistry
from pint.errors import DimensionalityError, PintError, UndefinedUnitError

SCHEMA_VERSION = "1.0.0"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_:-]{1,95}$")


class ModelError(ValueError):
    """Raised when typed model construction or decoding fails."""


class ProvenanceSource(StrEnum):
    USER = "user"
    DOCUMENTATION = "documentation"
    CATALOGUE = "catalogue"
    ESTIMATED = "estimated"
    CALCULATED = "calculated"
    GENERATED = "generated"
    IMPORTED = "imported"


class RequirementKind(StrEnum):
    PAYLOAD = "payload"
    DIMENSIONS = "dimensions"
    MASS = "mass"
    SPEED = "speed"
    ACCELERATION = "acceleration"
    RUNTIME = "runtime"
    RANGE = "range"
    ENVIRONMENT = "environment"
    TERRAIN = "terrain"
    INGRESS_PROTECTION = "ingress_protection"
    SENSING_COVERAGE = "sensing_coverage"
    COMPUTE = "compute"
    SAFETY_FACTOR = "safety_factor"
    USER_DEFINED = "user_defined"


class RequirementPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RequirementStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SATISFIED = "satisfied"
    UNVERIFIED = "unverified"


class VerificationMethod(StrEnum):
    INSPECTION = "inspection"
    ANALYSIS = "analysis"
    TEST = "test"
    SIMULATION = "simulation"
    DEMONSTRATION = "demonstration"
    UNSPECIFIED = "unspecified"


class GeometryKind(StrEnum):
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CAPSULE = "capsule"
    MESH = "mesh"
    PARAMETRIC = "parametric"


class MeshFormat(StrEnum):
    STL = "stl"
    OBJ = "obj"
    DAE = "dae"
    GLTF = "gltf"
    GLB = "glb"
    STEP = "step"
    IGES = "iges"
    BREP = "brep"


class GeometryRole(StrEnum):
    VISUAL = "visual"
    COLLISION = "collision"


class MassProvenance(StrEnum):
    EXPLICIT = "explicit"
    DERIVED = "derived"


class JointKind(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    PLANAR = "planar"
    FLOATING = "floating"
    SPHERICAL = "spherical"


class SensorModality(StrEnum):
    CAMERA = "camera"
    DEPTH = "depth"
    STEREO = "stereo"
    LIDAR_2D = "lidar_2d"
    LIDAR_3D = "lidar_3d"
    IMU = "imu"
    GNSS = "gnss"
    MAGNETOMETER = "magnetometer"
    ALTIMETER = "altimeter"
    SONAR = "sonar"
    CONTACT = "contact"
    FORCE_TORQUE = "force_torque"
    ENCODER = "encoder"
    CUSTOM = "custom"


class ActuatorKind(StrEnum):
    ROTARY = "rotary"
    LINEAR = "linear"
    PROPULSION = "propulsion"
    CUSTOM = "custom"


class BusKind(StrEnum):
    I2C = "i2c"
    SPI = "spi"
    UART = "uart"
    CAN = "can"
    ETHERNET = "ethernet"
    CUSTOM = "custom"


class EntityKind(StrEnum):
    ASSEMBLY = "assembly"
    BODY = "body"
    MECHANISM = "mechanism"
    ELECTRICAL = "electrical"
    COMPUTE = "compute"
    SOFTWARE = "software"
    CUSTOM = "custom"


Dimension = tuple[int, int, int, int, int]
DIMENSIONLESS: Dimension = (0, 0, 0, 0, 0)
LENGTH: Dimension = (1, 0, 0, 0, 0)
MASS: Dimension = (0, 1, 0, 0, 0)
TIME: Dimension = (0, 0, 1, 0, 0)
CURRENT: Dimension = (0, 0, 0, 1, 0)
ANGLE: Dimension = (0, 0, 0, 0, 1)
SPEED: Dimension = (1, 0, -1, 0, 0)
ACCELERATION: Dimension = (1, 0, -2, 0, 0)
FREQUENCY: Dimension = (0, 0, -1, 0, 0)
ANGULAR_VELOCITY: Dimension = (0, 0, -1, 0, 1)
FORCE: Dimension = (1, 1, -2, 0, 0)
TORQUE: Dimension = (2, 1, -2, 0, -1)
PRESSURE: Dimension = (-1, 1, -2, 0, 0)
VOLTAGE: Dimension = (2, 1, -3, -1, 0)
POWER: Dimension = (2, 1, -3, 0, 0)
ENERGY: Dimension = (2, 1, -2, 0, 0)
CAPACITY: Dimension = (0, 0, 1, 1, 0)
DENSITY: Dimension = (-3, 1, 0, 0, 0)
INERTIA: Dimension = (2, 1, 0, 0, 0)

_DIMENSION_NAMES: dict[str, Dimension] = {
    "dimensionless": DIMENSIONLESS,
    "length": LENGTH,
    "mass": MASS,
    "time": TIME,
    "current": CURRENT,
    "angle": ANGLE,
    "speed": SPEED,
    "acceleration": ACCELERATION,
    "frequency": FREQUENCY,
    "angular_velocity": ANGULAR_VELOCITY,
    "force": FORCE,
    "torque": TORQUE,
    "pressure": PRESSURE,
    "voltage": VOLTAGE,
    "power": POWER,
    "energy": ENERGY,
    "capacity": CAPACITY,
    "density": DENSITY,
    "inertia": INERTIA,
}
_DIMENSION_LABELS = {value: key for key, value in _DIMENSION_NAMES.items()}
_UNIT_REGISTRY = UnitRegistry(autoconvert_offset_to_baseunit=True, cache_folder=None)

_PINT_UNIT_ALIASES: dict[str, str] = {
    "1": "dimensionless",
    "%": "percent",
    "N*m": "newton * meter",
    "kg*m^2": "kilogram * meter ** 2",
    "kg/m^3": "kilogram / meter ** 3",
}
_DIMENSION_REPRESENTATIVE_UNITS: dict[Dimension, str] = {
    DIMENSIONLESS: "dimensionless",
    LENGTH: "meter",
    MASS: "kilogram",
    TIME: "second",
    CURRENT: "ampere",
    ANGLE: "radian",
    SPEED: "meter / second",
    ACCELERATION: "meter / second ** 2",
    FREQUENCY: "hertz",
    ANGULAR_VELOCITY: "radian / second",
    FORCE: "newton",
    TORQUE: "newton * meter",
    PRESSURE: "pascal",
    VOLTAGE: "volt",
    POWER: "watt",
    ENERGY: "joule",
    CAPACITY: "ampere * hour",
    DENSITY: "kilogram / meter ** 3",
    INERTIA: "kilogram * meter ** 2",
}
_SEMANTIC_UNIT_DIMENSIONS: dict[str, Dimension] = {
    "1": DIMENSIONLESS,
    "%": DIMENSIONLESS,
    "percent": DIMENSIONLESS,
    "rad": ANGLE,
    "radian": ANGLE,
    "deg": ANGLE,
    "degree": ANGLE,
    "Hz": FREQUENCY,
    "hertz": FREQUENCY,
    "rad/s": ANGULAR_VELOCITY,
    "radian/s": ANGULAR_VELOCITY,
    "deg/s": ANGULAR_VELOCITY,
    "degree/s": ANGULAR_VELOCITY,
    "N*m": TORQUE,
    "newton*meter": TORQUE,
    "newton * meter": TORQUE,
    "J": ENERGY,
    "joule": ENERGY,
    "Wh": ENERGY,
    "watt_hour": ENERGY,
    "Ah": CAPACITY,
    "ampere_hour": CAPACITY,
    "kg*m^2": INERTIA,
    "kilogram*meter^2": INERTIA,
    "kilogram * meter ** 2": INERTIA,
}


def _normalize_unit(unit: str) -> str:
    normalized = unit.strip().replace("²", "^2").replace("³", "^3").replace("·", "*")
    normalized = normalized.replace("**", "^")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _pint_unit(unit: str):
    normalized = _normalize_unit(unit)
    expression = _PINT_UNIT_ALIASES.get(normalized, normalized.replace("^", "**"))
    try:
        return _UNIT_REGISTRY.parse_units(expression)
    except (UndefinedUnitError, PintError) as exc:
        raise ModelError(f"unsupported unit: {unit}") from exc


def _pint_quantity(value: float, unit: str):
    return float(value) * _pint_unit(unit)


def _assert_unit_matches_dimension(unit: str, dimension: Dimension) -> None:
    normalized = _normalize_unit(unit)
    semantic_dimension = _SEMANTIC_UNIT_DIMENSIONS.get(normalized)
    if semantic_dimension is not None and semantic_dimension != dimension:
        raise ModelError(
            f"unit {unit!r} is semantic {_DIMENSION_LABELS[semantic_dimension]!r}, "
            f"not {_DIMENSION_LABELS.get(dimension, str(dimension))!r}"
        )
    unit_obj = _pint_unit(normalized)
    expected = _pint_unit(_DIMENSION_REPRESENTATIVE_UNITS[dimension])
    try:
        (1.0 * unit_obj).to(expected)
    except DimensionalityError as exc:
        raise ModelError(
            f"unit {unit!r} is dimensionally incompatible with "
            f"{_DIMENSION_LABELS.get(dimension, str(dimension))!r}"
        ) from exc


def assert_identifier(value: str, label: str = "identifier") -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ModelError(f"invalid {label}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Reference:
    ALLOWED_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "project",
            "entity",
            "frame",
            "component",
            "connection",
            "requirement",
            "scenario",
            "body",
            "joint",
            "actuator",
            "sensor",
            "electrical_device",
            "electrical_port",
            "bus",
            "compute_device",
            "software_component",
        }
    )
    kind: Literal[
        "project",
        "entity",
        "frame",
        "component",
        "connection",
        "requirement",
        "scenario",
        "body",
        "joint",
        "actuator",
        "sensor",
        "electrical_device",
        "electrical_port",
        "bus",
        "compute_device",
        "software_component",
    ]
    id: str

    def __post_init__(self) -> None:
        if self.kind not in self.ALLOWED_KINDS:
            raise ModelError(f"invalid reference kind: {self.kind!r}")
        assert_identifier(self.id, f"{self.kind} reference")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(data, {"kind", "id"}, {"kind", "id"}, "Reference")
        return cls(kind=data["kind"], id=data["id"])


@dataclass(frozen=True, slots=True)
class Provenance:
    source: ProvenanceSource
    detail: str = ""
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {"source": self.source.value, "detail": self.detail, "reference": self.reference}
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(
            data,
            {"source"},
            {"source", "detail", "reference"},
            "Provenance",
        )
        return cls(
            source=ProvenanceSource(data["source"]),
            detail=data.get("detail", ""),
            reference=data.get("reference"),
        )


@dataclass(frozen=True, slots=True)
class Quantity:
    value: float
    unit: str
    dimension: Dimension | str

    def __post_init__(self) -> None:
        unit = _normalize_unit(self.unit)
        dimension = (
            dimension_from_name(self.dimension)
            if isinstance(self.dimension, str)
            else self.dimension
        )
        numeric_value = float(self.value)
        if not math.isfinite(numeric_value):
            raise ModelError("quantity value must be finite")
        _assert_unit_matches_dimension(unit, dimension)
        object.__setattr__(self, "value", numeric_value)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "dimension", dimension)

    @property
    def si_value(self) -> float:
        try:
            return float(_pint_quantity(self.value, self.unit).to_base_units().magnitude)
        except PintError as exc:
            raise ModelError(f"could not convert {self.unit!r} to SI base units") from exc

    def to(self, unit: str) -> Quantity:
        target_unit = _normalize_unit(unit)
        _assert_unit_matches_dimension(target_unit, self.dimension)
        try:
            converted = _pint_quantity(self.value, self.unit).to(_pint_unit(target_unit))
        except DimensionalityError as exc:
            raise ModelError(
                f"cannot convert {self.unit!r} to dimensionally incompatible {target_unit!r}"
            ) from exc
        except PintError as exc:
            raise ModelError(f"could not convert {self.unit!r} to {target_unit!r}") from exc
        return Quantity(
            value=float(converted.magnitude), unit=target_unit, dimension=self.dimension
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "dimension": _DIMENSION_LABELS[self.dimension],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(
            data,
            {"value", "unit", "dimension"},
            {"value", "unit", "dimension"},
            "Quantity",
        )
        return cls(value=data["value"], unit=data["unit"], dimension=data["dimension"])


@dataclass(frozen=True, slots=True)
class QuantityRange:
    minimum: Quantity | None = None
    target: Quantity | None = None
    maximum: Quantity | None = None

    def __post_init__(self) -> None:
        quantities = [q for q in (self.minimum, self.target, self.maximum) if q is not None]
        if quantities:
            dimension = quantities[0].dimension
            if any(q.dimension != dimension for q in quantities):
                raise ModelError("quantity range bounds must share one dimension")
            minimum = self.minimum.si_value if self.minimum else None
            target = self.target.si_value if self.target else None
            maximum = self.maximum.si_value if self.maximum else None
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ModelError("quantity range minimum exceeds maximum")
            if target is not None and minimum is not None and target < minimum:
                raise ModelError("quantity range target is below minimum")
            if target is not None and maximum is not None and target > maximum:
                raise ModelError("quantity range target is above maximum")

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "minimum": _to_data(self.minimum),
                "target": _to_data(self.target),
                "maximum": _to_data(self.maximum),
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(
            data,
            set(),
            {"minimum", "target", "maximum"},
            "QuantityRange",
        )
        return cls(
            minimum=_optional(data.get("minimum"), Quantity.from_dict),
            target=_optional(data.get("target"), Quantity.from_dict),
            maximum=_optional(data.get("maximum"), Quantity.from_dict),
        )


@dataclass(frozen=True, slots=True)
class Vector3:
    x: Quantity
    y: Quantity
    z: Quantity

    def __post_init__(self) -> None:
        if self.x.dimension != self.y.dimension or self.x.dimension != self.z.dimension:
            raise ModelError("vector components must share one dimension")

    def si_tuple(self) -> tuple[float, float, float]:
        return (self.x.si_value, self.y.si_value, self.z.si_value)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x.to_dict(), "y": self.y.to_dict(), "z": self.z.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(data, {"x", "y", "z"}, {"x", "y", "z"}, "Vector3")
        return cls(
            x=Quantity.from_dict(data["x"]),
            y=Quantity.from_dict(data["y"]),
            z=Quantity.from_dict(data["z"]),
        )


@dataclass(frozen=True, slots=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for value in (self.w, self.x, self.y, self.z):
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ModelError("quaternion components must be finite numbers")
        object.__setattr__(self, "w", float(self.w))
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "z", float(self.z))

    @property
    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def is_valid_rotation(self) -> bool:
        return self.norm > 0.0

    def is_normalized(self, *, tolerance: float = 1e-6) -> bool:
        return abs(self.norm - 1.0) <= tolerance

    def to_dict(self) -> dict[str, float]:
        return {"order": "wxyz", "w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(
            data,
            {"w", "x", "y", "z"},
            {"order", "w", "x", "y", "z"},
            "Quaternion",
        )
        if data.get("order", "wxyz") != "wxyz":
            raise ModelError("unsupported quaternion component order")
        return cls(w=data["w"], x=data["x"], y=data["y"], z=data["z"])


@dataclass(frozen=True, slots=True)
class Pose:
    reference_frame: Reference
    position: Vector3
    orientation: Quaternion = field(default_factory=lambda: Quaternion(1.0, 0.0, 0.0, 0.0))

    def __post_init__(self) -> None:
        if self.reference_frame.kind != "frame":
            raise ModelError("pose reference_frame must reference a frame")
        if self.position.x.dimension != LENGTH:
            raise ModelError("pose position must use length units")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_frame": self.reference_frame.to_dict(),
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        data = _strict_object(
            data,
            {"reference_frame", "position", "orientation"},
            {"reference_frame", "position", "orientation"},
            "Pose",
        )
        return cls(
            reference_frame=Reference.from_dict(data["reference_frame"]),
            position=Vector3.from_dict(data["position"]),
            orientation=Quaternion.from_dict(data["orientation"]),
        )


@dataclass(frozen=True, slots=True)
class Appearance:
    color_rgba: tuple[float, float, float, float] | None = None
    texture_uri: str | None = None

    def __post_init__(self) -> None:
        if self.color_rgba is not None and (
            len(self.color_rgba) != 4 or any(c < 0.0 or c > 1.0 for c in self.color_rgba)
        ):
            raise ModelError("appearance color_rgba must contain four values in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict(
            {
                "color_rgba": list(self.color_rgba) if self.color_rgba else None,
                "texture_uri": self.texture_uri,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        color = data.get("color_rgba")
        return cls(
            color_rgba=tuple(color) if color is not None else None,
            texture_uri=data.get("texture_uri"),
        )


@dataclass(frozen=True, slots=True)
class Material:
    id: str
    name: str
    density: Quantity | None = None
    appearance: Appearance | None = None
    provenance: Provenance | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "material id")
        if self.density is not None and self.density.dimension != DENSITY:
            raise ModelError("material density must use density units")


@dataclass(frozen=True, slots=True)
class Geometry:
    id: str
    kind: GeometryKind
    dimensions: tuple[Quantity, ...] = ()
    radius: Quantity | None = None
    length: Quantity | None = None
    asset_uri: str | None = None
    mesh_format: MeshFormat | None = None
    parametric_kind: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    material_ref: Reference | None = None
    provenance: Provenance | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "geometry id")
        if self.kind == GeometryKind.BOX and len(self.dimensions) != 3:
            raise ModelError("box geometry requires three dimensions")
        if self.kind == GeometryKind.BOX and (
            self.radius is not None
            or self.length is not None
            or self.asset_uri is not None
            or self.mesh_format is not None
            or self.parametric_kind is not None
        ):
            raise ModelError("box geometry must not carry fields from other geometry variants")
        if self.kind in {GeometryKind.CYLINDER, GeometryKind.CAPSULE} and (
            self.radius is None or self.length is None
        ):
            raise ModelError(f"{self.kind.value} geometry requires radius and length")
        if self.kind in {GeometryKind.CYLINDER, GeometryKind.CAPSULE} and (
            self.dimensions
            or self.asset_uri is not None
            or self.mesh_format is not None
            or self.parametric_kind is not None
        ):
            raise ModelError(
                f"{self.kind.value} geometry must not carry fields from other variants"
            )
        if self.kind == GeometryKind.SPHERE and self.radius is None:
            raise ModelError("sphere geometry requires radius")
        if self.kind == GeometryKind.SPHERE and (
            self.dimensions
            or self.length is not None
            or self.asset_uri is not None
            or self.mesh_format is not None
            or self.parametric_kind is not None
        ):
            raise ModelError("sphere geometry must not carry fields from other geometry variants")
        if self.kind == GeometryKind.MESH and (not self.asset_uri or self.mesh_format is None):
            raise ModelError("mesh geometry requires asset_uri and mesh_format")
        if self.kind == GeometryKind.MESH and (
            self.dimensions
            or self.radius is not None
            or self.length is not None
            or self.parametric_kind is not None
        ):
            raise ModelError("mesh geometry must not carry fields from other geometry variants")
        if self.kind == GeometryKind.PARAMETRIC and not self.parametric_kind:
            raise ModelError("parametric geometry requires parametric_kind")
        if self.kind == GeometryKind.PARAMETRIC and (
            self.dimensions
            or self.radius is not None
            or self.length is not None
            or self.asset_uri is not None
            or self.mesh_format is not None
        ):
            raise ModelError(
                "parametric geometry must not carry fields from other geometry variants"
            )
        for quantity in (
            *self.dimensions,
            *(q for q in (self.radius, self.length) if q is not None),
        ):
            if quantity.dimension != LENGTH:
                raise ModelError("geometry dimensions must use length units")
        if self.material_ref is not None and self.material_ref.kind != "component":
            raise ModelError("geometry material_ref must use component references")

    @classmethod
    def box(cls, id: str, x: Quantity, y: Quantity, z: Quantity, **kwargs: Any) -> Geometry:
        return cls(id=id, kind=GeometryKind.BOX, dimensions=(x, y, z), **kwargs)

    @classmethod
    def mesh(cls, id: str, asset_uri: str, mesh_format: MeshFormat, **kwargs: Any) -> Geometry:
        return cls(
            id=id, kind=GeometryKind.MESH, asset_uri=asset_uri, mesh_format=mesh_format, **kwargs
        )


@dataclass(frozen=True, slots=True)
class GeometryBinding:
    geometry_ref: Reference
    role: GeometryRole
    pose: Pose | None = None

    def __post_init__(self) -> None:
        if self.geometry_ref.kind != "component":
            raise ModelError("geometry binding must reference a component")


@dataclass(frozen=True, slots=True)
class MassProperties:
    mass: Quantity
    center_of_mass: Vector3
    inertia_tensor: tuple[
        tuple[Quantity, Quantity, Quantity],
        tuple[Quantity, Quantity, Quantity],
        tuple[Quantity, Quantity, Quantity],
    ]
    provenance: MassProvenance = MassProvenance.EXPLICIT

    def __post_init__(self) -> None:
        if self.mass.dimension != MASS:
            raise ModelError("mass property must use mass units")
        if self.center_of_mass.x.dimension != LENGTH:
            raise ModelError("center of mass must use length units")
        if len(self.inertia_tensor) != 3 or any(len(row) != 3 for row in self.inertia_tensor):
            raise ModelError("inertia tensor must be 3x3")
        for row in self.inertia_tensor:
            for quantity in row:
                if quantity.dimension != INERTIA:
                    raise ModelError("inertia tensor values must use inertia units")


@dataclass(frozen=True, slots=True)
class Requirement:
    id: str
    kind: RequirementKind
    statement: str
    priority: RequirementPriority = RequirementPriority.MEDIUM
    verification_method: VerificationMethod = VerificationMethod.UNSPECIFIED
    status: RequirementStatus = RequirementStatus.PROPOSED
    value: Quantity | QuantityRange | str | None = None
    provenance: Provenance | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "requirement id")
        if not self.statement.strip():
            raise ModelError("requirement statement must not be empty")


@dataclass(frozen=True, slots=True)
class Assumption:
    id: str
    statement: str
    provenance: Provenance

    def __post_init__(self) -> None:
        assert_identifier(self.id, "assumption id")
        if not self.statement.strip():
            raise ModelError("assumption statement must not be empty")


@dataclass(frozen=True, slots=True)
class EngineeringDecision:
    id: str
    summary: str
    rationale: str
    provenance: Provenance

    def __post_init__(self) -> None:
        assert_identifier(self.id, "decision id")


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ModelError("project name must not be empty")


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    name: str
    kind: EntityKind = EntityKind.CUSTOM
    parent_entity: Reference | None = None
    frame_ref: Reference | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "entity id")
        if self.parent_entity is not None and self.parent_entity.kind != "entity":
            raise ModelError("parent_entity must reference an entity")
        if self.frame_ref is not None and self.frame_ref.kind != "frame":
            raise ModelError("frame_ref must reference a frame")


@dataclass(frozen=True, slots=True)
class Frame:
    id: str
    name: str
    attached_to: Reference | None = None

    def __post_init__(self) -> None:
        assert_identifier(self.id, "frame id")


@dataclass(frozen=True, slots=True)
class Transform:
    id: str
    parent_frame: Reference
    child_frame: Reference
    pose: Pose
    provenance: Provenance | None = None

    def __post_init__(self) -> None:
        assert_identifier(self.id, "transform id")
        if self.parent_frame.kind != "frame" or self.child_frame.kind != "frame":
            raise ModelError("transforms must reference frames")
        if self.pose.reference_frame.id != self.parent_frame.id:
            raise ModelError("transform pose reference_frame must match parent_frame")


@dataclass(frozen=True, slots=True)
class Body:
    id: str
    entity_ref: Reference
    frame_ref: Reference
    visual_geometry: tuple[GeometryBinding, ...] = ()
    collision_geometry: tuple[GeometryBinding, ...] = ()
    mass_properties: MassProperties | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "body id")
        if self.entity_ref.kind != "entity" or self.frame_ref.kind != "frame":
            raise ModelError("body must reference an entity and frame")


@dataclass(frozen=True, slots=True)
class Joint:
    id: str
    name: str
    kind: JointKind
    parent_body: Reference
    child_body: Reference
    axis: tuple[float, float, float] | None = None
    limits: dict[str, Quantity] = field(default_factory=dict)
    provenance: Provenance | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "joint id")
        if self.parent_body.kind != "body" or self.child_body.kind != "body":
            raise ModelError("joint bodies must reference bodies")
        if self.axis is not None:
            if len(self.axis) != 3 or any(not math.isfinite(v) for v in self.axis):
                raise ModelError("joint axis must contain three finite numbers")
            norm = math.sqrt(sum(float(v) * float(v) for v in self.axis))
            if norm > 0.0:
                object.__setattr__(self, "axis", tuple(float(v) / norm for v in self.axis))
            else:
                object.__setattr__(self, "axis", tuple(float(v) for v in self.axis))


@dataclass(frozen=True, slots=True)
class Connection:
    id: str
    kind: str
    source: Reference
    target: Reference
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "connection id")
        if not self.kind.strip():
            raise ModelError("connection kind must not be empty")


@dataclass(frozen=True, slots=True)
class Actuator:
    id: str
    name: str
    kind: ActuatorKind
    controlled_joint: Reference | None = None
    mount_frame: Reference | None = None
    interface_refs: tuple[Reference, ...] = ()
    limits: dict[str, Quantity] = field(default_factory=dict)
    vendor_reference: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "actuator id")
        if self.controlled_joint is not None and self.controlled_joint.kind != "joint":
            raise ModelError("actuator controlled_joint must reference a joint")
        if self.mount_frame is not None and self.mount_frame.kind != "frame":
            raise ModelError("actuator mount_frame must reference a frame")


@dataclass(frozen=True, slots=True)
class NoiseModel:
    kind: str
    parameters: dict[str, Quantity | float | str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FieldOfView:
    horizontal: Quantity | None = None
    vertical: Quantity | None = None

    def __post_init__(self) -> None:
        for quantity in (self.horizontal, self.vertical):
            if quantity is not None and quantity.dimension != ANGLE:
                raise ModelError("field of view must use angle units")


@dataclass(frozen=True, slots=True)
class Sensor:
    id: str
    name: str
    modality: SensorModality
    mounting_frame: Reference
    update_rate_hz: float | None = None
    latency: Quantity | None = None
    noise_model: NoiseModel | None = None
    field_of_view: FieldOfView | None = None
    range: QuantityRange | None = None
    resolution: tuple[int, ...] | None = None
    interface_refs: tuple[Reference, ...] = ()
    vendor_reference: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "sensor id")
        if self.mounting_frame.kind != "frame":
            raise ModelError("sensor mounting_frame must reference a frame")
        if self.update_rate_hz is not None and (
            not math.isfinite(self.update_rate_hz) or self.update_rate_hz < 0.0
        ):
            raise ModelError("sensor update_rate_hz must be finite and non-negative")
        if self.latency is not None and self.latency.dimension != TIME:
            raise ModelError("sensor latency must use time units")
        if self.latency is not None and self.latency.si_value < 0.0:
            raise ModelError("sensor latency must be non-negative")


@dataclass(frozen=True, slots=True)
class ElectricalPort:
    id: str
    device_ref: Reference
    name: str
    voltage_limit: Quantity | None = None
    current_limit: Quantity | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "electrical port id")
        if self.device_ref.kind != "electrical_device":
            raise ModelError("electrical port device_ref must reference an electrical_device")
        if self.voltage_limit is not None and self.voltage_limit.dimension != VOLTAGE:
            raise ModelError("electrical port voltage_limit must use voltage units")
        if self.current_limit is not None and self.current_limit.dimension != CURRENT:
            raise ModelError("electrical port current_limit must use current units")


@dataclass(frozen=True, slots=True)
class ElectricalDevice:
    id: str
    name: str
    device_type: str
    entity_ref: Reference | None = None
    power_source: bool = False
    voltage_nominal: Quantity | None = None
    power_limit: Quantity | None = None
    firmware_reference: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "electrical device id")
        if self.entity_ref is not None and self.entity_ref.kind != "entity":
            raise ModelError("electrical device entity_ref must reference an entity")
        if self.voltage_nominal is not None and self.voltage_nominal.dimension != VOLTAGE:
            raise ModelError("electrical device voltage_nominal must use voltage units")
        if self.power_limit is not None and self.power_limit.dimension != POWER:
            raise ModelError("electrical device power_limit must use power units")


@dataclass(frozen=True, slots=True)
class PowerConnection:
    id: str
    source_port: Reference
    target_port: Reference
    voltage: Quantity | None = None
    current_limit: Quantity | None = None

    def __post_init__(self) -> None:
        assert_identifier(self.id, "power connection id")
        if self.source_port.kind != "electrical_port" or self.target_port.kind != "electrical_port":
            raise ModelError("power connections must reference electrical ports")
        if self.voltage is not None and self.voltage.dimension != VOLTAGE:
            raise ModelError("power connection voltage must use voltage units")
        if self.current_limit is not None and self.current_limit.dimension != CURRENT:
            raise ModelError("power connection current_limit must use current units")


@dataclass(frozen=True, slots=True)
class CommunicationInterface:
    id: str
    owner_ref: Reference
    bus_ref: Reference | None = None
    name: str = ""
    update_rate_hz: float | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "communication interface id")
        if self.bus_ref is not None and self.bus_ref.kind != "bus":
            raise ModelError("communication interface bus_ref must reference a bus")
        if self.update_rate_hz is not None and (
            not math.isfinite(self.update_rate_hz) or self.update_rate_hz < 0.0
        ):
            raise ModelError("communication update_rate_hz must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CommunicationBus:
    id: str
    name: str
    kind: BusKind
    bitrate: Quantity | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "communication bus id")
        if self.bitrate is not None and self.bitrate.dimension != FREQUENCY:
            raise ModelError(
                "communication bus bitrate must use frequency units until bitrate units are modeled"
            )


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    id: str
    name: str
    device_ref: Reference | None = None
    processor: str = ""
    memory: Quantity | None = None
    interface_refs: tuple[Reference, ...] = ()
    software_refs: tuple[Reference, ...] = ()
    firmware_reference: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "compute device id")
        if self.device_ref is not None and self.device_ref.kind != "electrical_device":
            raise ModelError("compute device_ref must reference an electrical_device")
        for ref in self.software_refs:
            if ref.kind != "software_component":
                raise ModelError("compute software_refs must reference software components")


@dataclass(frozen=True, slots=True)
class SoftwareComponent:
    id: str
    name: str
    role: str
    compute_ref: Reference | None = None
    input_refs: tuple[Reference, ...] = ()
    output_refs: tuple[Reference, ...] = ()
    timing: dict[str, Quantity] = field(default_factory=dict)
    deployment_reference: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "software component id")
        if self.compute_ref is not None and self.compute_ref.kind != "compute_device":
            raise ModelError("software compute_ref must reference a compute_device")
        for name, quantity in self.timing.items():
            if quantity.dimension != TIME and quantity.dimension != FREQUENCY:
                raise ModelError(f"software timing {name} must use time or frequency units")
            if quantity.si_value < 0.0:
                raise ModelError(f"software timing {name} must be non-negative")


@dataclass(frozen=True, slots=True)
class SimulationScenario:
    id: str
    name: str
    description: str = ""
    environment: str = ""
    duration: Quantity | None = None
    initial_state_refs: tuple[Reference, ...] = ()
    validation_refs: tuple[Reference, ...] = ()
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_identifier(self.id, "simulation scenario id")
        if self.duration is not None and self.duration.dimension != TIME:
            raise ModelError("simulation scenario duration must use time units")
        if self.duration is not None and self.duration.si_value < 0.0:
            raise ModelError("simulation scenario duration must be non-negative")


@dataclass(frozen=True, slots=True)
class ValidationState:
    latest_summary: str = "not validated"
    finding_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EngineeringProject:
    id: str
    metadata: ProjectMetadata
    schema_version: str = SCHEMA_VERSION
    revision: int = 0
    requirements: tuple[Requirement, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    entities: tuple[Entity, ...] = ()
    frames: tuple[Frame, ...] = ()
    transforms: tuple[Transform, ...] = ()
    materials: tuple[Material, ...] = ()
    geometries: tuple[Geometry, ...] = ()
    bodies: tuple[Body, ...] = ()
    joints: tuple[Joint, ...] = ()
    connections: tuple[Connection, ...] = ()
    actuators: tuple[Actuator, ...] = ()
    sensors: tuple[Sensor, ...] = ()
    electrical_devices: tuple[ElectricalDevice, ...] = ()
    electrical_ports: tuple[ElectricalPort, ...] = ()
    power_connections: tuple[PowerConnection, ...] = ()
    communication_buses: tuple[CommunicationBus, ...] = ()
    communication_interfaces: tuple[CommunicationInterface, ...] = ()
    compute_devices: tuple[ComputeDevice, ...] = ()
    software_components: tuple[SoftwareComponent, ...] = ()
    simulation_scenarios: tuple[SimulationScenario, ...] = ()
    engineering_decisions: tuple[EngineeringDecision, ...] = ()
    validation_state: ValidationState = field(default_factory=ValidationState)
    extensions: dict[str, Any] = field(default_factory=dict)

    COLLECTIONS: ClassVar[tuple[str, ...]] = (
        "requirements",
        "assumptions",
        "entities",
        "frames",
        "transforms",
        "materials",
        "geometries",
        "bodies",
        "joints",
        "connections",
        "actuators",
        "sensors",
        "electrical_devices",
        "electrical_ports",
        "power_connections",
        "communication_buses",
        "communication_interfaces",
        "compute_devices",
        "software_components",
        "simulation_scenarios",
        "engineering_decisions",
    )

    def __post_init__(self) -> None:
        assert_identifier(self.id, "project id")
        if self.revision < 0:
            raise ModelError("project revision must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return _to_data(self)

    @classmethod
    def empty(cls, *, name: str, project_id: str = "project:hephaestus") -> EngineeringProject:
        return cls(id=project_id, metadata=ProjectMetadata(name=name))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineeringProject:
        allowed = {field.name for field in fields(cls) if field.init}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ModelError(f"unknown top-level project fields: {', '.join(unknown)}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(str(data.get("schema_version")))
        return _from_dataclass(cls, data)

    def collection_counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in self.COLLECTIONS}


class UnsupportedSchemaVersionError(ModelError):
    """Raised when a project document uses an unsupported schema version."""

    def __init__(self, schema_version: str) -> None:
        super().__init__(f"unsupported project schema version: {schema_version}")
        self.schema_version = schema_version


def q(value: float, unit: str, dimension: Dimension | str) -> Quantity:
    return Quantity(value=value, unit=unit, dimension=dimension)


def zero_vector(dimension: Dimension | str = LENGTH, unit: str = "m") -> Vector3:
    return Vector3(q(0.0, unit, dimension), q(0.0, unit, dimension), q(0.0, unit, dimension))


def identity_pose(reference_frame_id: str) -> Pose:
    return Pose(reference_frame=Reference("frame", reference_frame_id), position=zero_vector())


def dimension_from_name(value: str) -> Dimension:
    try:
        return _DIMENSION_NAMES[value]
    except KeyError as exc:
        raise ModelError(f"unknown dimension: {value}") from exc


def _clean_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if value is not None and value != {} and value != ()
    }


def _optional(value: Any, loader: Any) -> Any:
    return None if value is None else loader(value)


def _strict_object(
    data: Any, required: set[str], allowed: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ModelError(f"invalid {label}: expected object")
    missing = sorted(required - set(data))
    if missing:
        raise ModelError(f"missing required field for {label}: {', '.join(missing)}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ModelError(f"unknown fields for {label}: {', '.join(unknown)}")
    return data


def _to_data(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_to_data(item) for item in value]
    if isinstance(value, list):
        return [_to_data(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_data(value[key]) for key in sorted(value)}
    if hasattr(value, "to_dict") and not isinstance(value, EngineeringProject):
        return value.to_dict()
    if is_dataclass(value):
        result: dict[str, Any] = {}
        for item in fields(value):
            if not item.init or item.name == "COLLECTIONS":
                continue
            result[item.name] = _to_data(getattr(value, item.name))
        if isinstance(value, EngineeringProject):
            for name in EngineeringProject.COLLECTIONS:
                result[name] = sorted(result[name], key=lambda item: item["id"])
        return _clean_dict(result)
    return value


def _from_dataclass(cls: type[Any], data: Mapping[str, Any]) -> Any:
    if not isinstance(data, Mapping):
        raise ModelError(f"invalid {cls.__name__}: expected object")
    init_fields = {item.name: item for item in fields(cls) if item.init}
    unknown = sorted(set(data) - set(init_fields))
    if unknown:
        raise ModelError(f"unknown fields for {cls.__name__}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    try:
        for name, item in init_fields.items():
            if name in data:
                kwargs[name] = _from_value(type_hints.get(name, item.type), data[name])
            elif item.default is MISSING and item.default_factory is MISSING:
                raise ModelError(f"missing required field for {cls.__name__}: {name}")
        return cls(**kwargs)
    except (TypeError, ValueError, KeyError) as exc:
        raise ModelError(f"invalid {cls.__name__}: {exc}") from exc


def _from_value(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if value is None:
        return None
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ModelError("expected array")
        if args and len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_value(args[0], item) for item in value)
        if args:
            if len(args) != len(value):
                raise ModelError(f"expected array with {len(args)} items")
            return tuple(_from_value(arg, item) for arg, item in zip(args, value, strict=True))
        return tuple(value)
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ModelError("expected object")
        value_type = args[1] if len(args) == 2 else Any
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelError("object keys must be strings")
            converted[key] = _from_value(value_type, item)
        return converted
    if origin in {UnionType} or str(origin) == "typing.Union":
        errors: list[str] = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _from_value(arg, value)
            except Exception as exc:
                errors.append(str(exc))
        raise ModelError("value did not match any supported variant: " + "; ".join(errors))
    if annotation is Any:
        _assert_json_compatible(value)
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ModelError("expected string")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ModelError("expected integer")
        return value
    if annotation is float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ModelError("expected number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ModelError("number must be finite")
        return numeric
    if annotation is bool:
        if not isinstance(value, bool):
            raise ModelError("expected boolean")
        return value
    if annotation is Quantity:
        return Quantity.from_dict(value)
    if annotation is QuantityRange:
        return QuantityRange.from_dict(value)
    if annotation is Reference:
        return Reference.from_dict(value)
    if annotation is Vector3:
        return Vector3.from_dict(value)
    if annotation is Quaternion:
        return Quaternion.from_dict(value)
    if annotation is Pose:
        return Pose.from_dict(value)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _from_dataclass(annotation, value)
    return value


def _assert_json_compatible(value: Any) -> None:
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelError("JSON numeric extension values must be finite")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _assert_json_compatible(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelError("JSON object keys must be strings")
            _assert_json_compatible(item)
        return
    raise ModelError(f"value is not JSON serializable: {type(value).__name__}")


def coerce_value(value: Any) -> Any:
    if isinstance(value, dict) and {"value", "unit", "dimension"}.issubset(value):
        return Quantity.from_dict(value)
    if isinstance(value, dict) and any(key in value for key in ("minimum", "target", "maximum")):
        return QuantityRange.from_dict(value)
    return value
