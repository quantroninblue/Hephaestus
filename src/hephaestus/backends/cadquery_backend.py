"""CadQuery backend adapter for the Milestone 3.0 mobile robot slice."""

# ruff: noqa: E501

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hephaestus.artifacts import ArtifactRecord, ArtifactStager, record_for_file
from hephaestus.concept_robot import ConceptError, PrimitiveType, concept_backend_spec
from hephaestus.mobile_robot import mobile_robot_spec
from hephaestus.model import EngineeringProject

GENERATOR_VERSION = "hephaestus.cadquery_backend.v1"
M_TO_MM = 1000.0


class CadGenerationError(RuntimeError):
    """Raised when CAD generation cannot complete."""


@dataclass(frozen=True, slots=True)
class SolidCheck:
    name: str
    volume_mm3: float
    bounding_box_mm: tuple[float, float, float]
    expected_bounding_box_mm: tuple[float, float, float]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "volume_mm3": self.volume_mm3,
            "bounding_box_mm": list(self.bounding_box_mm),
            "expected_bounding_box_mm": list(self.expected_bounding_box_mm),
            "status": self.status,
        }


class CadAdapter(Protocol):
    version: str | None

    def box(self, x_mm: float, y_mm: float, z_mm: float) -> Any: ...

    def cylinder_y(self, radius_mm: float, width_mm: float) -> Any: ...

    def sphere(self, radius_mm: float) -> Any: ...

    def volume(self, solid: Any) -> float: ...

    def bounding_box(self, solid: Any) -> tuple[float, float, float]: ...

    def export_step(self, solid: Any, path: Path) -> None: ...

    def export_stl(self, solid: Any, path: Path) -> None: ...

    def export_assembly_step(self, solids: Mapping[str, Any], path: Path) -> None: ...


class RealCadQueryAdapter:
    def __init__(self) -> None:
        try:
            import cadquery as cq
            from cadquery import exporters
        except Exception as exc:  # pragma: no cover - host integration path
            raise CadGenerationError("CadQuery is not importable") from exc
        self._cq = cq
        self._exporters = exporters
        self.version = importlib.metadata.version("cadquery")

    def box(self, x_mm: float, y_mm: float, z_mm: float) -> Any:
        return self._cq.Workplane("XY").box(x_mm, y_mm, z_mm).val()

    def cylinder_y(self, radius_mm: float, width_mm: float) -> Any:
        return self._cq.Workplane("XZ").circle(radius_mm).extrude(width_mm).val()

    def sphere(self, radius_mm: float) -> Any:
        return self._cq.Workplane("XY").sphere(radius_mm).val()

    def volume(self, solid: Any) -> float:
        return float(solid.Volume())

    def bounding_box(self, solid: Any) -> tuple[float, float, float]:
        bbox = solid.BoundingBox()
        return (float(bbox.xlen), float(bbox.ylen), float(bbox.zlen))

    def export_step(self, solid: Any, path: Path) -> None:
        self._exporters.export(solid, str(path), exportType="STEP")
        _normalize_step_metadata(path)

    def export_stl(self, solid: Any, path: Path) -> None:
        self._exporters.export(
            solid,
            str(path),
            exportType="STL",
            tolerance=0.001,
            angularTolerance=0.05,
        )

    def export_assembly_step(self, solids: Mapping[str, Any], path: Path) -> None:
        assembly = self._cq.Assembly()
        for name in sorted(solids):
            assembly.add(solids[name], name=name)
        assembly.export(str(path))
        _normalize_step_metadata(path)


class FakeCadAdapter:
    """Deterministic non-CadQuery adapter for offline tests."""

    version = "fake-cadquery"

    def box(self, x_mm: float, y_mm: float, z_mm: float) -> dict[str, Any]:
        return {"kind": "box", "bbox": (x_mm, y_mm, z_mm), "volume": x_mm * y_mm * z_mm}

    def cylinder_y(self, radius_mm: float, width_mm: float) -> dict[str, Any]:
        return {
            "kind": "cylinder_y",
            "bbox": (2 * radius_mm, width_mm, 2 * radius_mm),
            "volume": 3.141592653589793 * radius_mm * radius_mm * width_mm,
        }

    def sphere(self, radius_mm: float) -> dict[str, Any]:
        return {
            "kind": "sphere",
            "bbox": (2 * radius_mm, 2 * radius_mm, 2 * radius_mm),
            "volume": 4.0 * 3.141592653589793 * radius_mm**3 / 3.0,
        }

    def volume(self, solid: dict[str, Any]) -> float:
        return float(solid["volume"])

    def bounding_box(self, solid: dict[str, Any]) -> tuple[float, float, float]:
        return tuple(float(item) for item in solid["bbox"])

    def export_step(self, solid: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps({"format": "STEP", **solid}, sort_keys=True) + "\n")

    def export_stl(self, solid: dict[str, Any], path: Path) -> None:
        path.write_text(
            "solid hephaestus\n"
            "facet normal 0 0 1\n"
            " outer loop\n"
            "  vertex 0 0 0\n"
            "  vertex 1 0 0\n"
            "  vertex 0 1 0\n"
            " endloop\n"
            "endfacet\n"
            "endsolid hephaestus\n"
        )

    def export_assembly_step(self, solids: Mapping[str, Any], path: Path) -> None:
        payload = {"format": "STEP_ASSEMBLY", "parts": sorted(solids)}
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def generate_cad_artifacts(
    project: EngineeringProject,
    stager: ArtifactStager,
    *,
    adapter: CadAdapter | None = None,
) -> tuple[ArtifactRecord, ...]:
    adapter = adapter or RealCadQueryAdapter()
    try:
        concept = concept_backend_spec(project)
    except ConceptError:
        pass
    else:
        return _generate_concept_cad_artifacts(concept, stager, adapter)
    spec = mobile_robot_spec(project)
    values = spec["values"]
    ids = spec["ids"]
    dims = _dimensions_mm(values)
    solids = {
        "chassis": (
            adapter.box(dims["chassis_length"], dims["chassis_width"], dims["chassis_height"]),
            (dims["chassis_length"], dims["chassis_width"], dims["chassis_height"]),
            (ids["chassis_body"], ids["chassis_geometry"]),
        ),
        "wheel": (
            adapter.cylinder_y(dims["wheel_radius"], dims["wheel_width"]),
            (2 * dims["wheel_radius"], dims["wheel_width"], 2 * dims["wheel_radius"]),
            (ids["left_wheel_body"], ids["right_wheel_body"], ids["wheel_geometry"]),
        ),
        "caster": (
            adapter.sphere(dims["caster_radius"]),
            (2 * dims["caster_radius"], 2 * dims["caster_radius"], 2 * dims["caster_radius"]),
            (ids["caster_body"], ids["caster_geometry"]),
        ),
    }
    checks: list[SolidCheck] = []
    records: list[ArtifactRecord] = []
    warnings: list[str] = []
    params_path = stager.write_text("cad/source/mobile_robot_parameters.json", _params_json(spec))
    source_path = stager.write_text("cad/source/mobile_robot_cad.py", _cad_source(spec))
    for rel, role in (
        ("cad/source/mobile_robot_parameters.json", "cad_parameters"),
        ("cad/source/mobile_robot_cad.py", "cad_source"),
    ):
        records.append(
            record_for_file(stager.staging, rel, role=role, owning_ids=(ids["robot_entity"],))
        )
    del params_path, source_path

    for name, (solid, expected, owning_ids) in solids.items():
        check = _check_solid(adapter, name, solid, expected)
        checks.append(check)
        if check.status != "passed":
            raise CadGenerationError(f"CAD solid validation failed for {name}")
        step_rel = f"cad/step/{name}.step"
        visual_rel = f"cad/stl/visual/{name}.stl"
        collision_rel = f"cad/stl/collision/{name}.stl"
        adapter.export_step(solid, stager.path(step_rel))
        adapter.export_stl(solid, stager.path(visual_rel))
        adapter.export_stl(solid, stager.path(collision_rel))
        records.append(
            record_for_file(
                stager.staging, step_rel, role="cad_step", owning_ids=owning_ids, units="mm"
            )
        )
        records.append(
            record_for_file(
                stager.staging,
                visual_rel,
                role="visual_mesh_stl",
                owning_ids=owning_ids,
                units="mm mesh vertices; URDF/SDF scale 0.001",
                dependencies=(step_rel,),
            )
        )
        records.append(
            record_for_file(
                stager.staging,
                collision_rel,
                role="collision_mesh_stl",
                owning_ids=owning_ids,
                units="mm mesh vertices; URDF/SDF scale 0.001",
                dependencies=(step_rel,),
            )
        )

    try:
        assembly_rel = "cad/step/assembly.step"
        adapter.export_assembly_step(
            {name: solid for name, (solid, _expected, _owners) in solids.items()},
            stager.path(assembly_rel),
        )
        records.append(
            record_for_file(
                stager.staging,
                assembly_rel,
                role="cad_assembly_step",
                owning_ids=(ids["robot_entity"],),
                units="mm",
                dependencies=tuple(f"cad/step/{name}.step" for name in sorted(solids)),
            )
        )
    except Exception as exc:
        warnings.append(f"assembly STEP export skipped: {exc}")

    report = {
        "schema": "hephaestus.cad_report.v1",
        "status": "passed",
        "backend": "cadquery",
        "backend_version": adapter.version,
        "unit_conversion": {"canonical": "m", "cadquery": "mm", "factor": M_TO_MM},
        "solids": [check.to_dict() for check in checks],
        "mass_property_source": "canonical analytical mass properties",
        "mesh_scaling": "STL exported in millimetres; URDF and SDF use scale 0.001 0.001 0.001",
        "assembly_step": "skipped" if warnings else "generated",
        "warnings": warnings,
    }
    report_rel = "reports/cad_report.json"
    stager.write_text(
        report_rel, json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    records.append(
        record_for_file(
            stager.staging, report_rel, role="cad_report", owning_ids=(ids["robot_entity"],)
        )
    )
    return tuple(records)


def _generate_concept_cad_artifacts(
    concept: dict[str, Any], stager: ArtifactStager, adapter: CadAdapter
) -> tuple[ArtifactRecord, ...]:
    spec = concept["spec"]
    ids = concept["ids"]
    bodies = list(spec["bodies"])
    records: list[ArtifactRecord] = []
    checks: list[SolidCheck] = []
    warnings: list[str] = []
    solids: dict[str, Any] = {}
    robot_owner = (str(ids.get("robot_entity", "entity:concept_robot")),)

    params_rel = "cad/source/concept_robot_parameters.json"
    source_rel = "cad/source/concept_robot_cad.py"
    stager.write_text(params_rel, _params_json(concept))
    stager.write_text(source_rel, _concept_cad_source(spec))
    records.append(
        record_for_file(stager.staging, params_rel, role="cad_parameters", owning_ids=robot_owner)
    )
    records.append(
        record_for_file(stager.staging, source_rel, role="cad_source", owning_ids=robot_owner)
    )

    for body in sorted(bodies, key=lambda item: item["semantic_key"]):
        key = str(body["semantic_key"])
        primitive = PrimitiveType(str(body["primitive_type"]))
        solid, expected = _concept_solid(adapter, body, primitive)
        check = _check_solid(adapter, key, solid, expected)
        checks.append(check)
        if check.status != "passed":
            raise CadGenerationError(f"CAD solid validation failed for concept body {key}")
        solids[key] = solid
        body_ids = ids.get("bodies", {}).get(key, {}) if isinstance(ids.get("bodies"), dict) else {}
        owning_ids = tuple(
            str(item) for item in (body_ids.get("body"), body_ids.get("geometry")) if item
        )
        if not owning_ids:
            owning_ids = robot_owner
        step_rel = f"cad/step/{key}.step"
        visual_rel = f"cad/stl/visual/{key}.stl"
        collision_rel = f"cad/stl/collision/{key}.stl"
        adapter.export_step(solid, stager.path(step_rel))
        adapter.export_stl(solid, stager.path(visual_rel))
        adapter.export_stl(solid, stager.path(collision_rel))
        records.append(
            record_for_file(
                stager.staging, step_rel, role="cad_step", owning_ids=owning_ids, units="mm"
            )
        )
        records.append(
            record_for_file(
                stager.staging,
                visual_rel,
                role="visual_mesh_stl",
                owning_ids=owning_ids,
                units="mm mesh vertices; canonical SI dimensions",
            )
        )
        records.append(
            record_for_file(
                stager.staging,
                collision_rel,
                role="collision_mesh_stl",
                owning_ids=owning_ids,
                units="mm mesh vertices; canonical SI dimensions",
            )
        )

    try:
        assembly_rel = "cad/step/assembly.step"
        adapter.export_assembly_step(solids, stager.path(assembly_rel))
        records.append(
            record_for_file(
                stager.staging,
                assembly_rel,
                role="cad_assembly_step",
                owning_ids=robot_owner,
                units="mm",
                dependencies=tuple(f"cad/step/{name}.step" for name in sorted(solids)),
            )
        )
    except Exception as exc:
        warnings.append(f"assembly STEP export skipped: {exc}")

    report = {
        "schema": "hephaestus.cad_report.v1",
        "status": "passed",
        "backend": "cadquery",
        "backend_version": adapter.version,
        "unit_conversion": {"canonical": "m", "cadquery": "mm", "factor": M_TO_MM},
        "robot_name": spec["robot_name"],
        "concept_body_count": len(bodies),
        "solids": [check.to_dict() for check in checks],
        "mass_property_source": "canonical analytical primitive mass properties",
        "mesh_scaling": "STL exported in millimetres; primitive URDF/SDF generated from canonical SI dimensions",
        "assembly_step": "skipped" if warnings else "generated",
        "warnings": [*concept.get("warnings", []), *warnings],
    }
    report_rel = "reports/cad_report.json"
    stager.write_text(
        report_rel, json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    records.append(
        record_for_file(stager.staging, report_rel, role="cad_report", owning_ids=robot_owner)
    )
    return tuple(records)


def _concept_solid(
    adapter: CadAdapter, body: dict[str, Any], primitive: PrimitiveType
) -> tuple[Any, tuple[float, float, float]]:
    if primitive is PrimitiveType.BOX:
        expected = (
            float(body["size_x_m"]) * M_TO_MM,
            float(body["size_y_m"]) * M_TO_MM,
            float(body["size_z_m"]) * M_TO_MM,
        )
        return adapter.box(*expected), expected
    if primitive is PrimitiveType.CYLINDER:
        radius = float(body["radius_m"]) * M_TO_MM
        length = float(body["length_m"]) * M_TO_MM
        return adapter.cylinder_y(radius, length), (2 * radius, length, 2 * radius)
    radius = float(body["radius_m"]) * M_TO_MM
    return adapter.sphere(radius), (2 * radius, 2 * radius, 2 * radius)


def _concept_cad_source(spec: dict[str, Any]) -> str:
    body_json = json.dumps(spec["bodies"], allow_nan=False, indent=2, sort_keys=True)
    return (
        '"""Deterministic CadQuery source generated by Hephaestus.\n\n'
        "This file is a derived artifact. Canonical values live in hephaestus.project.json.\n"
        "Cylinder primitives use a local Y axis.\n"
        '"""\n\n'
        "import cadquery as cq\n\n"
        "M_TO_MM = 1000.0\n"
        f"BODIES = {body_json}\n\n"
        "def body_solid(body):\n"
        '    primitive = body["primitive_type"]\n'
        '    if primitive == "box":\n'
        '        return cq.Workplane("XY").box(body["size_x_m"] * M_TO_MM, body["size_y_m"] * M_TO_MM, body["size_z_m"] * M_TO_MM)\n'
        '    if primitive == "cylinder":\n'
        '        return cq.Workplane("XZ").circle(body["radius_m"] * M_TO_MM).extrude(body["length_m"] * M_TO_MM)\n'
        '    if primitive == "sphere":\n'
        '        return cq.Workplane("XY").sphere(body["radius_m"] * M_TO_MM)\n'
        '    raise ValueError(f"unsupported primitive: {primitive}")\n'
    )


def _dimensions_mm(values: dict[str, Any]) -> dict[str, float]:
    return {
        "chassis_length": _value(values, "chassis_length") * M_TO_MM,
        "chassis_width": _value(values, "chassis_width") * M_TO_MM,
        "chassis_height": _value(values, "chassis_height") * M_TO_MM,
        "wheel_radius": _value(values, "wheel_radius") * M_TO_MM,
        "wheel_width": _value(values, "wheel_width") * M_TO_MM,
        "caster_radius": _value(values, "caster_radius") * M_TO_MM,
    }


def _value(values: dict[str, Any], key: str) -> float:
    item = values[key]
    return float(item["value"])


def _check_solid(
    adapter: CadAdapter, name: str, solid: Any, expected: tuple[float, float, float]
) -> SolidCheck:
    volume = adapter.volume(solid)
    bbox = adapter.bounding_box(solid)
    status = "passed"
    if not math.isfinite(volume) or volume <= 0.0:
        status = "failed"
    if any(not math.isfinite(actual) or actual <= 0.0 for actual in bbox):
        status = "failed"
    for actual, target in zip(bbox, expected, strict=True):
        if abs(actual - target) > max(0.5, target * 0.01):
            status = "failed"
    return SolidCheck(name, volume, bbox, expected, status)


def _params_json(spec: dict[str, Any]) -> str:
    return json.dumps(spec, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _cad_source(spec: dict[str, Any]) -> str:
    values = spec["values"]
    return f'''"""Deterministic CadQuery source generated by Hephaestus.

This file is a derived artifact. Canonical values live in hephaestus.project.json.
"""

import cadquery as cq

M_TO_MM = {M_TO_MM!r}
CHASSIS_LENGTH_MM = {_value(values, "chassis_length") * M_TO_MM!r}
CHASSIS_WIDTH_MM = {_value(values, "chassis_width") * M_TO_MM!r}
CHASSIS_HEIGHT_MM = {_value(values, "chassis_height") * M_TO_MM!r}
WHEEL_RADIUS_MM = {_value(values, "wheel_radius") * M_TO_MM!r}
WHEEL_WIDTH_MM = {_value(values, "wheel_width") * M_TO_MM!r}
CASTER_RADIUS_MM = {_value(values, "caster_radius") * M_TO_MM!r}


def chassis():
    return cq.Workplane("XY").box(CHASSIS_LENGTH_MM, CHASSIS_WIDTH_MM, CHASSIS_HEIGHT_MM)


def wheel():
    return cq.Workplane("XZ").circle(WHEEL_RADIUS_MM).extrude(WHEEL_WIDTH_MM)


def caster():
    return cq.Workplane("XY").sphere(CASTER_RADIUS_MM)
'''


def _normalize_step_metadata(path: Path) -> None:
    content = path.read_text(encoding="utf-8", errors="replace")
    normalized = re.sub(
        r"(FILE_NAME\('[^']*',)'[^']*'",
        r"\g<1>'1970-01-01T00:00:00'",
        content,
        count=1,
    )
    normalized = re.sub(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "hephaestus-deterministic-step-id",
        normalized,
    )
    if normalized != content:
        path.write_text(normalized, encoding="utf-8")
