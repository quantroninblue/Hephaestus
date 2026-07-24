from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hephaestus.changes import (
    AddConnectionOperation,
    AddEntityOperation,
    ChangeError,
    ChangeSet,
    ChangeSetLoadError,
    ConnectOperation,
    OperationKind,
    RemoveConnectionOperation,
    RemoveEntityOperation,
    UpdateEntityOperation,
    apply_changeset_file,
    canonical_changeset_json,
    fingerprint_project,
    load_changeset,
)
from hephaestus.fixtures import representative_project
from hephaestus.index import ProjectIndex
from hephaestus.model import (
    Connection,
    ElectricalDevice,
    EngineeringProject,
    Entity,
    EntityKind,
    Geometry,
    GeometryKind,
    Joint,
    JointKind,
    Material,
    MeshFormat,
    ModelError,
    PowerConnection,
    Provenance,
    ProvenanceSource,
    Quaternion,
    Reference,
    Sensor,
    SensorModality,
    Transform,
    UnsupportedSchemaVersionError,
    q,
)
from hephaestus.persistence import (
    PROJECT_FILENAME,
    ProjectPersistenceError,
    canonical_project_json,
    load_project,
    save_project,
)
from hephaestus.schema import (
    changeset_schema,
    engineering_project_schema,
    operation_schema,
    schema_json,
)
from hephaestus.validation import validate_project


def test_unit_conversion_and_dimensional_rejection() -> None:
    assert q(100.0, "cm", "length").to("m").value == pytest.approx(1.0)
    assert q(180.0, "deg", "angle").to("rad").value == pytest.approx(3.141592653589793)
    with pytest.raises(ModelError):
        q(1.0, "kg", "length")
    with pytest.raises(ModelError):
        q(1.0, "m", "length").to("kg")


@given(st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_quantity_length_round_trip_property(value: float) -> None:
    assert q(value, "m", "length").to("cm").to("m").value == pytest.approx(value)


def test_pose_and_quaternion_validation() -> None:
    project = representative_project()
    zero_transform = Transform(
        "transform:bad_quat",
        Reference("frame", "frame:world"),
        Reference("frame", "frame:base"),
        replace(project.transforms[0].pose, orientation=Quaternion(0.0, 0.0, 0.0, 0.0)),
    )
    invalid = replace(project, transforms=(zero_transform,))
    codes = {finding.code for finding in validate_project(invalid).findings}
    assert "pose.quaternion.invalid" in codes

def test_geometry_variants_and_mesh_formats() -> None:
    for mesh_format in MeshFormat:
        mesh = Geometry.mesh(
            f"geometry:{mesh_format.value}", f"asset.{mesh_format.value}", mesh_format
        )
        assert mesh.mesh_format is mesh_format
    sphere = Geometry("geometry:sphere", GeometryKind.SPHERE, radius=q(0.2, "m", "length"))
    capsule = Geometry(
        "geometry:capsule",
        GeometryKind.CAPSULE,
        radius=q(0.1, "m", "length"),
        length=q(0.4, "m", "length"),
    )
    parametric = Geometry(
        "geometry:parametric",
        GeometryKind.PARAMETRIC,
        parametric_kind="extrusion",
        parameters={"depth": q(0.1, "m", "length")},
    )
    assert {sphere.kind, capsule.kind, parametric.kind}


def test_stable_serialization_and_round_trip(tmp_path: Path) -> None:
    project = representative_project()
    first = canonical_project_json(project)
    second = canonical_project_json(project)
    assert first == second
    parsed = EngineeringProject.from_dict(json.loads(first))
    assert canonical_project_json(parsed) == first

    saved_path = save_project(project, tmp_path)
    loaded = load_project(tmp_path)
    assert saved_path.name == "hephaestus.project.json"
    assert canonical_project_json(loaded) == first


def test_unsupported_schema_version() -> None:
    data = representative_project().to_dict()
    data["schema_version"] = "999.0.0"
    with pytest.raises(UnsupportedSchemaVersionError):
        EngineeringProject.from_dict(data)


def test_validation_detects_dangling_and_physical_failures() -> None:
    project = representative_project()
    bad_joint = Joint(
        "joint:bad",
        "Bad joint",
        JointKind.REVOLUTE,
        Reference("body", "body:missing"),
        Reference("body", "body:base"),
        axis=(0.0, 0.0, 0.0),
    )
    bad_geometry = Geometry.box(
        "geometry:bad_box",
        q(0.0, "m", "length"),
        q(1.0, "m", "length"),
        q(1.0, "m", "length"),
    )
    invalid = replace(
        project, geometries=(*project.geometries, bad_geometry), joints=(*project.joints, bad_joint)
    )
    codes = {finding.code for finding in validate_project(invalid).findings}
    assert "reference.dangling" in codes
    assert "geometry.dimension.non_positive" in codes
    assert "joint.axis.invalid" in codes
    assert "joint.parent_body.missing" in codes


def test_electrical_and_sensor_reference_failures() -> None:
    project = representative_project()
    bad_sensor = Sensor(
        "sensor:bad",
        "Bad sensor",
        SensorModality.CUSTOM,
        Reference("frame", "frame:missing"),
    )
    bad_power = PowerConnection(
        "power:bad",
        Reference("electrical_port", "port:missing"),
        Reference("electrical_port", "port:driver_power"),
    )
    invalid = replace(
        project,
        sensors=(*project.sensors, bad_sensor),
        power_connections=(*project.power_connections, bad_power),
    )
    codes = {finding.code for finding in validate_project(invalid).findings}
    assert "sensor.mounting_frame.missing" in codes
    assert "electrical.connection.source_missing" in codes


def test_representative_project_fixture_has_breadth() -> None:
    project = representative_project()
    report = validate_project(project)
    assert report.is_valid
    assert len(project.bodies) >= 2
    assert len(project.sensors) >= 2
    assert project.electrical_devices
    assert project.communication_buses
    assert project.compute_devices
    assert project.software_components
    assert project.simulation_scenarios


def test_changeset_success_and_fingerprints() -> None:
    project = EngineeringProject.empty(name="Change Test")
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    changes = ChangeSet(
        "change:add_entity",
        0,
        "Add payload bay entity.",
        provenance,
        (
            AddEntityOperation(
                OperationKind.ADD_ENTITY,
                Entity("entity:payload_bay", "Payload bay", EntityKind.CUSTOM),
            ),
        ),
    )
    result = changes.apply(project)
    assert result.project.revision == 1
    assert result.before_fingerprint == fingerprint_project(project)
    assert result.after_fingerprint == fingerprint_project(result.project)
    assert result.before_fingerprint != result.after_fingerprint


def test_changeset_rollback_on_failure_and_revision_precondition() -> None:
    project = representative_project()
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    failing = ChangeSet(
        "change:bad_connection",
        project.revision,
        "Add invalid self connection.",
        provenance,
        (
            ConnectOperation(
                OperationKind.CONNECT,
                connection=Connection(
                    "connection:self",
                    "test",
                    Reference("entity", "entity:base"),
                    Reference("entity", "entity:base"),
                ),
            ),
        ),
    )
    before = fingerprint_project(project)
    with pytest.raises(ChangeError):
        failing.apply(project)
    assert fingerprint_project(project) == before

    stale = ChangeSet(
        "change:stale",
        999,
        "Stale",
        provenance,
        (AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:stale", "Stale")),),
    )
    with pytest.raises(ChangeError):
        stale.apply(project)


def test_remove_entity_requires_references_to_permit_it() -> None:
    project = representative_project()
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    changes = ChangeSet(
        "change:remove_referenced",
        project.revision,
        "Remove referenced entity.",
        provenance,
        (RemoveEntityOperation(OperationKind.REMOVE_ENTITY, "entity:base"),),
    )
    with pytest.raises(ChangeError):
        changes.apply(project)


def test_broad_quantity_units_and_incompatible_conversions() -> None:
    valid = [
        (1000.0, "mm", "length", "m", 1.0),
        (100.0, "cm", "length", "m", 1.0),
        (1.0, "km", "length", "m", 1000.0),
        (1000.0, "g", "mass", "kg", 1.0),
        (1000.0, "ms", "time", "s", 1.0),
        (1.0, "min", "time", "s", 60.0),
        (1.0, "h", "time", "s", 3600.0),
        (180.0, "deg", "angle", "rad", math.pi),
        (5.0, "Hz", "frequency", "Hz", 5.0),
        (2.0, "m/s", "speed", "m/s", 2.0),
        (3.0, "m/s²", "acceleration", "m/s^2", 3.0),
        (4.0, "rad/s", "angular_velocity", "rad/s", 4.0),
        (5.0, "N", "force", "N", 5.0),
        (6.0, "N·m", "torque", "N*m", 6.0),
        (7.0, "Pa", "pressure", "Pa", 7.0),
        (8.0, "W", "power", "W", 8.0),
        (9.0, "J", "energy", "J", 9.0),
        (10.0, "V", "voltage", "V", 10.0),
        (11.0, "A", "current", "A", 11.0),
        (2.0, "Ah", "capacity", "Ah", 2.0),
        (3.0, "Wh", "energy", "J", 10800.0),
        (2700.0, "kg/m³", "density", "kg/m^3", 2700.0),
        (0.12, "kg·m²", "inertia", "kg*m^2", 0.12),
        (50.0, "%", "dimensionless", "1", 0.5),
    ]
    for value, unit, dimension, target, expected in valid:
        assert q(value, unit, dimension).to(target).value == pytest.approx(expected)

    invalid = [
        ("m", "length", "s"),
        ("kg", "mass", "N"),
        ("rad/s", "angular_velocity", "m/s"),
        ("V", "voltage", "A"),
        ("J", "energy", "W"),
        ("kg*m^2", "inertia", "kg"),
    ]
    for unit, dimension, target in invalid:
        with pytest.raises(ModelError):
            q(1.0, unit, dimension).to(target)

    with pytest.raises(ModelError):
        q(float("nan"), "m", "length")
    with pytest.raises(ModelError):
        q(1.0, "not_a_real_unit", "length")
    with pytest.raises(ModelError):
        q(1.0, "J", "torque")
    with pytest.raises(ModelError):
        q(1.0, "N*m", "energy")
    with pytest.raises(ModelError):
        q(1.0, "kg*m^2", "energy")


def test_strict_model_loading_rejects_unknown_and_bad_reference_kind(tmp_path: Path) -> None:
    data = representative_project().to_dict()
    data["entities"][0]["unexpected"] = True
    project_dir = tmp_path / "strict"
    project_dir.mkdir()
    (project_dir / PROJECT_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(Exception, match="unknown fields"):
        load_project(project_dir)

    with pytest.raises(ModelError):
        Reference("ros_link", "entity:base")


def test_strict_model_loading_rejects_unknown_nested_core_fields(tmp_path: Path) -> None:
    data = representative_project().to_dict()
    data["bodies"][0]["entity_ref"]["unexpected"] = True
    project_dir = tmp_path / "bad_reference"
    project_dir.mkdir()
    (project_dir / PROJECT_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProjectPersistenceError, match="unknown fields for Reference"):
        load_project(project_dir)

    data = representative_project().to_dict()
    data["requirements"][0]["value"]["minimum"]["unexpected"] = True
    project_dir = tmp_path / "bad_quantity"
    project_dir.mkdir()
    (project_dir / PROJECT_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProjectPersistenceError, match="unknown fields for Quantity"):
        load_project(project_dir)


def test_strict_model_loading_rejects_malformed_fixed_tuple_cleanly(tmp_path: Path) -> None:
    data = representative_project().to_dict()
    tensor = data["bodies"][0]["mass_properties"]["inertia_tensor"]
    tensor[0] = tensor[0][:2]
    project_dir = tmp_path / "bad_tensor"
    project_dir.mkdir()
    (project_dir / PROJECT_FILENAME).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ProjectPersistenceError, match="expected array with 3 items"):
        load_project(project_dir)


def test_persistence_refuses_symlink_project_paths(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ProjectPersistenceError, match="symlink"):
        save_project(EngineeringProject.empty(name="Symlink"), link / "project")

    project_dir = target / "project"
    save_project(EngineeringProject.empty(name="Symlink"), project_dir)
    project_file_link = tmp_path / PROJECT_FILENAME
    project_file_link.symlink_to(project_dir / PROJECT_FILENAME)

    with pytest.raises(ProjectPersistenceError, match="symlink"):
        load_project(project_file_link)

def test_geometry_variant_exclusivity_and_material_density_units() -> None:
    with pytest.raises(ModelError):
        Geometry(
            "geometry:bad_variant",
            GeometryKind.MESH,
            dimensions=(q(1.0, "m", "length"),),
            asset_uri="asset.glb",
            mesh_format=MeshFormat.GLB,
        )
    with pytest.raises(ModelError):
        Material("material:bad_density", "Bad", density=q(1.0, "kg", "mass"))


def test_inertia_and_electrical_physical_validation_findings() -> None:
    project = representative_project()
    bad_inertia = (
        (q(10.0, "kg*m^2", "inertia"), q(1.0, "kg*m^2", "inertia"), q(0.0, "kg*m^2", "inertia")),
        (q(0.0, "kg*m^2", "inertia"), q(1.0, "kg*m^2", "inertia"), q(0.0, "kg*m^2", "inertia")),
        (q(0.0, "kg*m^2", "inertia"), q(0.0, "kg*m^2", "inertia"), q(1.0, "kg*m^2", "inertia")),
    )
    bad_body = replace(
        project.bodies[0],
        mass_properties=replace(project.bodies[0].mass_properties, inertia_tensor=bad_inertia),
    )
    bad_device = ElectricalDevice(
        "electrical:bad",
        "Bad supply",
        "supply",
        voltage_nominal=q(0.0, "V", "voltage"),
        power_limit=q(-1.0, "W", "power"),
    )
    invalid = replace(
        project,
        bodies=(bad_body, *project.bodies[1:]),
        electrical_devices=(*project.electrical_devices, bad_device),
    )
    codes = {finding.code for finding in validate_project(invalid).findings}
    assert "inertia.symmetry.invalid" in codes
    assert "inertia.principal_moments.implausible" in codes
    assert "electrical.device.voltage.non_positive" in codes
    assert "electrical.device.power.negative" in codes


def test_project_index_resolves_references_and_removal_blockers() -> None:
    project = representative_project()
    index = ProjectIndex.build(project)

    assert index.resolve(Reference("entity", "entity:base")) is not None
    assert index.resolve_component_as(Reference("component", "geometry:base_box"), {"geometries"})
    blockers = index.removal_blockers("entity", "entity:base")
    assert any("body" in blocker or "entities" in blocker for blocker in blockers)


def test_changeset_serialization_loading_and_all_supported_operations(tmp_path: Path) -> None:
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    changes = ChangeSet(
        "change:all_ops",
        0,
        "Exercise supported operations.",
        provenance,
        (
            AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:a", "A")),
            AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:b", "B")),
            AddConnectionOperation(
                OperationKind.ADD_CONNECTION,
                Connection(
                    "connection:a_b",
                    "logical",
                    Reference("entity", "entity:a"),
                    Reference("entity", "entity:b"),
                ),
            ),
            RemoveConnectionOperation(OperationKind.REMOVE_CONNECTION, "connection:a_b"),
            UpdateEntityOperation(OperationKind.UPDATE_ENTITY, "entity:b", name="B renamed"),
            RemoveEntityOperation(OperationKind.REMOVE_ENTITY, "entity:a"),
        ),
    )
    payload = canonical_changeset_json(changes)
    assert canonical_changeset_json(changes) == payload
    changeset_path = tmp_path / "changeset.json"
    changeset_path.write_text(payload, encoding="utf-8")

    loaded = load_changeset(changeset_path)
    result = loaded.apply(EngineeringProject.empty(name="Ops"))

    assert result.project.revision == 1
    assert [entity.id for entity in result.project.entities] == ["entity:b"]
    assert result.project.entities[0].name == "B renamed"


def test_malformed_changeset_and_unknown_operation_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_changeset.json"
    path.write_text(
        json.dumps(
            {
                "id": "change:bad",
                "expected_revision": 0,
                "summary": "Bad",
                "provenance": {"source": "user"},
                "operations": [{"operation": "run_python", "code": "print(1)"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChangeSetLoadError, match="unknown ChangeSet operation"):
        load_changeset(path)

    path.write_text(
        json.dumps(
            {
                "id": "change:extra_operation_field",
                "expected_revision": 0,
                "summary": "Bad",
                "provenance": {"source": "user"},
                "operations": [
                    {
                        "operation": "add_entity",
                        "entity": {"id": "entity:extra", "name": "Extra"},
                        "unexpected": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChangeSetLoadError, match="unknown fields for add_entity operation"):
        load_changeset(path)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ChangeSetLoadError, match="malformed"):
        load_changeset(path)


def test_changeset_file_apply_rechecks_project_fingerprint_before_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project = EngineeringProject.empty(name="Race")
    save_project(project, project_dir)
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    changes = ChangeSet(
        "change:race",
        0,
        "Add race entity.",
        provenance,
        (AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:race", "Race")),),
    )
    changeset_path = tmp_path / "changeset.json"
    changeset_path.write_text(canonical_changeset_json(changes), encoding="utf-8")

    first_load = load_project(project_dir)
    changed_load = replace(first_load, metadata=replace(first_load.metadata, description="changed"))
    loads = 0
    saved: list[object] = []

    def fake_load_project(path: Path) -> EngineeringProject:
        nonlocal loads
        loads += 1
        return first_load if loads == 1 else changed_load

    def fake_save_project(*args: object, **kwargs: object) -> None:
        saved.append((args, kwargs))

    monkeypatch.setattr("hephaestus.changes.load_project", fake_load_project)
    monkeypatch.setattr("hephaestus.changes.save_project", fake_save_project)

    with pytest.raises(ChangeError, match="project file changed after ChangeSet evaluation"):
        apply_changeset_file(project_dir, changeset_path)
    assert saved == []



def test_changeset_final_validation_rollback_and_dry_run_immutability() -> None:
    project = representative_project()
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    invalid = ChangeSet(
        "change:self_connection",
        project.revision,
        "Invalid connection.",
        provenance,
        (
            AddConnectionOperation(
                OperationKind.ADD_CONNECTION,
                Connection(
                    "connection:self_invalid",
                    "logical",
                    Reference("entity", "entity:base"),
                    Reference("entity", "entity:base"),
                ),
            ),
        ),
    )
    before = fingerprint_project(project)
    with pytest.raises(ChangeError, match="operation 0|final validation"):
        invalid.apply(project)
    assert fingerprint_project(project) == before

    valid = ChangeSet(
        "change:dry_run",
        project.revision,
        "Dry run add entity.",
        provenance,
        (AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:temporary", "Temporary")),),
    )
    result = valid.apply(project, dry_run=True)
    assert result.dry_run
    assert result.project.revision == project.revision + 1
    assert fingerprint_project(project) == before


def test_fingerprints_and_schema_generation_are_deterministic_and_parseable() -> None:
    project = representative_project()
    assert fingerprint_project(project) == fingerprint_project(project)

    project_schema = engineering_project_schema()
    changes_schema = changeset_schema()
    add_schema = operation_schema("add_entity")
    for schema in (project_schema, changes_schema, add_schema):
        encoded = schema_json(schema)
        assert json.loads(encoded) == schema
        assert "\x1b[" not in encoded
    assert changes_schema["properties"]["operations"]["items"]["oneOf"]
