from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.backends.cadquery_backend import FakeCadAdapter
from hephaestus.changes import fingerprint_project
from hephaestus.design import (
    DESIGN_EXTENSION_NS,
    DesignError,
    WarehouseInspectionRobotIntent,
    compile_warehouse_inspection_robot,
)
from hephaestus.generation import GenerationTarget, generate_project_artifacts
from hephaestus.mobile_robot import (
    DESIGN_EXTENSION_KEY as MOBILE_ROBOT_EXTENSION_KEY,
)
from hephaestus.mobile_robot import (
    MobileRobotIntent,
    compile_mobile_robot_design,
)
from hephaestus.model import (
    EngineeringProject,
    Entity,
    QuantityRange,
    Requirement,
    RequirementKind,
    q,
)
from hephaestus.persistence import canonical_project_json, load_project, save_project
from hephaestus.validation import validate_project

runner = CliRunner()


def _intent(**overrides: object) -> WarehouseInspectionRobotIntent:
    values = {
        "robot_family": "differential_drive",
        "robot_name": "Aisle Scout",
        "mission": "inspect pallet aisles for blocked exits",
        "warehouse_environment": "dry indoor warehouse with 2.4 m aisles",
        "payload_kg": 12.0,
        "max_speed_mps": 1.4,
        "chassis_length_m": 0.9,
        "chassis_width_m": 0.58,
        "chassis_height_m": 0.3,
        "wheel_diameter_m": 0.18,
        "wheel_width_m": 0.055,
        "has_lidar": True,
        "has_imu": True,
    }
    values.update(overrides)
    return WarehouseInspectionRobotIntent(**values)


def _mobile_intent(**overrides: object) -> MobileRobotIntent:
    values = {
        "robot_name": "Wide Scout",
        "mission_summary": "inspect warehouse aisles and report blocked exits",
        "payload_kg": 9.0,
        "max_speed_mps": 1.2,
        "chassis_length_m": 0.9,
        "chassis_width_m": 0.82,
        "chassis_height_m": 0.28,
        "wheel_radius_m": 0.1,
        "include_lidar": True,
        "include_imu": True,
        "indoor_warehouse": True,
    }
    values.update(overrides)
    return MobileRobotIntent(**values)


def test_mobile_robot_compiler_derives_wheel_separation_for_wide_chassis() -> None:
    compilation = compile_mobile_robot_design(
        EngineeringProject.empty(name="Warehouse Robot"),
        _mobile_intent(),
    )

    assert compilation.validation.is_valid
    spec = compilation.project.extensions[MOBILE_ROBOT_EXTENSION_KEY]
    values = spec["values"]
    assert values["wheel_separation"]["value"] > values["chassis_width"]["value"]
    assert values["wheel_separation"]["source"] == "application_default"
    assert validate_project(compilation.project).is_valid


def test_mobile_robot_reuses_existing_structured_payload_requirement() -> None:
    existing_payload = Requirement(
        "requirement:scanner_payload",
        RequirementKind.PAYLOAD,
        "Carry the existing scanner payload.",
        value=QuantityRange(minimum=q(18.0, "kg", "mass")),
    )
    base = replace(
        EngineeringProject.empty(name="Warehouse Robot"),
        requirements=(existing_payload,),
    )

    compilation = compile_mobile_robot_design(base, _mobile_intent(payload_kg=None))

    payload_requirements = [
        item for item in compilation.project.requirements if item.kind is RequirementKind.PAYLOAD
    ]
    assert [item.id for item in payload_requirements] == ["requirement:scanner_payload"]
    spec = compilation.project.extensions[MOBILE_ROBOT_EXTENSION_KEY]
    assert spec["ids"]["payload_requirement"] == "requirement:scanner_payload"
    assert spec["values"]["payload"] == {
        "dimension": "mass",
        "source": "existing_structured_requirement",
        "source_id": "requirement:scanner_payload",
        "unit": "kg",
        "value": 18.0,
    }




def test_mobile_robot_reuses_matching_explicit_existing_payload_requirement() -> None:
    existing_payload = Requirement(
        "requirement:payload_capacity_20kg",
        RequirementKind.PAYLOAD,
        "Payload capacity at least 20 kg.",
        value=QuantityRange(minimum=q(20.0, "kg", "mass")),
    )
    base = replace(
        EngineeringProject.empty(name="Genesis Live Recovery"),
        revision=1,
        requirements=(existing_payload,),
    )

    compilation = compile_mobile_robot_design(
        base,
        _mobile_intent(payload_kg=20.0, explicitly_supersedes_payload=True),
    )

    assert compilation.project.revision == 2
    payload_requirements = [
        item for item in compilation.project.requirements if item.kind is RequirementKind.PAYLOAD
    ]
    assert [item.id for item in payload_requirements] == ["requirement:payload_capacity_20kg"]
    spec = compilation.project.extensions[MOBILE_ROBOT_EXTENSION_KEY]
    assert spec["ids"]["payload_requirement"] == "requirement:payload_capacity_20kg"
    assert spec["values"]["payload"]["source"] == "existing_structured_requirement"
    assert spec["values"]["payload"]["value"] == 20.0


def test_mobile_robot_rejects_unsuperseded_payload_conflict() -> None:
    existing_payload = Requirement(
        "requirement:payload_capacity_20kg",
        RequirementKind.PAYLOAD,
        "Payload capacity at least 20 kg.",
        value=QuantityRange(minimum=q(20.0, "kg", "mass")),
    )
    base = replace(
        EngineeringProject.empty(name="Genesis Live Recovery"),
        revision=1,
        requirements=(existing_payload,),
    )

    try:
        compile_mobile_robot_design(base, _mobile_intent(payload_kg=15.0))
    except ValueError as exc:
        assert "contradicts an existing structured payload requirement" in str(exc)
    else:
        raise AssertionError("conflicting payload should fail before mutation")


def test_revision_one_existing_payload_recovery_generates_fresh_artifacts(
    tmp_path: Path,
) -> None:
    existing_payload = Requirement(
        "requirement:payload_capacity_20kg",
        RequirementKind.PAYLOAD,
        "Payload capacity at least 20 kg.",
        value=QuantityRange(minimum=q(20.0, "kg", "mass")),
    )
    project_dir = tmp_path / "genesis_live_recovery"
    base = replace(
        EngineeringProject.empty(name="Genesis Live Recovery"),
        revision=1,
        requirements=(existing_payload,),
    )
    save_project(base, project_dir)

    compilation = compile_mobile_robot_design(
        load_project(project_dir),
        _mobile_intent(payload_kg=20.0, max_speed_mps=1.5),
    )
    save_project(compilation.project, project_dir, overwrite=True)
    generation = generate_project_artifacts(
        project_dir,
        target=GenerationTarget.ALL,
        cad_adapter=FakeCadAdapter(),
        backend_versions={"cadquery": "fake"},
    )

    recovered = load_project(project_dir)
    payload_requirements = [
        item for item in recovered.requirements if item.kind is RequirementKind.PAYLOAD
    ]
    assert recovered.revision == 2
    assert [item.id for item in payload_requirements] == ["requirement:payload_capacity_20kg"]
    assert generation.status == "generated"
    assert generation.manifest is not None
    assert generation.manifest.project_revision == 2
    assert generation.manifest.project_fingerprint == fingerprint_project(recovered)
    assert all(record.status == "fresh" for record in generation.manifest.artifacts)

def test_mobile_robot_rejects_generated_id_collisions() -> None:
    base = replace(
        EngineeringProject.empty(name="Warehouse Robot"),
        entities=(Entity("entity:wide_scout", "Existing robot"),),
    )

    try:
        compile_mobile_robot_design(base, _mobile_intent())
    except ValueError as exc:
        assert "duplicate existing canonical IDs" in str(exc)
        assert "entity:wide_scout" in str(exc)
    else:
        raise AssertionError("mobile robot compiler should reject duplicate generated IDs")


def test_mobile_robot_speed_requirement_is_canonical_maximum() -> None:
    compilation = compile_mobile_robot_design(
        EngineeringProject.empty(name="Warehouse Robot"),
        _mobile_intent(max_speed_mps=1.35),
    )

    requirement = next(
        item for item in compilation.project.requirements if item.kind is RequirementKind.SPEED
    )
    assert isinstance(requirement.value, QuantityRange)
    assert requirement.value.minimum is None
    assert requirement.value.target is None
    assert requirement.value.maximum is not None
    assert requirement.value.maximum.value == 1.35
    assert requirement.value.maximum.unit == "m/s"


def test_warehouse_inspection_robot_compiles_valid_canonical_roundtrip() -> None:
    project = EngineeringProject.empty(name="Warehouse Robot")

    compilation = compile_warehouse_inspection_robot(project, _intent())
    compiled = compilation.project

    assert compilation.before_fingerprint == fingerprint_project(project)
    assert compilation.after_fingerprint == fingerprint_project(compiled)
    assert compilation.validation.is_valid
    assert compiled.revision == 1
    assert len(compiled.entities) >= 6
    assert len(compiled.bodies) == 3
    assert len(compiled.joints) == 2
    assert len(compiled.actuators) == 2
    assert {sensor.modality.value for sensor in compiled.sensors} == {"lidar_2d", "imu"}
    assert compiled.software_components
    assert compiled.simulation_scenarios[0].environment == "dry indoor warehouse with 2.4 m aisles"
    assert validate_project(compiled).is_valid

    canonical = canonical_project_json(compiled)
    loaded = EngineeringProject.from_dict(json.loads(canonical))
    assert canonical_project_json(loaded) == canonical
    assert (
        compiled.extensions[DESIGN_EXTENSION_NS]["value_sources"]["payload_kg"]["source"]
        == "user_provided"
    )


def test_warehouse_inspection_robot_records_defaults_and_existing_requirements() -> None:
    base = EngineeringProject.empty(name="Warehouse Robot")
    base = replace(
        base,
        requirements=(
            Requirement(
                "requirement:existing_payload",
                RequirementKind.PAYLOAD,
                "Carry the scanner payload.",
                value=QuantityRange(minimum=q(18.0, "kg", "mass")),
            ),
        ),
    )

    compilation = compile_warehouse_inspection_robot(
        base,
        _intent(payload_kg=None, max_speed_mps=None, has_lidar=None, has_imu=False),
    )
    sources = compilation.value_sources

    assert sources["payload_kg"].source.value == "existing_requirement"
    assert sources["payload_kg"].value == 18.0
    assert sources["max_speed_mps"].source.value == "default_policy"
    assert sources["has_lidar"].source.value == "default_policy"
    assert sources["has_imu"].source.value == "user_provided"
    assert {sensor.modality.value for sensor in compilation.project.sensors} == {"lidar_2d"}
    assert any(
        "max_speed_mps uses default_policy" in item.statement
        for item in compilation.project.assumptions
    )


def test_warehouse_inspection_robot_rejects_unsupported_family() -> None:
    try:
        compile_warehouse_inspection_robot(
            EngineeringProject.empty(name="Warehouse Robot"),
            _intent(robot_family="tracked"),
        )
    except DesignError as exc:
        assert "unsupported robot family" in str(exc)
        assert "differential_drive" in str(exc)
    else:
        raise AssertionError("unsupported robot family should fail")


def test_cli_design_dry_run_and_apply_json(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    save_project(EngineeringProject.empty(name="Warehouse Robot"), project_dir)
    before = fingerprint_project(load_project(project_dir))

    dry_run = runner.invoke(
        cli.app,
        [
            "project",
            "design",
            "warehouse-inspection-robot",
            str(project_dir),
            "--name",
            "Aisle Scout",
            "--mission",
            "inspect aisles",
            "--environment",
            "dry indoor warehouse",
            "--payload-kg",
            "10",
            "--max-speed-mps",
            "1.1",
            "--wheel-width-m",
            "0.06",
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["operation"] == "create_mobile_robot"
    assert dry_payload["status"] == "planned"
    assert dry_payload["dry_run"] is True
    assert load_project(project_dir).revision == 0
    assert fingerprint_project(load_project(project_dir)) == before
    assert "\x1b[" not in dry_run.output

    applied = runner.invoke(
        cli.app,
        [
            "project",
            "design",
            "warehouse-inspection-robot",
            str(project_dir),
            "--name",
            "Aisle Scout",
            "--mission",
            "inspect aisles",
            "--environment",
            "dry indoor warehouse",
            "--payload-kg",
            "10",
            "--max-speed-mps",
            "1.1",
            "--wheel-width-m",
            "0.06",
            "--yes",
            "--json",
        ],
    )
    assert applied.exit_code == 0
    payload = json.loads(applied.output)
    assert payload["status"] == "applied"
    assert payload["operation"] == "create_mobile_robot"
    assert payload["revision"] == 1
    project = load_project(project_dir)
    assert project.revision == 1
    assert MOBILE_ROBOT_EXTENSION_KEY in project.extensions
    assert project.extensions[MOBILE_ROBOT_EXTENSION_KEY]["values"]["wheel_width"]["value"] == 0.06
    assert len(project.simulation_scenarios) == 1
