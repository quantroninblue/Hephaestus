from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.changes import (
    AddConnectionOperation,
    AddEntityOperation,
    ChangeSet,
    OperationKind,
    canonical_changeset_json,
    fingerprint_project,
)
from hephaestus.fixtures import representative_project
from hephaestus.mobile_robot import MobileRobotIntent, compile_mobile_robot_design
from hephaestus.model import (
    Connection,
    EngineeringProject,
    Entity,
    Geometry,
    Provenance,
    ProvenanceSource,
    Reference,
    Sensor,
    SensorModality,
    q,
)
from hephaestus.persistence import PROJECT_FILENAME, load_project, save_project

runner = CliRunner()


def test_cli_project_init_and_overwrite_refusal(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"

    result = runner.invoke(cli.app, ["project", "init", str(project_dir), "--name", "Test Robot"])

    assert result.exit_code == 0
    project_file = project_dir / PROJECT_FILENAME
    assert str(project_file) in result.output
    project = load_project(project_dir)
    assert project.metadata.name == "Test Robot"

    second = runner.invoke(cli.app, ["project", "init", str(project_dir), "--name", "Again"])
    assert second.exit_code == 1
    assert "already exists" in second.output


def test_cli_project_inspect_human_and_json(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"
    save_project(representative_project(), project_dir)

    human = runner.invoke(cli.app, ["project", "inspect", str(project_dir)])
    assert human.exit_code == 0
    assert "Hephaestus Engineering Project" in human.output
    assert "Representative Fixture" in human.output

    json_result = runner.invoke(cli.app, ["project", "inspect", str(project_dir), "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["name"] == "Representative Fixture"
    assert payload["counts"]["sensors"] == 2
    assert "\x1b[" not in json_result.output


def test_cli_project_validate_human_and_json(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"
    save_project(representative_project(), project_dir)

    human = runner.invoke(cli.app, ["project", "validate", str(project_dir), "--strict"])
    assert human.exit_code == 0
    assert "Hephaestus Project Validation" in human.output
    assert "project.valid" in human.output

    json_result = runner.invoke(cli.app, ["project", "validate", str(project_dir), "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["status"] == "valid"
    assert "\x1b[" not in json_result.output


def test_cli_project_validate_invalid_project_has_findings_and_strict_nonzero(
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "bad_project"
    project = representative_project()
    bad_geometry = Geometry.box(
        "geometry:bad",
        q(0.0, "m", "length"),
        q(1.0, "m", "length"),
        q(1.0, "m", "length"),
    )
    bad_sensor = Sensor(
        "sensor:bad",
        "Bad sensor",
        SensorModality.CUSTOM,
        Reference("frame", "frame:missing"),
    )
    save_project(
        replace(
            project,
            geometries=(*project.geometries, bad_geometry),
            sensors=(*project.sensors, bad_sensor),
        ),
        project_dir,
        validate=False,
    )

    json_result = runner.invoke(cli.app, ["project", "validate", str(project_dir), "--json"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    codes = {finding["code"] for finding in payload["findings"]}
    assert payload["status"] == "invalid"
    assert "geometry.dimension.non_positive" in codes
    assert "sensor.mounting_frame.missing" in codes

    strict = runner.invoke(cli.app, ["project", "validate", str(project_dir), "--strict"])
    assert strict.exit_code == 1
    assert "geometry.dimension.non_positive" in strict.output


def test_cli_validate_malformed_json_has_clean_error(tmp_path: Path) -> None:
    project_dir = tmp_path / "malformed"
    project_dir.mkdir()
    (project_dir / PROJECT_FILENAME).write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(cli.app, ["project", "validate", str(project_dir), "--json", "--strict"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "invalid"
    assert payload["findings"][0]["code"] == "project.load.failed"


def test_cli_project_apply_dry_run_apply_and_failure_preserves_file(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"
    save_project(representative_project(), project_dir)
    project_file = project_dir / PROJECT_FILENAME
    before_bytes = project_file.read_bytes()
    before_fingerprint = fingerprint_project(load_project(project_dir))
    provenance = Provenance(ProvenanceSource.USER, "unit test")
    valid_changes = ChangeSet(
        "change:cli_apply",
        0,
        "Add CLI entity.",
        provenance,
        (AddEntityOperation(OperationKind.ADD_ENTITY, Entity("entity:cli_added", "CLI added")),),
    )
    changeset_path = tmp_path / "changeset.json"
    changeset_path.write_text(canonical_changeset_json(valid_changes), encoding="utf-8")

    dry_run = runner.invoke(
        cli.app,
        ["project", "apply", str(project_dir), str(changeset_path), "--dry-run", "--json"],
    )
    assert dry_run.exit_code == 0
    dry_payload = json.loads(dry_run.output)
    assert dry_payload["dry_run"] is True
    assert dry_payload["before_fingerprint"] == before_fingerprint
    assert project_file.read_bytes() == before_bytes
    assert "\x1b[" not in dry_run.output

    applied = runner.invoke(
        cli.app,
        ["project", "apply", str(project_dir), str(changeset_path), "--json"],
    )
    assert applied.exit_code == 0
    applied_payload = json.loads(applied.output)
    assert applied_payload["revision"] == 1
    assert applied_payload["before_fingerprint"] == before_fingerprint
    assert applied_payload["after_fingerprint"] != before_fingerprint
    assert load_project(project_dir).revision == 1

    failing = ChangeSet(
        "change:bad_cli_apply",
        1,
        "Add invalid connection.",
        provenance,
        (
            AddConnectionOperation(
                OperationKind.ADD_CONNECTION,
                Connection(
                    "connection:self_cli",
                    "logical",
                    Reference("entity", "entity:cli_added"),
                    Reference("entity", "entity:cli_added"),
                ),
            ),
        ),
    )
    failing_path = tmp_path / "bad_changeset.json"
    failing_path.write_text(canonical_changeset_json(failing), encoding="utf-8")
    bytes_before_failure = project_file.read_bytes()

    failed = runner.invoke(
        cli.app,
        ["project", "apply", str(project_dir), str(failing_path), "--json"],
    )
    assert failed.exit_code == 1
    failed_payload = json.loads(failed.output)
    assert failed_payload["status"] == "failed"
    assert project_file.read_bytes() == bytes_before_failure


def test_cli_generate_requires_mobile_robot_design(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"
    save_project(representative_project(), project_dir)

    result = runner.invoke(cli.app, ["generate", str(project_dir), "--dry-run", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "invalid"
    assert payload["dry_run"] is True
    codes = {finding["code"] for finding in payload["validation"]["findings"]}
    assert "generation.robot_design.missing" in codes
    assert "\x1b[" not in result.output


def test_cli_generate_dry_run_accepts_compiled_mobile_robot(tmp_path: Path) -> None:
    project_dir = tmp_path / "robot_project"
    compiled = compile_mobile_robot_design(
        EngineeringProject.empty(name="Warehouse Robot"),
        MobileRobotIntent(
            robot_name="Aisle Scout",
            mission_summary="inspect pallet aisles for blocked exits",
            payload_kg=10.0,
            max_speed_mps=1.0,
        ),
    )
    save_project(compiled.project, project_dir)

    result = runner.invoke(cli.app, ["generate", str(project_dir), "--dry-run", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True
    assert payload["manifest"] is None
    assert "\x1b[" not in result.output
