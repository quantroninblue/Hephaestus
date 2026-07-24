from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from hephaestus.artifacts import ArtifactStager, artifact_root, load_manifest
from hephaestus.backends.cadquery_backend import (
    GENERATOR_VERSION,
    CadGenerationError,
    FakeCadAdapter,
    _normalize_step_metadata,
    generate_cad_artifacts,
)
from hephaestus.mobile_robot import MobileRobotIntent, compile_mobile_robot_design
from hephaestus.model import EngineeringProject
from hephaestus.persistence import save_project


class BadBoundsAdapter(FakeCadAdapter):
    def bounding_box(self, solid: dict[str, object]) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)


def _mobile_robot_project() -> EngineeringProject:
    compiled = compile_mobile_robot_design(
        EngineeringProject.empty(name="CAD Test"),
        MobileRobotIntent(
            robot_name="CAD Test Bot",
            mission_summary="Inspect warehouse aisles offline.",
            payload_kg=5.0,
        ),
    )
    assert compiled.validation.is_valid
    return compiled.project


def test_generate_cad_artifacts_with_fake_adapter(tmp_path: Path) -> None:
    project = _mobile_robot_project()
    save_project(project, tmp_path)
    adapter = FakeCadAdapter()
    stager = ArtifactStager(tmp_path, project, backend_versions={"cadquery": adapter.version})

    records = generate_cad_artifacts(project, stager, adapter=adapter)
    manifest = stager.publish(
        records,
        generator_versions={"hephaestus.cadquery_backend": GENERATOR_VERSION},
    )
    root = artifact_root(tmp_path)

    paths = {record.path for record in manifest.artifacts}
    assert "cad/source/mobile_robot_cad.py" in paths
    assert "cad/source/mobile_robot_parameters.json" in paths
    assert "cad/step/chassis.step" in paths
    assert "cad/step/wheel.step" in paths
    assert "cad/step/caster.step" in paths
    assert "cad/step/assembly.step" in paths
    assert "cad/stl/visual/chassis.stl" in paths
    assert "cad/stl/collision/chassis.stl" in paths
    assert "reports/cad_report.json" in paths

    roles = {record.role for record in manifest.artifacts}
    assert "cad_assembly_step" in roles
    assert "visual_mesh_stl" in roles
    assert "collision_mesh_stl" in roles

    source = (root / "cad/source/mobile_robot_cad.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "import cadquery as cq" in source

    params = json.loads((root / "cad/source/mobile_robot_parameters.json").read_text())
    assert params["schema"] == "hephaestus.mobile_robot.v1"
    assert params["values"]["chassis_length"]["unit"] == "m"

    report = json.loads((root / "reports/cad_report.json").read_text())
    assert report["status"] == "passed"
    assert report["unit_conversion"] == {"canonical": "m", "cadquery": "mm", "factor": 1000.0}
    assert report["assembly_step"] == "generated"
    assert not report["warnings"]
    assert {solid["name"] for solid in report["solids"]} == {"chassis", "wheel", "caster"}
    assert all(solid["volume_mm3"] > 0.0 for solid in report["solids"])

    loaded, status = load_manifest(tmp_path)
    assert loaded is not None
    assert status == "fresh"
    assert loaded.project_fingerprint == manifest.project_fingerprint


def test_cad_generation_fails_when_solid_bounds_do_not_validate(tmp_path: Path) -> None:
    project = _mobile_robot_project()
    stager = ArtifactStager(tmp_path, project, backend_versions={"cadquery": "bad"})

    with pytest.raises(CadGenerationError, match="CAD solid validation failed"):
        generate_cad_artifacts(project, stager, adapter=BadBoundsAdapter())
    stager.cleanup()



def test_step_metadata_normalization_removes_export_timestamp_and_uuid(tmp_path: Path) -> None:
    step = tmp_path / "part.step"
    uuid = "a087f63c-8616-11f1-b9b8-bcfce7d2345d"
    file_name = (
        "FILE_NAME('Open CASCADE Shape Model','2026-07-23T03:06:43',"
        "('Author'),('Open CASCADE'),'processor','kernel','Unknown');\n"
    )
    step.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        f"{file_name}"
        f"#7 = PRODUCT('{uuid}','{uuid}','',(#8));\n"
        "ENDSEC;\n",
        encoding="utf-8",
    )

    _normalize_step_metadata(step)

    content = step.read_text(encoding="utf-8")
    assert "2026-07-23T03:06:43" not in content
    assert uuid not in content
    assert "hephaestus-deterministic-step-id" in content
    assert "FILE_NAME('Open CASCADE Shape Model','1970-01-01T00:00:00'" in content
