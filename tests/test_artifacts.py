from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hephaestus.artifacts import (
    ArtifactError,
    ArtifactStager,
    artifact_root,
    dumps_manifest,
    inspect_artifacts,
    load_manifest,
    record_for_file,
    validate_relative_artifact_path,
)
from hephaestus.fixtures import representative_project
from hephaestus.persistence import save_project


def test_artifact_stager_publishes_manifest_and_detects_stale_project(tmp_path: Path) -> None:
    project = representative_project()
    save_project(project, tmp_path)
    stager = ArtifactStager(tmp_path, project, backend_versions={"fake_backend": "1.0"})
    stager.write_text("cad/reports/report.json", "{}\n")
    record = record_for_file(
        stager.staging,
        "cad/reports/report.json",
        role="cad_report",
        owning_ids=("entity:base",),
        units="mm",
    )

    manifest = stager.publish((record,), generator_versions={"test.generator": "1.0"})
    first_json = dumps_manifest(manifest)

    loaded, status = load_manifest(tmp_path)
    assert loaded is not None
    assert status == "fresh"
    assert dumps_manifest(loaded) == first_json
    assert (artifact_root(tmp_path) / "hephaestus.artifacts.json").is_file()

    summary = inspect_artifacts(tmp_path)
    assert summary["status"] == "fresh"
    assert summary["roles"] == {"cad_report": 1}

    save_project(replace(project, revision=project.revision + 1), tmp_path, overwrite=True)
    stale, stale_status = load_manifest(tmp_path)
    assert stale is not None
    assert stale_status == "stale"
    assert {artifact.status for artifact in stale.artifacts} == {"stale"}


def test_load_manifest_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    project = representative_project()
    save_project(project, tmp_path)
    stager = ArtifactStager(tmp_path, project)
    stager.write_text("cad/reports/report.json", "{}\n")
    record = record_for_file(
        stager.staging,
        "cad/reports/report.json",
        role="cad_report",
        owning_ids=("entity:base",),
    )
    stager.publish((record,), generator_versions={"test.generator": "1.0"})

    (artifact_root(tmp_path) / "cad/reports/report.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactError, match="hash mismatch"):
        load_manifest(tmp_path)


@pytest.mark.parametrize(
    "path",
    (
        "/absolute.step",
        "../escape.step",
        "cad/../escape.step",
        "cad/bad space.step",
        "cad/$bad.step",
        "cad\\windows.step",
    ),
)
def test_artifact_paths_must_be_safe_relative_paths(path: str) -> None:
    with pytest.raises(ArtifactError):
        validate_relative_artifact_path(path)
