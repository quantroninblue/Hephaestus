"""Milestone 3.0 artifact generation orchestration."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from hephaestus.artifacts import ArtifactManifest, ArtifactStager, load_manifest
from hephaestus.backends.cadquery_backend import (
    GENERATOR_VERSION as CAD_GENERATOR_VERSION,
)
from hephaestus.backends.cadquery_backend import (
    CadAdapter,
    generate_cad_artifacts,
)
from hephaestus.backends.ros_gazebo import (
    GENERATOR_VERSION as ROS_GAZEBO_GENERATOR_VERSION,
)
from hephaestus.backends.ros_gazebo import (
    generate_ros_gazebo_artifacts,
)
from hephaestus.changes import fingerprint_project
from hephaestus.concept_robot import (
    CONCEPT_EXTENSION_KEY,
    CONCEPT_GENERATOR_VERSION,
    ConceptError,
    concept_robot_spec,
    validate_concept_project,
)
from hephaestus.mobile_robot import (
    DESIGN_EXTENSION_KEY,
    mobile_robot_spec,
    validate_mobile_robot_project,
)
from hephaestus.mobile_robot import GENERATOR_VERSION as MOBILE_ROBOT_GENERATOR_VERSION
from hephaestus.persistence import load_project, resolve_project_file
from hephaestus.validation import (
    ValidationFinding,
    ValidationReport,
    ValidationSeverity,
    validate_project,
)


class GenerationTarget(StrEnum):
    ALL = "all"
    CAD = "cad"
    ROS2 = "ros2"
    GAZEBO = "gazebo"


class GenerationError(RuntimeError):
    """Raised when project artifact generation cannot complete safely."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    target: GenerationTarget
    project_path: Path
    dry_run: bool
    validation: ValidationReport
    plan: tuple[str, ...]
    manifest: ArtifactManifest | None = None

    @property
    def status(self) -> str:
        if not self.validation.is_valid:
            return "invalid"
        if self.dry_run:
            return "planned"
        return "generated" if self.manifest is not None else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hephaestus.generation.result.v1",
            "status": self.status,
            "target": self.target.value,
            "project_path": str(self.project_path),
            "dry_run": self.dry_run,
            "plan": list(self.plan),
            "validation": self.validation.to_dict(),
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
        }


def generate_project_artifacts(
    project_path: Path,
    *,
    target: GenerationTarget | str = GenerationTarget.ALL,
    dry_run: bool = False,
    cad_adapter: CadAdapter | None = None,
    backend_versions: Mapping[str, str | None] | None = None,
) -> GenerationResult:
    selected = target if isinstance(target, GenerationTarget) else GenerationTarget(str(target))
    project = load_project(project_path)
    project_file = resolve_project_file(project_path)
    project_dir = project_file.parent
    validation = _combined_validation(project)
    plan = _plan_for(selected)
    if not validation.is_valid:
        return GenerationResult(selected, project_file, dry_run, validation, plan)
    if dry_run:
        return GenerationResult(selected, project_file, True, validation, plan)

    versions = dict(backend_versions or {})
    if "cadquery" not in versions:
        versions["cadquery"] = _package_version("cadquery")
    stager = ArtifactStager(project_dir, project, backend_versions=versions)
    try:
        records = []
        generator_versions = _design_generator_versions(project)
        if selected in {
            GenerationTarget.ALL,
            GenerationTarget.CAD,
            GenerationTarget.ROS2,
            GenerationTarget.GAZEBO,
        }:
            records.extend(generate_cad_artifacts(project, stager, adapter=cad_adapter))
            generator_versions["hephaestus.cadquery_backend"] = CAD_GENERATOR_VERSION
        if selected in {GenerationTarget.ALL, GenerationTarget.ROS2, GenerationTarget.GAZEBO}:
            records.extend(generate_ros_gazebo_artifacts(project, stager))
            generator_versions["hephaestus.ros_gazebo_backend"] = ROS_GAZEBO_GENERATOR_VERSION
        manifest = stager.publish(records, generator_versions=generator_versions)
    except Exception:
        stager.cleanup()
        raise
    return GenerationResult(selected, project_file, False, validation, plan, manifest)


def inspect_artifact_state(project_path: Path) -> dict[str, Any]:
    from hephaestus.artifacts import inspect_artifacts

    return inspect_artifacts(project_path)


def require_fresh_artifacts(project_path: Path) -> ArtifactManifest:
    manifest, status = load_manifest(project_path)
    if manifest is None:
        raise GenerationError("generated artifacts are missing; run hephaestus generate first")
    if status != "fresh":
        raise GenerationError("generated artifacts are stale; rerun hephaestus generate")
    project = load_project(project_path)
    if manifest.project_fingerprint != fingerprint_project(project):
        raise GenerationError("artifact manifest fingerprint does not match project")
    return manifest


def _combined_validation(project) -> ValidationReport:
    base = validate_project(project)
    findings: list[ValidationFinding] = list(base.findings)
    has_mobile = False
    has_concept = False
    try:
        mobile_robot_spec(project)
    except ValueError:
        pass
    else:
        has_mobile = True
        findings.extend(validate_mobile_robot_project(project))
    try:
        concept_robot_spec(project)
    except ConceptError:
        pass
    else:
        has_concept = True
        findings.extend(validate_concept_project(project))
    if not has_mobile and not has_concept:
        findings.append(
            ValidationFinding(
                "generation.robot_design.missing",
                ValidationSeverity.ERROR,
                "project does not contain a supported mobile or concept robot design",
                f"extensions.{DESIGN_EXTENSION_KEY}|{CONCEPT_EXTENSION_KEY}",
            )
        )
    return ValidationReport(tuple(findings))


def _design_generator_versions(project) -> dict[str, str]:
    try:
        concept_robot_spec(project)
    except ConceptError:
        return {"hephaestus.mobile_robot": MOBILE_ROBOT_GENERATOR_VERSION}
    return {"hephaestus.concept_robot": CONCEPT_GENERATOR_VERSION}


def _plan_for(target: GenerationTarget) -> tuple[str, ...]:
    if target is GenerationTarget.CAD:
        return (
            "validate canonical project",
            "generate CadQuery source",
            "export STEP/STL",
            "publish manifest",
        )
    if target is GenerationTarget.ROS2:
        return (
            "validate canonical project",
            "generate CAD mesh dependencies",
            "generate ROS 2 workspace",
            "validate URDF/Xacro XML templates",
            "publish manifest",
        )
    if target is GenerationTarget.GAZEBO:
        return (
            "validate canonical project",
            "generate CAD mesh dependencies",
            "generate Gazebo model and warehouse world",
            "validate SDF XML",
            "publish manifest",
        )
    return (
        "validate canonical project",
        "generate CadQuery CAD artifacts",
        "generate ROS 2 Jazzy workspace",
        "generate Gazebo Harmonic model/world",
        "publish artifact manifest",
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
