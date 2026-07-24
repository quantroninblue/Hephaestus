"""Declarative, non-executable setup planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hephaestus.capabilities import (
    CapabilityReport,
    CapabilityResult,
    CapabilityStatus,
    normalize_capability_id,
)

SETUP_PLAN_SCHEMA = "hephaestus.setup_plan.v1"


class InstallationMechanism(StrEnum):
    NONE = "none"
    APT = "apt"
    ISOLATED_PYTHON = "isolated_python"
    ENVIRONMENT_VARIABLE = "environment_variable"
    MANUAL = "manual"


class PrivilegeRequirement(StrEnum):
    NONE = "none"
    USER = "user"
    ADMINISTRATOR = "administrator"


@dataclass(frozen=True, slots=True)
class RemediationMetadata:
    remediation_id: str
    capability_id: str
    desired_state: str
    enables: str
    why_optional: str
    supported_platform: str
    target_version: str | None
    installation_mechanism: InstallationMechanism
    privilege_required: PrivilegeRequirement
    network_required: bool
    changes_package_repositories: bool
    expected_download: str | None
    expected_disk_impact: str | None
    reversible: str
    affected_paths: tuple[str, ...]
    official_docs: tuple[str, ...]
    verification_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "remediation_id": self.remediation_id,
            "capability_id": self.capability_id,
            "desired_state": self.desired_state,
            "enables": self.enables,
            "why_optional": self.why_optional,
            "supported_platform": self.supported_platform,
            "target_version": self.target_version,
            "installation_mechanism": self.installation_mechanism.value,
            "privilege_required": self.privilege_required.value,
            "network_required": self.network_required,
            "changes_package_repositories": self.changes_package_repositories,
            "expected_download": self.expected_download,
            "expected_disk_impact": self.expected_disk_impact,
            "reversible": self.reversible,
            "affected_paths": list(self.affected_paths),
            "official_docs": list(self.official_docs),
            "verification_checks": list(self.verification_checks),
        }


@dataclass(frozen=True, slots=True)
class SetupPlanItem:
    capability: CapabilityResult
    remediation: RemediationMetadata | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.to_dict(),
            "ready": self.capability.status is CapabilityStatus.ACTIVE,
            "remediation": self.remediation.to_dict() if self.remediation else None,
        }


@dataclass(frozen=True, slots=True)
class SetupPlan:
    schema: str
    required_capabilities: tuple[str, ...]
    items: tuple[SetupPlanItem, ...]
    executable: bool = False
    authorization_granted: bool = False
    summary: str = "Read-only setup plan. No system changes are performed."

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "summary": self.summary,
            "required_capabilities": list(self.required_capabilities),
            "executable": self.executable,
            "authorization_granted": self.authorization_granted,
            "items": [item.to_dict() for item in self.items],
        }


def build_setup_plan(
    report: CapabilityReport, *, required_capabilities: tuple[str, ...] = ()
) -> SetupPlan:
    required = tuple(normalize_capability_id(item) for item in required_capabilities)
    selected = set(required) if required else {item.capability_id for item in report.capabilities}
    indexed = report.by_id()
    unknown = sorted(selected.difference(indexed))
    if unknown:
        raise ValueError(f"unknown capability: {unknown[0]}")
    items = tuple(
        SetupPlanItem(capability, REMEDIATIONS.get(capability.remediation_id or ""))
        for capability in report.capabilities
        if capability.capability_id in selected
    )
    return SetupPlan(SETUP_PLAN_SCHEMA, required, items)


REMEDIATIONS: dict[str, RemediationMetadata] = {
    "remediation.core.environment": RemediationMetadata(
        remediation_id="remediation.core.environment",
        capability_id="core",
        desired_state=(
            "Ubuntu 24.04 on x86_64 with Python 3.12 and the project virtual environment active"
        ),
        enables="Hephaestus core CLI execution, project files, validation, and local diagnostics.",
        why_optional="This is required for supported development; other platforms are unverified.",
        supported_platform="Ubuntu 24.04 x86_64",
        target_version="Python 3.12",
        installation_mechanism=InstallationMechanism.MANUAL,
        privilege_required=PrivilegeRequirement.USER,
        network_required=False,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="Depends on the local Python and virtual-environment changes made manually.",
        affected_paths=("project .venv",),
        official_docs=("https://docs.python.org/3/library/venv.html",),
        verification_checks=("core",),
    ),
    "remediation.ai.groq": RemediationMetadata(
        remediation_id="remediation.ai.groq",
        capability_id="ai",
        desired_state="Groq SDK available and GROQ_API_KEY present in the runtime environment",
        enables="Project-aware AI discussion and deterministic tool requests through Groq.",
        why_optional="Core project inspection and validation work without hosted AI.",
        supported_platform="Any supported Hephaestus Python environment",
        target_version="openai/gpt-oss-120b through Groq",
        installation_mechanism=InstallationMechanism.ENVIRONMENT_VARIABLE,
        privilege_required=PrivilegeRequirement.USER,
        network_required=True,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="Remove the environment variable or user AI configuration.",
        affected_paths=("user shell environment", "user config directory"),
        official_docs=("https://console.groq.com/docs",),
        verification_checks=("ai",),
    ),
    "remediation.ros2.jazzy": RemediationMetadata(
        remediation_id="remediation.ros2.jazzy",
        capability_id="ros2",
        desired_state="ROS 2 Jazzy installed and sourced in the current shell",
        enables=(
            "ROS package inspection, workspace building, launch files, nodes, "
            "TF, controllers, and robotics communication checks."
        ),
        why_optional="Hephaestus Core and project validation work without ROS 2.",
        supported_platform="Ubuntu 24.04 x86_64",
        target_version="ROS 2 Jazzy",
        installation_mechanism=InstallationMechanism.APT,
        privilege_required=PrivilegeRequirement.ADMINISTRATOR,
        network_required=True,
        changes_package_repositories=True,
        expected_download=None,
        expected_disk_impact=None,
        reversible=(
            "APT removal is possible, but repository and dependency rollback "
            "may require manual cleanup."
        ),
        affected_paths=("/opt/ros/jazzy", "APT package database"),
        official_docs=("https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html",),
        verification_checks=("ros2",),
    ),
    "remediation.gazebo.harmonic": RemediationMetadata(
        remediation_id="remediation.gazebo.harmonic",
        capability_id="gazebo",
        desired_state="Gazebo Harmonic available through the modern gz command",
        enables=(
            "Simulation runtime inspection and later model/world execution "
            "when explicitly implemented."
        ),
        why_optional="Project modeling and validation work without Gazebo.",
        supported_platform="Ubuntu 24.04 x86_64 with ROS 2 Jazzy-compatible packages",
        target_version="Gazebo Harmonic",
        installation_mechanism=InstallationMechanism.APT,
        privilege_required=PrivilegeRequirement.ADMINISTRATOR,
        network_required=True,
        changes_package_repositories=True,
        expected_download=None,
        expected_disk_impact=None,
        reversible="APT removal is possible, but dependency rollback may require manual cleanup.",
        affected_paths=("APT package database", "Gazebo system package paths"),
        official_docs=("https://gazebosim.org/docs/harmonic/install_ubuntu/",),
        verification_checks=("gazebo",),
    ),
    "remediation.ros_gz_sim.jazzy_harmonic": RemediationMetadata(
        remediation_id="remediation.ros_gz_sim.jazzy_harmonic",
        capability_id="ros_gz_sim",
        desired_state="ros_gz_sim package available in a ROS 2 Jazzy environment",
        enables="Launching Gazebo simulations through ROS 2 when execution support is implemented.",
        why_optional="Inspection and project modeling work without ROS/Gazebo launch integration.",
        supported_platform="Ubuntu 24.04 x86_64, ROS 2 Jazzy, Gazebo Harmonic",
        target_version="Jazzy-compatible ros_gz_sim",
        installation_mechanism=InstallationMechanism.APT,
        privilege_required=PrivilegeRequirement.ADMINISTRATOR,
        network_required=True,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="APT removal is possible; dependency cleanup may be manual.",
        affected_paths=("/opt/ros/jazzy", "APT package database"),
        official_docs=("https://gazebosim.org/docs/harmonic/ros_installation/",),
        verification_checks=("ros_gz_sim",),
    ),
    "remediation.ros_gz_bridge.jazzy_harmonic": RemediationMetadata(
        remediation_id="remediation.ros_gz_bridge.jazzy_harmonic",
        capability_id="ros_gz_bridge",
        desired_state="ros_gz_bridge package available in a ROS 2 Jazzy environment",
        enables=(
            "Topic and message bridging between ROS 2 and Gazebo when "
            "execution support is implemented."
        ),
        why_optional="Project modeling and validation work without bridge execution.",
        supported_platform="Ubuntu 24.04 x86_64, ROS 2 Jazzy, Gazebo Harmonic",
        target_version="Jazzy-compatible ros_gz_bridge",
        installation_mechanism=InstallationMechanism.APT,
        privilege_required=PrivilegeRequirement.ADMINISTRATOR,
        network_required=True,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="APT removal is possible; dependency cleanup may be manual.",
        affected_paths=("/opt/ros/jazzy", "APT package database"),
        official_docs=("https://gazebosim.org/docs/harmonic/ros_installation/",),
        verification_checks=("ros_gz_bridge",),
    ),
    "remediation.gz_ros2_control.jazzy_harmonic": RemediationMetadata(
        remediation_id="remediation.gz_ros2_control.jazzy_harmonic",
        capability_id="gz_ros2_control",
        desired_state="gz_ros2_control package available in a ROS 2 Jazzy environment",
        enables=(
            "Simulated joint command/state integration with ros2_control "
            "when execution support is implemented."
        ),
        why_optional="Core project modeling does not require simulated joint control.",
        supported_platform="Ubuntu 24.04 x86_64, ROS 2 Jazzy, Gazebo Harmonic",
        target_version="Jazzy-compatible gz_ros2_control",
        installation_mechanism=InstallationMechanism.APT,
        privilege_required=PrivilegeRequirement.ADMINISTRATOR,
        network_required=True,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="APT removal is possible; dependency cleanup may be manual.",
        affected_paths=("/opt/ros/jazzy", "APT package database"),
        official_docs=("https://control.ros.org/",),
        verification_checks=("gz_ros2_control",),
    ),
    "remediation.cad.cadquery": RemediationMetadata(
        remediation_id="remediation.cad.cadquery",
        capability_id="cad",
        desired_state="CadQuery importable in an isolated Python or Conda-style environment",
        enables=(
            "Parametric CAD backend availability checks; CAD generation is "
            "not implemented in this milestone."
        ),
        why_optional="Project modeling and validation work without CAD dependencies.",
        supported_platform="Supported Hephaestus Python environment; isolation recommended",
        target_version="CadQuery",
        installation_mechanism=InstallationMechanism.ISOLATED_PYTHON,
        privilege_required=PrivilegeRequirement.USER,
        network_required=True,
        changes_package_repositories=False,
        expected_download=None,
        expected_disk_impact=None,
        reversible="Remove the isolated environment; host Python should remain untouched.",
        affected_paths=("isolated Python or Conda-style environment",),
        official_docs=("https://cadquery.readthedocs.io/",),
        verification_checks=("cad",),
    ),
}
