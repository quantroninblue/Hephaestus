"""Environment diagnostics for the Hephaestus CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hephaestus.capabilities import (
    CapabilityContext,
    CapabilityInspectionError,
    CapabilityReport,
    CapabilityResult,
    CapabilityStatus,
    collect_capabilities,
    normalize_capability_id,
)


class DiagnosticStatus(str):
    """Machine-readable status for a diagnostic check."""

    READY = "ready"
    MISSING = "missing"
    NOT_CONFIGURED = "not configured"
    MISMATCH = "mismatch"


class DiagnosticCategory(str):
    CORE = "core runtime"
    ROS2 = "ROS 2 tooling"
    GAZEBO = "Gazebo tooling"
    ROS_GAZEBO = "ROS/Gazebo integration"
    CAD = "CAD dependency availability"
    AI = "AI capability"


class DiagnosticItem:
    """One diagnostic check result."""

    def __init__(
        self,
        key: str,
        label: str,
        status: str,
        detail: str,
        category: str = DiagnosticCategory.CORE,
        required_for_strict: bool = True,
        capability_id: str | None = None,
        required_by_user: bool = False,
    ) -> None:
        self.key = key
        self.label = label
        self.status = status
        self.detail = detail
        self.category = category
        self.required_for_strict = required_for_strict
        self.capability_id = capability_id or key
        self.required_by_user = required_by_user

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "key": self.key,
            "label": self.label,
            "category": str(self.category),
            "status": str(self.status),
            "detail": self.detail,
            "required_for_strict": self.required_for_strict,
            "capability_id": self.capability_id,
            "required_by_user": self.required_by_user,
        }


class DiagnosticReport:
    """Collected diagnostics and readiness information."""

    def __init__(
        self,
        checks: tuple[DiagnosticItem, ...],
        *,
        capabilities: CapabilityReport | None = None,
        required_capabilities: tuple[str, ...] = (),
    ) -> None:
        self.checks = checks
        self.capabilities = capabilities
        self.required_capabilities = required_capabilities

    @property
    def strict_failures(self) -> tuple[DiagnosticItem, ...]:
        return tuple(
            check
            for check in self.checks
            if check.required_for_strict and check.status != DiagnosticStatus.READY
        )

    @property
    def requirement_failures(self) -> tuple[DiagnosticItem, ...]:
        return tuple(
            check
            for check in self.checks
            if check.required_by_user and check.status != DiagnosticStatus.READY
        )

    @property
    def is_strict_ready(self) -> bool:
        return not self.strict_failures

    @property
    def is_required_ready(self) -> bool:
        return not self.requirement_failures

    def to_dict(self) -> dict[str, Any]:
        status = (
            DiagnosticStatus.READY
            if self.is_strict_ready and self.is_required_ready
            else DiagnosticStatus.MISSING
        )
        data: dict[str, Any] = {
            "schema": "hephaestus.doctor.v2",
            "status": status,
            "strict_semantics": (
                "strict mode requires implemented core runtime checks; "
                "--require adds explicit capability requirements"
            ),
            "required_capabilities": list(self.required_capabilities),
            "requirement_failures": [check.capability_id for check in self.requirement_failures],
            "checks": [check.to_dict() for check in self.checks],
        }
        if self.capabilities is not None:
            data["capabilities"] = self.capabilities.to_dict()["capabilities"]
        return data


PythonVersion = tuple[int, int, int]


def collect_diagnostics(
    *,
    env: dict[str, str] | None = None,
    project_root: Path | None = None,
    sys_prefix: str | None = None,
    python_version: PythonVersion | None = None,
    system: str | None = None,
    machine: str | None = None,
    os_release_provider=None,
    which=None,
    find_spec=None,
    package_version=None,
    run=None,
    list_dir_names=None,
    required_capabilities: tuple[str, ...] = (),
) -> DiagnosticReport:
    """Collect diagnostics through the centralized capability registry."""

    context_kwargs: dict[str, Any] = {}
    if env is not None:
        context_kwargs["env"] = env
    if project_root is not None:
        context_kwargs["project_root"] = project_root
    if sys_prefix is not None:
        context_kwargs["sys_prefix"] = sys_prefix
    if python_version is not None:
        context_kwargs["python_version"] = python_version
    if system is not None:
        context_kwargs["system"] = system
    if machine is not None:
        context_kwargs["machine"] = machine
    if os_release_provider is not None:
        context_kwargs["os_release_provider"] = os_release_provider
    if which is not None:
        context_kwargs["which"] = which
    if find_spec is not None:
        context_kwargs["find_spec"] = find_spec
    if package_version is not None:
        context_kwargs["package_version"] = package_version
    if run is not None:
        context_kwargs["run"] = run
    if list_dir_names is not None:
        context_kwargs["list_dir_names"] = list_dir_names

    capabilities = collect_capabilities(CapabilityContext(**context_kwargs))
    required = tuple(normalize_capability_id(item) for item in required_capabilities)
    known = capabilities.by_id()
    for capability_id in required:
        if capability_id not in known:
            raise CapabilityInspectionError(f"unknown capability: {capability_id}")
    checks = tuple(
        _diagnostic_from_capability(item, item.capability_id in required)
        for item in capabilities.capabilities
    )
    return DiagnosticReport(checks, capabilities=capabilities, required_capabilities=required)


def dumps_diagnostics(report: DiagnosticReport) -> str:
    return json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True)


def _diagnostic_from_capability(
    capability: CapabilityResult, required_by_user: bool
) -> DiagnosticItem:
    return DiagnosticItem(
        key=capability.capability_id,
        label=capability.display_name,
        status=_diagnostic_status(capability.status),
        detail=_diagnostic_detail(capability),
        category=_diagnostic_category(capability),
        required_for_strict=capability.required_for_core,
        capability_id=capability.capability_id,
        required_by_user=required_by_user,
    )


def _diagnostic_status(status: CapabilityStatus) -> str:
    if status is CapabilityStatus.ACTIVE:
        return DiagnosticStatus.READY
    if status is CapabilityStatus.MISSING:
        return DiagnosticStatus.MISSING
    if status is CapabilityStatus.INSTALLED_INACTIVE:
        return DiagnosticStatus.NOT_CONFIGURED
    return DiagnosticStatus.MISMATCH


def _diagnostic_category(capability: CapabilityResult) -> str:
    categories = {
        "core": DiagnosticCategory.CORE,
        "ai": DiagnosticCategory.AI,
        "robotics_runtime": DiagnosticCategory.ROS2,
        "simulation": DiagnosticCategory.GAZEBO,
        "ros_gazebo_integration": DiagnosticCategory.ROS_GAZEBO,
        "cad": DiagnosticCategory.CAD,
    }
    return categories[str(capability.category)]


def _diagnostic_detail(capability: CapabilityResult) -> str:
    pieces: list[str] = []
    if capability.detected_version:
        pieces.append(str(capability.detected_version))
    if capability.compatibility:
        pieces.append(f"compatibility={capability.compatibility}")
    pieces.extend(capability.evidence[:3])
    return "; ".join(pieces) if pieces else capability.status
