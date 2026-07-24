"""Application-owned backend capability detection."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from hephaestus.ai.config import DEFAULT_MODEL, DEFAULT_PROVIDER, resolve_config
from hephaestus.ai.types import ProviderConfigurationError

SUPPORTED_OS_ID = "ubuntu"
SUPPORTED_OS_VERSION = "24.04"
SUPPORTED_ARCHITECTURE = "x86_64"
SUPPORTED_ROS_DISTRO = "jazzy"
SUPPORTED_GAZEBO_RELEASE = "Harmonic"
SUPPORTED_GAZEBO_MAJOR = 8
GAZEBO_RELEASE_BY_MAJOR: dict[int, str] = {
    6: "Fortress",
    7: "Garden",
    8: "Harmonic",
    9: "Ionic",
    10: "Jetty",
}
GAZEBO_VERSION_PATTERN = re.compile(
    r"(?<![\d.])(?P<major>0|[1-9]\d*)\.(?P<minor>\d+)\.(?P<patch>\d+)(?![\d.])"
)


class CapabilityStatus(StrEnum):
    ACTIVE = "active"
    INSTALLED_INACTIVE = "installed_inactive"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CompatibilityStatus(StrEnum):
    SUPPORTED = "supported"
    INCOMPATIBLE = "incompatible"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"


class CapabilityCategory(StrEnum):
    CORE = "core"
    AI = "ai"
    ROBOTICS_RUNTIME = "robotics_runtime"
    SIMULATION = "simulation"
    ROS_GAZEBO_INTEGRATION = "ros_gazebo_integration"
    CAD = "cad"


CAPABILITY_ORDER: tuple[str, ...] = (
    "core",
    "ai",
    "ros2",
    "gazebo",
    "ros_gz_sim",
    "ros_gz_bridge",
    "gz_ros2_control",
    "cad",
)


class CapabilityInspectionError(RuntimeError):
    """Raised for invalid capability requests, not detector failures."""


@dataclass(frozen=True, slots=True)
class SafeCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class GazeboVersionInfo:
    version: str | None
    major: int | None
    release: str | None


RunCommand = Callable[[Sequence[str], float], SafeCommandResult]
Which = Callable[[str], str | None]
FindSpec = Callable[[str], object | None]
PackageVersion = Callable[[str], str | None]
OsReleaseProvider = Callable[[], Mapping[str, str]]
ListDirNames = Callable[[Path], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    sys_prefix: str = field(default_factory=lambda: sys.prefix)
    sys_base_prefix: str = field(default_factory=lambda: sys.base_prefix)
    python_version: tuple[int, int, int] = field(default_factory=lambda: sys.version_info[:3])
    system: str = field(default_factory=platform.system)
    machine: str = field(default_factory=platform.machine)
    os_release_provider: OsReleaseProvider = field(default_factory=lambda: _freedesktop_os_release)
    which: Which = field(default_factory=lambda: shutil.which)
    find_spec: FindSpec = field(default_factory=lambda: importlib.util.find_spec)
    package_version: PackageVersion = field(default_factory=lambda: _package_version)
    run: RunCommand = field(default_factory=lambda: run_fixed_argv)
    list_dir_names: ListDirNames = field(default_factory=lambda: _list_dir_names)


@dataclass(frozen=True, slots=True)
class CapabilityResult:
    capability_id: str
    display_name: str
    category: CapabilityCategory
    status: CapabilityStatus
    compatibility: CompatibilityStatus
    required_for_core: bool = False
    detected_version: str | None = None
    expected: str | None = None
    evidence: tuple[str, ...] = ()
    remediation_id: str | None = None
    guidance: str | None = None

    @property
    def is_ready_for_requirement(self) -> bool:
        return self.status is CapabilityStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.capability_id,
            "display_name": self.display_name,
            "category": self.category.value,
            "status": self.status.value,
            "compatibility": self.compatibility.value,
            "required_for_core": self.required_for_core,
            "detected_version": self.detected_version,
            "expected": self.expected,
            "evidence": list(self.evidence),
            "remediation_id": self.remediation_id,
        }
        if self.guidance:
            data["guidance"] = self.guidance
        return data


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    capabilities: tuple[CapabilityResult, ...]

    def by_id(self) -> dict[str, CapabilityResult]:
        return {item.capability_id: item for item in self.capabilities}

    def require(self, capability_ids: Iterable[str]) -> tuple[CapabilityResult, ...]:
        indexed = self.by_id()
        failures: list[CapabilityResult] = []
        for capability_id in capability_ids:
            canonical = normalize_capability_id(capability_id)
            capability = indexed.get(canonical)
            if capability is None:
                raise CapabilityInspectionError(f"unknown capability: {capability_id}")
            if not capability.is_ready_for_requirement:
                failures.append(capability)
        return tuple(failures)

    @property
    def core_failures(self) -> tuple[CapabilityResult, ...]:
        return tuple(
            item
            for item in self.capabilities
            if item.required_for_core and not item.is_ready_for_requirement
        )

    @property
    def core_ready(self) -> bool:
        return not self.core_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hephaestus.capabilities.v1",
            "core_status": "ready" if self.core_ready else "not_ready",
            "capabilities": [item.to_dict() for item in self.capabilities],
        }


class CapabilityDetector:
    capability_id: str

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    detectors: tuple[CapabilityDetector, ...]

    def inspect(self, context: CapabilityContext | None = None) -> CapabilityReport:
        actual_context = context if context is not None else CapabilityContext()
        results = [_safe_inspect(detector, actual_context) for detector in self.detectors]
        order = {capability_id: index for index, capability_id in enumerate(CAPABILITY_ORDER)}
        return CapabilityReport(
            tuple(sorted(results, key=lambda item: order.get(item.capability_id, 10_000)))
        )


def default_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry(
        detectors=(
            CoreDetector(),
            AIDetector(),
            ROS2Detector(),
            GazeboDetector(),
            RosPackageDetector(
                "ros_gz_sim",
                "ros_gz_sim",
                "ROS/Gazebo launch integration",
                "remediation.ros_gz_sim.jazzy_harmonic",
                "ROS package that launches and integrates Gazebo simulations from ROS 2.",
            ),
            RosPackageDetector(
                "ros_gz_bridge",
                "ros_gz_bridge",
                "ROS/Gazebo communication bridge",
                "remediation.ros_gz_bridge.jazzy_harmonic",
                "ROS package that bridges topics and messages between ROS 2 and Gazebo.",
            ),
            RosPackageDetector(
                "gz_ros2_control",
                "gz_ros2_control",
                "Gazebo ros2_control integration",
                "remediation.gz_ros2_control.jazzy_harmonic",
                "ROS package that connects Gazebo simulated joints to ros2_control.",
            ),
            CADDetector(),
        )
    )


def collect_capabilities(context: CapabilityContext | None = None) -> CapabilityReport:
    return default_capability_registry().inspect(context)


def normalize_capability_id(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "ros": "ros2",
        "ros_2": "ros2",
        "gz": "gazebo",
        "gazebo_harmonic": "gazebo",
        "cadquery": "cad",
        "groq": "ai",
    }
    return aliases.get(normalized, normalized)


def stable_capability_ids() -> tuple[str, ...]:
    return CAPABILITY_ORDER


class CoreDetector(CapabilityDetector):
    capability_id = "core"

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        os_release = context.os_release_provider()
        distro_id = os_release.get("ID", "").lower()
        version_id = os_release.get("VERSION_ID", "")
        platform_ok = (
            context.system == "Linux"
            and distro_id == SUPPORTED_OS_ID
            and version_id == SUPPORTED_OS_VERSION
            and context.machine == SUPPORTED_ARCHITECTURE
        )
        python_ok = context.python_version[:2] == (3, 12)
        isolated_runtime_ok = _is_isolated_python_runtime(context)
        model_ok = _model_runtime_available()
        status = (
            CapabilityStatus.ACTIVE
            if all((platform_ok, python_ok, isolated_runtime_ok, model_ok))
            else CapabilityStatus.INCOMPATIBLE
        )
        evidence = (
            f"os={os_release.get('PRETTY_NAME') or context.system or 'unknown'}",
            f"architecture={context.machine or 'unknown'}",
            f"python={'.'.join(str(part) for part in context.python_version)}",
            "runtime_environment=isolated"
            if isolated_runtime_ok
            else "runtime_environment=not isolated",
            "canonical_model=available" if model_ok else "canonical_model=unavailable",
        )
        return CapabilityResult(
            "core",
            "Hephaestus core runtime",
            CapabilityCategory.CORE,
            status,
            CompatibilityStatus.SUPPORTED
            if status is CapabilityStatus.ACTIVE
            else CompatibilityStatus.INCOMPATIBLE,
            required_for_core=True,
            detected_version="Python " + ".".join(str(part) for part in context.python_version),
            expected="Ubuntu 24.04, x86_64, Python 3.12, isolated Python environment",
            evidence=evidence,
            remediation_id=None
            if status is CapabilityStatus.ACTIVE
            else "remediation.core.environment",
        )


class AIDetector(CapabilityDetector):
    capability_id = "ai"

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        sdk_available = context.find_spec("groq") is not None
        try:
            config = resolve_config(env=context.env)
            provider = config.provider
            model = config.model
            credential_present = config.credential_present
            config_valid = True
        except ProviderConfigurationError:
            provider = DEFAULT_PROVIDER
            model = DEFAULT_MODEL
            credential_present = False
            config_valid = False
        if not config_valid:
            status = CapabilityStatus.INCOMPATIBLE
            compatibility = CompatibilityStatus.INCOMPATIBLE
        elif not sdk_available:
            status = CapabilityStatus.MISSING
            compatibility = CompatibilityStatus.UNVERIFIED
        elif not credential_present:
            status = CapabilityStatus.INSTALLED_INACTIVE
            compatibility = CompatibilityStatus.SUPPORTED
        else:
            status = CapabilityStatus.ACTIVE
            compatibility = CompatibilityStatus.SUPPORTED
        return CapabilityResult(
            "ai",
            "Groq AI provider",
            CapabilityCategory.AI,
            status,
            compatibility,
            detected_version=context.package_version("groq") if sdk_available else None,
            expected=f"provider={DEFAULT_PROVIDER}; model={DEFAULT_MODEL}",
            evidence=(
                "groq_sdk=available" if sdk_available else "groq_sdk=missing",
                f"provider={provider}",
                f"model={model}",
                f"credential_present={str(credential_present).lower()}",
            ),
            remediation_id="remediation.ai.groq" if status is not CapabilityStatus.ACTIVE else None,
            guidance="Provider credentials are user-supplied; Hephaestus cannot create an API key.",
        )


class ROS2Detector(CapabilityDetector):
    capability_id = "ros2"

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        ros2_path = context.which("ros2")
        colcon_path = context.which("colcon")
        xacro_path = context.which("xacro")
        active_distro = context.env.get("ROS_DISTRO", "").strip()
        installed_distros = tuple(sorted(context.list_dir_names(Path("/opt/ros"))))
        if active_distro == SUPPORTED_ROS_DISTRO and ros2_path:
            status = CapabilityStatus.ACTIVE
            compatibility = CompatibilityStatus.SUPPORTED
        elif active_distro and active_distro != SUPPORTED_ROS_DISTRO:
            status = CapabilityStatus.INCOMPATIBLE
            compatibility = CompatibilityStatus.INCOMPATIBLE
        elif SUPPORTED_ROS_DISTRO in installed_distros or ros2_path:
            status = CapabilityStatus.INSTALLED_INACTIVE
            compatibility = (
                CompatibilityStatus.SUPPORTED
                if SUPPORTED_ROS_DISTRO in installed_distros
                else CompatibilityStatus.UNVERIFIED
            )
        else:
            status = CapabilityStatus.MISSING
            compatibility = CompatibilityStatus.UNVERIFIED
        evidence = (
            f"ros2={'found' if ros2_path else 'missing'}",
            f"colcon={'found' if colcon_path else 'missing'}",
            f"xacro={'found' if xacro_path else 'missing'}",
            f"active_ros_distro={active_distro or 'not set'}",
            "installed_distros="
            + (",".join(installed_distros) if installed_distros else "none detected"),
        )
        return CapabilityResult(
            "ros2",
            "ROS 2 Jazzy runtime",
            CapabilityCategory.ROBOTICS_RUNTIME,
            status,
            compatibility,
            detected_version=active_distro or None,
            expected="ROS_DISTRO=jazzy with ros2, colcon, and xacro available",
            evidence=evidence,
            remediation_id="remediation.ros2.jazzy"
            if status is not CapabilityStatus.ACTIVE
            else None,
        )


class GazeboDetector(CapabilityDetector):
    capability_id = "gazebo"

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        gz_path = context.which("gz")
        if not gz_path:
            return CapabilityResult(
                "gazebo",
                "Gazebo Harmonic runtime",
                CapabilityCategory.SIMULATION,
                CapabilityStatus.MISSING,
                CompatibilityStatus.UNVERIFIED,
                expected="Gazebo Harmonic using the modern gz command",
                evidence=("gz=missing",),
                remediation_id="remediation.gazebo.harmonic",
            )

        selected_command = context.run(("gz", "sim", "--version"), 3.0)
        selected_info = _parse_gazebo_version_info(
            selected_command.stdout + "\n" + selected_command.stderr
        )
        installed_command = context.run(("gz", "sim", "--versions"), 3.0)
        installed_infos: tuple[GazeboVersionInfo, ...] = ()
        if not installed_command.timed_out and installed_command.returncode == 0:
            installed_infos = _parse_gazebo_versions(
                installed_command.stdout + "\n" + installed_command.stderr
            )
        supported_installed = _supported_installed_gazebo_version(installed_infos)

        if selected_command.timed_out or installed_command.timed_out:
            status = CapabilityStatus.TIMEOUT
            compatibility = CompatibilityStatus.UNVERIFIED
            version = None
            evidence = _gazebo_evidence(selected_info, installed_infos) + (
                "version_check=timed out",
            )
        elif (
            selected_info.major == SUPPORTED_GAZEBO_MAJOR
            or selected_info.release == SUPPORTED_GAZEBO_RELEASE
        ):
            status = CapabilityStatus.ACTIVE
            compatibility = CompatibilityStatus.SUPPORTED
            version = selected_info.version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        elif supported_installed is not None:
            status = CapabilityStatus.INSTALLED_INACTIVE
            compatibility = CompatibilityStatus.SUPPORTED
            version = selected_info.version or supported_installed.version
            evidence = _gazebo_evidence(selected_info, installed_infos) + (
                f"supported_installed={supported_installed.version}",
            )
        elif selected_info.major is not None and selected_info.release is None:
            status = CapabilityStatus.UNKNOWN
            compatibility = CompatibilityStatus.UNVERIFIED
            version = selected_info.version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        elif selected_info.major is not None:
            status = CapabilityStatus.INCOMPATIBLE
            compatibility = CompatibilityStatus.INCOMPATIBLE
            version = selected_info.version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        elif selected_command.returncode != 0 and installed_command.returncode != 0:
            status = CapabilityStatus.DEGRADED
            compatibility = CompatibilityStatus.UNVERIFIED
            version = None
            selected_probe = _bounded_text(selected_command.stderr or selected_command.stdout)
            installed_probe = _bounded_text(installed_command.stderr or installed_command.stdout)
            evidence = _gazebo_evidence(selected_info, installed_infos) + (
                "version_check=failed",
                f"selected_probe={selected_probe}",
                f"installed_probe={installed_probe}",
            )
        elif selected_info.version is None and _only_known_incompatible_gazebo_versions(
            installed_infos
        ):
            status = CapabilityStatus.INCOMPATIBLE
            compatibility = CompatibilityStatus.INCOMPATIBLE
            version = installed_infos[0].version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        elif selected_info.release is None:
            status = CapabilityStatus.UNKNOWN
            compatibility = CompatibilityStatus.UNVERIFIED
            version = selected_info.version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        else:
            status = CapabilityStatus.INCOMPATIBLE
            compatibility = CompatibilityStatus.INCOMPATIBLE
            version = selected_info.version
            evidence = _gazebo_evidence(selected_info, installed_infos)
        return CapabilityResult(
            "gazebo",
            "Gazebo Harmonic runtime",
            CapabilityCategory.SIMULATION,
            status,
            compatibility,
            detected_version=version,
            expected="Gazebo Harmonic / major version 8",
            evidence=evidence,
            remediation_id="remediation.gazebo.harmonic"
            if status is not CapabilityStatus.ACTIVE
            else None,
        )


@dataclass(frozen=True, slots=True)
class RosPackageDetector(CapabilityDetector):
    capability_id: str
    package_name: str
    display_name: str
    remediation_id_value: str
    guidance_text: str

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        ros2_path = context.which("ros2")
        active_distro = context.env.get("ROS_DISTRO", "").strip()
        if not ros2_path:
            return _ros_package_result(
                self, CapabilityStatus.MISSING, CompatibilityStatus.UNVERIFIED, ("ros2=missing",)
            )
        if active_distro != SUPPORTED_ROS_DISTRO:
            return _ros_package_result(
                self,
                CapabilityStatus.INSTALLED_INACTIVE
                if not active_distro
                else CapabilityStatus.INCOMPATIBLE,
                CompatibilityStatus.UNVERIFIED
                if not active_distro
                else CompatibilityStatus.INCOMPATIBLE,
                (f"active_ros_distro={active_distro or 'not set'}", "package_check=not reliable"),
            )
        command = context.run(("ros2", "pkg", "prefix", self.package_name), 3.0)
        if command.timed_out:
            return _ros_package_result(
                self,
                CapabilityStatus.TIMEOUT,
                CompatibilityStatus.UNVERIFIED,
                ("package_check=timed out",),
            )
        if command.returncode == 0 and command.stdout.strip():
            return _ros_package_result(
                self,
                CapabilityStatus.ACTIVE,
                CompatibilityStatus.SUPPORTED,
                (f"package={self.package_name}", "prefix=found"),
                detected_version=active_distro,
            )
        output = _bounded_text(command.stderr or command.stdout or "")
        evidence = (f"package={self.package_name}", "prefix=missing")
        if output:
            evidence = (*evidence, f"message={output}")
        return _ros_package_result(
            self, CapabilityStatus.MISSING, CompatibilityStatus.UNVERIFIED, evidence
        )


class CADDetector(CapabilityDetector):
    capability_id = "cad"

    def inspect(self, context: CapabilityContext) -> CapabilityResult:
        available = context.find_spec("cadquery") is not None
        version = context.package_version("cadquery") if available else None
        return CapabilityResult(
            "cad",
            "CadQuery parametric CAD backend",
            CapabilityCategory.CAD,
            CapabilityStatus.ACTIVE if available else CapabilityStatus.MISSING,
            CompatibilityStatus.SUPPORTED if available else CompatibilityStatus.UNVERIFIED,
            detected_version=version,
            expected="CadQuery importable in an isolated Python environment",
            evidence=("cadquery=importable" if available else "cadquery=not importable",),
            remediation_id="remediation.cad.cadquery" if not available else None,
            guidance=(
                "CadQuery is optional for core operation and should be isolated from host Python."
            ),
        )


def run_fixed_argv(argv: Sequence[str], timeout_seconds: float) -> SafeCommandResult:
    if not argv:
        raise CapabilityInspectionError("argv must not be empty")
    try:
        completed = subprocess.run(
            tuple(argv),
            check=False,
            input=None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_inspection_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        return SafeCommandResult(
            tuple(argv),
            124,
            _bounded_text(exc.stdout or ""),
            _bounded_text(exc.stderr or ""),
            timed_out=True,
        )
    except OSError as exc:
        return SafeCommandResult(tuple(argv), 127, "", _bounded_text(str(exc)))
    return SafeCommandResult(
        tuple(argv),
        completed.returncode,
        _bounded_text(completed.stdout),
        _bounded_text(completed.stderr),
    )


def _safe_inspect(detector: CapabilityDetector, context: CapabilityContext) -> CapabilityResult:
    try:
        return detector.inspect(context)
    except Exception as exc:  # noqa: BLE001 - detector failures must be contained.
        return CapabilityResult(
            detector.capability_id,
            detector.capability_id,
            CapabilityCategory.CORE,
            CapabilityStatus.FAILED,
            CompatibilityStatus.UNVERIFIED,
            required_for_core=detector.capability_id == "core",
            evidence=(f"inspection_failed={type(exc).__name__}",),
            remediation_id=f"remediation.{detector.capability_id}.inspection_failed",
        )


def _ros_package_result(
    detector: RosPackageDetector,
    status: CapabilityStatus,
    compatibility: CompatibilityStatus,
    evidence: tuple[str, ...],
    *,
    detected_version: str | None = None,
) -> CapabilityResult:
    return CapabilityResult(
        detector.capability_id,
        detector.display_name,
        CapabilityCategory.ROS_GAZEBO_INTEGRATION,
        status,
        compatibility,
        detected_version=detected_version,
        expected=f"{detector.package_name} available in ROS 2 Jazzy environment",
        evidence=evidence,
        remediation_id=detector.remediation_id_value
        if status is not CapabilityStatus.ACTIVE
        else None,
        guidance=detector.guidance_text,
    )


def _parse_gazebo_version_info(output: str) -> GazeboVersionInfo:
    versions = _parse_gazebo_versions(output)
    if versions:
        return versions[0]
    release = _parse_gazebo_release_name(output)
    return GazeboVersionInfo(version=None, major=None, release=release)


def _parse_gazebo_versions(output: str) -> tuple[GazeboVersionInfo, ...]:
    versions: list[GazeboVersionInfo] = []
    seen: set[str] = set()
    for match in GAZEBO_VERSION_PATTERN.finditer(output):
        version = match.group(0)
        if version in seen:
            continue
        major = int(match.group("major"))
        versions.append(
            GazeboVersionInfo(
                version=version,
                major=major,
                release=GAZEBO_RELEASE_BY_MAJOR.get(major),
            )
        )
        seen.add(version)
    return tuple(versions)


def _parse_gazebo_release_name(output: str) -> str | None:
    lower = output.lower()
    for release in GAZEBO_RELEASE_BY_MAJOR.values():
        if release.lower() in lower:
            return release
    return None


def _supported_installed_gazebo_version(
    installed_infos: tuple[GazeboVersionInfo, ...],
) -> GazeboVersionInfo | None:
    for info in installed_infos:
        if info.major == SUPPORTED_GAZEBO_MAJOR:
            return info
    return None


def _only_known_incompatible_gazebo_versions(
    installed_infos: tuple[GazeboVersionInfo, ...],
) -> bool:
    return bool(installed_infos) and all(
        info.release is not None and info.major != SUPPORTED_GAZEBO_MAJOR
        for info in installed_infos
    )


def _gazebo_evidence(
    selected_info: GazeboVersionInfo, installed_infos: tuple[GazeboVersionInfo, ...]
) -> tuple[str, ...]:
    evidence = ["gz=found"]
    if selected_info.version:
        evidence.append(f"selected_version={selected_info.version}")
    release = selected_info.release or "unverified"
    evidence.append(f"release={release}")
    evidence.append(f"selected_release={release}")
    evidence.extend(_installed_gazebo_evidence(installed_infos))
    return tuple(evidence)


def _installed_gazebo_evidence(installed_infos: tuple[GazeboVersionInfo, ...]) -> tuple[str, ...]:
    if not installed_infos:
        return ()
    versions = ",".join(info.version or "unknown" for info in installed_infos)
    releases = ",".join(dict.fromkeys(info.release or "unverified" for info in installed_infos))
    return (f"installed_versions={versions}", f"installed_releases={releases}")


def _is_isolated_python_runtime(context: CapabilityContext) -> bool:
    return not _same_resolved_path(Path(context.sys_prefix), Path(context.sys_base_prefix))


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _freedesktop_os_release() -> Mapping[str, str]:
    try:
        return platform.freedesktop_os_release()
    except OSError:
        return {}


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _list_dir_names(path: Path) -> tuple[str, ...]:
    try:
        return tuple(sorted(child.name for child in path.iterdir() if child.is_dir()))
    except OSError:
        return ()


def _bounded_text(value: str | bytes | None, limit: int = 512) -> str:
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else value
    sanitized = " ".join(text.replace("\x00", "").split())
    return sanitized[:limit]


def _model_runtime_available() -> bool:
    try:
        __import__("hephaestus.model")
    except Exception:
        return False
    return True


def _inspection_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "ROS_DISTRO",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "GZ_CONFIG_PATH",
        "IGN_CONFIG_PATH",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}
