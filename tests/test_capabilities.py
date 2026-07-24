from __future__ import annotations

import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.capabilities import (
    CapabilityContext,
    CapabilityInspectionError,
    CapabilityRegistry,
    CapabilityStatus,
    CompatibilityStatus,
    SafeCommandResult,
    collect_capabilities,
    run_fixed_argv,
    stable_capability_ids,
)
from hephaestus.setup_plan import build_setup_plan

runner = CliRunner()


def _context(
    tmp_path: Path,
    *,
    env: dict[str, str] | None = None,
    sys_prefix: str | None = None,
    sys_base_prefix: str | None = None,
    which: dict[str, str | None] | None = None,
    specs: set[str] | None = None,
    versions: dict[str, str | None] | None = None,
    run_results: dict[tuple[str, ...], SafeCommandResult] | None = None,
    run_func=None,
    ros_dirs: tuple[str, ...] = (),
) -> CapabilityContext:
    environment = {"HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config")}
    if env:
        environment.update(env)
    found = which or {}
    modules = {"hephaestus.model", *(specs or set())}
    package_versions = versions or {}
    scripted = run_results or {}

    def run(argv, timeout):
        return scripted.get(tuple(argv), SafeCommandResult(tuple(argv), 1, "", "not found"))

    actual_run = run_func or run

    return CapabilityContext(
        env=environment,
        project_root=tmp_path,
        sys_prefix=sys_prefix or str(tmp_path / ".venv"),
        sys_base_prefix=sys_base_prefix or str(tmp_path / "base-python"),
        python_version=(3, 12, 3),
        system="Linux",
        machine="x86_64",
        os_release_provider=lambda: {
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "PRETTY_NAME": "Ubuntu 24.04 LTS",
        },
        which=lambda command: found.get(command),
        find_spec=lambda module_name: object() if module_name in modules else None,
        package_version=lambda name: package_versions.get(name),
        run=actual_run,
        list_dir_names=lambda path: ros_dirs if path == Path("/opt/ros") else (),
    )


def _by_id(report):
    return report.by_id()


def test_pipx_private_venv_core_runtime_is_ready_without_project_venv(
    monkeypatch, tmp_path: Path
) -> None:
    from hephaestus.doctor import collect_diagnostics

    external_project = tmp_path / "external_project"
    external_project.mkdir()
    monkeypatch.chdir(external_project)
    pipx_prefix = tmp_path / "pipx" / "venvs" / "hephaestus-ai"
    base_prefix = tmp_path / "python-base"
    context = _context(
        external_project,
        env={"HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config")},
        sys_prefix=str(pipx_prefix),
        sys_base_prefix=str(base_prefix),
    )

    assert "VIRTUAL_ENV" not in context.env
    assert not (external_project / ".venv").exists()

    report = collect_capabilities(context)
    core = report.by_id()["core"]
    capability_json = json.dumps(report.to_dict(), sort_keys=True)

    assert core.status is CapabilityStatus.ACTIVE
    assert core.compatibility is CompatibilityStatus.SUPPORTED
    assert report.to_dict()["core_status"] == "ready"
    assert "runtime_environment=isolated" in core.evidence
    assert "project_venv" not in capability_json
    assert "project .venv" not in capability_json
    assert str(tmp_path) not in capability_json
    assert "\x1b[" not in capability_json

    monkeypatch.setattr(cli, "collect_capabilities", lambda: report)
    capabilities_result = runner.invoke(cli.app, ["capabilities", "--json"])
    capabilities_payload = json.loads(capabilities_result.output)

    assert capabilities_result.exit_code == 0
    assert capabilities_payload["core_status"] == "ready"
    assert capabilities_payload["capabilities"][0]["status"] == "active"
    assert capabilities_payload["capabilities"][0]["compatibility"] == "supported"
    assert "\x1b[" not in capabilities_result.output
    assert str(tmp_path) not in capabilities_result.output

    def fake_collect_diagnostics(*, required_capabilities=()):
        return collect_diagnostics(
            env=context.env,
            project_root=context.project_root,
            sys_prefix=context.sys_prefix,
            sys_base_prefix=context.sys_base_prefix,
            python_version=context.python_version,
            system=context.system,
            machine=context.machine,
            os_release_provider=context.os_release_provider,
            which=context.which,
            find_spec=context.find_spec,
            package_version=context.package_version,
            run=context.run,
            list_dir_names=context.list_dir_names,
            required_capabilities=tuple(required_capabilities),
        )

    monkeypatch.setattr(cli, "collect_diagnostics", fake_collect_diagnostics)
    doctor_result = runner.invoke(cli.app, ["doctor", "--json"])
    doctor_payload = json.loads(doctor_result.output)

    assert doctor_result.exit_code == 0
    assert doctor_payload["status"] == "ready"
    assert doctor_payload["checks"][0]["status"] == "ready"
    assert doctor_payload["capabilities"][0]["status"] == "active"
    assert doctor_payload["capabilities"][0]["compatibility"] == "supported"
    assert "\x1b[" not in doctor_result.output
    assert str(tmp_path) not in doctor_result.output


def test_non_isolated_system_interpreter_is_not_reported_as_supported(
    tmp_path: Path,
) -> None:
    interpreter_prefix = tmp_path / "system-python"
    report = collect_capabilities(
        _context(
            tmp_path,
            sys_prefix=str(interpreter_prefix),
            sys_base_prefix=str(interpreter_prefix),
        )
    )
    core = report.by_id()["core"]
    payload = json.dumps(report.to_dict(), sort_keys=True)

    assert core.status is CapabilityStatus.INCOMPATIBLE
    assert core.compatibility is CompatibilityStatus.INCOMPATIBLE
    assert report.core_ready is False
    assert report.to_dict()["core_status"] == "not_ready"
    assert "runtime_environment=not isolated" in core.evidence
    assert "project_venv" not in payload


def test_capabilities_active_supported_stack_is_deterministic_and_secret_free(
    tmp_path: Path,
) -> None:
    groq_secret = "gsk_test_secret_value"
    report = collect_capabilities(
        _context(
            tmp_path,
            env={"ROS_DISTRO": "jazzy", "GROQ_API_KEY": groq_secret},
            which={
                "ros2": "/usr/bin/ros2",
                "colcon": "/usr/bin/colcon",
                "xacro": "/usr/bin/xacro",
                "gz": "/usr/bin/gz",
            },
            specs={"groq", "cadquery"},
            versions={"groq": "1.0", "cadquery": "2.5"},
            run_results={
                ("gz", "sim", "--version"): SafeCommandResult(
                    ("gz", "sim", "--version"), 0, "Gazebo Sim, version 8.9.0"
                ),
                ("gz", "sim", "--versions"): SafeCommandResult(
                    ("gz", "sim", "--versions"), 0, "8.9.0"
                ),
                ("ros2", "pkg", "prefix", "ros_gz_sim"): SafeCommandResult(
                    ("ros2", "pkg", "prefix", "ros_gz_sim"), 0, "/opt/ros/jazzy"
                ),
                ("ros2", "pkg", "prefix", "ros_gz_bridge"): SafeCommandResult(
                    ("ros2", "pkg", "prefix", "ros_gz_bridge"), 0, "/opt/ros/jazzy"
                ),
                ("ros2", "pkg", "prefix", "gz_ros2_control"): SafeCommandResult(
                    ("ros2", "pkg", "prefix", "gz_ros2_control"), 0, "/opt/ros/jazzy"
                ),
            },
            ros_dirs=("jazzy",),
        )
    )

    payload = report.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert [item["id"] for item in payload["capabilities"]] == list(stable_capability_ids())
    assert all(item.status is CapabilityStatus.ACTIVE for item in report.capabilities)
    assert groq_secret not in encoded
    assert "credential_present=true" in encoded


def test_ros_installed_but_unsourced_wrong_and_unknown_states(tmp_path: Path) -> None:
    unsourced = _by_id(
        collect_capabilities(_context(tmp_path, ros_dirs=("jazzy",), which={"ros2": None}))
    )["ros2"]
    wrong = _by_id(
        collect_capabilities(
            _context(tmp_path, env={"ROS_DISTRO": "humble"}, which={"ros2": "/usr/bin/ros2"})
        )
    )["ros2"]
    unknown = _by_id(collect_capabilities(_context(tmp_path, which={"ros2": "/usr/bin/ros2"})))[
        "ros2"
    ]

    assert unsourced.status is CapabilityStatus.INSTALLED_INACTIVE
    assert wrong.status is CapabilityStatus.INCOMPATIBLE
    assert unknown.status is CapabilityStatus.INSTALLED_INACTIVE


def test_gazebo_harmonic_semantic_version_is_active_and_supported(tmp_path: Path) -> None:
    report = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"),
                        0,
                        "Gazebo Sim, version 8.11.0\n"
                        "Copyright (C) 2018 Open Source Robotics Foundation.\n"
                        "Released under the Apache 2.0 License.\n",
                    ),
                    ("gz", "sim", "--versions"): SafeCommandResult(
                        ("gz", "sim", "--versions"), 0, "8.11.0\n"
                    ),
                },
            )
        )
    )

    gazebo = report["gazebo"]

    assert gazebo.status is CapabilityStatus.ACTIVE
    assert gazebo.compatibility.value == "supported"
    assert gazebo.detected_version == "8.11.0"
    assert "release=Harmonic" in gazebo.evidence
    assert "installed_versions=8.11.0" in gazebo.evidence



def test_gazebo_harmonic_release_name_without_semver_is_supported(tmp_path: Path) -> None:
    report = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 0, "Gazebo Harmonic\n"
                    ),
                    ("gz", "sim", "--versions"): SafeCommandResult(
                        ("gz", "sim", "--versions"), 0, ""
                    ),
                },
            )
        )
    )

    gazebo = report["gazebo"]

    assert gazebo.status is CapabilityStatus.ACTIVE
    assert gazebo.compatibility.value == "supported"
    assert gazebo.detected_version is None
    assert "release=Harmonic" in gazebo.evidence

def test_gazebo_supported_installed_but_wrong_default_is_inactive(tmp_path: Path) -> None:
    report = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 0, "Gazebo Sim, version 7.9.0\n"
                    ),
                    ("gz", "sim", "--versions"): SafeCommandResult(
                        ("gz", "sim", "--versions"), 0, "7.9.0\n8.11.0\n"
                    ),
                },
            )
        )
    )

    gazebo = report["gazebo"]

    assert gazebo.status is CapabilityStatus.INSTALLED_INACTIVE
    assert gazebo.compatibility.value == "supported"
    assert gazebo.detected_version == "7.9.0"
    assert "release=Garden" in gazebo.evidence
    assert "supported_installed=8.11.0" in gazebo.evidence


def test_gazebo_unknown_future_major_keeps_detected_version_unverified(tmp_path: Path) -> None:
    report = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 0, "Gazebo Sim, version 11.0.1\n"
                    ),
                    ("gz", "sim", "--versions"): SafeCommandResult(
                        ("gz", "sim", "--versions"), 0, "11.0.1\n"
                    ),
                },
            )
        )
    )

    gazebo = report["gazebo"]

    assert gazebo.status is CapabilityStatus.UNKNOWN
    assert gazebo.compatibility.value == "unverified"
    assert gazebo.detected_version == "11.0.1"
    assert "release=unverified" in gazebo.evidence


def test_gazebo_missing_unknown_incompatible_degraded_and_timeout(tmp_path: Path) -> None:
    missing = _by_id(collect_capabilities(_context(tmp_path)))["gazebo"]
    unknown = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 0, "???"
                    )
                },
            )
        )
    )["gazebo"]
    incompatible = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 0, "Gazebo Garden"
                    )
                },
            )
        )
    )["gazebo"]
    timeout = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 124, timed_out=True
                    )
                },
            )
        )
    )["gazebo"]
    degraded = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                which={"gz": "/usr/bin/gz"},
                run_results={
                    ("gz", "sim", "--version"): SafeCommandResult(
                        ("gz", "sim", "--version"), 1, "", "error"
                    )
                },
            )
        )
    )["gazebo"]

    assert missing.status is CapabilityStatus.MISSING
    assert unknown.status is CapabilityStatus.UNKNOWN
    assert incompatible.status is CapabilityStatus.INCOMPATIBLE
    assert timeout.status is CapabilityStatus.TIMEOUT
    assert degraded.status is CapabilityStatus.DEGRADED


def test_gazebo_harmonic_host_outputs_are_active_and_versioned(monkeypatch, tmp_path: Path) -> None:
    report = collect_capabilities(
        _context(
            tmp_path,
            which={"gz": "/usr/bin/gz"},
            run_results={
                ("gz", "sim", "--version"): SafeCommandResult(
                    ("gz", "sim", "--version"),
                    0,
                    "Gazebo Sim, version 8.11.0\nCopyright 2018 Open Source Robotics Foundation",
                ),
                ("gz", "sim", "--versions"): SafeCommandResult(
                    ("gz", "sim", "--versions"), 0, "8.11.0"
                ),
            },
        )
    )
    gazebo = report.by_id()["gazebo"]

    assert gazebo.status is CapabilityStatus.ACTIVE
    assert gazebo.compatibility is CompatibilityStatus.SUPPORTED
    assert gazebo.detected_version == "8.11.0"
    assert "selected_release=Harmonic" in gazebo.evidence
    assert "installed_versions=8.11.0" in gazebo.evidence

    def fake_collect(*, required_capabilities=()):
        from hephaestus.doctor import collect_diagnostics

        return collect_diagnostics(
            env={"HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config")},
            project_root=tmp_path,
            sys_prefix=str(tmp_path / ".venv"),
            python_version=(3, 12, 3),
            system="Linux",
            machine="x86_64",
            os_release_provider=lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
            which=lambda command: "/usr/bin/gz" if command == "gz" else None,
            run=lambda argv, timeout: {
                ("gz", "sim", "--version"): SafeCommandResult(
                    ("gz", "sim", "--version"), 0, "Gazebo Sim, version 8.11.0"
                ),
                ("gz", "sim", "--versions"): SafeCommandResult(
                    ("gz", "sim", "--versions"), 0, "8.11.0"
                ),
            }.get(tuple(argv), SafeCommandResult(tuple(argv), 1, "", "missing")),
            required_capabilities=tuple(required_capabilities),
        )

    monkeypatch.setattr(cli, "collect_diagnostics", fake_collect)
    result = runner.invoke(cli.app, ["doctor", "--require", "gazebo", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["requirement_failures"] == []
    assert payload["checks"][3]["status"] == "ready"
    assert "8.11.0" in result.output
    assert "\x1b[" not in result.output


def test_gazebo_installed_harmonic_without_selected_runtime_is_not_active(
    tmp_path: Path,
) -> None:
    report = collect_capabilities(
        _context(
            tmp_path,
            which={"gz": "/usr/bin/gz"},
            run_results={
                ("gz", "sim", "--version"): SafeCommandResult(
                    ("gz", "sim", "--version"), 0, "unrecognized version output"
                ),
                ("gz", "sim", "--versions"): SafeCommandResult(
                    ("gz", "sim", "--versions"), 0, "8.11.0"
                ),
            },
        )
    )
    gazebo = report.by_id()["gazebo"]

    assert gazebo.status is CapabilityStatus.INSTALLED_INACTIVE
    assert gazebo.compatibility is CompatibilityStatus.SUPPORTED
    assert gazebo.detected_version == "8.11.0"
    assert not report.require(("core",))
    assert report.require(("gazebo",)) == (gazebo,)


def test_ros_gz_packages_and_cad_ai_states(tmp_path: Path) -> None:
    report = _by_id(
        collect_capabilities(
            _context(
                tmp_path,
                env={"ROS_DISTRO": "jazzy"},
                which={"ros2": "/usr/bin/ros2"},
                specs={"groq"},
                run_results={
                    ("ros2", "pkg", "prefix", "ros_gz_sim"): SafeCommandResult(
                        ("ros2", "pkg", "prefix", "ros_gz_sim"), 0, "/opt/ros/jazzy"
                    ),
                    ("ros2", "pkg", "prefix", "ros_gz_bridge"): SafeCommandResult(
                        ("ros2", "pkg", "prefix", "ros_gz_bridge"), 1, "", "missing"
                    ),
                    ("ros2", "pkg", "prefix", "gz_ros2_control"): SafeCommandResult(
                        ("ros2", "pkg", "prefix", "gz_ros2_control"), 124, timed_out=True
                    ),
                },
            )
        )
    )

    assert report["ros_gz_sim"].status is CapabilityStatus.ACTIVE
    assert report["ros_gz_bridge"].status is CapabilityStatus.MISSING
    assert report["gz_ros2_control"].status is CapabilityStatus.TIMEOUT
    assert report["cad"].status is CapabilityStatus.MISSING
    assert report["ai"].status is CapabilityStatus.INSTALLED_INACTIVE


def test_detector_exception_is_contained(tmp_path: Path) -> None:
    class BrokenDetector:
        capability_id = "broken"

        def inspect(self, context):
            raise RuntimeError("boom")

    report = CapabilityRegistry((BrokenDetector(),)).inspect(_context(tmp_path))

    assert report.capabilities[0].status is CapabilityStatus.FAILED
    assert "RuntimeError" in report.capabilities[0].evidence[0]


def test_core_detector_failure_fails_core_readiness(tmp_path: Path) -> None:
    class BrokenCoreDetector:
        capability_id = "core"

        def inspect(self, context):
            raise RuntimeError("boom")

    report = CapabilityRegistry((BrokenCoreDetector(),)).inspect(_context(tmp_path))

    assert report.capabilities[0].status is CapabilityStatus.FAILED
    assert report.capabilities[0].required_for_core is True
    assert report.core_ready is False


def test_setup_plan_is_declarative_non_executable(tmp_path: Path) -> None:
    report = collect_capabilities(_context(tmp_path))
    plan = build_setup_plan(report, required_capabilities=("ros2", "gazebo", "cad"))
    payload = plan.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["schema"] == "hephaestus.setup_plan.v1"
    assert payload["executable"] is False
    assert payload["authorization_granted"] is False
    assert "sudo" not in encoded.lower()
    assert "argv" not in encoded
    assert {item["capability"]["id"] for item in payload["items"]} == {"ros2", "gazebo", "cad"}


def test_capabilities_cli_json_and_human_are_stable(monkeypatch, tmp_path: Path) -> None:
    report = collect_capabilities(_context(tmp_path))
    monkeypatch.setattr(cli, "collect_capabilities", lambda: report)

    human = runner.invoke(cli.app, ["capabilities"])
    machine = runner.invoke(cli.app, ["capabilities", "--json"])

    assert human.exit_code == 0
    assert "Hephaestus Backend Capabilities" in human.output
    assert machine.exit_code == 0
    payload = json.loads(machine.output)
    assert payload["schema"] == "hephaestus.capabilities.v1"
    assert "\x1b[" not in machine.output


def test_setup_plan_cli_json_contains_no_authorization(monkeypatch, tmp_path: Path) -> None:
    report = collect_capabilities(_context(tmp_path))
    monkeypatch.setattr(cli, "collect_capabilities", lambda: report)

    result = runner.invoke(cli.app, ["setup", "plan", "--require", "ros2", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["executable"] is False
    assert payload["authorization_granted"] is False
    assert payload["items"][0]["remediation"]["privilege_required"] == "administrator"
    assert "\x1b[" not in result.output


def test_unknown_capability_requirement_fails_cleanly(tmp_path: Path) -> None:
    report = collect_capabilities(_context(tmp_path))

    try:
        report.require(("does_not_exist",))
    except CapabilityInspectionError as exc:
        assert "unknown capability" in str(exc)
    else:
        raise AssertionError("expected unknown capability failure")


def test_doctor_require_failure_and_multiple_require_json(monkeypatch, tmp_path: Path) -> None:
    from hephaestus.doctor import collect_diagnostics

    def fake_collect(*, required_capabilities=()):
        return collect_diagnostics(
            env={"HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config")},
            project_root=tmp_path,
            sys_prefix=str(tmp_path / ".venv"),
            python_version=(3, 12, 3),
            system="Linux",
            machine="x86_64",
            os_release_provider=lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
            which=lambda command: None,
            find_spec=lambda module_name: object() if module_name == "hephaestus.model" else None,
            package_version=lambda name: None,
            run=lambda argv, timeout: SafeCommandResult(tuple(argv), 1, "", "missing"),
            list_dir_names=lambda path: (),
            required_capabilities=tuple(required_capabilities),
        )

    monkeypatch.setattr(cli, "collect_diagnostics", fake_collect)

    result = runner.invoke(cli.app, ["doctor", "--require", "ros2", "--require", "cad", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["required_capabilities"] == ["ros2", "cad"]
    assert payload["requirement_failures"] == ["ros2", "cad"]
    assert "\x1b[" not in result.output


def test_doctor_require_ai_success_and_strict_core_optional_absent(
    monkeypatch, tmp_path: Path
) -> None:
    from hephaestus.doctor import collect_diagnostics

    def fake_collect(*, required_capabilities=()):
        return collect_diagnostics(
            env={
                "HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config"),
                "GROQ_API_KEY": "gsk_secret_not_printed",
            },
            project_root=tmp_path,
            sys_prefix=str(tmp_path / ".venv"),
            python_version=(3, 12, 3),
            system="Linux",
            machine="x86_64",
            os_release_provider=lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
            which=lambda command: None,
            find_spec=lambda module_name: (
                object() if module_name in {"hephaestus.model", "groq"} else None
            ),
            package_version=lambda name: "1.0" if name == "groq" else None,
            run=lambda argv, timeout: SafeCommandResult(tuple(argv), 1, "", "missing"),
            list_dir_names=lambda path: (),
            required_capabilities=tuple(required_capabilities),
        )

    monkeypatch.setattr(cli, "collect_diagnostics", fake_collect)

    strict = runner.invoke(cli.app, ["doctor", "--strict"])
    ai = runner.invoke(cli.app, ["doctor", "--require", "ai", "--json"])

    assert strict.exit_code == 0
    assert ai.exit_code == 0
    assert "gsk_secret_not_printed" not in ai.output


def test_doctor_unknown_require_fails_cleanly(monkeypatch) -> None:
    def fake_collect(*, required_capabilities=()):
        raise CapabilityInspectionError("unknown capability: nope")

    monkeypatch.setattr(cli, "collect_diagnostics", fake_collect)

    result = runner.invoke(cli.app, ["doctor", "--require", "nope", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "unknown capability" in payload["error"]


def test_safe_subprocess_runner_uses_fixed_argv_and_filtered_environment(monkeypatch) -> None:
    captured = {}

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return Completed()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_not_for_children")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr("hephaestus.capabilities.subprocess.run", fake_run)

    result = run_fixed_argv(("gz", "sim", "--versions"), 1.0)

    assert result.stdout == "ok"
    assert captured["argv"] == ("gz", "sim", "--versions")
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["capture_output"] is True
    assert captured["timeout"] == 1.0
    assert captured["env"]["PATH"] == "/usr/bin"
    assert "GROQ_API_KEY" not in captured["env"]
    assert "shell" not in captured
    assert "GROQ_API_KEY" not in json.dumps(captured, default=str)


def test_ros_gz_packages_do_not_probe_when_ros_unsourced_or_wrong_distro(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def run(argv, timeout):
        calls.append(tuple(argv))
        return SafeCommandResult(tuple(argv), 0, "/opt/ros/jazzy")

    unsourced = collect_capabilities(
        _context(tmp_path, which={"ros2": "/usr/bin/ros2"}, run_results={})
    )
    wrong = collect_capabilities(
        _context(
            tmp_path,
            env={"ROS_DISTRO": "humble"},
            which={"ros2": "/usr/bin/ros2"},
            run_results={},
        )
    )
    explicit = collect_capabilities(
        _context(
            tmp_path,
            env={"ROS_DISTRO": "humble"},
            which={"ros2": "/usr/bin/ros2"},
            run_func=run,
        )
    )

    for report in (unsourced, wrong, explicit):
        indexed = report.by_id()
        assert indexed["ros_gz_sim"].status is not CapabilityStatus.ACTIVE
        assert indexed["ros_gz_bridge"].status is not CapabilityStatus.ACTIVE
        assert indexed["gz_ros2_control"].status is not CapabilityStatus.ACTIVE
    assert calls == []


def test_doctor_require_gazebo_fails_for_non_ready_states(monkeypatch, tmp_path: Path) -> None:
    from hephaestus.doctor import collect_diagnostics

    states = [
        SafeCommandResult(("gz", "sim", "--version"), 0, "???"),
        SafeCommandResult(("gz", "sim", "--version"), 1, "", "error"),
        SafeCommandResult(("gz", "sim", "--version"), 124, timed_out=True),
        SafeCommandResult(("gz", "sim", "--version"), 0, "Gazebo Garden"),
    ]

    for command_result in states:

        def fake_collect(*, required_capabilities=(), command_result=command_result):
            return collect_diagnostics(
                env={"HEPHAESTUS_CONFIG_DIR": str(tmp_path / "config")},
                project_root=tmp_path,
                sys_prefix=str(tmp_path / ".venv"),
                python_version=(3, 12, 3),
                system="Linux",
                machine="x86_64",
                os_release_provider=lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"},
                which=lambda command: "/usr/bin/gz" if command == "gz" else None,
                run=lambda argv, timeout: command_result,
                required_capabilities=tuple(required_capabilities),
            )

        monkeypatch.setattr(cli, "collect_diagnostics", fake_collect)
        result = runner.invoke(cli.app, ["doctor", "--require", "gazebo", "--json"])
        assert result.exit_code == 1
        assert json.loads(result.output)["requirement_failures"] == ["gazebo"]


def test_setup_plan_unknown_requirement_fails_cleanly(monkeypatch, tmp_path: Path) -> None:
    report = collect_capabilities(_context(tmp_path))
    monkeypatch.setattr(cli, "collect_capabilities", lambda: report)

    result = runner.invoke(cli.app, ["setup", "plan", "--require", "nope", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "failed"
    assert "unknown capability" in payload["error"]
