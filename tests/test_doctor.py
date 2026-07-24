from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hephaestus import __version__, cli
from hephaestus.doctor import (
    DiagnosticCategory,
    DiagnosticItem,
    DiagnosticReport,
    DiagnosticStatus,
    collect_diagnostics,
)

runner = CliRunner()


def _complete_core_report(**kwargs) -> DiagnosticReport:
    return DiagnosticReport(
        checks=(
            DiagnosticItem(
                key="platform",
                label="Operating system",
                status=DiagnosticStatus.READY,
                detail="Ubuntu 24.04 LTS; architecture x86_64",
                category=DiagnosticCategory.CORE,
                required_for_strict=True,
            ),
            DiagnosticItem(
                key="python",
                label="Python",
                status=DiagnosticStatus.READY,
                detail="3.12.3",
                category=DiagnosticCategory.CORE,
                required_for_strict=True,
            ),
            DiagnosticItem(
                key="python_module_cadquery",
                label="Python module cadquery",
                status=DiagnosticStatus.MISSING,
                detail="not importable",
                category=DiagnosticCategory.CAD,
                required_for_strict=False,
            ),
        )
    )


def _incomplete_core_report(**kwargs) -> DiagnosticReport:
    return DiagnosticReport(
        checks=(
            DiagnosticItem(
                key="platform",
                label="Operating system",
                status=DiagnosticStatus.MISMATCH,
                detail="Other OS; architecture x86_64",
                category=DiagnosticCategory.CORE,
                required_for_strict=True,
            ),
        )
    )


def test_collect_diagnostics_uses_mocked_environment_without_secret_values() -> None:
    secret = "sk-test-secret-value"
    project_root = Path("/workspace/hephaestus")
    report = collect_diagnostics(
        env={"ROS_DISTRO": "jazzy", "OPENAI_API_KEY": secret, "GROQ_API_KEY": secret},
        project_root=project_root,
        sys_prefix=str(project_root / ".venv"),
        python_version=(3, 12, 3),
        system="Linux",
        machine="x86_64",
        os_release_provider=lambda: {
            "ID": "ubuntu",
            "VERSION_ID": "24.04",
            "PRETTY_NAME": "Ubuntu 24.04 LTS",
        },
        which=lambda command: f"/usr/bin/{command}",
        find_spec=lambda module_name: object() if module_name == "cadquery" else None,
    )

    encoded = json.dumps(report.to_dict())

    assert report.is_strict_ready
    assert secret not in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "GROQ_API_KEY" not in encoded
    assert "CAD dependency availability" in encoded
    assert "jazzy" in encoded


def test_optional_cad_missing_does_not_fail_core_strict_readiness() -> None:
    assert _complete_core_report().is_strict_ready


def test_human_doctor_output_renders_table(monkeypatch) -> None:
    monkeypatch.setattr(cli, "collect_diagnostics", _complete_core_report)

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Hephaestus Environment Diagnostics" in result.output
    assert "core runtime" in result.output
    assert "CAD dependency availability" in result.output
    assert "ready" in result.output


def test_doctor_json_is_valid_plain_json_without_rich_formatting(monkeypatch) -> None:
    monkeypatch.setattr(cli, "collect_diagnostics", _complete_core_report)

    result = runner.invoke(cli.app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert "strict_semantics" in payload
    assert "\x1b[" not in result.output
    assert "[green]" not in result.output


def test_doctor_strict_succeeds_with_complete_core_environment(monkeypatch) -> None:
    monkeypatch.setattr(cli, "collect_diagnostics", _complete_core_report)

    result = runner.invoke(cli.app, ["doctor", "--strict"])

    assert result.exit_code == 0


def test_doctor_strict_exits_nonzero_with_incomplete_core_environment(monkeypatch) -> None:
    monkeypatch.setattr(cli, "collect_diagnostics", _incomplete_core_report)

    result = runner.invoke(cli.app, ["doctor", "--strict"])

    assert result.exit_code == 1


def test_version_works() -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"hephaestus {__version__}"


def test_help_works() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "Hephaestus AI robotics engineering CLI" in result.output
    assert "doctor" in result.output
    assert "project" in result.output
