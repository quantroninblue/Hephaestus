from __future__ import annotations

import email
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

from hephaestus import __version__
from hephaestus.local_workbench import LEGAL_DOCUMENTS, LocalWorkbenchSession

ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "0.1.1"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LEGAL_RESOURCE_NAMES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "PRIVACY.md",
    "SAFETY.md",
    "SECURITY.md",
    "RESPONSIBLE_USE.md",
    "THIRD_PARTY_NOTICES.md",
}
FORBIDDEN_ARCHIVE_PARTS = {
    ".env",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "generated",
    "project_state.txt",
    "tests",
}


def test_legal_command_uses_installed_resources_outside_repository(tmp_path: Path) -> None:
    session = LocalWorkbenchSession(tmp_path)
    result = session.dispatch("/legal")
    session.close()

    documents = result.payload["documents"]
    assert result.status == "ok"
    assert all(item["available"] for item in documents)
    assert {item["path"] for item in documents if item["path"] in LEGAL_RESOURCE_NAMES} == (
        LEGAL_RESOURCE_NAMES
    )


def test_wheel_sdist_metadata_contents_and_installed_cli(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist_dir)],
        cwd=ROOT,
        check=True,
    )
    wheel = _single(dist_dir.glob(f"hephaestus_ai-{CURRENT_VERSION}-py3-none-any.whl"))
    sdist = _single(dist_dir.glob(f"hephaestus_ai-{CURRENT_VERSION}.tar.gz"))

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        metadata_name = _single(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        entry_points_name = _single(
            name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
        )
        metadata = email.message_from_string(archive.read(metadata_name).decode("utf-8"))
        entry_points = archive.read(entry_points_name).decode("utf-8")

    assert metadata["Name"] == "hephaestus-ai"
    assert metadata["Version"] == CURRENT_VERSION
    assert metadata["License-Expression"] == "Apache-2.0"
    assert metadata["Author"] == "quantroninblue"
    assert not metadata.get_all("Author-email")
    assert "@" not in metadata["Author"]
    assert metadata.get_all("Provides-Extra") == ["cad", "dev", "release"]
    assert "hephaestus = hephaestus.cli:app" in entry_points
    assert __version__ == metadata["Version"]

    runtime_requires = metadata.get_all("Requires-Dist") or []
    assert any(
        item.startswith("cadquery") and 'extra == "cad"' in item for item in runtime_requires
    )
    assert not any(
        item.startswith(("build", "twine", "pytest", "ruff", "hypothesis"))
        and "extra ==" not in item
        for item in runtime_requires
    )
    _assert_archive_policy(wheel_names)
    _assert_legal_resources_in_wheel(wheel_names)

    with tarfile.open(sdist) as archive:
        sdist_names = archive.getnames()
    _assert_archive_policy(sdist_names)
    for resource_name in LEGAL_RESOURCE_NAMES:
        assert f"hephaestus_ai-{CURRENT_VERSION}/{resource_name}" in sdist_names
        assert (
            f"hephaestus_ai-{CURRENT_VERSION}/src/hephaestus/legal/{resource_name}"
            in sdist_names
        )

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = venv_dir / "bin" / "python"
    hephaestus = venv_dir / "bin" / "hephaestus"
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True)

    env = _installed_cli_env()
    work_dir = tmp_path / "external_project"
    work_dir.mkdir()
    imported = subprocess.run(
        [
            str(python),
            "-c",
            "import hephaestus, pathlib; print(pathlib.Path(hephaestus.__file__).resolve())",
        ],
        cwd=work_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert not Path(imported.stdout.strip()).is_relative_to(ROOT / "src")

    assert shutil.which("hephaestus", path=str(venv_dir / "bin")) == str(hephaestus)
    help_result = _run_installed(hephaestus, "--help", cwd=work_dir, env=env)
    assert "Hephaestus AI robotics engineering CLI" in help_result.stdout
    version_result = _run_installed(hephaestus, "--version", cwd=work_dir, env=env)
    assert version_result.stdout.strip() == f"hephaestus {CURRENT_VERSION}"

    capabilities_result = _run_installed(
        hephaestus, "capabilities", "--json", cwd=work_dir, env=env
    )
    assert not ANSI_RE.search(capabilities_result.stdout)
    capabilities_payload = json.loads(capabilities_result.stdout)
    core = _capability_by_id(capabilities_payload, "core")
    assert capabilities_payload["core_status"] == "ready"
    assert core["status"] == "active"
    assert core["compatibility"] == "supported"
    assert "runtime_environment=isolated" in core["evidence"]
    assert "project_venv" not in capabilities_result.stdout
    assert "project .venv" not in capabilities_result.stdout

    doctor_result = _run_installed(hephaestus, "doctor", "--json", cwd=work_dir, env=env)
    assert not ANSI_RE.search(doctor_result.stdout)
    doctor_payload = json.loads(doctor_result.stdout)
    doctor_core = _capability_by_id(doctor_payload, "core")
    assert doctor_payload["status"] == "ready"
    assert doctor_core["status"] == "active"
    assert doctor_core["compatibility"] == "supported"
    assert "project_venv" not in doctor_result.stdout
    assert "project .venv" not in doctor_result.stdout

    status_result = _run_installed(hephaestus, "ai", "status", "--json", cwd=work_dir, env=env)
    assert not ANSI_RE.search(status_result.stdout)
    payload = json.loads(status_result.stdout)
    assert payload["provider"] == "groq"
    assert payload["credential_present"] is False

    legal_result = subprocess.run(
        [str(hephaestus), "--plain"],
        input="/legal\n/exit\n",
        cwd=work_dir,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    for label, path in LEGAL_DOCUMENTS:
        assert label in legal_result.stdout
        if path is not None:
            assert path in legal_result.stdout
    assert "False" not in legal_result.stdout
    assert "unavailable" in legal_result.stdout


def _single(values) -> Path:
    items = list(values)
    assert len(items) == 1
    return items[0]


def _capability_by_id(payload: dict[str, object], capability_id: str) -> dict[str, object]:
    capabilities = payload["capabilities"]
    assert isinstance(capabilities, list)
    for item in capabilities:
        assert isinstance(item, dict)
        if item.get("id") == capability_id:
            return item
    raise AssertionError(f"missing capability: {capability_id}")


def _assert_archive_policy(names: list[str]) -> None:
    for name in names:
        normalized = name.replace("\\", "/")
        parts = set(normalized.split("/"))
        assert FORBIDDEN_ARCHIVE_PARTS.isdisjoint(parts), normalized
        assert ".env" not in parts
        assert not normalized.endswith((".pyc", ".pyo"))
        assert "GROQ_API_KEY=" not in normalized
        assert "gsk_" not in normalized


def _assert_legal_resources_in_wheel(names: list[str]) -> None:
    for resource_name in LEGAL_RESOURCE_NAMES:
        assert f"hephaestus/legal/{resource_name}" in names


def _installed_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GROQ_API_KEY", None)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    dependency_paths = [path for path in site.getsitepackages() if Path(path).is_dir()]
    user_site = site.getusersitepackages()
    if user_site and Path(user_site).is_dir():
        dependency_paths.append(user_site)
    env["PYTHONPATH"] = os.pathsep.join(dependency_paths)
    return env


def _run_installed(
    executable: Path,
    *args: str,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *args],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
