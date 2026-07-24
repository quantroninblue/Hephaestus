from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import hephaestus.tools.cadquery_build as cadquery_build
from hephaestus.ai.types import ToolCall
from hephaestus.tools.cadquery_build import (
    MESH_SCALE,
    CadQueryBuildContext,
    CadQueryBuildError,
    _run_cadquery_process,
    validate_cadquery_source,
)
from hephaestus.tools.workspace import WorkspaceToolExecutor


def _robot_source(*, assembly: bool = False) -> str:
    assembly_text = ""
    if assembly:
        assembly_text = """
assembly = cq.Assembly()
for name in sorted(parts):
    assembly.add(parts[name], name=name)
"""
    return f"""import cadquery as cq

parts = {{
    "chassis": cq.Workplane("XY").box(600, 400, 180).edges("|Z").fillet(20),
    "head": cq.Workplane("XY").box(240, 200, 160).translate((0, 0, 220)),
    "left_wheel": cq.Workplane("YZ").circle(120).extrude(60).translate((0, -260, 0)),
    "right_wheel": cq.Workplane("YZ").circle(120).extrude(60).translate((0, 260, 0)),
}}
{assembly_text}
"""


def _six_part_worker_source() -> str:
    return """import cadquery as cq

body_length = 300
body_width = 200
body_height = 150
head_radius = 80
wheel_radius = 50
wheel_width = 30
arm_length = 120
arm_width = 30
arm_thickness = 20

body = (
    cq.Workplane("XY")
    .box(body_length, body_width, body_height)
    .edges("|Z")
    .fillet(30)
)
head = (
    cq.Workplane("XY")
    .center(-body_length / 2 + head_radius, 0)
    .sphere(head_radius)
    .translate((0, 0, body_height / 2 + head_radius / 2))
)
left_wheel = (
    cq.Workplane("XY")
    .center(0, -body_width / 2 - wheel_width / 2)
    .circle(wheel_radius)
    .extrude(wheel_width)
    .translate((0, 0, -body_height / 2 + wheel_radius))
)
right_wheel = (
    cq.Workplane("XY")
    .center(0, body_width / 2 + wheel_width / 2)
    .circle(wheel_radius)
    .extrude(wheel_width)
    .translate((0, 0, -body_height / 2 + wheel_radius))
)
left_arm = (
    cq.Workplane("XY")
    .center(-body_length / 2 + arm_length / 2, -body_width / 2 - arm_width / 2)
    .box(arm_length, arm_width, arm_thickness)
    .translate((0, 0, body_height / 4))
)
right_arm = (
    cq.Workplane("XY")
    .center(-body_length / 2 + arm_length / 2, body_width / 2 + arm_width / 2)
    .box(arm_length, arm_width, arm_thickness)
    .translate((0, 0, body_height / 4))
)

parts = {
    "body": body,
    "head": head,
    "left_wheel": left_wheel,
    "right_wheel": right_wheel,
    "left_arm": left_arm,
    "right_arm": right_arm,
}
"""


def test_valid_multi_part_cadquery_script_exports_step_stl_assembly_and_si_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cad" / "robot.py"
    source.parent.mkdir()
    source.write_text(_robot_source(assembly=True), encoding="utf-8")
    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path, env={"GROQ_API_KEY": "gsk_no_leak"}),
    )

    result = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "ok"
    payload = result["result"]
    assert payload["success"] is True
    assert payload["committed"] is True
    assert payload["recommended_sdf_mesh_scale"] == MESH_SCALE
    assert payload["generated_part_names"] == ["chassis", "head", "left_wheel", "right_wheel"]
    assert payload["manifest_path"] == "generated/cad/robot/manifest.json"
    assert payload["assembly_step"]["path"] == "generated/cad/robot/assembly.step"
    assert payload["parts"][0]["bounds_m"][0] > 0
    assert payload["parts"][0]["volume_m3"] > 0
    assert payload["parts"][0]["center_of_mass_m"] is not None
    for part in payload["parts"]:
        step = tmp_path / part["step_path"]
        stl = tmp_path / part["stl_path"]
        assert step.stat().st_size > 0
        assert stl.stat().st_size > 0
        assert len(part["hashes"]["step_sha256"]) == 64
        assert len(part["hashes"]["stl_sha256"]) == 64

    manifest = json.loads((tmp_path / payload["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["unit_convention"]["cadquery_source_dimensions"] == "millimetres"
    assert manifest["unit_convention"]["gazebo_stl_mesh_scale"] == MESH_SCALE
    first_hashes = {
        part["name"]: (part["hashes"]["step_sha256"], part["hashes"]["stl_sha256"])
        for part in payload["parts"]
    }
    second = executor.execute(
        ToolCall("build2", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    second_hashes = {
        part["name"]: (part["hashes"]["step_sha256"], part["hashes"]["stl_sha256"])
        for part in second["result"]["parts"]
    }
    assert second_hashes == first_hashes


def test_production_worker_exports_six_part_filleted_sphere_cylinder_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cad" / "warehouse_robot.py"
    source.parent.mkdir()
    source.write_text(_six_part_worker_source(), encoding="utf-8")
    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(
            tmp_path,
            env={
                "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
                "PYTHONPATH": str(Path(cadquery_build.__file__).parents[2]),
                "HOME": str(tmp_path),
            },
        ),
    )

    result = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/warehouse_robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "ok"
    payload = result["result"]
    assert payload["generated_part_names"] == [
        "body",
        "head",
        "left_arm",
        "left_wheel",
        "right_arm",
        "right_wheel",
    ]
    manifest_path = tmp_path / payload["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert len(manifest["parts"]) == 6
    for part in manifest["parts"]:
        step = tmp_path / part["step_path"]
        stl = tmp_path / part["stl_path"]
        assert step.stat().st_size > 0
        assert stl.stat().st_size > 0


def test_confirmation_decline_and_stale_source_create_no_outputs(tmp_path: Path) -> None:
    source = tmp_path / "cad" / "robot.py"
    source.parent.mkdir()
    source.write_text(_robot_source(), encoding="utf-8")
    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path),
    )

    declined = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: False,
    )
    assert declined["status"] == "ok"
    assert declined["result"]["committed"] is False
    assert not (tmp_path / "generated").exists()

    def mutate(_: dict[str, Any]) -> bool:
        source.write_text(_robot_source() + "\n# changed\n", encoding="utf-8")
        return True

    stale = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=mutate,
    )
    assert stale["status"] == "error"
    assert "changed after CadQuery build preview" in stale["error"]
    assert not (tmp_path / "generated").exists()


def test_atomic_preservation_of_previous_outputs_on_failure(tmp_path: Path) -> None:
    source = tmp_path / "cad" / "robot.py"
    source.parent.mkdir()
    source.write_text(_robot_source(), encoding="utf-8")
    previous = tmp_path / "generated" / "cad" / "robot"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text("previous\n", encoding="utf-8")

    def fail(argv, **kwargs):
        return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="bad cad\n")

    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path, run_process=fail),
    )

    failed = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert failed["status"] == "error"
    assert "bad cad" in failed["error"]
    assert (previous / "manifest.json").read_text(encoding="utf-8") == "previous\n"


def test_signal_crash_reports_signal_and_preserves_previous_outputs(tmp_path: Path) -> None:
    source = tmp_path / "cad" / "warehouse_robot.py"
    source.parent.mkdir()
    source.write_text(_six_part_worker_source(), encoding="utf-8")
    previous = tmp_path / "generated" / "cad" / "warehouse_robot"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text("previous\n", encoding="utf-8")

    def signal_crash(argv, **kwargs):
        return subprocess.CompletedProcess(
            tuple(argv),
            -int(signal.SIGSEGV),
            stdout="",
            stderr=f"OpenCascade allocator crash near {source}\napi_key=supersecretvalue\n",
        )

    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path, run_process=signal_crash),
    )

    failed = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/warehouse_robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert failed["status"] == "error"
    assert "SIGSEGV (11)" in failed["error"]
    assert "OpenCascade allocator crash" in failed["error"]
    assert str(source) not in failed["error"]
    assert "supersecretvalue" not in failed["error"]
    assert (previous / "manifest.json").read_text(encoding="utf-8") == "previous\n"


def test_fake_runner_reports_hashes_and_filters_environment(tmp_path: Path) -> None:
    source = tmp_path / "cad" / "robot.py"
    source.parent.mkdir()
    source.write_text(_robot_source(), encoding="utf-8")
    captured_env: dict[str, str] = {}

    def fake_success(argv, **kwargs):
        captured_env.update(kwargs["env"])
        output = Path(argv[-1])
        (output / "step").mkdir(parents=True)
        (output / "stl").mkdir(parents=True)
        (output / "step" / "chassis.step").write_text("step\n", encoding="utf-8")
        (output / "stl" / "chassis.stl").write_text("solid x\nendsolid x\n", encoding="utf-8")
        (output / ".worker_manifest.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "backend": "cadquery",
                    "backend_version": "fake",
                    "parts": [
                        {
                            "name": "chassis",
                            "bounds_m": [0.6, 0.4, 0.18],
                            "volume_m3": 0.0432,
                            "center_of_mass_m": [0.0, 0.0, 0.0],
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            tuple(argv), 0, stdout='{"success": true, "backend_version": "fake"}\n', stderr=""
        )

    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(
            tmp_path,
            env={"GROQ_API_KEY": "gsk_should_not_leak", "PATH": "/usr/bin"},
            run_process=fake_success,
        ),
    )
    result = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "ok"
    assert result["result"]["backend_version"] == "fake"
    assert len(result["result"]["parts"][0]["hashes"]["stl_sha256"]) == 64
    assert "GROQ_API_KEY" not in captured_env


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("import os\nparts = {}\n", "import is not allowed"),
        (
            "import cadquery as cq\nfrom cadquery import exporters\nparts = {}\n",
            "exporter",
        ),
        ("import cadquery as cq\nvalue = open('x')\nparts = {}\n", "builtin"),
        ("import cadquery as cq\nx = cq.Workplane('XY').__class__\nparts = {}\n", "dunder"),
        ("import cadquery as cq\noutput_path = 'x.step'\nparts = {}\n", "output paths"),
    ],
)
def test_restricted_source_rejections(source: str, message: str) -> None:
    with pytest.raises(CadQueryBuildError, match=message):
        validate_cadquery_source(source)


def test_runtime_rejects_missing_parts_too_many_parts_and_zero_volume(tmp_path: Path) -> None:
    cases = {
        "bad_name.py": (
            "import cadquery as cq\nparts = {'bad-name': cq.Workplane('XY').box(1, 1, 1)}\n"
        ),
        "missing.py": "import cadquery as cq\n",
        "too_many.py": (
            "import cadquery as cq\nparts = {"
            + ", ".join(f"'p{i}': cq.Workplane('XY').box(1, 1, 1)" for i in range(65))
            + "}\n"
        ),
        "zero.py": "import cadquery as cq\nparts = {'line': cq.Workplane('XY').lineTo(1, 0)}\n",
    }
    for name, content in cases.items():
        source = tmp_path / name
        source.write_text(content, encoding="utf-8")
        executor = WorkspaceToolExecutor(
            tmp_path,
            cadquery_context=CadQueryBuildContext(tmp_path),
        )
        result = executor.execute(
            ToolCall("build", "cadquery_build", {"relative_path": name}),
            allow_mutation=True,
            confirm=lambda preview: True,
        )
        assert result["status"] == "error"


def test_source_syntax_size_ast_and_artifact_limits(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(CadQueryBuildError, match="invalid CadQuery syntax"):
        validate_cadquery_source("import cadquery as cq\nparts = {\n")
    with pytest.raises(CadQueryBuildError, match="AST complexity"):
        validate_cadquery_source(
            "import cadquery as cq\n" + "\n".join(f"a{i} = {i}" for i in range(6000))
        )

    source = tmp_path / "large.py"
    source.write_text("#" * (cadquery_build.MAX_CADQUERY_SOURCE_BYTES + 1), encoding="utf-8")
    executor = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path),
    )
    too_large = executor.execute(
        ToolCall("build", "cadquery_build", {"relative_path": "large.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert too_large["status"] == "error"
    assert "source exceeds size limit" in too_large["error"]

    monkeypatch.setattr(cadquery_build, "MAX_ARTIFACT_BYTES", 4)
    valid = tmp_path / "valid.py"
    valid.write_text(_robot_source(), encoding="utf-8")

    def fake_success(argv, **kwargs):
        output = Path(argv[-1])
        (output / "step").mkdir(parents=True)
        (output / "stl").mkdir(parents=True)
        (output / "step" / "chassis.step").write_text("step-over-limit\n", encoding="utf-8")
        (output / "stl" / "chassis.stl").write_text("solid x\nendsolid x\n", encoding="utf-8")
        (output / ".worker_manifest.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "parts": [
                        {
                            "name": "chassis",
                            "bounds_m": [0.6, 0.4, 0.18],
                            "volume_m3": 0.0432,
                            "center_of_mass_m": [0, 0, 0],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(tuple(argv), 0, stdout='{"success": true}\n', stderr="")

    limited = WorkspaceToolExecutor(
        tmp_path,
        cadquery_context=CadQueryBuildContext(tmp_path, run_process=fake_success),
    ).execute(
        ToolCall("build", "cadquery_build", {"relative_path": "valid.py"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert limited["status"] == "error"
    assert "artifact exceeds size limit" in limited["error"]
    assert not (tmp_path / "generated" / "cad" / "valid").exists()


def test_worker_process_timeout_cleans_process_group(monkeypatch, tmp_path: Path) -> None:
    class HangingProcess:
        pid = 7788
        returncode = None

        def __init__(self) -> None:
            self.communicate_calls = 0
            self.kill_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(("python",), timeout)
            self.returncode = -int(signal.SIGTERM)
            return "", "terminated"

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

    process = HangingProcess()
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr("hephaestus.tools.cadquery_build.subprocess.Popen", lambda *a, **k: process)
    monkeypatch.setattr(
        "hephaestus.execution.os.killpg", lambda pid, sig: killed.append((pid, sig))
    )

    with pytest.raises(subprocess.TimeoutExpired):
        _run_cadquery_process(
            ("python", "-m", "hephaestus.tools.cadquery_worker", "x.py", str(tmp_path)),
            stdin=subprocess.DEVNULL,
            text=True,
            env={},
            cwd=str(tmp_path),
            start_new_session=True,
            timeout=0.01,
            capture_output=True,
            check=False,
        )

    assert killed == [(process.pid, signal.SIGTERM)]
    assert process.kill_calls == 0
