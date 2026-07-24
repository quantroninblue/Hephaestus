from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.ai.agent import AgentLoopError, HephaestusAgentSession
from hephaestus.ai.groq_provider import GroqProvider
from hephaestus.ai.identity import (
    DEFAULT_AGENT_IDENTITY_VERSION,
    DEFAULT_AGENT_SYSTEM_INSTRUCTION,
)
from hephaestus.ai.types import (
    FinishReason,
    ModelRequest,
    ModelResponse,
    ProviderCapabilities,
    ProviderErrorDetails,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderToolUseError,
    ToolCall,
)
from hephaestus.capabilities import (
    CapabilityCategory,
    CapabilityReport,
    CapabilityResult,
    CapabilityStatus,
    CompatibilityStatus,
)
from hephaestus.local_workbench import LocalWorkbenchSession
from hephaestus.tools.cadquery_build import CadQueryBuildContext
from hephaestus.tools.gazebo import GazeboWorkbench
from hephaestus.tools.ros import RosWorkbench
from hephaestus.tools.workspace import (
    MAX_READ_BYTES,
    MAX_WRITE_BYTES,
    WorkspaceToolExecutor,
)

runner = CliRunner()


class ScriptedProvider:
    provider_id = "fake"
    model_id = "fake-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    def capabilities(self):
        return ProviderCapabilities(tool_calls=True, streaming=False)

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            return ModelResponse("done", FinishReason.STOP)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        pass


@dataclass
class Obj:
    pass


@dataclass
class EmptyCapabilities:
    capabilities = ()

    def to_dict(self):
        return {"capabilities": []}


def _agent(provider: ScriptedProvider, root: Path, *, max_tool_rounds: int = 16):
    return HephaestusAgentSession(
        provider=provider,
        startup_directory=root,
        tool_executor=WorkspaceToolExecutor(root),
        capabilities=CapabilityReport(()),
        max_tool_rounds=max_tool_rounds,
    )


def _active_gazebo_capabilities(version: str | None = "8.11.0") -> CapabilityReport:
    return CapabilityReport(
        (
            CapabilityResult(
                "gazebo",
                "Gazebo Harmonic runtime",
                CapabilityCategory.SIMULATION,
                CapabilityStatus.ACTIVE,
                CompatibilityStatus.SUPPORTED,
                detected_version=version,
                evidence=("gz=found", "selected_release=Harmonic"),
            ),
        )
    )


def _inactive_gazebo_capabilities() -> CapabilityReport:
    return CapabilityReport(
        (
            CapabilityResult(
                "gazebo",
                "Gazebo Harmonic runtime",
                CapabilityCategory.SIMULATION,
                CapabilityStatus.MISSING,
                CompatibilityStatus.UNVERIFIED,
                evidence=("gz=missing",),
            ),
        )
    )


def _active_cad_capabilities(version: str | None = "2.8.0") -> CapabilityReport:
    return CapabilityReport(
        (
            CapabilityResult(
                "cad",
                "CadQuery parametric CAD backend",
                CapabilityCategory.CAD,
                CapabilityStatus.ACTIVE,
                CompatibilityStatus.SUPPORTED,
                detected_version=version,
                evidence=("cadquery=importable",),
            ),
        )
    )


def _active_cad_and_gazebo_capabilities() -> CapabilityReport:
    return CapabilityReport(
        (*_active_cad_capabilities().capabilities, *_active_gazebo_capabilities().capabilities)
    )


def _gazebo_agent(
    provider: ScriptedProvider,
    root: Path,
    *,
    run_command=None,
    popen_factory=None,
    capabilities: CapabilityReport | None = None,
    env: dict[str, str] | None = None,
    max_tool_rounds: int = 16,
) -> HephaestusAgentSession:
    workbench = GazeboWorkbench(
        root,
        gazebo_version="8.11.0",
        env=env or {"DISPLAY": ":99", "GROQ_API_KEY": "gsk_should_not_leak"},
        run_command=run_command or _valid_sdf_runner(),
        popen_factory=popen_factory or FakePopenFactory(),
    )
    return HephaestusAgentSession(
        provider=provider,
        startup_directory=root,
        tool_executor=WorkspaceToolExecutor(root, gazebo_workbench=workbench),
        capabilities=capabilities or _active_gazebo_capabilities(),
        max_tool_rounds=max_tool_rounds,
    )


def _minimal_sdf() -> str:
    return """<sdf version="1.10">
  <world name="forge">
    <gravity>0 0 -9.8</gravity>
  </world>
</sdf>
"""


def _valid_sdf_runner(calls: list[tuple[Sequence[str], dict[str, object]]] | None = None):
    def run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append((tuple(argv), dict(kwargs)))
        return subprocess.CompletedProcess(tuple(argv), 0, stdout="valid\n", stderr="")

    return run


def _invalid_then_valid_runner():
    calls: list[Sequence[str]] = []

    def run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(argv))
        if len(calls) == 1:
            return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="bad sdf\n")
        return subprocess.CompletedProcess(tuple(argv), 0, stdout="valid\n", stderr="")

    run.calls = calls
    return run


class FakeProcess:
    pid = 40001

    def __init__(self, *, returncode: int | None = None, stdout: str = "", stderr: str = ""):
        self.pid = FakeProcess.pid
        FakeProcess.pid += 1
        self.returncode = returncode
        self.stdout = StringIO(stdout)
        self.stderr = StringIO(stderr)
        self.wait_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired(["fake"], timeout)
        return self.returncode

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


class FakePopenFactory:
    def __init__(self, process: FakeProcess | None = None):
        self.process = process or FakeProcess()
        self.calls: list[tuple[Sequence[str], dict[str, object]]] = []

    def __call__(self, argv: Sequence[str], **kwargs: object) -> FakeProcess:
        self.calls.append((tuple(argv), dict(kwargs)))
        return self.process


class FakeRosProcess(FakeProcess):
    def communicate(self, timeout: float | None = None):
        return "", ""


def _ros_controller_output() -> str:
    return """controller:
- name: joint_state_broadcaster
  type: joint_state_broadcaster/JointStateBroadcaster
  state: active
- name: diff_drive_controller
  type: diff_drive_controller/DiffDriveController
  state: active
"""


def _fake_ros_run(records: list[tuple[Sequence[str], dict[str, object]]], *, twist_type: str):
    def run(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        records.append((tuple(argv), dict(kwargs)))
        if tuple(argv[:3]) == ("ros2", "topic", "list"):
            return subprocess.CompletedProcess(
                tuple(argv),
                0,
                stdout=f"/diff_drive_controller/cmd_vel [{twist_type}]\n",
                stderr="",
            )
        if tuple(argv[:3]) == ("ros2", "service", "call"):
            return subprocess.CompletedProcess(
                tuple(argv), 0, stdout=_ros_controller_output(), stderr=""
            )
        if tuple(argv[:3]) == ("ros2", "topic", "pub"):
            return subprocess.CompletedProcess(tuple(argv), 0, stdout="published\n", stderr="")
        return subprocess.CompletedProcess(tuple(argv), 0, stdout="", stderr="")

    return run


def _groq_tool_response(name: str, arguments: str):
    call = Obj()
    call.id = "call_1"
    call.type = "function"
    call.function = Obj()
    call.function.name = name
    call.function.arguments = arguments
    message = Obj()
    message.content = None
    message.tool_calls = [call]
    choice = Obj()
    choice.message = message
    choice.finish_reason = "tool_calls"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    return response


def _groq_text_response(content: str):
    message = Obj()
    message.content = content
    message.tool_calls = []
    choice = Obj()
    choice.message = message
    choice.finish_reason = "stop"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    return response


class FakeGroqCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeGroqClient:
    def __init__(self, responses):
        self.chat = Obj()
        self.chat.completions = FakeGroqCompletions(responses)


def test_default_agent_identity_contains_domain_and_physics_policy() -> None:
    assert DEFAULT_AGENT_IDENTITY_VERSION == "hephaestus.agent.identity.v1"
    assert "Identity version: hephaestus.agent.identity.v1" in DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert "AI-native robotics engineer" in DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert "Politely redirect unrelated requests" in DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert "Physics is enabled and physically plausible by default" in (
        DEFAULT_AGENT_SYSTEM_INSTRUCTION
    )
    assert "positive mass" in DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert "valid inertia" in DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert "may not transfer to real hardware" in DEFAULT_AGENT_SYSTEM_INSTRUCTION


def test_hephaestus_no_subcommand_enters_agent_mode_and_binds_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr("hephaestus.ai.agent.collect_capabilities", lambda: CapabilityReport(()))
    fake = ScriptedProvider([ModelResponse("Hello from Hephaestus.", FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, [], input="hello\n/exit\n")

    assert result.exit_code == 0
    assert "Hephaestus | provider=fake model=fake-model" in result.output
    assert f"workspace={tmp_path}" in result.output
    assert "Hello from Hephaestus." in result.output
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request.system_instruction == DEFAULT_AGENT_SYSTEM_INSTRUCTION
    assert str(tmp_path) in request.messages[0].content
    assert "workspace_list_files" in request.messages[0].content
    assert "workspace_read_text" in request.messages[0].content
    assert "workspace_write_text" in request.messages[0].content
    assert "workspace_replace_text" in request.messages[0].content
    assert "gsk_test_secret" not in result.output


def test_hephaestus_no_subcommand_starts_without_groq_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(
        "hephaestus.local_workbench.collect_capabilities", lambda: CapabilityReport(())
    )

    result = runner.invoke(cli.app, [], input="/help\n/exit\n")

    assert result.exit_code == 0
    assert "AI unavailable; local engineering commands remain available." in result.output
    assert "/cad build SOURCE" in result.output
    assert "/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE" in result.output


def test_slash_commands_make_zero_provider_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(
        "hephaestus.local_workbench.collect_capabilities", lambda: CapabilityReport(())
    )
    fake = ScriptedProvider([ModelResponse("should not be used", FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, [], input="/status\n/help\n/exit\n")

    assert result.exit_code == 0
    assert fake.requests == []
    assert "Local workbench status." in result.output


def test_provider_429_then_local_commands_remain_usable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(
        "hephaestus.local_workbench.collect_capabilities", lambda: CapabilityReport(())
    )
    fake = ScriptedProvider(
        [
            ProviderRateLimitError(
                "AI provider rate limit reached",
                ProviderErrorDetails(
                    status_code=429,
                    provider_code="rate_limit_exceeded",
                    retry_after_seconds=3.0,
                ),
            )
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, [], input="hello\n/status\n/exit\n")

    assert result.exit_code == 0
    assert "status=429" in result.output
    assert "Local workbench status." in result.output
    assert len(fake.requests) == 1


def test_local_help_includes_every_slash_command(tmp_path: Path) -> None:
    session = LocalWorkbenchSession(tmp_path, capabilities=CapabilityReport(()))

    result = session.dispatch("/help")

    for command in (
        "/cad build SOURCE",
        "/gazebo validate WORLD",
        "/gazebo launch WORLD",
        "/gazebo launch WORLD --headless",
        "/ros status",
        "/ros topics",
        "/ros controllers",
        "/ros build ROS_WORKSPACE",
        "/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE",
        "/cmdvel LINEAR_X_MPS ANGULAR_Z_RADPS DURATION_SECONDS",
        "/move forward SPEED_MPS for DURATION_SECONDS",
        "/stop",
    ):
        assert command in result.message


def test_local_strict_parser_rejects_invalid_motion_arguments(tmp_path: Path) -> None:
    session = LocalWorkbenchSession(tmp_path, capabilities=CapabilityReport(()))

    nan = session.dispatch("/cmdvel nan 0 1")
    ambiguous = session.dispatch("/move forward fast please")

    assert nan.status == "error"
    assert "finite decimal" in nan.message
    assert ambiguous.status == "error"
    assert "expected /move forward" in ambiguous.message


def test_standalone_cad_json_without_yes_requires_confirmation(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / "robot.py").write_text(
        "import cadquery as cq\nparts = {'body': cq.Workplane('XY').box(1, 1, 1)}\n",
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["cad", "build", "robot.py", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "confirmation_required"
    assert payload["committed"] is False
    assert not (tmp_path / "generated").exists()


def test_standalone_gazebo_validate_json_is_provider_free(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fake_validate(root: Path, world: str) -> dict[str, object]:
        assert root == tmp_path
        assert world == "world.sdf"
        return {
            "status": "ok",
            "tool": "gazebo_validate_sdf",
            "result": {"valid": True, "relative_path": "world.sdf"},
        }

    monkeypatch.setattr(cli, "standalone_gazebo_validate", fake_validate)

    result = runner.invoke(cli.app, ["gazebo", "validate", "world.sdf", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "succeeded"
    assert payload["valid"] is True


def test_ros_workbench_build_launch_and_isolated_status(tmp_path: Path) -> None:
    workspace = tmp_path / "ros_ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "install").mkdir()
    (workspace / "install" / "setup.bash").write_text("# setup\n", encoding="utf-8")
    popen = FakePopenFactory(FakeRosProcess())
    records: list[tuple[Sequence[str], dict[str, object]]] = []
    ros = RosWorkbench(
        tmp_path,
        env={"PATH": "/usr/bin", "ROS_DISTRO": "jazzy", "GROQ_API_KEY": "gsk_secret"},
        popen_factory=popen,
        run_command=_fake_ros_run(records, twist_type="geometry_msgs/msg/TwistStamped"),
    )
    session = LocalWorkbenchSession(
        tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        ros_workbench=ros,
        capabilities=CapabilityReport(()),
    )

    build = session.dispatch("/ros build ros_ws", confirm=lambda preview: True)
    ros.built_workspace = workspace
    launch = session.dispatch(
        "/ros launch ros_ws robot_bringup gazebo_headless.launch.py",
        confirm=lambda preview: True,
    )
    status = session.dispatch("/ros status")

    assert build.payload["argv"][0] == "colcon"
    assert launch.status == "ok"
    assert launch.payload["env"]["ROS_LOCALHOST_ONLY"] == "1"
    assert launch.payload["env"]["GZ_IP"] == "127.0.0.1"
    assert status.payload["ros_distro"] == "jazzy"
    assert status.payload["isolated_domain_id"] == ros.domain_id
    assert "GROQ_API_KEY" not in json.dumps(launch.payload)


def test_ros_movement_requires_owned_simulation_and_bounds(tmp_path: Path) -> None:
    records: list[tuple[Sequence[str], dict[str, object]]] = []
    ros = RosWorkbench(
        tmp_path,
        env={"PATH": "/usr/bin", "ROS_DISTRO": "jazzy"},
        run_command=_fake_ros_run(records, twist_type="geometry_msgs/msg/TwistStamped"),
    )
    session = LocalWorkbenchSession(
        tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        ros_workbench=ros,
        capabilities=CapabilityReport(()),
    )

    no_sim = session.dispatch("/cmdvel 0.1 0 1")
    too_fast = session.dispatch("/cmdvel 0.6 0 1")

    assert no_sim.status == "error"
    assert "owned active ROS simulation" in no_sim.message
    assert too_fast.status == "error"
    assert "0.5 m/s" in too_fast.message


def test_ros_cmdvel_twiststamped_and_stop_publish_zero_velocity(tmp_path: Path) -> None:
    workspace = tmp_path / "ros_ws"
    (workspace / "install").mkdir(parents=True)
    (workspace / "install" / "setup.bash").write_text("# setup\n", encoding="utf-8")
    records: list[tuple[Sequence[str], dict[str, object]]] = []
    ros = RosWorkbench(
        tmp_path,
        env={"PATH": "/usr/bin", "ROS_DISTRO": "jazzy"},
        popen_factory=FakePopenFactory(FakeRosProcess()),
        run_command=_fake_ros_run(records, twist_type="geometry_msgs/msg/TwistStamped"),
    )
    ros.built_workspace = workspace
    session = LocalWorkbenchSession(
        tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        ros_workbench=ros,
        capabilities=CapabilityReport(()),
    )
    launched = session.dispatch(
        "/ros launch ros_ws robot_bringup gazebo_headless.launch.py",
        confirm=lambda preview: True,
    )
    assert launched.status == "ok"

    move = session.dispatch("/cmdvel 0.1 0.2 0.2")
    stop = session.dispatch("/stop")

    pub_calls = [
        tuple(call[0]) for call in records if tuple(call[0][:3]) == ("ros2", "topic", "pub")
    ]
    assert move.status == "ok"
    assert move.payload["topic"] == "/diff_drive_controller/cmd_vel"
    assert move.payload["message_type"] == "geometry_msgs/msg/TwistStamped"
    assert move.payload["zero_velocity_sent"] is True
    assert stop.status == "ok"
    assert any("0.100000" in call[-1] and "0.200000" in call[-1] for call in pub_calls)
    assert any("0.000000" in call[-1] for call in pub_calls)


def test_ros_cmdvel_supports_twist_and_deterministic_move_grammar(tmp_path: Path) -> None:
    workspace = tmp_path / "ros_ws"
    (workspace / "install").mkdir(parents=True)
    (workspace / "install" / "setup.bash").write_text("# setup\n", encoding="utf-8")
    records: list[tuple[Sequence[str], dict[str, object]]] = []
    ros = RosWorkbench(
        tmp_path,
        env={"PATH": "/usr/bin", "ROS_DISTRO": "jazzy"},
        popen_factory=FakePopenFactory(FakeRosProcess()),
        run_command=_fake_ros_run(records, twist_type="geometry_msgs/msg/Twist"),
    )
    ros.built_workspace = workspace
    session = LocalWorkbenchSession(
        tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        ros_workbench=ros,
        capabilities=CapabilityReport(()),
    )
    session.dispatch(
        "/ros launch ros_ws robot_bringup gazebo_headless.launch.py",
        confirm=lambda preview: True,
    )

    moved = session.dispatch("/move turn-left 0.3 for 0.2")

    pub_calls = [
        tuple(call[0]) for call in records if tuple(call[0][:3]) == ("ros2", "topic", "pub")
    ]
    assert moved.status == "ok"
    assert moved.payload["message_type"] == "geometry_msgs/msg/Twist"
    assert any("geometry_msgs/msg/Twist" in call for call in pub_calls)
    assert any("0.300000" in call[-1] for call in pub_calls)


def test_cadquery_build_schema_and_capability_exposure(tmp_path: Path) -> None:
    provider = ScriptedProvider([])
    unavailable = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=CapabilityReport(()),
    )
    available = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=_active_cad_capabilities(),
    )

    unavailable_names = {tool.name for tool in unavailable._tool_definitions()}
    available_tools = {tool.name: tool for tool in available._tool_definitions()}

    assert "cadquery_build" not in unavailable_names
    assert "cadquery_build" in available_tools
    schema = available_tools["cadquery_build"]
    assert schema.input_schema["additionalProperties"] is False
    assert schema.input_schema["required"] == ["relative_path"]
    assert "millimetres" in schema.description
    assert "0.001 0.001 0.001" in schema.description
    assert "cadquery_tool_guidance" in available.dynamic_context()


def test_scripted_agent_workflow_builds_cad_then_writes_and_validates_sdf(
    tmp_path: Path,
) -> None:
    (tmp_path / "cad").mkdir()
    cad_source = """import cadquery as cq
parts = {"chassis": cq.Workplane("XY").box(600, 400, 180)}
"""
    sdf = """<sdf version="1.10">
  <world name="forge">
    <gravity>0 0 -9.8</gravity>
    <model name="robot">
      <link name="base">
        <inertial><mass>20</mass><inertia><ixx>1</ixx><iyy>1</iyy><izz>1</izz></inertia></inertial>
        <collision name="box"><geometry><box><size>0.6 0.4 0.18</size></box></geometry></collision>
        <visual name="mesh"><geometry><mesh>
          <uri>generated/cad/robot/stl/chassis.stl</uri>
          <scale>0.001 0.001 0.001</scale>
        </mesh></geometry></visual>
      </link>
    </model>
  </world>
</sdf>
"""

    def fake_cad(argv, **kwargs):
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
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            tuple(argv), 0, stdout='{"success": true, "backend_version": "fake"}\n', stderr=""
        )

    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "write_cad",
                        "workspace_write_text",
                        {"relative_path": "cad/robot.py", "content": cad_source},
                    ),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "write_sdf",
                        "workspace_write_text",
                        {"relative_path": "world.sdf", "content": sdf},
                    ),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),
                ),
            ),
            ModelResponse("Ready for launch.", FinishReason.STOP),
        ]
    )
    workbench = GazeboWorkbench(
        tmp_path,
        gazebo_version="8.11.0",
        env={"DISPLAY": ":99"},
        run_command=_valid_sdf_runner(),
        popen_factory=FakePopenFactory(),
    )
    session = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(
            tmp_path,
            gazebo_workbench=workbench,
            cadquery_context=CadQueryBuildContext(tmp_path, run_process=fake_cad),
        ),
        capabilities=_active_cad_and_gazebo_capabilities(),
    )

    result = session.ask("create robot", confirm=lambda preview: True)

    assert result.content.startswith("Operation committed; AI explanation unavailable.")
    assert [item["name"] for item in result.tool_activity] == [
        "workspace_write_text",
        "cadquery_build",
    ]
    assert len(provider.requests) == 2
    assert (tmp_path / "generated/cad/robot/manifest.json").exists()
    assert "generated/cad/robot/stl/chassis.stl" in result.content


def test_failed_cadquery_build_is_terminal_and_not_reported_as_success(tmp_path: Path) -> None:
    (tmp_path / "cad").mkdir()
    (tmp_path / "cad" / "robot.py").write_text(
        "import cadquery as cq\nparts = {'chassis': cq.Workplane('XY').box(1, 1, 1)}\n",
        encoding="utf-8",
    )

    def fail(argv, **kwargs):
        return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="export failed\n")

    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("build", "cadquery_build", {"relative_path": "cad/robot.py"}),
                ),
            ),
            ModelResponse("Artifacts created.", FinishReason.STOP),
        ]
    )
    session = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(
            tmp_path,
            cadquery_context=CadQueryBuildContext(tmp_path, run_process=fail),
        ),
        capabilities=_active_cad_capabilities(),
    )

    result = session.ask("build cad", confirm=lambda preview: True)

    assert result.finish_reason is FinishReason.ERROR
    assert "Tool failed (cadquery_build)" in result.content
    assert "export failed" in result.content
    assert len(provider.requests) == 1


def test_agent_conversation_history_persists_between_turns(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ModelResponse("First answer.", FinishReason.STOP),
            ModelResponse("Second answer.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    first = session.ask("first")
    second = session.ask("second")

    assert first.content == "First answer."
    assert second.content == "Second answer."
    second_request_messages = provider.requests[1].messages
    assert [message.content for message in second_request_messages if message.content] == [
        second_request_messages[0].content,
        "first",
        "First answer.",
        "second",
    ]
    assert "messages=2" in second_request_messages[0].content


def test_agent_preserves_assistant_tool_call_and_tool_result_order(tmp_path: Path) -> None:
    (tmp_path / "design.md").write_text("wheel torque note\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "list_1",
                        "workspace_list_files",
                        {"relative_path": ".", "limit": 10},
                    ),
                    ToolCall("read_1", "workspace_read_text", {"relative_path": "design.md"}),
                ),
            ),
            ModelResponse("Read the note.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("inspect the workspace")

    assert result.content == "Read the note."
    assert [item["name"] for item in result.tool_activity] == [
        "workspace_list_files",
        "workspace_read_text",
    ]
    second = provider.requests[1].messages
    assert second[0].role.value == "system"
    assert second[1].role.value == "user"
    assert second[2].role.value == "assistant"
    assert [call.id for call in second[2].tool_calls] == ["list_1", "read_1"]
    assert second[3].role.value == "tool"
    assert second[3].tool_call_id == "list_1"
    assert second[4].role.value == "tool"
    assert second[4].tool_call_id == "read_1"


def test_workspace_list_read_write_confirmation_and_atomic_replace(
    monkeypatch, tmp_path: Path
) -> None:
    existing = tmp_path / "notes.md"
    existing.write_text("old\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    listed = executor.execute(
        ToolCall("list", "workspace_list_files", {"relative_path": ".", "limit": 5}),
        allow_mutation=True,
    )
    assert listed["status"] == "ok"
    assert listed["result"]["files"] == [{"relative_path": "notes.md", "bytes": 4}]

    read = executor.execute(
        ToolCall("read", "workspace_read_text", {"relative_path": "notes.md"}),
        allow_mutation=True,
    )
    assert read["status"] == "ok"
    assert read["result"]["content"] == "old\n"

    replacements: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def record_replace(src, dst):
        replacements.append((Path(src), Path(dst)))
        original_replace(src, dst)

    monkeypatch.setattr("hephaestus.tools.workspace.os.replace", record_replace)
    previews: list[dict] = []
    written = executor.execute(
        ToolCall(
            "write",
            "workspace_write_text",
            {"relative_path": "notes.md", "content": "new robot note\n"},
        ),
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or True,
    )

    assert previews[0]["result"]["relative_path"] == "notes.md"
    assert previews[0]["result"]["operation"] == "replacing"
    assert written["status"] == "ok"
    assert written["result"]["operation"] == "replaced"
    assert written["result"]["committed"] is True
    assert existing.read_text(encoding="utf-8") == "new robot note\n"
    assert replacements
    assert replacements[0][1] == existing


def test_workspace_replace_text_schema_is_provider_safe(tmp_path: Path) -> None:
    executor = WorkspaceToolExecutor(tmp_path)

    schema = next(
        tool
        for tool in executor.definitions(allow_mutation=True)
        if tool.name == "workspace_replace_text"
    )

    assert schema.input_schema["additionalProperties"] is False
    assert schema.input_schema["required"] == ["relative_path", "old_text", "new_text"]
    assert schema.input_schema["properties"].keys() == {
        "relative_path",
        "old_text",
        "new_text",
    }
    assert "workspace_read_text before editing" in schema.description
    assert "prefer it over workspace_write_text" in schema.description
    assert "read, edit, observe the committed result, validate, then launch" in (schema.description)


def test_workspace_replace_text_unique_replacement_and_committed_fingerprint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.md"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)
    previews: list[dict[str, object]] = []

    result = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {
                "relative_path": "notes.md",
                "old_text": "beta\n",
                "new_text": "delta\n",
            },
        ),
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or True,
    )

    committed = target.read_bytes()
    assert previews[0]["result"] == {
        "relative_path": "notes.md",
        "operation": "targeted replacement",
        "old_byte_count": 5,
        "new_byte_count": 6,
        "resulting_file_byte_count": len(committed),
        "before_fingerprint": hashlib.sha256(b"alpha\nbeta\ngamma\n").hexdigest(),
    }
    assert "beta" not in json.dumps(previews[0])
    assert result["status"] == "ok"
    assert result["result"]["committed"] is True
    assert result["result"]["occurrences_replaced"] == 1
    assert result["result"]["after_fingerprint"] == hashlib.sha256(committed).hexdigest()
    assert target.read_text(encoding="utf-8") == "alpha\ndelta\ngamma\n"


def test_workspace_replace_text_conflict_ambiguity_empty_and_deletion(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.md"
    target.write_text("keep\nremove\nrepeat\nrepeat\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    zero = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "missing", "new_text": "x"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    ambiguous = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "repeat", "new_text": "x"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    empty_old = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "", "new_text": "x"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    deleted = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "remove\n", "new_text": ""},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert zero["status"] == "error"
    assert "zero times" in zero["error"]
    assert zero["evidence"] == {"code": "read_write_conflict"}
    assert ambiguous["status"] == "error"
    assert "ambiguous" in ambiguous["error"]
    assert empty_old["status"] == "error"
    assert empty_old["error"] == "invalid arguments"
    assert deleted["status"] == "ok"
    assert deleted["result"]["new_byte_count"] == 0
    assert target.read_text(encoding="utf-8") == "keep\nrepeat\nrepeat\n"


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/outside.txt",
        "../outside.txt",
        ".git/config",
        ".venv/config",
        ".env",
        ".env.local",
        "project_state.txt",
        "robot.secret",
        "token.txt",
    ],
)
def test_workspace_replace_text_rejects_traversal_and_protected_paths(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "token.txt").write_text("old\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": path, "old_text": "old", "new_text": "new"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "error"
    assert (tmp_path / "token.txt").read_text(encoding="utf-8") == "old\n"


def test_workspace_replace_text_rejects_symlink_and_oversized_inputs_and_result(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside_replace.txt"
    outside.write_text("old\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8")
    large = tmp_path / "large.txt"
    large.write_text("a" * (MAX_WRITE_BYTES - 3) + "old", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    symlink = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "link.txt", "old_text": "old", "new_text": "new"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    too_big_arg = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {
                "relative_path": "notes.md",
                "old_text": "old",
                "new_text": "x" * (MAX_WRITE_BYTES + 1),
            },
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    too_big_result = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {
                "relative_path": "large.txt",
                "old_text": "old",
                "new_text": "newer",
            },
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert symlink["status"] == "error"
    assert "escapes startup directory" in symlink["error"]
    assert too_big_arg["status"] == "error"
    assert too_big_result["status"] == "error"
    assert "resulting file exceeds" in too_big_result["error"]
    assert target.read_text(encoding="utf-8") == "old\n"


def test_workspace_replace_text_decline_stale_and_atomic_failure_preserve_bytes(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    declined = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "old", "new_text": "new"},
        ),
        allow_mutation=True,
        confirm=lambda preview: False,
    )
    assert declined["status"] == "ok"
    assert declined["result"]["committed"] is False
    assert target.read_text(encoding="utf-8") == "old\n"

    calls = 0
    real_fingerprint = hashlib.sha256

    def changing_fingerprint(value: bytes):
        nonlocal calls
        calls += 1
        payload = value if calls == 1 else value + b"changed"
        return real_fingerprint(payload)

    monkeypatch.setattr("hephaestus.tools.workspace.hashlib.sha256", changing_fingerprint)
    stale = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "old", "new_text": "new"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert stale["status"] == "error"
    assert "changed after replacement preview" in stale["error"]
    assert target.read_text(encoding="utf-8") == "old\n"
    monkeypatch.undo()

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("hephaestus.tools.workspace.os.replace", fail_replace)
    failed = executor.execute(
        ToolCall(
            "replace",
            "workspace_replace_text",
            {"relative_path": "notes.md", "old_text": "old", "new_text": "new"},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert failed["status"] == "error"
    assert failed["error"] == "workspace persistence failed"
    assert target.read_text(encoding="utf-8") == "old\n"


@pytest.mark.parametrize("arguments", [{}, {"relative_path": ""}, {"relative_path": "."}])
def test_workspace_list_files_accepts_workspace_root_forms(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    (tmp_path / "notes.md").write_text("old\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall("list", "workspace_list_files", arguments),
        allow_mutation=True,
    )

    assert result["status"] == "ok"
    assert result["result"]["relative_path"] == "."
    assert result["result"]["files"] == [{"relative_path": "notes.md", "bytes": 4}]


def test_workspace_read_and_write_still_reject_empty_paths(tmp_path: Path) -> None:
    executor = WorkspaceToolExecutor(tmp_path)

    read = executor.execute(
        ToolCall("read", "workspace_read_text", {"relative_path": ""}),
        allow_mutation=True,
    )
    write = executor.execute(
        ToolCall("write", "workspace_write_text", {"relative_path": "", "content": "x"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert read["status"] == "error"
    assert read["error"] == "invalid arguments"
    assert write["status"] == "error"
    assert write["error"] == "invalid arguments"
    assert not (tmp_path / "x").exists()


def test_default_agent_repairs_empty_root_list_then_reads_grounded_note(
    tmp_path: Path,
) -> None:
    note = (
        "LiDAR mounting height: 0.42 m above the base deck.\n"
        "IMU placement: centered near the chassis center of mass.\n"
    )
    (tmp_path / "design_note.md").write_text(note, encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("list_root", "workspace_list_files", {"relative_path": ""}),),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "read_note",
                        "workspace_read_text",
                        {"relative_path": "design_note.md"},
                    ),
                ),
            ),
            ModelResponse(
                "The LiDAR mounting height is 0.42 m above the base deck. "
                "The IMU is centered near the chassis center of mass.",
                FinishReason.STOP,
            ),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask(
        "Read design_note.md from the workspace and tell me the LiDAR mounting "
        "height and IMU placement."
    )

    assert result.content == (
        "The LiDAR mounting height is 0.42 m above the base deck. "
        "The IMU is centered near the chassis center of mass."
    )
    assert "invalid arguments" not in result.content
    second_request = provider.requests[1].messages
    list_result = json.loads(second_request[3].content)
    assert second_request[2].tool_calls[0].name == "workspace_list_files"
    assert second_request[2].tool_calls[0].arguments == {"relative_path": ""}
    assert list_result["status"] == "ok"
    assert list_result["result"]["relative_path"] == "."
    assert list_result["result"]["files"] == [
        {"relative_path": "design_note.md", "bytes": len(note.encode("utf-8"))}
    ]
    third_request = provider.requests[2].messages
    read_result = json.loads(third_request[5].content)
    assert read_result["status"] == "ok"
    assert read_result["result"]["content"] == note


def test_default_agent_write_clear_then_read_accepts_groq_relative_path_arguments(
    tmp_path: Path,
) -> None:
    note = (
        "LiDAR mounting height: 0.42 m above the base deck.\n"
        "IMU placement: centered near the chassis center of mass.\n"
    )
    client = FakeGroqClient(
        [
            _groq_tool_response(
                "workspace_write_text",
                json.dumps({"relative_path": "design_note.md", "content": note}),
            ),
            _groq_text_response("Created design_note.md."),
            _groq_tool_response(
                "workspace_read_text",
                json.dumps({"relative_path": "design_note.md"}),
            ),
            _groq_text_response(
                "The LiDAR is mounted at 0.42 m. The IMU is centered near the "
                "chassis center of mass."
            ),
        ]
    )
    provider = GroqProvider(model="fake-model", client=client)
    session = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=CapabilityReport(()),
    )

    write_result = session.ask("Create design_note.md", confirm=lambda preview: True)
    session.clear()
    read_result = session.ask(
        "Read design_note.md from the workspace and tell me the LiDAR mounting "
        "height and IMU placement."
    )

    assert write_result.content == "Created design_note.md."
    assert (tmp_path / "design_note.md").read_text(encoding="utf-8") == note
    assert "invalid arguments" not in read_result.content
    assert "0.42 m" in read_result.content
    assert "center of mass" in read_result.content
    read_tool_result = client.chat.completions.calls[3]["messages"][4]
    assert read_tool_result["role"] == "tool"
    assert json.loads(read_tool_result["content"])["result"]["content"] == note
    assert client.chat.completions.calls[2]["tools"][1]["function"]["parameters"]["required"] == [
        "relative_path"
    ]
    assert client.chat.completions.calls[2]["tools"][1]["function"]["parameters"][
        "properties"
    ].keys() == {"relative_path"}
    assert str(tmp_path) in client.chat.completions.calls[2]["messages"][1]["content"]


def test_default_agent_groq_requests_disable_parallel_tools_and_use_default_temperature(
    tmp_path: Path,
) -> None:
    client = FakeGroqClient([_groq_text_response("Ready.")])
    provider = GroqProvider(model="fake-model", client=client)
    session = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=CapabilityReport(()),
    )

    result = session.ask("status")

    assert result.content == "Ready."
    kwargs = client.chat.completions.calls[0]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["temperature"] == 0.2
    assert "max_completion_tokens" not in kwargs
    assert "observe manifest" in kwargs["messages"][1]["content"]
    assert "validate SDF -> launch Gazebo" in kwargs["messages"][1]["content"]


def test_default_agent_groq_request_uses_explicit_configured_temperature(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr("hephaestus.ai.agent.collect_capabilities", lambda: CapabilityReport(()))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "ai.json").write_text(
        json.dumps(
            {
                "provider": "groq",
                "model": "fake-model",
                "timeout_seconds": 30.0,
                "default_agent_tool_temperature": 0.05,
            }
        ),
        encoding="utf-8",
    )
    client = FakeGroqClient([_groq_text_response("Configured.")])
    provider = GroqProvider(model="fake-model", client=client)
    monkeypatch.setattr(cli, "_create_provider", lambda config: provider)

    result = runner.invoke(cli.app, [], input="hello\n/exit\n")

    assert result.exit_code == 0
    assert client.chat.completions.calls[0]["temperature"] == 0.05
    assert client.chat.completions.calls[0]["parallel_tool_calls"] is False


def test_workspace_accepts_valid_decoded_object_arguments(tmp_path: Path) -> None:
    (tmp_path / "design_note.md").write_text("LiDAR mounting height: 0.42 m\n", encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall(
            "read",
            "workspace_read_text",
            {"relative_path": "design_note.md"},
        ),
        allow_mutation=True,
    )

    assert result["status"] == "ok"
    assert result["result"]["relative_path"] == "design_note.md"
    assert result["result"]["content"] == "LiDAR mounting height: 0.42 m\n"


def test_agent_declined_workspace_write_is_terminal_and_preserves_bytes(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8")
    before = target.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "write",
                        "workspace_write_text",
                        {"relative_path": "notes.md", "content": "new\n"},
                    ),
                ),
            ),
            ModelResponse("I wrote the file.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("write it", confirm=lambda preview: False)

    assert result.content == "Workspace write not applied."
    assert result.finish_reason is FinishReason.ERROR
    assert len(provider.requests) == 1
    assert target.read_bytes() == before


def test_agent_declined_workspace_replace_is_terminal_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.md"
    target.write_text("old\n", encoding="utf-8")
    before = target.read_bytes()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "replace",
                        "workspace_replace_text",
                        {
                            "relative_path": "notes.md",
                            "old_text": "old",
                            "new_text": "new",
                        },
                    ),
                ),
            ),
            ModelResponse("I edited the file.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("edit it", confirm=lambda preview: False)

    assert result.content == "Workspace write not applied."
    assert result.finish_reason is FinishReason.ERROR
    assert len(provider.requests) == 1
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({}, "relative_path: Field required"),
        ({"path": "design_note.md"}, "path: Extra inputs are not permitted"),
        (
            {"relative_path": "design_note.md", "extra": "nope"},
            "extra: Extra inputs are not permitted",
        ),
    ],
)
def test_agent_reports_workspace_argument_validation_details(
    tmp_path: Path, arguments: dict[str, object], expected: str
) -> None:
    (tmp_path / "design_note.md").write_text("note\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("read", "workspace_read_text", arguments),),
            ),
            ModelResponse("Use a valid relative_path.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("read design_note.md")

    assert result.content == "Use a valid relative_path."
    assert len(provider.requests) == 2
    repair_request = provider.requests[1].messages
    assert repair_request[2].role.value == "assistant"
    assert repair_request[2].tool_calls[0].name == "workspace_read_text"
    assert repair_request[3].role.value == "tool"
    tool_error = json.loads(repair_request[3].content)
    assert tool_error["status"] == "error"
    assert tool_error["tool"] == "workspace_read_text"
    assert tool_error["error"] == "invalid arguments"
    assert expected in _validation_text(tool_error)


def test_groq_malformed_json_arguments_fail_before_workspace_dispatch(tmp_path: Path) -> None:
    client = FakeGroqClient(
        [_groq_tool_response("workspace_read_text", '{"relative_path": "design_note.md"')]
    )
    provider = GroqProvider(model="fake-model", client=client)
    session = HephaestusAgentSession(
        provider=provider,
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=CapabilityReport(()),
    )

    with pytest.raises(ProviderMalformedResponseError, match="malformed"):
        session.ask("read design_note.md")


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/outside.txt",
        "../outside.txt",
        ".git/config",
        ".venv/config",
        ".env",
        ".env.local",
        "project_state.txt",
        "robot.secret",
        "token.txt",
    ],
)
def test_workspace_rejects_unsafe_and_protected_paths(tmp_path: Path, path: str) -> None:
    executor = WorkspaceToolExecutor(tmp_path)
    result = executor.execute(
        ToolCall("write", "workspace_write_text", {"relative_path": path, "content": "x"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "error"
    assert not (tmp_path / "token.txt").exists()


def test_workspace_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    executor = WorkspaceToolExecutor(tmp_path)

    result = executor.execute(
        ToolCall("read", "workspace_read_text", {"relative_path": "link.txt"}),
        allow_mutation=True,
    )

    assert result["status"] == "error"
    assert "escapes startup directory" in result["error"]


def test_workspace_bounds_read_write_and_list_results(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file_{index}.txt").write_text(str(index), encoding="utf-8")
    big = tmp_path / "big.txt"
    big.write_text("x" * (MAX_READ_BYTES + 1), encoding="utf-8")
    executor = WorkspaceToolExecutor(tmp_path)

    listed = executor.execute(
        ToolCall("list", "workspace_list_files", {"relative_path": ".", "limit": 2}),
        allow_mutation=True,
    )
    assert listed["result"]["count"] == 2
    assert listed["result"]["truncated"] is True

    read = executor.execute(
        ToolCall("read", "workspace_read_text", {"relative_path": "big.txt"}),
        allow_mutation=True,
    )
    assert read["status"] == "error"
    assert "read size limit" in read["error"]

    write = executor.execute(
        ToolCall(
            "write",
            "workspace_write_text",
            {"relative_path": "too_big.txt", "content": "x" * (MAX_WRITE_BYTES + 1)},
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert write["status"] == "error"


def test_agent_validates_complete_tool_batch_before_any_execution(tmp_path: Path) -> None:
    safe = tmp_path / "safe.txt"
    safe.write_text("safe\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "write",
                        "workspace_write_text",
                        {"relative_path": "safe.txt", "content": "changed\n"},
                    ),
                    ToolCall("bad", "workspace_read_text", {"relative_path": "../outside.txt"}),
                ),
            )
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("write and read", confirm=lambda preview: True)

    assert result.finish_reason is FinishReason.ERROR
    assert "workspace_read_text" in result.content
    assert "parent traversal" in result.content
    assert safe.read_text(encoding="utf-8") == "safe\n"


def test_agent_read_only_validation_error_allows_one_repair(tmp_path: Path) -> None:
    (tmp_path / "design_note.md").write_text("LiDAR height 0.42 m\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("bad_read", "workspace_read_text", {"relative_path": ""}),),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "good_read",
                        "workspace_read_text",
                        {"relative_path": "design_note.md"},
                    ),
                ),
            ),
            ModelResponse("LiDAR height is 0.42 m.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("read design_note.md")

    assert result.content == "LiDAR height is 0.42 m."
    assert len(provider.requests) == 3
    repair_request = provider.requests[1].messages
    tool_error = json.loads(repair_request[3].content)
    assert tool_error["tool"] == "workspace_read_text"
    assert tool_error["error"] == "invalid arguments"


def test_agent_read_only_invalid_batch_does_not_partially_execute(
    tmp_path: Path,
) -> None:
    (tmp_path / "safe.txt").write_text("safe content\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("good_read", "workspace_read_text", {"relative_path": "safe.txt"}),
                    ToolCall("bad_read", "workspace_read_text", {"relative_path": ""}),
                ),
            ),
            ModelResponse("Repair acknowledged.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("read two files")

    assert result.content == "Repair acknowledged."
    repair_request = provider.requests[1].messages
    good_result = json.loads(repair_request[3].content)
    bad_result = json.loads(repair_request[4].content)
    assert good_result["tool"] == "workspace_read_text"
    assert good_result["tool_call_id"] == "good_read"
    assert good_result["error"] == "tool batch rejected before execution"
    assert "safe content" not in repair_request[3].content
    assert bad_result["tool_call_id"] == "bad_read"
    assert bad_result["error"] == "invalid arguments"


def test_agent_provider_error_is_visible_and_cli_recovers(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr("hephaestus.ai.agent.collect_capabilities", lambda: CapabilityReport(()))
    fake = ScriptedProvider(
        [
            ProviderToolUseError(
                "AI provider could not produce a valid engineering operation. "
                "No changes were made.",
                ProviderErrorDetails(
                    status_code=400,
                    provider_code="tool_use_failed",
                    provider_error_type="invalid_request_error",
                    request_id="req_test",
                ),
            ),
            ModelResponse("Recovered.", FinishReason.STOP),
        ]
    )
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, [], input="bad\ngood\n/exit\n")

    assert result.exit_code == 0
    assert "valid engineering operation" in result.output
    assert "tool_use_failed" in result.output
    assert "Recovered." in result.output
    assert "gsk_test_secret" not in result.output


def test_default_agent_max_tool_rounds_default_is_16(tmp_path: Path) -> None:
    session = _agent(ScriptedProvider([]), tmp_path)

    assert session.max_tool_rounds == 16
    assert "max_tool_rounds=16" in session.dynamic_context()


def test_legitimate_six_step_workflow_does_not_exhaust_default_guard(tmp_path: Path) -> None:
    for index in range(6):
        (tmp_path / f"step_{index}.txt").write_text(f"{index}\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            *(
                ModelResponse(
                    "",
                    FinishReason.TOOL_CALLS,
                    tool_calls=(
                        ToolCall(
                            f"read_{index}",
                            "workspace_read_text",
                            {"relative_path": f"step_{index}.txt"},
                        ),
                    ),
                )
                for index in range(6)
            ),
            ModelResponse("Six sequential tool rounds completed.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path)

    result = session.ask("run six dependent inspection steps")

    assert result.finish_reason is FinishReason.STOP
    assert result.content == "Six sequential tool rounds completed."
    assert len(provider.requests) == 7
    assert len(result.tool_activity) == 6


def test_agent_tool_round_guard_reports_without_claiming_completion(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("list1", "workspace_list_files", {"relative_path": ".", "limit": 1}),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("list2", "workspace_list_files", {"relative_path": ".", "limit": 1}),
                ),
            ),
        ]
    )
    session = _agent(provider, tmp_path, max_tool_rounds=1)

    result = session.ask("loop")

    assert result.finish_reason is FinishReason.ERROR
    assert "tool-call budget" in result.content
    assert len(provider.requests) == 2
    assert result.tool_activity == ({"name": "workspace_list_files", "status": "ok"},)


def test_agent_tool_round_guard_terminates_only_current_turn(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1\n", encoding="utf-8")
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("list1", "workspace_list_files", {"relative_path": ".", "limit": 1}),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall("list2", "workspace_list_files", {"relative_path": ".", "limit": 2}),
                ),
            ),
            ModelResponse("Next turn still works.", FinishReason.STOP),
        ]
    )
    session = _agent(provider, tmp_path, max_tool_rounds=1)

    exhausted = session.ask("loop")
    continued = session.ask("continue")

    assert exhausted.finish_reason is FinishReason.ERROR
    assert "tool-call budget" in exhausted.content
    assert continued.content == "Next turn still works."
    assert len(provider.requests) == 3


def test_agent_tool_round_guard_stops_infinite_read_only_repairs(tmp_path: Path) -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("bad1", "workspace_read_text", {"relative_path": ""}),),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("bad2", "workspace_read_text", {"relative_path": ""}),),
            ),
        ]
    )
    session = _agent(provider, tmp_path, max_tool_rounds=1)

    result = session.ask("loop on bad read")

    assert result.finish_reason is FinishReason.ERROR
    assert "tool-call budget" in result.content
    assert len(provider.requests) == 2
    assert result.tool_activity == ({"name": "workspace_read_text", "status": "error"},)


def test_gazebo_tool_schema_exposure_is_capability_gated(tmp_path: Path) -> None:
    available = _gazebo_agent(ScriptedProvider([]), tmp_path)
    unavailable = HephaestusAgentSession(
        provider=ScriptedProvider([]),
        startup_directory=tmp_path,
        tool_executor=WorkspaceToolExecutor(tmp_path),
        capabilities=_inactive_gazebo_capabilities(),
    )

    available_names = {tool.name for tool in available._tool_definitions()}
    unavailable_names = {tool.name for tool in unavailable._tool_definitions()}

    assert {
        "gazebo_validate_sdf",
        "gazebo_launch",
        "gazebo_status",
        "gazebo_read_logs",
        "gazebo_stop",
    } <= available_names
    assert all(not name.startswith("gazebo_") for name in unavailable_names)
    schema = next(tool for tool in available._tool_definitions() if tool.name == "gazebo_launch")
    assert schema.input_schema["additionalProperties"] is False
    assert schema.input_schema["required"] == ["relative_path"]
    assert "gz sim -r" in schema.description
    assert "gz=missing" in unavailable.dynamic_context()


def test_gazebo_validate_sdf_returns_valid_result(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    world.write_text(_minimal_sdf(), encoding="utf-8")
    calls: list[tuple[Sequence[str], dict[str, object]]] = []
    session = _gazebo_agent(ScriptedProvider([]), tmp_path, run_command=_valid_sdf_runner(calls))

    result = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),
        allow_mutation=True,
    )

    assert result["status"] == "ok"
    assert result["result"]["valid"] is True
    assert result["result"]["exit_code"] == 0
    assert result["result"]["relative_path"] == "world.sdf"
    assert result["result"]["gazebo_version"] == "8.11.0"
    assert calls[0][0] == ("gz", "sdf", "-k", str(world))
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert calls[0][1]["timeout"] == 15.0
    assert calls[0][1]["env"]["GZ_SIM_RESOURCE_PATH"] == str(tmp_path)
    assert calls[0][1]["env"]["SDF_PATH"] == str(tmp_path)
    assert "GROQ_API_KEY" not in calls[0][1]["env"]


def test_gazebo_validate_resolves_workspace_relative_meshes(tmp_path: Path) -> None:
    mesh = tmp_path / "generated" / "cad" / "robot" / "stl" / "chassis.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("solid x\nendsolid x\n", encoding="utf-8")
    world = tmp_path / "world.sdf"
    world.write_text(
        """<sdf version="1.10">
  <world name="forge">
    <model name="robot">
      <link name="base">
        <visual name="mesh">
          <geometry><mesh><uri>generated/cad/robot/stl/chassis.stl</uri></mesh></geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )
    session = _gazebo_agent(ScriptedProvider([]), tmp_path)

    result = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),
        allow_mutation=True,
    )

    assert result["status"] == "ok"
    assert result["result"]["valid"] is True
    assert result["result"]["local_meshes"] == [
        {
            "uri": "generated/cad/robot/stl/chassis.stl",
            "relative_path": "generated/cad/robot/stl/chassis.stl",
        }
    ]


def test_gazebo_missing_outside_and_symlink_meshes_block_validation_and_launch(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside_mesh.stl"
    outside.write_text("solid x\nendsolid x\n", encoding="utf-8")
    (tmp_path / "link_mesh.stl").symlink_to(outside)
    missing_world = tmp_path / "missing.sdf"
    missing_world.write_text(
        (
            '<sdf version="1.10"><world name="forge"><model name="robot"><link name="base">\n'
            '<visual name="mesh"><geometry><mesh><uri>generated/cad/robot/stl/missing.stl'
            "</uri></mesh></geometry></visual>\n"
            "</link></model></world></sdf>\n"
        ),
        encoding="utf-8",
    )
    symlink_world = tmp_path / "symlink.sdf"
    symlink_world.write_text(
        """<sdf version="1.10"><world name="forge"><model name="robot"><link name="base">
<visual name="mesh"><geometry><mesh><uri>link_mesh.stl</uri></mesh></geometry></visual>
</link></model></world></sdf>
""",
        encoding="utf-8",
    )
    absolute_world = tmp_path / "absolute.sdf"
    absolute_world.write_text(
        f"""<sdf version="1.10"><world name="forge"><model name="robot"><link name="base">
<visual name="mesh"><geometry><mesh><uri>{outside}</uri></mesh></geometry></visual>
</link></model></world></sdf>
""",
        encoding="utf-8",
    )
    popen = FakePopenFactory()
    session = _gazebo_agent(ScriptedProvider([]), tmp_path, popen_factory=popen)

    missing = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "missing.sdf"}),
        allow_mutation=True,
    )
    symlink = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "symlink.sdf"}),
        allow_mutation=True,
    )
    absolute = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "absolute.sdf"}),
        allow_mutation=True,
    )
    launch = session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "missing.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert missing["result"]["valid"] is False
    assert "missing" in missing["result"]["stderr"]
    assert symlink["result"]["valid"] is False
    assert "symlink" in symlink["result"]["stderr"]
    assert absolute["result"]["valid"] is False
    assert "escapes startup directory" in absolute["result"]["stderr"]
    assert launch["status"] == "error"
    assert "missing" in launch["error"]
    assert popen.calls == []


def test_invalid_sdf_is_returned_to_provider_for_repair(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text("<sdf>", encoding="utf-8")

    def invalid(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="line 1 bad\n")

    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("v1", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),),
            ),
            ModelResponse("I will repair the SDF.", FinishReason.STOP),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, run_command=invalid)

    result = session.ask("validate world.sdf")

    assert result.content == "I will repair the SDF."
    tool_result = json.loads(provider.requests[1].messages[3].content)
    assert tool_result["status"] == "ok"
    assert tool_result["result"]["valid"] is False
    assert "line 1 bad" in tool_result["result"]["stderr"]


def test_scripted_sdf_repair_uses_targeted_replacements_then_validates(
    tmp_path: Path,
) -> None:
    sun_include = """    <include>
      <uri>model://sun</uri>
    </include>
"""
    sun_light = """    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
"""
    ground_include = """    <include>
      <uri>model://ground_plane</uri>
    </include>
"""
    ground_model = """    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </visual>
      </link>
    </model>
"""
    world = tmp_path / "forge_world.sdf"
    world.write_text(
        f"""<sdf version="1.10">
  <world name="forge">
{sun_include}{ground_include}  </world>
</sdf>
""",
        encoding="utf-8",
    )
    validations: list[Sequence[str]] = []

    def validate(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        validations.append(tuple(argv))
        text = Path(argv[-1]).read_text(encoding="utf-8")
        if "model://sun" in text or "model://ground_plane" in text:
            return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="external model")
        return subprocess.CompletedProcess(tuple(argv), 0, stdout="valid\n", stderr="")

    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "read",
                        "workspace_read_text",
                        {"relative_path": "forge_world.sdf"},
                    ),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "replace_sun",
                        "workspace_replace_text",
                        {
                            "relative_path": "forge_world.sdf",
                            "old_text": sun_include,
                            "new_text": sun_light,
                        },
                    ),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "replace_ground",
                        "workspace_replace_text",
                        {
                            "relative_path": "forge_world.sdf",
                            "old_text": ground_include,
                            "new_text": ground_model,
                        },
                    ),
                ),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "validate",
                        "gazebo_validate_sdf",
                        {"relative_path": "forge_world.sdf"},
                    ),
                ),
            ),
            ModelResponse("The repaired SDF validates.", FinishReason.STOP),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, run_command=validate)

    result = session.ask("repair external model references", confirm=lambda preview: True)

    assert result.content == "The repaired SDF validates."
    assert [item["name"] for item in result.tool_activity] == [
        "workspace_read_text",
        "workspace_replace_text",
        "workspace_replace_text",
        "gazebo_validate_sdf",
    ]
    assert "workspace_write_text" not in {item["name"] for item in result.tool_activity}
    repaired = world.read_text(encoding="utf-8")
    assert "model://sun" not in repaired
    assert "model://ground_plane" not in repaired
    assert '<light name="sun" type="directional">' in repaired
    assert '<model name="ground_plane">' in repaired
    first_replace_result = json.loads(provider.requests[2].messages[-1].content)
    assert first_replace_result["result"]["committed"] is True
    assert "after_fingerprint" in first_replace_result["result"]
    final_validate = json.loads(provider.requests[4].messages[-1].content)
    assert final_validate["result"]["valid"] is True
    assert len(validations) == 1


def test_repaired_sdf_can_be_written_then_validated_successfully(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text("<sdf>", encoding="utf-8")
    runner = _invalid_then_valid_runner()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("v1", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        "w1",
                        "workspace_write_text",
                        {"relative_path": "world.sdf", "content": _minimal_sdf()},
                    ),
                    ToolCall("v2", "gazebo_validate_sdf", {"relative_path": "world.sdf"}),
                ),
            ),
            ModelResponse("The repaired SDF validates.", FinishReason.STOP),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, run_command=runner)

    result = session.ask("repair and validate", confirm=lambda preview: True)

    assert result.content == "The repaired SDF validates."
    assert len(runner.calls) == 2
    repaired_result = json.loads(provider.requests[2].messages[-1].content)
    assert repaired_result["result"]["valid"] is True


def test_gazebo_launch_confirmation_accepted_uses_fixed_argv_and_env(tmp_path: Path) -> None:
    world = tmp_path / "world.sdf"
    world.write_text(_minimal_sdf(), encoding="utf-8")
    popen = FakePopenFactory()
    session = _gazebo_agent(
        ScriptedProvider([]),
        tmp_path,
        popen_factory=popen,
        env={
            "DISPLAY": ":99",
            "GZ_IP": "10.0.0.9",
            "GROQ_API_KEY": "gsk_should_not_leak",
            "UNRELATED_SETTING": "filtered",
        },
    )
    previews: list[dict[str, object]] = []

    result = session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: previews.append(preview) or True,
    )

    assert result["status"] == "ok"
    assert result["result"]["active"] is True
    assert result["result"]["pid"] == popen.process.pid
    assert previews[0]["result"]["relative_path"] == "world.sdf"
    assert previews[0]["result"]["gui"] is True
    assert previews[0]["result"]["argv_summary"] == ["gz", "sim", "-r", "world.sdf"]
    argv, kwargs = popen.calls[0]
    assert argv == ("gz", "sim", "-r", str(world))
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.PIPE
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["GZ_IP"] == "127.0.0.1"
    assert kwargs["env"]["DISPLAY"] == ":99"
    assert "GROQ_API_KEY" not in kwargs["env"]
    assert "UNRELATED_SETTING" not in kwargs["env"]
    assert result["result"]["env"] == {"GZ_IP": "127.0.0.1"}
    assert popen.process.wait_calls == 0


def test_gazebo_launch_inherits_valid_xauthority_and_wayland_session_values(
    tmp_path: Path,
) -> None:
    world = tmp_path / "world.sdf"
    world.write_text(_minimal_sdf(), encoding="utf-8")
    xauthority = tmp_path / ".Xauthority"
    xauthority.write_text("cookie", encoding="utf-8")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    bus = runtime / "bus"
    bus.write_text("", encoding="utf-8")
    popen = FakePopenFactory()
    session = _gazebo_agent(
        ScriptedProvider([]),
        tmp_path,
        popen_factory=popen,
        env={
            "DISPLAY": ":99",
            "XAUTHORITY": str(xauthority),
            "WAYLAND_DISPLAY": "wayland-0",
            "XDG_RUNTIME_DIR": str(runtime),
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
            "XDG_SESSION_TYPE": "wayland",
            "LD_LIBRARY_PATH": "/opt/ros/jazzy/lib",
            "GZ_SIM_RESOURCE_PATH": "/opt/ros/jazzy/share",
            "QT_QPA_PLATFORM": "xcb",
            "UNRELATED_SETTING": "filtered",
        },
    )

    result = session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "ok"
    env = popen.calls[0][1]["env"]
    assert env["DISPLAY"] == ":99"
    assert env["XAUTHORITY"] == str(xauthority)
    assert env["WAYLAND_DISPLAY"] == "wayland-0"
    assert env["XDG_RUNTIME_DIR"] == str(runtime)
    assert env["DBUS_SESSION_BUS_ADDRESS"] == f"unix:path={bus}"
    assert env["XDG_SESSION_TYPE"] == "wayland"
    assert env["LD_LIBRARY_PATH"] == "/opt/ros/jazzy/lib"
    assert env["GZ_SIM_RESOURCE_PATH"] == str(tmp_path)
    assert env["SDF_PATH"] == str(tmp_path)
    assert env["IGN_GAZEBO_RESOURCE_PATH"] == str(tmp_path)
    assert env["GZ_IP"] == "127.0.0.1"
    assert "QT_QPA_PLATFORM" not in env
    assert "UNRELATED_SETTING" not in env
    assert result["result"]["env"] == {"GZ_IP": "127.0.0.1"}


def test_gazebo_launch_without_display_or_wayland_fails_before_spawn_after_validation(
    tmp_path: Path,
) -> None:
    world = tmp_path / "world.sdf"
    world.write_text(_minimal_sdf(), encoding="utf-8")
    calls: list[tuple[Sequence[str], dict[str, object]]] = []
    popen = FakePopenFactory()
    session = _gazebo_agent(
        ScriptedProvider([]),
        tmp_path,
        run_command=_valid_sdf_runner(calls),
        popen_factory=popen,
        env={"PATH": "/usr/bin", "GZ_IP": "10.0.0.9"},
    )

    result = session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert result["status"] == "error"
    assert result["tool"] == "gazebo_launch"
    assert result["evidence"] == {"code": "gui_unavailable"}
    assert "graphical session is unavailable" in result["error"]
    assert "headless" in result["error"]
    assert calls[0][0] == ("gz", "sdf", "-k", str(world))
    assert popen.calls == []


def test_provider_cannot_inject_or_override_gazebo_display_environment(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    popen = FakePopenFactory()
    session = _gazebo_agent(
        ScriptedProvider([]),
        tmp_path,
        popen_factory=popen,
        env={"DISPLAY": ":99", "WAYLAND_DISPLAY": "wayland-0"},
    )

    injected = session.tool_executor.execute(
        ToolCall(
            "launch",
            "gazebo_launch",
            {
                "relative_path": "world.sdf",
                "DISPLAY": ":1",
                "XAUTHORITY": "/tmp/fake",
                "env": {"DISPLAY": ":1", "GZ_IP": "1.2.3.4"},
            },
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert injected["status"] == "error"
    assert injected["error"] == "invalid arguments"
    assert popen.calls == []


def test_gazebo_launch_declined_is_terminal_and_starts_no_process(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    popen = FakePopenFactory()
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),),
            ),
            ModelResponse("Gazebo launched.", FinishReason.STOP),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, popen_factory=popen)

    result = session.ask("open gazebo", confirm=lambda preview: False)

    assert result.finish_reason is FinishReason.ERROR
    assert "launch was previewed but not approved" in result.content
    assert popen.calls == []
    assert len(provider.requests) == 1


def test_active_gazebo_process_survives_tool_round_exhaustion(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    process = FakeProcess()
    popen = FakePopenFactory(process)
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),),
            ),
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("status", "gazebo_status", {}),),
            ),
        ]
    )
    session = _gazebo_agent(
        provider,
        tmp_path,
        popen_factory=popen,
        max_tool_rounds=1,
    )

    result = session.ask("launch then inspect", confirm=lambda preview: True)
    status = session.tool_executor.gazebo_workbench.status()

    assert result.finish_reason is FinishReason.ERROR
    assert "tool-call budget" in result.content
    assert result.tool_activity == ({"name": "gazebo_launch", "status": "ok"},)
    assert len(popen.calls) == 1
    assert status["active"] is True
    assert process.poll() is None


def test_active_gazebo_process_survives_provider_rate_limit_after_launch(
    tmp_path: Path,
) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    process = FakeProcess()
    popen = FakePopenFactory(process)
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),),
            ),
            ProviderRateLimitError(
                "AI provider rate limit reached",
                ProviderErrorDetails(status_code=429, provider_code="rate_limit_exceeded"),
            ),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, popen_factory=popen)

    with pytest.raises(ProviderRateLimitError):
        session.ask("launch then summarize", confirm=lambda preview: True)

    assert len(popen.calls) == 1
    assert session.tool_executor.gazebo_workbench.status()["active"] is True
    assert process.poll() is None


def test_active_gazebo_process_survives_completion_failure_after_launch(
    tmp_path: Path,
) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    process = FakeProcess()
    popen = FakePopenFactory(process)
    provider = ScriptedProvider(
        [
            ModelResponse(
                "",
                FinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),),
            ),
            ModelResponse("", FinishReason.STOP),
        ]
    )
    session = _gazebo_agent(provider, tmp_path, popen_factory=popen)

    with pytest.raises(AgentLoopError, match="no assistant text"):
        session.ask("launch then malformed completion", confirm=lambda preview: True)

    assert len(popen.calls) == 1
    assert session.tool_executor.gazebo_workbench.status()["active"] is True
    assert process.poll() is None


def test_gazebo_launch_validation_failure_and_changed_file_start_no_process(
    tmp_path: Path,
) -> None:
    world = tmp_path / "world.sdf"
    world.write_text("<sdf>", encoding="utf-8")
    popen = FakePopenFactory()

    def invalid(argv: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(tuple(argv), 1, stdout="", stderr="invalid")

    session = _gazebo_agent(
        ScriptedProvider([]), tmp_path, run_command=invalid, popen_factory=popen
    )
    failed = session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    assert failed["status"] == "error"
    assert "validation failed" in failed["error"]
    assert popen.calls == []

    world.write_text(_minimal_sdf(), encoding="utf-8")
    changed_session = _gazebo_agent(ScriptedProvider([]), tmp_path, popen_factory=popen)

    def mutate_and_accept(preview: dict[str, object]) -> bool:
        world.write_text(_minimal_sdf() + "<!-- changed -->\n", encoding="utf-8")
        return True

    changed = changed_session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=mutate_and_accept,
    )
    assert changed["status"] == "error"
    assert "changed after launch preview" in changed["error"]
    assert popen.calls == []


def test_gazebo_rejects_injected_args_outside_paths_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    outside = tmp_path.parent / "outside.sdf"
    outside.write_text(_minimal_sdf(), encoding="utf-8")
    (tmp_path / "link.sdf").symlink_to(outside)
    session = _gazebo_agent(ScriptedProvider([]), tmp_path)

    injected = session.tool_executor.execute(
        ToolCall(
            "launch",
            "gazebo_launch",
            {
                "relative_path": "world.sdf",
                "argv": ["--render-engine", "ogre"],
                "env": {"GZ_IP": "1.2.3.4"},
            },
        ),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    traversal = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "../outside.sdf"}),
        allow_mutation=True,
    )
    symlink = session.tool_executor.execute(
        ToolCall("validate", "gazebo_validate_sdf", {"relative_path": "link.sdf"}),
        allow_mutation=True,
    )

    assert injected["status"] == "error"
    assert injected["error"] == "invalid arguments"
    assert traversal["status"] == "error"
    assert "parent traversal" in traversal["error"]
    assert symlink["status"] == "error"
    assert "escapes startup directory" in symlink["error"]


def test_gazebo_allows_only_one_active_owned_simulation_per_session(tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    popen = FakePopenFactory()
    session = _gazebo_agent(ScriptedProvider([]), tmp_path, popen_factory=popen)

    first = session.tool_executor.execute(
        ToolCall("launch1", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    second = session.tool_executor.execute(
        ToolCall("launch2", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )

    assert first["status"] == "ok"
    assert second["status"] == "error"
    assert "already active" in second["error"]
    assert len(popen.calls) == 1


def test_gazebo_status_running_exited_logs_and_repeated_stop(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "world.sdf").write_text(_minimal_sdf(), encoding="utf-8")
    process = FakeProcess(stdout="ready\n", stderr="gsk_secret_value_should_hide\n")
    popen = FakePopenFactory(process)
    killed: list[tuple[int, signal.Signals]] = []

    def fake_killpg(pid: int, sig: signal.Signals) -> None:
        killed.append((pid, sig))
        process.returncode = -int(sig)

    monkeypatch.setattr("hephaestus.execution.os.killpg", fake_killpg)
    session = _gazebo_agent(ScriptedProvider([]), tmp_path, popen_factory=popen)
    session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    time.sleep(0.05)

    status = session.tool_executor.execute(
        ToolCall("status", "gazebo_status", {}),
        allow_mutation=True,
    )
    logs = session.tool_executor.execute(
        ToolCall("logs", "gazebo_read_logs", {"max_lines": 100}),
        allow_mutation=True,
    )
    stopped = session.tool_executor.execute(
        ToolCall("stop", "gazebo_stop", {}), allow_mutation=True
    )
    stopped_again = session.tool_executor.execute(
        ToolCall("stop2", "gazebo_stop", {}), allow_mutation=True
    )

    assert status["result"]["active"] is True
    assert status["result"]["state"] in {"starting", "running"}
    assert logs["result"]["lines"]
    assert "[REDACTED]" in "\n".join(logs["result"]["lines"])
    assert "gsk_secret_value_should_hide" not in "\n".join(logs["result"]["lines"])
    assert stopped["result"]["stopped"] is True
    assert stopped["result"]["owned_process_exists"] is False
    assert stopped_again["result"]["stopped"] is False
    assert killed == [(process.pid, signal.SIGTERM)]

    exited_process = FakeProcess(returncode=0)
    exited_session = _gazebo_agent(
        ScriptedProvider([]), tmp_path, popen_factory=FakePopenFactory(exited_process)
    )
    exited_session.tool_executor.execute(
        ToolCall("launch", "gazebo_launch", {"relative_path": "world.sdf"}),
        allow_mutation=True,
        confirm=lambda preview: True,
    )
    exited = exited_session.tool_executor.execute(
        ToolCall("status", "gazebo_status", {}), allow_mutation=True
    )
    assert exited["result"]["active"] is False
    assert exited["result"]["state"] == "exited"
    assert exited["result"]["exit_code"] == 0


def test_gazebo_log_read_without_simulation_is_structured(tmp_path: Path) -> None:
    session = _gazebo_agent(ScriptedProvider([]), tmp_path)

    result = session.tool_executor.execute(
        ToolCall("logs", "gazebo_read_logs", {"max_lines": 10}),
        allow_mutation=True,
    )

    assert result["status"] == "ok"
    assert result["result"]["active"] is False
    assert result["result"]["lines"] == []
    assert "no Hephaestus-owned" in result["result"]["message"]


def test_default_agent_exit_eof_and_ctrl_c_cleanup_owned_gazebo(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(
        "hephaestus.local_workbench.collect_capabilities", _active_gazebo_capabilities
    )

    closed: list[bool] = []

    class ClosingWorkbench(GazeboWorkbench):
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("hephaestus.local_workbench.GazeboWorkbench", ClosingWorkbench)
    monkeypatch.setattr(cli, "_create_provider", lambda config: ScriptedProvider([]))

    exit_result = runner.invoke(cli.app, [], input="/exit\n")
    eof_result = runner.invoke(cli.app, [], input="")

    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    cli._run_default_agent_shell()

    assert exit_result.exit_code == 0
    assert eof_result.exit_code == 0
    assert len(closed) == 3


def test_gazebo_stop_without_owned_process_does_not_kill_unrelated_process(
    monkeypatch, tmp_path: Path
) -> None:
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        "hephaestus.execution.os.killpg", lambda pid, sig: killed.append((pid, sig))
    )
    session = _gazebo_agent(ScriptedProvider([]), tmp_path)

    result = session.tool_executor.execute(ToolCall("stop", "gazebo_stop", {}), allow_mutation=True)

    assert result["status"] == "ok"
    assert result["result"]["state"] == "inactive"
    assert killed == []


def test_default_agent_clear_status_and_eof_exit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr(
        "hephaestus.local_workbench.collect_capabilities", lambda: CapabilityReport(())
    )
    fake = ScriptedProvider([ModelResponse("Before clear.", FinishReason.STOP)])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    result = runner.invoke(cli.app, [], input="/help\n/status\nask\n/clear\n/status\n")

    assert result.exit_code == 0
    assert "/help" in result.output
    assert "/clear" in result.output
    assert "/status" in result.output
    assert "/exit" in result.output
    assert f"workspace={tmp_path}" in result.output
    assert "Local workbench status." in result.output
    assert "Before clear." in result.output
    assert "Conversation cleared." in result.output


def test_default_agent_ctrl_c_exit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret")
    monkeypatch.setattr("hephaestus.ai.agent.collect_capabilities", lambda: CapabilityReport(()))
    fake = ScriptedProvider([])
    monkeypatch.setattr(cli, "_create_provider", lambda config: fake)

    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    cli._run_default_agent_shell()

    assert fake.requests == []


def test_explicit_help_and_existing_json_command_still_work_without_ansi(tmp_path: Path) -> None:
    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "doctor" in help_result.output

    json_result = runner.invoke(cli.app, ["ai", "status", "--json"], env={"GROQ_API_KEY": ""})
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["provider"] == "groq"
    assert payload["default_agent_max_tool_rounds"] == 16
    assert "\x1b[" not in json_result.output


def test_ai_status_reports_configured_default_agent_tool_round_limit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEPHAESTUS_CONFIG_DIR", str(tmp_path / "config"))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "ai.json").write_text(
        json.dumps(
            {
                "provider": "groq",
                "model": "fake-model",
                "timeout_seconds": 30.0,
                "default_agent_max_tool_rounds": 64,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["ai", "status", "--json"], env={"GROQ_API_KEY": ""})

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["default_agent_max_tool_rounds"] == 64


def _validation_text(result: dict[str, object]) -> str:
    evidence = result.get("evidence")
    assert isinstance(evidence, list)
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in evidence
        if isinstance(item, dict)
    )
