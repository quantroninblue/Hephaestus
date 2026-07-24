from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hephaestus import cli
from hephaestus.artifacts import ArtifactStager
from hephaestus.backends.cadquery_backend import FakeCadAdapter, generate_cad_artifacts
from hephaestus.backends.ros_gazebo import GENERATOR_VERSION, generate_ros_gazebo_artifacts
from hephaestus.execution import ConstrainedRunner, ExecutionError
from hephaestus.mobile_robot import DESIGN_EXTENSION_KEY
from hephaestus.model import EngineeringProject
from hephaestus.persistence import save_project

runner = CliRunner()


class FakeProcess:
    def __init__(
        self,
        stdout: str = "ok",
        stderr: str = "",
        returncode: int = 0,
        *,
        timeout_once: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timeout_once = timeout_once
        self.pid = 999_999_999
        self.killed = False
        self.calls = 0

    def communicate(self, timeout: float | None = None):
        self.calls += 1
        if self.timeout_once and timeout is not None and self.calls == 1:
            raise subprocess.TimeoutExpired(["fake"], timeout)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True


def _mobile_robot_project() -> EngineeringProject:
    ids = {
        "robot_entity": "entity:execution_bot",
        "chassis_body": "body:execution_bot_chassis",
        "chassis_geometry": "geometry:execution_bot_chassis",
        "left_wheel_body": "body:execution_bot_left_wheel",
        "right_wheel_body": "body:execution_bot_right_wheel",
        "wheel_geometry": "geometry:execution_bot_wheel",
        "caster_body": "body:execution_bot_caster",
        "caster_geometry": "geometry:execution_bot_caster",
    }
    values = {
        "chassis_length": {"value": 0.7, "unit": "m", "dimension": "length"},
        "chassis_width": {"value": 0.48, "unit": "m", "dimension": "length"},
        "chassis_height": {"value": 0.22, "unit": "m", "dimension": "length"},
        "wheel_radius": {"value": 0.08, "unit": "m", "dimension": "length"},
        "wheel_width": {"value": 0.04, "unit": "m", "dimension": "length"},
        "wheel_separation": {"value": 0.42, "unit": "m", "dimension": "length"},
        "caster_radius": {"value": 0.035, "unit": "m", "dimension": "length"},
        "ground_clearance": {"value": 0.04, "unit": "m", "dimension": "length"},
        "chassis_mass": {"value": 10.0, "unit": "kg", "dimension": "mass"},
        "wheel_mass": {"value": 0.8, "unit": "kg", "dimension": "mass"},
        "caster_mass": {"value": 0.3, "unit": "kg", "dimension": "mass"},
        "lidar_update_rate": {"value": 10.0, "unit": "Hz", "dimension": "frequency"},
        "imu_update_rate": {"value": 100.0, "unit": "Hz", "dimension": "frequency"},
        "simulation_update_rate": {"value": 100.0, "unit": "Hz", "dimension": "frequency"},
    }
    extension = {
        "schema": "hephaestus.mobile_robot.v1",
        "supported_family": "differential_drive",
        "robot_name": "Execution Bot",
        "mission_summary": "Inspect warehouse aisles offline.",
        "values": values,
        "ids": ids,
    }
    base = EngineeringProject.empty(name="Execution Bot", project_id="project:execution_bot")
    return replace(base, extensions={DESIGN_EXTENSION_KEY: extension})


def _published_project(tmp_path: Path) -> tuple[EngineeringProject, Path, Path]:
    project_dir = tmp_path / "robot_project"
    project = _mobile_robot_project()
    save_project(project, project_dir)
    stager = ArtifactStager(project_dir, project, backend_versions={"cadquery": "fake"})
    cad_records = generate_cad_artifacts(project, stager, adapter=FakeCadAdapter())
    ros_records = generate_ros_gazebo_artifacts(project, stager)
    stager.publish(
        (*cad_records, *ros_records),
        generator_versions={"hephaestus.ros_gazebo_backend": GENERATOR_VERSION},
    )
    return project, project_dir, project_dir / "generated" / "ros2_ws"


def test_build_ros_gazebo_uses_fixed_argv_sanitized_env_and_project_cwd(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def popen(argv, **kwargs):
        calls.append((tuple(argv), kwargs))
        install_dir = Path(str(kwargs["cwd"])) / "install"
        install_dir.mkdir(parents=True, exist_ok=True)
        (install_dir / "setup.bash").write_text("# fake colcon setup\n", encoding="utf-8")
        return FakeProcess(stdout="built")

    runner = ConstrainedRunner(
        project_dir=project_dir,
        env={"PATH": "/usr/bin", "GROQ_API_KEY": "gsk_secret_should_not_pass"},
        timeout_seconds=3.0,
        popen_factory=popen,
    )

    result = runner.build_ros_gazebo()

    assert result.succeeded
    assert result.stdout == "built"
    argv, kwargs = calls[0]
    assert argv[0] == "colcon"
    assert "build" in argv[:4]
    assert "--base-paths" in argv
    assert str(workspace_dir / "src") in argv
    assert kwargs["cwd"] == str(workspace_dir)
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert kwargs["env"] == {"PATH": "/usr/bin"}
    assert "shell" not in kwargs



def test_build_ros_gazebo_clears_stale_build_outputs_before_colcon(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    for dirname in ("build", "install", "log"):
        stale_dir = workspace_dir / dirname
        stale_dir.mkdir(parents=True, exist_ok=True)
        (stale_dir / "stale.txt").write_text("old generated build output", encoding="utf-8")

    def popen(argv, **kwargs):
        del argv
        cwd = Path(str(kwargs["cwd"]))
        assert not (cwd / "build" / "stale.txt").exists()
        assert not (cwd / "install" / "stale.txt").exists()
        assert not (cwd / "log" / "stale.txt").exists()
        install_dir = cwd / "install"
        install_dir.mkdir(parents=True, exist_ok=True)
        (install_dir / "setup.bash").write_text("# fake colcon setup\n", encoding="utf-8")
        return FakeProcess(stdout="rebuilt")

    result = ConstrainedRunner(project_dir, popen_factory=popen).build_ros_gazebo()

    assert result.succeeded
    assert result.stdout == "rebuilt"


def test_simulate_ros_gazebo_uses_generated_launch_target_and_world(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def popen(argv, **kwargs):
        del kwargs
        calls.append(tuple(argv))
        return FakeProcess(stdout="sim")

    result = ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo()

    assert result.succeeded
    argv = calls[0]
    assert argv[:4] == ("ros2", "launch", "execution_bot_bringup", "gazebo.launch.py")
    assert argv[4].startswith("world:=")
    assert (
        str(workspace_dir / "src/execution_bot_simulation/worlds/warehouse_inspection.sdf")
        in argv[4]
    )




def test_gui_and_headless_simulation_use_distinct_generated_launch_files(
    tmp_path: Path,
) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def popen(argv, **kwargs):
        del kwargs
        calls.append(tuple(argv))
        return FakeProcess(stdout="sim")

    runner = ConstrainedRunner(project_dir, popen_factory=popen)

    gui = runner.simulate_ros_gazebo(headless=False)
    headless = runner.simulate_ros_gazebo(headless=True)

    assert gui.succeeded
    assert headless.succeeded
    assert calls[0][:4] == ("ros2", "launch", "execution_bot_bringup", "gazebo.launch.py")
    assert calls[1][:4] == (
        "ros2",
        "launch",
        "execution_bot_bringup",
        "gazebo_headless.launch.py",
    )
    assert calls[0][4] == calls[1][4]
    assert calls[0][4].startswith("world:=")
    world_path = workspace_dir / "src/execution_bot_simulation/worlds/warehouse_inspection.sdf"
    assert str(world_path) in calls[0][4]

def test_simulate_rejects_stale_generated_artifacts_before_launch(tmp_path: Path) -> None:
    project, project_dir, _workspace_dir = _published_project(tmp_path)
    save_project(replace(project, revision=project.revision + 1), project_dir, overwrite=True)

    def popen(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError((argv, kwargs))

    with pytest.raises(ExecutionError, match="stale"):
        ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo()


def test_execution_refuses_missing_and_escaping_workspace_paths(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "src").mkdir()

    runner = ConstrainedRunner(project_dir)

    with pytest.raises(ExecutionError, match="path does not exist"):
        runner.build_ros_gazebo(project_dir / "missing_ws")
    with pytest.raises(ExecutionError, match="escapes project scope"):
        runner.build_ros_gazebo(outside)


def test_execution_rejects_symlink_paths_before_resolution(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    link = project_dir / "workspace_link"
    link.symlink_to(workspace_dir, target_is_directory=True)

    with pytest.raises(ExecutionError, match="symlink"):
        ConstrainedRunner(project_dir).build_ros_gazebo(link)


def test_timeout_cleans_process_group_and_redacts_captured_output(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)
    process = FakeProcess(
        stdout="token=supersecretvalue should disappear",
        stderr="api_key=supersecretvalue should disappear",
        returncode=0,
        timeout_once=True,
    )

    def popen(argv, **kwargs):
        del argv, kwargs
        return process

    result = ConstrainedRunner(
        project_dir, timeout_seconds=0.01, popen_factory=popen
    ).build_ros_gazebo()

    assert result.returncode == 124
    assert result.timed_out is True
    assert process.killed is True
    assert "supersecretvalue" not in result.stdout
    assert "supersecretvalue" not in result.stderr
    assert "[REDACTED]" in result.stdout
    assert "[REDACTED]" in result.stderr


def test_missing_controller_packages_are_reported_as_failed_execution(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")

    def popen(argv, **kwargs):
        del argv, kwargs
        return FakeProcess(stderr="package diff_drive_controller not found", returncode=1)

    result = ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo()

    assert not result.succeeded
    assert result.returncode == 1
    assert "diff_drive_controller not found" in result.stderr


def test_timeout_escalates_to_kill_when_process_ignores_sigterm(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)

    class StubbornProcess(FakeProcess):
        def communicate(self, timeout: float | None = None):
            self.calls += 1
            if self.calls <= 2:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return self.stdout, self.stderr

    process = StubbornProcess(stdout="terminated", timeout_once=True)

    def popen(argv, **kwargs):
        del argv, kwargs
        return process

    result = ConstrainedRunner(
        project_dir, timeout_seconds=0.01, popen_factory=popen
    ).build_ros_gazebo()

    assert result.returncode == 124
    assert result.timed_out is True
    assert process.calls == 3


def test_timeout_returns_result_when_process_never_exits_after_kill(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)

    class NeverExitsProcess(FakeProcess):
        def communicate(self, timeout: float | None = None):
            self.calls += 1
            raise subprocess.TimeoutExpired(["fake"], timeout)

    process = NeverExitsProcess()

    def popen(argv, **kwargs):
        del argv, kwargs
        return process

    result = ConstrainedRunner(
        project_dir, timeout_seconds=0.01, popen_factory=popen
    ).build_ros_gazebo()

    assert result.returncode == 124
    assert result.timed_out is True
    assert process.killed is True
    assert process.calls == 3
    assert "did not exit after SIGKILL" in result.stderr


def test_simulation_env_overlay_uses_runner_env_and_sets_isolation_values(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def popen(argv, **kwargs):
        del argv
        calls.append(kwargs)
        return FakeProcess(stdout="sim")

    result = ConstrainedRunner(
        project_dir,
        env={
            "AMENT_PREFIX_PATH": "/runner/ament",
            "GZ_CONFIG_PATH": "/runner/gz",
            "PYTHONPATH": "/runner/python",
        },
        popen_factory=popen,
    ).simulate_ros_gazebo()

    assert result.succeeded
    env = calls[0]["env"]
    assert isinstance(env, dict)
    assert env["AMENT_PREFIX_PATH"].startswith(str(install))
    assert env["AMENT_PREFIX_PATH"].endswith("/runner/ament")
    assert env["PYTHONPATH"].endswith("/runner/python")
    assert env["SDF_PATH"] == str(project_dir / "generated" / "gazebo" / "models")
    assert env["GZ_CONFIG_PATH"] == "/runner/gz"
    assert env["GZ_PARTITION"].startswith("hephaestus_")
    assert env["GZ_IP"] == "127.0.0.1"
    assert env["ROS_DOMAIN_ID"] == "42"


def test_simulate_marks_ros_launch_child_errors_failed(tmp_path: Path) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")

    def popen(argv, **kwargs):
        del argv, kwargs
        return FakeProcess(
            stdout="[ERROR] [gazebo-1]: process has died unexpectedly",
            returncode=0,
        )

    result = ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo(headless=True)

    assert not result.succeeded
    assert result.returncode == 1
    assert "ros2 launch reported child process errors" in result.stderr


def test_smoke_test_requires_headless_mode(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)

    def popen(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError((argv, kwargs))

    with pytest.raises(ExecutionError, match="headless=True"):
        ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo(smoke_test=True)


def _successful_smoke_run(argv, **kwargs):
    assert kwargs["stdin"] == subprocess.DEVNULL
    if argv[:3] == ("ros2", "pkg", "prefix"):
        return subprocess.CompletedProcess(argv, 0, stdout="/opt/ros/jazzy", stderr="")
    if argv == ("ros2", "topic", "list"):
        _successful_smoke_run.stamp_count = 0
        _successful_smoke_run.odom_count = 0
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "/clock\n/scan\n/imu\n/joint_states\n"
                "/diff_drive_controller/cmd_vel\n/diff_drive_controller/odom\n"
            ),
            stderr="",
        )
    if argv[:5] == (
        "ros2",
        "service",
        "call",
        "/controller_manager/list_controllers",
        "controller_manager_msgs/srv/ListControllers",
    ):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "controller:\n"
                "- name: joint_state_broadcaster\n  state: active\n"
                "- name: diff_drive_controller\n  state: active\n"
            ),
            stderr="",
        )
    if argv[:6] == ("ros2", "topic", "echo", "--once", "--qos-reliability", "best_effort"):
        topic = argv[6]
        stamp_count = getattr(_successful_smoke_run, "stamp_count", 0) + 1
        _successful_smoke_run.stamp_count = stamp_count
        if topic == "/clock":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=f"clock:\n  sec: {stamp_count}\n  nanosec: 0\n",
                stderr="",
            )
        if topic == "/joint_states":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=f"header:\n  stamp:\n    sec: {stamp_count}\n    nanosec: 0\n",
                stderr="",
            )
        if topic == "/diff_drive_controller/odom":
            odom_count = getattr(_successful_smoke_run, "odom_count", 0) + 1
            _successful_smoke_run.odom_count = odom_count
            x = 0.0 if odom_count == 1 else 0.05
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=f"pose:\n  pose:\n    position:\n      x: {x}\n      y: 0.0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(argv, 0, stdout=f"message on {topic}\n", stderr="")
    if argv[:4] == ("ros2", "topic", "pub", "--times"):
        assert argv[10] == "/diff_drive_controller/cmd_vel"
        assert argv[11] == "geometry_msgs/msg/TwistStamped"
        assert "header: auto" in argv[12]
        return subprocess.CompletedProcess(argv, 0, stdout="published stamped velocity", stderr="")
    raise AssertionError(argv)


def test_headless_smoke_checks_full_runtime_package_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")
    checked: list[str] = []

    def fake_run(argv, **kwargs):
        if argv[:3] == ("ros2", "pkg", "prefix"):
            assert kwargs["timeout"] == 5.0
            checked.append(argv[3])
        return _successful_smoke_run(argv, **kwargs)

    process = FakeProcess(stdout="smoke")

    def popen(argv, **kwargs):
        del argv, kwargs
        return process

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo(
        headless=True, smoke_test=True
    )

    assert result.succeeded
    assert process.killed is True
    assert "clock messages received and increasing" in result.stdout
    assert "LiDAR message received" in result.stdout
    assert "IMU message received" in result.stdout
    assert "joint states received with advancing timestamps" in result.stdout
    assert "bounded stamped velocity command accepted" in result.stdout
    assert "observable odometry movement verified" in result.stdout
    assert "stop command accepted" in result.stdout
    assert checked == [
        "ros_gz_sim",
        "ros_gz_bridge",
        "gz_ros2_control",
        "robot_state_publisher",
        "xacro",
        "controller_manager",
        "joint_state_broadcaster",
        "diff_drive_controller",
    ]


def test_headless_smoke_reports_launch_stdout_and_controller_topic_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project, project_dir, workspace_dir = _published_project(tmp_path)
    install = workspace_dir / "install"
    install.mkdir(parents=True)
    (install / "setup.bash").write_text("# fake install\n", encoding="utf-8")

    def popen(argv, **kwargs):
        del argv, kwargs
        return FakeProcess(stdout="Robot initialized")

    monkeypatch.setattr(subprocess, "run", _successful_smoke_run)

    result = ConstrainedRunner(
        project_dir, timeout_seconds=6.0, popen_factory=popen
    ).simulate_ros_gazebo(headless=True, smoke_test=True)

    assert result.succeeded
    assert "controllers active: joint_state_broadcaster, diff_drive_controller" in result.stdout
    assert "odometry received" in result.stdout
    assert "--- launch stdout ---" in result.stdout
    assert "Robot initialized" in result.stdout


def test_headless_smoke_reports_missing_runtime_packages_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)

    def fake_run(argv, **kwargs):
        del kwargs
        returncode = 1 if argv[3] in {"ros_gz_bridge", "diff_drive_controller"} else 0
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr="missing")

    def popen(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError((argv, kwargs))

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ExecutionError) as exc_info:
        ConstrainedRunner(project_dir, popen_factory=popen).simulate_ros_gazebo(
            headless=True, smoke_test=True
        )

    message = str(exc_info.value)
    assert "runtime packages" in message
    assert "ros_gz_bridge" in message
    assert "diff_drive_controller" in message


def test_cli_simulate_json_requires_confirmation_without_human_output(tmp_path: Path) -> None:
    _project, project_dir, _workspace_dir = _published_project(tmp_path)

    result = runner.invoke(cli.app, ["simulate", str(project_dir), "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "confirmation_required"
    assert payload["command"] == "simulate_ros_gazebo"
    assert "Hephaestus Constrained Execution" not in result.output
    assert "[" not in result.output
