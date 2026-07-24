"""Constrained execution for application-owned ROS/Gazebo commands."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from hephaestus.artifacts import ArtifactError, ArtifactManifest, load_manifest
from hephaestus.backends.ros_gazebo import default_workspace_dir

MAX_CAPTURE_CHARS = 12_000
DEFAULT_TIMEOUT_SECONDS = 120.0
ALLOWED_ENV_KEYS = frozenset(
    {
        "AMENT_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "GZ_CONFIG_PATH",
        "GZ_IP",
        "GZ_PARTITION",
        "GZ_SIM_RESOURCE_PATH",
        "HOME",
        "IGN_GAZEBO_RESOURCE_PATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "RMW_IMPLEMENTATION",
        "ROS_DOMAIN_ID",
        "ROS_DISTRO",
        "ROS_PYTHON_VERSION",
        "ROS_VERSION",
        "SDF_PATH",
        "TERM",
    }
)
SECRET_PATTERN = re.compile(
    r"(?i)(gsk_[a-z0-9_=-]{8,}|sk-[a-z0-9_=-]{8,}|api[_-]?key[=:][^\s]+|token[=:][^\s]+)"
)


class ExecutionCommand(StrEnum):
    BUILD_ROS_GAZEBO = "build_ros_gazebo"
    SIMULATE_ROS_GAZEBO = "simulate_ros_gazebo"


class ExecutionError(RuntimeError):
    """Raised when constrained execution cannot start safely."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    command: ExecutionCommand
    argv: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command.value,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "succeeded": self.succeeded,
        }


class _PopenLike(Protocol):
    returncode: int | None
    pid: int

    def communicate(self, timeout: float | None = None) -> tuple[str | bytes, str | bytes]: ...

    def kill(self) -> None: ...


class _PopenFactory(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: object) -> _PopenLike: ...


@dataclass(frozen=True, slots=True)
class ConstrainedRunner:
    project_dir: Path
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    popen_factory: _PopenFactory = subprocess.Popen

    def build_ros_gazebo(self, workspace_dir: Path | None = None) -> ExecutionResult:
        workspace = _workspace(self.project_dir, workspace_dir)
        _fresh_artifact_manifest(self.project_dir)
        src_dir = _project_scoped_existing_path(self.project_dir, workspace / "src")
        _clear_workspace_build_outputs(self.project_dir, workspace)
        argv = (
            "colcon",
            "--log-base",
            str(workspace / "log"),
            "build",
            "--merge-install",
            "--base-paths",
            str(src_dir),
            "--build-base",
            str(workspace / "build"),
            "--install-base",
            str(workspace / "install"),
        )
        result = self._run(ExecutionCommand.BUILD_ROS_GAZEBO, argv, cwd=workspace)
        setup = workspace / "install" / "setup.bash"
        if result.succeeded and not setup.exists():
            return ExecutionResult(
                result.command,
                result.argv,
                result.cwd,
                1,
                result.stdout,
                _bounded_text(result.stderr + "\nexpected install/setup.bash was not created"),
                result.timed_out,
            )
        return result

    def simulate_ros_gazebo(
        self, workspace_dir: Path | None = None, *, headless: bool = False, smoke_test: bool = False
    ) -> ExecutionResult:
        workspace = _workspace(self.project_dir, workspace_dir)
        manifest = _fresh_artifact_manifest(self.project_dir)
        if smoke_test:
            if not headless:
                raise ExecutionError("smoke test execution requires headless=True")
            _check_smoke_runtime_packages(self.env)
        launch_package, launch_file = _gazebo_launch_target(manifest, headless=headless)
        world = _project_scoped_existing_path(
            self.project_dir, workspace / _world_relative_path(manifest)
        )
        argv = (
            "ros2",
            "launch",
            launch_package,
            launch_file,
            f"world:={world}",
        )
        env_overlay = _simulation_env_overlay(self.project_dir, workspace, manifest, self.env)
        if smoke_test:
            robot_name = launch_package.removesuffix("_bringup")
            return self._run_headless_smoke(
                argv, cwd=workspace, env_overlay=env_overlay, _robot_name=robot_name
            )
        return self._run(
            ExecutionCommand.SIMULATE_ROS_GAZEBO,
            argv,
            cwd=workspace,
            env_overlay=env_overlay,
        )

    def _run(
        self,
        command: ExecutionCommand,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env_overlay: Mapping[str, str] | None = None,
    ) -> ExecutionResult:
        _validate_command(command, argv)
        scoped_cwd = _project_scoped_existing_path(self.project_dir, cwd)
        env = _sanitized_env(self.env)
        if env_overlay:
            env.update(_sanitized_env(env_overlay))
        try:
            process = self.popen_factory(
                argv,
                cwd=str(scoped_cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return ExecutionResult(
                command,
                argv,
                scoped_cwd,
                127,
                "",
                _bounded_text(str(exc)),
            )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            returncode = int(process.returncode or 0)
            timed_out = False
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, signal.SIGKILL)
                try:
                    stdout, stderr = process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = "", "process did not exit after SIGKILL"
            returncode = 124
            timed_out = True
        stdout_text = _to_text(stdout)
        stderr_text = _to_text(stderr)
        if command is ExecutionCommand.SIMULATE_ROS_GAZEBO and _ros_launch_reported_child_error(
            stdout_text + "\n" + stderr_text
        ):
            returncode = 1
            stderr_text = _append_execution_diagnostic(
                stderr_text, "ros2 launch reported child process errors"
            )
        return ExecutionResult(
            command,
            argv,
            scoped_cwd,
            returncode,
            _bounded_text(stdout_text),
            _bounded_text(stderr_text),
            timed_out,
        )


    def _run_headless_smoke(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env_overlay: Mapping[str, str],
        _robot_name: str,
    ) -> ExecutionResult:
        command = ExecutionCommand.SIMULATE_ROS_GAZEBO
        _validate_command(command, argv)
        scoped_cwd = _project_scoped_existing_path(self.project_dir, cwd)
        env = _sanitized_env(self.env)
        env.update(_sanitized_env(env_overlay))
        try:
            process = self.popen_factory(
                argv,
                cwd=str(scoped_cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            return ExecutionResult(command, argv, scoped_cwd, 127, "", _bounded_text(str(exc)))

        report: list[str] = ["headless smoke started"]
        returncode = 1
        failure = "headless smoke checks did not complete"
        try:
            deadline = time.monotonic() + self.timeout_seconds
            _sleep_until_ready(deadline)
            early = _poll_returncode(process)
            if early not in (None, 0):
                failure = f"ros2 launch exited before smoke checks completed: {early}"
            else:
                topic_list = _wait_for_topics(
                    env, deadline, ("/clock", "/scan", "/imu", "/joint_states")
                )
                report.append("topics present: " + ", ".join(topic_list))
                controllers = _wait_for_active_controllers(
                    env, deadline, ("joint_state_broadcaster", "diff_drive_controller")
                )
                report.append("controllers active: " + ", ".join(controllers))
                _require_advancing_message(
                    env, deadline, "/clock", "rosgraph_msgs/msg/Clock", "clock"
                )
                report.append("clock messages received and increasing")
                _require_topic_message(env, deadline, "/scan", "sensor_msgs/msg/LaserScan")
                report.append("LiDAR message received")
                _require_topic_message(env, deadline, "/imu", "sensor_msgs/msg/Imu")
                report.append("IMU message received")
                _require_advancing_message(
                    env, deadline, "/joint_states", "sensor_msgs/msg/JointState", "joint states"
                )
                report.append("joint states received with advancing timestamps")
                cmd_topic = "/diff_drive_controller/cmd_vel"
                odom_topic = "/diff_drive_controller/odom"
                odom_before = _require_topic_message(
                    env, deadline, odom_topic, "nav_msgs/msg/Odometry"
                )
                report.append("odometry received")
                try:
                    _publish_velocity_command(env, deadline, cmd_topic, linear_x=0.10, times=20)
                    report.append("bounded stamped velocity command accepted")
                    displacement = _wait_for_odom_movement(env, deadline, odom_topic, odom_before)
                    report.append(f"observable odometry movement verified: {displacement:.6f} m")
                finally:
                    _publish_velocity_command(env, deadline, cmd_topic, linear_x=0.0, times=3)
                    report.append("stop command accepted")
                returncode = 0
                failure = ""
        except ExecutionError as exc:
            failure = str(exc)
        finally:
            _terminate_process_group(process, signal.SIGTERM)
            try:
                launch_stdout, launch_stderr = process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, signal.SIGKILL)
                try:
                    launch_stdout, launch_stderr = process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    launch_stdout, launch_stderr = "", "process did not exit after SIGKILL"
        stdout = "\n".join(report)
        stderr = _to_text(launch_stderr)
        launch_text = _to_text(launch_stdout)
        if launch_text:
            stdout = stdout + "\n--- launch stdout ---\n" + launch_text
        if failure:
            stderr = _append_execution_diagnostic(stderr, failure)
        if _ros_launch_reported_child_error(stdout + "\n" + stderr):
            returncode = 1
            stderr = _append_execution_diagnostic(
                stderr, "ros2 launch reported child process errors"
            )
        return ExecutionResult(
            command,
            argv,
            scoped_cwd,
            returncode,
            _bounded_text(stdout),
            _bounded_text(stderr),
            False,
        )

def build_ros_gazebo_workspace(
    project_dir: Path,
    workspace_dir: Path | None = None,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ExecutionResult:
    return ConstrainedRunner(
        project_dir=project_dir, timeout_seconds=timeout_seconds
    ).build_ros_gazebo(workspace_dir)


def simulate_ros_gazebo_workspace(
    project_dir: Path,
    workspace_dir: Path | None = None,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    headless: bool = False,
    smoke_test: bool = False,
) -> ExecutionResult:
    runner = ConstrainedRunner(project_dir=project_dir, timeout_seconds=timeout_seconds)
    return runner.simulate_ros_gazebo(workspace_dir, headless=headless, smoke_test=smoke_test)


def _fresh_artifact_manifest(project_dir: Path) -> ArtifactManifest:
    try:
        manifest, status = load_manifest(project_dir)
    except ArtifactError as exc:
        raise ExecutionError(f"generated artifact manifest is invalid: {exc}") from exc
    if manifest is None:
        raise ExecutionError("generated artifacts are missing; generate ROS/Gazebo artifacts first")
    if status != "fresh":
        raise ExecutionError("generated artifacts are stale; regenerate before build or simulation")
    return manifest


def _gazebo_launch_target(manifest: ArtifactManifest, *, headless: bool) -> tuple[str, str]:
    launch_name = "gazebo_headless.launch.py" if headless else "gazebo.launch.py"
    for record in manifest.artifacts:
        if record.role == "ros2_launch" and record.path.endswith(f"/launch/{launch_name}"):
            parts = Path(record.path).parts
            if len(parts) >= 5 and parts[0:2] == ("ros2_ws", "src"):
                return parts[2], parts[-1]
    raise ExecutionError(f"generated artifacts do not include {launch_name}")


def _world_relative_path(manifest: ArtifactManifest) -> Path:
    for record in manifest.artifacts:
        if record.role == "gazebo_world_sdf" and record.path.startswith("ros2_ws/"):
            return Path(*Path(record.path).parts[1:])
    raise ExecutionError("generated artifacts do not include a ROS workspace world SDF")


def _check_smoke_runtime_packages(env: Mapping[str, str]) -> None:
    missing: list[str] = []
    safe_env = _sanitized_env(env)
    for package in (
        "ros_gz_sim",
        "ros_gz_bridge",
        "gz_ros2_control",
        "robot_state_publisher",
        "xacro",
        "controller_manager",
        "joint_state_broadcaster",
        "diff_drive_controller",
    ):
        try:
            completed = subprocess.run(
                ("ros2", "pkg", "prefix", package),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=safe_env,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExecutionError(f"could not verify ROS package {package}: {exc}") from exc
        if completed.returncode != 0:
            missing.append(package)
    if missing:
        raise ExecutionError(
            "headless smoke test requires installed ROS 2 runtime packages: " + ", ".join(missing)
        )


def _simulation_env_overlay(
    project_dir: Path,
    workspace: Path,
    manifest: ArtifactManifest,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    install = _project_scoped_existing_path(project_dir, workspace / "install")
    generated_models = _project_scoped_existing_path(
        project_dir, project_dir / "generated" / "gazebo" / "models"
    )
    python_path = install / "lib" / "python3.12" / "site-packages"
    existing_ament = base_env.get("AMENT_PREFIX_PATH", "")
    existing_python = base_env.get("PYTHONPATH", "")
    generation_part = re.sub(r"[^A-Za-z0-9_]", "_", manifest.generation_id)[-24:]
    return {
        "AMENT_PREFIX_PATH": _prepend_path(str(install), existing_ament),
        "PYTHONPATH": _prepend_path(str(python_path), existing_python),
        "GZ_SIM_RESOURCE_PATH": str(generated_models),
        "SDF_PATH": str(generated_models),
        "GZ_PARTITION": f"hephaestus_{generation_part}",
        "GZ_IP": "127.0.0.1",
        "ROS_DOMAIN_ID": "42",
    }


def _prepend_path(value: str, existing: str) -> str:
    if not existing:
        return value
    return value + os.pathsep + existing


def _workspace(project_dir: Path, workspace_dir: Path | None) -> Path:
    candidate = workspace_dir if workspace_dir is not None else default_workspace_dir(project_dir)
    return _project_scoped_existing_path(project_dir, candidate)


def _clear_workspace_build_outputs(project_dir: Path, workspace: Path) -> None:
    for name in ("build", "install", "log"):
        path = workspace / name
        if not path.exists():
            continue
        scoped = _project_scoped_existing_path(project_dir, path)
        if scoped.is_dir():
            shutil.rmtree(scoped)
        else:
            scoped.unlink()


def _validate_command(command: ExecutionCommand, argv: tuple[str, ...]) -> None:
    if any(not part or "\x00" in part for part in argv):
        raise ExecutionError("argv contains an invalid argument")
    if command is ExecutionCommand.BUILD_ROS_GAZEBO:
        if len(argv) < 4 or argv[0] != "colcon" or "build" not in argv[:4]:
            raise ExecutionError("argv does not match application-owned build command")
        return
    if command is ExecutionCommand.SIMULATE_ROS_GAZEBO:
        if len(argv) != 5 or argv[:2] != ("ros2", "launch"):
            raise ExecutionError("argv does not match application-owned simulation command")
        if not argv[2].endswith("_bringup") or not argv[3].startswith("gazebo"):
            raise ExecutionError("argv does not match generated Gazebo launch target")
        if not argv[4].startswith("world:="):
            raise ExecutionError("simulation argv must include an application-owned world path")
        return
    raise ExecutionError(f"unsupported execution command: {command.value}")


def _project_scoped_existing_path(root: Path, path: Path) -> Path:
    if not path.exists():
        raise ExecutionError(f"path does not exist: {path}")
    resolved_root = root.resolve()
    _reject_symlink_ancestors(path, "execute")
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ExecutionError(f"path escapes project scope: {path}") from exc
    return resolved


def _reject_symlink_ancestors(path: Path, operation: str) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ExecutionError(
                f"refusing to {operation} through generated path symlink: {current}"
            )


def _sleep_until_ready(deadline: float) -> None:
    time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))


def _poll_returncode(process: _PopenLike) -> int | None:
    poll = getattr(process, "poll", None)
    if callable(poll):
        value = poll()
        return None if value is None else int(value)
    value = process.returncode
    return None if value is None else int(value)


def _remaining(deadline: float, *, default: float = 5.0) -> float:
    return min(default, max(0.1, deadline - time.monotonic()))


def _run_ros_probe(
    argv: tuple[str, ...],
    env: Mapping[str, str],
    deadline: float,
    *,
    default_timeout: float = 5.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            timeout=_remaining(deadline, default=default_timeout),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionError(f"ROS probe failed for {' '.join(argv[:3])}: {exc}") from exc


def _wait_for_topics(
    env: Mapping[str, str], deadline: float, expected: tuple[str, ...]
) -> tuple[str, ...]:
    last_output = ""
    while time.monotonic() < deadline:
        completed = _run_ros_probe(("ros2", "topic", "list"), env, deadline)
        last_output = completed.stdout + completed.stderr
        topics = set(completed.stdout.split())
        if completed.returncode == 0 and all(topic in topics for topic in expected):
            return expected
        time.sleep(0.5)
    missing = [topic for topic in expected if topic not in set(last_output.split())]
    raise ExecutionError("required ROS topics were not observed: " + ", ".join(missing))


def _wait_for_active_controllers(
    env: Mapping[str, str], deadline: float, expected: tuple[str, ...]
) -> tuple[str, ...]:
    last_output = ""
    while time.monotonic() < deadline:
        completed = _run_ros_probe(
            (
                "ros2",
                "service",
                "call",
                "/controller_manager/list_controllers",
                "controller_manager_msgs/srv/ListControllers",
                "{}",
            ),
            env,
            deadline,
        )
        last_output = completed.stdout + completed.stderr
        if completed.returncode == 0 and all(
            _controller_is_active(last_output, controller) for controller in expected
        ):
            return expected
        time.sleep(0.5)
    missing = [
        controller for controller in expected if not _controller_is_active(last_output, controller)
    ]
    raise ExecutionError("required controllers were not active: " + ", ".join(missing))


def _controller_is_active(output: str, controller: str) -> bool:
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if controller not in line:
            continue
        window = "\n".join(lines[index : index + 6])
        if "active" in window:
            return True
    return False


def _require_topic_message(
    env: Mapping[str, str], deadline: float, topic: str, message_type: str
) -> str:
    while time.monotonic() < deadline:
        message = _optional_topic_message(env, deadline, topic, message_type)
        if message is not None:
            return message
        time.sleep(0.5)
    raise ExecutionError(f"no message received on {topic}")


def _optional_topic_message(
    env: Mapping[str, str], deadline: float, topic: str, message_type: str
) -> str | None:
    try:
        completed = subprocess.run(
            (
                "ros2",
                "topic",
                "echo",
                "--once",
                "--qos-reliability",
                "best_effort",
                topic,
                message_type,
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=env,
            timeout=_remaining(deadline),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout
    return None


def _require_advancing_message(
    env: Mapping[str, str], deadline: float, topic: str, message_type: str, label: str
) -> str:
    before = _require_topic_message(env, deadline, topic, message_type)
    before_stamp = _message_stamp(before)
    if before_stamp is None:
        raise ExecutionError(f"{label} message did not include a parseable timestamp")
    while time.monotonic() < deadline:
        time.sleep(0.25)
        after = _optional_topic_message(env, deadline, topic, message_type)
        if after is None:
            continue
        after_stamp = _message_stamp(after)
        if after_stamp is not None and after_stamp > before_stamp:
            return after
    raise ExecutionError(f"{label} timestamp did not advance")


def _message_stamp(message: str) -> tuple[int, int] | None:
    match = re.search(r"sec:\s*(?P<sec>\d+).*?nanosec:\s*(?P<nanosec>\d+)", message, re.S)
    if match is None:
        return None
    return int(match.group("sec")), int(match.group("nanosec"))


def _publish_velocity_command(
    env: Mapping[str, str], deadline: float, topic: str, *, linear_x: float, times: int
) -> None:
    message = (
        "{header: auto, twist: {linear: {x: "
        + f"{linear_x:.3f}"
        + "}, angular: {z: 0.0}}}"
    )
    completed = _run_ros_probe(
        (
            "ros2",
            "topic",
            "pub",
            "--times",
            str(times),
            "--rate",
            "10",
            "--use-sim-time",
            "--qos-reliability",
            "best_effort",
            topic,
            "geometry_msgs/msg/TwistStamped",
            message,
        ),
        env,
        deadline,
        default_timeout=10.0,
    )
    if completed.returncode != 0:
        raise ExecutionError(f"bounded velocity command was not accepted on {topic}")


def _wait_for_odom_movement(
    env: Mapping[str, str], deadline: float, topic: str, before: str
) -> float:
    while time.monotonic() < deadline:
        after = _optional_topic_message(env, deadline, topic, "nav_msgs/msg/Odometry")
        if after is not None:
            displacement = _odom_displacement(before, after)
            if displacement > 0.01:
                return displacement
        time.sleep(0.25)
    raise ExecutionError(f"observable odometry movement was not verified on {topic}")


def _odom_displacement(before: str, after: str) -> float:
    first = _odom_xy(before)
    second = _odom_xy(after)
    if first is None or second is None:
        return 0.0
    return ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5


def _odom_xy(message: str) -> tuple[float, float] | None:
    x_match = re.search(r"position:\s*\n\s*x: (?P<x>[-+0-9.eE]+)", message)
    y_match = re.search(r"position:\s*\n(?:.*\n){0,3}\s*y: (?P<y>[-+0-9.eE]+)", message)
    if x_match is None or y_match is None:
        return None
    try:
        return float(x_match.group("x")), float(y_match.group("y"))
    except ValueError:
        return None


def _interpret_headless_smoke_result(result: ExecutionResult) -> ExecutionResult:
    output = result.stdout + "\n" + result.stderr
    if result.timed_out and result.returncode == 124 and _headless_smoke_markers_present(output):
        return ExecutionResult(
            result.command,
            result.argv,
            result.cwd,
            0,
            result.stdout,
            result.stderr,
            False,
        )
    return result


def _headless_smoke_markers_present(output: str) -> bool:
    markers = (
        "Robot initialized",
        "Creating GZ->ROS Bridge: [/clock",
        "Creating GZ->ROS Bridge: [/scan",
        "Creating GZ->ROS Bridge: [/imu",
        "Configured and activated joint_state_broadcaster",
        "Configured and activated diff_drive_controller",
    )
    return all(marker in output for marker in markers)


def _ros_launch_reported_child_error(output: str) -> bool:
    markers = (
        "[ERROR]",
        "process has died",
        "I cannot find any available gz command",
    )
    return any(marker in output for marker in markers)


def _append_execution_diagnostic(stderr: str, message: str) -> str:
    if not stderr.strip():
        return message
    return stderr.rstrip() + "\n" + message


def _sanitized_env(env: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if key in ALLOWED_ENV_KEYS and "\x00" not in key and "\x00" not in value
    }


def _terminate_process_group(process: _PopenLike, sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except OSError:
        process.kill()


def _to_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _bounded_text(value: str) -> str:
    redacted = SECRET_PATTERN.sub("[REDACTED]", value.replace("\x00", ""))
    if len(redacted) <= MAX_CAPTURE_CHARS:
        return redacted
    return redacted[:MAX_CAPTURE_CHARS] + "\n[truncated]"
