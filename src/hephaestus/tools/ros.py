"""Provider-independent ROS 2 workbench for local simulation sessions."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from hephaestus.execution import _bounded_text, _sanitized_env, _terminate_process_group

ROS_LOG_MAX_LINES = 240
ROS_LOG_RETURN_LINES = 120
ROS_LOG_MAX_BYTES = 48_000
ROS_DOMAIN_MIN = 1
ROS_DOMAIN_MAX = 101
DEFAULT_CMDVEL_RATE_HZ = 10
DEFAULT_MAX_LINEAR_X_MPS = 0.5
DEFAULT_MAX_ANGULAR_Z_RADPS = 1.0
MAX_CMDVEL_DURATION_SECONDS = 10.0
ROS_BUILD_TIMEOUT_SECONDS = 180.0
ROS_LAUNCH_STOP_TIMEOUT_SECONDS = 5.0
MAX_RELATIVE_PATH_LENGTH = 300
_SAFE_PACKAGE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
_SAFE_LAUNCH = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,119}\.launch\.py$")


class RosWorkbenchError(RuntimeError):
    """Raised when a local ROS operation is rejected."""


class _PopenLike(Protocol):
    pid: int
    returncode: int | None
    stdout: TextIO | None
    stderr: TextIO | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...

    def communicate(self, timeout: float | None = None) -> tuple[str | bytes, str | bytes]: ...


class _PopenFactory(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: object) -> _PopenLike: ...


@dataclass(slots=True)
class _BoundedLog:
    max_lines: int = ROS_LOG_MAX_LINES
    max_bytes: int = ROS_LOG_MAX_BYTES
    _lines: deque[str] = field(default_factory=deque)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, source: str, line: str | bytes) -> None:
        text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        clean = _bounded_text(text.rstrip("\n"))
        if not clean:
            return
        entry = f"{source}: {clean}"
        size = len(entry.encode("utf-8"))
        with self._lock:
            self._lines.append(entry)
            self._bytes += size
            while len(self._lines) > self.max_lines or self._bytes > self.max_bytes:
                removed = self._lines.popleft()
                self._bytes -= len(removed.encode("utf-8"))

    def recent(self, max_lines: int) -> list[str]:
        limit = min(max(1, max_lines), ROS_LOG_RETURN_LINES)
        with self._lock:
            return list(self._lines)[-limit:]


@dataclass(slots=True)
class _OwnedRosProcess:
    process: _PopenLike
    workspace: Path
    package: str
    launch_file: str
    started_at: float
    env: dict[str, str]
    log: _BoundedLog = field(default_factory=_BoundedLog)
    stopped: bool = False

    def poll(self) -> int | None:
        try:
            return self.process.poll()
        except OSError:
            return self.process.returncode

    @property
    def active(self) -> bool:
        return self.poll() is None and not self.stopped


@dataclass(slots=True)
class RosWorkbench:
    workspace_root: Path
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    popen_factory: _PopenFactory = subprocess.Popen
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    domain_id: int | None = None
    built_workspace: Path | None = None
    owned_process: _OwnedRosProcess | None = None

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve(strict=True)
        if self.domain_id is None:
            digest = hashlib.sha256(str(self.workspace_root).encode("utf-8")).digest()
            value = ROS_DOMAIN_MIN + digest[0] % (ROS_DOMAIN_MAX - ROS_DOMAIN_MIN + 1)
            self.domain_id = value

    @property
    def session_env(self) -> dict[str, str]:
        return _session_ros_env(self.env, self.domain_id or 42, self.workspace_root)

    def status(self) -> dict[str, Any]:
        ros_distro = self.session_env.get("ROS_DISTRO")
        controller_available = self._controller_manager_available()
        return {
            "ros_distro": ros_distro,
            "ros_jazzy_available": ros_distro == "jazzy" and _command_exists("ros2", self.env),
            "isolated_domain_id": self.domain_id,
            "localhost_only": True,
            "runtime_state": self._runtime_state(),
            "owned_ros_process": self._owned_summary(),
            "built_workspace": _relative_or_none(self.workspace_root, self.built_workspace),
            "controller_manager_available": controller_available,
            "simulated_motion_permitted": self._motion_permitted(controller_available),
        }

    def quick_status(self) -> dict[str, Any]:
        """Return session-owned state without ROS graph probes."""

        return {
            "isolated_domain_id": self.domain_id,
            "runtime_state": self._runtime_state(),
            "owned_ros_process": self._owned_summary(),
            "built_workspace": _relative_or_none(self.workspace_root, self.built_workspace),
        }

    def topics(self) -> dict[str, Any]:
        completed = self._run_probe(
            ("ros2", "topic", "list", "--no-daemon", "--spin-time", "2", "-t"),
            timeout=8.0,
        )
        topics = _parse_topic_list(completed.stdout)
        return {
            "topics": topics[:200],
            "count": min(len(topics), 200),
            "truncated": len(topics) > 200,
            "domain_id": self.domain_id,
        }

    def controllers(self) -> dict[str, Any]:
        output = self._list_controllers_output()
        controllers = _parse_controllers(output)
        if not controllers:
            controllers = self._controllers_from_owned_logs()
        return {"controllers": controllers, "count": len(controllers), "domain_id": self.domain_id}

    def build(
        self,
        workspace: Path,
        relative_workspace: str,
        *,
        confirm: Callable[[dict[str, Any]], bool] | None,
    ) -> dict[str, Any]:
        workspace = _safe_workspace_dir(self.workspace_root, workspace)
        src = workspace / "src"
        if not src.is_dir():
            raise RosWorkbenchError("ROS workspace must contain a src directory")
        argv = (
            "colcon",
            "--log-base",
            str(workspace / "log"),
            "build",
            "--merge-install",
            "--base-paths",
            str(src),
            "--build-base",
            str(workspace / "build"),
            "--install-base",
            str(workspace / "install"),
        )
        preview = {
            "status": "preview",
            "tool": "ros_build",
            "permission": "ros build",
            "result": {
                "relative_workspace": relative_workspace,
                "argv": list(argv),
                "domain_id": self.domain_id,
                "warning": "colcon build may execute package build logic from the workspace",
            },
        }
        if confirm is None or not confirm(preview):
            return {
                "success": False,
                "committed": False,
                "relative_workspace": relative_workspace,
                "message": "ROS build was previewed but not approved",
            }
        completed = self._run_bounded_process(
            argv, cwd=workspace, timeout=ROS_BUILD_TIMEOUT_SECONDS
        )
        setup = workspace / "install" / "setup.bash"
        success = completed.returncode == 0 and setup.exists()
        if success:
            self.built_workspace = workspace
        stderr = completed.stderr
        if completed.returncode == 0 and not setup.exists():
            stderr = _append(stderr, "expected install/setup.bash was not created")
        return {
            "success": success,
            "committed": success,
            "relative_workspace": relative_workspace,
            "argv": list(argv),
            "returncode": completed.returncode,
            "stdout": _bounded_text(completed.stdout),
            "stderr": _bounded_text(stderr),
            "built_workspace": _relative_or_none(self.workspace_root, self.built_workspace),
        }

    def launch(
        self,
        workspace: Path,
        relative_workspace: str,
        package: str,
        launch_file: str,
        *,
        confirm: Callable[[dict[str, Any]], bool] | None,
    ) -> dict[str, Any]:
        workspace = _safe_workspace_dir(self.workspace_root, workspace)
        if self.built_workspace is None or workspace.resolve() != self.built_workspace.resolve():
            raise RosWorkbenchError("ROS launch requires a workspace built in this session")
        _check_package_name(package)
        _check_launch_file(launch_file)
        if self._active_owned_process() is not None:
            raise RosWorkbenchError("a Hephaestus-owned ROS simulation is already active")
        setup = workspace / "install" / "setup.bash"
        if not setup.exists():
            raise RosWorkbenchError("built workspace is missing install/setup.bash")
        argv = ("ros2", "launch", package, launch_file)
        preview = {
            "status": "preview",
            "tool": "ros_launch",
            "permission": "ros launch",
            "result": {
                "relative_workspace": relative_workspace,
                "package": package,
                "launch_file": launch_file,
                "argv": list(argv),
                "domain_id": self.domain_id,
            },
        }
        if confirm is None or not confirm(preview):
            return {
                "success": False,
                "active": False,
                "relative_workspace": relative_workspace,
                "message": "ROS launch was previewed but not approved",
            }
        env = _workspace_launch_env(self.session_env, workspace)
        try:
            process = self.popen_factory(
                argv,
                cwd=str(workspace),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise RosWorkbenchError(
                f"ROS launch failed to start: {_bounded_text(str(exc))}"
            ) from exc
        owned = _OwnedRosProcess(process, workspace, package, launch_file, time.monotonic(), env)
        self.owned_process = owned
        _start_log_reader(owned, "stdout", process.stdout)
        _start_log_reader(owned, "stderr", process.stderr)
        return {
            "success": True,
            "active": owned.active,
            "state": self._state(owned),
            "relative_workspace": relative_workspace,
            "package": package,
            "launch_file": launch_file,
            "pid": process.pid,
            "argv": list(argv),
            "env": {
                "ROS_DOMAIN_ID": str(self.domain_id),
                "ROS_LOCALHOST_ONLY": "1",
                "GZ_IP": "127.0.0.1",
            },
        }

    def logs(self, max_lines: int) -> dict[str, Any]:
        owned = self.owned_process
        if owned is None:
            return {"active": False, "state": "inactive", "lines": []}
        return {
            "active": owned.active,
            "state": self._state(owned),
            "lines": owned.log.recent(max_lines),
            "max_lines": min(max(1, max_lines), ROS_LOG_RETURN_LINES),
        }

    def stop_launch(self) -> dict[str, Any]:
        stop_result: dict[str, Any] | None = None
        try:
            stop_result = self.stop_motion()
        except RosWorkbenchError as exc:
            stop_result = {"stopped": False, "error": str(exc)}
        owned = self.owned_process
        if owned is None:
            return {
                "active": False,
                "state": "inactive",
                "stopped": False,
                "zero_velocity": stop_result,
            }
        if owned.poll() is not None or owned.stopped:
            owned.stopped = True
            return {
                "active": False,
                "state": "stopped",
                "stopped": False,
                "exit_code": owned.poll(),
                "zero_velocity": stop_result,
            }
        _terminate_process_group(owned.process, signal.SIGTERM)
        terminated = _wait_for_exit(owned, ROS_LAUNCH_STOP_TIMEOUT_SECONDS)
        killed = False
        if not terminated:
            _terminate_process_group(owned.process, signal.SIGKILL)
            killed = True
            terminated = _wait_for_exit(owned, 2.0)
        if not terminated:
            owned.process.kill()
            _wait_for_exit(owned, 1.0)
        owned.stopped = True
        return {
            "active": False,
            "state": "stopped",
            "stopped": True,
            "sigkill_used": killed,
            "exit_code": owned.poll(),
            "owned_process_exists": owned.poll() is None,
            "zero_velocity": stop_result,
        }

    def cmd_vel(self, linear_x: float, angular_z: float, duration: float) -> dict[str, Any]:
        _validate_cmdvel_numbers(linear_x, angular_z, duration)
        self._require_motion_allowed()
        topic, msg_type = self._discover_command_topic()
        times = max(1, int(math.ceil(duration * DEFAULT_CMDVEL_RATE_HZ)))
        env = self._motion_env()
        deadline = time.monotonic() + duration + 8.0
        stopped = False
        try:
            _publish_velocity(
                self.run_command,
                env,
                deadline,
                topic,
                msg_type,
                linear_x=linear_x,
                angular_z=angular_z,
                times=times,
            )
        finally:
            stopped = self._publish_zero_velocity(topic, msg_type)
        return {
            "commanded_linear_x_mps": linear_x,
            "commanded_angular_z_radps": angular_z,
            "duration_seconds": duration,
            "rate_hz": DEFAULT_CMDVEL_RATE_HZ,
            "topic": topic,
            "message_type": msg_type,
            "zero_velocity_sent": stopped,
        }

    def stop_motion(self) -> dict[str, Any]:
        self._require_motion_allowed()
        topic, msg_type = self._discover_command_topic()
        stopped = self._publish_zero_velocity(topic, msg_type)
        return {"topic": topic, "message_type": msg_type, "zero_velocity_sent": stopped}

    def close(self) -> None:
        self.stop_launch()

    def _owned_summary(self) -> dict[str, Any] | None:
        owned = self.owned_process
        if owned is None:
            return None
        return {
            "active": owned.active,
            "state": self._state(owned),
            "pid": owned.process.pid,
            "package": owned.package,
            "launch_file": owned.launch_file,
        }

    def _runtime_state(self) -> str:
        owned = self.owned_process
        if owned is None:
            return "inactive"
        return self._state(owned)

    def _motion_permitted(self, controller_available: bool | None = None) -> bool:
        if self._active_owned_process() is None:
            return False
        if self.session_env.get("ROS_DISTRO") != "jazzy":
            return False
        if not (controller_available if controller_available is not None else True):
            return False
        try:
            controllers = self.controllers()["controllers"]
            if not any(
                item["name"] == "diff_drive_controller" and item["state"] == "active"
                for item in controllers
            ):
                return False
            self._discover_command_topic()
        except RosWorkbenchError:
            return False
        return True

    def _require_motion_allowed(self) -> None:
        if self._active_owned_process() is None:
            raise RosWorkbenchError("movement requires a Hephaestus-owned active ROS simulation")
        if self.session_env.get("ROS_DISTRO") != "jazzy":
            raise RosWorkbenchError("movement requires ROS 2 Jazzy in the isolated session")
        controllers = self.controllers()["controllers"]
        if not any(
            item["name"] == "diff_drive_controller" and item["state"] == "active"
            for item in controllers
        ):
            raise RosWorkbenchError("active differential-drive controller was not discovered")

    def _discover_command_topic(self) -> tuple[str, str]:
        topics = self.topics()["topics"]
        candidates = {
            str(item["name"]): str(item["type"])
            for item in topics
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        preferred = "/diff_drive_controller/cmd_vel"
        for topic in (preferred, *sorted(candidates)):
            msg_type = candidates.get(topic)
            if msg_type in {
                "geometry_msgs/msg/Twist",
                "geometry_msgs/msg/TwistStamped",
            } and (topic.endswith("/cmd_vel") or topic.endswith("/cmd_vel_unstamped")):
                return topic, msg_type
        controllers = self.controllers()["controllers"]
        if any(
            item["name"] == "diff_drive_controller" and item["state"] == "active"
            for item in controllers
        ):
            return preferred, "geometry_msgs/msg/TwistStamped"
        raise RosWorkbenchError("controller command topic was not discovered in isolated graph")

    def _motion_env(self) -> dict[str, str]:
        owned = self._active_owned_process()
        if owned is None:
            raise RosWorkbenchError("no active owned ROS process")
        return dict(owned.env)

    def _publish_zero_velocity(self, topic: str, msg_type: str) -> bool:
        try:
            _publish_velocity(
                self.run_command,
                self._motion_env(),
                time.monotonic() + 5.0,
                topic,
                msg_type,
                linear_x=0.0,
                angular_z=0.0,
                times=3,
            )
            return True
        except RosWorkbenchError:
            return False

    def _run_probe(
        self, argv: tuple[str, ...], *, timeout: float = 5.0
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = self.run_command(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=(
                    self._motion_env()
                    if self._active_owned_process() is not None
                    else self.session_env
                ),
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RosWorkbenchError(f"ROS probe failed for {' '.join(argv[:3])}: {exc}") from exc
        if completed.returncode != 0:
            raise RosWorkbenchError(
                "ROS probe failed: " + _bounded_text(completed.stderr or completed.stdout)
            )
        return completed

    def _list_controllers_output(self) -> str:
        completed = self._run_probe(
            (
                "ros2",
                "service",
                "call",
                "/controller_manager/list_controllers",
                "controller_manager_msgs/srv/ListControllers",
                "{}",
            ),
            timeout=6.0,
        )
        return completed.stdout + completed.stderr

    def _controller_manager_available(self) -> bool:
        try:
            self._list_controllers_output()
        except RosWorkbenchError:
            return False
        return True

    def _controllers_from_owned_logs(self) -> list[dict[str, str]]:
        owned = self.owned_process
        if owned is None:
            return []
        text = "\n".join(owned.log.recent(ROS_LOG_MAX_LINES))
        if "Configured and activated all the parsed controllers list" not in text:
            return []
        controllers: list[dict[str, str]] = []
        if "joint_state_broadcaster" in text:
            controllers.append(
                {
                    "name": "joint_state_broadcaster",
                    "type": "joint_state_broadcaster/JointStateBroadcaster",
                    "state": "active",
                    "source": "session_owned_launch_log",
                }
            )
        if "diff_drive_controller" in text:
            controllers.append(
                {
                    "name": "diff_drive_controller",
                    "type": "diff_drive_controller/DiffDriveController",
                    "state": "active",
                    "source": "session_owned_launch_log",
                }
            )
        return controllers

    def _run_bounded_process(
        self, argv: tuple[str, ...], *, cwd: Path, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        env = self.session_env
        try:
            process = self.popen_factory(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise RosWorkbenchError(
                f"ROS process failed to start: {_bounded_text(str(exc))}"
            ) from exc
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = int(process.returncode or 0)
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
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout=_bounded_text(_to_text(stdout)),
            stderr=_bounded_text(_to_text(stderr)),
        )

    def _active_owned_process(self) -> _OwnedRosProcess | None:
        owned = self.owned_process
        if owned is not None and owned.active:
            return owned
        return None

    def _state(self, owned: _OwnedRosProcess) -> str:
        if owned.stopped:
            return "stopped"
        if owned.poll() is not None:
            return "exited"
        if time.monotonic() - owned.started_at < 0.5:
            return "starting"
        return "running"


def _session_ros_env(
    parent: Mapping[str, str], domain_id: int, workspace_root: Path
) -> dict[str, str]:
    if domain_id < ROS_DOMAIN_MIN or domain_id > ROS_DOMAIN_MAX:
        raise RosWorkbenchError("ROS_DOMAIN_ID is outside the bounded local session range")
    env = _sanitized_env(parent)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["GZ_IP"] = "127.0.0.1"
    partition = hashlib.sha256(str(workspace_root).encode("utf-8")).hexdigest()[:16]
    env["GZ_PARTITION"] = f"hephaestus_{partition}"
    return env


def _workspace_launch_env(base_env: Mapping[str, str], workspace: Path) -> dict[str, str]:
    env = dict(base_env)
    install = workspace / "install"
    python_path = install / "lib" / "python3.12" / "site-packages"
    env["AMENT_PREFIX_PATH"] = _prepend(str(install), env.get("AMENT_PREFIX_PATH", ""))
    env["PYTHONPATH"] = _prepend(str(python_path), env.get("PYTHONPATH", ""))
    return env


def _safe_workspace_dir(root: Path, path: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    if not path.exists():
        raise RosWorkbenchError("ROS workspace path does not exist")
    if any(part == ".." for part in path.parts):
        raise RosWorkbenchError("parent traversal is not allowed")
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise RosWorkbenchError("ROS workspace path traverses a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RosWorkbenchError("ROS workspace path escapes startup directory") from exc
    if any(part in {".git", ".venv"} for part in relative.parts):
        raise RosWorkbenchError("ROS workspace path is protected")
    if not resolved.is_dir():
        raise RosWorkbenchError("ROS workspace path is not a directory")
    return resolved


def _check_package_name(value: str) -> None:
    if not _SAFE_PACKAGE.fullmatch(value):
        raise RosWorkbenchError("PACKAGE must be a safe ROS package identifier")


def _check_launch_file(value: str) -> None:
    if "/" in value or "\\" in value or not _SAFE_LAUNCH.fullmatch(value):
        raise RosWorkbenchError("LAUNCH_FILE must be a launch file name, not a path")


def _validate_cmdvel_numbers(linear_x: float, angular_z: float, duration: float) -> None:
    if not all(math.isfinite(item) for item in (linear_x, angular_z, duration)):
        raise RosWorkbenchError("movement arguments must be finite decimal numbers")
    if abs(linear_x) > DEFAULT_MAX_LINEAR_X_MPS:
        raise RosWorkbenchError("linear_x exceeds the 0.5 m/s simulation limit")
    if abs(angular_z) > DEFAULT_MAX_ANGULAR_Z_RADPS:
        raise RosWorkbenchError("angular_z exceeds the 1.0 rad/s simulation limit")
    if duration <= 0.0 or duration > MAX_CMDVEL_DURATION_SECONDS:
        raise RosWorkbenchError("duration must be greater than 0 and no more than 10 seconds")


def _publish_velocity(
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    env: Mapping[str, str],
    deadline: float,
    topic: str,
    msg_type: str,
    *,
    linear_x: float,
    angular_z: float,
    times: int,
) -> None:
    if msg_type == "geometry_msgs/msg/TwistStamped":
        message = (
            "{header: auto, twist: {linear: {x: "
            + f"{linear_x:.6f}"
            + "}, angular: {z: "
            + f"{angular_z:.6f}"
            + "}}}"
        )
        argv = (
            "ros2",
            "topic",
            "pub",
            "--times",
            str(times),
            "--rate",
            str(DEFAULT_CMDVEL_RATE_HZ),
            "--use-sim-time",
            "--qos-reliability",
            "best_effort",
            topic,
            msg_type,
            message,
        )
    elif msg_type == "geometry_msgs/msg/Twist":
        message = (
            "{linear: {x: "
            + f"{linear_x:.6f}"
            + "}, angular: {z: "
            + f"{angular_z:.6f}"
            + "}}"
        )
        argv = (
            "ros2",
            "topic",
            "pub",
            "--times",
            str(times),
            "--rate",
            str(DEFAULT_CMDVEL_RATE_HZ),
            "--qos-reliability",
            "best_effort",
            topic,
            msg_type,
            message,
        )
    else:
        raise RosWorkbenchError("unsupported controller command message type")
    try:
        completed = run_command(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=min(12.0, max(0.1, deadline - time.monotonic())),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RosWorkbenchError(f"velocity publish failed: {exc}") from exc
    if completed.returncode != 0:
        raise RosWorkbenchError(
            "velocity publish was not accepted: " + _bounded_text(completed.stderr)
        )


def _parse_topic_list(output: str) -> list[dict[str, str]]:
    topics: list[dict[str, str]] = []
    for line in output.splitlines():
        text = line.strip()
        if not text or "[" not in text or "]" not in text:
            continue
        name, rest = text.split("[", 1)
        msg_type = rest.split("]", 1)[0].strip()
        topics.append({"name": name.strip(), "type": msg_type})
    return topics


def _parse_controllers(output: str) -> list[dict[str, str]]:
    controllers: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        text = line.strip()
        if text.startswith("- name:"):
            if current:
                controllers.append(current)
            current = {"name": _yaml_scalar(text.removeprefix("- name:"))}
        elif text.startswith("name:"):
            if current:
                controllers.append(current)
            current = {"name": _yaml_scalar(text.removeprefix("name:"))}
        elif current and text.startswith("type:"):
            current["type"] = _yaml_scalar(text.removeprefix("type:"))
        elif current and text.startswith("state:"):
            current["state"] = _yaml_scalar(text.removeprefix("state:"))
    if current:
        controllers.append(current)
    return controllers


def _yaml_scalar(value: str) -> str:
    return value.strip().strip("'\"")


def _start_log_reader(owned: _OwnedRosProcess, source: str, stream: TextIO | None) -> None:
    if stream is None:
        return

    def read_stream() -> None:
        try:
            while True:
                line = stream.readline()
                if not line:
                    break
                owned.log.append(source, line)
        except Exception:
            owned.log.append(source, "log reader stopped")

    threading.Thread(target=read_stream, daemon=True).start()


def _wait_for_exit(owned: _OwnedRosProcess, timeout: float) -> bool:
    try:
        owned.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return owned.poll() is not None
    return owned.poll() is not None


def _relative_or_none(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=True)).as_posix()
    except ValueError:
        return None


def _prepend(value: str, existing: str) -> str:
    return value if not existing else value + os.pathsep + existing


def _append(stderr: str, message: str) -> str:
    return message if not stderr.strip() else stderr.rstrip() + "\n" + message


def _to_text(value: str | bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def _command_exists(name: str, env: Mapping[str, str]) -> bool:
    path = env.get("PATH") or os.environ.get("PATH", "")
    for directory in path.split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return True
    return False


def parse_decimal(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise RosWorkbenchError(f"not a decimal number: {value}") from exc
    if not math.isfinite(number):
        raise RosWorkbenchError("movement arguments must be finite decimal numbers")
    return number


def parse_move_command(tokens: Sequence[str]) -> tuple[float, float, float]:
    if len(tokens) != 4 or tokens[2] != "for":
        raise RosWorkbenchError(
            "expected /move forward SPEED_MPS for DURATION_SECONDS, backward, "
            "turn-left, or turn-right"
        )
    action = tokens[0]
    value = parse_decimal(tokens[1])
    duration = parse_decimal(tokens[3])
    if action == "forward":
        return value, 0.0, duration
    if action == "backward":
        return -value, 0.0, duration
    if action == "turn-left":
        return 0.0, value, duration
    if action == "turn-right":
        return 0.0, -value, duration
    raise RosWorkbenchError("unsupported move direction")


def shell_split(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError as exc:
        raise RosWorkbenchError(f"could not parse command: {exc}") from exc
