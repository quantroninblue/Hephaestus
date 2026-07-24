"""Session-owned Gazebo workbench tools for the default agent shell."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from hephaestus.execution import _bounded_text, _sanitized_env, _terminate_process_group

GAZEBO_PERMISSION_READ = "gazebo read"
GAZEBO_PERMISSION_LAUNCH = "gazebo launch"
GAZEBO_VALIDATE_TIMEOUT_SECONDS = 15.0
GAZEBO_STOP_TIMEOUT_SECONDS = 5.0
MAX_SDF_BYTES = 1_000_000
MAX_LOG_LINES = 200
MAX_LOG_RETURN_LINES = 100
MAX_LOG_BYTES = 32_000
MAX_INHERITED_ENV_VALUE_BYTES = 4096
LOCAL_MESH_SUFFIXES = (".stl", ".dae", ".obj")


class GazeboWorkbenchError(RuntimeError):
    """Raised when a Gazebo workbench operation is rejected."""


class GazeboGuiUnavailable(GazeboWorkbenchError):
    """Raised when a GUI launch cannot be attached to a desktop session."""


class GazeboValidateSdfArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=300)


class GazeboLaunchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=300)
    headless: bool = False


class GazeboReadLogsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    max_lines: int = Field(default=MAX_LOG_RETURN_LINES, ge=1, le=MAX_LOG_RETURN_LINES)


class GazeboEmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RunCommand(Protocol):
    def __call__(
        self, argv: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]: ...


class _PopenLike(Protocol):
    pid: int
    returncode: int | None
    stdout: TextIO | None
    stderr: TextIO | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class _PopenFactory(Protocol):
    def __call__(self, argv: Sequence[str], **kwargs: object) -> _PopenLike: ...


@dataclass(slots=True)
class _BoundedSessionLog:
    max_lines: int = MAX_LOG_LINES
    max_bytes: int = MAX_LOG_BYTES
    _lines: deque[str] = field(default_factory=deque)
    _bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def append(self, source: str, line: str | bytes) -> None:
        text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        clean = _bounded_text(text.rstrip("\n"))
        if not clean:
            return
        entry = f"{source}: {clean}"
        entry_bytes = len(entry.encode("utf-8"))
        with self._lock:
            self._lines.append(entry)
            self._bytes += entry_bytes
            while len(self._lines) > self.max_lines or self._bytes > self.max_bytes:
                removed = self._lines.popleft()
                self._bytes -= len(removed.encode("utf-8"))

    def recent(self, max_lines: int) -> list[str]:
        limit = min(max(max_lines, 1), MAX_LOG_RETURN_LINES)
        with self._lock:
            return list(self._lines)[-limit:]


@dataclass(slots=True)
class _OwnedGazeboProcess:
    process: _PopenLike
    relative_path: str
    absolute_path: Path
    started_at: float
    log: _BoundedSessionLog = field(default_factory=_BoundedSessionLog)
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
class GazeboWorkbench:
    workspace_root: Path
    gazebo_version: str | None = None
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)
    run_command: _RunCommand = subprocess.run
    popen_factory: _PopenFactory = subprocess.Popen
    owned_process: _OwnedGazeboProcess | None = None

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve(strict=True)

    def validate_sdf(self, path: Path, relative_path: str) -> dict[str, Any]:
        self._check_sdf_file(path)
        mesh_report = self._local_mesh_report(path)
        if mesh_report["errors"]:
            return {
                "valid": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "\n".join(mesh_report["errors"]),
                "relative_path": relative_path,
                "gazebo_version": self.gazebo_version,
                "timed_out": False,
                "local_meshes": mesh_report["meshes"],
            }
        return self._validate_sdf_checked(path, relative_path)

    def launch(
        self,
        path: Path,
        relative_path: str,
        *,
        headless: bool = False,
        confirm: Callable[[dict[str, Any]], bool] | None,
    ) -> dict[str, Any]:
        self._check_sdf_file(path)
        mesh_report = self._local_mesh_report(path)
        if mesh_report["errors"]:
            raise GazeboWorkbenchError("; ".join(mesh_report["errors"][:3]))
        if self._active_owned_process() is not None:
            raise GazeboWorkbenchError("a Hephaestus-owned Gazebo simulation is already active")
        before_digest = _file_digest(path)
        validation = self._validate_sdf_checked(path, relative_path)
        if not headless:
            _gazebo_gui_launch_env(self.env, self.workspace_root)
        argv_preview = ["gz", "sim", "-r", "-s", relative_path] if headless else [
            "gz",
            "sim",
            "-r",
            relative_path,
        ]
        preview = {
            "status": "preview",
            "tool": "gazebo_launch",
            "permission": GAZEBO_PERMISSION_LAUNCH,
            "result": {
                "relative_path": relative_path,
                "gui": not headless,
                "headless": headless,
                "executable": "gz",
                "argv_summary": argv_preview,
                "validation_passed": bool(validation["valid"]),
                "validation_exit_code": validation["exit_code"],
                "before_fingerprint": before_digest,
            },
        }
        if confirm is None or not confirm(preview):
            raise GazeboWorkbenchError("launch was previewed but not approved")
        if _file_digest(path) != before_digest:
            raise GazeboWorkbenchError("world file changed after launch preview")
        mesh_report = self._local_mesh_report(path)
        if mesh_report["errors"]:
            raise GazeboWorkbenchError("local mesh check failed before launch")
        if not validation["valid"]:
            raise GazeboWorkbenchError("SDF validation failed; launch refused")
        current_validation = self._validate_sdf_checked(path, relative_path)
        if not current_validation["valid"]:
            raise GazeboWorkbenchError("SDF validation failed during final launch check")
        argv = ("gz", "sim", "-r", "-s", str(path)) if headless else ("gz", "sim", "-r", str(path))
        env = (
            _gazebo_headless_launch_env(self.env, self.workspace_root)
            if headless
            else _gazebo_gui_launch_env(self.env, self.workspace_root)
        )
        try:
            process = self.popen_factory(
                argv,
                cwd=str(self.workspace_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise GazeboWorkbenchError(
                f"Gazebo launch failed to start: {_bounded_text(str(exc))}"
            ) from exc
        owned = _OwnedGazeboProcess(
            process=process,
            relative_path=relative_path,
            absolute_path=path,
            started_at=time.monotonic(),
        )
        self.owned_process = owned
        _start_log_reader(owned, "stdout", process.stdout)
        _start_log_reader(owned, "stderr", process.stderr)
        return {
            "active": owned.active,
            "state": self._state(owned),
            "relative_path": relative_path,
            "pid": process.pid,
            "argv": argv_preview,
            "headless": headless,
            "env": {"GZ_IP": "127.0.0.1"},
            "validation": current_validation,
        }

    def status(self) -> dict[str, Any]:
        owned = self.owned_process
        if owned is None:
            return {
                "active": False,
                "state": "inactive",
                "relative_path": None,
                "exit_code": None,
                "runtime_seconds": 0.0,
                "owned_process_exists": False,
            }
        poll = owned.poll()
        active = poll is None and not owned.stopped
        return {
            "active": active,
            "state": self._state(owned),
            "relative_path": owned.relative_path,
            "exit_code": None if poll is None else int(poll),
            "runtime_seconds": round(max(0.0, time.monotonic() - owned.started_at), 3),
            "owned_process_exists": active,
            "pid": owned.process.pid,
        }

    def read_logs(self, max_lines: int) -> dict[str, Any]:
        owned = self.owned_process
        if owned is None:
            return {
                "active": False,
                "state": "inactive",
                "relative_path": None,
                "lines": [],
                "message": "no Hephaestus-owned Gazebo simulation has been launched",
            }
        return {
            "active": owned.active,
            "state": self._state(owned),
            "relative_path": owned.relative_path,
            "lines": owned.log.recent(max_lines),
            "max_lines": min(max(max_lines, 1), MAX_LOG_RETURN_LINES),
        }

    def stop(self) -> dict[str, Any]:
        owned = self.owned_process
        if owned is None:
            return {
                "active": False,
                "state": "inactive",
                "stopped": False,
                "message": "no Hephaestus-owned Gazebo simulation is active",
            }
        if owned.poll() is not None or owned.stopped:
            owned.stopped = True
            return {
                "active": False,
                "state": "stopped",
                "stopped": False,
                "relative_path": owned.relative_path,
                "exit_code": owned.poll(),
                "owned_process_exists": False,
            }
        _terminate_process_group(owned.process, signal.SIGTERM)
        terminated = self._wait_for_exit(owned, GAZEBO_STOP_TIMEOUT_SECONDS)
        killed = False
        if not terminated:
            _terminate_process_group(owned.process, signal.SIGKILL)
            terminated = self._wait_for_exit(owned, 2.0)
            killed = True
        if not terminated:
            owned.process.kill()
            self._wait_for_exit(owned, 1.0)
        owned.stopped = True
        return {
            "active": False,
            "state": "stopped",
            "stopped": True,
            "sigkill_used": killed,
            "relative_path": owned.relative_path,
            "exit_code": owned.poll(),
            "owned_process_exists": owned.poll() is None,
        }

    def close(self) -> None:
        self.stop()

    def context_summary(self) -> str:
        status = self.status()
        path = status.get("relative_path") or "none"
        pid = status.get("pid")
        pid_part = f"; pid={pid}" if pid is not None else ""
        return (
            f"gazebo={status['state']}; active={str(status['active']).lower()}; "
            f"world={path}{pid_part}"
        )

    def _validate_sdf_checked(self, path: Path, relative_path: str) -> dict[str, Any]:
        argv = ("gz", "sdf", "-k", str(path))
        mesh_report = self._local_mesh_report(path)
        try:
            completed = self.run_command(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=GAZEBO_VALIDATE_TIMEOUT_SECONDS,
                check=False,
                env=_gazebo_resource_env(self.env, self.workspace_root),
            )
            exit_code = int(completed.returncode)
            stdout = _bounded_text(completed.stdout or "")
            stderr = _bounded_text(completed.stderr or "")
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = _bounded_text((exc.stdout or "") if isinstance(exc.stdout, str) else "")
            stderr = _bounded_text((exc.stderr or "") if isinstance(exc.stderr, str) else "")
            timed_out = True
        except OSError as exc:
            exit_code = 127
            stdout = ""
            stderr = _bounded_text(str(exc))
            timed_out = False
        return {
            "valid": exit_code == 0 and not timed_out,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "relative_path": relative_path,
            "gazebo_version": self.gazebo_version,
            "timed_out": timed_out,
            "local_meshes": mesh_report["meshes"],
        }

    def _check_sdf_file(self, path: Path) -> None:
        if path.suffix.lower() not in {".sdf", ".world"}:
            raise GazeboWorkbenchError("Gazebo tools accept only .sdf or .world files")
        if not path.is_file():
            raise GazeboWorkbenchError("Gazebo world path is not a regular file")
        size = path.stat().st_size
        if size > MAX_SDF_BYTES:
            raise GazeboWorkbenchError("SDF file exceeds Gazebo validation size limit")
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise GazeboWorkbenchError("SDF file is not valid UTF-8 text") from exc

    def _local_mesh_report(self, path: Path) -> dict[str, Any]:
        try:
            root = ET.fromstring(path.read_text(encoding="utf-8"))
        except ET.ParseError:
            return {"meshes": [], "errors": []}
        meshes: list[dict[str, str]] = []
        errors: list[str] = []
        for uri_node in root.findall(".//mesh/uri"):
            raw = (uri_node.text or "").strip()
            if not raw:
                errors.append("mesh uri is empty")
                continue
            if not _looks_like_local_mesh(raw):
                errors.append(f"mesh uri is not a workspace-local mesh: {_bounded_text(raw)}")
                continue
            candidates = self._mesh_candidates(raw)
            existing = next((candidate for candidate in candidates if candidate.exists()), None)
            if existing is None:
                errors.append(f"workspace mesh is missing: {_bounded_text(raw)}")
                continue
            try:
                resolved = _resolve_workspace_mesh(self.workspace_root, existing)
            except GazeboWorkbenchError as exc:
                errors.append(str(exc))
                continue
            meshes.append(
                {
                    "uri": raw,
                    "relative_path": resolved.relative_to(self.workspace_root).as_posix(),
                }
            )
        return {"meshes": meshes, "errors": errors}

    def _mesh_candidates(self, uri: str) -> tuple[Path, ...]:
        relative = uri
        if uri.startswith("model://"):
            relative = uri.removeprefix("model://")
        elif uri.startswith("file://"):
            relative = uri.removeprefix("file://")
        path = Path(relative)
        if path.is_absolute():
            return (path,)
        candidates = [self.workspace_root / path]
        parts = path.parts
        if parts:
            candidates.append(self.workspace_root / "generated" / "gazebo" / "models" / path)
        return tuple(candidates)

    def _active_owned_process(self) -> _OwnedGazeboProcess | None:
        owned = self.owned_process
        if owned is not None and owned.active:
            return owned
        return None

    def _state(self, owned: _OwnedGazeboProcess) -> str:
        if owned.stopped:
            return "stopped"
        if owned.poll() is not None:
            return "exited"
        if time.monotonic() - owned.started_at < 0.5:
            return "starting"
        return "running"

    def _wait_for_exit(self, owned: _OwnedGazeboProcess, timeout: float) -> bool:
        try:
            owned.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return owned.poll() is not None
        return owned.poll() is not None


def gazebo_tool_definitions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "gazebo_validate_sdf",
            "description": (
                "Validate one workspace .sdf or .world file with fixed Gazebo Harmonic "
                "`gz sdf -k` argv. Return validation errors for repair. Worlds should normally "
                "include gravity, ground collision, positive mass and valid inertia for dynamic "
                "links, collision geometry, friction, and joint limits/damping where applicable."
            ),
            "permission": GAZEBO_PERMISSION_READ,
            "args_model": GazeboValidateSdfArgs,
            "input_schema": _relative_path_schema(),
        },
        {
            "name": "gazebo_launch",
            "description": (
                "Request a confirmed launch of one validated workspace .sdf or .world file. "
                "Hephaestus revalidates and launches only fixed argv `gz sim -r WORLD_PATH` "
                "or headless `gz sim -r -s WORLD_PATH` with GZ_IP=127.0.0.1; no custom "
                "executable, flags, argv, or environment are accepted."
            ),
            "permission": GAZEBO_PERMISSION_LAUNCH,
            "args_model": GazeboLaunchArgs,
            "input_schema": _launch_schema(),
        },
        {
            "name": "gazebo_status",
            "description": "Inspect only the Gazebo process owned by this agent session.",
            "permission": GAZEBO_PERMISSION_READ,
            "args_model": GazeboEmptyArgs,
            "input_schema": _empty_schema(),
        },
        {
            "name": "gazebo_read_logs",
            "description": (
                "Read recent bounded logs from only this agent session's Gazebo process."
            ),
            "permission": GAZEBO_PERMISSION_READ,
            "args_model": GazeboReadLogsArgs,
            "input_schema": _read_logs_schema(),
        },
        {
            "name": "gazebo_stop",
            "description": "Stop only the Gazebo process group launched by this agent session.",
            "permission": GAZEBO_PERMISSION_LAUNCH,
            "args_model": GazeboEmptyArgs,
            "input_schema": _empty_schema(),
        },
    )


def _start_log_reader(owned: _OwnedGazeboProcess, source: str, stream: TextIO | None) -> None:
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


def _gazebo_gui_launch_env(parent_env: Mapping[str, str], workspace_root: Path) -> dict[str, str]:
    env = _gazebo_resource_env(parent_env, workspace_root)
    desktop_env = _desktop_session_env(parent_env)
    if "DISPLAY" not in desktop_env and "WAYLAND_DISPLAY" not in desktop_env:
        raise GazeboGuiUnavailable(
            "graphical session is unavailable for Gazebo GUI launch; run headless simulation "
            "instead or start Hephaestus from a graphical desktop session"
        )
    env.update(desktop_env)
    env["GZ_IP"] = "127.0.0.1"
    return env


def _gazebo_headless_launch_env(
    parent_env: Mapping[str, str], workspace_root: Path
) -> dict[str, str]:
    env = _gazebo_resource_env(parent_env, workspace_root)
    env["GZ_IP"] = "127.0.0.1"
    return env


def _gazebo_resource_env(parent_env: Mapping[str, str], workspace_root: Path) -> dict[str, str]:
    env = _sanitized_env(parent_env)
    resource_root = str(workspace_root.resolve(strict=True))
    env["GZ_SIM_RESOURCE_PATH"] = resource_root
    env["SDF_PATH"] = resource_root
    env["IGN_GAZEBO_RESOURCE_PATH"] = resource_root
    return env


def _looks_like_local_mesh(uri: str) -> bool:
    lowered = uri.lower()
    if not lowered.endswith(LOCAL_MESH_SUFFIXES):
        return False
    if "://" in uri and not uri.startswith(("model://", "file://")):
        return False
    return not uri.startswith("file:///")


def _resolve_workspace_mesh(workspace_root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else workspace_root / path
    if any(part == ".." for part in candidate.parts):
        raise GazeboWorkbenchError("mesh path uses parent traversal")
    current = Path(candidate.anchor) if candidate.is_absolute() else Path(".")
    parts = candidate.parts[1:] if candidate.is_absolute() else candidate.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise GazeboWorkbenchError(f"mesh path traverses a symlink: {current}")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(workspace_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise GazeboWorkbenchError("mesh path escapes startup directory") from exc
    if not resolved.is_file():
        raise GazeboWorkbenchError("mesh path is not a regular file")
    return resolved


def _desktop_session_env(parent_env: Mapping[str, str]) -> dict[str, str]:
    inherited: dict[str, str] = {}
    display = parent_env.get("DISPLAY")
    if _valid_env_value(display):
        inherited["DISPLAY"] = str(display)
    xauthority = parent_env.get("XAUTHORITY")
    if _valid_existing_file_path(xauthority):
        inherited["XAUTHORITY"] = str(xauthority)
    runtime_dir = parent_env.get("XDG_RUNTIME_DIR")
    if _valid_existing_dir_path(runtime_dir):
        inherited["XDG_RUNTIME_DIR"] = str(runtime_dir)
    wayland = parent_env.get("WAYLAND_DISPLAY")
    if _valid_wayland_display(wayland, inherited.get("XDG_RUNTIME_DIR")):
        inherited["WAYLAND_DISPLAY"] = str(wayland)
    bus = parent_env.get("DBUS_SESSION_BUS_ADDRESS")
    if _valid_dbus_session_bus(bus):
        inherited["DBUS_SESSION_BUS_ADDRESS"] = str(bus)
    session_type = parent_env.get("XDG_SESSION_TYPE")
    if _valid_simple_session_value(session_type):
        inherited["XDG_SESSION_TYPE"] = str(session_type)
    return inherited


def _valid_env_value(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if "\x00" in value:
        return False
    return len(value.encode("utf-8")) <= MAX_INHERITED_ENV_VALUE_BYTES


def _valid_existing_file_path(value: object) -> bool:
    if not _valid_env_value(value):
        return False
    path = Path(str(value))
    try:
        return path.is_absolute() and path.is_file()
    except OSError:
        return False


def _valid_existing_dir_path(value: object) -> bool:
    if not _valid_env_value(value):
        return False
    path = Path(str(value))
    try:
        return path.is_absolute() and path.is_dir()
    except OSError:
        return False


def _valid_wayland_display(value: object, runtime_dir: str | None) -> bool:
    if not _valid_env_value(value):
        return False
    text = str(value)
    if "/" not in text:
        return runtime_dir is not None
    path = Path(text)
    if not path.is_absolute():
        return False
    try:
        return path.exists()
    except OSError:
        return False


def _valid_dbus_session_bus(value: object) -> bool:
    if not _valid_env_value(value):
        return False
    text = str(value)
    if text.startswith("unix:path="):
        path_value = text.removeprefix("unix:path=").split(",", 1)[0]
        path = Path(path_value)
        try:
            return path.is_absolute() and path.exists()
        except OSError:
            return False
    return text.startswith(("unix:abstract=", "unix:tmpdir="))


def _valid_simple_session_value(value: object) -> bool:
    if not _valid_env_value(value):
        return False
    text = str(value)
    return all(ch.isascii() and (ch.isalnum() or ch in {"_", "-"}) for ch in text)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Relative .sdf or .world file beneath the startup directory.",
            }
        },
        "required": ["relative_path"],
    }


def _launch_schema() -> dict[str, Any]:
    schema = _relative_path_schema()
    schema["properties"] = {
        **schema["properties"],
        "headless": {
            "type": "boolean",
            "description": "Use fixed headless server mode instead of a GUI launch.",
        },
    }
    return schema


def _empty_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }


def _read_logs_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "max_lines": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LOG_RETURN_LINES,
                "description": "Maximum recent Gazebo log lines to return.",
            }
        },
        "required": [],
    }
