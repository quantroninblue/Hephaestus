"""Application-owned local workbench command parsing and dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from hephaestus.ai.types import ToolCall
from hephaestus.capabilities import CapabilityReport, CapabilityStatus, collect_capabilities
from hephaestus.tools.cadquery_build import CadQueryBuildContext
from hephaestus.tools.gazebo import GazeboWorkbench
from hephaestus.tools.ros import (
    ROS_LOG_RETURN_LINES,
    RosWorkbench,
    RosWorkbenchError,
    parse_decimal,
    parse_move_command,
    shell_split,
)
from hephaestus.tools.workspace import WorkspaceToolExecutor

ConfirmCallback = Callable[[dict[str, Any]], bool]

LEGAL_DOCUMENTS: tuple[tuple[str, str | None], ...] = (
    ("Repository", "https://github.com/quantroninblue/Hephaestus"),
    ("README/legal overview", "README.md"),
    ("SAFETY.md", "SAFETY.md"),
    ("PRIVACY.md", "PRIVACY.md"),
    ("SECURITY.md", "SECURITY.md"),
    ("RESPONSIBLE_USE.md", "RESPONSIBLE_USE.md"),
    ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ("LICENSE", "LICENSE"),
    ("NOTICE", "NOTICE"),
)
LEGAL_RESOURCE_DIR = "legal"


@dataclass(frozen=True, slots=True)
class LocalCommandResult:
    command: str
    status: str
    message: str
    payload: Mapping[str, Any]
    exit_requested: bool = False
    clear_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "message": self.message,
            "payload": dict(self.payload),
            "exit_requested": self.exit_requested,
            "clear_requested": self.clear_requested,
        }


@dataclass(slots=True)
class LocalWorkbenchSession:
    workspace_root: Path
    tool_executor: WorkspaceToolExecutor | None = None
    ros_workbench: RosWorkbench | None = None
    capabilities: CapabilityReport | None = None
    _closed: bool = False

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve(strict=True)
        if self.capabilities is None:
            self.capabilities = collect_capabilities()
        if self.tool_executor is None:
            self.tool_executor = WorkspaceToolExecutor(self.workspace_root)
        self._configure_tool_executor()
        if self.ros_workbench is None:
            self.ros_workbench = RosWorkbench(self.workspace_root)

    def help_text(self) -> str:
        return "\n".join(
            (
                "Commands:",
                "/help",
                "/clear",
                "/status",
                "/legal",
                "/exit",
                "/quit",
                "/cad build SOURCE",
                "/gazebo validate WORLD",
                "/gazebo launch WORLD",
                "/gazebo launch WORLD --headless",
                "/gazebo status",
                "/gazebo logs [MAX_LINES]",
                "/gazebo stop",
                "/ros status",
                "/ros topics",
                "/ros controllers",
                "/ros build ROS_WORKSPACE",
                "/ros launch ROS_WORKSPACE PACKAGE LAUNCH_FILE",
                "/ros logs [MAX_LINES]",
                "/ros stop",
                "/cmdvel LINEAR_X_MPS ANGULAR_Z_RADPS DURATION_SECONDS",
                "/move forward SPEED_MPS for DURATION_SECONDS",
                "/move backward SPEED_MPS for DURATION_SECONDS",
                "/move turn-left ANGULAR_SPEED_RADPS for DURATION_SECONDS",
                "/move turn-right ANGULAR_SPEED_RADPS for DURATION_SECONDS",
                "/stop",
            )
        )

    def dispatch(
        self,
        command_text: str,
        *,
        confirm: ConfirmCallback | None = None,
    ) -> LocalCommandResult:
        tokens = shell_split(command_text)
        if not tokens:
            return _result("empty", "error", "empty command")
        command = tokens[0]
        args = tokens[1:]
        try:
            if command in {"/exit", "/quit"}:
                self.close()
                return LocalCommandResult(command, "ok", "Exiting.", {}, exit_requested=True)
            if command == "/help":
                return _result(
                    command,
                    "ok",
                    self.help_text(),
                    {"commands": self.help_text().splitlines()[1:]},
                )
            if command == "/clear":
                return LocalCommandResult(
                    command,
                    "ok",
                    "Conversation cleared.",
                    {},
                    clear_requested=True,
                )
            if command == "/status":
                return self._status()
            if command == "/legal":
                return self._legal()
            if command == "/cad":
                return self._cad(args, confirm)
            if command == "/gazebo":
                return self._gazebo(args, confirm)
            if command == "/ros":
                return self._ros(args, confirm)
            if command == "/cmdvel":
                return self._cmdvel(args)
            if command == "/move":
                return self._move(args)
            if command == "/stop":
                return self._stop()
        except (RosWorkbenchError, ValueError) as exc:
            return _result(command, "error", str(exc))
        return _result(command, "error", f"unknown command: {command}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.ros_workbench is not None:
            self.ros_workbench.close()
        if self.tool_executor is not None:
            self.tool_executor.close()

    def _status(self) -> LocalCommandResult:
        payload: dict[str, Any] = {
            "workspace": str(self.workspace_root),
            "gazebo": (
                self.tool_executor.gazebo_context_summary()
                if self.tool_executor
                else "unavailable"
            ),
            "ros": self.ros_workbench.status() if self.ros_workbench else None,
        }
        return _result("/status", "ok", "Local workbench status.", payload)

    def quick_status(self) -> dict[str, Any]:
        return {
            "workspace": str(self.workspace_root),
            "gazebo": (
                self.tool_executor.gazebo_context_summary()
                if self.tool_executor
                else "unavailable"
            ),
            "ros": self.ros_workbench.quick_status() if self.ros_workbench else None,
        }

    def _legal(self) -> LocalCommandResult:
        documents = [
            {"label": label, "path": path, "available": self._legal_path_available(path)}
            for label, path in LEGAL_DOCUMENTS
        ]
        lines = [
            (
                f"- {item['label']}: {item['path']}"
                if item["path"] is not None
                else f"- {item['label']}"
            )
            for item in documents
        ]
        return _result(
            "/legal",
            "ok",
            "\n".join(lines),
            {
                "repository": "https://github.com/quantroninblue/Hephaestus",
                "documents": documents,
            },
        )

    def _legal_path_available(self, path: str | None) -> bool:
        if path is None or path.startswith("https://"):
            return True
        candidate = self.workspace_root / path
        if candidate.is_file():
            return True
        return _packaged_legal_resource_available(path)

    def _cad(self, args: list[str], confirm: ConfirmCallback | None) -> LocalCommandResult:
        if len(args) != 2 or args[0] != "build":
            return _result("/cad", "error", "expected /cad build SOURCE")
        assert self.tool_executor is not None
        result = self.tool_executor.execute(
            ToolCall("local_cad", "cadquery_build", {"relative_path": args[1]}),
            allow_mutation=True,
            confirm=confirm,
        )
        return _tool_result("/cad build", result)

    def _gazebo(self, args: list[str], confirm: ConfirmCallback | None) -> LocalCommandResult:
        if not args:
            return _result(
                "/gazebo",
                "error",
                "expected /gazebo validate, launch, status, logs, or stop",
            )
        assert self.tool_executor is not None
        sub = args[0]
        if sub == "validate" and len(args) == 2:
            result = self.tool_executor.execute(
                ToolCall(
                    "local_gazebo_validate",
                    "gazebo_validate_sdf",
                    {"relative_path": args[1]},
                ),
                allow_mutation=True,
            )
            return _tool_result("/gazebo validate", result)
        if sub == "launch" and len(args) in {2, 3}:
            headless = False
            if len(args) == 3:
                if args[2] != "--headless":
                    return _result(
                        "/gazebo launch",
                        "error",
                        "expected /gazebo launch WORLD [--headless]",
                    )
                headless = True
            result = self.tool_executor.execute(
                ToolCall(
                    "local_gazebo_launch",
                    "gazebo_launch",
                    {"relative_path": args[1], "headless": headless},
                ),
                allow_mutation=True,
                confirm=confirm,
            )
            return _tool_result("/gazebo launch", result)
        if sub == "status" and len(args) == 1:
            return _tool_result(
                "/gazebo status",
                self.tool_executor.execute(
                    ToolCall("local_gazebo_status", "gazebo_status", {}),
                    allow_mutation=True,
                ),
            )
        if sub == "logs" and len(args) in {1, 2}:
            max_lines = _optional_int(args[1], "MAX_LINES") if len(args) == 2 else 100
            return _tool_result(
                "/gazebo logs",
                self.tool_executor.execute(
                    ToolCall("local_gazebo_logs", "gazebo_read_logs", {"max_lines": max_lines}),
                    allow_mutation=True,
                ),
            )
        if sub == "stop" and len(args) == 1:
            return _tool_result(
                "/gazebo stop",
                self.tool_executor.execute(
                    ToolCall("local_gazebo_stop", "gazebo_stop", {}),
                    allow_mutation=True,
                ),
            )
        return _result("/gazebo", "error", "invalid /gazebo syntax")

    def _ros(self, args: list[str], confirm: ConfirmCallback | None) -> LocalCommandResult:
        if self.ros_workbench is None:
            return _result("/ros", "error", "ROS workbench is unavailable")
        if not args:
            return _result(
                "/ros",
                "error",
                "expected /ros status, topics, controllers, build, launch, logs, or stop",
            )
        sub = args[0]
        if sub == "status" and len(args) == 1:
            return _result("/ros status", "ok", "ROS status.", self.ros_workbench.status())
        if sub == "topics" and len(args) == 1:
            return _result("/ros topics", "ok", "ROS topics.", self.ros_workbench.topics())
        if sub == "controllers" and len(args) == 1:
            return _result(
                "/ros controllers",
                "ok",
                "ROS controllers.",
                self.ros_workbench.controllers(),
            )
        if sub == "build" and len(args) == 2:
            workspace = self._resolve_relative_workspace(args[1])
            payload = self.ros_workbench.build(workspace, args[1], confirm=confirm)
            status = "ok" if payload.get("success") else "error"
            return _result("/ros build", status, _json_message(payload), payload)
        if sub == "launch" and len(args) == 4:
            workspace = self._resolve_relative_workspace(args[1])
            payload = self.ros_workbench.launch(
                workspace,
                args[1],
                args[2],
                args[3],
                confirm=confirm,
            )
            return _result(
                "/ros launch",
                "ok" if payload.get("success") else "error",
                _json_message(payload),
                payload,
            )
        if sub == "logs" and len(args) in {1, 2}:
            max_lines = (
                _optional_int(args[1], "MAX_LINES")
                if len(args) == 2
                else ROS_LOG_RETURN_LINES
            )
            payload = self.ros_workbench.logs(max_lines)
            return _result("/ros logs", "ok", "ROS logs.", payload)
        if sub == "stop" and len(args) == 1:
            payload = self.ros_workbench.stop_launch()
            return _result("/ros stop", "ok", "ROS launch stopped.", payload)
        return _result("/ros", "error", "invalid /ros syntax")

    def _cmdvel(self, args: list[str]) -> LocalCommandResult:
        if len(args) != 3:
            return _result(
                "/cmdvel",
                "error",
                "expected /cmdvel LINEAR_X_MPS ANGULAR_Z_RADPS DURATION_SECONDS",
            )
        if self.ros_workbench is None:
            return _result("/cmdvel", "error", "ROS workbench is unavailable")
        linear = parse_decimal(args[0])
        angular = parse_decimal(args[1])
        duration = parse_decimal(args[2])
        payload = self.ros_workbench.cmd_vel(linear, angular, duration)
        return _result("/cmdvel", "ok", "Velocity command completed.", payload)

    def _move(self, args: list[str]) -> LocalCommandResult:
        if self.ros_workbench is None:
            return _result("/move", "error", "ROS workbench is unavailable")
        linear, angular, duration = parse_move_command(args)
        payload = self.ros_workbench.cmd_vel(linear, angular, duration)
        return _result("/move", "ok", "Move command completed.", payload)

    def _stop(self) -> LocalCommandResult:
        if self.ros_workbench is None:
            return _result("/stop", "error", "ROS workbench is unavailable")
        payload = self.ros_workbench.stop_motion()
        return _result("/stop", "ok", "Zero velocity command sent.", payload)

    def _resolve_relative_workspace(self, value: str) -> Path:
        if not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
            raise RosWorkbenchError("ROS_WORKSPACE must be a safe relative path")
        return self.workspace_root / value

    def _configure_tool_executor(self) -> None:
        if self.tool_executor is None:
            return
        capability_map = (self.capabilities or CapabilityReport(())).by_id()
        cad = capability_map.get("cad")
        if cad is not None and cad.status is CapabilityStatus.ACTIVE:
            self.tool_executor.cadquery_context = CadQueryBuildContext(self.workspace_root)
        gazebo = capability_map.get("gazebo")
        if gazebo is not None and gazebo.status is CapabilityStatus.ACTIVE:
            self.tool_executor.gazebo_workbench = GazeboWorkbench(
                self.workspace_root, gazebo_version=gazebo.detected_version
            )


def standalone_cad_build(
    workspace_root: Path,
    source: str,
    *,
    confirm: ConfirmCallback | None,
) -> dict[str, Any]:
    executor = WorkspaceToolExecutor(
        workspace_root,
        cadquery_context=CadQueryBuildContext(workspace_root),
    )
    return executor.execute(
        ToolCall("standalone_cad", "cadquery_build", {"relative_path": source}),
        allow_mutation=True,
        confirm=confirm,
    )


def standalone_gazebo_validate(workspace_root: Path, world: str) -> dict[str, Any]:
    caps = collect_capabilities().by_id()
    gazebo = caps.get("gazebo")
    workbench = GazeboWorkbench(
        workspace_root,
        gazebo_version=gazebo.detected_version if gazebo is not None else None,
    )
    executor = WorkspaceToolExecutor(workspace_root, gazebo_workbench=workbench)
    return executor.execute(
        ToolCall("standalone_gazebo_validate", "gazebo_validate_sdf", {"relative_path": world}),
        allow_mutation=True,
    )


def _packaged_legal_resource_available(path: str) -> bool:
    try:
        return resources.files("hephaestus").joinpath(LEGAL_RESOURCE_DIR, path).is_file()
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        return False


def _tool_result(command: str, result: Mapping[str, Any]) -> LocalCommandResult:
    status = "ok" if result.get("status") == "ok" else "error"
    if status == "error":
        return _result(command, "error", str(result.get("error", "tool failed")), result)
    payload = result.get("result")
    return _result(
        command,
        "ok",
        _json_message(payload),
        payload if isinstance(payload, Mapping) else result,
    )


def _result(
    command: str, status: str, message: str, payload: Mapping[str, Any] | None = None
) -> LocalCommandResult:
    return LocalCommandResult(command, status, message, payload or {})


def _optional_int(value: str, label: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def _json_message(payload: object) -> str:
    if isinstance(payload, Mapping):
        if "message" in payload:
            return str(payload["message"])
        if payload.get("success") is True:
            return "Operation completed."
        if payload.get("valid") is True:
            return "Validation passed."
        if payload.get("valid") is False:
            return "Validation failed."
    return json.dumps(payload, sort_keys=True)
