"""Bounded workspace tools for the default Hephaestus agent session."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hephaestus.ai.types import ToolCall, ToolDefinition
from hephaestus.tools.cadquery_build import (
    CADQUERY_PERMISSION_BUILD,
    CadQueryBuildArgs,
    CadQueryBuildContext,
    CadQueryBuildError,
    cadquery_tool_definition,
)
from hephaestus.tools.gazebo import (
    GAZEBO_PERMISSION_LAUNCH,
    GAZEBO_PERMISSION_READ,
    GazeboEmptyArgs,
    GazeboGuiUnavailable,
    GazeboLaunchArgs,
    GazeboReadLogsArgs,
    GazeboValidateSdfArgs,
    GazeboWorkbench,
    GazeboWorkbenchError,
    gazebo_tool_definitions,
)

MAX_LIST_LIMIT = 200
DEFAULT_LIST_LIMIT = 50
MAX_READ_BYTES = 64_000
MAX_WRITE_BYTES = 64_000
MAX_RELATIVE_PATH_LENGTH = 300

WORKSPACE_PERMISSION_READ = "workspace read"
WORKSPACE_PERMISSION_WRITE = "workspace write"


class WorkspaceToolError(RuntimeError):
    """Raised when a workspace operation is rejected at the application boundary."""


class WorkspaceToolConflict(WorkspaceToolError):
    """Raised when an edit cannot apply to the current file content."""


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(default=".", max_length=MAX_RELATIVE_PATH_LENGTH)
    limit: int = Field(default=DEFAULT_LIST_LIMIT, ge=1, le=MAX_LIST_LIMIT)


class ReadTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH_LENGTH)


class WriteTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH_LENGTH)
    content: str = Field(max_length=MAX_WRITE_BYTES)


class ReplaceTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    relative_path: str = Field(min_length=1, max_length=MAX_RELATIVE_PATH_LENGTH)
    old_text: str = Field(min_length=1, max_length=MAX_WRITE_BYTES)
    new_text: str = Field(max_length=MAX_WRITE_BYTES)


@dataclass(frozen=True, slots=True)
class WorkspaceTool:
    name: str
    description: str
    permission: str
    args_model: type[BaseModel]
    input_schema: dict[str, Any]

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name, self.description, self.input_schema)


def _gazebo_workspace_tools() -> tuple[WorkspaceTool, ...]:
    return tuple(
        WorkspaceTool(
            str(item["name"]),
            str(item["description"]),
            str(item["permission"]),
            item["args_model"],
            item["input_schema"],
        )
        for item in gazebo_tool_definitions()
    )


def _cadquery_workspace_tool() -> WorkspaceTool:
    item = cadquery_tool_definition()
    return WorkspaceTool(
        str(item["name"]),
        str(item["description"]),
        str(item["permission"]),
        item["args_model"],
        item["input_schema"],
    )


@dataclass(slots=True)
class WorkspaceToolExecutor:
    startup_directory: Path
    installed_source_root: Path = Path(__file__).resolve().parents[1]
    gazebo_workbench: GazeboWorkbench | None = None
    cadquery_context: CadQueryBuildContext | None = None

    def __post_init__(self) -> None:
        self.startup_directory = self.startup_directory.resolve(strict=True)
        self.installed_source_root = self.installed_source_root.resolve(strict=True)

    def definitions(
        self, *, allow_mutation: bool, tool_names: Iterable[str] | None = None
    ) -> tuple[ToolDefinition, ...]:
        selected = frozenset(tool_names) if tool_names is not None else None
        tools = self._available_tools()
        return tuple(
            tool.definition()
            for tool in tools
            if (
                allow_mutation
                or tool.permission not in {WORKSPACE_PERMISSION_WRITE, GAZEBO_PERMISSION_LAUNCH}
            )
            and (selected is None or tool.name in selected)
        )

    def validate_batch(
        self, calls: Sequence[ToolCall], *, allow_mutation: bool
    ) -> dict[str, Any] | None:
        available = {tool.name: tool for tool in self._available_tools()}
        exposed = {
            definition.name for definition in self.definitions(allow_mutation=allow_mutation)
        }
        pending_writes: set[str] = set()
        for call in calls:
            tool = available.get(call.name)
            if tool is None or call.name not in exposed:
                return _failure(call.name, "unknown or unavailable workspace tool", None, call.id)
            if (
                tool.permission
                in {WORKSPACE_PERMISSION_WRITE, GAZEBO_PERMISSION_LAUNCH, CADQUERY_PERMISSION_BUILD}
                and not allow_mutation
            ):
                return _failure(
                    call.name, "workspace mutation permission denied", tool.permission, call.id
                )
            try:
                args = tool.args_model.model_validate(dict(call.arguments))
                self._preflight_args(tool.name, args, pending_writes=pending_writes)
                if isinstance(args, WriteTextArgs):
                    pending_writes.add(self._relative(self._resolve_write_path(args.relative_path)))
                elif isinstance(args, ReplaceTextArgs):
                    pending_writes.add(
                        self._relative(self._resolve_existing_path(args.relative_path))
                    )
            except ValidationError as exc:
                return _failure(
                    call.name,
                    "invalid arguments",
                    tool.permission,
                    call.id,
                    _safe_validation_errors(exc),
                )
            except WorkspaceToolError as exc:
                return _failure(call.name, str(exc), tool.permission, call.id)
            except GazeboWorkbenchError as exc:
                return _failure(call.name, str(exc), tool.permission, call.id)
        return None

    def is_read_only_batch(self, calls: Sequence[ToolCall]) -> bool:
        available = {tool.name: tool for tool in self._available_tools()}
        return all(
            (tool := available.get(call.name)) is not None
            and tool.permission in {WORKSPACE_PERMISSION_READ, GAZEBO_PERMISSION_READ}
            for call in calls
        )

    def execute(
        self,
        call: ToolCall,
        *,
        allow_mutation: bool,
        confirm: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]:
        tool = {item.name: item for item in self._available_tools()}.get(call.name)
        if tool is None:
            return _failure(call.name, "unknown workspace tool", None, call.id)
        if (
            tool.permission
            in {WORKSPACE_PERMISSION_WRITE, GAZEBO_PERMISSION_LAUNCH, CADQUERY_PERMISSION_BUILD}
            and not allow_mutation
        ):
            return _failure(
                call.name, "workspace mutation permission denied", tool.permission, call.id
            )
        try:
            args = tool.args_model.model_validate(dict(call.arguments))
            self._preflight_args(tool.name, args)
            if isinstance(args, ListFilesArgs):
                payload = self._list_files(args)
            elif isinstance(args, ReadTextArgs):
                payload = self._read_text(args)
            elif isinstance(args, WriteTextArgs):
                payload = self._write_text(args, confirm)
            elif isinstance(args, ReplaceTextArgs):
                payload = self._replace_text(args, confirm)
            elif isinstance(args, GazeboValidateSdfArgs):
                payload = self._gazebo_validate_sdf(args)
            elif isinstance(args, GazeboLaunchArgs):
                payload = self._gazebo_launch(args, confirm)
            elif isinstance(args, GazeboReadLogsArgs):
                payload = self._gazebo_read_logs(args)
            elif isinstance(args, GazeboEmptyArgs) and tool.name == "gazebo_status":
                payload = self._gazebo_status()
            elif isinstance(args, GazeboEmptyArgs) and tool.name == "gazebo_stop":
                payload = self._gazebo_stop()
            elif isinstance(args, CadQueryBuildArgs):
                payload = self._cadquery_build(args, confirm)
            else:
                return _failure(tool.name, "unsupported workspace tool arguments", tool.permission)
        except ValidationError as exc:
            return _failure(
                tool.name,
                "invalid arguments",
                tool.permission,
                call.id,
                _safe_validation_errors(exc),
            )
        except WorkspaceToolConflict as exc:
            return _failure(
                tool.name,
                str(exc),
                tool.permission,
                call.id,
                {"code": "read_write_conflict"},
            )
        except GazeboGuiUnavailable as exc:
            return _failure(
                tool.name,
                str(exc),
                tool.permission,
                call.id,
                {"code": "gui_unavailable"},
            )
        except (WorkspaceToolError, GazeboWorkbenchError) as exc:
            return _failure(tool.name, str(exc), tool.permission, call.id)
        except CadQueryBuildError as exc:
            return _failure(tool.name, str(exc), tool.permission, call.id)
        except Exception:
            return _failure(tool.name, "workspace tool execution failed", tool.permission, call.id)
        result = {
            "status": "ok",
            "tool": tool.name,
            "permission": tool.permission,
            "result": payload,
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result

    def close(self) -> None:
        if self.gazebo_workbench is not None:
            self.gazebo_workbench.close()

    def gazebo_context_summary(self) -> str:
        if self.gazebo_workbench is None:
            return "gazebo=unavailable"
        return self.gazebo_workbench.context_summary()

    def _available_tools(self) -> tuple[WorkspaceTool, ...]:
        tools = _TOOLS
        if self.cadquery_context is not None:
            tools = (*tools, _CADQUERY_TOOL)
        if self.gazebo_workbench is not None:
            tools = (*tools, *_GAZEBO_TOOLS)
        return tools

    def _preflight_args(
        self,
        tool_name: str,
        args: BaseModel,
        *,
        pending_writes: set[str] | None = None,
    ) -> None:
        path_value = getattr(args, "relative_path", None)
        if isinstance(path_value, str):
            if tool_name == "workspace_list_files":
                self._resolve_list_path(path_value)
            elif tool_name == "workspace_write_text":
                self._resolve_write_path(path_value)
            elif tool_name == "workspace_replace_text":
                self._resolve_existing_path(path_value)
            elif tool_name == "cadquery_build":
                self._preflight_cadquery_path(path_value)
            elif tool_name.startswith("gazebo_"):
                self._preflight_gazebo_path(path_value, pending_writes or set())
            else:
                self._resolve_existing_path(path_value)
        content = getattr(args, "content", None)
        if isinstance(content, str) and len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise WorkspaceToolError("content exceeds workspace write size limit")
        for field_name in ("old_text", "new_text"):
            value = getattr(args, field_name, None)
            if isinstance(value, str) and len(value.encode("utf-8")) > MAX_WRITE_BYTES:
                raise WorkspaceToolError(f"{field_name} exceeds workspace write size limit")

    def _list_files(self, args: ListFilesArgs) -> dict[str, Any]:
        base = self._resolve_list_path(args.relative_path)
        if not base.is_dir():
            raise WorkspaceToolError("list path is not a directory")
        entries: list[dict[str, Any]] = []
        truncated = False
        for current, dirnames, filenames in os.walk(base, followlinks=False):
            current_path = Path(current)
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not self._is_protected_relative_path(self._lexical_relative(current_path / name))
                and not (current_path / name).is_symlink()
            ]
            for name in sorted(filenames):
                candidate = current_path / name
                try:
                    relative = self._relative(candidate)
                    resolved = candidate.resolve(strict=True)
                    self._reject_unsafe_resolved(relative, resolved)
                    stat = resolved.stat()
                except (OSError, WorkspaceToolError):
                    continue
                if not resolved.is_file():
                    continue
                entries.append(
                    {
                        "relative_path": relative,
                        "bytes": stat.st_size,
                    }
                )
                if len(entries) >= args.limit:
                    truncated = True
                    break
            if truncated:
                break
        return {
            "root": str(self.startup_directory),
            "relative_path": self._relative(base) or ".",
            "files": entries,
            "count": len(entries),
            "truncated": truncated,
            "limit": args.limit,
        }

    def _read_text(self, args: ReadTextArgs) -> dict[str, Any]:
        path = self._resolve_existing_path(args.relative_path)
        if not path.is_file():
            raise WorkspaceToolError("read path is not a regular file")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise WorkspaceToolError("file exceeds workspace read size limit")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceToolError("file is not valid UTF-8 text") from exc
        encoded_size = len(content.encode("utf-8"))
        if encoded_size > MAX_READ_BYTES:
            raise WorkspaceToolError("file exceeds workspace read size limit")
        return {
            "relative_path": self._relative(path),
            "bytes": encoded_size,
            "content": content,
        }

    def _write_text(
        self, args: WriteTextArgs, confirm: Callable[[dict[str, Any]], bool] | None
    ) -> dict[str, Any]:
        target = self._resolve_write_path(args.relative_path)
        content_bytes = args.content.encode("utf-8")
        if len(content_bytes) > MAX_WRITE_BYTES:
            raise WorkspaceToolError("content exceeds workspace write size limit")
        operation = "replacing" if target.exists() else "creating"
        preview = {
            "status": "preview",
            "tool": "workspace_write_text",
            "permission": WORKSPACE_PERMISSION_WRITE,
            "result": {
                "relative_path": self._relative(target),
                "operation": operation,
                "bytes": len(content_bytes),
            },
        }
        if confirm is None or not confirm(preview):
            return {
                "relative_path": self._relative(target),
                "operation": operation,
                "committed": False,
                "message": "write was previewed but not applied",
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=False,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except Exception:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise
        committed_size = target.stat().st_size
        return {
            "relative_path": self._relative(target),
            "operation": "created" if operation == "creating" else "replaced",
            "bytes_written": committed_size,
            "committed": True,
        }

    def _replace_text(
        self, args: ReplaceTextArgs, confirm: Callable[[dict[str, Any]], bool] | None
    ) -> dict[str, Any]:
        target = self._resolve_existing_path(args.relative_path)
        if not target.is_file():
            raise WorkspaceToolError("replace target is not a regular file")
        original_bytes = target.read_bytes()
        if len(original_bytes) > MAX_READ_BYTES:
            raise WorkspaceToolError("file exceeds workspace read size limit")
        before_fingerprint = _bytes_fingerprint(original_bytes)
        try:
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceToolError("file is not valid UTF-8 text") from exc
        if not args.old_text:
            raise WorkspaceToolError("old_text must not be empty")
        old_bytes = args.old_text.encode("utf-8")
        new_bytes = args.new_text.encode("utf-8")
        if len(old_bytes) > MAX_WRITE_BYTES:
            raise WorkspaceToolError("old_text exceeds workspace write size limit")
        if len(new_bytes) > MAX_WRITE_BYTES:
            raise WorkspaceToolError("new_text exceeds workspace write size limit")
        matches = original.count(args.old_text)
        if matches == 0:
            raise WorkspaceToolConflict("replacement conflict: old_text occurs zero times")
        if matches > 1:
            raise WorkspaceToolError("replacement is ambiguous: old_text occurs more than once")
        updated = original.replace(args.old_text, args.new_text, 1)
        updated_bytes = updated.encode("utf-8")
        if len(updated_bytes) > MAX_WRITE_BYTES:
            raise WorkspaceToolError("resulting file exceeds workspace write size limit")
        preview = {
            "status": "preview",
            "tool": "workspace_replace_text",
            "permission": WORKSPACE_PERMISSION_WRITE,
            "result": {
                "relative_path": self._relative(target),
                "operation": "targeted replacement",
                "old_byte_count": len(old_bytes),
                "new_byte_count": len(new_bytes),
                "resulting_file_byte_count": len(updated_bytes),
                "before_fingerprint": before_fingerprint,
            },
        }
        if confirm is None or not confirm(preview):
            return {
                "relative_path": self._relative(target),
                "operation": "targeted replacement",
                "committed": False,
                "message": "replacement was previewed but not applied",
                "before_fingerprint": before_fingerprint,
            }
        current_bytes = target.read_bytes()
        if _bytes_fingerprint(current_bytes) != before_fingerprint:
            raise WorkspaceToolError("file changed after replacement preview")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=False,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(updated_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        except Exception as exc:
            with suppress(OSError):
                temp_path.unlink(missing_ok=True)
            raise WorkspaceToolError("workspace persistence failed") from exc
        committed_bytes = target.read_bytes()
        committed_fingerprint = _bytes_fingerprint(committed_bytes)
        return {
            "relative_path": self._relative(target),
            "operation": "targeted replacement",
            "occurrences_replaced": 1,
            "old_byte_count": len(old_bytes),
            "new_byte_count": len(new_bytes),
            "bytes_written": len(committed_bytes),
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": committed_fingerprint,
            "committed": True,
        }

    def _gazebo_validate_sdf(self, args: GazeboValidateSdfArgs) -> dict[str, Any]:
        workbench = self._require_gazebo_workbench()
        path = self._resolve_existing_path(args.relative_path)
        return workbench.validate_sdf(path, self._relative(path))

    def _gazebo_launch(
        self, args: GazeboLaunchArgs, confirm: Callable[[dict[str, Any]], bool] | None
    ) -> dict[str, Any]:
        workbench = self._require_gazebo_workbench()
        path = self._resolve_existing_path(args.relative_path)
        return workbench.launch(path, self._relative(path), headless=args.headless, confirm=confirm)

    def _gazebo_status(self) -> dict[str, Any]:
        return self._require_gazebo_workbench().status()

    def _gazebo_read_logs(self, args: GazeboReadLogsArgs) -> dict[str, Any]:
        return self._require_gazebo_workbench().read_logs(args.max_lines)

    def _gazebo_stop(self) -> dict[str, Any]:
        return self._require_gazebo_workbench().stop()

    def _cadquery_build(
        self, args: CadQueryBuildArgs, confirm: Callable[[dict[str, Any]], bool] | None
    ) -> dict[str, Any]:
        context = self._require_cadquery_context()
        path = self._resolve_existing_path(args.relative_path)
        return context.build(path, self._relative(path), confirm=confirm)

    def _require_gazebo_workbench(self) -> GazeboWorkbench:
        if self.gazebo_workbench is None:
            raise GazeboWorkbenchError("Gazebo capability is not active for this agent session")
        return self.gazebo_workbench

    def _require_cadquery_context(self) -> CadQueryBuildContext:
        if self.cadquery_context is None:
            raise CadQueryBuildError("CadQuery capability is not active for this agent session")
        return self.cadquery_context

    def _preflight_cadquery_path(self, value: str) -> None:
        path = self._resolve_existing_path(value)
        if path.suffix.lower() != ".py":
            raise CadQueryBuildError("cadquery_build accepts only .py source files")

    def _preflight_gazebo_path(self, value: str, pending_writes: set[str]) -> None:
        relative = _clean_relative_path(value)
        if self._is_protected_relative_path(relative):
            raise WorkspaceToolError("path is protected")
        suffix = Path(relative).suffix.lower()
        if suffix not in {".sdf", ".world"}:
            raise GazeboWorkbenchError("Gazebo tools accept only .sdf or .world files")
        if relative in pending_writes:
            self._resolve_write_path(relative)
            return
        self._resolve_existing_path(relative)

    def _resolve_existing_path(self, value: str) -> Path:
        relative = _clean_relative_path(value)
        if self._is_protected_relative_path(relative):
            raise WorkspaceToolError("path is protected")
        candidate = self.startup_directory / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceToolError("path does not exist") from exc
        self._reject_unsafe_resolved(relative, resolved)
        return resolved

    def _resolve_list_path(self, value: str) -> Path:
        relative = _clean_list_relative_path(value)
        if self._is_protected_relative_path(relative):
            raise WorkspaceToolError("path is protected")
        candidate = self.startup_directory / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceToolError("path does not exist") from exc
        self._reject_unsafe_resolved(relative, resolved)
        return resolved

    def _resolve_write_path(self, value: str) -> Path:
        relative = _clean_relative_path(value)
        if self._is_protected_relative_path(relative):
            raise WorkspaceToolError("path is protected")
        parent = self.startup_directory / Path(relative).parent
        try:
            resolved_parent = parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceToolError("parent directory does not exist") from exc
        self._reject_unsafe_resolved(self._relative(resolved_parent), resolved_parent)
        target = resolved_parent / Path(relative).name
        if target.exists():
            resolved_target = target.resolve(strict=True)
            self._reject_unsafe_resolved(relative, resolved_target)
            if not resolved_target.is_file():
                raise WorkspaceToolError("write target is not a regular file")
            return resolved_target
        return target

    def _reject_unsafe_resolved(self, relative: str, resolved: Path) -> None:
        if not _is_relative_to(resolved, self.startup_directory):
            raise WorkspaceToolError("path escapes startup directory")
        if _is_relative_to(resolved, self.installed_source_root):
            raise WorkspaceToolError("installed Hephaestus source is protected")
        if self._is_protected_relative_path(relative):
            raise WorkspaceToolError("path is protected")
        try:
            mode_path = resolved if resolved.exists() else resolved.parent
            if not mode_path.is_dir() and not mode_path.is_file():
                raise WorkspaceToolError("path is not a regular file or directory")
        except OSError as exc:
            raise WorkspaceToolError("path could not be inspected") from exc

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(self.startup_directory).as_posix()
        except ValueError as exc:
            raise WorkspaceToolError("path escapes startup directory") from exc

    def _lexical_relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.startup_directory).as_posix()
        except ValueError as exc:
            raise WorkspaceToolError("path escapes startup directory") from exc

    def _is_protected_relative_path(self, value: str) -> bool:
        parts = tuple(part.lower() for part in Path(value).parts if part not in {"", "."})
        if any(part in {".git", ".venv", "project_state.txt"} for part in parts):
            return True
        filename = parts[-1] if parts else ""
        if filename == ".env" or filename.startswith(".env."):
            return True
        credential_markers = (
            "credential",
            "credentials",
            "secret",
            "secrets",
            "token",
            "password",
            "api_key",
            "apikey",
            "id_rsa",
            "id_dsa",
            "id_ed25519",
        )
        return any(marker in filename for marker in credential_markers)


def _clean_relative_path(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise WorkspaceToolError("path must not be empty")
    path = Path(raw)
    if path.is_absolute():
        raise WorkspaceToolError("absolute paths are not allowed")
    parts = path.parts
    if any(part == ".." for part in parts):
        raise WorkspaceToolError("parent traversal is not allowed")
    if any(part == "" for part in parts):
        raise WorkspaceToolError("path is malformed")
    return "." if raw == "." else str(path)


def _clean_list_relative_path(value: str) -> str:
    if value in {"", "."}:
        return "."
    return _clean_relative_path(value)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _failure(
    tool: str,
    message: str,
    permission: str | None,
    tool_call_id: str | None = None,
    evidence: Any | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "error",
        "tool": tool,
        "error": message,
    }
    if permission is not None:
        result["permission"] = permission
    if tool_call_id is not None:
        result["tool_call_id"] = tool_call_id
    if evidence is not None:
        result["evidence"] = evidence
    return result


def _safe_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()]


def _list_files_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Relative directory beneath the startup directory.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_LIST_LIMIT,
                "description": "Maximum number of files to return.",
            },
        },
        "required": [],
    }


def _read_text_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Relative UTF-8 text file beneath the startup directory.",
            }
        },
        "required": ["relative_path"],
    }


def _write_text_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Relative UTF-8 text file beneath the startup directory.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 file content to create or replace.",
            },
        },
        "required": ["relative_path", "content"],
    }


def _replace_text_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Relative existing UTF-8 text file beneath the startup directory.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact existing text to replace. Must occur exactly once.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text. May be empty for intentional deletion.",
            },
        },
        "required": ["relative_path", "old_text", "new_text"],
    }


def _bytes_fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


_TOOLS = (
    WorkspaceTool(
        "workspace_list_files",
        "List regular files beneath the startup directory with a bounded result count.",
        WORKSPACE_PERMISSION_READ,
        ListFilesArgs,
        _list_files_input_schema(),
    ),
    WorkspaceTool(
        "workspace_read_text",
        "Read one bounded UTF-8 text file beneath the startup directory.",
        WORKSPACE_PERMISSION_READ,
        ReadTextArgs,
        _read_text_input_schema(),
    ),
    WorkspaceTool(
        "workspace_write_text",
        "Create or replace one UTF-8 text file beneath the startup directory after "
        "application-controlled user confirmation.",
        WORKSPACE_PERMISSION_WRITE,
        WriteTextArgs,
        _write_text_input_schema(),
    ),
    WorkspaceTool(
        "workspace_replace_text",
        "Repair an existing workspace file with one exact targeted replacement after "
        "application-controlled user confirmation. Use workspace_read_text before editing "
        "an existing file. Use workspace_replace_text for small targeted edits and prefer "
        "it over workspace_write_text when only part of an existing file needs changing. "
        "old_text must match exactly once; new_text may be empty for deletion. Perform "
        "dependent edits, validation, and launch sequentially: read, edit, observe the "
        "committed result, validate, then launch.",
        WORKSPACE_PERMISSION_WRITE,
        ReplaceTextArgs,
        _replace_text_input_schema(),
    ),
)

_GAZEBO_TOOLS = _gazebo_workspace_tools()
_CADQUERY_TOOL = _cadquery_workspace_tool()
