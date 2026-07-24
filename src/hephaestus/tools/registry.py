"""Provider-independent engineering tool registry."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from hephaestus.ai.types import ToolCall, ToolDefinition
from hephaestus.changes import fingerprint_project
from hephaestus.persistence import load_project


class PermissionClass(StrEnum):
    READ_ONLY = "read-only"
    PROJECT_MUTATION = "project mutation"


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ToolContext:
    project_path: Path | None = None
    expected_fingerprint: str | None = None

    def require_project_path(self) -> Path:
        if self.project_path is None:
            raise ToolExecutionError("an active project is required")
        return self.project_path


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


@dataclass(frozen=True, slots=True)
class ToolWorkflowPolicy:
    name: str
    allowed_tool_names: frozenset[str]
    allowed_permissions: frozenset[PermissionClass]
    expected_operation_family: str | None = None
    expected_terminal_tools: frozenset[str] = frozenset()


def mobile_robot_design_tool_policy() -> ToolWorkflowPolicy:
    return ToolWorkflowPolicy(
        name="design",
        allowed_tool_names=frozenset(
            {"project.get_summary", "project.validate", "project_create_mobile_robot"}
        ),
        allowed_permissions=frozenset(
            {PermissionClass.READ_ONLY, PermissionClass.PROJECT_MUTATION}
        ),
        expected_operation_family="mobile_robot_design",
        expected_terminal_tools=frozenset({"project_create_mobile_robot"}),
    )


ToolHandler = Callable[[BaseModel, ToolContext], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    name: str
    description: str
    permission: PermissionClass
    args_model: type[BaseModel]
    handler: ToolHandler
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    input_schema: Mapping[str, Any] | None = None
    model_facing: bool = True
    preview_handler: ToolHandler | None = None
    operation_family: str | None = None

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            input_schema=dict(self.input_schema)
            if self.input_schema is not None
            else self.args_model.model_json_schema(),
        )


@dataclass(slots=True)
class ToolRegistry:
    tools: dict[str, RegisteredTool] = field(default_factory=dict)

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def list_tools(
        self,
        *,
        allow_mutation: bool = False,
        model_facing_only: bool = False,
        tool_names: Iterable[str] | None = None,
    ) -> tuple[RegisteredTool, ...]:
        allowed = frozenset(tool_names) if tool_names is not None else None
        return tuple(
            self.tools[name]
            for name in sorted(self.tools)
            if (allow_mutation or self.tools[name].permission is PermissionClass.READ_ONLY)
            and (not model_facing_only or self.tools[name].model_facing)
            and (allowed is None or name in allowed)
        )

    def definitions(
        self,
        *,
        allow_mutation: bool = False,
        model_facing_only: bool = False,
        tool_names: Iterable[str] | None = None,
    ) -> tuple[ToolDefinition, ...]:
        return tuple(
            tool.definition()
            for tool in self.list_tools(
                allow_mutation=allow_mutation,
                model_facing_only=model_facing_only,
                tool_names=tool_names,
            )
        )

    def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        allow_mutation: bool = False,
    ) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            return _failure(call.name, "unknown tool", None)
        if tool.permission is PermissionClass.PROJECT_MUTATION and not allow_mutation:
            return _failure(tool.name, "mutation permission denied", tool.permission)
        try:
            args = tool.args_model.model_validate(dict(call.arguments))
            payload = tool.handler(args, context)
        except ValidationError as exc:
            return _failure(
                tool.name, "invalid arguments", tool.permission, _safe_validation_errors(exc)
            )
        except (ToolExecutionError, ValueError) as exc:
            return _failure(tool.name, str(exc), tool.permission)
        except Exception:
            return _failure(tool.name, "tool execution failed", tool.permission)
        result = {
            "status": "ok",
            "tool": tool.name,
            "permission": tool.permission.value,
            "result": payload,
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result


def _failure(
    tool: str, message: str, permission: PermissionClass | None, evidence: Any | None = None
) -> dict[str, Any]:
    result = {
        "status": "error",
        "tool": tool,
        "error": message,
    }
    if permission is not None:
        result["permission"] = permission.value
    if evidence is not None:
        result["evidence"] = evidence
    return result


def _safe_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()]


@dataclass(slots=True)
class ProjectToolExecutor:
    registry: ToolRegistry
    context: ToolContext
    workflow_policy: ToolWorkflowPolicy | None = None

    def definitions(
        self, *, allow_mutation: bool, tool_names: Iterable[str] | None = None
    ) -> tuple[ToolDefinition, ...]:
        selected = tool_names
        if selected is None and self.workflow_policy is not None:
            selected = self.workflow_policy.allowed_tool_names
        return self.registry.definitions(
            allow_mutation=allow_mutation, model_facing_only=True, tool_names=selected
        )

    def execute(
        self, call: ToolCall, *, allow_mutation: bool, confirm: Any = None
    ) -> dict[str, Any]:
        tool = self.registry.tools.get(call.name)
        policy_error = self._policy_error(call, tool)
        if policy_error is not None:
            return policy_error
        if tool is None:
            return self.registry.execute(call, self.context, allow_mutation=allow_mutation)
        if not tool.model_facing:
            return {
                "status": "error",
                "tool": call.name,
                "error": "tool is not available to model-facing operations",
            }
        expected_before: str | None = None
        if tool.permission is PermissionClass.PROJECT_MUTATION:
            if not allow_mutation:
                return {
                    "status": "denied",
                    "tool": call.name,
                    "permission": tool.permission.value,
                    "error": "mutation tools require interactive confirmation",
                }
            preview = self._preview(tool, call)
            if preview.get("status") != "ok":
                return preview
            expected = (preview.get("result") or {}).get("before_fingerprint")
            if not isinstance(expected, str):
                return {
                    "status": "error",
                    "tool": call.name,
                    "permission": tool.permission.value,
                    "error": "mutation preview did not include before_fingerprint",
                }
            if confirm is None or not confirm(preview):
                return {
                    "status": "denied",
                    "tool": call.name,
                    "permission": tool.permission.value,
                    "preview": _declined_preview(preview),
                    "message": "change was previewed but not applied",
                }
            current = fingerprint_project(load_project(self.context.require_project_path()))
            if current != expected:
                return {
                    "status": "error",
                    "tool": call.name,
                    "permission": tool.permission.value,
                    "error": "active project changed after preview",
                }
            expected_before = expected
        context = (
            replace(self.context, expected_fingerprint=expected_before)
            if expected_before is not None
            else self.context
        )
        return self.registry.execute(call, context, allow_mutation=allow_mutation)

    def _policy_error(
        self, call: ToolCall, tool: RegisteredTool | None
    ) -> dict[str, Any] | None:
        policy = self.workflow_policy
        if policy is None:
            return None
        if call.name not in policy.allowed_tool_names:
            return self._workflow_rejection(call, tool, "tool is not allowed in this workflow")
        if tool is None:
            return self._workflow_rejection(call, tool, "unknown tool")
        if tool.permission not in policy.allowed_permissions:
            return self._workflow_rejection(
                call,
                tool,
                "tool permission is not allowed in this workflow",
            )
        if (
            tool.permission is PermissionClass.PROJECT_MUTATION
            and policy.expected_operation_family is not None
            and tool.operation_family != policy.expected_operation_family
        ):
            return self._workflow_rejection(
                call,
                tool,
                "mutation operation family is not allowed in this workflow",
            )
        return None

    def _workflow_rejection(
        self, call: ToolCall, tool: RegisteredTool | None, reason: str
    ) -> dict[str, Any]:
        policy = self.workflow_policy
        expected = sorted(policy.expected_terminal_tools) if policy is not None else []
        expected_text = ", ".join(expected) if expected else "the workflow policy"
        result: dict[str, Any] = {
            "status": "wrong_operation_rejected",
            "tool": call.name,
            "error": f"{reason}; expected {expected_text}",
            "terminal": True,
        }
        if policy is not None:
            result["workflow"] = policy.name
        if tool is not None:
            result["permission"] = tool.permission.value
            if tool.operation_family is not None:
                result["operation_family"] = tool.operation_family
        return result

    def _preview(self, tool: RegisteredTool, call: ToolCall) -> dict[str, Any]:
        if tool.preview_handler is None:
            preview_call = ToolCall(call.id, call.name.replace("apply", "preview"), call.arguments)
            return self.registry.execute(preview_call, self.context, allow_mutation=False)
        try:
            args = tool.args_model.model_validate(dict(call.arguments))
            payload = tool.preview_handler(args, self.context)
        except ValidationError as exc:
            return _failure(
                tool.name, "invalid arguments", tool.permission, _safe_validation_errors(exc)
            )
        except (ToolExecutionError, ValueError) as exc:
            return _failure(tool.name, str(exc), tool.permission)
        except Exception:
            return _failure(tool.name, "tool execution failed", tool.permission)
        result = {
            "status": "ok",
            "tool": tool.name,
            "permission": tool.permission.value,
            "result": payload,
        }
        json.dumps(result, allow_nan=False, sort_keys=True)
        return result


def _declined_preview(preview: dict[str, Any]) -> dict[str, Any]:
    result = preview.get("result")
    if not isinstance(result, dict):
        return {}
    data: dict[str, Any] = {"not_applied": True}
    for key in ("summary", "before_fingerprint"):
        if key in result:
            data[key] = result[key]
    validation = result.get("validation")
    if isinstance(validation, dict):
        data["validation"] = validation
    return data
