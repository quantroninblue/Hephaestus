"""Default conversational Hephaestus agent loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hephaestus.ai.config import DEFAULT_AGENT_MAX_TOOL_ROUNDS, DEFAULT_AGENT_TOOL_TEMPERATURE
from hephaestus.ai.identity import (
    DEFAULT_AGENT_IDENTITY_VERSION,
    DEFAULT_AGENT_SYSTEM_INSTRUCTION,
)
from hephaestus.ai.types import (
    ChatMessage,
    FinishReason,
    MessageRole,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)
from hephaestus.capabilities import CapabilityReport, CapabilityStatus, collect_capabilities
from hephaestus.tools.cadquery_build import CadQueryBuildContext
from hephaestus.tools.gazebo import GazeboWorkbench
from hephaestus.tools.workspace import (
    CADQUERY_PERMISSION_BUILD,
    WORKSPACE_PERMISSION_READ,
    WORKSPACE_PERMISSION_WRITE,
    WorkspaceToolExecutor,
)


class AgentLoopError(RuntimeError):
    """Raised when the default agent cannot produce a visible outcome."""


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    content: str
    provider: str
    model: str
    finish_reason: FinishReason
    usage: dict[str, int]
    tool_activity: tuple[dict[str, str], ...] = ()


@dataclass(slots=True)
class HephaestusAgentSession:
    provider: ModelProvider
    startup_directory: Path
    tool_executor: WorkspaceToolExecutor | None = None
    max_tool_rounds: int = DEFAULT_AGENT_MAX_TOOL_ROUNDS
    tool_calling_temperature: float = DEFAULT_AGENT_TOOL_TEMPERATURE
    history: list[ChatMessage] = field(default_factory=list)
    capabilities: CapabilityReport | None = None

    def __post_init__(self) -> None:
        self.startup_directory = self.startup_directory.resolve(strict=True)
        if self.tool_executor is None:
            self.tool_executor = WorkspaceToolExecutor(self.startup_directory)
        if self.capabilities is None:
            self.capabilities = collect_capabilities()
        self._configure_cadquery_builder()
        self._configure_gazebo_workbench()

    def ask(
        self,
        prompt: str,
        *,
        confirm: Callable[[dict[str, Any]], bool] | None = None,
    ) -> AgentTurnResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        working_messages = [*self.history, ChatMessage(MessageRole.USER, prompt)]
        tools = self._tool_definitions()
        activity: list[dict[str, str]] = []
        seen_calls: set[tuple[int, str, str]] = set()
        workspace_epoch = 0
        last_response: ModelResponse | None = None
        last_committed_result: dict[str, Any] | None = None
        tool_rounds = 0

        while True:
            try:
                response = self.provider.complete(
                    ModelRequest(
                        messages=(
                            ChatMessage(
                                MessageRole.SYSTEM,
                                self.dynamic_context(tool_definitions=tools),
                            ),
                            *working_messages,
                        ),
                        system_instruction=DEFAULT_AGENT_SYSTEM_INSTRUCTION,
                        tools=tools,
                        temperature=self.tool_calling_temperature,
                        parallel_tool_calls=False,
                    )
                )
            except ProviderError:
                if last_committed_result is not None:
                    content = _committed_operation_fallback(last_committed_result)
                    self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                    fallback_response = last_response or ModelResponse("", FinishReason.ERROR)
                    return self._result(fallback_response, content, activity, FinishReason.ERROR)
                raise
            last_response = response
            if not response.tool_calls:
                if not response.content.strip():
                    raise AgentLoopError("Provider returned no assistant text or tool calls")
                working_messages.append(ChatMessage(MessageRole.ASSISTANT, response.content))
                self.history = working_messages
                return self._result(response, response.content, activity)

            if tool_rounds >= self.max_tool_rounds:
                content = "Stopped after tool-call budget was exhausted."
                self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                return self._result(last_response, content, activity, FinishReason.ERROR)

            if self.tool_executor is None:
                content = "Tool call requested, but no workspace tool executor is active."
                self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                return self._result(response, content, activity, FinishReason.ERROR)

            exposed_names = frozenset(tool.name for tool in tools)
            unknown = next(
                (call for call in response.tool_calls if call.name not in exposed_names),
                None,
            )
            if unknown is not None:
                content = f"Unexpected tool requested: {unknown.name}."
                self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                activity.append({"name": unknown.name, "status": "unexpected"})
                return self._result(response, content, activity, FinishReason.ERROR)

            validation_error = self.tool_executor.validate_batch(
                response.tool_calls, allow_mutation=True
            )
            if validation_error is not None:
                if self.tool_executor.is_read_only_batch(response.tool_calls):
                    working_messages.append(
                        ChatMessage(
                            MessageRole.ASSISTANT,
                            response.content,
                            tool_calls=response.tool_calls,
                        )
                    )
                    for result in _read_only_batch_error_results(
                        response.tool_calls, validation_error
                    ):
                        activity.append(
                            {
                                "name": str(result.get("tool", "unknown")),
                                "status": str(result.get("status", "error")),
                            }
                        )
                        working_messages.append(
                            ToolResultMessage(
                                tool_call_id=str(result.get("tool_call_id", "")),
                                tool_name=str(result.get("tool", "unknown")),
                                content=json.dumps(result, allow_nan=False, sort_keys=True),
                            ).as_message()
                        )
                    tool_rounds += 1
                    continue
                content = _tool_error_message(validation_error)
                self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                activity.append(
                    {
                        "name": str(validation_error.get("tool", "unknown")),
                        "status": str(validation_error.get("status", "error")),
                    }
                )
                return self._result(response, content, activity, FinishReason.ERROR)

            working_messages.append(
                ChatMessage(
                    MessageRole.ASSISTANT,
                    response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                call_key = (
                    workspace_epoch,
                    call.name,
                    json.dumps(dict(call.arguments), sort_keys=True),
                )
                if call_key in seen_calls:
                    content = "Stopped repeated identical tool call."
                    self.history = [
                        *working_messages,
                        ChatMessage(MessageRole.ASSISTANT, content),
                    ]
                    return self._result(response, content, activity, FinishReason.ERROR)
                seen_calls.add(call_key)
                result = self.tool_executor.execute(call, allow_mutation=True, confirm=confirm)
                status = str(result.get("status", "ok"))
                activity.append({"name": call.name, "status": status})
                if _is_declined_workspace_write(result):
                    content = _declined_tool_message(result)
                    working_messages.append(
                        ToolResultMessage(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            content=json.dumps(result, allow_nan=False, sort_keys=True),
                        ).as_message()
                    )
                    self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                    return self._result(response, content, activity, FinishReason.ERROR)
                if status != "ok":
                    content = _tool_error_message(result)
                    working_messages.append(
                        ToolResultMessage(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            content=json.dumps(result, allow_nan=False, sort_keys=True),
                        ).as_message()
                    )
                    self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                    return self._result(response, content, activity, FinishReason.ERROR)
                if _is_committed_workspace_write(result):
                    workspace_epoch += 1
                    last_committed_result = result
                working_messages.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=json.dumps(result, allow_nan=False, sort_keys=True),
                    ).as_message()
                )
                if _is_terminal_cadquery_build_success(result):
                    content = _committed_operation_fallback(result)
                    self.history = [*working_messages, ChatMessage(MessageRole.ASSISTANT, content)]
                    return self._result(response, content, activity, FinishReason.STOP)
            tool_rounds += 1

    def clear(self) -> None:
        self.history.clear()

    def close(self) -> None:
        if self.tool_executor is not None:
            self.tool_executor.close()

    def dynamic_context(self, *, tool_definitions: Sequence[ToolDefinition] | None = None) -> str:
        tools = (
            tuple(tool_definitions) if tool_definitions is not None else self._tool_definitions()
        )
        capabilities = (
            self.capabilities if self.capabilities is not None else collect_capabilities()
        )
        capability_lines = ", ".join(
            f"{item.capability_id}={item.status.value}/{item.compatibility.value}"
            for item in capabilities.capabilities
        )
        tool_names = ", ".join(tool.name for tool in tools)
        state = f"messages={len(self.history)}; max_tool_rounds={self.max_tool_rounds}"
        gazebo = capabilities.by_id().get("gazebo")
        cad = capabilities.by_id().get("cad")
        gazebo_context = (
            self.tool_executor.gazebo_context_summary()
            if self.tool_executor is not None
            else "gazebo=unavailable"
        )
        gazebo_tool_context = "gazebo_tools=unavailable"
        if gazebo is not None and gazebo.status is not CapabilityStatus.ACTIVE:
            evidence = "; ".join(gazebo.evidence[:4])
            gazebo_tool_context = (
                f"gazebo_tool_blocker: status={gazebo.status.value}; "
                f"compatibility={gazebo.compatibility.value}; evidence={evidence}"
            )
        elif self.tool_executor is not None and self.tool_executor.gazebo_workbench is not None:
            gazebo_tool_context = (
                "gazebo_tool_guidance: validate SDF before launch; generated worlds should "
                "normally include gravity, ground collision, positive mass, valid inertia, "
                "collision geometry, friction, and joint limits/damping where applicable."
            )
        cadquery_tool_context = "cadquery_tool=unavailable"
        if cad is not None and cad.status is not CapabilityStatus.ACTIVE:
            evidence = "; ".join(cad.evidence[:4])
            cadquery_tool_context = (
                f"cadquery_tool_blocker: status={cad.status.value}; "
                f"compatibility={cad.compatibility.value}; evidence={evidence}"
            )
        elif self.tool_executor is not None and self.tool_executor.cadquery_context is not None:
            cadquery_tool_context = (
                "cadquery_tool_guidance: for recognizable robot designs, write editable "
                "CadQuery source with multiple named parts in millimetres, call cadquery_build, "
                "observe the real manifest, then write SDF visuals using the returned STL paths "
                "with mesh scale 0.001 0.001 0.001. Use simpler collision geometry, plausible "
                "inertia, distinct SDF materials/colors, believable robot scale, an inline "
                "ground plane, suitable light, gravity/collisions enabled, and a clear camera "
                "pose. State important simplifications; do not claim fidelity not produced."
            )
        return "\n".join(
            (
                "Dynamic context:",
                f"startup_working_directory: {self.startup_directory}",
                f"detected_capabilities: {capability_lines}",
                f"available_tools: {tool_names}",
                f"current_session_state: {state}",
                f"gazebo_session_state: {gazebo_context}",
                cadquery_tool_context,
                gazebo_tool_context,
                "tool_sequence_guidance: perform dependent operations one tool round at a "
                "time: write CAD source -> cadquery_build -> observe manifest -> write SDF "
                "with real mesh paths -> validate SDF -> launch Gazebo.",
                f"identity_version: {DEFAULT_AGENT_IDENTITY_VERSION}",
            )
        )

    def _tool_definitions(self) -> tuple[ToolDefinition, ...]:
        if self.tool_executor is None:
            return ()
        return self.tool_executor.definitions(allow_mutation=True)

    def _configure_gazebo_workbench(self) -> None:
        if self.tool_executor is None or self.capabilities is None:
            return
        gazebo = self.capabilities.by_id().get("gazebo")
        if gazebo is None or gazebo.status is not CapabilityStatus.ACTIVE:
            self.tool_executor.gazebo_workbench = None
            return
        if self.tool_executor.gazebo_workbench is None:
            self.tool_executor.gazebo_workbench = GazeboWorkbench(
                self.startup_directory,
                gazebo_version=gazebo.detected_version,
            )

    def _configure_cadquery_builder(self) -> None:
        if self.tool_executor is None or self.capabilities is None:
            return
        cad = self.capabilities.by_id().get("cad")
        if cad is None or cad.status is not CapabilityStatus.ACTIVE:
            self.tool_executor.cadquery_context = None
            return
        if self.tool_executor.cadquery_context is None:
            self.tool_executor.cadquery_context = CadQueryBuildContext(self.startup_directory)

    def _result(
        self,
        response: ModelResponse,
        content: str,
        activity: list[dict[str, str]],
        finish_reason: FinishReason | None = None,
    ) -> AgentTurnResult:
        return AgentTurnResult(
            content=content,
            provider=self.provider.provider_id,
            model=self.provider.model_id,
            finish_reason=finish_reason or response.finish_reason,
            usage=response.usage.to_dict(),
            tool_activity=tuple(activity),
        )


def _tool_error_message(result: dict[str, Any]) -> str:
    message = str(result.get("error") or result.get("message") or "unknown error")
    tool_name = str(result.get("tool") or "unknown")
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        details: list[str] = []
        for item in evidence[:3]:
            if not isinstance(item, dict):
                continue
            loc = item.get("loc")
            field = ".".join(str(part) for part in loc) if isinstance(loc, list) else ""
            detail = str(item.get("msg", "")).strip()
            if field and detail:
                details.append(f"{field}: {detail}")
            elif detail:
                details.append(detail)
        if details:
            message = f"{message}: {'; '.join(details)}"
    return f"Tool failed ({tool_name}): {message}"


def _read_only_batch_error_results(
    calls: Sequence[ToolCall], validation_error: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    failed_call_id = validation_error.get("tool_call_id")
    results: list[dict[str, Any]] = []
    for call in calls:
        if call.id == failed_call_id:
            result = dict(validation_error)
            result["tool_call_id"] = call.id
            results.append(result)
            continue
        results.append(
            {
                "status": "error",
                "tool": call.name,
                "permission": str(validation_error.get("permission", WORKSPACE_PERMISSION_READ)),
                "tool_call_id": call.id,
                "error": "tool batch rejected before execution",
            }
        )
    return tuple(results)


def _is_declined_workspace_write(result: dict[str, Any]) -> bool:
    if result.get("tool") not in {
        "workspace_write_text",
        "workspace_replace_text",
        "cadquery_build",
    }:
        return False
    payload = result.get("result")
    if result.get("tool") == "cadquery_build":
        return (
            result.get("status") == "ok"
            and result.get("permission") == CADQUERY_PERMISSION_BUILD
            and isinstance(payload, dict)
            and payload.get("committed") is False
        )
    return (
        result.get("status") == "ok"
        and result.get("permission") == WORKSPACE_PERMISSION_WRITE
        and isinstance(payload, dict)
        and payload.get("committed") is False
    )


def _is_committed_workspace_write(result: dict[str, Any]) -> bool:
    if result.get("tool") not in {
        "workspace_write_text",
        "workspace_replace_text",
        "cadquery_build",
    }:
        return False
    payload = result.get("result")
    if result.get("tool") == "cadquery_build":
        return (
            result.get("status") == "ok"
            and result.get("permission") == CADQUERY_PERMISSION_BUILD
            and isinstance(payload, dict)
            and payload.get("committed") is True
        )
    return (
        result.get("status") == "ok"
        and result.get("permission") == WORKSPACE_PERMISSION_WRITE
        and isinstance(payload, dict)
        and payload.get("committed") is True
    )


def _is_terminal_cadquery_build_success(result: dict[str, Any]) -> bool:
    return result.get("tool") == "cadquery_build" and _is_committed_workspace_write(result)


def _committed_operation_fallback(result: dict[str, Any]) -> str:
    payload = result.get("result")
    summary = (
        json.dumps(payload, allow_nan=False, sort_keys=True)
        if isinstance(payload, dict)
        else "{}"
    )
    return "Operation committed; AI explanation unavailable.\n" + summary


def _declined_tool_message(result: dict[str, Any]) -> str:
    if result.get("tool") == "cadquery_build":
        return "CadQuery build not approved; no artifacts were created."
    return "Workspace write not applied."
