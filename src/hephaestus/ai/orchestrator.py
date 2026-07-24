"""Provider-independent Hephaestus orchestration loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from hephaestus.ai.identity import SYSTEM_INSTRUCTION
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
from hephaestus.changes import fingerprint_project
from hephaestus.persistence import load_project


class OrchestrationError(RuntimeError):
    """Raised when a turn cannot reach a visible terminal outcome."""


class ToolExecutor(Protocol):
    def definitions(
        self, *, allow_mutation: bool, tool_names: Iterable[str] | None = None
    ) -> Sequence[ToolDefinition]: ...

    def execute(
        self,
        call: ToolCall,
        *,
        allow_mutation: bool,
        confirm: Callable[[dict[str, Any]], bool] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ToolActivity:
    name: str
    status: str
    permission: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"name": self.name, "status": self.status}
        if self.permission is not None:
            data["permission"] = self.permission
        return data


@dataclass(frozen=True, slots=True)
class OrchestrationEvent:
    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event, "data": dict(self.data)}


@dataclass(frozen=True, slots=True)
class ProjectMarker:
    revision: int | None
    fingerprint: str | None


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    attempted: bool
    tool: str
    tool_call_id: str
    permission: str | None
    applied: bool
    status: str
    reason: str
    before_revision: int | None
    before_fingerprint: str | None
    after_revision: int | None
    after_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "tool": self.tool,
            "tool_call_id": self.tool_call_id,
            "permission": self.permission,
            "applied": self.applied,
            "status": self.status,
            "reason": self.reason,
            "before_revision": self.before_revision,
            "before_fingerprint": self.before_fingerprint,
            "after_revision": self.after_revision,
            "after_fingerprint": self.after_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    content: str
    provider: str
    model: str
    finish_reason: FinishReason
    usage: dict[str, int]
    tool_activity: tuple[ToolActivity, ...] = ()
    events: tuple[OrchestrationEvent, ...] = ()
    mutation_outcome: MutationOutcome | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage,
            "tool_activity": [item.to_dict() for item in self.tool_activity],
        }
        if self.mutation_outcome is not None:
            data["mutation_outcome"] = self.mutation_outcome.to_dict()
        return data


@dataclass(slots=True)
class HephaestusOrchestrator:
    provider: ModelProvider
    tool_executor: ToolExecutor | None = None
    project_path: Path | None = None
    max_tool_rounds: int = 6
    history: list[ChatMessage] = field(default_factory=list)
    events: list[OrchestrationEvent] = field(default_factory=list)

    def ask(
        self,
        prompt: str,
        *,
        allow_mutation: bool = False,
        confirm: Callable[[dict[str, Any]], bool] | None = None,
        expected_tool_names: Iterable[str] | None = None,
    ) -> OrchestrationResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        self.events.clear()
        self._record("turn_started", project_active=self.project_path is not None)
        working_history = [*self.history, ChatMessage(MessageRole.USER, prompt)]
        messages = [*working_history]
        project_context = self._project_context()
        if project_context:
            messages.insert(0, ChatMessage(MessageRole.SYSTEM, project_context))
        expected_tools = _expected_tool_set(expected_tool_names)
        tools = tuple(
            self.tool_executor.definitions(allow_mutation=allow_mutation, tool_names=expected_tools)
            if self.tool_executor is not None
            else ()
        )
        activity: list[ToolActivity] = []
        seen_calls: set[tuple[str, str]] = set()
        last_response: ModelResponse | None = None

        for _round in range(self.max_tool_rounds + 1):
            self._record("provider_request_started", tool_count=len(tools))
            try:
                response = self.provider.complete(
                    ModelRequest(
                        messages=tuple(messages),
                        system_instruction=SYSTEM_INSTRUCTION,
                        tools=tools,
                    )
                )
            except ProviderError:
                raise
            self._record(
                "provider_response_completed",
                has_text=bool(response.content.strip()),
                tool_call_count=len(response.tool_calls),
                finish_reason=response.finish_reason.value,
            )
            last_response = response
            if not response.tool_calls:
                if not response.content.strip():
                    self._record("invalid_empty_response")
                    raise OrchestrationError("Provider returned no assistant text or tool calls")
                working_history.append(ChatMessage(MessageRole.ASSISTANT, response.content))
                self.history = working_history
                self._record("final_response_ready")
                return OrchestrationResult(
                    content=response.content,
                    provider=self.provider.provider_id,
                    model=self.provider.model_id,
                    finish_reason=response.finish_reason,
                    usage=response.usage.to_dict(),
                    tool_activity=tuple(activity),
                    events=tuple(self.events),
                )
            if self.tool_executor is None:
                return self._stop_with_error(
                    "Tool call requested, but no tool executor is active", response, activity
                )
            exposed_tool_names = frozenset(tool.name for tool in tools)
            for call in response.tool_calls:
                if call.name not in exposed_tool_names:
                    allowed = expected_tools if expected_tools is not None else exposed_tool_names
                    self._record(
                        "unexpected_tool_requested",
                        tool=call.name,
                        tool_call_id=call.id,
                        expected_tools=sorted(allowed),
                    )
                    activity.append(ToolActivity(name=call.name, status="unexpected"))
                    return self._stop_with_error(
                        _unexpected_tool_message(call.name, allowed), response, activity
                    )
            messages.append(
                ChatMessage(
                    MessageRole.ASSISTANT,
                    response.content,
                    tool_calls=response.tool_calls,
                )
            )
            for call in response.tool_calls:
                call_key = (call.name, json.dumps(dict(call.arguments), sort_keys=True))
                if call_key in seen_calls:
                    return self._stop_with_error(
                        "Stopped repeated identical tool call", response, activity
                    )
                seen_calls.add(call_key)
                self._record("tool_requested", tool=call.name, tool_call_id=call.id)
                if expected_tools is not None and call.name not in expected_tools:
                    self._record(
                        "unexpected_tool_requested",
                        tool=call.name,
                        tool_call_id=call.id,
                        expected_tools=sorted(expected_tools),
                    )
                    activity.append(ToolActivity(name=call.name, status="unexpected"))
                    return self._stop_with_error(
                        _unexpected_tool_message(call.name, expected_tools), response, activity
                    )
                before_marker = self._project_marker()
                result = self.tool_executor.execute(
                    call, allow_mutation=allow_mutation, confirm=confirm
                )
                status = str(result.get("status", "ok"))
                permission = str(result.get("permission")) if result.get("permission") else None
                self._record(
                    "tool_result_returned",
                    tool=call.name,
                    tool_call_id=call.id,
                    status=status,
                    permission=permission,
                )
                activity.append(
                    ToolActivity(
                        name=call.name,
                        status=status,
                        permission=permission,
                    )
                )
                if result.get("terminal") is True:
                    message = str(result.get("error") or "Tool call rejected by workflow policy")
                    self._record(
                        "tool_rejected_terminal",
                        tool=call.name,
                        tool_call_id=call.id,
                        status=status,
                    )
                    return self._stop_with_error(message, response, activity)
                if permission == "project mutation":
                    outcome = self._mutation_outcome(call, result, before_marker, permission)
                    if outcome.status == "applied":
                        self._record("confirmation_accepted", tool_call_id=call.id)
                        self._record("mutation_applied", tool_call_id=call.id)
                    elif outcome.status == "declined":
                        self._record("confirmation_declined", tool_call_id=call.id)
                    else:
                        self._record("mutation_failed", tool_call_id=call.id)
                    return self._terminal_mutation_result(outcome, activity, working_history)
                messages.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        content=json.dumps(result, allow_nan=False, sort_keys=True),
                    ).as_message()
                )
        return self._stop_with_error(
            "Stopped after tool-call budget was exhausted", last_response, activity
        )

    def clear(self) -> None:
        self.history.clear()
        self.events.clear()

    def _project_context(self) -> str:
        if self.project_path is None:
            return ""
        project = load_project(self.project_path)
        return (
            "Active engineering project summary: "
            f"name={project.metadata.name}; schema_version={project.schema_version}; "
            f"revision={project.revision}; counts={project.collection_counts()}"
        )

    def _project_marker(self) -> ProjectMarker:
        if self.project_path is None:
            return ProjectMarker(None, None)
        project = load_project(self.project_path)
        return ProjectMarker(project.revision, fingerprint_project(project))

    def _mutation_outcome(
        self,
        call: ToolCall,
        result: dict[str, Any],
        before_marker: ProjectMarker,
        permission: str | None,
    ) -> MutationOutcome:
        status = str(result.get("status", "error"))
        if status == "ok":
            after_marker = self._project_marker()
            payload = result.get("result") if isinstance(result.get("result"), dict) else {}
            expected_revision = payload.get("revision") if isinstance(payload, dict) else None
            expected_fingerprint = (
                payload.get("after_fingerprint") if isinstance(payload, dict) else None
            )
            verified = (
                after_marker.revision == expected_revision
                and after_marker.fingerprint == expected_fingerprint
            )
            if verified:
                return MutationOutcome(
                    attempted=True,
                    tool=call.name,
                    tool_call_id=call.id,
                    permission=permission,
                    applied=True,
                    status="applied",
                    reason="mutation applied",
                    before_revision=before_marker.revision,
                    before_fingerprint=before_marker.fingerprint,
                    after_revision=after_marker.revision,
                    after_fingerprint=after_marker.fingerprint,
                )
            return MutationOutcome(
                attempted=True,
                tool=call.name,
                tool_call_id=call.id,
                permission=permission,
                applied=False,
                status="failed",
                reason="committed project state could not be verified",
                before_revision=before_marker.revision,
                before_fingerprint=before_marker.fingerprint,
                after_revision=after_marker.revision,
                after_fingerprint=after_marker.fingerprint,
            )
        if status == "denied":
            reason = str(
                result.get("message") or result.get("error") or "user declined confirmation"
            )
            after_marker = self._project_marker()
            return MutationOutcome(
                attempted=True,
                tool=call.name,
                tool_call_id=call.id,
                permission=permission,
                applied=False,
                status="declined",
                reason=reason,
                before_revision=before_marker.revision,
                before_fingerprint=before_marker.fingerprint,
                after_revision=after_marker.revision,
                after_fingerprint=after_marker.fingerprint,
            )
        after_marker = self._project_marker()
        return MutationOutcome(
            attempted=True,
            tool=call.name,
            tool_call_id=call.id,
            permission=permission,
            applied=False,
            status="failed",
            reason=str(result.get("error") or "mutation failed"),
            before_revision=before_marker.revision,
            before_fingerprint=before_marker.fingerprint,
            after_revision=after_marker.revision,
            after_fingerprint=after_marker.fingerprint,
        )

    def _terminal_mutation_result(
        self,
        outcome: MutationOutcome,
        activity: Iterable[ToolActivity],
        working_history: Sequence[ChatMessage],
    ) -> OrchestrationResult:
        content = _mutation_outcome_message(outcome)
        self.history = [*working_history, ChatMessage(MessageRole.ASSISTANT, content)]
        self._record("mutation_outcome_terminal", status=outcome.status, applied=outcome.applied)
        return OrchestrationResult(
            content=content,
            provider=self.provider.provider_id,
            model=self.provider.model_id,
            finish_reason=FinishReason.STOP if outcome.applied else FinishReason.ERROR,
            usage={},
            tool_activity=tuple(activity),
            events=tuple(self.events),
            mutation_outcome=outcome,
        )

    def _stop_with_error(
        self,
        message: str,
        response: ModelResponse | None,
        activity: Iterable[ToolActivity],
    ) -> OrchestrationResult:
        self._record("loop_terminated", reason=message)
        return OrchestrationResult(
            content=message,
            provider=self.provider.provider_id,
            model=self.provider.model_id,
            finish_reason=FinishReason.ERROR,
            usage={} if response is None else response.usage.to_dict(),
            tool_activity=tuple(activity),
            events=tuple(self.events),
        )

    def _record(self, event: str, **data: Any) -> None:
        self.events.append(OrchestrationEvent(event, data))


def _mutation_outcome_message(outcome: MutationOutcome) -> str:
    if outcome.status == "applied":
        return "Change applied."
    if outcome.status == "declined":
        return "Change not applied."
    return f"Change failed: {outcome.reason}"


def _expected_tool_set(tool_names: Iterable[str] | None) -> frozenset[str] | None:
    if tool_names is None:
        return None
    names = frozenset(name for name in tool_names if name)
    if not names:
        raise ValueError("expected_tool_names must not be empty when provided")
    return names


def _unexpected_tool_message(actual: str, expected: frozenset[str]) -> str:
    allowed = ", ".join(sorted(expected))
    return f"Unexpected tool requested: expected {allowed}; got {actual}."
