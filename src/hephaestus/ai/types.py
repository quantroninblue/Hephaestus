"""Provider-independent AI contract types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return data


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    tool_calls: bool = True
    streaming: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {"tool_calls": self.tool_calls, "streaming": self.streaming}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str

    def as_message(self) -> ChatMessage:
        return ChatMessage(
            role=MessageRole.TOOL,
            content=self.content,
            tool_call_id=self.tool_call_id,
            name=self.tool_name,
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: Sequence[ChatMessage]
    system_instruction: str
    tools: Sequence[ToolDefinition] = ()
    tool_results: Sequence[ToolResultMessage] = ()
    timeout_seconds: float = 30.0
    temperature: float | None = None
    parallel_tool_calls: bool | None = None


@dataclass(frozen=True, slots=True)
class StructuredPlanningRequest:
    system_instruction: str
    user_prompt: str
    schema_name: str
    json_schema: Mapping[str, Any]
    timeout_seconds: float = 30.0
    strict_schema: bool = True
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StructuredPlanningResponse:
    payload: Mapping[str, Any]
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": dict(self.payload),
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str
    finish_reason: FinishReason
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: tuple[ToolCall, ...] = ()
    raw_provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "finish_reason": self.finish_reason.value,
            "usage": self.usage.to_dict(),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass(frozen=True, slots=True)
class ProviderErrorDetails:
    status_code: int | None = None
    provider_code: str | None = None
    provider_error_type: str | None = None
    provider_reason: str | None = None
    attempted_tool_name: str | None = None
    request_id: str | None = None
    retry_after_seconds: float | None = None
    token_reset_seconds: float | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"retryable": self.retryable}
        if self.status_code is not None:
            data["status_code"] = self.status_code
        if self.provider_code is not None:
            data["provider_code"] = self.provider_code
        if self.provider_error_type is not None:
            data["provider_error_type"] = self.provider_error_type
        if self.provider_reason is not None:
            data["provider_reason"] = self.provider_reason
        if self.attempted_tool_name is not None:
            data["attempted_tool_name"] = self.attempted_tool_name
        if self.request_id is not None:
            data["request_id"] = self.request_id
        if self.retry_after_seconds is not None:
            data["retry_after_seconds"] = self.retry_after_seconds
        if self.token_reset_seconds is not None:
            data["token_reset_seconds"] = self.token_reset_seconds
        return data


class ProviderError(RuntimeError):
    """Base class for application-owned provider failures."""

    def __init__(self, message: str, details: ProviderErrorDetails | None = None) -> None:
        super().__init__(message)
        self.details = details


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is missing or unsupported."""


class ProviderAuthenticationError(ProviderError):
    """Raised for authentication failures."""


class ProviderRateLimitError(ProviderError):
    """Raised for provider rate limits."""


class ProviderTimeoutError(ProviderError):
    """Raised for provider timeouts."""


class ProviderServiceUnavailableError(ProviderError):
    """Raised when the provider service is unavailable."""


class ProviderMalformedResponseError(ProviderError):
    """Raised when provider data cannot be translated safely."""


class ProviderStructuredGenerationError(ProviderMalformedResponseError):
    """Raised when a structured-output provider cannot satisfy the requested schema."""


class ProviderToolUseError(ProviderError):
    """Raised when the provider rejects generated tool-use output."""


class ModelProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def capabilities(self) -> ProviderCapabilities: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...

    def complete_structured(
        self, request: StructuredPlanningRequest
    ) -> StructuredPlanningResponse: ...

    def close(self) -> None: ...
