"""Groq provider adapter isolated behind Hephaestus AI types."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import groq

from hephaestus.ai.config import DEFAULT_MODEL
from hephaestus.ai.types import (
    ChatMessage,
    FinishReason,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderAuthenticationError,
    ProviderCapabilities,
    ProviderConfigurationError,
    ProviderErrorDetails,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderServiceUnavailableError,
    ProviderStructuredGenerationError,
    ProviderTimeoutError,
    ProviderToolUseError,
    StructuredPlanningRequest,
    StructuredPlanningResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


def provider_from_environment(
    *, model: str = DEFAULT_MODEL, timeout_seconds: float = 30.0, client: Any | None = None
) -> GroqProvider:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key and client is None:
        raise ProviderConfigurationError("GROQ_API_KEY is not configured")
    return GroqProvider(
        model=model, api_key=api_key, timeout_seconds=timeout_seconds, client=client
    )


class GroqProvider(ModelProvider):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if client is None and not clean_key:
            raise ProviderConfigurationError("GROQ_API_KEY is not configured")
        self._client = client if client is not None else groq.Groq(api_key=clean_key)

    @property
    def provider_id(self) -> str:
        return "groq"

    @property
    def model_id(self) -> str:
        return self._model

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(tool_calls=True, streaming=False)

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            internal_to_provider, provider_to_internal = _tool_name_maps(request.tools)
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": _messages_payload(
                    request.system_instruction, request.messages, internal_to_provider
                ),
                "reasoning_format": "hidden",
                "timeout": request.timeout_seconds or self._timeout_seconds,
            }
            if request.tools:
                payload["tools"] = _tools_payload(request.tools, internal_to_provider)
                payload["tool_choice"] = "auto"
                if request.parallel_tool_calls is not None:
                    payload["parallel_tool_calls"] = request.parallel_tool_calls
            if request.temperature is not None:
                payload["temperature"] = request.temperature
            response = self._client.chat.completions.create(**payload)
        except groq.AuthenticationError as exc:
            raise ProviderAuthenticationError("AI provider authentication failed") from exc
        except groq.RateLimitError as exc:
            details = _provider_error_details(exc, retryable=False)
            message = (
                "AI provider account limit reached"
                if _is_provider_account_limit(details)
                else "AI provider rate limit reached"
            )
            raise ProviderRateLimitError(message, details) from exc
        except groq.APITimeoutError as exc:
            raise ProviderTimeoutError("AI provider request timed out") from exc
        except (groq.APIConnectionError, groq.InternalServerError) as exc:
            raise ProviderServiceUnavailableError("AI provider service is unavailable") from exc
        except groq.BadRequestError as exc:
            details = _provider_error_details(
                exc, retryable=False, registered_tool_names=provider_to_internal
            )
            if details.provider_code == "tool_use_failed":
                raise ProviderToolUseError(
                    "AI provider could not produce a valid engineering operation. "
                    "No changes were made.",
                    details,
                ) from exc
            raise ProviderMalformedResponseError(
                "AI provider rejected the request", details
            ) from exc
        except groq.GroqError as exc:
            raise ProviderServiceUnavailableError(
                "AI provider request failed", _provider_error_details(exc)
            ) from exc
        return _response_from_groq(response, provider_to_internal)

    def complete_structured(self, request: StructuredPlanningRequest) -> StructuredPlanningResponse:
        try:
            payload: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": request.system_instruction},
                    {"role": "user", "content": request.user_prompt},
                ],
                "reasoning_format": "hidden",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.schema_name,
                        "schema": dict(request.json_schema),
                        "strict": request.strict_schema,
                    },
                },
                "timeout": request.timeout_seconds or self._timeout_seconds,
            }
            if request.max_output_tokens is not None:
                payload["max_completion_tokens"] = request.max_output_tokens
            response = self._client.chat.completions.create(**payload)
        except groq.AuthenticationError as exc:
            raise ProviderAuthenticationError("AI provider authentication failed") from exc
        except groq.RateLimitError as exc:
            details = _provider_error_details(exc, retryable=False)
            message = (
                "AI provider account limit reached"
                if _is_provider_account_limit(details)
                else "AI provider rate limit reached"
            )
            raise ProviderRateLimitError(message, details) from exc
        except groq.APITimeoutError as exc:
            raise ProviderTimeoutError("AI provider request timed out") from exc
        except (groq.APIConnectionError, groq.InternalServerError) as exc:
            raise ProviderServiceUnavailableError("AI provider service is unavailable") from exc
        except groq.BadRequestError as exc:
            details = _provider_error_details(exc, retryable=False)
            if details.status_code == 400 and details.provider_code == "json_validate_failed":
                raise ProviderStructuredGenerationError(
                    "AI provider could not produce schema-valid structured output",
                    details,
                ) from exc
            raise ProviderMalformedResponseError(
                "AI provider rejected the structured planning request",
                details,
            ) from exc
        except groq.GroqError as exc:
            raise ProviderServiceUnavailableError(
                "AI provider structured planning request failed", _provider_error_details(exc)
            ) from exc
        return _structured_response_from_groq(response)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


_PROVIDER_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _tool_name_maps(
    tools: Sequence[ToolDefinition],
) -> tuple[dict[str, str], dict[str, str]]:
    internal_to_provider: dict[str, str] = {}
    provider_to_internal: dict[str, str] = {}
    for tool in tools:
        provider_name = _provider_tool_name(tool.name)
        existing = provider_to_internal.get(provider_name)
        if existing is not None and existing != tool.name:
            raise ProviderConfigurationError(
                f"tool names collide after provider sanitization: {existing}, {tool.name}"
            )
        internal_to_provider[tool.name] = provider_name
        provider_to_internal[provider_name] = tool.name
    return internal_to_provider, provider_to_internal


def _provider_tool_name(name: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]
    if not candidate or not _PROVIDER_TOOL_NAME_PATTERN.fullmatch(candidate):
        raise ProviderConfigurationError(f"tool name cannot be mapped for provider: {name}")
    return candidate


def _messages_payload(
    system_instruction: str,
    messages: Sequence[ChatMessage],
    internal_to_provider: Mapping[str, str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]
    for message in messages:
        item = message.to_dict()
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_payload(call, internal_to_provider) for call in message.tool_calls
            ]
            if message.role.value == "assistant" and not message.content:
                item["content"] = None
        if message.role.value == "tool" and message.name is not None:
            item["name"] = internal_to_provider.get(message.name, message.name)
        payload.append({key: value for key, value in item.items() if value is not None})
    return payload


def _tool_call_payload(call: ToolCall, internal_to_provider: Mapping[str, str]) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": internal_to_provider.get(call.name, call.name),
            "arguments": json.dumps(dict(call.arguments), allow_nan=False, sort_keys=True),
        },
    }


def _tools_payload(
    tools: Sequence[ToolDefinition], internal_to_provider: Mapping[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": internal_to_provider[tool.name],
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            },
        }
        for tool in tools
    ]


def _structured_response_from_groq(response: Any) -> StructuredPlanningResponse:
    try:
        choices = _get(response, "choices")
        if not choices:
            raise ProviderMalformedResponseError("Provider response did not include choices")
        choice = choices[0]
        message = _get(choice, "message")
        raw_content = _get(message, "content", None)
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ProviderMalformedResponseError("Provider structured response was empty")
        payload = _loads_json_object_no_duplicates(raw_content)
        usage = _usage(_get(response, "usage", None))
    except ProviderMalformedResponseError:
        raise
    except json.JSONDecodeError as exc:
        raise ProviderMalformedResponseError(
            "Provider structured response was malformed JSON"
        ) from exc
    except ValueError as exc:
        raise ProviderMalformedResponseError(str(exc)) from exc
    except Exception as exc:
        raise ProviderMalformedResponseError(
            "Provider structured response could not be translated"
        ) from exc
    return StructuredPlanningResponse(payload=payload, usage=usage, raw_provider="groq")


def _response_from_groq(
    response: Any, provider_to_internal: Mapping[str, str] | None = None
) -> ModelResponse:
    try:
        choices = _get(response, "choices")
        if not choices:
            raise ProviderMalformedResponseError("Provider response did not include choices")
        choice = choices[0]
        message = _get(choice, "message")
        raw_content = _get(message, "content", None)
        content = "" if raw_content is None else str(raw_content)
        finish = _finish_reason(_get(choice, "finish_reason", None))
        usage = _usage(_get(response, "usage", None))
        tool_calls = tuple(
            _tool_call(call, provider_to_internal or {})
            for call in (_get(message, "tool_calls", None) or ())
        )
        if finish is FinishReason.TOOL_CALLS and not tool_calls:
            raise ProviderMalformedResponseError(
                "Provider response finished with tool calls but did not include tool calls"
            )
        if tool_calls and finish is not FinishReason.TOOL_CALLS:
            raise ProviderMalformedResponseError(
                "Provider response included tool calls with a non-tool finish reason"
            )
        if len({call.id for call in tool_calls}) != len(tool_calls):
            raise ProviderMalformedResponseError(
                "Provider response included duplicate tool call ids"
            )
        if not content.strip() and not tool_calls:
            raise ProviderMalformedResponseError(
                "Provider response contained neither assistant text nor tool calls"
            )
    except ProviderMalformedResponseError:
        raise
    except Exception as exc:
        raise ProviderMalformedResponseError("Provider response could not be translated") from exc
    return ModelResponse(
        content=content,
        finish_reason=finish,
        usage=usage,
        tool_calls=tool_calls,
        raw_provider="groq",
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _finish_reason(value: Any) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALLS,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(str(value), FinishReason.UNKNOWN)


def _usage(value: Any) -> TokenUsage:
    if value is None:
        return TokenUsage()
    input_tokens = int(_get(value, "prompt_tokens", _get(value, "input_tokens", 0)) or 0)
    output_tokens = int(_get(value, "completion_tokens", _get(value, "output_tokens", 0)) or 0)
    total = int(_get(value, "total_tokens", input_tokens + output_tokens) or 0)
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total)


def _tool_call(call: Any, provider_to_internal: Mapping[str, str] | None = None) -> ToolCall:
    raw_type = _get(call, "type", "function")
    if raw_type is not None and str(raw_type) != "function":
        raise ProviderMalformedResponseError("Provider tool call type was not function")
    function = _get(call, "function", {})
    raw_args = _get(function, "arguments", {}) or {}
    if isinstance(raw_args, str):
        try:
            arguments = _loads_json_object_no_duplicates(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise ProviderMalformedResponseError(
                "Provider tool call arguments were malformed"
            ) from exc
        except ValueError as exc:
            raise ProviderMalformedResponseError(str(exc)) from exc
    elif isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
    else:
        raise ProviderMalformedResponseError("Provider tool call arguments were not an object")
    if not isinstance(arguments, dict):
        raise ProviderMalformedResponseError(
            "Provider tool call arguments must decode to an object"
        )
    tool_call_id = str(_get(call, "id", ""))
    provider_name = str(_get(function, "name", ""))
    if provider_to_internal and provider_name not in provider_to_internal:
        raise ProviderMalformedResponseError("Provider tool call referenced an unknown tool name")
    name = (provider_to_internal or {}).get(provider_name, provider_name)
    if not tool_call_id or not name:
        raise ProviderMalformedResponseError(
            "Provider tool call was missing an id or function name"
        )
    return ToolCall(id=tool_call_id, name=name, arguments=arguments)


def _loads_json_object_no_duplicates(value: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Provider tool call arguments contained duplicate keys")
            result[key] = item
        return result

    loaded = json.loads(value, object_pairs_hook=pairs_hook)
    if not isinstance(loaded, dict):
        raise ProviderMalformedResponseError(
            "Provider tool call arguments must decode to an object"
        )
    return loaded


def _provider_error_details(
    exc: BaseException,
    *,
    retryable: bool | None = None,
    registered_tool_names: Mapping[str, str] | None = None,
) -> ProviderErrorDetails:
    status_code = _maybe_int(_get(exc, "status_code", None))
    body = _get(exc, "body", None)
    error = body.get("error", body) if isinstance(body, Mapping) else {}
    provider_code = _safe_metadata(error.get("code")) if isinstance(error, Mapping) else None
    provider_error_type = _safe_metadata(error.get("type")) if isinstance(error, Mapping) else None
    provider_reason = _safe_provider_reason(error) if isinstance(error, Mapping) else None
    request_id = _safe_metadata(_get(exc, "request_id", None)) or _response_request_id(exc)
    retry_after = _response_float_header(exc, ("retry-after", "Retry-After"))
    token_reset = _response_float_header(
        exc,
        (
            "x-ratelimit-reset-tokens",
            "X-RateLimit-Reset-Tokens",
            "x-ratelimit-reset-requests",
            "X-RateLimit-Reset-Requests",
        ),
    )
    attempted_tool_name = (
        _safe_attempted_tool_name(error, registered_tool_names)
        if isinstance(error, Mapping)
        else None
    )
    is_retryable = retryable if retryable is not None else bool(status_code and status_code >= 500)
    return ProviderErrorDetails(
        status_code=status_code,
        provider_code=provider_code,
        provider_error_type=provider_error_type,
        provider_reason=provider_reason,
        attempted_tool_name=attempted_tool_name,
        request_id=request_id,
        retry_after_seconds=retry_after,
        token_reset_seconds=token_reset,
        retryable=is_retryable,
    )


def _is_provider_account_limit(details: ProviderErrorDetails) -> bool:
    return details.status_code == 413 and details.provider_code == "rate_limit_exceeded"


def _response_request_id(exc: BaseException) -> str | None:
    response = _get(exc, "response", None)
    headers = _get(response, "headers", None)
    if isinstance(headers, Mapping):
        return _safe_metadata(headers.get("x-request-id") or headers.get("X-Request-Id"))
    return None


def _response_float_header(exc: BaseException, names: Sequence[str]) -> float | None:
    response = _get(exc, "response", None)
    headers = _get(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    for name in names:
        value = headers.get(name)
        if value is None:
            continue
        try:
            number = float(str(value).strip().removesuffix("s"))
        except ValueError:
            continue
        if 0.0 <= number <= 86_400.0:
            return number
    return None


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


_SAFE_METADATA_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")
_SAFE_REASON_PATTERN = re.compile(r"^[A-Za-z0-9 .,_:;()/-]{1,220}$")
_SECRET_METADATA_PATTERN = re.compile(
    r"(api[_-]?key|argument|bearer|content|credential|failed_generation|gsk_|password|prompt|secret|sk-|token)",
    re.IGNORECASE,
)
_MAX_METADATA_LENGTH = 128


def _safe_metadata(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or _SECRET_METADATA_PATTERN.search(text):
        return None
    if not _SAFE_METADATA_PATTERN.fullmatch(text):
        return None
    return text[:_MAX_METADATA_LENGTH]


def _safe_provider_reason(error: Mapping[str, Any]) -> str | None:
    for key in ("reason", "message"):
        value = error.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text or _SECRET_METADATA_PATTERN.search(text):
            continue
        if any(marker in text for marker in ("{", "}", "[", "]", "<", ">", "\n", "\r")):
            continue
        if _SAFE_REASON_PATTERN.fullmatch(text):
            return text
    return None


def _safe_attempted_tool_name(
    error: Mapping[str, Any], registered_tool_names: Mapping[str, str] | None
) -> str | None:
    if not registered_tool_names:
        return None
    haystack = " ".join(
        str(error.get(key, ""))
        for key in ("message", "reason", "failed_generation")
        if error.get(key) is not None
    )
    matches = {
        internal_name
        for provider_name, internal_name in registered_tool_names.items()
        if provider_name
        and re.search(rf"(?<![A-Za-z0-9_-]){re.escape(provider_name)}(?![A-Za-z0-9_-])", haystack)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None
