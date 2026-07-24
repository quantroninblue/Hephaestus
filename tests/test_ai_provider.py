from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from hephaestus.ai.config import (
    DEFAULT_AGENT_MAX_TOOL_ROUNDS,
    DEFAULT_MODEL,
    ProviderConfigurationError,
    resolve_config,
    save_user_config,
)
from hephaestus.ai.groq_provider import GroqProvider, provider_from_environment
from hephaestus.ai.types import (
    ChatMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ProviderMalformedResponseError,
    ProviderRateLimitError,
    ProviderStructuredGenerationError,
    ProviderToolUseError,
    StructuredPlanningRequest,
    ToolDefinition,
)
from hephaestus.tools.project import default_project_tool_registry
from hephaestus.tools.registry import ProjectToolExecutor, ToolContext


@dataclass
class Obj:
    pass


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, list) else [responses]
        self.kwargs = None
        self.calls = []

    def create(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, response):
        self.chat = Obj()
        self.chat.completions = FakeCompletions(response)


def _tool_call(
    *,
    call_id: str = "call_1",
    name: str = "project_get_summary",
    arguments: str | dict = "{}",
    call_type: str | None = None,
):
    call = Obj()
    call.id = call_id
    if call_type is not None:
        call.type = call_type
    call.function = Obj()
    call.function.name = name
    call.function.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return call


def _response(
    name: str = "project_get_summary",
    content=None,
    *,
    finish_reason: str = "tool_calls",
    calls=None,
):
    message = Obj()
    message.content = content
    message.tool_calls = list(calls) if calls is not None else [_tool_call(name=name)]
    choice = Obj()
    choice.message = message
    choice.finish_reason = finish_reason
    usage = Obj()
    usage.prompt_tokens = 3
    usage.completion_tokens = 4
    usage.total_tokens = 7
    response = Obj()
    response.choices = [choice]
    response.usage = usage
    return response


def test_groq_request_and_response_translation_keeps_owned_types() -> None:
    client = FakeClient(_response())
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)
    response = provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(ChatMessage(MessageRole.USER, "unchanged user"),),
            tools=(ToolDefinition("project.get_summary", "summary", {"type": "object"}),),
        )
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["messages"][0] == {"role": "system", "content": "system text"}
    assert kwargs["messages"][1]["content"] == "unchanged user"
    assert kwargs["reasoning_format"] == "hidden"
    assert kwargs["tools"][0]["function"]["name"] == "project_get_summary"
    assert "." not in kwargs["tools"][0]["function"]["name"]
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.usage.total_tokens == 7
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "project.get_summary"
    assert response.tool_calls[0].arguments == {}


def test_groq_chat_request_sends_optional_tool_controls_without_token_limit() -> None:
    final_message = Obj()
    final_message.content = "Ready."
    final_message.tool_calls = []
    final_choice = Obj()
    final_choice.message = final_message
    final_choice.finish_reason = "stop"
    final_response = Obj()
    final_response.choices = [final_choice]
    final_response.usage = None
    client = FakeClient(final_response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(ChatMessage(MessageRole.USER, "inspect"),),
            tools=(ToolDefinition("workspace_replace_text", "replace", {"type": "object"}),),
            temperature=0.2,
            parallel_tool_calls=False,
        )
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["temperature"] == 0.2
    assert kwargs["parallel_tool_calls"] is False
    assert "max_completion_tokens" not in kwargs


def test_missing_key_and_config_resolution_do_not_expose_secret(monkeypatch) -> None:
    secret = "gsk_test_secret_value"
    config = resolve_config(env={"GROQ_API_KEY": secret})
    encoded = json.dumps(config.safe_dict())
    assert config.credential_present is True
    assert secret not in encoded
    assert resolve_config(env={"GROQ_API_KEY": "   "}).credential_present is False
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ProviderConfigurationError, match="GROQ_API_KEY"):
        provider_from_environment(client=None)
    with pytest.raises(ProviderConfigurationError, match="GROQ_API_KEY"):
        GroqProvider(api_key="   ")


def test_config_rejects_invalid_max_completion_tokens(tmp_path) -> None:
    path = tmp_path / "ai.json"

    with pytest.raises(ProviderConfigurationError, match="max_completion_tokens"):
        save_user_config(
            provider="groq",
            model=DEFAULT_MODEL,
            timeout_seconds=30.0,
            max_completion_tokens=0,
            path=path,
        )

    path.write_text(
        json.dumps(
            {
                "provider": "groq",
                "model": DEFAULT_MODEL,
                "timeout_seconds": 30.0,
                "max_completion_tokens": -1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigurationError, match="max_completion_tokens"):
        resolve_config(path=path)

    path.write_text(
        json.dumps(
            {
                "provider": "groq",
                "model": DEFAULT_MODEL,
                "timeout_seconds": 30.0,
                "default_agent_tool_temperature": 3.0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigurationError, match="default_agent_tool_temperature"):
        resolve_config(path=path)


def test_config_resolves_default_agent_tool_temperature_override(tmp_path) -> None:
    path = tmp_path / "ai.json"
    save_user_config(
        provider="groq",
        model=DEFAULT_MODEL,
        timeout_seconds=30.0,
        max_completion_tokens=None,
        default_agent_tool_temperature=0.05,
        path=path,
    )

    config = resolve_config(path=path)

    assert config.default_agent_tool_temperature == 0.05
    assert config.safe_dict()["default_agent_tool_temperature"] == 0.05


def test_config_validates_default_agent_max_tool_rounds(tmp_path) -> None:
    path = tmp_path / "ai.json"
    assert resolve_config(path=path).default_agent_max_tool_rounds == DEFAULT_AGENT_MAX_TOOL_ROUNDS

    for value in (1, 64):
        save_user_config(
            provider="groq",
            model=DEFAULT_MODEL,
            timeout_seconds=30.0,
            max_completion_tokens=None,
            default_agent_max_tool_rounds=value,
            path=path,
        )
        config = resolve_config(path=path)
        assert config.default_agent_max_tool_rounds == value
        assert config.safe_dict()["default_agent_max_tool_rounds"] == value

    for value in (0, -1, 65, "six", 1.5):
        path.write_text(
            json.dumps(
                {
                    "provider": "groq",
                    "model": DEFAULT_MODEL,
                    "timeout_seconds": 30.0,
                    "default_agent_max_tool_rounds": value,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ProviderConfigurationError, match="default_agent_max_tool_rounds"):
            resolve_config(path=path)


def test_groq_rejects_empty_and_malformed_tool_call_responses() -> None:
    empty_message = Obj()
    empty_message.content = None
    empty_message.tool_calls = []
    empty_choice = Obj()
    empty_choice.message = empty_message
    empty_choice.finish_reason = "stop"
    empty_response = Obj()
    empty_response.choices = [empty_choice]
    empty_response.usage = None

    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(empty_response))
    with pytest.raises(
        ProviderMalformedResponseError, match="neither assistant text nor tool calls"
    ):
        provider.complete(ModelRequest(system_instruction="s", messages=()))

    bad_tool_choice = Obj()
    bad_tool_choice.message = empty_message
    bad_tool_choice.finish_reason = "tool_calls"
    bad_tool_response = Obj()
    bad_tool_response.choices = [bad_tool_choice]
    bad_tool_response.usage = None
    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(bad_tool_response))
    with pytest.raises(ProviderMalformedResponseError, match="finished with tool calls"):
        provider.complete(ModelRequest(system_instruction="s", messages=()))


def test_groq_assistant_tool_call_messages_are_sent_in_provider_shape() -> None:
    first = _response()
    final_message = Obj()
    final_message.content = "Done."
    final_message.tool_calls = []
    final_choice = Obj()
    final_choice.message = final_message
    final_choice.finish_reason = "stop"
    final_response = Obj()
    final_response.choices = [final_choice]
    final_response.usage = None
    client = FakeClient([first, final_response])
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    first_response = provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(ChatMessage(MessageRole.USER, "inspect"),),
            tools=(ToolDefinition("project.get_summary", "summary", {"type": "object"}),),
        )
    )
    provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(
                ChatMessage(MessageRole.USER, "inspect"),
                ChatMessage(MessageRole.ASSISTANT, "", tool_calls=first_response.tool_calls),
                ChatMessage(
                    MessageRole.TOOL, "{}", tool_call_id="call_1", name="project.get_summary"
                ),
            ),
            tools=(ToolDefinition("project.get_summary", "summary", {"type": "object"}),),
        )
    )

    second_messages = client.chat.completions.calls[1]["messages"]
    assistant = second_messages[2]
    tool_result = second_messages[3]
    assert assistant["role"] == "assistant"
    assert "content" not in assistant
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "project_get_summary"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {}
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_1"
    assert tool_result["name"] == "project_get_summary"


def test_groq_rejects_unmapped_provider_tool_name() -> None:
    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(_response(name="unknown_tool")))
    with pytest.raises(ProviderMalformedResponseError, match="unknown tool name"):
        provider.complete(
            ModelRequest(
                system_instruction="system text",
                messages=(ChatMessage(MessageRole.USER, "inspect"),),
                tools=(ToolDefinition("project.get_summary", "summary", {"type": "object"}),),
            )
        )


def test_groq_rejects_inconsistent_or_ambiguous_tool_call_envelopes() -> None:
    tools = (ToolDefinition("project.get_summary", "summary", {"type": "object"}),)

    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(_response(finish_reason="stop", calls=[_tool_call()])),
    )
    with pytest.raises(ProviderMalformedResponseError, match="non-tool finish reason"):
        provider.complete(ModelRequest(system_instruction="s", messages=(), tools=tools))

    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(_response(calls=[_tool_call(call_id="dup"), _tool_call(call_id="dup")])),
    )
    with pytest.raises(ProviderMalformedResponseError, match="duplicate tool call ids"):
        provider.complete(ModelRequest(system_instruction="s", messages=(), tools=tools))

    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(_response(calls=[_tool_call(call_type="custom")])),
    )
    with pytest.raises(ProviderMalformedResponseError, match="type was not function"):
        provider.complete(ModelRequest(system_instruction="s", messages=(), tools=tools))

    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(_response(calls=[_tool_call(arguments='{"limit":1,"limit":2}')])),
    )
    with pytest.raises(ProviderMalformedResponseError, match="duplicate keys"):
        provider.complete(ModelRequest(system_instruction="s", messages=(), tools=tools))


def test_groq_payload_uses_provider_safe_model_facing_tool_names() -> None:
    final_message = Obj()
    final_message.content = "Ready."
    final_message.tool_calls = []
    final_choice = Obj()
    final_choice.message = final_message
    final_choice.finish_reason = "stop"
    final_response = Obj()
    final_response.choices = [final_choice]
    final_response.usage = None
    client = FakeClient(final_response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)
    tools = ProjectToolExecutor(default_project_tool_registry(), ToolContext()).definitions(
        allow_mutation=True
    )

    provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(ChatMessage(MessageRole.USER, "inspect"),),
            tools=tools,
        )
    )

    names = [tool["function"]["name"] for tool in client.chat.completions.kwargs["tools"]]
    assert "project_add_requirement" in names
    assert "project_apply_changeset" not in names
    assert "project_preview_changeset" not in names
    assert all("." not in name for name in names)


def test_groq_chat_request_does_not_set_default_completion_token_limit() -> None:
    final_message = Obj()
    final_message.content = "Ready."
    final_message.tool_calls = []
    final_choice = Obj()
    final_choice.message = final_message
    final_choice.finish_reason = "stop"
    final_response = Obj()
    final_response.choices = [final_choice]
    final_response.usage = None
    client = FakeClient(final_response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    provider.complete(
        ModelRequest(
            system_instruction="system text",
            messages=(ChatMessage(MessageRole.USER, "inspect"),),
        )
    )

    assert "max_completion_tokens" not in client.chat.completions.kwargs


def test_groq_tool_use_failed_is_classified_and_redacted(monkeypatch) -> None:
    from hephaestus.ai import groq_provider as provider_module

    class FakeBadRequestError(Exception):
        status_code = 400
        request_id = "req_test_123"
        body = {
            "error": {
                "code": "tool_use_failed",
                "type": "invalid_request_error",
                "message": "tool call could not be validated",
                "failed_generation": (
                    "workspace_replace_text old_text=<sdf>SECRET</sdf> new_text=gsk_should_not_leak"
                ),
            }
        }

    monkeypatch.setattr(provider_module.groq, "BadRequestError", FakeBadRequestError)
    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(FakeBadRequestError("raw failed_generation should not leak")),
    )

    with pytest.raises(ProviderToolUseError) as raised:
        provider.complete(
            ModelRequest(
                system_instruction="s",
                messages=(),
                tools=(ToolDefinition("workspace_replace_text", "replace", {"type": "object"}),),
            )
        )

    error = raised.value
    assert "valid engineering operation" in str(error)
    assert "failed_generation" not in str(error)
    assert "SECRET" not in str(error)
    assert error.details is not None
    assert error.details.status_code == 400
    assert error.details.provider_code == "tool_use_failed"
    assert error.details.provider_error_type == "invalid_request_error"
    assert error.details.provider_reason == "tool call could not be validated"
    assert error.details.attempted_tool_name == "workspace_replace_text"
    assert error.details.request_id == "req_test_123"
    assert error.details.retryable is False
    encoded = json.dumps(error.details.to_dict())
    assert "failed_generation" not in encoded
    assert "SECRET" not in encoded
    assert "gsk_should_not_leak" not in encoded
    assert "<sdf>" not in encoded


def test_provider_error_metadata_is_bounded_and_secret_safe(monkeypatch) -> None:
    from hephaestus.ai import groq_provider as provider_module

    class FakeBadRequestError(Exception):
        status_code = 400
        request_id = "req_" + "x" * 400
        body = {
            "error": {
                "code": "bad secret gsk_should_not_leak",
                "type": "invalid_request_error",
                "message": "unsafe details",
                "failed_generation": "SECRET failed_generation payload",
            }
        }

    monkeypatch.setattr(provider_module.groq, "BadRequestError", FakeBadRequestError)
    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(FakeBadRequestError()))

    with pytest.raises(ProviderMalformedResponseError) as raised:
        provider.complete(ModelRequest(system_instruction="s", messages=()))

    error = raised.value
    assert str(error) == "AI provider rejected the request"
    assert error.details is not None
    payload = json.dumps(error.details.to_dict())
    assert "gsk_should_not_leak" not in payload
    assert "failed_generation" not in payload
    assert "SECRET" not in payload
    assert "bad secret" not in payload
    assert "invalid_request_error" in payload
    assert error.details.provider_code is None
    assert error.details.request_id is not None
    assert len(error.details.request_id) == 128


def test_groq_structured_request_uses_strict_json_schema_without_tools_or_token_limit() -> None:
    message = Obj()
    message.content = '{"robot_name":"Concept Bot"}'
    message.tool_calls = []
    choice = Obj()
    choice.message = message
    choice.finish_reason = "stop"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    client = FakeClient(response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    result = provider.complete_structured(
        StructuredPlanningRequest(
            system_instruction="Return JSON only.",
            user_prompt="Create a robot",
            schema_name="concept_robot_spec",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"robot_name": {"type": "string"}},
                "required": ["robot_name"],
            },
        )
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["messages"] == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Create a robot"},
    ]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "concept_robot_spec",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"robot_name": {"type": "string"}},
                "required": ["robot_name"],
            },
            "strict": True,
        },
    }
    assert "max_completion_tokens" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "stream" not in kwargs
    assert result.payload == {"robot_name": "Concept Bot"}


def test_groq_structured_request_sends_explicit_token_limit_exactly() -> None:
    message = Obj()
    message.content = '{"robot_name":"Concept Bot"}'
    message.tool_calls = []
    choice = Obj()
    choice.message = message
    choice.finish_reason = "stop"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    client = FakeClient(response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    provider.complete_structured(
        StructuredPlanningRequest(
            system_instruction="Return JSON only.",
            user_prompt="Create a robot",
            schema_name="concept_robot_spec",
            json_schema={"type": "object"},
            max_output_tokens=12345,
        )
    )

    assert client.chat.completions.kwargs["max_completion_tokens"] == 12345


def test_groq_structured_request_can_use_best_effort_json_schema() -> None:
    message = Obj()
    message.content = '{"robot_name":"Concept Bot"}'
    message.tool_calls = []
    choice = Obj()
    choice.message = message
    choice.finish_reason = "stop"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    client = FakeClient(response)
    provider = GroqProvider(model=DEFAULT_MODEL, client=client)

    provider.complete_structured(
        StructuredPlanningRequest(
            system_instruction="Return JSON only.",
            user_prompt="Create a robot",
            schema_name="concept_robot_spec",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"robot_name": {"type": "string"}},
                "required": ["robot_name"],
            },
            strict_schema=False,
            max_output_tokens=1024,
        )
    )

    kwargs = client.chat.completions.kwargs
    assert kwargs["response_format"]["json_schema"]["strict"] is False
    assert kwargs["max_completion_tokens"] == 1024
    assert "tools" not in kwargs
    assert "stream" not in kwargs


def test_groq_structured_json_validate_failed_is_classified_and_redacted(monkeypatch) -> None:
    from hephaestus.ai import groq_provider as provider_module

    class FakeBadRequestError(Exception):
        status_code = 400
        request_id = "req_structured_123"
        body = {
            "error": {
                "code": "json_validate_failed",
                "type": "invalid_request_error",
                "message": "schema validation failed",
                "failed_generation": "SECRET failed_generation payload",
            }
        }

    monkeypatch.setattr(provider_module.groq, "BadRequestError", FakeBadRequestError)
    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(FakeBadRequestError("raw failed_generation should not leak")),
    )

    with pytest.raises(ProviderStructuredGenerationError) as raised:
        provider.complete_structured(
            StructuredPlanningRequest(
                system_instruction="s",
                user_prompt="u",
                schema_name="schema",
                json_schema={"type": "object"},
            )
        )

    error = raised.value
    assert "schema-valid structured output" in str(error)
    assert "failed_generation" not in str(error)
    assert "SECRET" not in str(error)
    assert error.details is not None
    assert error.details.status_code == 400
    assert error.details.provider_code == "json_validate_failed"
    assert error.details.provider_error_type == "invalid_request_error"
    assert error.details.request_id == "req_structured_123"
    assert error.details.retryable is False


def test_groq_structured_413_rate_limit_is_account_limit_and_redacted(monkeypatch) -> None:
    from hephaestus.ai import groq_provider as provider_module

    class FakeRateLimitError(Exception):
        status_code = 413
        request_id = "req_rate_123"
        body = {
            "error": {
                "code": "rate_limit_exceeded",
                "type": "rate_limit_error",
                "message": "too many tokens",
                "failed_generation": "SECRET failed_generation payload",
            }
        }

    monkeypatch.setattr(provider_module.groq, "RateLimitError", FakeRateLimitError)
    provider = GroqProvider(
        model=DEFAULT_MODEL,
        client=FakeClient(FakeRateLimitError("raw failed_generation should not leak")),
    )

    with pytest.raises(ProviderRateLimitError) as raised:
        provider.complete_structured(
            StructuredPlanningRequest(
                system_instruction="s",
                user_prompt="u",
                schema_name="schema",
                json_schema={"type": "object"},
            )
        )

    error = raised.value
    assert str(error) == "AI provider account limit reached"
    assert "failed_generation" not in str(error)
    assert "SECRET" not in str(error)
    assert error.details is not None
    assert error.details.status_code == 413
    assert error.details.provider_code == "rate_limit_exceeded"
    assert error.details.provider_error_type == "rate_limit_error"
    assert error.details.request_id == "req_rate_123"
    assert error.details.retryable is False


def test_groq_rate_limit_preserves_safe_retry_timing(monkeypatch) -> None:
    from hephaestus.ai import groq_provider as provider_module

    class FakeResponse:
        headers = {
            "retry-after": "7",
            "x-ratelimit-reset-tokens": "12.5s",
            "x-request-id": "req_retry_123",
        }

    class FakeRateLimitError(Exception):
        status_code = 429
        request_id = None
        response = FakeResponse()
        body = {"error": {"code": "rate_limit_exceeded", "type": "rate_limit_error"}}

    monkeypatch.setattr(provider_module.groq, "RateLimitError", FakeRateLimitError)
    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(FakeRateLimitError("limit")))

    with pytest.raises(ProviderRateLimitError) as raised:
        provider.complete(ModelRequest(messages=[], system_instruction="s"))

    details = raised.value.details
    assert details is not None
    assert details.status_code == 429
    assert details.retry_after_seconds == 7.0
    assert details.token_reset_seconds == 12.5
    assert details.request_id == "req_retry_123"


def test_groq_structured_response_rejects_malformed_json() -> None:
    message = Obj()
    message.content = "not json"
    message.tool_calls = []
    choice = Obj()
    choice.message = message
    choice.finish_reason = "stop"
    response = Obj()
    response.choices = [choice]
    response.usage = None
    provider = GroqProvider(model=DEFAULT_MODEL, client=FakeClient(response))

    with pytest.raises(ProviderMalformedResponseError, match="malformed JSON"):
        provider.complete_structured(
            StructuredPlanningRequest(
                system_instruction="s",
                user_prompt="u",
                schema_name="schema",
                json_schema={"type": "object"},
            )
        )
