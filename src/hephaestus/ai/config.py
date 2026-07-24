"""Safe AI configuration resolution."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

from hephaestus.ai.types import ProviderConfigurationError

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_AGENT_TOOL_TEMPERATURE = 0.2
DEFAULT_AGENT_MAX_TOOL_ROUNDS = 16
MIN_AGENT_MAX_TOOL_ROUNDS = 1
MAX_AGENT_MAX_TOOL_ROUNDS = 64
CONFIG_FILENAME = "ai.json"
CONFIG_DIR_ENV = "HEPHAESTUS_CONFIG_DIR"


@dataclass(frozen=True, slots=True)
class AIConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_completion_tokens: int | None = None
    default_agent_tool_temperature: float = DEFAULT_AGENT_TOOL_TEMPERATURE
    default_agent_max_tool_rounds: int = DEFAULT_AGENT_MAX_TOOL_ROUNDS
    config_path: Path | None = None
    credential_present: bool = False

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_completion_tokens": (
                self.max_completion_tokens
                if self.max_completion_tokens is not None
                else "provider default"
            ),
            "default_agent_tool_temperature": self.default_agent_tool_temperature,
            "default_agent_max_tool_rounds": self.default_agent_max_tool_rounds,
            "configuration_path": str(self.config_path) if self.config_path else None,
            "credential_present": self.credential_present,
        }


def config_path(env: Mapping[str, str] | None = None) -> Path:
    environment = env if env is not None else os.environ
    override = environment.get(CONFIG_DIR_ENV)
    base = Path(override) if override else Path(user_config_dir("hephaestus-ai"))
    return base / CONFIG_FILENAME


def load_user_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderConfigurationError("AI configuration could not be read") from exc
    if not isinstance(data, dict):
        raise ProviderConfigurationError("AI configuration must be a JSON object")
    return data


def save_user_config(
    *,
    provider: str,
    model: str,
    timeout_seconds: float,
    max_completion_tokens: int | None,
    default_agent_tool_temperature: float | None = None,
    default_agent_max_tool_rounds: int | None = None,
    path: Path,
) -> Path:
    if provider != DEFAULT_PROVIDER:
        raise ProviderConfigurationError("only provider 'groq' is supported")
    if not model.strip():
        raise ProviderConfigurationError("model must not be empty")
    if max_completion_tokens is not None:
        if not isinstance(max_completion_tokens, int) or isinstance(max_completion_tokens, bool):
            raise ProviderConfigurationError("max_completion_tokens must be an integer")
        if max_completion_tokens <= 0:
            raise ProviderConfigurationError("max_completion_tokens must be positive")
    if default_agent_tool_temperature is not None:
        _validate_temperature(default_agent_tool_temperature)
    if default_agent_max_tool_rounds is not None:
        _validate_agent_max_tool_rounds(default_agent_max_tool_rounds)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "provider": provider,
        "timeout_seconds": float(timeout_seconds),
    }
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = int(max_completion_tokens)
    if default_agent_tool_temperature is not None:
        payload["default_agent_tool_temperature"] = float(default_agent_tool_temperature)
    if default_agent_max_tool_rounds is not None:
        payload["default_agent_max_tool_rounds"] = int(default_agent_max_tool_rounds)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def resolve_config(
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    timeout_override: float | None = None,
    max_completion_tokens_override: int | None = None,
    default_agent_tool_temperature_override: float | None = None,
    default_agent_max_tool_rounds_override: int | None = None,
    env: Mapping[str, str] | None = None,
    path: Path | None = None,
) -> AIConfig:
    environment = env if env is not None else os.environ
    resolved_path = path if path is not None else config_path(environment)
    user = load_user_config(resolved_path)
    provider = (
        provider_override
        or environment.get("HEPHAESTUS_AI_PROVIDER")
        or user.get("provider")
        or DEFAULT_PROVIDER
    )
    model = (
        model_override
        or environment.get("HEPHAESTUS_AI_MODEL")
        or user.get("model")
        or DEFAULT_MODEL
    )
    timeout = (
        timeout_override
        if timeout_override is not None
        else user.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    )
    max_completion_tokens = (
        max_completion_tokens_override
        if max_completion_tokens_override is not None
        else user.get("max_completion_tokens")
    )
    default_agent_tool_temperature = (
        default_agent_tool_temperature_override
        if default_agent_tool_temperature_override is not None
        else environment.get("HEPHAESTUS_DEFAULT_AGENT_TOOL_TEMPERATURE")
        or user.get("default_agent_tool_temperature", DEFAULT_AGENT_TOOL_TEMPERATURE)
    )
    default_agent_max_tool_rounds = (
        default_agent_max_tool_rounds_override
        if default_agent_max_tool_rounds_override is not None
        else environment.get("HEPHAESTUS_DEFAULT_AGENT_MAX_TOOL_ROUNDS")
        or user.get("default_agent_max_tool_rounds", DEFAULT_AGENT_MAX_TOOL_ROUNDS)
    )
    if provider != DEFAULT_PROVIDER:
        raise ProviderConfigurationError("only provider 'groq' is supported")
    try:
        timeout_seconds = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError("timeout_seconds must be numeric") from exc
    if timeout_seconds <= 0:
        raise ProviderConfigurationError("timeout_seconds must be positive")
    if max_completion_tokens is not None:
        if not isinstance(max_completion_tokens, int) or isinstance(max_completion_tokens, bool):
            raise ProviderConfigurationError("max_completion_tokens must be an integer")
        if max_completion_tokens <= 0:
            raise ProviderConfigurationError("max_completion_tokens must be positive")
    try:
        temperature = float(default_agent_tool_temperature)
    except (TypeError, ValueError) as exc:
        raise ProviderConfigurationError("default_agent_tool_temperature must be numeric") from exc
    _validate_temperature(temperature)
    max_tool_rounds = _parse_agent_max_tool_rounds(default_agent_max_tool_rounds)
    return AIConfig(
        provider=provider,
        model=str(model),
        timeout_seconds=timeout_seconds,
        max_completion_tokens=max_completion_tokens,
        default_agent_tool_temperature=temperature,
        default_agent_max_tool_rounds=max_tool_rounds,
        config_path=resolved_path,
        credential_present=bool(environment.get("GROQ_API_KEY", "").strip()),
    )


def _validate_temperature(value: float) -> None:
    if isinstance(value, bool) or not 0.0 <= float(value) <= 2.0:
        raise ProviderConfigurationError("default_agent_tool_temperature must be between 0 and 2")


def _parse_agent_max_tool_rounds(value: object) -> int:
    if isinstance(value, bool):
        raise ProviderConfigurationError("default_agent_max_tool_rounds must be an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError as exc:
            raise ProviderConfigurationError(
                "default_agent_max_tool_rounds must be an integer"
            ) from exc
        if str(parsed) != value.strip():
            raise ProviderConfigurationError("default_agent_max_tool_rounds must be an integer")
    else:
        raise ProviderConfigurationError("default_agent_max_tool_rounds must be an integer")
    _validate_agent_max_tool_rounds(parsed)
    return parsed


def _validate_agent_max_tool_rounds(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProviderConfigurationError("default_agent_max_tool_rounds must be an integer")
    if not MIN_AGENT_MAX_TOOL_ROUNDS <= value <= MAX_AGENT_MAX_TOOL_ROUNDS:
        raise ProviderConfigurationError("default_agent_max_tool_rounds must be between 1 and 64")
