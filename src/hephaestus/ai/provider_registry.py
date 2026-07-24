"""Provider registry and factory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from hephaestus.ai.config import AIConfig
from hephaestus.ai.groq_provider import GroqProvider, provider_from_environment
from hephaestus.ai.types import ModelProvider, ProviderConfigurationError

ProviderFactory = Callable[[AIConfig], ModelProvider]


@dataclass(slots=True)
class ProviderRegistry:
    factories: dict[str, ProviderFactory] = field(default_factory=dict)

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        if provider_id in self.factories:
            raise ProviderConfigurationError(f"provider already registered: {provider_id}")
        self.factories[provider_id] = factory

    def create(self, config: AIConfig) -> ModelProvider:
        try:
            factory = self.factories[config.provider]
        except KeyError as exc:
            raise ProviderConfigurationError(f"unsupported provider: {config.provider}") from exc
        return factory(config)


def default_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        "groq",
        lambda config: provider_from_environment(
            model=config.model, timeout_seconds=config.timeout_seconds
        ),
    )
    return registry


def injected_groq_registry(provider: GroqProvider | ModelProvider) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("groq", lambda config: provider)
    return registry
