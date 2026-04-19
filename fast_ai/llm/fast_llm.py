import os
import re
from pathlib import Path

import yaml

from fast_ai.exceptions import AllProvidersFailedError, ConfigurationError, ProviderError
from fast_ai.llm.base import LLMProvider, LLMResponse
from fast_ai.llm.providers.ollama import OllamaProvider
from fast_ai.llm.providers.openai import OpenAIProvider
from fast_ai.logger import get_logger

log = get_logger("llm.fast_llm")


class FastLLM:
    """Orchestrator that tries providers in order, with configurable rounds.

    Args:
        providers: Ordered list of LLMProvider instances.
        max_rounds: How many full passes through the provider list before raising.
    """

    def __init__(self, providers: list[LLMProvider], max_rounds: int = 1):
        if not providers:
            raise ConfigurationError("providers list must not be empty")
        if max_rounds < 1:
            raise ConfigurationError("max_rounds must be >= 1")

        self.providers = providers
        self.max_rounds = max_rounds

        log.info(
            "initialized with %d provider(s), max_rounds=%d: %s",
            len(providers), max_rounds, [p.provider_name for p in providers],
        )

    PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def build(cls, config_path: str | Path) -> "FastLLM":
        """Build a FastLLM instance from a YAML config file.

        Config format::

            max_rounds: 2
            providers:
              - type: ollama
                base_url: http://localhost:11434
                model: qwen2.5vl:7b
                retries: 2
                timeout: 60
              - type: openai
                base_url: https://api.openai.com
                model: gpt-4o
                token: ${OPENAI_API_KEY}

        Values containing ``${VAR}`` are expanded from environment variables.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not cfg or "providers" not in cfg:
            raise ConfigurationError("config must contain a 'providers' list")

        providers: list[LLMProvider] = []
        for entry in cfg["providers"]:
            entry = {k: cls._expand_env(v) for k, v in entry.items()}
            ptype = entry.pop("type", None)
            if ptype not in cls.PROVIDER_REGISTRY:
                raise ConfigurationError(
                    f"unknown provider type {ptype!r}, "
                    f"available: {list(cls.PROVIDER_REGISTRY)}"
                )
            provider_cls = cls.PROVIDER_REGISTRY[ptype]
            providers.append(provider_cls(**entry))

        max_rounds = cfg.get("max_rounds", 1)
        return cls(providers=providers, max_rounds=max_rounds)

    @staticmethod
    def _expand_env(value: object) -> object:
        if not isinstance(value, str):
            return value
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Try each provider in order; repeat up to max_rounds times.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            timeout: Override per-provider timeout for this call.
            **kwargs: Provider-specific parameters forwarded as-is.

        Returns:
            LLMResponse from the first provider that succeeds.

        Raises:
            AllProvidersFailedError: Every provider failed across all rounds.
        """
        errors: list[ProviderError] = []

        for round_num in range(1, self.max_rounds + 1):
            log.debug("round %d/%d", round_num, self.max_rounds)

            for provider in self.providers:
                try:
                    result = provider.generate(messages, timeout=timeout, **kwargs)
                    log.info("provider %s succeeded in round %d", provider.provider_name, round_num)
                    return result
                except ProviderError as exc:
                    log.warning(
                        "provider %s failed in round %d: %s",
                        provider.provider_name, round_num, exc,
                    )
                    errors.append(exc)

        raise AllProvidersFailedError(errors)

    def generate_vision(
        self,
        prompt: str,
        images_base64: list[str],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> str:
        """Vision-запрос через цепочку провайдеров.

        Каждый провайдер сам формирует сообщения (`format_vision_messages`)
        и извлекает текст ответа (`extract_content`).

        Raises:
            AllProvidersFailedError: все провайдеры упали во всех раундах.
        """
        errors: list[ProviderError] = []

        for round_num in range(1, self.max_rounds + 1):
            log.debug("vision round %d/%d", round_num, self.max_rounds)

            for provider in self.providers:
                messages = provider.format_vision_messages(prompt, images_base64)
                try:
                    response = provider.generate(messages, timeout=timeout, **kwargs)
                    text = provider.extract_content(response)
                    log.info(
                        "provider %s succeeded in round %d",
                        provider.provider_name, round_num,
                    )
                    return text
                except ProviderError as exc:
                    log.warning(
                        "provider %s failed in round %d: %s",
                        provider.provider_name, round_num, exc,
                    )
                    errors.append(exc)

        raise AllProvidersFailedError(errors)
