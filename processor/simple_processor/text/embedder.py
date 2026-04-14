import os
import re
from pathlib import Path

import yaml

from fast_ai.exceptions import AllProvidersFailedError, ConfigurationError, ProviderError
from fast_ai.logger import get_logger
from processor.simple_processor.text.base import EmbeddingProvider, EmbeddingResponse
from processor.simple_processor.text.providers.ollama import OllamaEmbeddingProvider
from processor.simple_processor.text.providers.openai import OpenAIEmbeddingProvider

log = get_logger("embedder")


class Embedder:
    """Fault-tolerant embedding orchestrator.

    Mirrors ``FastLLM``: tries providers in order, repeats up to ``max_rounds``
    times, returns the first successful ``EmbeddingResponse``.

    Config format (``embedder_config.yaml``)::

        max_rounds: 2
        providers:
          - type: ollama
            base_url: http://localhost:11434
            model: nomic-embed-text
            retries: 2
            timeout: 60
          - type: openai
            base_url: https://api.openai.com
            model: text-embedding-3-small
            token: ${OPENAI_API_KEY}
    """

    PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {
        "ollama": OllamaEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
    }

    def __init__(self, providers: list[EmbeddingProvider], max_rounds: int = 1):
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

    @classmethod
    def build(cls, config_path: str | Path) -> "Embedder":
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not cfg or "providers" not in cfg:
            raise ConfigurationError("config must contain a 'providers' list")

        providers: list[EmbeddingProvider] = []
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

    def embed(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        """Embed a batch of texts, trying providers until one succeeds.

        Raises:
            AllProvidersFailedError: Every provider failed across all rounds.
        """
        errors: list[ProviderError] = []

        for round_num in range(1, self.max_rounds + 1):
            log.debug("round %d/%d", round_num, self.max_rounds)

            for provider in self.providers:
                try:
                    result = provider.embed(texts, timeout=timeout)
                    log.info(
                        "provider %s succeeded in round %d",
                        provider.provider_name, round_num,
                    )
                    return result
                except ProviderError as exc:
                    log.warning(
                        "provider %s failed in round %d: %s",
                        provider.provider_name, round_num, exc,
                    )
                    errors.append(exc)

        raise AllProvidersFailedError(errors)
