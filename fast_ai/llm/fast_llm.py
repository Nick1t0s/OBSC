from fast_ai.exceptions import AllProvidersFailedError, ConfigurationError, ProviderError
from fast_ai.llm.base import LLMProvider, LLMResponse
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
