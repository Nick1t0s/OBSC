import os
import re
from pathlib import Path

import yaml

from fast_ai.exceptions import AllProvidersFailedError, ConfigurationError, ProviderError
from fast_ai.logger import get_logger
from fast_ai.stt.base import STTProvider, STTResponse
from fast_ai.stt.providers.faster_whisper import FasterWhisperProvider
from fast_ai.stt.providers.openai import OpenAIWhisperProvider

log = get_logger("stt")


class STT:
    """Fault-tolerant speech-to-text orchestrator.

    Tries providers in order with configurable retry rounds,
    same pattern as FastLLM / OCR / ImageDescriber.

    Args:
        providers: Ordered list of STTProvider instances.
        max_rounds: How many full passes through the provider list before raising.

    Config format (``stt_config.yaml``)::

        max_rounds: 2
        providers:
          - type: faster_whisper
            model: base
            device: auto
            compute_type: default
            beam_size: 5
            vad_filter: true
            timeout: 120
          - type: openai_whisper
            base_url: https://api.openai.com
            model: whisper-1
            token: ${OPENAI_API_KEY}
            timeout: 60
    """

    PROVIDER_REGISTRY: dict[str, type[STTProvider]] = {
        "faster_whisper": FasterWhisperProvider,
        "openai_whisper": OpenAIWhisperProvider,
    }

    def __init__(self, providers: list[STTProvider], max_rounds: int = 1):
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
    def build(cls, config_path: str | Path) -> "STT":
        """Build an STT instance from a YAML config file.

        Values containing ``${VAR}`` are expanded from environment variables.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not cfg or "providers" not in cfg:
            raise ConfigurationError("config must contain a 'providers' list")

        providers: list[STTProvider] = []
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

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> STTResponse:
        """Transcribe an audio file using the provider chain.

        Args:
            audio: Path to the audio file.
            language: Language code (e.g. "ru", "en"). None for auto-detect.
            timeout: Override per-provider timeout for this call.
            **kwargs: Provider-specific parameters forwarded as-is.

        Returns:
            STTResponse from the first provider that succeeds.

        Raises:
            AllProvidersFailedError: Every provider failed across all rounds.
        """
        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        errors: list[ProviderError] = []

        for round_num in range(1, self.max_rounds + 1):
            log.debug("round %d/%d", round_num, self.max_rounds)

            for provider in self.providers:
                try:
                    result = provider.transcribe(
                        audio_path,
                        language=language,
                        timeout=timeout,
                        **kwargs,
                    )
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
