import base64
from pathlib import Path

from fast_ai.exceptions import AllProvidersFailedError, ConfigurationError, ProviderError
from fast_ai.llm.fast_llm import FastLLM
from fast_ai.logger import get_logger

log = get_logger("describer")

DEFAULT_PROMPT = (
    "Опиши подробно, что изображено на картинке. "
    "Укажи основные объекты, их расположение, цвета и общую атмосферу."
)


class ImageDescriber:
    """Vision-based image description built on top of FastLLM.

    Reads a YAML config with ordered providers, encodes images,
    and generates descriptions using the fault-tolerant provider chain.

    Config format (``describer_config.yaml``)::

        prompt: "Опиши подробно, что изображено на картинке..."
        max_rounds: 2
        providers:
          - type: ollama
            base_url: http://localhost:11434
            model: qwen2.5vl:7b
            retries: 2
            timeout: 120
    """

    def __init__(self, llm: FastLLM, prompt: str = DEFAULT_PROMPT):
        self.llm = llm
        self.prompt = prompt

    @classmethod
    def build(cls, config_path: str | Path) -> "ImageDescriber":
        """Build an ImageDescriber instance from a YAML config file.

        The config is the same format as ``FastLLM.build`` accepts,
        with an optional top-level ``prompt`` field for the description instruction.
        """
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        prompt = cfg.get("prompt", DEFAULT_PROMPT)
        llm = FastLLM.build(config_path)
        return cls(llm=llm, prompt=prompt)

    def describe(
        self,
        image: str | Path | bytes,
        *,
        prompt: str | None = None,
    ) -> str:
        """Generate a description of an image.

        Args:
            image: File path to an image, or raw bytes.
            prompt: Override the default description prompt for this call.

        Returns:
            Description text string.

        Raises:
            AllProvidersFailedError: Every provider failed across all rounds.
        """
        image_b64 = self._encode(image)
        effective_prompt = prompt or self.prompt
        errors: list[ProviderError] = []

        for round_num in range(1, self.llm.max_rounds + 1):
            log.debug("round %d/%d", round_num, self.llm.max_rounds)

            for provider in self.llm.providers:
                messages = provider.format_vision_messages(
                    effective_prompt, [image_b64],
                )
                try:
                    response = provider.generate(messages)
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

    @staticmethod
    def _encode(image: str | Path | bytes) -> str:
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return base64.b64encode(path.read_bytes()).decode()
