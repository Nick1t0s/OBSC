import base64
from pathlib import Path

from fast_ai.exceptions import ConfigurationError
from fast_ai.llm.fast_llm import FastLLM
from fast_ai.logger import get_logger

log = get_logger("ocr.ocr")

DEFAULT_PROMPT = (
    "Распознай весь текст на изображении. "
    "Верни только распознанный текст, без пояснений."
)


class OCR:
    """Vision-based OCR built on top of FastLLM.

    Reads a YAML config with ordered providers, encodes images,
    and extracts text using the fault-tolerant provider chain.

    Config format (``ocr_config.yaml``)::

        prompt: "Распознай весь текст на изображении..."
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
    def build(cls, config_path: str | Path) -> "OCR":
        """Build an OCR instance from a YAML config file.

        The config is the same format as ``FastLLM.build`` accepts,
        with an optional top-level ``prompt`` field for the OCR instruction.
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

    def recognize(
        self,
        image: str | Path | bytes,
        *,
        prompt: str | None = None,
    ) -> str:
        """Extract text from an image.

        Args:
            image: File path to an image, or raw bytes.
            prompt: Override the default OCR prompt for this call.

        Returns:
            Extracted text string.

        Raises:
            AllProvidersFailedError: Every provider failed across all rounds.
        """
        image_b64 = self._encode(image)
        effective_prompt = prompt or self.prompt
        return self.llm.generate_vision(effective_prompt, [image_b64])

    @staticmethod
    def _encode(image: str | Path | bytes) -> str:
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        return base64.b64encode(path.read_bytes()).decode()
