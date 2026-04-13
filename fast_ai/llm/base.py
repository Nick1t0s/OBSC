from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Raw response from an LLM provider."""

    provider_name: str
    body: dict
    status_code: int


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Args:
        base_url: API base URL (e.g. "http://localhost:11434" for Ollama).
        model: Model name to use.
        retries: How many times this provider retries on failure before giving up.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        retries: int = 1,
        timeout: float = 30.0,
    ):
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if retries < 1:
            raise ValueError("retries must be >= 1")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retries = retries
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send messages and return the raw response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            timeout: Override instance-level timeout for this call.
            **kwargs: Provider-specific parameters forwarded to the API.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r})"
