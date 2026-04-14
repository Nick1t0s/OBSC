from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbeddingResponse:
    """Result of an embedding call for a batch of texts."""

    provider_name: str
    model: str
    embeddings: list[list[float]]
    dim: int


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Args:
        base_url: API base URL.
        model: Embedding model name.
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
    def embed(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        """Compute embeddings for a batch of texts.

        Args:
            texts: List of input strings.
            timeout: Override instance-level timeout for this call.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r}, model={self.model!r})"
