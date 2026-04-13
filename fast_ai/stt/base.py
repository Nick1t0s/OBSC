from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class STTSegment:
    """A single transcription segment with timing info."""

    start: float
    end: float
    text: str


@dataclass
class STTResponse:
    """Response from an STT provider."""

    provider_name: str
    text: str
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    segments: list[STTSegment] = field(default_factory=list)


class STTProvider(ABC):
    """Abstract base class for speech-to-text providers.

    Args:
        model: Model name to use.
        retries: How many times this provider retries on failure before giving up.
        timeout: Processing timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        retries: int = 1,
        timeout: float = 120.0,
    ):
        if not model:
            raise ValueError("model must not be empty")
        if retries < 1:
            raise ValueError("retries must be >= 1")

        self.model = model
        self.retries = retries
        self.timeout = timeout

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> STTResponse:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g. "ru", "en"). None for auto-detect.
            timeout: Override instance-level timeout for this call.
            **kwargs: Provider-specific parameters.

        Returns:
            STTResponse with transcription text and metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
