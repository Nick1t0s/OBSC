from pathlib import Path

import requests

from fast_ai.exceptions import ProviderError
from fast_ai.logger import get_logger
from fast_ai.stt.base import STTProvider, STTResponse, STTSegment

log = get_logger("stt.openai_whisper")


class OpenAIWhisperProvider(STTProvider):
    """Speech-to-text provider using the OpenAI Audio API.

    Uses the ``/v1/audio/transcriptions`` endpoint.

    Args:
        base_url: API base URL (e.g. "https://api.openai.com").
        model: Whisper model name (e.g. "whisper-1").
        token: API bearer token.
        response_format: Response format — "verbose_json" gives segments/duration.
        retries: How many times to retry on failure.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "whisper-1",
        token: str = "",
        response_format: str = "verbose_json",
        retries: int = 1,
        timeout: float = 120.0,
    ):
        super().__init__(model=model, retries=retries, timeout=timeout)
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not token:
            raise ValueError("token must not be empty")

        self.base_url = base_url.rstrip("/")
        self.token = token
        self.response_format = response_format

    @property
    def provider_name(self) -> str:
        return f"openai_whisper({self.base_url}, {self.model})"

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> STTResponse:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        url = f"{self.base_url}/v1/audio/transcriptions"
        effective_timeout = timeout if timeout is not None else self.timeout

        headers = {"Authorization": f"Bearer {self.token}"}

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  POST %s  model=%s  timeout=%.1fs",
                attempt, self.retries, url, self.model, effective_timeout,
            )
            try:
                with open(audio_path, "rb") as f:
                    data = {
                        "model": (None, self.model),
                        "response_format": (None, self.response_format),
                    }
                    if language:
                        data["language"] = (None, language)

                    # Forward any extra kwargs as form fields
                    for key, val in kwargs.items():
                        data[key] = (None, str(val))

                    resp = requests.post(
                        url,
                        headers=headers,
                        files={"file": (audio_path.name, f)},
                        data={k: v[1] for k, v in data.items()},
                        timeout=effective_timeout,
                    )
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_err = exc
                log.warning("attempt %d/%d failed: %s", attempt, self.retries, exc)
                continue

            body = resp.json()
            segments = [
                STTSegment(start=s["start"], end=s["end"], text=s["text"].strip())
                for s in body.get("segments", [])
            ]

            log.info("success on attempt %d/%d", attempt, self.retries)
            return STTResponse(
                provider_name=self.provider_name,
                text=body.get("text", ""),
                language=body.get("language"),
                duration=body.get("duration"),
                segments=segments,
            )

        raise ProviderError(
            self.provider_name,
            f"all {self.retries} attempts failed",
            cause=last_err,
        )
