import requests

from fast_ai.exceptions import ProviderError
from fast_ai.llm.base import LLMProvider, LLMResponse
from fast_ai.logger import get_logger

log = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """Ollama provider using native /api/chat endpoint."""

    @property
    def provider_name(self) -> str:
        return f"ollama({self.base_url}, {self.model})"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        effective_timeout = timeout if timeout is not None else self.timeout

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            **kwargs,
        }

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  POST %s  model=%s  timeout=%.1fs",
                attempt, self.retries, url, self.model, effective_timeout,
            )
            try:
                resp = requests.post(url, json=payload, timeout=effective_timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_err = exc
                log.warning("attempt %d/%d failed: %s", attempt, self.retries, exc)
                continue

            body = resp.json()
            log.info("success on attempt %d/%d", attempt, self.retries)
            return LLMResponse(
                provider_name=self.provider_name,
                body=body,
                status_code=resp.status_code,
            )

        raise ProviderError(self.provider_name, f"all {self.retries} attempts failed", cause=last_err)
