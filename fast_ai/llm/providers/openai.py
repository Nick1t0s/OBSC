import requests

from fast_ai.exceptions import ProviderError
from fast_ai.llm.base import LLMProvider, LLMResponse
from fast_ai.logger import get_logger

log = get_logger("llm.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (OpenAI, Together, etc.)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        token: str,
        retries: int = 1,
        timeout: float = 30.0,
    ):
        super().__init__(base_url=base_url, model=model, retries=retries, timeout=timeout)
        if not token:
            raise ValueError("token must not be empty")
        self.token = token

    @property
    def provider_name(self) -> str:
        return f"openai({self.base_url}, {self.model})"

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> LLMResponse:
        url = f"{self.base_url}/v1/chat/completions"
        effective_timeout = timeout if timeout is not None else self.timeout

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  POST %s  model=%s  timeout=%.1fs",
                attempt, self.retries, url, self.model, effective_timeout,
            )
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=effective_timeout,
                )
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
