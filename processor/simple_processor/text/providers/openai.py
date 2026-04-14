import requests

from fast_ai.exceptions import ProviderError
from fast_ai.logger import get_logger
from processor.simple_processor.text.base import EmbeddingProvider, EmbeddingResponse

log = get_logger("embedder.openai")


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embeddings provider (OpenAI, Together, etc.).

    Hits ``POST {base_url}/v1/embeddings``.
    """

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
        return f"openai-embed({self.base_url}, {self.model})"

    def embed(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        url = f"{self.base_url}/v1/embeddings"
        effective_timeout = timeout if timeout is not None else self.timeout

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  POST %s  model=%s  n=%d  timeout=%.1fs",
                attempt, self.retries, url, self.model, len(texts), effective_timeout,
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
            embeddings = [item["embedding"] for item in body["data"]]
            dim = len(embeddings[0]) if embeddings else 0
            log.info("success on attempt %d/%d (n=%d, dim=%d)", attempt, self.retries, len(embeddings), dim)
            return EmbeddingResponse(
                provider_name=self.provider_name,
                model=body.get("model", self.model),
                embeddings=embeddings,
                dim=dim,
            )

        raise ProviderError(
            self.provider_name, f"all {self.retries} attempts failed", cause=last_err,
        )
