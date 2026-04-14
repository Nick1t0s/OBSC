import requests

from fast_ai.exceptions import ProviderError
from fast_ai.logger import get_logger
from processor.simple_processor.text.base import EmbeddingProvider, EmbeddingResponse

log = get_logger("embedder.ollama")


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embeddings provider.

    Hits ``POST {base_url}/api/embed`` (batch-capable endpoint; requires
    Ollama >= 0.2.0).
    """

    @property
    def provider_name(self) -> str:
        return f"ollama-embed({self.base_url}, {self.model})"

    def embed(
        self,
        texts: list[str],
        *,
        timeout: float | None = None,
    ) -> EmbeddingResponse:
        url = f"{self.base_url}/api/embed"
        effective_timeout = timeout if timeout is not None else self.timeout
        payload = {"model": self.model, "input": texts}

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  POST %s  model=%s  n=%d  timeout=%.1fs",
                attempt, self.retries, url, self.model, len(texts), effective_timeout,
            )
            try:
                resp = requests.post(url, json=payload, timeout=effective_timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                last_err = exc
                log.warning("attempt %d/%d failed: %s", attempt, self.retries, exc)
                continue

            body = resp.json()
            embeddings = body.get("embeddings") or []
            dim = len(embeddings[0]) if embeddings else 0
            log.info("success on attempt %d/%d (n=%d, dim=%d)", attempt, self.retries, len(embeddings), dim)
            return EmbeddingResponse(
                provider_name=self.provider_name,
                model=self.model,
                embeddings=embeddings,
                dim=dim,
            )

        raise ProviderError(
            self.provider_name, f"all {self.retries} attempts failed", cause=last_err,
        )
