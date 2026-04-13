class FastAIError(Exception):
    """Base exception for fast_ai module."""


class ProviderError(FastAIError):
    """Raised when a single provider fails."""

    def __init__(self, provider_name: str, message: str, cause: Exception | None = None):
        self.provider_name = provider_name
        self.cause = cause
        super().__init__(f"[{provider_name}] {message}")


class AllProvidersFailedError(FastAIError):
    """Raised when every provider in the chain has failed across all rounds."""

    def __init__(self, errors: list[ProviderError]):
        self.errors = errors
        details = "; ".join(str(e) for e in errors)
        super().__init__(f"All providers failed: {details}")


class ConfigurationError(FastAIError):
    """Raised on invalid configuration (missing url, bad params, etc.)."""
