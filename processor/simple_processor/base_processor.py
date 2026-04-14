from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class BaseProcessor(ABC):
    """Abstract base for simple processors."""

    @abstractmethod
    def run(self, source: str | Path | bytes) -> dataclass:
        ...
