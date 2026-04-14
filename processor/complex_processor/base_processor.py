from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class BaseComplexProcessor(ABC):
    """Abstract base for complex processors."""

    @abstractmethod
    def run(self, source: str | Path | bytes) -> dataclass:
        ...
