from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Attachment:
    """A processed attachment within an object document.

    ``source`` is a human-readable identifier (file name or synthetic id).
    ``kind`` is the result dataclass produced by the underlying processor
    (``Photo``, ``PDF``, ``Word``, ...) — every such dataclass exposes
    ``render() -> str``.
    """

    source: str
    kind: Any

    def render(self) -> str:
        return self.kind.render()


class BaseObjectProcessor(ABC):
    """Abstract base for object processors.

    An object processor handles a whole document from some source
    (an email, a Telegram post, an article, ...) consisting of a
    free-form text body and an arbitrary list of attachments.
    """

    @abstractmethod
    def run(self, text, attachments, **metadata):
        ...

    @abstractmethod
    def render(self, obj) -> str:
        ...
