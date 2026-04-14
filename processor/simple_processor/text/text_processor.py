from dataclasses import dataclass
from pathlib import Path

from fast_ai.exceptions import ConfigurationError
from processor.simple_processor.base_processor import BaseProcessor
from processor.simple_processor.text.chunker import chunk_text
from processor.simple_processor.text.embedder import Embedder


@dataclass
class Chunk:
    """A chunk of source text with its embedding and position.

    ``start`` and ``end`` are character offsets into the original source,
    useful for reconstructing surrounding context at retrieval time.
    """

    index: int
    text: str
    start: int
    end: int
    embedding: list[float]


@dataclass
class Text:
    """Result of processing a text: original source + per-chunk embeddings."""

    source: str
    chunks: list[Chunk]
    model: str
    dim: int


class TextProcessor(BaseProcessor):
    """Splits text into overlapping chunks and computes an embedding for each.

    Config format (``text_processor_config.yaml``)::

        embedder_config: embedder_config.yaml
        chunking:
          size: 800
          overlap: 100
    """

    def __init__(
        self,
        embedder: Embedder,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def build(cls, config_path: str | Path) -> "TextProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "embedder_config" not in cfg:
            raise ConfigurationError("missing required key: embedder_config")

        config_dir = path.parent
        embedder = Embedder.build(config_dir / cfg["embedder_config"])

        chunking = cfg.get("chunking") or {}
        return cls(
            embedder=embedder,
            chunk_size=chunking.get("size", 800),
            chunk_overlap=chunking.get("overlap", 100),
        )

    def run(self, text: str) -> Text:
        raw = chunk_text(text, size=self.chunk_size, overlap=self.chunk_overlap)
        if not raw:
            return Text(source=text, chunks=[], model="", dim=0)

        response = self.embedder.embed([rc.text for rc in raw])
        chunks = [
            Chunk(
                index=i,
                text=rc.text,
                start=rc.start,
                end=rc.end,
                embedding=emb,
            )
            for i, (rc, emb) in enumerate(zip(raw, response.embeddings))
        ]
        return Text(
            source=text,
            chunks=chunks,
            model=response.model,
            dim=response.dim,
        )
