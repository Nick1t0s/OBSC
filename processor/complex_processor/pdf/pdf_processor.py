from dataclasses import dataclass, field
from pathlib import Path

from fast_ai.exceptions import ConfigurationError
from processor.complex_processor.base_processor import BaseComplexProcessor
from processor.simple_processor.photo.photo_processor import Photo, PhotoProcessor


@dataclass
class PDF:
    photos: list[Photo] = field(default_factory=list)
    file_name: str = ""

    def render(self) -> str:
        header = f"PDF: {self.file_name}"
        body = "\n\n".join(
            f"Страница {i}: {p.result}" for i, p in enumerate(self.photos, start=1)
        )
        return f"{header}\n\n{body}" if body else header


class PDFProcessor(BaseComplexProcessor):
    """Renders a PDF into images and processes each page as a photo.

    Config format (``pdf_processor_config.yaml``)::

        photo_processor_config: photo_processor_config.yaml
        dpi: 200

    Paths are resolved relative to this config file's directory.
    """

    def __init__(self, photo_processor: PhotoProcessor, dpi: int = 200):
        self.photo_processor = photo_processor
        self.dpi = dpi

    @classmethod
    def build(cls, config_path: str | Path) -> "PDFProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "photo_processor_config" not in cfg:
            raise ConfigurationError("missing required key: photo_processor_config")

        config_dir = path.parent
        photo_processor = PhotoProcessor.build(
            config_dir / cfg["photo_processor_config"],
        )
        dpi = int(cfg.get("dpi", 200))
        return cls(photo_processor=photo_processor, dpi=dpi)

    def run(self, source: str | Path | bytes) -> PDF:
        images = self._render_pages(source)
        photos = [self.photo_processor.run(img) for img in images]
        file_name = "" if isinstance(source, bytes) else Path(source).name
        return PDF(photos=photos, file_name=file_name)

    def _render_pages(self, source: str | Path | bytes) -> list[bytes]:
        try:
            import fitz
        except ImportError as exc:
            raise ConfigurationError(
                "PyMuPDF (fitz) is required for PDFProcessor. "
                "Install it with `pip install pymupdf`.",
            ) from exc

        if isinstance(source, bytes):
            doc = fitz.open(stream=source, filetype="pdf")
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"pdf not found: {path}")
            doc = fitz.open(path)

        try:
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pages: list[bytes] = []
            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                pages.append(pix.tobytes("png"))
            return pages
        finally:
            doc.close()
