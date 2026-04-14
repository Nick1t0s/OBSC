import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fast_ai.exceptions import ConfigurationError
from processor.complex_processor.base_processor import BaseComplexProcessor
from processor.complex_processor.pdf.pdf_processor import PDF, PDFProcessor
from processor.simple_processor.photo.photo_processor import Photo


@dataclass
class ODP:
    photos: list[Photo] = field(default_factory=list)
    file_name: str = ""

    def render(self) -> str:
        header = f"ODP: {self.file_name}"
        body = "\n\n".join(
            f"Слайд {i}: {p.result}" for i, p in enumerate(self.photos, start=1)
        )
        return f"{header}\n\n{body}" if body else header


class ODPProcessor(BaseComplexProcessor):
    """Converts an OpenDocument Presentation (.odp) to PDF and processes it as a PDF.

    Conversion is performed by LibreOffice (``soffice --headless --convert-to pdf``),
    so LibreOffice must be installed and reachable on ``PATH`` (or the binary
    path provided via the ``soffice`` config key).

    Config format (``odp_processor_config.yaml``)::

        pdf_processor_config: pdf_processor_config.yaml
        soffice: soffice

    Paths are resolved relative to this config file's directory.
    """

    def __init__(self, pdf_processor: PDFProcessor, soffice: str = "soffice"):
        self.pdf_processor = pdf_processor
        self.soffice = soffice

    @classmethod
    def build(cls, config_path: str | Path) -> "ODPProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "pdf_processor_config" not in cfg:
            raise ConfigurationError("missing required key: pdf_processor_config")

        config_dir = path.parent
        pdf_processor = PDFProcessor.build(
            config_dir / cfg["pdf_processor_config"],
        )
        soffice = str(cfg.get("soffice", "soffice"))
        return cls(pdf_processor=pdf_processor, soffice=soffice)

    def run(self, source: str | Path | bytes) -> ODP:
        pdf_bytes = self._convert_to_pdf(source)
        pdf: PDF = self.pdf_processor.run(pdf_bytes)
        file_name = "" if isinstance(source, bytes) else Path(source).name
        return ODP(photos=pdf.photos, file_name=file_name)

    def _convert_to_pdf(self, source: str | Path | bytes) -> bytes:
        if shutil.which(self.soffice) is None and not Path(self.soffice).exists():
            raise ConfigurationError(
                f"LibreOffice binary not found: {self.soffice}. "
                "Install LibreOffice or set `soffice` in the config.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            if isinstance(source, bytes):
                src_path = tmp_dir / "input.odp"
                src_path.write_bytes(source)
            else:
                src_path = Path(source)
                if not src_path.exists():
                    raise FileNotFoundError(f"odp document not found: {src_path}")

            out_dir = tmp_dir / "out"
            out_dir.mkdir()

            result = subprocess.run(
                [
                    self.soffice,
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", str(out_dir),
                    str(src_path),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ConfigurationError(
                    f"soffice conversion failed (code {result.returncode}): "
                    f"{result.stderr.strip() or result.stdout.strip()}",
                )

            pdfs = list(out_dir.glob("*.pdf"))
            if not pdfs:
                raise ConfigurationError(
                    "soffice produced no PDF output for the presentation",
                )
            return pdfs[0].read_bytes()
