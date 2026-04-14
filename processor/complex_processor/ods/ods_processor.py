import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fast_ai.exceptions import ConfigurationError
from processor.complex_processor.base_processor import BaseComplexProcessor
from processor.complex_processor.xlsx.xlsx_processor import XLSX, Sheet, XLSXProcessor


@dataclass
class ODS:
    sheets: list[Sheet] = field(default_factory=list)
    file_name: str = ""

    @property
    def result(self) -> str:
        return "\n\n".join(f"{s.name}:{s.csv}" for s in self.sheets)

    def render(self) -> str:
        header = f"ODS: {self.file_name}"
        body = "\n\n".join(f"{s.name}: {s.csv}" for s in self.sheets)
        return f"{header}\n{body}" if body else header


class ODSProcessor(BaseComplexProcessor):
    """Converts an OpenDocument Spreadsheet (.ods) to XLSX and processes it as XLSX.

    Conversion is performed by LibreOffice (``soffice --headless --convert-to xlsx``),
    so LibreOffice must be installed and reachable on ``PATH`` (or the binary
    path provided via the ``soffice`` config key). The resulting workbook is then
    delegated to an internal :class:`XLSXProcessor`.

    Config format (``ods_processor_config.yaml``)::

        xlsx_processor_config: xlsx_processor_config.yaml
        soffice: soffice

    ``xlsx_processor_config`` is optional; if omitted, a default-configured
    :class:`XLSXProcessor` is used. Paths are resolved relative to this config
    file's directory.
    """

    def __init__(self, xlsx_processor: XLSXProcessor, soffice: str = "soffice"):
        self.xlsx_processor = xlsx_processor
        self.soffice = soffice

    @classmethod
    def build(cls, config_path: str | Path | None = None) -> "ODSProcessor":
        if config_path is None:
            return cls(xlsx_processor=XLSXProcessor())

        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        config_dir = path.parent
        xlsx_cfg = cfg.get("xlsx_processor_config")
        if xlsx_cfg:
            xlsx_processor = XLSXProcessor.build(config_dir / xlsx_cfg)
        else:
            xlsx_processor = XLSXProcessor()
        soffice = str(cfg.get("soffice", "soffice"))
        return cls(xlsx_processor=xlsx_processor, soffice=soffice)

    def run(self, source: str | Path | bytes) -> ODS:
        xlsx_bytes = self._convert_to_xlsx(source)
        xlsx: XLSX = self.xlsx_processor.run(xlsx_bytes)
        file_name = "" if isinstance(source, bytes) else Path(source).name
        return ODS(sheets=xlsx.sheets, file_name=file_name)

    def _convert_to_xlsx(self, source: str | Path | bytes) -> bytes:
        if shutil.which(self.soffice) is None and not Path(self.soffice).exists():
            raise ConfigurationError(
                f"LibreOffice binary not found: {self.soffice}. "
                "Install LibreOffice or set `soffice` in the config.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            if isinstance(source, bytes):
                src_path = tmp_dir / "input.ods"
                src_path.write_bytes(source)
            else:
                src_path = Path(source)
                if not src_path.exists():
                    raise FileNotFoundError(f"ods document not found: {src_path}")

            out_dir = tmp_dir / "out"
            out_dir.mkdir()

            result = subprocess.run(
                [
                    self.soffice,
                    "--headless",
                    "--convert-to", "xlsx",
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

            xlsxs = list(out_dir.glob("*.xlsx"))
            if not xlsxs:
                raise ConfigurationError(
                    "soffice produced no XLSX output for the spreadsheet",
                )
            return xlsxs[0].read_bytes()
