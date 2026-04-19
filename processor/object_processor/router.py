from pathlib import Path
from typing import Any

from fast_ai.exceptions import ConfigurationError
from processor.complex_processor.odg.odg_processor import ODGProcessor
from processor.complex_processor.odp.odp_processor import ODPProcessor
from processor.complex_processor.ods.ods_processor import ODSProcessor
from processor.complex_processor.odt.odt_processor import ODTProcessor
from processor.complex_processor.pdf.pdf_processor import PDFProcessor
from processor.complex_processor.powerpoint.powerpoint_processor import PowerPointProcessor
from processor.complex_processor.word.word_processor import WordProcessor
from processor.complex_processor.xlsx.xlsx_processor import XLSXProcessor
from processor.object_processor.base_processor import Attachment
from processor.simple_processor.audio.audio_processor import AudioProcessor
from processor.simple_processor.photo.photo_processor import PhotoProcessor


_KIND_REGISTRY: dict[str, Any] = {
    "photo": PhotoProcessor,
    "audio": AudioProcessor,
    "pdf": PDFProcessor,
    "word": WordProcessor,
    "powerpoint": PowerPointProcessor,
    "xlsx": XLSXProcessor,
    "odt": ODTProcessor,
    "odp": ODPProcessor,
    "odg": ODGProcessor,
    "ods": ODSProcessor,
}


class AttachmentRouter:
    """Routes raw attachments to the right processor by file extension.

    Built from a config block of the form::

        attachment_processors:
          photo: {extensions: [jpg, jpeg, png, webp], config: photo_processor_config.yaml}
          pdf:   {extensions: [pdf], config: pdf_processor_config.yaml}
          word:  {extensions: [doc, docx], config: word_processor_config.yaml}

    The kind name (``photo``, ``pdf``, ...) selects the processor class from
    a built-in registry; ``config`` is loaded with that class's ``build()``.
    """

    def __init__(self, ext_to_processor: dict[str, Any]):
        self.ext_to_processor = ext_to_processor

    @classmethod
    def build(cls, cfg: dict, config_dir: Path) -> "AttachmentRouter":
        ext_to_processor: dict[str, Any] = {}
        for kind, entry in cfg.items():
            if kind not in _KIND_REGISTRY:
                raise ConfigurationError(
                    f"unknown attachment kind: {kind!r}. "
                    f"Known kinds: {sorted(_KIND_REGISTRY)}",
                )
            if "extensions" not in entry or "config" not in entry:
                raise ConfigurationError(
                    f"attachment_processors.{kind} must contain "
                    "'extensions' and 'config'",
                )
            processor_cls = _KIND_REGISTRY[kind]
            processor = processor_cls.build(config_dir / entry["config"])
            for ext in entry["extensions"]:
                ext_to_processor[ext.lower().lstrip(".")] = processor
        return cls(ext_to_processor)

    def route(self, source, *, ext: str | None = None, name: str | None = None) -> Attachment:
        if ext is None:
            if isinstance(source, bytes):
                raise ConfigurationError(
                    "ext must be provided explicitly when routing raw bytes",
                )
            ext = Path(source).suffix
        ext = ext.lower().lstrip(".")
        if ext not in self.ext_to_processor:
            raise ConfigurationError(
                f"no processor configured for extension: {ext!r}",
            )
        processor = self.ext_to_processor[ext]
        result = processor.run(source)
        if name is None:
            name = (
                f"attachment.{ext}"
                if isinstance(source, bytes)
                else Path(source).name
            )
        return Attachment(source=name, kind=result)
