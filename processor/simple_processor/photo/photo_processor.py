from dataclasses import dataclass
from pathlib import Path
from string import Template

from fast_ai.describer import ImageDescriber
from fast_ai.ocr import OCR
from fast_ai.exceptions import ConfigurationError
from processor.simple_processor.base_processor import BaseProcessor


@dataclass
class Photo:
    ocr: str
    description: str
    result: str

    def render(self) -> str:
        return self.result


class PhotoProcessor(BaseProcessor):
    """Processes a photo through OCR and ImageDescriber, then fills a template.

    Config format (``photo_processor_config.yaml``)::

        ocr_config: ocr_config.yaml
        describer_config: describer_config.yaml
        template_path: templates/photo.txt

    The template file uses ``$ocr`` and ``$description`` placeholders
    (Python `string.Template` syntax).
    """

    def __init__(self, ocr: OCR, describer: ImageDescriber, template: str):
        self.ocr = ocr
        self.describer = describer
        self.template = template

    @classmethod
    def build(cls, config_path: str | Path) -> "PhotoProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for key in ("ocr_config", "describer_config", "template_path"):
            if key not in cfg:
                raise ConfigurationError(f"missing required key: {key}")

        config_dir = path.parent

        ocr = OCR.build(config_dir / cfg["ocr_config"])
        describer = ImageDescriber.build(config_dir / cfg["describer_config"])

        template_file = config_dir / cfg["template_path"]
        if not template_file.exists():
            raise ConfigurationError(f"template file not found: {template_file}")

        template = template_file.read_text(encoding="utf-8")
        return cls(ocr=ocr, describer=describer, template=template)

    def run(self, image: str | Path | bytes) -> Photo:
        ocr_text = self.ocr.recognize(image)
        description_text = self.describer.describe(image)
        result = Template(self.template).safe_substitute(
            ocr=ocr_text,
            description=description_text,
        )
        return Photo(ocr=ocr_text, description=description_text, result=result)
