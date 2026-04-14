from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from fast_ai.exceptions import ConfigurationError
from fast_ai.stt import STT
from fast_ai.stt.base import STTSegment
from processor.simple_processor.base_processor import BaseProcessor


@dataclass
class Audio:
    text: str
    language: str | None
    duration: float | None
    segments: list[STTSegment] = field(default_factory=list)
    result: str = ""


class AudioProcessor(BaseProcessor):
    """Processes an audio file through STT, then fills a template.

    Config format (``audio_processor_config.yaml``)::

        stt_config: stt_config.yaml
        template_path: templates/audio.txt

    The template file uses the ``$text`` placeholder
    (Python `string.Template` syntax).
    """

    def __init__(self, stt: STT, template: str):
        self.stt = stt
        self.template = template

    @classmethod
    def build(cls, config_path: str | Path) -> "AudioProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for key in ("stt_config", "template_path"):
            if key not in cfg:
                raise ConfigurationError(f"missing required key: {key}")

        config_dir = path.parent

        stt = STT.build(config_dir / cfg["stt_config"])

        template_file = config_dir / cfg["template_path"]
        if not template_file.exists():
            raise ConfigurationError(f"template file not found: {template_file}")

        template = template_file.read_text(encoding="utf-8")
        return cls(stt=stt, template=template)

    def run(self, source: str | Path) -> Audio:
        response = self.stt.transcribe(source)
        result = Template(self.template).safe_substitute(text=response.text)
        return Audio(
            text=response.text,
            language=response.language,
            duration=response.duration,
            segments=response.segments,
            result=result,
        )
