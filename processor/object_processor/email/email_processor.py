from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from fast_ai.exceptions import ConfigurationError
from processor.object_processor.base_processor import Attachment, BaseObjectProcessor
from processor.object_processor.router import AttachmentRouter


@dataclass
class Email:
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    sender: str = ""
    recipient: str = ""
    subject: str = ""
    date: str = ""
    result: str = ""


class EmailProcessor(BaseObjectProcessor):
    """Processes an email: a body text plus a list of attachments.

    Config format (``email_processor_config.yaml``)::

        header_template_path: templates/email_header.txt
        attachment_processors:
          photo: {extensions: [jpg, jpeg, png, webp], config: photo_processor_config.yaml}
          pdf:   {extensions: [pdf], config: pdf_processor_config.yaml}
          word:  {extensions: [doc, docx], config: word_processor_config.yaml}

    Header template uses ``$sender``, ``$recipient``, ``$subject``, ``$date``.
    """

    def __init__(self, router: AttachmentRouter, header_template: str):
        self.router = router
        self.header_template = header_template

    @classmethod
    def build(cls, config_path: str | Path) -> "EmailProcessor":
        import yaml

        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for key in ("header_template_path", "attachment_processors"):
            if key not in cfg:
                raise ConfigurationError(f"missing required key: {key}")

        config_dir = path.parent

        template_file = config_dir / cfg["header_template_path"]
        if not template_file.exists():
            raise ConfigurationError(f"template file not found: {template_file}")
        header_template = template_file.read_text(encoding="utf-8")

        router = AttachmentRouter.build(cfg["attachment_processors"], config_dir)
        return cls(router=router, header_template=header_template)

    def run(
        self,
        text: str,
        attachments: list = None,
        *,
        sender: str = "",
        recipient: str = "",
        subject: str = "",
        date: str = "",
    ) -> Email:
        attachments = attachments or []
        processed = [self.router.route(a) for a in attachments]
        email = Email(
            text=text,
            attachments=processed,
            sender=sender,
            recipient=recipient,
            subject=subject,
            date=date,
        )
        email.result = self.render(email)
        return email

    def render(self, email: Email) -> str:
        header = Template(self.header_template).safe_substitute(
            sender=email.sender,
            recipient=email.recipient,
            subject=email.subject,
            date=email.date,
        )
        parts = [header.rstrip(), "", email.text.rstrip()]
        for i, att in enumerate(email.attachments, start=1):
            parts.append("")
            parts.append(f"Вложение {i}: {att.render()}")
        return "\n".join(parts)
