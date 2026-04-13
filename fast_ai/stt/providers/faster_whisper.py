import json
import subprocess
import sys
from pathlib import Path

from fast_ai.exceptions import ProviderError
from fast_ai.logger import get_logger
from fast_ai.stt.base import STTProvider, STTResponse, STTSegment

log = get_logger("stt.faster_whisper")

_WORKER_MODULE = "fast_ai.stt._faster_whisper_worker"


class FasterWhisperProvider(STTProvider):
    """Speech-to-text provider using faster-whisper via subprocess.

    Each transcription spawns a new process so the model is unloaded
    from memory after every call.

    Args:
        model: Model size or path (e.g. "base", "large-v3", "/path/to/model").
        device: Device to run on: "auto", "cpu", "cuda".
        compute_type: Compute type: "default", "float16", "int8", etc.
        beam_size: Beam size for decoding.
        vad_filter: Enable voice activity detection filter.
        word_timestamps: Include word-level timestamps.
        temperature: Sampling temperature.
        initial_prompt: Initial prompt to condition the model.
        retries: How many times to retry on failure.
        timeout: Subprocess timeout in seconds.
    """

    def __init__(
        self,
        model: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        vad_filter: bool = False,
        word_timestamps: bool = False,
        temperature: float = 0.0,
        initial_prompt: str | None = None,
        retries: int = 1,
        timeout: float = 120.0,
    ):
        super().__init__(model=model, retries=retries, timeout=timeout)
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.word_timestamps = word_timestamps
        self.temperature = temperature
        self.initial_prompt = initial_prompt

    @property
    def provider_name(self) -> str:
        return f"faster_whisper({self.model})"

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> STTResponse:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        effective_timeout = timeout if timeout is not None else self.timeout

        cmd = [
            sys.executable, "-m", _WORKER_MODULE,
            str(audio_path),
            "--model", self.model,
            "--device", self.device,
            "--compute-type", self.compute_type,
            "--beam-size", str(self.beam_size),
            "--temperature", str(self.temperature),
        ]
        if language:
            cmd.extend(["--language", language])
        if self.vad_filter:
            cmd.append("--vad-filter")
        if self.word_timestamps:
            cmd.append("--word-timestamps")
        if self.initial_prompt:
            cmd.extend(["--initial-prompt", self.initial_prompt])

        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            log.debug(
                "attempt %d/%d  model=%s  audio=%s  timeout=%.1fs",
                attempt, self.retries, self.model, audio_path, effective_timeout,
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
                )

                if proc.returncode != 0:
                    stderr = proc.stderr.strip()
                    # Worker may still print JSON error to stdout
                    error_msg = stderr
                    try:
                        data = json.loads(proc.stdout)
                        if "error" in data:
                            error_msg = data["error"]
                    except (json.JSONDecodeError, KeyError):
                        pass
                    raise RuntimeError(error_msg or f"worker exited with code {proc.returncode}")

                data = json.loads(proc.stdout)
                if "error" in data:
                    raise RuntimeError(data["error"])

                segments = [
                    STTSegment(start=s["start"], end=s["end"], text=s["text"])
                    for s in data.get("segments", [])
                ]

                log.info("success on attempt %d/%d", attempt, self.retries)
                return STTResponse(
                    provider_name=self.provider_name,
                    text=data["text"],
                    language=data.get("language"),
                    language_probability=data.get("language_probability"),
                    duration=data.get("duration"),
                    segments=segments,
                )

            except subprocess.TimeoutExpired as exc:
                last_err = exc
                log.warning("attempt %d/%d timed out after %.1fs", attempt, self.retries, effective_timeout)
            except (json.JSONDecodeError, RuntimeError, KeyError) as exc:
                last_err = exc
                log.warning("attempt %d/%d failed: %s", attempt, self.retries, exc)

        raise ProviderError(
            self.provider_name,
            f"all {self.retries} attempts failed",
            cause=last_err,
        )
