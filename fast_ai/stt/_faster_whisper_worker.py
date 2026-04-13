"""Standalone worker script for faster-whisper transcription.

Runs in a separate process so that the model is loaded on start
and fully unloaded from memory when the process exits.

Usage:
    python -m fast_ai.stt._faster_whisper_worker <audio_path> [options]

Outputs JSON to stdout with keys:
    text, segments, language, language_probability, duration
"""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper",
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument("--model", default="base", help="Model size or path")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    parser.add_argument("--compute-type", default="default", help="Compute type: default, float16, int8, etc.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--language", default=None, help="Language code (None for auto-detect)")
    parser.add_argument("--initial-prompt", default=None, help="Initial prompt to condition the model")
    parser.add_argument("--vad-filter", action="store_true", help="Enable VAD filter")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")

    args = parser.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            json.dumps({"error": "faster-whisper is not installed. Install with: pip install faster-whisper"}),
            file=sys.stdout,
        )
        sys.exit(1)

    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )

        segments_iter, info = model.transcribe(
            args.audio_path,
            beam_size=args.beam_size,
            language=args.language,
            initial_prompt=args.initial_prompt,
            vad_filter=args.vad_filter,
            word_timestamps=args.word_timestamps,
            temperature=args.temperature,
        )

        segments = []
        full_text_parts = []
        for segment in segments_iter:
            segments.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
            })
            full_text_parts.append(segment.text.strip())

        result = {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
        }

        print(json.dumps(result, ensure_ascii=False))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
