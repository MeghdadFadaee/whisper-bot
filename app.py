"""Transcribe a local audio file with Whisper."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from whisper_runtime import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", default="output.mp3", type=Path)
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "large"))
    parser.add_argument("--device", default=os.getenv("WHISPER_DEVICE", "auto"))
    parser.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE", "fa"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")

    model = load_model(args.model, args.device)
    result = model.transcribe(str(args.audio), language=args.language)
    print(result["text"].strip())


if __name__ == "__main__":
    main()
