"""Transcribe a local audio file with Whisper."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from whisper_runtime import load_model

OSC_PROGRESS = "\033]9;4;{state};{value}\033\\"


@contextmanager
def terminal_activity() -> Iterator[None]:
    """Show indeterminate progress in terminals that support OSC 9;4."""
    enabled = sys.stderr.isatty() and os.getenv("TERM") != "dumb"
    if enabled:
        sys.stderr.write(OSC_PROGRESS.format(state=3, value=0))
        sys.stderr.flush()
    try:
        yield
    finally:
        if enabled:
            sys.stderr.write(OSC_PROGRESS.format(state=0, value=0))
            sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", default="output.mp3", type=Path)
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "large"))
    parser.add_argument("--device", default=os.getenv("WHISPER_DEVICE", "auto"))
    parser.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE", "fa"))
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show transcription progress (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")

    with terminal_activity():
        model = load_model(args.model, args.device)
        result = model.transcribe(
            str(args.audio),
            language=args.language,
            verbose=False if args.progress else None,
        )
    print(result["text"].strip())


if __name__ == "__main__":
    main()
