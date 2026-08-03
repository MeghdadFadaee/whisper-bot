"""Transcribe every supported media file with safe resume support."""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from whisper_runtime import load_model

AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".amr",
    ".caf",
    ".flac",
    ".m4a",
    ".mka",
    ".mp3",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mts",
    ".ts",
    ".vob",
    ".webm",
    ".wmv",
}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


@contextmanager
def exclusive_batch_lock(output_directory: Path) -> Iterator[None]:
    """Prevent two batch processes from writing to the same output directory."""
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / "_batch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "unknown"
            raise RuntimeError(
                f"Another batch process is already using {output_directory} "
                f"(PID {owner})"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        yield


def discover_media(
    input_directory: Path, recursive: bool, follow_symlinks: bool
) -> list[Path]:
    if not recursive:
        return sorted(
            (
                path
                for path in input_directory.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )

    media_files: list[Path] = []
    visited_directories: set[tuple[int, int]] = set()

    def report_walk_error(error: OSError) -> None:
        logger.warning("Unable to scan a directory, continuing: %s", error)

    for root, directories, filenames in os.walk(
        input_directory,
        followlinks=follow_symlinks,
        onerror=report_walk_error,
    ):
        root_path = Path(root)
        try:
            stat = root_path.stat()
        except OSError as error:
            logger.warning("Unable to inspect directory, continuing: %s", error)
            directories.clear()
            continue

        directory_key = (stat.st_dev, stat.st_ino)
        if directory_key in visited_directories:
            directories.clear()
            continue
        visited_directories.add(directory_key)

        directories.sort(key=str.casefold)
        if follow_symlinks:
            directories[:] = [
                name
                for name in directories
                if _directory_key(root_path / name) not in visited_directories
            ]

        for filename in sorted(filenames, key=str.casefold):
            path = root_path / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                media_files.append(path)

    return media_files


def _directory_key(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def output_paths(
    source: Path, input_directory: Path, output_directory: Path
) -> tuple[Path, Path]:
    relative_path = source.relative_to(input_directory)
    transcript = output_directory / relative_path.with_suffix(
        relative_path.suffix + ".txt"
    )
    metadata = transcript.with_suffix(transcript.suffix + ".json")
    return transcript, metadata


def source_metadata(
    source: Path, model_name: str, language: str | None
) -> dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "model": model_name,
        "language": language,
    }


def is_completed(
    transcript: Path, metadata_path: Path, expected_metadata: dict[str, Any]
) -> bool:
    if not transcript.is_file() or not metadata_path.is_file():
        return False
    try:
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        actual_metadata.get(key) == value
        for key, value in expected_metadata.items()
    )


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def compact_error(error: Exception, limit: int = 4000) -> str:
    message = str(error).strip()
    if len(message) <= limit:
        return message
    return message[:limit] + "\n... error output truncated ..."


def validate_media(source: Path) -> None:
    """Confirm FFmpeg can find an audio stream before invoking Whisper."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        reason = result.stderr.strip().splitlines()[-1:] or ["invalid media file"]
        raise ValueError(reason[0])
    if not result.stdout.strip():
        raise ValueError("media file contains no audio stream")


def start_caffeinate(enabled: bool) -> subprocess.Popen[bytes] | None:
    if not enabled or sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(
        ["caffeinate", "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("temp"))
    parser.add_argument("--output", type=Path, default=Path("transcripts"))
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "large"))
    parser.add_argument("--device", default=os.getenv("WHISPER_DEVICE", "auto"))
    parser.add_argument("--language", default=os.getenv("WHISPER_LANGUAGE", "fa"))
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan subdirectories (default: enabled)",
    )
    parser.add_argument(
        "--follow-symlinks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan linked directories without following loops (default: enabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="transcribe files even when matching completed output exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list discovered media without loading Whisper or writing output",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="additional attempts after a failed transcription (default: 1)",
    )
    parser.add_argument(
        "--keep-awake",
        action=argparse.BooleanOptionalAction,
        default=sys.platform == "darwin",
        help="prevent macOS sleep while running (default: enabled on macOS)",
    )
    return parser.parse_args()


class BatchTranscriber:
    def __init__(self, args: argparse.Namespace) -> None:
        self.input_directory = args.input.expanduser().resolve()
        self.output_directory = args.output.expanduser().resolve()
        self.model_name = args.model
        self.device = args.device
        self.language = args.language or None
        self.recursive = args.recursive
        self.follow_symlinks = args.follow_symlinks
        self.overwrite = args.overwrite
        self.dry_run = args.dry_run
        self.retries = args.retries
        self.keep_awake = args.keep_awake
        self.state_path = self.output_directory / "_state.json"
        self.summary_path = self.output_directory / "_summary.json"
        self.errors_path = self.output_directory / "_errors.jsonl"
        self.started_at = utc_now()
        self.stats = {"succeeded": 0, "skipped": 0, "failed": 0}
        self.total = 0
        self.current_index = 0
        self.current_file: str | None = None

    def validate(self) -> None:
        if not self.input_directory.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")
        if self.retries < 0:
            raise ValueError("--retries cannot be negative")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg was not found; install it with: brew install ffmpeg")
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe was not found; install it with: brew install ffmpeg")
        try:
            self.output_directory.relative_to(self.input_directory)
        except ValueError:
            pass
        else:
            raise ValueError("The output directory cannot be inside the input directory")

    def write_state(
        self,
        status: str,
        total: int,
        current_index: int = 0,
        current_file: str | None = None,
    ) -> None:
        atomic_write_json(
            self.state_path,
            {
                "status": status,
                "pid": os.getpid(),
                "started_at": self.started_at,
                "updated_at": utc_now(),
                "input": str(self.input_directory),
                "output": str(self.output_directory),
                "model": self.model_name,
                "device": self.device,
                "language": self.language,
                "follow_symlinks": self.follow_symlinks,
                "total": total,
                "current_index": current_index,
                "current_file": current_file,
                **self.stats,
            },
        )

    def transcribe_with_retries(self, model: Any, source: Path) -> str:
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                result = model.transcribe(
                    str(source),
                    language=self.language,
                    verbose=False,
                )
                return result["text"].strip()
            except KeyboardInterrupt:
                raise
            except Exception as error:
                if attempt == attempts:
                    raise
                delay = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s; retrying in %d seconds",
                    attempt,
                    attempts,
                    source,
                    compact_error(error),
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def validate_with_retries(self, source: Path) -> None:
        attempts = self.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                validate_media(source)
                return
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                if attempt == attempts:
                    raise
                delay = min(2 ** (attempt - 1), 30)
                logger.warning(
                    "Media validation %d/%d failed for %s: %s; "
                    "retrying in %d seconds",
                    attempt,
                    attempts,
                    source,
                    compact_error(error),
                    delay,
                )
                time.sleep(delay)

    def record_failure(self, source: Path, error: Exception) -> None:
        self.stats["failed"] += 1
        append_json_line(
            self.errors_path,
            {
                "timestamp": utc_now(),
                "source": str(source),
                "error_type": type(error).__name__,
                "error": compact_error(error),
                "model": self.model_name,
                "language": self.language,
                "attempts": self.retries + 1,
            },
        )

    def has_matching_output(self, source: Path) -> bool:
        transcript_path, metadata_path = output_paths(
            source, self.input_directory, self.output_directory
        )
        try:
            metadata = source_metadata(source, self.model_name, self.language)
        except OSError:
            return False
        return is_completed(transcript_path, metadata_path, metadata)

    def process_file(self, model: Any, source: Path) -> str:
        transcript_path, metadata_path = output_paths(
            source, self.input_directory, self.output_directory
        )
        metadata = source_metadata(source, self.model_name, self.language)
        if not self.overwrite and is_completed(
            transcript_path, metadata_path, metadata
        ):
            self.stats["skipped"] += 1
            return "skipped"

        transcript = self.transcribe_with_retries(model, source)
        metadata.update(
            {
                "transcribed_at": utc_now(),
                "device": str(model.device),
                "transcript": str(transcript_path),
            }
        )
        atomic_write_text(transcript_path, transcript + "\n")
        atomic_write_json(metadata_path, metadata)
        self.stats["succeeded"] += 1
        return "completed"

    def run(self) -> int:
        self.validate()
        media_files = discover_media(
            self.input_directory, self.recursive, self.follow_symlinks
        )
        total = len(media_files)
        self.total = total
        logger.info("Found %d supported media file(s) in %s", total, self.input_directory)
        if self.dry_run:
            for source in media_files:
                logger.info("Would transcribe: %s", source.relative_to(self.input_directory))
            return 0

        self.write_state("starting", total)
        if not media_files:
            self.finish("completed", total)
            return 0

        self.write_state("running", total)
        model: Any | None = None

        for index, source in enumerate(media_files, start=1):
            relative_path = source.relative_to(self.input_directory)
            self.current_index = index
            self.current_file = str(relative_path)
            self.write_state("running", total, index, str(relative_path))
            logger.info("[%d/%d] %s", index, total, relative_path)
            if not self.overwrite and self.has_matching_output(source):
                self.stats["skipped"] += 1
                logger.info("[%d/%d] skipped: %s", index, total, relative_path)
                self.current_file = None
                self.write_state("running", total, index)
                continue
            try:
                self.validate_with_retries(source)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                logger.error("Invalid media, continuing: %s: %s", source, error)
                self.record_failure(source, error)
                self.current_file = None
                self.write_state("running", total, index)
                continue

            if model is None:
                try:
                    model = load_model(self.model_name, self.device)
                except Exception as error:
                    self.finish("failed_to_start", total)
                    raise RuntimeError(
                        f"Unable to load Whisper model: {compact_error(error)}"
                    ) from error

            try:
                outcome = self.process_file(model, source)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                logger.error(
                    "Failed permanently, continuing: %s: %s",
                    source,
                    compact_error(error),
                )
                self.record_failure(source, error)
                outcome = "failed"
            logger.info("[%d/%d] %s: %s", index, total, outcome, relative_path)
            self.current_file = None
            self.write_state("running", total, index)

        status = "completed_with_errors" if self.stats["failed"] else "completed"
        self.finish(status, total)
        return 1 if self.stats["failed"] else 0

    def finish(self, status: str, total: int) -> None:
        self.write_state(status, total, total)
        atomic_write_json(
            self.summary_path,
            {
                "status": status,
                "started_at": self.started_at,
                "finished_at": utc_now(),
                "input": str(self.input_directory),
                "output": str(self.output_directory),
                "model": self.model_name,
                "device": self.device,
                "language": self.language,
                "total": total,
                **self.stats,
            },
        )
        logger.info(
            "Finished: %d succeeded, %d skipped, %d failed",
            self.stats["succeeded"],
            self.stats["skipped"],
            self.stats["failed"],
        )


def raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> int:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)
    transcriber = BatchTranscriber(args)
    caffeinate_process = start_caffeinate(args.keep_awake and not args.dry_run)

    try:
        if args.dry_run:
            return transcriber.run()
        with exclusive_batch_lock(transcriber.output_directory):
            return transcriber.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted; completed files are safe and will be skipped on resume")
        if transcriber.state_path.exists():
            transcriber.write_state(
                "interrupted",
                transcriber.total,
                transcriber.current_index,
                transcriber.current_file,
            )
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        logger.error("Batch setup failed: %s", error)
        return 2
    finally:
        if caffeinate_process is not None:
            caffeinate_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
