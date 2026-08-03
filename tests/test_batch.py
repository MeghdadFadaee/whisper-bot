from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from batch import BatchTranscriber, discover_media


class FakeModel:
    device = "test-device"

    def __init__(self, failing_names: set[str] | None = None) -> None:
        self.failing_names = failing_names or set()
        self.calls: list[str] = []

    def transcribe(self, source: str, **_kwargs: object) -> dict[str, str]:
        name = Path(source).name
        self.calls.append(name)
        if name in self.failing_names:
            raise RuntimeError(f"cannot decode {name}")
        return {"text": f"transcript for {name}"}


class BatchTranscriberTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_directory = self.root / "input"
        self.output_directory = self.root / "output"
        (self.input_directory / "nested").mkdir(parents=True)
        (self.input_directory / "nested" / "good.mp3").write_bytes(b"good")
        (self.input_directory / "bad.mp4").write_bytes(b"bad")
        (self.input_directory / "ignored.txt").write_text(
            "not media", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def arguments(self) -> argparse.Namespace:
        return argparse.Namespace(
            input=self.input_directory,
            output=self.output_directory,
            model="tiny",
            device="cpu",
            language="fa",
            recursive=True,
            follow_symlinks=True,
            overwrite=False,
            dry_run=False,
            retries=1,
            keep_awake=False,
        )

    @patch("batch.shutil.which", return_value="/usr/bin/ffmpeg")
    @patch("batch.time.sleep")
    def test_failures_do_not_stop_batch_and_successes_resume(
        self, _sleep: object, _which: object
    ) -> None:
        first_model = FakeModel({"bad.mp4"})
        with patch("batch.validate_media"), patch(
            "batch.load_model", return_value=first_model
        ):
            exit_code = BatchTranscriber(self.arguments()).run()

        self.assertEqual(exit_code, 1)
        self.assertEqual(first_model.calls.count("bad.mp4"), 2)
        self.assertEqual(first_model.calls.count("good.mp3"), 1)
        transcript = self.output_directory / "nested" / "good.mp3.txt"
        self.assertEqual(
            transcript.read_text(encoding="utf-8"),
            "transcript for good.mp3\n",
        )
        self.assertFalse((self.output_directory / "bad.mp4.txt").exists())

        summary = json.loads(
            (self.output_directory / "_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["failed"], 1)

        second_model = FakeModel({"bad.mp4"})
        with patch("batch.validate_media"), patch(
            "batch.load_model", return_value=second_model
        ):
            second_exit_code = BatchTranscriber(self.arguments()).run()

        self.assertEqual(second_exit_code, 1)
        self.assertNotIn("good.mp3", second_model.calls)
        second_summary = json.loads(
            (self.output_directory / "_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(second_summary["skipped"], 1)
        self.assertEqual(second_summary["failed"], 1)

    @patch("batch.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_all_matching_outputs_skip_model_loading(self, _which: object) -> None:
        model = FakeModel()
        with patch("batch.validate_media"), patch(
            "batch.load_model", return_value=model
        ):
            self.assertEqual(BatchTranscriber(self.arguments()).run(), 0)

        with patch("batch.validate_media"), patch(
            "batch.load_model", side_effect=AssertionError("must not load")
        ):
            transcriber = BatchTranscriber(self.arguments())
            self.assertEqual(transcriber.run(), 0)

        self.assertEqual(transcriber.stats["skipped"], 2)

    def test_recursive_discovery_follows_symlinks_without_loops(self) -> None:
        external = self.root / "external"
        external.mkdir()
        linked_video = external / "linked.mp4"
        linked_video.write_bytes(b"video")
        (external / "loop").symlink_to(self.input_directory, target_is_directory=True)
        (self.input_directory / "downloads").symlink_to(
            external, target_is_directory=True
        )

        files = discover_media(
            self.input_directory,
            recursive=True,
            follow_symlinks=True,
        )
        relative_files = {
            str(path.relative_to(self.input_directory)) for path in files
        }

        self.assertEqual(
            relative_files,
            {"bad.mp4", "nested/good.mp3", "downloads/linked.mp4"},
        )


if __name__ == "__main__":
    unittest.main()
