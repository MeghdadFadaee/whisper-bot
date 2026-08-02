"""Bale messenger bot that transcribes voice and audio messages."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from colorama import Style
from dotenv import load_dotenv
from whisper_runtime import load_model

BALE_BASE_URL = "https://tapi.bale.ai"
TEMP_DIR = Path("temp")
REQUEST_TIMEOUT = 30

logger = logging.getLogger(__name__)


class Messages:
    HELLO = "👋 سلام"
    HELP = "یک نجوا کافی است."
    FILE_RECEIVED = "✅ فایل با موفقیت دریافت شد، در حال پردازش..."
    TRANSCRIPTION_RESULT = "متن استخراج‌شده:\n"
    INVALID_FILE_TYPE = "فایل ارسالی باید صوتی یا ویدیویی باشد."
    REQUEST_AUDIO = "لطفاً یک فایل صوتی یا ویدیویی ارسال کنید."


class BaleWhisperBot:
    def __init__(self) -> None:
        token = os.getenv("BOT_TOKEN")
        if not token:
            raise RuntimeError("BOT_TOKEN is missing; copy .env.example to .env and set it")

        self.api_url = f"{BALE_BASE_URL}/bot{token}"
        self.download_url = f"{BALE_BASE_URL}/file/bot{token}"
        self.poll_timeout = int(os.getenv("BALE_POLL_TIMEOUT", "30"))
        self.language = os.getenv("WHISPER_LANGUAGE", "fa") or None
        self.session = requests.Session()
        self.model = None

    def api_request(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        response = self.session.request(
            kwargs.pop("http_method", "GET"),
            f"{self.api_url}/{operation}",
            timeout=kwargs.pop("timeout", REQUEST_TIMEOUT),
            **kwargs,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", True):
            raise RuntimeError(f"Bale API error from {operation}: {payload}")
        return payload

    def get_me(self) -> dict[str, Any]:
        return self.api_request("getMe").get("result", {})

    def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": self.poll_timeout}
        if offset is not None:
            params["offset"] = offset
        payload = self.api_request(
            "getUpdates",
            params=params,
            timeout=self.poll_timeout + 10,
        )
        return payload.get("result", [])

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = reply_to_message_id
        return self.api_request("sendMessage", http_method="POST", json=data)

    def download_file(self, file_id: str) -> bytes:
        file_info = self.api_request("getFile", params={"file_id": file_id})
        file_path = file_info["result"]["file_path"]
        response = self.session.get(
            f"{self.download_url}/{file_path}", timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.content

    def transcribe_audio(self, audio_data: bytes) -> str:
        TEMP_DIR.mkdir(exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=TEMP_DIR, suffix=".wav", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    str(temporary_path),
                ],
                input=audio_data,
                check=True,
            )
            if self.model is None:
                raise RuntimeError("Whisper model has not been loaded")
            result = self.model.transcribe(
                str(temporary_path), language=self.language
            )
            return result["text"].strip()
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def handle_audio(
        self,
        chat_id: int,
        message_id: int,
        file_id: str,
    ) -> None:
        self.send_message(chat_id, Messages.FILE_RECEIVED, message_id)
        audio_data = self.download_file(file_id)
        transcript = self.transcribe_audio(audio_data)
        self.send_message(
            chat_id,
            f"{Messages.TRANSCRIPTION_RESULT}{transcript}",
            message_id,
        )

    def process_message(self, message: dict[str, Any]) -> None:
        chat_id = message["chat"]["id"]
        message_id = message["message_id"]
        sender = message.get("from", {})
        username = sender.get("username") or sender.get("first_name") or "unknown"
        logger.info(
            ">%s%s (ID: %s)%s sent a message",
            Style.BRIGHT,
            username,
            sender.get("id", "unknown"),
            Style.RESET_ALL,
        )

        command = message.get("text", "").lower()
        if command in {"/start", "/hi", "/hello"}:
            name = " ".join(
                part
                for part in (sender.get("first_name"), sender.get("last_name"))
                if part
            )
            self.send_message(chat_id, f"{Messages.HELLO} {name}".rstrip())
        elif command == "/help":
            self.send_message(chat_id, Messages.HELP)
        elif command == "ping":
            self.send_message(chat_id, "pong")
        elif "voice" in message:
            self.handle_audio(chat_id, message_id, message["voice"]["file_id"])
        elif "document" in message:
            document = message["document"]
            mime_type = document.get("mime_type", "")
            if not mime_type.startswith(("audio/", "video/")):
                self.send_message(chat_id, Messages.INVALID_FILE_TYPE, message_id)
                return
            self.handle_audio(chat_id, message_id, document["file_id"])
        else:
            self.send_message(chat_id, Messages.REQUEST_AUDIO, message_id)

    def run(self) -> None:
        identity = self.get_me()
        logger.info(
            "Logged in as %s (ID: %s)",
            identity.get("username", "unknown"),
            identity.get("id", "unknown"),
        )
        self.model = load_model()

        offset: int | None = None
        while True:
            try:
                for update in self.get_updates(offset):
                    offset = update["update_id"] + 1
                    message = update.get("message")
                    if message:
                        self.process_message(message)
            except (
                requests.RequestException,
                RuntimeError,
                subprocess.SubprocessError,
                KeyError,
                ValueError,
            ):
                logger.exception("Bot polling failed; retrying in 3 seconds")
                time.sleep(3)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    BaleWhisperBot().run()


if __name__ == "__main__":
    main()
