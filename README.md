# Whisper Bale Bot

Transcribe Persian voice messages, audio files, and videos from
[Bale](https://bale.ai/) with OpenAI Whisper. The project also includes a small
command-line utility for transcribing local media files.

## Requirements

- macOS on Apple Silicon (the checked-in setup targets this machine)
- Python 3.14
- FFmpeg
- A Bale bot token when running `bot.py`

## macOS setup

Install FFmpeg and create the virtual environment:

```bash
brew install ffmpeg
python3.14 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The runtime chooses an accelerator automatically: Apple Metal (`mps`) on this
Mac, CUDA on supported Linux systems, or CPU as a fallback.

## Transcribe a local file

```bash
python app.py recording.mp3
python app.py video.mp4
```

Useful options:

```bash
python app.py recording.mp3 --model small --language fa --device auto
```

The CLI shows a transcription progress bar and marks supported terminal tabs as
busy while it is working. Disable the in-terminal percentage bar for scripts:

```bash
python app.py recording.mp3 --no-progress
```

Whisper downloads model weights on first use. `large` needs roughly 10 GB of
memory; use `small` or `turbo` if the machine does not have enough memory.

## Run the Bale bot

Create the local configuration:

```bash
cp .env.example .env
```

Set `BOT_TOKEN` in `.env`, then run:

```bash
python bot.py
```

Configuration variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BOT_TOKEN` | required | Bale bot authentication token |
| `WHISPER_MODEL` | `large` | Whisper model name |
| `WHISPER_DEVICE` | `auto` | `auto`, `mps`, `cuda`, or `cpu` |
| `WHISPER_LANGUAGE` | `fa` | Input language; leave empty for detection |
| `BALE_POLL_TIMEOUT` | `30` | Long-poll timeout in seconds |

## Docker

Docker runs the bot on CPU and keeps downloaded model weights in a named volume:

```bash
./make.sh
./run.sh
```

## Notes

- FFmpeg is required by Whisper and extracts incoming audio or video tracks to
  16 kHz mono WAV.
- Temporary converted audio is stored under `temp/` and removed after each job.
- Python 3.14-compatible transitive dependencies are resolved by Whisper instead
  of being duplicated and pinned in `requirements.txt`.
