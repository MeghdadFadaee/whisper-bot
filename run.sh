#!/usr/bin/env bash
set -euo pipefail

docker run --rm --env-file .env -v whisper-models:/root/.cache/whisper whisper-bot
