#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT_DIR/venv/bin/python"
PID_FILE="$ROOT_DIR/temp/batch.pid"
LOG_FILE="$ROOT_DIR/temp/batch.log"

cd "$ROOT_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual environment not found. Run the setup steps in README.md first."
    exit 1
fi

is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local batch_pid process_command
    batch_pid="$(<"$PID_FILE")"
    [[ "$batch_pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$batch_pid" 2>/dev/null || return 1
    process_command="$(ps -p "$batch_pid" -o command= 2>/dev/null || true)"
    [[ "$process_command" == *"$ROOT_DIR/batch.py"* ]]
}

show_status() {
    if is_running; then
        echo "Batch transcriber is running (PID $(<"$PID_FILE"))."
    else
        echo "Batch transcriber is not running."
    fi
    if [[ -f "$LOG_FILE" ]]; then
        echo
        tail -n 8 "$LOG_FILE"
    fi
}

command="${1:-status}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command" in
    start)
        if is_running; then
            echo "Batch transcriber is already running (PID $(<"$PID_FILE"))."
            exit 1
        fi
        mkdir -p "$ROOT_DIR/temp"
        : >"$LOG_FILE"
        nohup "$PYTHON" -u "$ROOT_DIR/batch.py" "$@" >>"$LOG_FILE" 2>&1 &
        batch_pid=$!
        echo "$batch_pid" >"$PID_FILE"
        sleep 1
        if ! kill -0 "$batch_pid" 2>/dev/null; then
            set +e
            wait "$batch_pid"
            exit_code=$?
            set -e
            echo "Batch transcriber finished quickly (exit code $exit_code)."
            echo "Recent log output:"
            tail -n 20 "$LOG_FILE"
            exit "$exit_code"
        fi
        echo "Started batch transcriber (PID $batch_pid)."
        echo "Log: $LOG_FILE"
        ;;
    status)
        show_status
        ;;
    logs)
        touch "$LOG_FILE"
        tail -f "$LOG_FILE"
        ;;
    stop)
        if ! is_running; then
            echo "Batch transcriber is not running."
            exit 0
        fi
        batch_pid="$(<"$PID_FILE")"
        kill "$batch_pid"
        echo "Asked batch transcriber (PID $batch_pid) to stop safely."
        echo "Completed transcripts will be reused next time."
        ;;
    *)
        echo "Usage: ./batchctl.sh {start|status|logs|stop} [batch.py options]"
        exit 2
        ;;
esac
