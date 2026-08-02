"""Shared Whisper model and device configuration."""

from __future__ import annotations

import os

import torch
import whisper


def select_device(requested: str | None = None) -> str:
    """Resolve ``auto`` to the best accelerator available on this machine."""
    device = (requested or os.getenv("WHISPER_DEVICE", "auto")).lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but Apple Metal acceleration is unavailable")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("WHISPER_DEVICE must be one of: auto, cpu, cuda, mps")
    return device


def load_model(model_name: str | None = None, device: str | None = None):
    """Load the configured Whisper model on the selected device."""
    name = model_name or os.getenv("WHISPER_MODEL", "large")
    selected_device = select_device(device)
    print(f"Loading Whisper model {name!r} on {selected_device}...")
    return whisper.load_model(name, device=selected_device)
