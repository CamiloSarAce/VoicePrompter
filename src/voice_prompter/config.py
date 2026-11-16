"""Loads config.toml into a typed Settings object."""
from __future__ import annotations

import tomllib  # TOML reader built into Python 3.11+
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    """Every option in config.toml. The defaults here apply if a line is missing from the file."""

    hotkey: str = "ctrl+alt+v"  # key combo that toggles listening on/off
    newline_key: str = "shift+enter"  # key pressed for "new line" (plain Enter would send chat messages)
    model: str = "small"  # Whisper model size: bigger = more accurate but slower
    language: str = "en"  # spoken language code; "" lets Whisper auto-detect
    device: str = "auto"  # "auto", "cpu" or "cuda" (NVIDIA GPU)
    compute_type: str = "int8"  # numeric precision: "int8" for CPU, "float16" for GPU
    initial_prompt: str = ""  # hint text that nudges Whisper toward your command words
    strip_auto_punctuation: bool = True  # drop punctuation Whisper adds, so only spoken commands add it
    energy_threshold: float = 0.01  # loudness that counts as speech; raise it in noisy rooms
    silence_seconds: float = 0.8  # silence length that ends an utterance


def load_settings(path: str | Path) -> Settings:
    """Read config.toml and return Settings, rejecting unknown option names (catches typos)."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(Settings)}  # the option names we actually support
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
    return Settings(**data)  # missing options keep the defaults above
