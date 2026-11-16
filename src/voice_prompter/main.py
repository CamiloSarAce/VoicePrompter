"""Entry point: wires hotkey -> microphone -> Whisper -> parser -> keyboard."""
from __future__ import annotations

import threading
from pathlib import Path

import keyboard

from .audio import utterances
from .config import load_settings
from .parser import Control, Parser, load_commands
from .recognizer import Recognizer
from .typer import Typer

# Project root (two folders above this file); config.toml and commands.toml live here.
ROOT = Path(__file__).resolve().parents[2]


def _beep(freq: int) -> None:
    """Short audible cue for ON/OFF so you don't need to look at the console."""
    try:
        import winsound  # Windows-only standard library module

        winsound.Beep(freq, 120)  # frequency in Hz, 120 ms long
    except Exception:  # not on Windows, or no sound device: silently skip
        pass


def main() -> None:
    cfg = load_settings(ROOT / "config.toml")  # your options
    parser = Parser(load_commands(ROOT / "commands.toml"), cfg.strip_auto_punctuation)
    typer = Typer(cfg.newline_key)

    print(f"Loading Whisper model '{cfg.model}' (first run downloads it)...")
    recognizer = Recognizer(cfg.model, cfg.language, cfg.device, cfg.compute_type, cfg.initial_prompt)

    active = threading.Event()  # set = listening is ON
    stop = threading.Event()  # set = shut the program down

    def toggle() -> None:
        """Hotkey handler: flip listening on/off."""
        if active.is_set():
            active.clear()
            _beep(600)  # low tone = off
            print("[listening OFF]")
        else:
            active.set()
            _beep(1000)  # high tone = on
            print("[listening ON]")

    keyboard.add_hotkey(cfg.hotkey, toggle)  # register the global hotkey from config.toml
    print(f"Ready. Press {cfg.hotkey} to toggle listening. Say 'stop listening' to quit.")

    # Each loop pass handles one spoken utterance.
    for audio in utterances(active, stop, cfg.energy_threshold, cfg.silence_seconds):
        text = recognizer.transcribe(audio)  # audio -> raw words
        if not text:  # Whisper heard nothing usable
            continue
        actions = parser.process(text)  # raw words -> typing actions
        print(f"heard: {text!r}")  # shown in the console to help debugging
        typer.run(actions)  # perform the keystrokes
        for action in actions:  # react to voice control commands
            if isinstance(action, Control):
                print(f"[{action.name}]")
                if action.name == "stop":
                    stop.set()  # ends the loop (utterances() sees it and returns)


if __name__ == "__main__":
    main()
