"""Send parser actions to whichever window has focus."""
from __future__ import annotations

import re
from collections.abc import Iterable

import keyboard  # simulates real keystrokes and watches global hotkeys (Windows)

from .parser import Action, Backspace, Type

# Splits text at newlines and tabs but keeps them, so each can be sent as a real key press.
_SPLIT = re.compile(r"(\n|\t)")


class Typer:
    def __init__(self, newline_key: str = "shift+enter"):
        self.newline_key = newline_key  # what "new line" presses (configurable in config.toml)

    def run(self, actions: Iterable[Action]) -> None:
        """Perform each action in order."""
        for action in actions:
            if isinstance(action, Type):
                self._type(action.text)
            elif isinstance(action, Backspace):
                for _ in range(action.count):  # one key press per character to undo
                    keyboard.send("backspace")
            # Control actions are handled in main.py, not here.

    def _type(self, text: str) -> None:
        """Type text, translating newline/tab characters into key presses."""
        for part in _SPLIT.split(text):
            if part == "\n":
                keyboard.send(self.newline_key)  # Shift+Enter by default, so it doesn't submit
            elif part == "\t":
                keyboard.send("tab")
            elif part:  # ordinary characters (including unicode) are typed directly
                keyboard.write(part)
