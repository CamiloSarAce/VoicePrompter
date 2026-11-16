"""Speech to raw text, using Whisper running locally on your machine."""
from __future__ import annotations

import numpy as np


class Recognizer:
    """Speech to raw text using faster-whisper (runs locally)."""

    def __init__(self, model: str, language: str, device: str, compute_type: str, initial_prompt: str):
        from faster_whisper import WhisperModel  # heavy import, so it is done lazily here

        # Downloads the model on first use (cached under ~/.cache/huggingface, outside the repo).
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language or None  # None = auto-detect the language
        self._prompt = initial_prompt or None  # None = no hint

    def transcribe(self, audio: np.ndarray) -> str:
        """Return the text of one utterance (16 kHz mono float32 audio)."""
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            initial_prompt=self._prompt,  # nudges Whisper toward your command vocabulary
            vad_filter=True,  # skip stretches with no speech
            beam_size=5,  # search width: higher is a bit more accurate but slower
        )
        # Whisper returns several segments per utterance; glue them into one line.
        return " ".join(s.text.strip() for s in segments).strip()
