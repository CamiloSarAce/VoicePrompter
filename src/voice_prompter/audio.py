"""Microphone capture with simple energy-based utterance detection."""
from __future__ import annotations

import queue  # thread-safe hand-off between the audio callback and our loop
import threading
from collections import deque  # fixed-length buffer for the pre-roll
from collections.abc import Iterator

import numpy as np  # audio arrays and loudness maths
import sounddevice as sd  # microphone access

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio
BLOCK_S = 0.05  # we examine the audio in 50 ms slices
PRE_ROLL_BLOCKS = 6  # keep ~0.3 s before speech starts so the first syllable isn't clipped
MIN_SPEECH_S = 0.25  # ignore blips shorter than this (clicks, coughs)
MAX_UTTERANCE_S = 30.0  # force a cut if someone talks without pausing this long


def utterances(
    active: threading.Event,
    stop: threading.Event,
    threshold: float = 0.01,
    silence_s: float = 0.8,
) -> Iterator[np.ndarray]:
    """Yield mono float32 16 kHz arrays, one per spoken utterance.

    Audio is only collected while `active` is set. Ends when `stop` is set.
    """
    blocks: queue.Queue[np.ndarray] = queue.Queue()  # raw mic slices waiting to be examined

    def callback(indata, frames, time, status):
        # Runs on the audio thread for every slice: copy it and hand it to the main loop.
        blocks.put(indata[:, 0].copy())

    pre_roll: deque[np.ndarray] = deque(maxlen=PRE_ROLL_BLOCKS)  # recent quiet slices
    speech: list[np.ndarray] = []  # slices of the utterance being recorded
    speaking = False  # are we currently inside an utterance?
    quiet = 0  # consecutive quiet slices since the last loud one

    def finish() -> np.ndarray | None:
        """Close the current utterance and return its audio, or None if it was too short."""
        nonlocal speech, speaking, quiet
        audio = np.concatenate(speech) if speech else None
        speech, speaking, quiet = [], False, 0  # reset for the next utterance
        pre_roll.clear()
        # The trailing silence is included in the clip, so the minimum length allows for it.
        if audio is not None and len(audio) / SAMPLE_RATE >= MIN_SPEECH_S + silence_s:
            return audio
        return None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,  # mono
        dtype="float32",
        blocksize=int(SAMPLE_RATE * BLOCK_S),  # samples per slice
        callback=callback,
    ):
        while not stop.is_set():
            try:
                block = blocks.get(timeout=0.2)  # wake periodically to re-check `stop`
            except queue.Empty:
                continue
            if not active.is_set():  # listening is toggled off: throw everything away
                speech, speaking, quiet = [], False, 0
                pre_roll.clear()
                continue

            # RMS (average loudness) of this slice compared against the threshold.
            loud = float(np.sqrt(np.mean(block**2))) > threshold
            if not speaking:
                if loud:  # speech just started: include the quiet lead-in we saved
                    speaking = True
                    speech = list(pre_roll) + [block]
                else:  # still silence: keep a rolling lead-in
                    pre_roll.append(block)
                continue

            speech.append(block)  # inside an utterance: keep everything, even short pauses
            quiet = 0 if loud else quiet + 1
            too_long = len(speech) * BLOCK_S >= MAX_UTTERANCE_S
            if quiet * BLOCK_S >= silence_s or too_long:  # long enough pause (or hit the cap)
                audio = finish()
                if audio is not None:
                    yield audio  # hand the finished utterance to the caller
