# Voice Prompter: design notes

Local Python app. Toggle hotkey starts/stops listening; speech is transcribed by
faster-whisper, run through `parser.py`, and typed into the focused window.

## Pipeline

```
mic -> audio.py (utterance detection) -> recognizer.py (Whisper)
    -> parser.py (commands -> actions) -> typer.py (keystrokes)
```

`parser.py` is pure text-in/actions-out and is the only part with tests
(`python -m unittest discover -s tests`). Commands live in `commands.toml`,
settings in `config.toml`; neither needs code changes to edit.

## Decisions

- **Whisper punctuation is stripped** (`strip_auto_punctuation`), so only spoken
  commands insert punctuation. Capitalisation is kept as Whisper/you produce it.
- **Tag name = the next single word**, case as spoken. Multi-word names chain with
  "underscore" / "hyphen" / "dash": "user underscore name" -> `<user_name>`.
- **"new line" sends Shift+Enter**, not Enter, so it doesn't submit chat boxes.
- **"literal X"** types a command word as plain text.
- Words like "period", "dot", "tab", "dash" are always commands.
  "file dot txt" gives `file. txt`: use "literal dot" or a different phrase.

## Not verified yet

Unit tests cover the parser only. Microphone capture, Whisper, hotkey and typing
have not been run. Expect to tune `energy_threshold` and `silence_seconds`.

## Open questions

- Final hotkey (placeholder: ctrl+alt+v).
- Should "period" capitalise the next word?
- Does `keyboard` need admin rights to type into elevated windows? (Yes for those.)
