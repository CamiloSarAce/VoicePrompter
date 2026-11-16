"""Turn raw speech-to-text output into typing actions.

Pure logic, no audio or keyboard dependencies, so it is fully unit-testable.
"""
from __future__ import annotations

import re  # regular expressions: used to strip punctuation and find the last word
import tomllib  # reads commands.toml (built into Python 3.11+)
from dataclasses import dataclass  # small classes that just hold data
from pathlib import Path  # file paths that work on Windows and elsewhere

# The five ways a symbol can attach to the words around it (see commands.toml).
SPACE_MODES = {"standalone", "closing", "opening", "joining", "raw"}
# Modes that must NOT get a space before them: "hello" + "," -> "hello,"
NO_SPACE_BEFORE = {"closing", "joining", "raw"}
# Modes after which the NEXT item must not get a leading space: "(" + "note" -> "(note"
GLUE_AFTER = {"opening", "joining", "raw"}
# Words that chain a multi-word XML tag name: "user underscore name" -> user_name
NAME_JOINERS = {"underscore": "_", "hyphen": "-", "dash": "-"}
# Matches punctuation at the very start or end of a word (Whisper adds these itself)
_EDGE = re.compile(r"^\W+|\W+$")
# Matches the last word plus the whitespace around it (used by "delete last word")
_LAST_WORD = re.compile(r"\s*\S+\s*$")


@dataclass(frozen=True)
class Type:
    """Action: type this text into the focused window."""

    text: str


@dataclass(frozen=True)
class Backspace:
    """Action: press Backspace this many times (used to undo earlier typing)."""

    count: int


@dataclass(frozen=True)
class Control:
    """Action: a program-control event for main.py ("pause", "resume", "stop")."""

    name: str


# Anything the parser can hand back to the caller.
Action = Type | Backspace | Control


@dataclass(frozen=True)
class Entry:
    """One command definition loaded from commands.toml."""

    type: str  # symbol | xml | edit | modifier | literal | control
    text: str = ""  # the characters to type (symbols only)
    space: str = "standalone"  # how it attaches to neighbouring words
    action: str = ""  # sub-kind for xml/edit/modifier/control commands


def load_commands(path: str | Path) -> dict[tuple[str, ...], Entry]:
    """Read commands.toml into a lookup table: spoken phrase (as a tuple of words) -> Entry."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    commands: dict[tuple[str, ...], Entry] = {}
    for raw in data.get("command", []):  # each [[command]] block in the file
        entry = Entry(
            type=raw["type"],  # required: a missing "type" raises a KeyError on purpose
            text=raw.get("text", ""),  # optional fields fall back to defaults
            space=raw.get("space", "standalone"),
            action=raw.get("action", ""),
        )
        if entry.space not in SPACE_MODES:  # catch typos like space = "closng"
            raise ValueError(f"Unknown space mode {entry.space!r} in {raw['say']}")
        for phrase in raw["say"]:  # one entry can have several spoken phrases
            key = tuple(phrase.lower().split())  # "new line" -> ("new", "line")
            if key in commands:  # two commands sharing a phrase would be ambiguous
                raise ValueError(f"Duplicate command phrase: {phrase!r}")
            commands[key] = entry
    return commands


def _glue_from(text: str) -> bool:
    """Guess from the last typed character whether the next item should skip its leading space."""
    return text[-1:] in ("\n", "\t", "(", "[", "/", "-")


def _trim_last_word(text: str) -> tuple[str, int]:
    """Remove the last word (and its surrounding space). Returns (shorter text, characters removed)."""
    m = _LAST_WORD.search(text)
    if not m:  # nothing to remove
        return text, 0
    return text[: m.start()], len(m.group())


@dataclass
class _Hist:
    """One chunk of text we typed earlier, kept so it can be undone."""

    text: str
    glue_after: bool  # whether the next item after this chunk gets no leading space


class Parser:
    """Stateful: remembers what it typed so "scratch that" and "delete last word" work."""

    def __init__(self, commands, strip_auto_punctuation: bool = True, history_limit: int = 50):
        self.commands = commands  # phrase -> Entry table from load_commands()
        self.strip = strip_auto_punctuation  # drop punctuation Whisper adds on its own
        self.paused = False  # True after "pause listening": everything but resume/stop is ignored
        self._max = max(len(k) for k in commands)  # longest phrase, in words
        self._limit = history_limit  # how many typed chunks we remember
        self._history: list[_Hist] = []  # chunks typed so far, oldest first
        self._glue = False  # True: the next item must not get a leading space
        self._started = False  # has anything been typed yet? (no leading space on the first word)
        self._reset_run()

    # ---------- public ----------

    def process(self, raw: str) -> list[Action]:
        """Convert one transcribed utterance into a list of actions."""
        self._reset_run()
        toks = self._tokenize(raw)
        mod: str | None = None  # pending "capitalize next" / "all caps next"
        i = 0
        while i < len(toks):
            entry, length = self._match(toks, i)  # is a command spoken at this position?
            if self.paused:
                # While paused only resume/stop are honoured; every other word is dropped.
                if entry and entry.type == "control" and entry.action in ("resume", "stop"):
                    self._control(entry.action)
                i += length
                continue
            if entry is None:  # an ordinary word: type it as-is
                self._word(toks[i], mod)
                mod = None  # a modifier applies to one word only
                i += 1
                continue
            i += length  # skip past the words that made up the command phrase
            if entry.type == "symbol":
                self._emit(entry.text, entry.space)
            elif entry.type == "xml":
                i = self._xml(entry, toks, i)  # consumes the tag-name word(s) too
            elif entry.type == "modifier":
                mod = entry.action  # remember it for the next ordinary word
            elif entry.type == "literal":
                if i < len(toks):  # "literal period" types the word "period"
                    self._word(toks[i], mod)
                    mod = None
                    i += 1
            elif entry.type == "edit":
                (self._scratch if entry.action == "scratch" else self._delete_word)()
            elif entry.type == "control":
                self._control(entry.action)
        self._flush()  # turn whatever text is buffered into a final Type action
        return self._actions

    # ---------- tokens ----------

    def _tokenize(self, raw: str) -> list[tuple[str, str, str]]:
        """Split into (normalised, clean, plain) per word; punctuation-only tokens are dropped.

        normalised = lowercase, used to match commands
        clean      = original case, punctuation trimmed (used for XML tag names)
        plain      = what gets typed for ordinary words (clean, or raw if stripping is off)
        """
        out = []
        for tok in raw.split():
            clean = _EDGE.sub("", tok)  # "TagName." -> "TagName"
            if clean:
                out.append((clean.lower(), clean, clean if self.strip else tok))
        return out

    def _match(self, toks, i):
        """Find the longest command phrase starting at position i. Returns (Entry or None, words used)."""
        # Try the longest possible phrase first so "end quote" wins over "end" + "quote".
        for n in range(min(self._max, len(toks) - i), 0, -1):
            key = tuple(t[0] for t in toks[i : i + n])
            if key in self.commands:
                return self.commands[key], n
        return None, 1  # no command here: consume one ordinary word

    # ---------- emitting ----------

    def _reset_run(self) -> None:
        """Clear the per-utterance scratch state before processing a new utterance."""
        self._actions: list[Action] = []  # finished actions for this utterance
        self._cur = ""  # text buffered for the current Type action
        self._bs = 0  # backspaces waiting to be sent before the text
        self._snap = (self._glue, self._started)  # state to restore if the buffer is discarded

    def _emit(self, text: str, space: str) -> None:
        """Append text to the buffer, adding a leading space only when appropriate."""
        lead = "" if (space in NO_SPACE_BEFORE or self._glue or not self._started) else " "
        self._cur += lead + text
        self._glue = space in GLUE_AFTER  # does the next item attach tightly to this one?
        self._started = True

    def _word(self, tok, mod: str | None) -> None:
        """Emit an ordinary word, applying a pending capitalisation modifier."""
        word = tok[2]
        if mod == "capitalize":
            word = word[:1].upper() + word[1:]  # first letter only
        elif mod == "allcaps":
            word = word.upper()
        self._emit(word, "standalone")

    def _xml(self, entry: Entry, toks, i: int) -> int:
        """Handle "XML tag start/end/empty <name>". Returns the new token position."""
        if i >= len(toks):
            return i  # no tag name was spoken: ignore the command
        name = toks[i][1]  # the next word is the tag name, case as spoken
        i += 1
        # "user underscore name" -> keep joining while a joiner word is followed by another word
        while i + 1 < len(toks) and toks[i][0] in NAME_JOINERS:
            name += NAME_JOINERS[toks[i][0]] + toks[i + 1][1]
            i += 2
        text = {"start": f"<{name}>", "end": f"</{name}>", "empty": f"<{name} />"}[entry.action]
        self._emit(text, entry.space)
        return i

    def _flush(self) -> None:
        """Move buffered backspaces/text into the action list and record the text in history."""
        if self._bs:
            self._actions.append(Backspace(self._bs))  # backspaces must come before new text
            self._bs = 0
        if self._cur:
            self._actions.append(Type(self._cur))
            self._history.append(_Hist(self._cur, self._glue))
            del self._history[: -self._limit]  # keep only the newest `history_limit` chunks
            self._cur = ""
        self._snap = (self._glue, self._started)  # new restore point

    def _control(self, action: str) -> None:
        """Emit a pause/resume/stop event (after flushing any text typed before it)."""
        self._flush()
        if action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        self._actions.append(Control(action))

    # ---------- editing ----------

    def _sync_from_history(self) -> None:
        """Re-derive spacing state from the newest remembered chunk (or reset if none)."""
        if self._history:
            self._glue, self._started = self._history[-1].glue_after, True
        else:
            self._glue, self._started = False, False

    def _scratch(self) -> None:
        """"scratch that": drop the current phrase, or undo the previous utterance."""
        if self._cur:  # text from this same utterance hasn't been typed yet: just discard it
            self._cur = ""
            self._glue, self._started = self._snap
        elif self._history:  # otherwise erase what was already typed
            self._bs += len(self._history.pop().text)
            self._sync_from_history()

    def _delete_word(self) -> None:
        """"delete last word": remove the last word from this utterance or from earlier typing."""
        if self._cur.strip():  # there is un-typed text in this utterance: trim it directly
            self._cur = _trim_last_word(self._cur)[0]
            if self._cur:
                self._glue = _glue_from(self._cur)
            else:
                self._glue, self._started = self._snap
            return
        if self._cur:  # buffer holds only whitespace: discard it, then fall through to history
            self._cur = ""
            self._glue, self._started = self._snap
        if self._history:  # erase the last word that was already typed
            hist = self._history[-1]
            hist.text, removed = _trim_last_word(hist.text)
            self._bs += removed  # backspace exactly as many characters as we cut
            if hist.text:
                hist.glue_after = _glue_from(hist.text)
            else:
                self._history.pop()  # the whole chunk is gone
            self._sync_from_history()
