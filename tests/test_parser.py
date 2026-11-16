import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voice_prompter.parser import Backspace, Control, Parser, Type, load_commands  # noqa: E402

COMMANDS = load_commands(ROOT / "commands.toml")


def new_parser(**kw):
    return Parser(COMMANDS, **kw)


def typed(raw, parser=None):
    """Process one utterance and return the text typed (assumes only Type actions)."""
    actions = (parser or new_parser()).process(raw)
    assert all(isinstance(a, Type) for a in actions), actions
    return "".join(a.text for a in actions)


class XmlTags(unittest.TestCase):
    def test_start_keeps_case(self):
        self.assertEqual(typed("XML Tag Start: TagName"), "<TagName>")

    def test_end_keeps_case(self):
        self.assertEqual(typed("XML tag end: TagName."), "</TagName>")

    def test_empty(self):
        self.assertEqual(typed("xml tag empty br"), "<br />")

    def test_joined_name(self):
        self.assertEqual(typed("xml tag start user underscore name"), "<user_name>")

    def test_spelled_xml(self):
        self.assertEqual(typed("x m l tag start task"), "<task>")

    def test_missing_name_is_ignored(self):
        self.assertEqual(new_parser().process("xml tag start"), [])

    def test_full_block(self):
        raw = "XML tag start task new line write a summary new line XML tag end task"
        self.assertEqual(typed(raw), "<task>\nwrite a summary\n</task>")


class Punctuation(unittest.TestCase):
    def test_comma_period(self):
        self.assertEqual(typed("Hello comma world period"), "Hello, world.")

    def test_whisper_case_and_punct_on_command(self):
        self.assertEqual(typed("yes Comma, no"), "yes, no")

    def test_auto_punctuation_stripped(self):
        self.assertEqual(typed("Hello, world."), "Hello world")

    def test_auto_punctuation_kept_when_disabled(self):
        self.assertEqual(typed("Hello, world.", new_parser(strip_auto_punctuation=False)), "Hello, world.")

    def test_ellipsis(self):
        self.assertEqual(typed("wait ellipsis"), "wait...")

    def test_colon_semicolon(self):
        self.assertEqual(typed("a colon b semicolon c"), "a: b; c")

    def test_parentheses(self):
        self.assertEqual(typed("open parenthesis note close parenthesis"), "(note)")

    def test_quotes(self):
        self.assertEqual(typed("she said quote hi end quote"), 'she said "hi"')

    def test_slash_joins(self):
        self.assertEqual(typed("and slash or"), "and/or")

    def test_hyphen_vs_dash(self):
        self.assertEqual(typed("well hyphen known"), "well-known")
        self.assertEqual(typed("a dash b"), "a - b")

    def test_question_exclamation(self):
        self.assertEqual(typed("really question mark yes exclamation mark"), "really? yes!")

    def test_layout(self):
        self.assertEqual(typed("one new line two new paragraph three"), "one\ntwo\n\nthree")


class Modifiers(unittest.TestCase):
    def test_capitalize(self):
        self.assertEqual(typed("capitalize next hello world"), "Hello world")

    def test_all_caps(self):
        self.assertEqual(typed("all caps next api key"), "API key")

    def test_literal(self):
        self.assertEqual(typed("say literal comma"), "say comma")


class Spacing(unittest.TestCase):
    def test_space_between_utterances(self):
        p = new_parser()
        self.assertEqual(typed("Hello", p), "Hello")
        self.assertEqual(typed("world", p), " world")

    def test_glue_across_utterances(self):
        p = new_parser()
        typed("see", p)
        self.assertEqual(typed("open parenthesis", p), " (")
        self.assertEqual(typed("note", p), "note")


class Editing(unittest.TestCase):
    def test_scratch_same_utterance(self):
        self.assertEqual(new_parser().process("hello world scratch that"), [])

    def test_scratch_previous_utterance(self):
        p = new_parser()
        p.process("hello world")
        self.assertEqual(p.process("scratch that"), [Backspace(11)])

    def test_scratch_restores_spacing(self):
        p = new_parser()
        p.process("see")
        p.process("open parenthesis")
        self.assertEqual(p.process("scratch that"), [Backspace(2)])
        self.assertEqual(p.process("x"), [Type(" x")])

    def test_delete_word_same_utterance(self):
        self.assertEqual(typed("one two three delete last word"), "one two")

    def test_delete_word_previous_utterance(self):
        p = new_parser()
        p.process("one two three")
        self.assertEqual(p.process("delete last word"), [Backspace(6)])
        self.assertEqual(p.process("four"), [Type(" four")])

    def test_delete_with_nothing_typed(self):
        self.assertEqual(new_parser().process("delete last word"), [])


class Control_(unittest.TestCase):
    def test_pause_resume_in_one_utterance(self):
        actions = new_parser().process("pause listening hello world resume listening goodbye")
        self.assertEqual(actions, [Control("pause"), Control("resume"), Type("goodbye")])

    def test_pause_persists_across_utterances(self):
        p = new_parser()
        self.assertEqual(p.process("pause listening"), [Control("pause")])
        self.assertEqual(p.process("hello there"), [])
        self.assertEqual(p.process("resume listening"), [Control("resume")])
        self.assertEqual(p.process("hi"), [Type("hi")])

    def test_stop(self):
        self.assertEqual(new_parser().process("stop listening"), [Control("stop")])


class Config(unittest.TestCase):
    def test_longest_phrase_wins(self):
        # "end quote" must not become "end" + "quote"
        self.assertEqual(typed("quote a end quote"), '"a"')

    def test_duplicate_phrase_rejected(self):
        import tempfile

        toml = '[[command]]\ntype="symbol"\nsay=["x"]\ntext="1"\n[[command]]\ntype="symbol"\nsay=["x"]\ntext="2"\n'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.toml"
            p.write_text(toml, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_commands(p)


if __name__ == "__main__":
    unittest.main()
