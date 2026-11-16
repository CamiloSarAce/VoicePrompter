"""Launcher: python run.py"""
import sys
from pathlib import Path

# Make the `voice_prompter` package (inside src/) importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_prompter.main import main  # noqa: E402  (import must come after the path tweak)

if __name__ == "__main__":
    main()
