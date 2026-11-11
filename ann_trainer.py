#!/usr/bin/env python3
"""Python entry point for LRAEP S1 and connection checks.

Example: .venv/bin/python ann_trainer.py --config config.json S3
"""

from pathlib import Path
import sys


def main(argv=None) -> int:
    source = str(Path(__file__).resolve().parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    # Configuration is read after importing the entry point. Avoid creating
    # bytecode during bootstrap, before the training settings are known.
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from LRAEP_trainer.__main__ import main as run

        return run(argv)
    finally:
        sys.dont_write_bytecode = previous


if __name__ == "__main__":
    raise SystemExit(main())
