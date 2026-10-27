#!/usr/bin/env python3
"""Python entry point for LRAEP mining and connection checks.

Example: .venv/bin/python ann_trainer.py --config config.json pool
"""

from pathlib import Path
import sys


def main(argv=None) -> int:
    source = str(Path(__file__).resolve().parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from LRAEP_trainer.__main__ import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
