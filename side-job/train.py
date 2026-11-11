#!/usr/bin/env python3
"""Run the CNN job from the source checkout, including from another directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main(argv=None):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from LRAEP_trainer.side_job import main as run

        return run(argv)
    finally:
        sys.dont_write_bytecode = previous


if __name__ == "__main__":
    raise SystemExit(main())
