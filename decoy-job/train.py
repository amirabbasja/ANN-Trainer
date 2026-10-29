#!/usr/bin/env python3
"""Run the CNN job from the source checkout, including from another directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from LRAEP_trainer.decoy import main

if __name__ == "__main__":
    raise SystemExit(main())
