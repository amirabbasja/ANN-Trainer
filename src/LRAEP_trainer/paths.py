"""Locations of bundled source and generated LRAEP application files."""

import os
from pathlib import Path


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "LRAEP_helpers"


def cache_dir() -> Path:
    override = os.environ.get("LRAEP_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "LRAEP"


def artifacts_dir() -> Path:
    override = os.environ.get("LRAEP_DATA_DIR")
    if override:
        return Path(override).expanduser() / "artifacts"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "LRAEP" / "artifacts"
