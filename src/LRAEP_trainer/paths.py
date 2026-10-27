"""Locations of bundled source and generated LRAEP application files."""

import os
from pathlib import Path
import platform


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "LRAEP_helpers"


def default_config_path() -> Path:
    """Use the checkout's shared config even when launched from another folder.

    Installed packages use the current folder's config.json. An explicit CLI
    configuration path can still select a shared file stored elsewhere.
    """
    project = Path(__file__).resolve().parents[2]
    if (project / "ann_trainer.py").is_file():
        return project / "config.json"
    return Path.cwd() / "config.json"


def bundled_xray() -> Path | None:
    """Locate a compatible core beside the package after copying or installing.

    Platform directories prevent a bundled Linux binary from being selected
    on another operating system or CPU. Missing assets use the existing
    explicit-path, PATH or application-data discovery instead.
    """
    if platform.system() != "Linux":
        return None
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    if machine not in {"x86_64", "aarch64"}:
        return None
    executable = Path(__file__).resolve().parent / "bin" / f"linux-{machine}" / "xray"
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    return None


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
