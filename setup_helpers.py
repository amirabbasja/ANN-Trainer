"""Fetch a pinned official PEARL checkout and build its Python proof helper.

Run with Python 3.12+ after installing Rust/cargo and a C build toolchain.
This never builds or starts a GPU trainer application from the checkout.
"""

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

from pearl_trainer import OFFICIAL_COMMIT

ROOT = Path(__file__).resolve().parent
OFFICIAL_URL = "https://github.com/pearl-research-labs/pearl.git"


def run(*command, **kwargs):
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".official" / "pearl")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required")
    if not shutil.which("git") or not shutil.which("cargo"):
        parser.error("install git, Rust/cargo (stable), and a C build toolchain first")
    source = args.source.expanduser().resolve()
    environment = args.venv.expanduser().resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.mkdir()
        run("git", "-C", source, "init")
        run("git", "-C", source, "remote", "add", "origin", OFFICIAL_URL)
        run("git", "-C", source, "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT)
        run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip()
    if revision != OFFICIAL_COMMIT or dirty:
        parser.error(f"source checkout must be clean at {OFFICIAL_COMMIT}")
    # Reuse an existing CUDA-enabled system PyTorch installation when available.
    if not environment.exists():
        venv.create(environment, with_pip=True, system_site_packages=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(python, "-m", "pip", "install", "-r", ROOT / "requirements.txt", "maturin>=1.7")
    build_env = os.environ.copy()
    build_env.setdefault("CARGO_BUILD_JOBS", "4")
    wheels = environment / "official-helper-wheels"
    wheels.mkdir(exist_ok=True)
    run(python, "-m", "maturin", "build", "--release", "--locked", "--interpreter", python, "--manifest-path",
        source / "py-pearl-mining" / "Cargo.toml", "--out", wheels, env=build_env)
    built = list(wheels.glob("py_pearl_mining-0.3.1-*.whl"))
    if len(built) != 1:
        raise RuntimeError("expected one official helper wheel in the build directory")
    run(python, "-m", "pip", "install", "--force-reinstall", built[0])
    print(f"Ready. Run from {ROOT}:")
    print(f"{python} -m pearl_trainer --official-source {source} self-test")


if __name__ == "__main__":
    main()
