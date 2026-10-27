"""Build the attributed proof helpers bundled in src/LRAEP_helpers.

Run with Python 3.12+, Rust/cargo, and a C build toolchain. No upstream Git
checkout or downloaded miner executable is used.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from LRAEP_trainer.helpers import verify_bundle
from LRAEP_trainer.paths import DEFAULT_SOURCE, cache_dir


def run(*command, **kwargs):
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--offline", action="store_true", help="use cached Rust dependencies")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Python 3.12 or newer is required")
    if not shutil.which("cargo"):
        parser.error("install Rust/cargo and a C build toolchain first")
    try:
        source = verify_bundle(args.source)
    except RuntimeError as exc:
        parser.error(str(exc))
    environment = args.venv.expanduser().resolve()
    if not environment.exists():
        venv.create(environment, with_pip=True, system_site_packages=True)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(python, "-m", "pip", "install", "setuptools>=68", "wheel", "maturin>=1.7")
    run(python, "-m", "pip", "install", "--no-build-isolation", "-e", ROOT)
    build_env = os.environ.copy()
    build_env.setdefault("CARGO_BUILD_JOBS", "4")
    build_env.setdefault("CARGO_TARGET_DIR", str(cache_dir() / "rust"))
    wheels = cache_dir() / "helper-wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    command = [python, "-m", "maturin", "build", "--release", "--locked", "--interpreter", python,
               "--manifest-path", source / "rust" / "py-LRAEP-mining" / "Cargo.toml", "--out", wheels]
    if args.offline:
        command.append("--offline")
    run(*command, env=build_env)
    built = list(wheels.glob("py_lraep_mining-0.3.1-*.whl"))
    if len(built) != 1:
        raise RuntimeError("expected one LRAEP proof helper wheel in the build directory")
    run(python, "-m", "pip", "install", "--force-reinstall", "--no-deps", built[0])
    print(f"Ready: {python} -m LRAEP_trainer self-test")


if __name__ == "__main__":
    main()
