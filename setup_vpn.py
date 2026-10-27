"""Install the pinned official Xray core into LRAEP's application directory."""

import argparse
import hashlib
import io
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from LRAEP_trainer.paths import artifacts_dir, cache_dir

VERSION = "v26.10.21"
# Official GitHub release asset SHA-256 values, pinned with the release.
ASSETS = {
    "x86_64": ("Xray-linux-64.zip", "0eec3ed4163c273d8c8bf821f7d219e257921cf584aa3fb970715e458e631978"),
    "aarch64": ("Xray-linux-arm64-v8a.zip", "d1304946c486d98e6b886e9b931eaaaebb23835b30e217638037b7f2721d4e9c"),
}
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024


def verified_archive(data: bytes, digest: str) -> zipfile.ZipFile:
    if len(data) > MAX_ARCHIVE_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError("Xray download failed SHA-256 verification")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
        binary = archive.getinfo("xray")
        if (binary.is_dir() or stat.S_ISLNK(binary.external_attr >> 16)
                or binary.file_size > MAX_ARCHIVE_BYTES):
            raise ValueError("invalid executable")
        return archive
    except (zipfile.BadZipFile, KeyError, ValueError):
        raise RuntimeError("Xray release archive is invalid") from None


def _write_atomic(path: Path, data: bytes, mode: int):
    descriptor, temporary = tempfile.mkstemp(prefix=".LRAEP-install-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            os.fchmod(output.fileno(), mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def install_core(destination: Path | None = None, *, offline: bool = False) -> Path:
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
    if platform.system() != "Linux" or machine not in ASSETS:
        raise RuntimeError("Automatic Xray installation supports Linux x86_64/arm64; install Xray manually and set vpn.core_path")
    filename, digest = ASSETS[machine]
    cache = cache_dir() / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    saved = cache / f"{VERSION}-{filename}"
    data = saved.read_bytes() if saved.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != digest:
        if offline:
            raise RuntimeError("Verified Xray archive is not cached; run setup_vpn.py without --offline")
        url = f"https://github.com/XTLS/Xray-core/releases/download/{VERSION}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "LRAEP-setup/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_ARCHIVE_BYTES + 1)
        with verified_archive(data, digest):
            pass
        _write_atomic(saved, data, 0o600)
    destination = (destination or artifacts_dir().parent / "bin").expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with verified_archive(data, digest) as archive:
        binary_data = archive.read("xray")
        # Manual extraction writes exactly these files; archive paths are never followed.
        license_name = next((name for name in ("LICENSE", "LICENSE.txt") if name in archive.namelist()), None)
        if license_name:
            if archive.getinfo(license_name).file_size > 2 * 1024 * 1024:
                raise RuntimeError("Xray license entry is unexpectedly large")
            _write_atomic(destination / "Xray-LICENSE", archive.read(license_name), 0o644)
        _write_atomic(destination / "xray", binary_data, 0o755)
    return destination / "xray"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, help="directory for the Xray executable")
    parser.add_argument("--offline", action="store_true", help="install only from the verified local download cache")
    args = parser.parse_args()
    try:
        binary = install_core(args.destination, offline=args.offline)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Xray setup failed: {exc}\n")
    print(f"Installed official Xray {VERSION} at {binary}")


if __name__ == "__main__":
    main()
