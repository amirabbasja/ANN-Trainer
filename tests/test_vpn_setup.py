"""Pinned executable download verification and safe installer extraction."""

import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import setup_vpn


def release_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("xray", b"fixture executable")
        archive.writestr("LICENSE", b"original upstream license")
        archive.writestr("../outside", b"must never be extracted")
    return output.getvalue()


class VPNInstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = release_archive()
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"LRAEP_CACHE_DIR": str(self.root / "cache")}).start()
        patch.object(setup_vpn.platform, "system", return_value="Linux").start()
        patch.object(setup_vpn.platform, "machine", return_value="x86_64").start()
        patch.object(setup_vpn, "ASSETS", {"x86_64": ("fixture.zip", self.digest)}).start()

    def test_verified_install_writes_only_selected_files_and_preserves_license(self):
        with patch.object(setup_vpn.urllib.request, "urlopen", return_value=io.BytesIO(self.data)):
            binary = setup_vpn.install_core(self.root / "bin")
        self.assertEqual(binary.read_bytes(), b"fixture executable")
        self.assertEqual(binary.stat().st_mode & 0o777, 0o755)
        self.assertEqual((binary.parent / "Xray-LICENSE").read_bytes(), b"original upstream license")
        self.assertFalse((self.root / "outside").exists())
        saved = self.root / "cache" / "downloads" / f"{setup_vpn.VERSION}-fixture.zip"
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)

    def test_modified_download_cannot_replace_existing_executable(self):
        destination = self.root / "bin"
        destination.mkdir()
        (destination / "xray").write_bytes(b"existing executable")
        with patch.object(setup_vpn.urllib.request, "urlopen", return_value=io.BytesIO(b"changed")):
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                setup_vpn.install_core(destination)
        self.assertEqual((destination / "xray").read_bytes(), b"existing executable")

    def test_offline_install_requires_and_reuses_a_verified_cache(self):
        with self.assertRaisesRegex(RuntimeError, "not cached"):
            setup_vpn.install_core(self.root / "bin", offline=True)
        cache = self.root / "cache" / "downloads"
        (cache / f"{setup_vpn.VERSION}-fixture.zip").write_bytes(self.data)
        with patch.object(setup_vpn.urllib.request, "urlopen") as request:
            setup_vpn.install_core(self.root / "bin", offline=True)
        request.assert_not_called()

    def test_correct_checksum_does_not_accept_missing_executable(self):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("LICENSE", b"license")
        data = output.getvalue()
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            setup_vpn.verified_archive(data, hashlib.sha256(data).hexdigest())

    def test_unsupported_platform_does_not_download_or_modify_files(self):
        with patch.object(setup_vpn.platform, "system", return_value="Other"), \
                patch.object(setup_vpn.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "manually"):
                setup_vpn.install_core(self.root / "bin")
        request.assert_not_called()
        self.assertFalse((self.root / "bin").exists())


if __name__ == "__main__":
    unittest.main()
