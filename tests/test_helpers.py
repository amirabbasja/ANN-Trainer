"""Vendored helper integrity and relocation checks without Git or GPU work."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from LRAEP_trainer import OFFICIAL_COMMIT
from LRAEP_trainer.helpers import verify_bundle


class VendoredHelperIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "helpers"
        self.source.mkdir()
        self.relative = "python/helper.py"
        self.helper = self.source / self.relative
        self.helper.parent.mkdir()
        self.helper.write_bytes(b"VALUE = 123\n")
        self.manifest = {
            "revision": OFFICIAL_COMMIT,
            "files": {self.relative: hashlib.sha256(self.helper.read_bytes()).hexdigest()},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def test_bundle_can_move_without_an_upstream_git_checkout(self):
        verify_bundle(self.source)
        moved = self.root / "relocated" / "src" / "LRAEP_helpers"
        moved.parent.mkdir(parents=True)
        shutil.move(self.source, moved)
        verify_bundle(moved)
        self.assertFalse((moved / ".git").exists())

    def test_modified_helper_is_rejected(self):
        self.helper.write_bytes(b"VALUE = 456\n")
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)

    def test_missing_helper_is_rejected(self):
        self.helper.unlink()
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)

    def test_symlinked_helper_is_rejected_even_if_bytes_match(self):
        external = self.root / "external.py"
        self.helper.replace(external)
        self.helper.symlink_to(external)
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)

    def test_symlinked_parent_directory_is_rejected(self):
        external = self.root / "external-python"
        self.helper.parent.replace(external)
        (self.source / "python").symlink_to(external, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)

    def test_revision_must_match_the_reviewed_upstream_revision(self):
        self.manifest["revision"] = "0" * 40
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)

    def test_absolute_and_parent_paths_are_rejected(self):
        for relative in ("/tmp/helper.py", "../helper.py", "python/../../helper.py"):
            with self.subTest(relative=relative):
                self.manifest["files"] = {relative: hashlib.sha256(b"irrelevant").hexdigest()}
                self.write_manifest()
                with self.assertRaises(RuntimeError):
                    verify_bundle(self.source)

    def test_missing_and_malformed_manifests_are_rejected(self):
        manifest = self.source / "manifest.json"
        manifest.unlink()
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)
        manifest.write_text("{invalid json")
        with self.assertRaises(RuntimeError):
            verify_bundle(self.source)


if __name__ == "__main__":
    unittest.main()
