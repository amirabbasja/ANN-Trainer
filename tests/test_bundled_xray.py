"""Packaged VPN core discovery and integrity, without downloads or remote I/O."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer import paths


class BundledXrayDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.package = Path(temporary.name) / 'relocated' / 'LRAEP_trainer'
        self.binary = self.package / 'bin' / 'linux-x86_64' / 'xray'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'local fixture executable; never launched')
        self.binary.chmod(0o755)
        self.source = self.package / 'paths.py'
        self.source.write_text('# relocated package path fixture\n')

    def discover(self, system='Linux', machine='x86_64'):
        with patch.object(paths, '__file__', str(self.source)), \
                patch('platform.system', return_value=system), \
                patch('platform.machine', return_value=machine), \
                patch.dict('os.environ', {'PATH': '', 'LRAEP_DATA_DIR': '/unused-old-install'}):
            return paths.bundled_xray()

    def test_relocated_package_is_found_without_path_or_generated_artifacts(self):
        self.assertEqual(self.discover(), self.binary)
        self.assertTrue(self.discover().is_absolute())

    def test_supported_x86_64_architecture_aliases_select_the_same_binary(self):
        for machine in ('x86_64', 'amd64', 'AMD64'):
            with self.subTest(machine=machine):
                self.assertEqual(self.discover(machine=machine), self.binary)

    def test_incompatible_platforms_cannot_select_the_x86_binary(self):
        for system in ('Windows', 'Darwin', 'FreeBSD'):
            with self.subTest(system=system):
                self.assertIsNone(self.discover(system=system))
        for machine in ('aarch64', 'arm64', 'i686', 'ppc64le', 'riscv64'):
            with self.subTest(machine=machine):
                self.assertIsNone(self.discover(machine=machine))

    def test_missing_and_nonexecutable_cores_are_unavailable(self):
        self.binary.chmod(0o644)
        self.assertIsNone(self.discover())
        self.binary.unlink()
        self.assertIsNone(self.discover())


class BundledXrayIntegrityTests(unittest.TestCase):
    def test_distributed_core_and_upstream_license_match_manifest(self):
        directory = Path(paths.__file__).resolve().parent / 'bin'
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(manifest['license'], 'MPL-2.0')
        self.assertEqual(manifest['platform'], 'linux-x86_64')
        self.assertEqual(set(manifest['files']), {'linux-x86_64/xray', 'Xray-LICENSE'})
        for name, expected in manifest['files'].items():
            with self.subTest(file=name):
                self.assertRegex(expected, r'^[0-9a-f]{64}$')
                path = directory / name
                self.assertTrue(path.is_file())
                with path.open('rb') as handle:
                    self.assertEqual(hashlib.file_digest(handle, 'sha256').hexdigest(), expected)
        license_text = (directory / 'Xray-LICENSE').read_text()
        self.assertIn('Mozilla Public License', license_text)
        self.assertIn('2.0', license_text)
        self.assertRegex(manifest['archive']['sha256'], r'^[0-9a-f]{64}$')
        self.assertTrue(manifest['source_url'].startswith('https://github.com/XTLS/Xray-core/'))
        self.assertTrue(manifest['release_url'].startswith('https://github.com/XTLS/Xray-core/'))


if __name__ == '__main__':
    unittest.main()
