import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer import config_transfer as transfer
from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, save_config
from LRAEP_trainer.config_export import decode_config_document


class ConfigTransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "trainer" / "config.json"
        self.source.parent.mkdir()
        self.destination = self.directory / "WR" / "config.json"
        self.values = {
            "version": 1, "mode": "solo", "official_source": "src/LRAEP_helpers",
            "training": {"max_batches": 2, "output_dir": "results"},
            "S3": {"protocol": "KEX", "WR_S2": "PRIVATE-IDENTITY",
                   "password": "PRIVATE-S3-PASSWORD",
                   "WR": {"host": "wr.example", "port": 9000, "tls": False}},
            "vpn": {"enabled": True, "link": "vless://fixture@example.invalid:443"},
            "telegram": {"enabled": False, "tgBot_key": "123:VISIBLE_TOKEN"},
            "side_job": {"repo_url": "https://huggingface.co/datasets/fixture/thesis"},
        }
        self.source.write_text(json.dumps(self.values))

    def expected(self):
        values = copy.deepcopy(self.values)
        values["mode"] = "S3"
        values["S3"]["protocol"] = "WR"
        values.setdefault("side_job", {}).setdefault("repo_url", DEFAULT_CONFIG["side_job"]["repo_url"])
        return values

    def test_default_export_encrypts_only_s3_and_loads_in_isolated_standalone(self):
        before = self.source.read_bytes()
        self.assertEqual(transfer.export_wr_config(self.source, self.destination), self.destination)
        document = json.loads(self.destination.read_text())
        expected = self.expected()
        self.assertNotIn("S3", document)
        self.assertEqual(decode_config_document(document), expected)
        for name in expected.keys() - {"S3"}:
            self.assertEqual(document[name], expected[name])
        self.assertNotIn("PRIVATE-S3-PASSWORD", self.destination.read_text())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination})
        self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o600)
        standalone_source = Path(__file__).resolve().parents[2] / "LRAEP-WR-Trainer" / "src"
        if not (standalone_source / "workrunner" / "config.py").is_file():
            self.skipTest("standalone checkout unavailable")
        script = """
import json, sys
from workrunner.config import read_config, load_config, resolve_S3_settings
raw = read_config(sys.argv[1])
runtime = load_config(sys.argv[1])
selected = resolve_S3_settings(runtime["S3"])
assert selected["protocol"] == "WR" and selected["host"] == "wr.example"
print(json.dumps(raw))
"""
        environment = dict(os.environ, PYTHONPATH=str(standalone_source),
                           open_text="wrong", LRAEP_CONFIG_KEY="wrong")
        result = subprocess.run([sys.executable, "-c", script, str(self.destination)],
                                cwd=self.directory, env=environment, capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), expected)

    def test_custom_output_key_and_encrypted_source_are_independent(self):
        save_config(self.source, self.values, key="source key")
        before = self.source.read_bytes()
        transfer.export_wr_config(self.source, self.destination, source_key="source key", key="output key")
        document = json.loads(self.destination.read_text())
        self.assertEqual(decode_config_document(document, "output key"), self.expected())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.destination.parent / ".env").exists())

    def test_existing_dotenv_is_neither_read_nor_modified(self):
        self.destination.parent.mkdir()
        env = self.destination.parent / ".env"
        env.write_text("open_text=obsolete\nUNRELATED=keep\n")
        before = env.read_bytes()
        transfer.export_wr_config(self.source, self.destination)
        self.assertEqual(env.read_bytes(), before)
        self.assertEqual(decode_config_document(json.loads(self.destination.read_text())), self.expected())

    def test_invalid_key_or_runtime_settings_do_not_replace_export(self):
        self.destination.parent.mkdir()
        self.destination.write_text("previous")
        before = self.source.read_bytes()
        for key in ("", "  ", 123):
            with self.assertRaises(ConfigError):
                transfer.export_wr_config(self.source, self.destination, key=key)
            self.assertEqual(self.destination.read_text(), "previous")
        for changes in ({"enabled": False}, {"enabled": True, "link": ""}):
            values = copy.deepcopy(self.values)
            values["vpn"] = changes
            self.source.write_text(json.dumps(values))
            with self.assertRaises(ConfigError):
                transfer.export_wr_config(self.source, self.destination)
            self.assertEqual(self.destination.read_text(), "previous")
        self.source.write_bytes(before)

    def test_failed_atomic_write_preserves_source_and_destination(self):
        self.destination.parent.mkdir()
        self.destination.write_text("previous")
        before = self.source.read_bytes()
        with patch("LRAEP_trainer.config.os.replace", side_effect=OSError("PRIVATE-ERROR")):
            with self.assertRaises(ConfigError):
                transfer.export_wr_config(self.source, self.destination)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(self.destination.read_text(), "previous")
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination})

    def test_export_rejects_source_alias_and_dotenv_destinations(self):
        alias = self.source.parent / "alias.json"
        alias.symlink_to(self.source)
        before = self.source.read_bytes()
        for destination in (self.source, alias, self.source.parent / ".env"):
            with self.assertRaises(ConfigError):
                transfer.export_wr_config(self.source, destination)
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
