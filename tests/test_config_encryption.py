"""Plain source persistence and read-only support for legacy encrypted configs."""

import copy
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer import __main__ as cli
from LRAEP_trainer.config import (
    ConfigError, load_config, read_config, save_config, update_config,
)
from LRAEP_trainer.config_crypto import (
    EncryptionError, decode_string, encode_string, generate_key,
)


class StringEncryptionTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.key = generate_key()

    def test_unicode_and_empty_strings_round_trip(self):
        for value in ("", "plain text", "سلام 🌍\npassword=secret"):
            with self.subTest(value=value):
                encoded = encode_string(value, self.key)
                self.assertIsInstance(encoded, str)
                self.assertNotEqual(encoded, value)
                self.assertEqual(decode_string(encoded, self.key), value)

    def test_repeated_encryption_uses_fresh_randomness(self):
        self.assertNotEqual(encode_string("secret", self.key),
                            encode_string("secret", self.key))

    def test_environment_key_and_explicit_override(self):
        with patch.dict(os.environ, {"LRAEP_CONFIG_KEY": self.key}):
            token = encode_string("secret")
            self.assertEqual(decode_string(token), "secret")
            other_key = generate_key()
            explicit = encode_string("another secret", other_key)
            self.assertEqual(decode_string(explicit, other_key), "another secret")

    def test_missing_wrong_and_invalid_keys_fail_without_echoing_secrets(self):
        secret = "PRIVATE-STRING-THAT-MUST-NOT-APPEAR"
        token = encode_string(secret, self.key)
        for operation in (
            lambda: encode_string(secret),
            lambda: decode_string(token, generate_key()),
            lambda: encode_string(secret, "INVALID-SECRET-KEY"),
            lambda: decode_string("SECRET-NOT-A-TOKEN", self.key),
        ):
            with self.subTest(operation=operation), self.assertRaises(EncryptionError) as caught:
                operation()
            message = str(caught.exception)
            for value in (secret, token, self.key, "INVALID-SECRET-KEY", "SECRET-NOT-A-TOKEN"):
                self.assertNotIn(value, message)


class ConfigPersistenceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "config.json"
        self.key_path = self.path.with_name(self.path.name + ".key")
        self.key = generate_key()
        self.settings = {
            "version": 1,
            "official_source": "helpers",
            "training": {"output_dir": "results"},
            "pool": {
                "protocol": "KEX", "KEX_uname": "PRIVATE-KEX-IDENTITY",
                "WR_wallet": "PRIVATE-WR-IDENTITY", "password": "PRIVATE-POOL-PASSWORD",
                "KEX": {"host": "kex.example", "port": 8048, "tls": True},
                "WR": {"host": "wr.example", "port": 9000, "tls": False},
            },
            "node": {"username": "PRIVATE-USER", "password": "PRIVATE-NODE-PASSWORD"},
        }

    def write_legacy(self, raw=None):
        if raw is None:
            raw = json.dumps(self.settings)
        self.path.write_text(json.dumps({
            "format": "LRAEP-config-fernet-v1",
            "ciphertext": encode_string(raw, self.key),
        }))

    def test_save_writes_pretty_plain_json_without_generating_keys(self):
        self.settings["pool"]["password"] = "رمز عبور"
        original = copy.deepcopy(self.settings)
        save_config(self.path, self.settings)
        stored = self.path.read_text()
        self.assertEqual(json.loads(stored), original)
        self.assertIn('\n  "version": 1,\n', stored)
        self.assertIn("رمز عبور", stored)
        self.assertTrue(stored.endswith("\n"))
        self.assertEqual(read_config(self.path), original)
        self.assertEqual(self.settings, original)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(set(self.directory.iterdir()), {self.path})
        self.path.chmod(0o644)
        save_config(self.path, self.settings)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_runtime_defaults_and_relative_paths_survive_plain_save(self):
        save_config(self.path, self.settings)
        before = self.path.read_bytes()
        loaded = load_config(self.path)
        self.assertEqual(loaded["official_source"], str(self.directory / "helpers"))
        self.assertEqual(loaded["training"]["output_dir"], str(self.directory / "results"))
        self.assertEqual(loaded["device"], "cuda:0")
        self.assertEqual(loaded["pool"]["KEX_uname"], "PRIVATE-KEX-IDENTITY")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_plain_source_loads_without_a_key(self):
        self.path.write_text(json.dumps(self.settings))
        self.assertEqual(read_config(self.path), self.settings)
        self.assertEqual(load_config(self.path)["version"], 1)
        self.assertFalse(self.key_path.exists())

    def test_plain_saves_updates_and_reads_ignore_legacy_environment_or_key_file(self):
        self.key_path.write_text("UNRELATED-INVALID-KEY\n")
        original_key = self.key_path.read_bytes()
        with patch.dict(os.environ, {"LRAEP_CONFIG_KEY": "INVALID-ENV-KEY"}):
            save_config(self.path, self.settings)
            update_config(self.path, {"training": {"max_batches": 4}})
            self.assertEqual(load_config(self.path)["training"]["max_batches"], 4)
            self.assertEqual(json.loads(self.path.read_text())["training"]["max_batches"], 4)
        self.assertEqual(self.key_path.read_bytes(), original_key)
        self.assertEqual(set(self.directory.iterdir()), {self.path, self.key_path})

    def test_legacy_encrypted_read_is_in_memory_and_does_not_rewrite(self):
        self.write_legacy()
        before = self.path.read_bytes()
        self.assertEqual(read_config(self.path, key=self.key), self.settings)
        self.assertEqual(load_config(self.path, key=self.key)["pool"]["KEX_uname"],
                         "PRIVATE-KEX-IDENTITY")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_legacy_document_can_be_migrated_to_plain_without_generated_keys_or_backups(self):
        self.write_legacy()
        original = read_config(self.path, key=self.key)
        save_config(self.path, original)
        self.assertEqual(json.loads(self.path.read_text()), self.settings)
        self.assertEqual(read_config(self.path), self.settings)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_update_legacy_document_with_existing_key_saves_plain_and_preserves_key(self):
        self.write_legacy()
        self.key_path.write_text(self.key + "\n")
        original_key = self.key_path.read_bytes()
        update_config(self.path, {"training": {"max_batches": 4}})
        self.assertEqual(json.loads(self.path.read_text())["training"]["max_batches"], 4)
        self.assertEqual(read_config(self.path)["pool"], self.settings["pool"])
        self.assertEqual(self.key_path.read_bytes(), original_key)
        self.assertEqual(set(self.directory.iterdir()), {self.path, self.key_path})

    def test_missing_legacy_key_never_generates_a_replacement(self):
        self.write_legacy()
        before = self.path.read_bytes()
        with self.assertRaises(ConfigError):
            load_config(self.path)
        self.assertFalse(self.key_path.exists())
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_environment_key_precedence_and_explicit_override(self):
        self.write_legacy()
        self.key_path.write_text(self.key + "\n")
        before = self.path.read_bytes()
        with patch.dict(os.environ, {"LRAEP_CONFIG_KEY": "INVALID-ENV-KEY"}):
            with self.assertRaises(ConfigError) as caught:
                load_config(self.path)
            self.assertNotIn("INVALID-ENV-KEY", str(caught.exception))
            self.assertEqual(read_config(self.path, key=self.key), self.settings)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(read_config(self.path), self.settings)

    def test_wrong_legacy_key_and_tampering_fail_with_secret_safe_errors(self):
        self.write_legacy()
        with self.assertRaises(ConfigError) as caught:
            load_config(self.path, key=generate_key())
        self.assertNotIn(self.key, str(caught.exception))
        envelope = json.loads(self.path.read_text())
        token = envelope["ciphertext"]
        index = len(token) // 2
        envelope["ciphertext"] = token[:index] + ("A" if token[index] != "A" else "B") + token[index + 1:]
        self.path.write_text(json.dumps(envelope))
        with self.assertRaises(ConfigError) as caught:
            load_config(self.path, key=self.key)
        for secret in (self.key, token, "PRIVATE-POOL-PASSWORD"):
            self.assertNotIn(secret, str(caught.exception))

    def test_nested_plain_update_preserves_siblings_and_does_not_mutate_input(self):
        save_config(self.path, self.settings)
        changes = {"pool": {"KEX": {"port": 12345}}, "training": {"max_batches": 3}}
        original = copy.deepcopy(changes)
        update_config(self.path, changes)
        expected = copy.deepcopy(self.settings)
        expected["pool"]["KEX"]["port"] = 12345
        expected["training"]["max_batches"] = 3
        self.assertEqual(json.loads(self.path.read_text()), expected)
        self.assertEqual(read_config(self.path), expected)
        self.assertEqual(changes, original)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_invalid_save_or_update_preserves_existing_plain_document(self):
        save_config(self.path, self.settings)
        before = self.path.read_bytes()
        for operation in (
            lambda: save_config(self.path, {"version": 1, "pool": {"port": 0}}),
            lambda: update_config(self.path, {"pool": {"port": 0}}),
            lambda: save_config(self.path, {"version": 1, "SECRET-UNKNOWN-FIELD": "PRIVATE"}),
        ):
            with self.subTest(operation=operation), self.assertRaises(ConfigError) as caught:
                operation()
            self.assertNotIn("SECRET-UNKNOWN-FIELD", str(caught.exception))
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_failed_atomic_replace_preserves_plain_document_and_cleans_temp_file(self):
        save_config(self.path, self.settings)
        before = self.path.read_bytes()
        with patch("LRAEP_trainer.config.os.replace", side_effect=OSError("write failed")):
            with self.assertRaises(ConfigError):
                update_config(self.path, {"training": {"max_batches": 4}})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_duplicate_fields_and_nonfinite_numbers_remain_invalid_in_legacy_reads(self):
        for raw in ('{"version":1,"version":1}',
                    '{"version":1,"training":{"max_seconds":NaN}}'):
            with self.subTest(raw=raw):
                self.write_legacy(raw)
                with self.assertRaises(ConfigError):
                    load_config(self.path, key=self.key)

    def test_plain_cli_dispatch_receives_settings_without_writing_source(self):
        save_config(self.path, self.settings)
        before = self.path.read_bytes()
        with patch.object(cli, "pool_train", return_value=0) as train, \
                patch.object(cli, "load_official_helpers") as helpers:
            self.assertEqual(cli.main(["--config", str(self.path), "pool", "--miner", "KEX"]), 0)
        effective = train.call_args.args[0]
        self.assertEqual(effective["pool"]["host"], "kex.example")
        self.assertEqual(effective["pool"]["KEX_uname"], "PRIVATE-KEX-IDENTITY")
        self.assertEqual(effective["pool"]["password"], "PRIVATE-POOL-PASSWORD")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.key_path.exists())
        helpers.assert_not_called()

    def test_legacy_cli_dispatch_decrypts_in_memory_and_leaves_document_unchanged(self):
        self.write_legacy()
        self.key_path.write_text(self.key + "\n")
        before = self.path.read_bytes()
        with patch.object(cli, "pool_train", return_value=0) as train:
            self.assertEqual(cli.main(["--config", str(self.path), "pool", "--miner", "KEX"]), 0)
        self.assertEqual(train.call_args.args[0]["pool"]["KEX_uname"], "PRIVATE-KEX-IDENTITY")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
