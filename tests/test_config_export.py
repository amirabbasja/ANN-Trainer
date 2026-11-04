import base64
import copy
import json
import math
import os
import unittest

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from LRAEP_trainer import config_export as export


class ConfigExportTests(unittest.TestCase):
    def setUp(self):
        self.text = '  رمز "key" 🌍\nsecond line  '
        self.values = {
            "version": 1, "training": {"max_batches": 2},
            "S3": {"WR_S2": "PRIVATE-S2", "password": "PRIVATE-PASSWORD"},
            "side_job": {"repo_url": "https://huggingface.co/datasets/owner/dataset"},
            "telegram": {"tgBot_key": "123:VISIBLE_TOKEN"},
            "relative_path": "../helpers", "unicode": "سلام",
        }

    def test_only_s3_is_encrypted_and_other_values_remain_editable(self):
        original = copy.deepcopy(self.values)
        document = export.encode_config_document(self.values)
        self.assertNotIn("S3", document)
        self.assertEqual(set(document["S_3"]), {"format", "salt", "ciphertext"})
        self.assertEqual({k: v for k, v in document.items() if k != "S_3"},
                         {k: v for k, v in original.items() if k != "S3"})
        for secret in ("PRIVATE-S2", "PRIVATE-PASSWORD", "WR_S2"):
            self.assertNotIn(secret, json.dumps(document))
        self.assertEqual(export.decode_config_document(document), original)
        self.assertEqual(export.decode_config_document(document, "qwe"), original)
        self.assertEqual(self.values, original)

    def test_custom_text_and_key_derivation_preserve_exact_unicode(self):
        document = export.encode_config_document(self.values, self.text)
        envelope = document["S_3"]
        salt = base64.urlsafe_b64decode(envelope["salt"])
        key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(self.text.encode())
        decoded = Fernet(base64.urlsafe_b64encode(key)).decrypt(envelope["ciphertext"].encode())
        self.assertEqual(json.loads(decoded), self.values["S3"])
        self.assertEqual(export.decode_config_document(document, self.text), self.values)
        with self.assertRaises(export.ExportError):
            export.decode_config_document(document, self.text.strip())

    def test_fresh_randomness_is_confined_to_s3(self):
        first = export.encode_config_document(self.values)
        second = export.encode_config_document(self.values)
        self.assertNotEqual(first["S_3"]["salt"], second["S_3"]["salt"])
        self.assertNotEqual(first["S_3"]["ciphertext"], second["S_3"]["ciphertext"])
        self.assertEqual(export.decode_config_document(first), export.decode_config_document(second))

    def test_plain_and_legacy_whole_documents_are_readable_for_migration(self):
        self.assertEqual(export.decode_config_document(self.values), self.values)
        legacy = export._encrypt_object(self.values, self.text)
        self.assertEqual(export.decode_config_document(legacy, self.text), self.values)
        self.assertNotIn("format", export.encode_config_document(
            export.decode_config_document(legacy, self.text)))

    def test_wrong_key_and_tampering_fail_without_echoing_values(self):
        document = export.encode_config_document(self.values, self.text)
        damaged = copy.deepcopy(document)
        token = bytearray(base64.urlsafe_b64decode(damaged["S_3"]["ciphertext"]))
        token[-5] ^= 1
        damaged["S_3"]["ciphertext"] = base64.urlsafe_b64encode(token).decode()
        for value, key in ((document, "WRONG-PRIVATE-KEY"), (damaged, self.text)):
            with self.assertRaises(export.ExportError) as caught:
                export.decode_config_document(value, key)
            for secret in (key, self.text, "PRIVATE-PASSWORD", document["S_3"]["ciphertext"]):
                self.assertNotIn(secret, str(caught.exception))

    def test_ambiguous_or_malformed_encrypted_sections_fail(self):
        encoded = export.encode_config_document(self.values)
        invalid = [None, [], {"S_3": "PRIVATE"}, {"S_3": {}},
                   dict(encoded, S3={}), {"S_3": dict(encoded["S_3"], extra=True)},
                   {"S_3": dict(encoded["S_3"], salt="bad")},
                   {"S_3": dict(encoded["S_3"], format="PRIVATE")}]
        for value in invalid:
            with self.subTest(value=type(value).__name__), self.assertRaises(export.ExportError):
                export.decode_config_document(value)
        with self.assertRaises(export.ExportError):
            export.encode_config_document(dict(encoded, S3={}))

    def test_authenticated_json_rejects_duplicates_nonfinite_and_non_objects(self):
        salt = os.urandom(16)
        cipher = export._cipher(self.text, salt)
        for plaintext in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'bad'):
            envelope = {"format": export.FORMAT, "salt": base64.urlsafe_b64encode(salt).decode(),
                        "ciphertext": cipher.encrypt(plaintext).decode()}
            with self.assertRaises(export.ExportError):
                export.decode_config_document({"version": 1, "S_3": envelope}, self.text)

    def test_invalid_documents_and_keys_fail_before_encoding(self):
        recursive = {}
        recursive["recursive"] = recursive
        for value in (None, [], {"bad": math.nan}, {"bad": object()}, {"bad": "\ud800"},
                      {1: "a", "1": "b"}, recursive, {"S3": "PRIVATE"}):
            with self.subTest(value=type(value).__name__), self.assertRaises(export.ExportError):
                export.encode_config_document(value)
        for key in (None, 123, b"bytes", "", "  ", "\ud800",
                    "a" * (export.MAX_OPEN_TEXT_BYTES + 1)):
            with self.subTest(key=type(key).__name__), self.assertRaises(export.ExportError):
                export.encode_config_document(self.values, key)


if __name__ == "__main__":
    unittest.main()
