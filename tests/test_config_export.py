import base64
import copy
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from LRAEP_trainer import config_export as export


class ConfigExportTests(unittest.TestCase):
    def setUp(self):
        self.text = '  متن دلخواه "quoted" \'single\' $HOME #tag 🌍\nsecond line  '
        self.values = {
            'version': 1,
            'training': {'main_effort': [[60, 80], [30, 0], [0, -1]], 'decoy_effort': []},
            'pool': {'WR_wallet': 'PRIVATE-WALLET', 'password': 'PRIVATE-PASSWORD'},
            'relative_path': '../helpers', 'unicode': 'سلام', 'null': None,
            'enabled': True, 'fraction': 0.5,
        }

    def test_complete_raw_document_round_trips_without_mutation_or_plaintext_fields(self):
        original = copy.deepcopy(self.values)
        encoded = export.encode_config_document(self.values, self.text)
        self.assertEqual(encoded['format'], 'LRAEP-config-text-v1')
        self.assertEqual(set(encoded), {'format', 'salt', 'ciphertext'})
        self.assertEqual(len(base64.urlsafe_b64decode(encoded['salt'])), 16)
        for secret in ('PRIVATE-WALLET', 'PRIVATE-PASSWORD', 'WR_wallet', 'relative_path', self.text):
            self.assertNotIn(secret, json.dumps(encoded, ensure_ascii=False))
        self.assertEqual(export.decode_config_document(encoded, self.text), original)
        self.assertEqual(self.values, original)

    def test_every_export_has_fresh_salt_and_ciphertext(self):
        first = export.encode_config_document(self.values, self.text)
        second = export.encode_config_document(self.values, self.text)
        self.assertNotEqual(first['salt'], second['salt'])
        self.assertNotEqual(first['ciphertext'], second['ciphertext'])
        self.assertEqual(export.decode_config_document(second, self.text), self.values)

    def test_key_derivation_uses_the_declared_scrypt_parameters(self):
        encoded = export.encode_config_document(self.values, self.text)
        salt = base64.urlsafe_b64decode(encoded['salt'])
        key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(self.text.encode('utf-8'))
        decoded = Fernet(base64.urlsafe_b64encode(key)).decrypt(encoded['ciphertext'].encode('ascii'))
        self.assertEqual(json.loads(decoded), self.values)

    def test_whitespace_and_unicode_passphrases_are_used_exactly(self):
        encoded = export.encode_config_document(self.values, self.text)
        with self.assertRaises(export.ExportError):
            export.decode_config_document(encoded, self.text.strip())
        self.assertEqual(export.decode_config_document(encoded, self.text), self.values)

    def test_wrong_text_and_tampering_fail_without_echoing_values(self):
        encoded = export.encode_config_document(self.values, self.text)
        damaged = copy.deepcopy(encoded)
        token = bytearray(base64.urlsafe_b64decode(damaged['ciphertext']))
        token[-5] ^= 1
        damaged['ciphertext'] = base64.urlsafe_b64encode(token).decode('ascii')
        wrong_salt = dict(encoded, salt=base64.urlsafe_b64encode(bytes(16)).decode('ascii'))
        for document, text in ((encoded, 'WRONG-PRIVATE-TEXT'), (damaged, self.text),
                               (wrong_salt, self.text)):
            with self.subTest(text=text), self.assertRaises(export.ExportError) as caught:
                export.decode_config_document(document, text)
            for secret in (text, self.text, encoded['ciphertext'], 'PRIVATE-PASSWORD'):
                self.assertNotIn(secret, str(caught.exception))

    def test_invalid_envelopes_and_noncanonical_salts_are_rejected(self):
        encoded = export.encode_config_document(self.values, self.text)
        invalid = [
            None, [], {}, dict(encoded, extra='PRIVATE'), dict(encoded, format='PRIVATE'),
            dict(encoded, salt=None), dict(encoded, salt='!bad!'),
            dict(encoded, salt=base64.urlsafe_b64encode(b'short').decode('ascii')),
            dict(encoded, salt=encoded['salt'].rstrip('=')),
            dict(encoded, salt=encoded['salt'] + '\n'), dict(encoded, salt='秘密'),
            dict(encoded, ciphertext=None), dict(encoded, ciphertext='秘密'),
            dict(encoded, ciphertext='invalid token'),
        ]
        for value in invalid:
            with self.subTest(value=type(value).__name__), self.assertRaises(export.ExportError):
                export.decode_config_document(value, self.text)

    def test_export_rejects_invalid_documents_and_nonfinite_values(self):
        invalid = [None, [], 'PRIVATE', {'bad': math.nan}, {'bad': math.inf},
                   {'bad': object()}, {'bad': '\ud800'}, {1: 'a', '1': 'b'}]
        recursive = {}
        recursive['recursive'] = recursive
        invalid.append(recursive)
        for value in invalid:
            with self.subTest(value=type(value).__name__), self.assertRaises(export.ExportError) as caught:
                export.encode_config_document(value, self.text)
            self.assertNotIn('PRIVATE', str(caught.exception))

    def test_authenticated_json_still_rejects_duplicates_nonfinite_and_non_objects(self):
        salt = os.urandom(16)
        cipher = export._cipher(self.text, salt)
        for plaintext in (b'{"version":1,"version":2}', b'{"x":{"a":1,"a":2}}',
                          b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'[]', b'null', b'bad'):
            envelope = {'format': export.FORMAT,
                        'salt': base64.urlsafe_b64encode(salt).decode('ascii'),
                        'ciphertext': cipher.encrypt(plaintext).decode('ascii')}
            with self.subTest(plaintext=plaintext), self.assertRaises(export.ExportError):
                export.decode_config_document(envelope, self.text)

    def test_empty_invalid_or_oversized_text_is_rejected_without_truncation(self):
        for text in (None, 123, b'bytes', '', ' \t\n', '\u2003', '\ud800',
                     'a' * (export.MAX_OPEN_TEXT_BYTES + 1)):
            with self.subTest(value=type(text).__name__), self.assertRaises(export.ExportError):
                export.encode_config_document(self.values, text)
        maximum = 'a' * export.MAX_OPEN_TEXT_BYTES
        encoded = export.encode_config_document({}, maximum)
        self.assertEqual(export.decode_config_document(encoded, maximum), {})


class OpenTextFileTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.path = self.directory / '.env'

    def test_literal_values_round_trip_in_one_private_line(self):
        values = ['plain text', ' leading and trailing ', '秘密 🌍 # $HOME',
                  'a"b\'c\\d', 'one\ntwo\rthree\tfour', '\u2028one\u2029two',
                  '#starts-with-hash', '$(touch should-not-exist) `echo nope`']
        for value in values:
            with self.subTest(value=value):
                export.write_open_text(self.path, value)
                self.assertEqual(export.read_open_text(self.path), value)
                stored = self.path.read_text()
                self.assertEqual(stored.count('\n'), 1)
                self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
                self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_double_single_and_unquoted_values_with_optional_export_are_supported(self):
        cases = [
            ('open_text=plain value # trailing comment\n', 'plain value'),
            (' export open_text = "spaces $HOME # inside" # outside\n', 'spaces $HOME # inside'),
            ("open_text='single $HOME # inside' # outside\n", 'single $HOME # inside'),
            ('open_text=abc#literal\n', 'abc#literal'),
            ('open_text="line\\nnext"\n', 'line\nnext'),
            ('OTHER=ignored\nOPEN_TEXT=wrong\nopen_text=exact\n', 'exact'),
        ]
        for stored, expected in cases:
            with self.subTest(stored=stored):
                self.path.write_text(stored)
                self.assertEqual(export.read_open_text(self.path), expected)

    def test_only_the_given_file_is_used_even_if_process_environment_disagrees(self):
        self.path.write_text('open_text=file-specific\n')
        before = dict(os.environ)
        with patch.dict(os.environ, {'open_text': 'PROCESS-PRIVATE', 'OPEN_TEXT': 'WRONG-PRIVATE'}):
            self.assertEqual(export.read_open_text(self.path), 'file-specific')
            self.assertEqual(os.environ['open_text'], 'PROCESS-PRIVATE')
            missing = self.directory / 'other.env'
            with self.assertRaises(export.ExportError):
                export.read_open_text(missing)
            self.assertFalse(missing.exists())
        self.assertEqual(dict(os.environ), before)

    def test_shell_substitution_and_interpolation_are_never_executed(self):
        marker = self.directory / 'executed'
        value = f'$(touch {marker}) `touch {marker}` $HOME ${{PRIVATE_ENV}}'
        self.path.write_text('open_text=' + value + '\n')
        self.assertEqual(export.read_open_text(self.path), value)
        self.assertFalse(marker.exists())
        export.write_open_text(self.path, value)
        self.assertEqual(export.read_open_text(self.path), value)
        self.assertFalse(marker.exists())

    def test_writer_preserves_unrelated_assignments_comments_and_crlf(self):
        prefix = '# Keep this comment\r\nOTHER="keep $HOME # literal"\r\n'
        suffix = 'LAST=unchanged\r\n'
        self.path.write_bytes((prefix + ' export open_text = old # replaced\r\n' + suffix).encode())
        export.write_open_text(self.path, ' new "秘密" ')
        expected = prefix + 'open_text=' + json.dumps(' new "秘密" ', ensure_ascii=False) + '\r\n' + suffix
        self.assertEqual(self.path.read_bytes(), expected.encode('utf-8'))
        self.assertEqual(export.read_open_text(self.path), ' new "秘密" ')

    def test_writer_appends_missing_key_without_dropping_existing_last_line(self):
        self.path.write_text('# comment\nOTHER=original')
        export.write_open_text(self.path, 'new text')
        self.assertEqual(self.path.read_text(), '# comment\nOTHER=original\nopen_text="new text"\n')
        self.assertEqual(export.read_open_text(self.path), 'new text')

    def test_missing_duplicate_malformed_and_empty_values_fail_safely(self):
        invalid = [
            '', 'OTHER=value\n', 'OPEN_TEXT=uppercase\n', 'open_text\n',
            'export open_text wrong\n', 'open_text="unterminated\n',
            "open_text='unterminated\n", 'open_text="value" extra\n',
            "open_text='value' extra\n", 'open_text=\n', 'open_text="  "\n',
            'open_text=PRIVATE-one\nopen_text=PRIVATE-two\n',
            'open_text=PRIVATE-one\nexport open_text=PRIVATE-two\n',
        ]
        for stored in invalid:
            with self.subTest(stored=stored):
                self.path.write_text(stored)
                with self.assertRaises(export.ExportError) as caught:
                    export.read_open_text(self.path)
                self.assertNotIn('PRIVATE', str(caught.exception))

    def test_invalid_existing_assignment_does_not_get_silently_overwritten(self):
        for stored in ('open_text=one\nopen_text=two\n', 'open_text="unterminated\n'):
            self.path.write_text(stored)
            with self.assertRaises(export.ExportError):
                export.write_open_text(self.path, 'replacement')
            self.assertEqual(self.path.read_text(), stored)

    def test_atomic_write_failure_preserves_existing_file_and_removes_tempfile(self):
        self.path.write_text('OTHER=keep\nopen_text="original"\n')
        original = self.path.read_bytes()
        with patch.object(export.os, 'replace', side_effect=OSError('PRIVATE-SYSTEM-DETAILS')):
            with self.assertRaises(export.ExportError) as caught:
                export.write_open_text(self.path, 'new text')
        self.assertNotIn('PRIVATE-SYSTEM-DETAILS', str(caught.exception))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_invalid_text_and_path_do_not_modify_files_or_environment(self):
        self.path.write_text('OTHER=keep\n')
        before = self.path.read_bytes()
        environment = dict(os.environ)
        for text in ('', '  \n', None, '\ud800', 'x' * (export.MAX_OPEN_TEXT_BYTES + 1)):
            with self.assertRaises(export.ExportError):
                export.write_open_text(self.path, text)
        with self.assertRaises(export.ExportError):
            export.read_open_text(None)
        with self.assertRaises(export.ExportError):
            export.write_open_text(None, 'valid text')
        with self.assertRaises(export.ExportError):
            export.read_open_text('invalid\x00path')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(dict(os.environ), environment)


if __name__ == '__main__':
    unittest.main()
