import copy
from contextlib import chdir
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config, read_config, save_config, update_config
from LRAEP_trainer.config_export import FORMAT, encode_config_document, write_open_text
from LRAEP_trainer.paths import default_config_path


class ConfigDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / 'config.json'
        self.text = 'test arbitrary text key Ω # $ literal'
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config['pool'].update(host='pool.example', WR_wallet='fixture-wallet')
        self.config['vpn']['link'] = 'vless://fixture@example.invalid:443'
        write_open_text(self.directory / '.env', self.text)

    def write_envelope(self, key=None):
        envelope = encode_config_document(self.config, self.text if key is None else key)
        self.path.write_text(json.dumps(envelope), encoding='utf-8')
        return envelope

    def test_encoded_json_roundtrip_keeps_authoring_fields_encrypted(self):
        save_config(self.path, self.config)
        envelope = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(envelope['format'], FORMAT)
        self.assertEqual(set(envelope), {'format', 'salt', 'ciphertext'})
        self.assertNotIn('fixture-wallet', self.path.read_text(encoding='utf-8'))
        self.assertEqual(read_config(self.path), self.config)
        self.assertEqual(load_config(self.path)['training']['main_effort'], [(60, 80), (120, 70), (180, 10)])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.directory / 'config.key').exists())

    def test_shared_export_can_be_opened_by_wr_runtime(self):
        source = Path(__file__).resolve().parents[2] / 'src/LRAEP_trainer/config_export.py'
        spec = importlib.util.spec_from_file_location('shared_text_export_fixture', source)
        shared_export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shared_export)
        document = shared_export.encode_config_document(self.config, self.text)
        self.path.write_text(json.dumps(document), encoding='utf-8')
        self.assertEqual(read_config(self.path), self.config)

    def test_adjacent_dotenv_wins_over_cwd_and_process_environment(self):
        self.write_envelope()
        other = self.directory / 'other'
        other.mkdir()
        write_open_text(other / '.env', 'wrong cwd key')
        with chdir(other), patch.dict(os.environ, {
            'open_text': 'wrong process key', 'OPEN_TEXT': 'wrong upper-case key',
            'LRAEP_CONFIG_KEY': 'wrong legacy key',
        }):
            self.assertEqual(load_config(self.path)['pool']['WR_wallet'], 'fixture-wallet')

    def test_missing_dotenv_does_not_fall_back_to_process_or_config_key(self):
        self.write_envelope()
        (self.directory / '.env').unlink()
        (self.directory / 'config.key').write_text(self.text, encoding='utf-8')
        with patch.dict(os.environ, {
            'open_text': self.text, 'OPEN_TEXT': self.text, 'LRAEP_CONFIG_KEY': self.text,
        }), self.assertRaisesRegex(ConfigError, 'open_text') as caught:
            load_config(self.path)
        self.assertNotIn(self.text, str(caught.exception))

    def test_wrong_case_dotenv_assignment_is_rejected(self):
        self.write_envelope()
        (self.directory / '.env').write_text('OPEN_TEXT="' + self.text + '"\n', encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'open_text'):
            load_config(self.path)

    def test_explicit_api_key_can_override_missing_dotenv(self):
        self.write_envelope()
        (self.directory / '.env').unlink()
        self.assertEqual(read_config(self.path, key=self.text), self.config)
        save_config(self.path, self.config, key=self.text)
        self.assertEqual(read_config(self.path, key=self.text), self.config)
        self.assertFalse((self.directory / '.env').exists())
        self.assertFalse((self.directory / 'config.key').exists())

    def test_wrong_key_and_tampered_ciphertext_fail_without_echoing_secrets(self):
        envelope = self.write_envelope()
        write_open_text(self.directory / '.env', 'PRIVATE_WRONG_KEY')
        with self.assertRaisesRegex(ConfigError, 'decode') as caught:
            load_config(self.path)
        for value in (self.text, 'PRIVATE_WRONG_KEY', 'fixture-wallet', envelope['ciphertext']):
            self.assertNotIn(value, str(caught.exception))
        write_open_text(self.directory / '.env', self.text)
        envelope['ciphertext'] = envelope['ciphertext'][:-8] + 'AAAAAAAA'
        self.path.write_text(json.dumps(envelope), encoding='utf-8')
        with self.assertRaisesRegex(ConfigError, 'decode'):
            load_config(self.path)

    def test_plain_and_legacy_token_documents_are_not_accepted(self):
        for document in (json.dumps(self.config), 'gAAAAA_PRIVATE_LEGACY_TOKEN'):
            with self.subTest(document=document[:1]):
                self.path.write_text(document, encoding='utf-8')
                with self.assertRaises(ConfigError) as caught:
                    load_config(self.path)
                self.assertNotIn('PRIVATE_LEGACY_TOKEN', str(caught.exception))
                self.assertNotIn('fixture-wallet', str(caught.exception))

    def test_invalid_envelope_errors_do_not_include_untrusted_values(self):
        document = {'format': 'PRIVATE_FORMAT', 'salt': 'PRIVATE_SALT', 'ciphertext': 'PRIVATE_CIPHERTEXT'}
        self.path.write_text(json.dumps(document), encoding='utf-8')
        with self.assertRaises(ConfigError) as caught:
            load_config(self.path)
        for value in document.values():
            self.assertNotIn(value, str(caught.exception))

    def test_update_merges_fields_and_preserves_adjacent_dotenv(self):
        save_config(self.path, self.config)
        dotenv_before = (self.directory / '.env').read_bytes()
        update_config(self.path, {'training': {'main_effort': [[5, 50], [0, -1]]}})
        updated = read_config(self.path)
        self.assertEqual(updated['training']['main_effort'], [[5, 50], [0, -1]])
        self.assertEqual(updated['pool'], self.config['pool'])
        self.assertEqual(updated['vpn'], self.config['vpn'])
        self.assertEqual((self.directory / '.env').read_bytes(), dotenv_before)
        self.assertEqual({entry.name for entry in self.directory.iterdir()}, {'.env', 'config.json'})

    def test_save_requires_existing_dotenv_or_explicit_text_key(self):
        (self.directory / '.env').unlink()
        with self.assertRaisesRegex(ConfigError, 'open_text'):
            save_config(self.path, self.config)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_default_filename_is_config_json(self):
        self.assertEqual(default_config_path().name, 'config.json')


if __name__ == '__main__':
    unittest.main()
