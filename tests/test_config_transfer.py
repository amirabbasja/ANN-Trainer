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

from LRAEP_trainer.config import ConfigError
from LRAEP_trainer.config_export import FORMAT, decode_config_document, read_open_text
from LRAEP_trainer import config_transfer as transfer


class ConfigTransferTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.source = self.directory / 'trainer' / 'config.json'
        self.source.parent.mkdir()
        self.destination = self.directory / 'WR' / 'config.json'
        self.env_path = self.destination.parent / '.env'
        self.text = '  متن رمز "quotes" \'single\' $HOME #literal\nsecond line 🌍  '
        self.values = {
            'version': 1,
            'mode': 'solo',
            'official_source': 'src/LRAEP_helpers',
            'device': 'cuda:0',
            'training': {
                'main_effort': [[60, 80], [120, 70], [180, 10]],
                'decoy_effort': [],
                'output_dir': 'results',
                'max_batches': 2,
            },
            'pool': {
                'protocol': 'KEX',
                'KEX_uname': 'PRIVATE-KEX-IDENTITY',
                'WR_wallet': 'PRIVATE-WR-IDENTITY',
                'worker': 'test-rig',
                'password': 'PRIVATE-POOL-PASSWORD',
                'KEX': {'host': 'kex.example', 'port': 8048, 'tls': True},
                'WR': {'host': 'wr.example', 'port': 9000, 'tls': False,
                       'certificate_version': 3},
            },
            'node': {'username': 'PRIVATE-NODE-USER', 'password': 'PRIVATE-NODE-PASSWORD'},
            'vpn': {'enabled': True, 'link': 'vless://PRIVATE-VPN-ID@vpn.example:443',
                    'links_file': 'links.txt'},
        }
        self.save_source()

    def save_source(self):
        self.source.write_text(json.dumps(self.values, ensure_ascii=False, indent=2) + '\n')

    def exported_values(self):
        expected = copy.deepcopy(self.values)
        expected['mode'] = 'pool'
        expected['pool']['protocol'] = 'WR'
        return expected

    def isolated_wr_load(self, *, expected_success=True):
        wr_source = Path(__file__).resolve().parents[2] / 'LRAEP-WR-Trainer' / 'src'
        if not (wr_source / 'LRAEP_trainer' / 'config.py').is_file():
            self.skipTest('standalone WR checkout unavailable for interoperability test')
        script = '''
import json
from pathlib import Path
import sys
import LRAEP_trainer.config as config
assert Path(config.__file__).resolve().parent.parent == Path(sys.argv[2]).resolve()
try:
    raw = config.read_config(sys.argv[1])
    runtime = config.load_config(sys.argv[1])
    pool = config.resolve_pool_settings(runtime['pool'])
except config.ConfigError as error:
    print(str(error))
    raise SystemExit(1)
print(json.dumps({'raw': raw, 'runtime': runtime, 'pool': pool}, ensure_ascii=False))
'''
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(wr_source)
        environment['open_text'] = 'WRONG-PROCESS-TEXT'
        environment['OPEN_TEXT'] = 'WRONG-UPPERCASE-TEXT'
        environment['LRAEP_CONFIG_KEY'] = 'WRONG-LEGACY-KEY'
        result = subprocess.run([sys.executable, '-c', script, str(self.destination), str(wr_source)],
                                cwd=self.directory, env=environment, capture_output=True,
                                text=True, timeout=20)
        if expected_success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def test_plain_source_exports_to_private_encoded_wr_config_and_loads_in_wr_process(self):
        source_before = self.source.read_bytes()
        expected = self.exported_values()
        returned = transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        self.assertEqual(returned, self.destination)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(json.loads(self.source.read_text()), self.values)
        document = json.loads(self.destination.read_text())
        self.assertEqual(document['format'], FORMAT)
        self.assertEqual(set(document), {'format', 'salt', 'ciphertext'})
        self.assertEqual(decode_config_document(document, self.text), expected)
        self.assertEqual(read_open_text(self.env_path), self.text)
        for path in (self.destination, self.env_path):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination, self.env_path})
        for secret in ('PRIVATE-WR-IDENTITY', 'PRIVATE-POOL-PASSWORD', 'PRIVATE-VPN-ID',
                       'PRIVATE-NODE-PASSWORD', self.text, 'main_effort', 'WR_wallet'):
            self.assertNotIn(secret, self.destination.read_text())
        loaded = self.isolated_wr_load()
        self.assertEqual(loaded['raw'], expected)
        self.assertEqual(loaded['runtime']['mode'], 'pool')
        self.assertEqual(loaded['runtime']['training']['main_effort'], self.values['training']['main_effort'])
        self.assertEqual(loaded['runtime']['training']['decoy_effort'], [])
        self.assertEqual(loaded['pool']['protocol'], 'WR')
        self.assertEqual(loaded['pool']['host'], 'wr.example')
        self.assertEqual(loaded['pool']['port'], 9000)
        self.assertEqual(loaded['pool']['WR_wallet'], 'PRIVATE-WR-IDENTITY')
        self.assertEqual(loaded['runtime']['official_source'], str(self.destination.parent / 'src/LRAEP_helpers'))
        self.assertEqual(loaded['runtime']['training']['output_dir'], str(self.destination.parent / 'results'))
        self.assertEqual(loaded['raw']['official_source'], 'src/LRAEP_helpers')
        self.assertEqual(loaded['raw']['training']['output_dir'], 'results')

    def test_export_preserves_unrelated_env_assignments_and_comments(self):
        self.destination.parent.mkdir()
        self.env_path.write_text('# Keep comment\nUNRELATED="literal $HOME"\nopen_text="old"\nLAST=keep\n')
        transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        stored = self.env_path.read_text()
        self.assertTrue(stored.startswith('# Keep comment\nUNRELATED="literal $HOME"\n'))
        self.assertTrue(stored.endswith('LAST=keep\n'))
        self.assertEqual(read_open_text(self.env_path), self.text)
        self.assertNotIn('open_text="old"', stored)

    def test_save_key_false_writes_no_env_and_does_not_modify_an_existing_env(self):
        transfer.export_wr_config(self.source, self.destination, open_text=self.text, save_key=False)
        self.assertTrue(self.destination.is_file())
        self.assertFalse(self.env_path.exists())
        self.assertEqual(decode_config_document(json.loads(self.destination.read_text()), self.text),
                         self.exported_values())
        self.env_path.write_text('# old\nopen_text="existing"\n')
        before = self.env_path.read_bytes()
        transfer.export_wr_config(self.source, self.destination, open_text=self.text, save_key=False)
        self.assertEqual(self.env_path.read_bytes(), before)

    def test_repeated_export_is_fresh_and_does_not_change_authoring_config(self):
        source_before = self.source.read_bytes()
        transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        first = json.loads(self.destination.read_text())
        transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        second = json.loads(self.destination.read_text())
        self.assertNotEqual(first['salt'], second['salt'])
        self.assertNotEqual(first['ciphertext'], second['ciphertext'])
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(decode_config_document(second, self.text), self.exported_values())

    def test_same_source_and_destination_or_env_destination_are_rejected_before_writing(self):
        source_before = self.source.read_bytes()
        for destination in (self.source, self.source.parent / '.env'):
            with self.subTest(destination=destination.name), self.assertRaises(ConfigError):
                transfer.export_wr_config(self.source, destination, open_text=self.text)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertFalse((self.source.parent / '.env').exists())
        alias = self.source.parent / 'alias.json'
        alias.symlink_to(self.source)
        with self.assertRaises(ConfigError):
            transfer.export_wr_config(self.source, alias, open_text=self.text)
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_preflight_requires_wr_endpoint_identity_and_enabled_vpn_without_creating_output(self):
        cases = [
            ('WR host', lambda values: values['pool']['WR'].update(host='')),
            ('WR identity', lambda values: values['pool'].update(WR_wallet='')),
            ('VPN enabled', lambda values: values['vpn'].update(enabled=False)),
            ('VPN link', lambda values: values['vpn'].update(link='')),
        ]
        original = copy.deepcopy(self.values)
        for name, mutate in cases:
            with self.subTest(case=name):
                self.values = copy.deepcopy(original)
                mutate(self.values)
                self.save_source()
                source_before = self.source.read_bytes()
                with self.assertRaises((ConfigError, ValueError)) as caught:
                    transfer.export_wr_config(self.source, self.destination, open_text=self.text)
                self.assertEqual(self.source.read_bytes(), source_before)
                self.assertFalse(self.destination.exists())
                self.assertFalse(self.env_path.exists())
                self.assertNotIn('PRIVATE-', str(caught.exception))

    def test_invalid_key_and_save_key_flag_fail_before_output_or_env_changes(self):
        for arguments in ({'open_text': ''}, {'open_text': '  '},
                          {'open_text': self.text, 'save_key': 'yes'},
                          {'open_text': self.text, 'save_key': 1}):
            with self.subTest(arguments=type(arguments.get('save_key')).__name__):
                with self.assertRaises((ConfigError, ValueError)):
                    transfer.export_wr_config(self.source, self.destination, **arguments)
                self.assertFalse(self.destination.exists())
                self.assertFalse(self.env_path.exists())

    def test_env_write_failure_rolls_back_existing_export_exactly(self):
        self.destination.parent.mkdir()
        original = b'{\r\n  "existing": "unchanged"\r\n}\r\n'
        self.destination.write_bytes(original)
        self.env_path.write_text('OTHER=keep\nopen_text="original"\n')
        env_before = self.env_path.read_bytes()
        with patch.object(transfer, 'write_open_text', side_effect=ValueError('PRIVATE-FAILURE')):
            with self.assertRaisesRegex(ConfigError, 'rolled back') as caught:
                transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        self.assertNotIn('PRIVATE-FAILURE', str(caught.exception))
        self.assertEqual(self.destination.read_bytes(), original)
        self.assertEqual(self.env_path.read_bytes(), env_before)
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination, self.env_path})

    def test_env_encoding_failure_rolls_back_and_does_not_expose_passphrase(self):
        self.destination.parent.mkdir()
        original = b'previous encoded document\r\n'
        self.destination.write_bytes(original)
        source_before = self.source.read_bytes()
        error = UnicodeEncodeError('utf-8', self.text, 0, 1, 'PRIVATE-ENCODING-DETAILS')
        with patch.object(transfer, 'write_open_text', side_effect=error):
            with self.assertRaisesRegex(ConfigError, 'rolled back') as caught:
                transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        self.assertNotIn(self.text, str(caught.exception))
        self.assertNotIn('PRIVATE-ENCODING-DETAILS', str(caught.exception))
        self.assertEqual(self.destination.read_bytes(), original)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertFalse(self.env_path.exists())
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination})

    def test_env_write_failure_removes_a_new_export(self):
        source_before = self.source.read_bytes()
        with patch.object(transfer, 'write_open_text', side_effect=OSError('PRIVATE-FAILURE')):
            with self.assertRaisesRegex(ConfigError, 'rolled back'):
                transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.env_path.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_wrong_adjacent_file_text_fails_wr_load_without_environment_fallback(self):
        transfer.export_wr_config(self.source, self.destination, open_text=self.text)
        self.env_path.write_text('open_text="incorrect file value"\n')
        result = self.isolated_wr_load(expected_success=False)
        for secret in (self.text, 'incorrect file value', 'PRIVATE-POOL-PASSWORD'):
            self.assertNotIn(secret, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
