import copy
from contextlib import chdir, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer.config import DEFAULT_CONFIG, save_config
from LRAEP_trainer.config_export import encode_config_document, write_open_text


ROOT = Path(__file__).resolve().parents[3] / 'LRAEP-WR-Trainer'


class NotebookStartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / ROOT.name
        self.root.mkdir()
        shutil.copy2(ROOT / 'train.py', self.root / 'train.py')
        shutil.copy2(ROOT / 'requirements.txt', self.root / 'requirements.txt')
        (self.root / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
        self.notebook = json.loads((ROOT / 'train.ipynb').read_text())
        self.key = 'notebook fixture key Ω'
        self.config_path = self.root / 'config.json'
        write_open_text(self.root / '.env', self.key)
        write_open_text(self.directory / '.env', 'wrong working-directory key')
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config['vpn']['link'] = 'vless://fixture@example.invalid:443'
        self.config['training']['job_sequence'] = [['side-job', 0, -1]]
        save_config(self.config_path, self.config)
        shared = self.directory / 'trainer'
        shared.mkdir()
        (shared / 'config.json').write_text(json.dumps({
            'training': {'job_sequence': [['main', 60, 100]]},
            'private_source_marker': 'SHARED-SOURCE-MUST-NOT-BE-READ',
        }))
        self.namespace = {'__name__': '__wr_notebook_test__'}
        self.addCleanup(self.stop_worker)
        self.original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.original_path))

    def stop_worker(self):
        trainer = self.namespace.get('trainer')
        if trainer is not None:
            trainer.stop_training()

    def execute_notebook(self, *, setup=True):
        cells = self.notebook['cells'] if setup else self.notebook['cells'][-1:]
        output = io.StringIO()
        with chdir(self.directory), redirect_stdout(output), patch.dict(os.environ, {
                'open_text': 'wrong process key', 'OPEN_TEXT': 'wrong uppercase key',
                'LRAEP_CONFIG_KEY': 'wrong legacy key',
                'PYTHONPATH': str(self.root / 'src'),
        }), patch('subprocess.check_call', side_effect=AssertionError('unexpected dependency installation')), \
                patch('logging.basicConfig'):
            for index, cell in enumerate(cells):
                self.assertEqual(cell['cell_type'], 'code')
                source = ''.join(cell['source'])
                exec(compile(source, f'train.ipynb:cell-{index}', 'exec'), self.namespace)
            code = self.namespace['exit_code']
        self.assertNotIn(self.key, output.getvalue())
        self.assertNotIn('SHARED-SOURCE-MUST-NOT-BE-READ', output.getvalue())
        return code, output.getvalue()

    def test_notebook_decrypts_its_own_config_without_the_launcher_file(self):
        (self.root / 'train.py').unlink()
        key_before = (self.root / '.env').read_bytes()
        config_before = self.config_path.read_bytes()
        code, output = self.execute_notebook()
        self.assertEqual(code, 0, output)
        self.assertEqual(self.namespace['ROOT'], self.root)
        self.assertIn('Training started', output)
        self.assertIn('Training stopped (exit code 0).', output)
        self.assertNotIn('worker_code', self.namespace)
        self.assertNotIn('process', self.namespace)
        self.assertNotIn('VPN enabled', output)
        self.assertNotIn('GPU:', output)
        self.assertEqual(self.config_path.read_bytes(), config_before)
        self.assertEqual((self.root / '.env').read_bytes(), key_before)
        self.assertEqual(set(json.loads(config_before)), {'format', 'salt', 'ciphertext'})

    def test_training_cell_restart_reads_the_current_adjacent_key(self):
        self.assertEqual(self.execute_notebook()[0], 0)
        write_open_text(self.root / '.env', 'different fixture key')
        code, output = self.execute_notebook(setup=False)
        self.assertEqual(code, 1, output)
        self.assertIn('could not decode', output)
        self.assertNotIn('different fixture key', output)
        self.assertNotIn('VPN enabled', output)
        write_open_text(self.root / '.env', self.key)
        self.assertEqual(self.execute_notebook(setup=False)[0], 0)

    def test_training_cell_restart_validates_the_new_decoded_sequence(self):
        self.assertEqual(self.execute_notebook()[0], 0)
        updated = copy.deepcopy(self.config)
        updated['training']['job_sequence'] = [['main', 0, 100]]
        self.config_path.write_text(json.dumps(encode_config_document(updated, self.key)))
        code, output = self.execute_notebook(setup=False)
        self.assertEqual(code, 1, output)
        self.assertIn('training.job_sequence', output)
        self.assertNotIn('VPN enabled', output)


if __name__ == '__main__':
    unittest.main()
