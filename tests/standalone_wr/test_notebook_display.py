import copy
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import time
import unittest

from LRAEP_trainer.config import DEFAULT_CONFIG, save_config
from LRAEP_trainer.config_export import write_open_text


ROOT = Path(__file__).resolve().parents[3] / 'LRAEP-WR-Trainer'


@unittest.skipUnless(importlib.util.find_spec('jupyter_client') and importlib.util.find_spec('ipykernel'),
                     'requires the notebook runtime')
class NotebookDisplayTests(unittest.TestCase):
    def exercise_cell(self, *, interrupt):
        from jupyter_client import KernelManager

        notebook = json.loads((ROOT / 'train.ipynb').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'src').symlink_to(ROOT / 'src', target_is_directory=True)
            go = directory / 'go'
            config = copy.deepcopy(DEFAULT_CONFIG)
            config['pool'].update(protocol='WR', host='fixture.example', WR_wallet='fixture-wallet')
            config['vpn'].update(enabled=True, link='vless://fixture@example.invalid:443')
            config['side_job']['repo_url'] = 'https://huggingface.co/datasets/fixture/thesis'
            config['training']['job_sequence'] = [['side-job', 2, 100], ['main', 1, 100]]
            write_open_text(directory / '.env', 'notebook kernel fixture key')
            save_config(directory / 'config.json', config)
            manager = KernelManager(connection_file=str(directory / 'kernel.json'))
            manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            manager.start_kernel(cwd=temporary)
            client = manager.client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=20)
                source = (
                    'import threading, time\n'
                    'from pathlib import Path\n'
                    'from unittest.mock import patch\n'
                    f"exec({''.join(notebook['cells'][0]['source'])!r}, globals())\n"
                    'from LRAEP_trainer.pool_training import _signal_stop\n'
                    'def exercise(configuration):\n'
                    "    assert configuration['training']['job_sequence'] == [('side-job', 2, 100), ('main', 1, 100)]\n"
                    '    stopped = threading.Event()\n'
                    '    with _signal_stop(stopped):\n'
                    "        print('Side-job | Epoch 1 | Step 1/3', flush=True)\n"
                    '        for _ in range(3000):\n'
                    f'            if stopped.is_set() or Path({str(go)!r}).exists(): break\n'
                    '            time.sleep(0.01)\n'
                    '        if not stopped.is_set():\n'
                    "            print('Side-job | Epoch 1 | Step 2/3 (66.7%) | GPU 80%', flush=True)\n"
                    "            print('Fixture epoch complete', flush=True)\n"
                    '    return 0\n'
                    "with patch('LRAEP_trainer.vpn.ManagedVPN') as vpn, patch('LRAEP_trainer.network.routed') as route, patch('LRAEP_trainer.pool_training.run_pool_training', side_effect=exercise):\n"
                    f"    exec({''.join(notebook['cells'][-1]['source'])!r}, globals())\n"
                    '    assert vpn.return_value.__exit__.called\n'
                    '    assert route.return_value.__exit__.called\n'
                    "    print('Fixture resources closed', flush=True)\n"
                )
                message_id = client.execute(source)
                chunks = []
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        message = client.get_iopub_msg(timeout=1)
                    except queue.Empty:
                        continue
                    if message.get('parent_header', {}).get('msg_id') != message_id:
                        continue
                    kind = message['msg_type']
                    if kind == 'error':
                        self.fail('\n'.join(message['content']['traceback']))
                    self.assertNotIn(kind, ('display_data', 'update_display_data'))
                    if kind == 'stream':
                        chunks.append(message['content']['text'])
                        if 'Step 1/3' in ''.join(chunks):
                            break
                    if kind == 'status' and message['content']['execution_state'] == 'idle':
                        self.fail('training cell returned before its direct loop completed')
                else:
                    self.fail('direct progress did not reach the notebook output stream')
                if interrupt:
                    manager.interrupt_kernel()
                else:
                    go.write_text('go')
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        message = client.get_iopub_msg(timeout=1)
                    except queue.Empty:
                        continue
                    if message.get('parent_header', {}).get('msg_id') != message_id:
                        continue
                    kind = message['msg_type']
                    if kind == 'error':
                        self.fail('\n'.join(message['content']['traceback']))
                    self.assertNotIn(kind, ('display_data', 'update_display_data'))
                    if kind == 'stream':
                        chunks.append(message['content']['text'])
                    if kind == 'status' and message['content']['execution_state'] == 'idle':
                        break
                else:
                    self.fail('training cell did not finish cleaning up after stopping')
                output = ''.join(chunks)
                self.assertIn('Training stopped (exit code 0).', output)
                self.assertIn('Fixture resources closed', output)
                if not interrupt:
                    self.assertIn('Step 2/3', output)
                    self.assertIn('Fixture epoch complete', output)
            finally:
                go.touch()
                manager.shutdown_kernel(now=True)
                client.stop_channels()

    def test_real_kernel_prints_direct_progress_while_the_cell_is_busy(self):
        self.exercise_cell(interrupt=False)

    def test_interrupting_direct_training_closes_its_resources(self):
        self.exercise_cell(interrupt=True)


if __name__ == '__main__':
    unittest.main()
