"""Execute notebook cells safely and exercise an explicit local WR export."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from LRAEP_trainer import config, config_transfer
from LRAEP_trainer.config_export import decode_config_document, read_open_text


class ConfigNotebookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.notebook = json.loads((self.root / "config_manager.ipynb").read_text())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "trainer" / "config.json"
        self.source.parent.mkdir()
        self.destination = self.directory / "WR" / "config.json"
        self.settings = {
            "version": 1,
            "mode": "solo",
            "official_source": "src/LRAEP_helpers",
            "training": {
                "output_dir": "results",
                "job_sequence": [["main", 5, 80], ["decoy", 1, 0], ["main", 0, -1]],
            },
            "pool": {
                "protocol": "KEX", "WR_wallet": "test-wr-wallet",
                "password": "PRIVATE-SOURCE-PASSWORD",
                "KEX": {"host": "kex.example", "port": 8048, "tls": True},
                "WR": {"host": "wr.example", "port": 9000, "tls": False},
            },
            "vpn": {"enabled": True, "link": "vless://test-id@vpn.example:443"},
        }

    def execute_cells(self):
        namespace = {"__name__": "__config_notebook__"}
        with patch.object(Path, "cwd", return_value=self.root):
            for index, cell in enumerate(self.notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    exec(compile(source, f"config_manager.ipynb:cell-{index}", "exec"), namespace)
        return namespace

    def test_running_all_cells_never_reads_writes_exports_or_prompts(self):
        output = io.StringIO()
        with patch.object(config, "read_config", side_effect=AssertionError("unexpected read")) as read, \
                patch.object(config, "save_config", side_effect=AssertionError("unexpected save")) as save, \
                patch.object(config, "update_config", side_effect=AssertionError("unexpected update")) as update, \
                patch.object(config_transfer, "export_wr_config", side_effect=AssertionError("unexpected export")) as export, \
                patch("getpass.getpass", side_effect=AssertionError("unexpected key prompt")) as prompt, \
                patch.object(Path, "read_text", side_effect=AssertionError("unexpected file read")), \
                patch.object(Path, "write_text", side_effect=AssertionError("unexpected file write")), \
                redirect_stdout(output):
            namespace = self.execute_cells()
        for operation in (read, save, update, export, prompt):
            operation.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertNotIn("decode_key", namespace)
        self.assertNotIn("open_text", namespace)
        self.assertEqual(namespace["CONFIG_PATH"], self.root / "config.json")
        self.assertEqual(namespace["WR_CONFIG_PATH"], self.root.parent / "LRAEP-WR-Trainer" / "config.json")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_explicit_prompted_export_preserves_plain_source_and_exact_text(self):
        namespace = self.execute_cells()
        config.save_config(self.source, self.settings)
        before = self.source.read_bytes()
        self.destination.parent.mkdir()
        env_path = self.destination.parent / ".env"
        env_path.write_text('KEEP_SETTING="preserve exactly"\n')
        namespace["CONFIG_PATH"] = self.source
        key_text = '  رمز notebook "$value" #\n\'word  '
        prompt = Mock(return_value=key_text)
        output = io.StringIO()
        with patch.dict(namespace, {"getpass": prompt}), redirect_stdout(output):
            result = namespace["export_wr_config"](output_path=self.destination)
        prompt.assert_called_once_with("Decode key text: ")
        self.assertEqual(result, self.destination)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(json.loads(self.source.read_text()), self.settings)
        envelope = json.loads(self.destination.read_text())
        expected = copy.deepcopy(self.settings)
        expected["mode"] = "pool"
        expected["pool"]["protocol"] = "WR"
        self.assertEqual(decode_config_document(envelope, key_text), expected)
        self.assertEqual(read_open_text(env_path), key_text)
        self.assertTrue(env_path.read_text().startswith('KEEP_SETTING="preserve exactly"\n'))
        self.assertEqual(output.getvalue(),
                         f"Encoded WR configuration: {self.destination}\n"
                         f"Decode key saved in: {env_path}\n")
        self.assertNotIn("PRIVATE-SOURCE-PASSWORD", output.getvalue())
        self.assertNotIn(key_text, output.getvalue())
        self.assertNotIn("decode_key", namespace)
        self.assertNotIn("open_text", namespace)

    def test_plain_edit_helpers_do_not_create_an_export(self):
        namespace = self.execute_cells()
        config.save_config(self.source, self.settings)
        namespace["CONFIG_PATH"] = self.source
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["update_from_dict"]({"pool": {"worker": "updated-worker"}})
        expected = copy.deepcopy(self.settings)
        expected["pool"]["worker"] = "updated-worker"
        self.assertEqual(json.loads(self.source.read_text()), expected)
        with redirect_stdout(output):
            namespace["replace_from_dict"](self.settings)
        self.assertEqual(json.loads(self.source.read_text()), self.settings)
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.source.parent / ".env").exists())
        self.assertNotIn("PRIVATE-SOURCE-PASSWORD", output.getvalue())

    def test_cancelled_key_prompt_does_not_write_either_config(self):
        namespace = self.execute_cells()
        config.save_config(self.source, self.settings)
        before = self.source.read_bytes()
        namespace["CONFIG_PATH"] = self.source
        with patch.dict(namespace, {"getpass": Mock(side_effect=KeyboardInterrupt)}), \
                self.assertRaises(KeyboardInterrupt):
            namespace["export_wr_config"](output_path=self.destination)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse(self.destination.exists())
        self.assertFalse((self.destination.parent / ".env").exists())


if __name__ == "__main__":
    unittest.main()
