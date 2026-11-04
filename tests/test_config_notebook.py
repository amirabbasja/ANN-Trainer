import copy
from contextlib import redirect_stdout
import io
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from LRAEP_trainer import config, config_transfer
from LRAEP_trainer.config_export import decode_config_document


class ConfigNotebookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        standalone_source = self.root.parent / "LRAEP-WR-Trainer" / "src"
        with patch.object(sys, "path", [str(standalone_source), *sys.path]):
            self.standalone_config = importlib.import_module("workrunner.config")
        self.notebook = json.loads((self.root / "config_manager.ipynb").read_text())
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "trainer" / "config.json"
        self.source.parent.mkdir()
        self.destination = self.directory / "LRAEP-WR-Trainer" / "config.json"
        self.destination.parent.mkdir()
        self.values = {
            "version": 1, "mode": "S3", "training": {"output_dir": "results"},
            "S3": {"protocol": "WR", "WR_S2": "PRIVATE-IDENTITY", "password": "PRIVATE-PASSWORD",
                   "WR": {"host": "wr.example", "port": 9000, "tls": False}},
            "vpn": {"enabled": True, "link": "vless://fixture@example.invalid:443"},
            "side_job": {"repo_url": "https://huggingface.co/datasets/fixture/thesis"},
        }

    def execute_cells(self):
        namespace = {"__name__": "__config_notebook__"}
        with patch.object(Path, "cwd", return_value=self.root):
            for index, cell in enumerate(self.notebook["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"config_manager.ipynb:{index}", "exec"), namespace)
        return namespace

    def test_running_all_cells_does_not_read_write_export_or_prompt(self):
        output = io.StringIO()
        with patch.object(self.standalone_config, "read_config", side_effect=AssertionError("read")) as read, \
                patch.object(self.standalone_config, "save_config", side_effect=AssertionError("save")) as save, \
                patch.object(self.standalone_config, "update_config", side_effect=AssertionError("update")) as update, \
                patch.object(config_transfer, "export_wr_config", side_effect=AssertionError("export")) as export, \
                patch("getpass.getpass", side_effect=AssertionError("prompt")) as prompt, redirect_stdout(output):
            namespace = self.execute_cells()
        for operation in (read, save, update, export, prompt):
            operation.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        expected = self.root.parent / "LRAEP-WR-Trainer" / "config.json"
        self.assertEqual(namespace["CONFIG_PATH"], expected)
        self.assertEqual(namespace["WR_CONFIG_PATH"], expected)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_replace_and_update_save_directly_to_standalone_with_default_key(self):
        namespace = self.execute_cells()
        self.source.write_text('{"unchanged":true}')
        before = self.source.read_bytes()
        namespace["CONFIG_PATH"] = self.destination
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["replace_from_dict"](self.values)
            namespace["update_from_dict"]({"S3": {"worker": "device"}, "training": {"max_batches": 4}})
        expected = copy.deepcopy(self.values)
        expected["S3"]["worker"] = "device"
        expected["training"]["max_batches"] = 4
        document = json.loads(self.destination.read_text())
        self.assertEqual(decode_config_document(document), expected)
        self.assertEqual(document["training"]["max_batches"], 4)
        self.assertIn("S_3", document)
        self.assertNotIn("PRIVATE-PASSWORD", output.getvalue())
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(set(self.destination.parent.iterdir()), {self.destination})

    def test_invalid_standalone_settings_do_not_replace_its_config(self):
        namespace = self.execute_cells()
        namespace["CONFIG_PATH"] = self.destination
        self.standalone_config.save_config(self.destination, self.values)
        before = self.destination.read_bytes()
        with self.assertRaisesRegex(self.standalone_config.ConfigError, "VPN must be enabled"):
            namespace["update_from_dict"]({"vpn": {"enabled": False}})
        self.assertEqual(self.destination.read_bytes(), before)

    def test_explicit_import_uses_shared_source_and_does_not_create_dotenv(self):
        namespace = self.execute_cells()
        config.save_config(self.source, self.values)
        before = self.source.read_bytes()
        namespace["TRAINER_CONFIG_PATH"] = self.source
        output = io.StringIO()
        with patch("getpass.getpass", side_effect=AssertionError("unexpected prompt")), redirect_stdout(output):
            result = namespace["export_wr_config"](output_path=self.destination)
        self.assertEqual(result, self.destination)
        self.assertEqual(config.read_config(self.destination), self.values)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(output.getvalue(), f"WR configuration saved: {self.destination}\n")
        self.assertFalse((self.destination.parent / ".env").exists())


if __name__ == "__main__":
    unittest.main()
