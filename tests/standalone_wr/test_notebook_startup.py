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

from workrunner.config import DEFAULT_CONFIG, save_config


ROOT = Path(__file__).resolve().parents[3] / "LRAEP-WR-Trainer"


class NotebookStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / ROOT.name
        self.root.mkdir()
        shutil.copy2(ROOT / "train.py", self.root / "train.py")
        shutil.copy2(ROOT / "requirements.txt", self.root / "requirements.txt")
        (self.root / "src").symlink_to(ROOT / "src", target_is_directory=True)
        self.notebook = json.loads((ROOT / "train.ipynb").read_text())
        self.path = self.root / "config.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["vpn"]["link"] = "vless://fixture@example.invalid:443"
        self.config["training"]["job_sequence"] = [{"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]
        save_config(self.path, self.config)
        shared = self.directory / "trainer"
        shared.mkdir()
        (shared / "config.json").write_text('{"invalid_shared_config":true}')
        self.namespace = {"__name__": "__wr_notebook_test__"}
        original_path = sys.path[:]
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))

    def execute_notebook(self, *, setup=True):
        cells = self.notebook["cells"] if setup else self.notebook["cells"][-1:]
        output = io.StringIO()
        with chdir(self.directory), redirect_stdout(output), patch.dict(os.environ, {
            "open_text": "wrong", "LRAEP_CONFIG_KEY": "wrong",
        }), patch("subprocess.check_call", side_effect=AssertionError("unexpected installation")):
            for index, cell in enumerate(cells):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"train.ipynb:{index}", "exec"), self.namespace)
        return self.namespace["exit_code"], output.getvalue()

    def test_notebook_reads_its_own_partially_encrypted_config_without_dotenv(self):
        before = self.path.read_bytes()
        code, output = self.execute_notebook()
        self.assertEqual(code, 0, output)
        self.assertEqual(self.namespace["ROOT"], self.root)
        self.assertEqual(self.path.read_bytes(), before)
        document = json.loads(before)
        self.assertIn("S_3", document)
        self.assertEqual(document["training"]["job_sequence"], [{"title": 'job-0', "time": 0, "max_gpu_utilization": -1}])
        self.assertFalse((self.root / ".env").exists())

    def test_restart_uses_qwe_and_does_not_read_a_dotenv_override(self):
        self.assertEqual(self.execute_notebook()[0], 0)
        env = self.root / ".env"
        env.write_text("open_text=custom key\n")
        save_config(self.path, self.config, key="custom key")
        code, output = self.execute_notebook(setup=False)
        self.assertEqual(code, 1, output)
        self.assertIn("could not decrypt", output)
        self.assertNotIn("custom key", output)
        save_config(self.path, self.config)
        self.assertEqual(self.execute_notebook(setup=False)[0], 0)
        self.assertEqual(env.read_text(), "open_text=custom key\n")

    def test_restart_validates_edits_to_plain_schedule(self):
        self.assertEqual(self.execute_notebook()[0], 0)
        document = json.loads(self.path.read_text())
        document["training"]["job_sequence"] = [{"title": 'job-0', "time": 0, "max_gpu_utilization": 100}]
        self.path.write_text(json.dumps(document))
        code, output = self.execute_notebook(setup=False)
        self.assertEqual(code, 1, output)
        self.assertIn("training.job_sequence", output)


if __name__ == "__main__":
    unittest.main()
