import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from workrunner.config import ConfigError, DEFAULT_CONFIG, load_config, read_config, save_config, update_config
from workrunner.paths import default_config_path


class ConfigDecodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "config.json"
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config["vpn"]["link"] = "vless://fixture@example.invalid:443"

    def test_shared_codec_and_standalone_use_the_same_default_key(self):
        source = Path(__file__).resolve().parents[2] / "src/LRAEP_trainer/config_export.py"
        spec = importlib.util.spec_from_file_location("shared_export_fixture", source)
        shared = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(shared)
        self.path.write_text(json.dumps(shared.encode_config_document(self.config)))
        self.assertEqual(read_config(self.path), self.config)

    def test_cnn_repository_and_vpn_are_readable_while_s3_is_encrypted(self):
        self.config["S3"]["WR_S2"] = "PRIVATE-IDENTITY"
        self.config["side_job"]["repo_url"] = "https://huggingface.co/datasets/fixture/thesis"
        self.config["side_job"]["vpn"].update(enabled=True, link="ss://fixture", core_path="bin/xray")
        save_config(self.path, self.config)
        stored = json.loads(self.path.read_text())
        self.assertEqual(stored["side_job"], self.config["side_job"])
        self.assertNotIn("PRIVATE-IDENTITY", self.path.read_text())
        self.assertEqual(read_config(self.path), self.config)
        self.assertEqual(load_config(self.path)["side_job"]["vpn"]["core_path"], str(self.directory / "bin/xray"))
        self.assertFalse((self.directory / ".env").exists())

    def test_cnn_schedule_requires_explicit_repository_with_safe_errors(self):
        self.config["training"]["job_sequence"] = [{"title": 'job-1', "time": 60, "max_gpu_utilization": 100}]
        for repo in ("", "https://PRIVATE.invalid/SECRET"):
            self.config["side_job"]["repo_url"] = repo
            with self.assertRaisesRegex(ConfigError, "side_job.repo_url") as caught:
                save_config(self.path, self.config)
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertNotIn("SECRET", str(caught.exception))
        self.assertFalse(self.path.exists())

    def test_schedules_update_without_creating_dotenv_or_changing_connection_settings(self):
        save_config(self.path, self.config)
        update_config(self.path, {"training": {"job_sequence": [{"title": 'job-0', "time": 5, "max_gpu_utilization": 50}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]}})
        updated = read_config(self.path)
        self.assertEqual(updated["S3"], self.config["S3"])
        self.assertEqual(updated["training"]["job_sequence"], [{"title": 'job-0', "time": 5, "max_gpu_utilization": 50}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}])
        self.assertEqual(set(self.directory.iterdir()), {self.path})

    def test_default_filename_is_config_json(self):
        self.assertEqual(default_config_path().name, "config.json")


if __name__ == "__main__":
    unittest.main()
