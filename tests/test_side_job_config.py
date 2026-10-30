"""CNN config defaults, independent routes and validation without network I/O."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config, read_config, update_config


class SideJobConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "config.json"

    def load(self, side_job=None, **sections):
        values = {"version": 1, **sections}
        if side_job is not None:
            values["side_job"] = side_job
        self.path.write_text(json.dumps(values))
        return load_config(self.path)

    def test_old_configs_gain_independent_side_job_defaults(self):
        settings = self.load(vpn={"enabled": True, "link": "vless://PRIVATE_MAIN_VPN"})
        side_job = settings["side_job"]
        self.assertEqual(side_job["channels"], [64, 128, 256, 512])
        self.assertEqual(side_job["blocks_per_stage"], [2, 2, 2, 2])
        self.assertEqual(side_job["epochs"], 0)
        self.assertEqual(side_job["prefetch_batches"], 2)
        self.assertEqual(side_job["passes_per_shard"], 1)
        self.assertEqual(side_job["cache_limit_mb"], 2048)
        self.assertFalse(side_job["vpn"]["enabled"])
        self.assertEqual(side_job["vpn"]["link"], "")
        self.assertIsNone(side_job["vpn"]["core_path"])
        self.assertEqual(side_job["vpn"]["startup_timeout"], 10)

    def test_defaults_do_not_share_mutable_values(self):
        first = self.load()["side_job"]
        first["channels"].append(1024)
        first["blocks_per_stage"][0] = 3
        first["vpn"]["link"] = "PRIVATE_LINK"
        again = self.load()["side_job"]
        self.assertEqual(again, DEFAULT_CONFIG["side_job"])

    def test_partial_overrides_keep_other_defaults_and_resolve_vpn_core(self):
        settings = self.load({"channels": [32, 64], "blocks_per_stage": [1, 2],
                              "weight_decay": 0, "vpn": {"enabled": True, "link": "ss://PRIVATE_SIDE_JOB",
                                                          "core_path": "bin/xray"}},
                             vpn={"enabled": True, "link": "ss://PRIVATE_MAIN"})
        side_job = settings["side_job"]
        self.assertEqual(side_job["batch_size"], 16)
        self.assertEqual(side_job["channels"], [32, 64])
        self.assertEqual(side_job["vpn"]["link"], "ss://PRIVATE_SIDE_JOB")
        self.assertEqual(side_job["vpn"]["core_path"], str(self.path.parent / "bin/xray"))
        self.assertEqual(settings["vpn"]["link"], "ss://PRIVATE_MAIN")

    def test_dataset_root_and_tree_urls_are_accepted(self):
        for suffix in ("", "/", "/tree/main/train", "/tree/revision/other/train"):
            url = "https://huggingface.co/datasets/BDRC/dataset" + suffix
            with self.subTest(suffix=suffix):
                self.assertEqual(self.load({"repo_url": url})["side_job"]["repo_url"], url)

    def test_invalid_dataset_urls_are_rejected_without_echoing_values(self):
        for url in ("PRIVATE_REPO", "https://example.com/PRIVATE_REPO", "http://huggingface.co/datasets/a/b",
                    "https://huggingface.co/a/b", "https://huggingface.co/datasets/a/b?token=PRIVATE_REPO",
                    "https://PRIVATE_REPO@huggingface.co/datasets/a/b", "https://huggingface.co/datasets/a/b#PRIVATE_REPO"):
            with self.subTest(url=url), self.assertRaisesRegex(ConfigError, "side_job.repo_url") as caught:
                self.load({"repo_url": url})
            self.assertNotIn("PRIVATE_REPO", str(caught.exception))

    def test_network_layout_types_lengths_and_limits_are_validated(self):
        invalid = ({"channels": []}, {"channels": "PRIVATE_LAYOUT"}, {"channels": [0, 128, 256, 512]},
                   {"channels": [64, True, 256, 512]}, {"channels": [4096, 128, 256, 512]},
                   {"channels": [32, 64]}, {"channels": [1] * 7}, {"blocks_per_stage": [1, 1, 1, 9]},
                   {"blocks_per_stage": [1, 1]}, {"blocks_per_stage": None})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ConfigError) as caught:
                self.load(values)
            self.assertNotIn("PRIVATE_LAYOUT", str(caught.exception))

    def test_optimizer_and_training_ranges_are_validated(self):
        invalid = (("hidden_size", 0), ("hidden_size", 8193), ("image_size", 31), ("image_size", 1025),
                   ("batch_size", 0), ("batch_size", 513), ("epochs", -1), ("epochs", True),
                   ("prefetch_batches", -1), ("prefetch_batches", 9), ("prefetch_batches", True),
                   ("passes_per_shard", 0), ("passes_per_shard", 101), ("passes_per_shard", True),
                   ("cache_limit_mb", 15), ("cache_limit_mb", 16385), ("cache_limit_mb", True),
                   ("seed", -1), ("seed", 2**32), ("learning_rate", 0), ("learning_rate", 1.1),
                   ("learning_rate", "PRIVATE_NUMBER"), ("weight_decay", -0.1), ("weight_decay", 2),
                   ("use_amp", 1))
        for field, value in invalid:
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, field) as caught:
                self.load({field: value})
            self.assertNotIn("PRIVATE_NUMBER", str(caught.exception))

    def test_side_job_vpn_validation_does_not_borrow_miner_link(self):
        invalid = ({"enabled": True}, {"enabled": 1}, {"link": "https://PRIVATE_VPN"},
                   {"startup_timeout": 0}, {"startup_timeout": 61}, {"core_path": 17})
        for vpn in invalid:
            with self.subTest(vpn=vpn), self.assertRaisesRegex(ConfigError, "side_job.vpn") as caught:
                self.load({"vpn": vpn}, vpn={"enabled": True, "link": "ss://PRIVATE_MAIN"})
            self.assertNotIn("PRIVATE_VPN", str(caught.exception))
            self.assertNotIn("PRIVATE_MAIN", str(caught.exception))

    def test_unknown_fields_are_rejected(self):
        for side_job in ({"PRIVATE_FIELD": 1}, {"vpn": {"PRIVATE_FIELD": 1}}):
            with self.subTest(side_job=side_job), self.assertRaises(ConfigError) as caught:
                self.load(side_job)
            self.assertNotIn("PRIVATE_FIELD", str(caught.exception))

    def test_updates_preserve_source_paths_and_nested_settings(self):
        source = {"channels": [32, 64], "blocks_per_stage": [1, 1], "vpn": {"core_path": "bin/xray"}}
        original = copy.deepcopy(source)
        self.load(source)
        update_config(self.path, {"side_job": {"batch_size": 8}})
        saved = read_config(self.path)["side_job"]
        self.assertEqual(saved, {**original, "batch_size": 8})
        self.assertEqual(source, original)


if __name__ == "__main__":
    unittest.main()
