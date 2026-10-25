"""Configuration dispatch checks without GPU, credentials, or remote services."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pearl_trainer import __main__ as cli
from pearl_trainer.config import DEFAULT_CONFIG


class ConfigDispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.file = Path(self.directory.name) / "settings.json"
        self.settings = copy.deepcopy(DEFAULT_CONFIG)

    def write_config(self):
        self.file.write_text(json.dumps(self.settings))

    def test_pool_check_uses_only_configured_transport(self):
        self.settings["pool"].update(host="localhost", port=19000,
                                     connect_timeout=2, read_timeout=3)
        self.write_config()
        with patch("pearl_trainer.pool_transport.PoolTransport") as transport, \
                patch.object(cli, "load_official_helpers") as helpers, \
                patch.object(cli, "require_cuda") as cuda:
            self.assertEqual(cli.main(["--config", str(self.file), "pool-check"]), 0)
        transport.assert_called_once_with(
            host="localhost", port=19000, tls=False, ca_file=None,
            connect_timeout=2, read_timeout=3,
        )
        transport.return_value.__enter__.assert_called_once()
        transport.return_value.__exit__.assert_called_once()
        helpers.assert_not_called()
        cuda.assert_not_called()

    def test_pool_mode_requires_payout_identity_before_gpu_or_network(self):
        self.write_config()
        with patch("pearl_trainer.pool_transport.PoolTransport") as transport, \
                patch.object(cli, "load_official_helpers") as helpers, \
                self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main(["--config", str(self.file)]), 1)
        self.assertIn("fill pool.wallet", " ".join(messages.output))
        transport.assert_not_called()
        helpers.assert_not_called()

    def test_pool_mode_rejects_unknown_pearlhash_adapter_before_network(self):
        self.settings["pool"]["host"] = "pool.pearlhash.xyz"
        self.settings["pool"]["protocol"] = "pearlhash"
        self.write_config()
        with patch("pearl_trainer.pool_transport.PoolTransport") as transport, \
                self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main(["--config", str(self.file)]), 1)
        self.assertIn("Pearlhash uses a separate", " ".join(messages.output))
        transport.assert_not_called()

    def test_pool_dispatch_honors_global_device_and_helper_overrides(self):
        self.write_config()
        with patch.object(cli, "pool_train", return_value=0) as train:
            self.assertEqual(cli.main([
                "--config", str(self.file), "--device", "cuda:2",
                "--official-source", "/tmp/another-official-checkout", "pool",
            ]), 0)
        effective = train.call_args.args[0]
        self.assertEqual(effective["device"], "cuda:2")
        self.assertEqual(effective["official_source"], "/tmp/another-official-checkout")

    def test_solo_mode_reads_file_credentials_and_cli_overrides(self):
        self.settings["mode"] = "solo"
        self.settings["training"].update(m=128, n=128, max_batches=2)
        self.settings["node"].update(username="file-user", password="file-password",
                                    address="file-address")
        self.write_config()
        with patch.object(cli, "load_official_helpers") as helpers, \
                patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main(["--config", str(self.file), "train", "--m", "512"]), 0)
        args = train.call_args.args[0]
        self.assertEqual(args.m, 512)
        self.assertEqual(args.n, 128)
        self.assertEqual(args.max_batches, 2)
        self.assertEqual(args.rpc_username, "file-user")
        self.assertEqual(args.rpc_password, "file-password")
        self.assertEqual(args.address, "file-address")
        self.assertEqual(args.output_dir, str(self.file.parent / "artifacts"))
        helpers.assert_called_once_with(str(self.file.parent / ".official/pearl"))

    def test_solo_mode_without_subcommand_runs_solo_training(self):
        self.settings["mode"] = "solo"
        self.write_config()
        with patch.object(cli, "load_official_helpers"), \
                patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main(["--config", str(self.file)]), 0)
        self.assertEqual(train.call_args.args[0].command, "train")
        self.assertFalse(train.call_args.args[0].no_submit)

    def test_pool_check_requires_configuration(self):
        with self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main(["pool-check"]), 1)
        self.assertIn("require --config", " ".join(messages.output))


if __name__ == "__main__":
    unittest.main()
