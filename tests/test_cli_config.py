"""Configuration dispatch checks without GPU, credentials, or remote services."""

import copy
import json
import sys
from pathlib import Path
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import LRAEP_trainer
from LRAEP_trainer import __main__ as cli
from LRAEP_trainer.config import DEFAULT_CONFIG
from LRAEP_trainer.paths import artifacts_dir


class ConfigDispatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.file = Path(self.directory.name) / "settings.json"
        self.settings = copy.deepcopy(DEFAULT_CONFIG)
        self.settings["pool"]["host"] = "pool.example"
        canonical = patch.object(cli, "default_config_path",
                                 return_value=Path(self.directory.name) / "absent-config.json")
        canonical.start()
        self.addCleanup(canonical.stop)

    def write_config(self):
        self.file.write_text(json.dumps(self.settings))

    def test_pool_check_uses_only_configured_transport(self):
        self.settings["pool"].update(host="localhost", port=19000,
                                     connect_timeout=2, read_timeout=3)
        self.write_config()
        with patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
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
        with patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
                patch.object(cli, "load_official_helpers") as helpers, \
                self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main(["--config", str(self.file)]), 1)
        self.assertIn("fill pool.KEX_uname", " ".join(messages.output))
        transport.assert_not_called()
        helpers.assert_not_called()

    def test_missing_pool_host_fails_before_vpn_gpu_or_network(self):
        self.settings["pool"].update(host="", KEX_uname="local-test-account")
        self.settings["vpn"].update(enabled=True, link="vless://SECRET@vpn.example:443")
        self.write_config()
        for command in ("pool", "pool-check", "pool-login-check"):
            with self.subTest(command=command), \
                    patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
                    patch("LRAEP_trainer.vpn.ManagedVPN") as vpn, \
                    patch.object(cli, "load_official_helpers") as helpers, \
                    self.assertLogs(cli.LOG, level="ERROR") as messages:
                self.assertEqual(cli.main(["--config", str(self.file), command]), 1)
            self.assertIn("fill pool.host", " ".join(messages.output))
            transport.assert_not_called()
            vpn.assert_not_called()
            helpers.assert_not_called()

    def test_pearlhash_requires_identity_before_network(self):
        self.settings["pool"]["host"] = "pool.example"
        self.settings["pool"]["protocol"] = "pearlhash"
        self.write_config()
        with patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
                self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main(["--config", str(self.file)]), 1)
        self.assertIn("fill pool.WR_wallet", " ".join(messages.output))
        transport.assert_not_called()

    def test_selected_pool_requires_its_own_identity_before_helpers_or_vpn(self):
        for selected, required, other in (("KEX", "KEX_uname", "WR_wallet"),
                                          ("WR", "WR_wallet", "KEX_uname")):
            self.settings["pool"].update({required: "", other: "PRIVATE_OTHER_POOL_IDENTITY"})
            self.settings["vpn"].update(enabled=True, link="vless://SECRET@vpn.example:443")
            self.write_config()
            for command in ("pool", "pool-login-check"):
                with self.subTest(selected=selected, command=command), \
                        patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
                        patch("LRAEP_trainer.vpn.ManagedVPN") as vpn, \
                        patch.object(cli, "load_official_helpers") as helpers, \
                        self.assertLogs(cli.LOG, level="ERROR") as messages:
                    self.assertEqual(cli.main(["--config", str(self.file), command, "--miner", selected]), 1)
                output = " ".join(messages.output)
                self.assertIn(f"fill pool.{required}", output)
                self.assertNotIn("PRIVATE_OTHER_POOL_IDENTITY", output)
                transport.assert_not_called()
                vpn.assert_not_called()
                helpers.assert_not_called()

    def test_miner_selection_overrides_adapter_and_preserves_endpoint(self):
        self.settings["pool"].update(host="pool.example", port=9000,
                                     KEX_uname="test-kex-account", WR_wallet="test-wr-address")
        self.write_config()
        for miner in ("KEX", "WR"):
            for arguments in (["pool", "--miner", miner], ["--miner", miner, "pool"]):
                with self.subTest(arguments=arguments), patch.object(cli, "pool_train", return_value=0) as train:
                    self.assertEqual(cli.main(["--config", str(self.file), *arguments]), 0)
                    effective = train.call_args.args[0]["pool"]
                    self.assertEqual(effective["protocol"], miner)
                    self.assertEqual(effective["host"], "pool.example")
                    self.assertEqual(effective["port"], 9000)

    def add_profiles(self):
        self.settings["pool"].update(
            KEX_uname="test-kex-account", WR_wallet="test-wr-address", worker="shared-worker",
            KEX={"host": "kex.example", "port": 8048, "tls": True},
            WR={"host": "wr.example", "port": 9000, "tls": False, "certificate_version": 3},
        )

    def test_one_unchanged_config_selects_each_miners_endpoint_and_identity(self):
        self.add_profiles()
        self.settings["pool"]["host"] = "stale-shared.example"
        self.write_config()
        original = self.file.read_bytes()
        for miner, host, port, tls in (("WR", "wr.example", 9000, False),
                                       ("KEX", "kex.example", 8048, True)):
            for arguments in (["--miner", miner, "pool"], ["pool", "--miner", miner]):
                with self.subTest(arguments=arguments), \
                        patch.object(cli, "pool_train", return_value=0) as train:
                    self.assertEqual(cli.main(["--config", str(self.file), *arguments]), 0)
                effective = train.call_args.args[0]
                self.assertEqual((effective["pool"]["protocol"], effective["pool"]["host"],
                                  effective["pool"]["port"], effective["pool"]["tls"]),
                                 (miner, host, port, tls))
                self.assertEqual(effective["pool"]["worker"], "shared-worker")
                self.assertEqual(effective["pool"]["WR_wallet"], "test-wr-address")
                self.assertEqual(effective["pool"]["KEX_uname"], "test-kex-account")
                self.assertEqual(effective["training"]["m"], self.settings["training"]["m"])
                self.assertEqual(effective["vpn"]["enabled"], self.settings["vpn"]["enabled"])
                self.assertNotIn("KEX", effective["pool"])
                self.assertNotIn("WR", effective["pool"])
        self.assertEqual(self.file.read_bytes(), original)

    def test_selected_profile_cannot_borrow_another_host_before_vpn(self):
        self.add_profiles()
        self.settings["pool"]["WR"] = {}
        self.settings["vpn"].update(enabled=True, link="vless://SECRET@vpn.example:443")
        self.write_config()
        for command in ("pool", "pool-check", "pool-login-check"):
            with self.subTest(command=command), \
                    patch("LRAEP_trainer.vpn.ManagedVPN") as vpn, \
                    patch("LRAEP_trainer.pool_transport.PoolTransport") as transport, \
                    patch.object(cli, "load_official_helpers") as helpers, \
                    self.assertLogs(cli.LOG, level="ERROR"):
                self.assertEqual(cli.main(["--config", str(self.file), "--miner", "WR", command]), 1)
            vpn.assert_not_called()
            transport.assert_not_called()
            helpers.assert_not_called()

    def test_profile_pool_check_resolves_direct_api_calls(self):
        self.add_profiles()
        self.settings["pool"]["protocol"] = "WR"
        before = copy.deepcopy(self.settings)
        with patch("LRAEP_trainer.pool_transport.PoolTransport") as transport:
            self.assertEqual(cli.pool_check(self.settings), 0)
        self.assertEqual(transport.call_args.kwargs["host"], "wr.example")
        self.assertEqual(transport.call_args.kwargs["port"], 9000)
        self.assertFalse(transport.call_args.kwargs["tls"])
        self.assertEqual(self.settings, before)

    def test_canonical_config_is_used_without_config_flag_or_subcommand(self):
        self.add_profiles()
        self.file = self.file.with_name("config.json")
        self.write_config()
        with patch.object(cli, "default_config_path", return_value=self.file), \
                patch.object(cli, "pool_train", return_value=0) as train:
            self.assertEqual(cli.main(["--miner", "WR"]), 0)
        self.assertEqual(train.call_args.args[0]["pool"]["protocol"], "WR")
        self.assertEqual(train.call_args.args[0]["pool"]["host"], "wr.example")

    def test_explicit_config_never_reads_a_different_canonical_file(self):
        self.write_config()
        with patch.object(cli, "default_config_path", side_effect=AssertionError("explicit file wins")), \
                patch.object(cli, "pool_check", return_value=0) as check:
            self.assertEqual(cli.main(["--config", str(self.file), "pool-check"]), 0)
        self.assertEqual(check.call_args.args[0]["pool"]["host"], "pool.example")

    def test_offline_self_test_never_autoloads_private_canonical_config(self):
        with patch.object(cli, "default_config_path", side_effect=AssertionError("self-test is offline")), \
                patch.object(cli, "load_official_helpers"), \
                patch.object(cli, "self_test", return_value=0) as check:
            self.assertEqual(cli.main(["self-test"]), 0)
        check.assert_called_once()

    def test_invalid_canonical_config_does_not_fall_back_or_start_work(self):
        self.file.write_text('{"invalid": true}')
        with patch.object(cli, "default_config_path", return_value=self.file), \
                patch.object(cli, "pool_train") as train, \
                patch("LRAEP_trainer.vpn.ManagedVPN") as vpn, \
                self.assertLogs(cli.LOG, level="ERROR"):
            self.assertEqual(cli.main(["--miner", "WR", "pool"]), 1)
        train.assert_not_called()
        vpn.assert_not_called()

    def test_pool_dispatch_honors_global_device_and_helper_overrides(self):
        self.settings["pool"]["KEX_uname"] = "test-kex-account"
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
        self.settings["training"]["job_sequence"] = [["main", 10, 65], ["side-job", 3, 15], ["main", 5, 0], ["main", 20, 35]]
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
        self.assertEqual(args.job_sequence, [("main", 10, 65), ("side-job", 3, 15), ("main", 5, 0), ("main", 20, 35)])
        self.assertFalse(hasattr(args, "side_job_effort"))
        self.assertEqual(args.rpc_username, "file-user")
        self.assertEqual(args.rpc_password, "file-password")
        self.assertEqual(args.address, "file-address")
        self.assertEqual(args.output_dir, str(artifacts_dir().resolve()))
        helpers.assert_called_once_with(str(self.file.parent / "src/LRAEP_helpers"))

    def test_solo_gpu_limit_can_be_overridden_on_cli(self):
        self.settings["mode"] = "solo"
        self.write_config()
        with patch.object(cli, "load_official_helpers"), patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main(["--config", str(self.file), "train", "--max-gpu-utilization", "60"]), 0)
        self.assertEqual(train.call_args.args[0].max_gpu_utilization, 60)
        self.assertEqual(train.call_args.args[0].job_sequence, [("main", 60, 60)])
        with patch.object(cli, "load_official_helpers") as helpers, self.assertLogs(cli.LOG, level="ERROR"):
            self.assertEqual(cli.main(["--config", str(self.file), "train", "--max-gpu-utilization", "101"]), 1)
        helpers.assert_not_called()

    def test_solo_effort_schedule_can_be_overridden_with_json(self):
        self.settings["mode"] = "solo"
        self.write_config()
        with patch.object(cli, "load_official_helpers"), patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main([
                "--config", str(self.file), "train", "--job-sequence",
                '[["main", 2, 70], ["side-job", 3, 0], ["main", 0, -1]]',
            ]), 0)
        self.assertEqual(train.call_args.args[0].job_sequence, [("main", 2, 70), ("side-job", 3, 0), ("main", 0, -1)])
        self.assertFalse(hasattr(train.call_args.args[0], "max_gpu_utilization"))

    def test_solo_effort_override_is_validated_before_network_or_helpers(self):
        self.settings["mode"] = "solo"
        self.settings["vpn"].update(enabled=True, link="vless://SECRET@vpn.example:443")
        self.write_config()
        for schedule in ("not-json", "[]", "[[0, 70]]", '[["MAIN", 1, 80]]',
                         '[["main", 0, 70]]', '[["side-job", 1, 101]]',
                         '[["main", 1, true]]', '[["main", 1, -2]]'):
            with self.subTest(schedule=schedule), \
                    patch("LRAEP_trainer.vpn.ManagedVPN") as vpn, \
                    patch.object(cli, "load_official_helpers") as helpers, \
                    patch.object(cli, "train") as train, \
                    self.assertLogs(cli.LOG, level="ERROR"):
                self.assertEqual(cli.main([
                    "--config", str(self.file), "train", "--job-sequence", schedule,
                ]), 1)
            vpn.assert_not_called()
            helpers.assert_not_called()
            train.assert_not_called()

    def test_solo_conflicting_effort_overrides_are_rejected(self):
        self.settings["mode"] = "solo"
        self.write_config()
        with patch.object(cli, "load_official_helpers") as helpers, \
                patch.object(cli, "train") as train, \
                self.assertLogs(cli.LOG, level="ERROR") as messages:
            self.assertEqual(cli.main([
                "--config", str(self.file), "train", "--job-sequence", '[["main", 1, 70]]',
                "--max-gpu-utilization", "60",
            ]), 1)
        self.assertIn("use only one", " ".join(messages.output))
        helpers.assert_not_called()
        train.assert_not_called()

    def test_solo_without_configuration_uses_default_effort_schedule(self):
        with patch.object(cli, "load_official_helpers"), patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main(["train"]), 0)
        self.assertEqual(train.call_args.args[0].job_sequence, [("main", 60, 80), ("main", 120, 70), ("main", 180, 10)])
        self.assertFalse(hasattr(train.call_args.args[0], "side_job_effort"))

    def test_immediate_exit_skips_network_helpers_and_mining_entrypoints(self):
        self.settings["training"]["job_sequence"] = [["main", 0, -1]]
        # The requested exit also bypasses unavailable mining credentials/endpoints.
        self.settings["pool"].update(host="", KEX_uname="", WR_wallet="")
        self.write_config()
        for command in ("train", "pool"):
            with self.subTest(command=command), \
                    patch.object(cli, "_network_scope") as network_scope, \
                    patch.object(cli, "load_official_helpers") as helpers, \
                    patch.object(cli, "require_cuda") as cuda, \
                    patch.object(cli, "pool_train") as pool_train, \
                    patch.object(cli, "train") as train:
                self.assertEqual(cli.main(["--config", str(self.file), command]), 0)
            network_scope.assert_not_called()
            helpers.assert_not_called()
            cuda.assert_not_called()
            pool_train.assert_not_called()
            train.assert_not_called()

    def test_legacy_config_gpu_limit_loads_as_a_constant_schedule(self):
        self.settings["mode"] = "solo"
        del self.settings["training"]["job_sequence"]
        self.settings["training"]["max_gpu_utilization"] = 65
        self.write_config()
        with patch.object(cli, "load_official_helpers"), patch.object(cli, "train", return_value=0) as train:
            self.assertEqual(cli.main(["--config", str(self.file), "train"]), 0)
        self.assertEqual(train.call_args.args[0].job_sequence, [("main", 60, 65)])

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


class DirectRuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.configuration = copy.deepcopy(DEFAULT_CONFIG)
        self.configuration["pool"].update(
            protocol="WR", host="obsolete-shared.example", KEX_uname="test-kex-account",
            WR_wallet="test-wr-address", worker="shared-worker",
            KEX={"host": "kex.example", "port": 8048, "tls": True},
            WR={"host": "wr.example", "port": 9000, "tls": False},
        )

    def test_direct_pool_login_resolves_selected_profile_without_mutating_config(self):
        from LRAEP_trainer import pool_training as worker
        from LRAEP_trainer.pool_client import JobSnapshot
        from LRAEP_trainer.pool_protocol import PoolJob

        session = MagicMock()
        session.__enter__.return_value = session
        session.negotiated_gzip = False
        session.wait_job.return_value = JobSnapshot(PoolJob("job", bytes(76), 1, 3, 1), 1)
        native = ModuleType("LRAEP_trainer.primitives")
        native.IncompleteBlockHeader = SimpleNamespace(from_bytes=MagicMock())
        before = copy.deepcopy(self.configuration)
        for miner, target, host in (("WR", "LRAEP_trainer.wr_client.WRSession", "wr.example"),
                                     ("KEX", "LRAEP_trainer.pool_client.PoolSession", "kex.example")):
            self.configuration["pool"]["protocol"] = miner
            with self.subTest(miner=miner), \
                    patch.object(worker, "load_official_helpers"), \
                    patch.object(worker, "pool_share_target", return_value=1), \
                    patch.dict(sys.modules, {"LRAEP_trainer.primitives": native}), \
                    patch.object(LRAEP_trainer, "primitives", native, create=True), \
                    patch(target, return_value=session) as factory:
                self.assertEqual(worker.run_pool_training(self.configuration, login_only=True), 0)
            resolved = factory.call_args.args[0]
            self.assertEqual(resolved["protocol"], miner)
            self.assertEqual(resolved["host"], host)
            self.assertEqual(resolved["worker"], "shared-worker")
            self.assertNotIn("WR", resolved)
            self.assertNotIn("KEX", resolved)
        self.configuration["pool"]["protocol"] = before["pool"]["protocol"]
        self.assertEqual(self.configuration, before)

    def test_direct_training_rejects_missing_selected_endpoint_before_helpers_or_session(self):
        from LRAEP_trainer import pool_training as worker

        self.configuration["pool"]["WR"] = {}
        with patch.object(worker, "load_official_helpers") as helpers, \
                patch("LRAEP_trainer.wr_client.WRSession") as session, \
                self.assertRaisesRegex(ValueError, "fill pool.host"):
            worker.run_pool_training(self.configuration, login_only=True)
        helpers.assert_not_called()
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
