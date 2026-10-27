"""VPN settings and command routing are explicit, scoped, and fail closed."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from LRAEP_trainer import __main__ as cli
from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config
from LRAEP_trainer.network import SocksProxy, current_proxy
from LRAEP_trainer.vpn import VPNError


class VPNConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "config.json"

    def load(self, vpn=None):
        settings = {"version": 1}
        if vpn is not None:
            settings["vpn"] = vpn
        self.path.write_text(json.dumps(settings))
        return load_config(self.path)

    def test_existing_configuration_keeps_vpn_disabled_by_default(self):
        loaded = self.load()
        self.assertFalse(loaded["vpn"]["enabled"])
        self.assertEqual(loaded["vpn"]["link"], "")
        self.assertIsNone(loaded["vpn"]["core_path"])
        self.assertEqual(loaded["vpn"]["check_url"], "https://api.ipify.org")
        self.assertEqual(loaded["vpn"]["links_file"], str(self.directory / "links.txt"))

    def test_core_and_links_paths_resolve_beside_selected_config(self):
        loaded = self.load({"enabled": True, "link": "vless://SECRET@vpn.example:443",
                            "core_path": "bin/xray", "links_file": "private/links.txt",
                            "startup_timeout": 5, "check_url": None})
        self.assertEqual(loaded["vpn"]["core_path"], str(self.directory / "bin/xray"))
        self.assertEqual(loaded["vpn"]["links_file"], str(self.directory / "private/links.txt"))
        self.assertIsNone(loaded["vpn"]["check_url"])
        self.assertEqual(loaded["vpn"]["link"], "vless://SECRET@vpn.example:443")

    def test_invalid_vpn_settings_do_not_echo_links_or_credentials(self):
        invalid = [
            [], {"enabled": "true"}, {"enabled": True}, {"enabled": True, "link": " "},
            {"link": None}, {"link": 1}, {"link": "vless://SECRET\n@vpn.example:443"},
            {"link": "https://SECRET@vpn.example"}, {"core_path": 1}, {"links_file": None},
            {"startup_timeout": 0}, {"startup_timeout": -1}, {"startup_timeout": 61},
            {"startup_timeout": True}, {"startup_timeout": "5"},
            {"check_url": "http://SECRET.example/ip"},
            {"check_url": "https://user:SECRET@check.example/ip"},
            {"check_url": "https://check.example/ip#SECRET"},
            {"SECRET": "SECRET"},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ConfigError) as error:
                self.load(settings)
            self.assertNotIn("SECRET", str(error.exception))

    def test_defaults_are_not_mutated_by_loading_vpn_settings(self):
        before = copy.deepcopy(DEFAULT_CONFIG)
        self.load({"link": "ss://SECRET@vpn.example:443", "links_file": "other.txt"})
        self.assertEqual(DEFAULT_CONFIG, before)


class VPNCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "config.json"
        self.settings = copy.deepcopy(DEFAULT_CONFIG)
        self.settings["vpn"].update(enabled=True, link="vless://SECRET@vpn.example:443")
        self.settings["pool"].pop("KEX", None)
        self.settings["pool"].pop("WR", None)
        self.settings["pool"]["KEX_uname"] = "public-test-wallet"
        self.settings["pool"]["host"] = "pool.example"
        self.proxy = SocksProxy("127.0.0.1", 19080, "private-user", "private-password")
        self.managed = MagicMock()
        self.managed.__enter__.return_value = self.proxy

    def write(self):
        self.path.write_text(json.dumps(self.settings))

    def arguments(self, command):
        return ["--config", str(self.path), *command]

    def assert_route(self, *_args, **_kwargs):
        self.assertIs(current_proxy(), self.proxy)
        return 0

    def test_pool_and_transport_check_execute_inside_managed_route(self):
        self.write()
        for command, function in (("pool", "pool_train"), ("pool-check", "pool_check")):
            with self.subTest(command=command), \
                 patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed) as core, \
                 patch.object(cli, function, side_effect=self.assert_route) as operation, \
                 patch.object(cli, "require_cuda") as cuda:
                self.assertEqual(cli.main(self.arguments([command])), 0)
                operation.assert_called_once()
                core.assert_called_once()
                self.managed.__exit__.assert_called()
                self.assertIsNone(current_proxy())
                cuda.assert_not_called()

    def test_pool_login_check_executes_inside_route_without_gpu_initialization(self):
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed), \
             patch("LRAEP_trainer.pool_training.run_pool_training", side_effect=self.assert_route) as login, \
             patch.object(cli, "require_cuda") as cuda:
            self.assertEqual(cli.main(self.arguments(["pool-login-check"])), 0)
        self.assertTrue(login.call_args.kwargs["login_only"])
        self.assertIsNone(current_proxy())
        cuda.assert_not_called()

    def test_solo_training_and_helpers_execute_inside_route(self):
        self.settings["mode"] = "solo"
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed), \
             patch.object(cli, "load_official_helpers", side_effect=self.assert_route) as helpers, \
             patch.object(cli, "train", side_effect=self.assert_route) as train, \
             patch.object(cli.torch, "set_num_threads"):
            self.assertEqual(cli.main(self.arguments(["train"])), 0)
        helpers.assert_called_once()
        train.assert_called_once()
        self.assertIsNone(current_proxy())

    def test_config_selected_default_mode_also_uses_vpn(self):
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed), \
             patch.object(cli, "pool_train", side_effect=self.assert_route) as train:
            self.assertEqual(cli.main(self.arguments([])), 0)
        train.assert_called_once()
        self.assertIsNone(current_proxy())

    def test_failed_core_never_runs_any_network_command(self):
        self.write()
        self.managed.__enter__.side_effect = VPNError("proxy core unavailable")
        for command, function in (("pool", "pool_train"), ("pool-check", "pool_check"),
                                  ("train", "train")):
            with self.subTest(command=command), \
                 patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed), \
                 patch.object(cli, function) as operation, \
                 patch.object(cli, "load_official_helpers") as helpers, \
                 self.assertLogs(cli.LOG, level="ERROR") as logs:
                self.assertEqual(cli.main(self.arguments([command])), 1)
            operation.assert_not_called()
            helpers.assert_not_called()
            self.assertNotIn("SECRET", " ".join(logs.output))
            self.assertIsNone(current_proxy())

    def test_command_failure_closes_core_and_restores_route(self):
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN", return_value=self.managed), \
             patch.object(cli, "pool_check", side_effect=OSError("pool unavailable")), \
             self.assertLogs(cli.LOG, level="ERROR"):
            self.assertEqual(cli.main(self.arguments(["pool-check"])), 1)
        self.managed.__exit__.assert_called_once()
        self.assertIsNone(current_proxy())

    def test_offline_self_test_never_starts_configured_vpn(self):
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
             patch.object(cli, "load_official_helpers"), \
             patch.object(cli, "self_test", return_value=0) as operation, \
             patch.object(cli.torch, "set_num_threads"):
            self.assertEqual(cli.main(self.arguments(["self-test"])), 0)
        core.assert_not_called()
        operation.assert_called_once()

    def test_vpn_check_uses_config_file_options_without_starting_an_outer_tunnel(self):
        self.settings["vpn"]["links_file"] = "private/links.txt"
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
             patch("LRAEP_trainer.vpn_check.run_vpn_check", return_value=0) as operation, \
             patch.object(cli, "load_official_helpers") as helpers, \
             patch.object(cli, "require_cuda") as cuda:
            self.assertEqual(cli.main(self.arguments(["vpn-check"])), 0)
        core.assert_not_called()
        helpers.assert_not_called()
        cuda.assert_not_called()
        self.assertEqual(operation.call_args.args[1], str(self.directory / "private/links.txt"))
        self.assertEqual(operation.call_args.kwargs, {"timeout": 5, "attempts": 3, "output": None})

    def test_vpn_check_miner_flag_selects_profile_from_same_config_in_both_positions(self):
        self.settings["pool"].update(
            protocol="KEX", WR_wallet="wr-only-wallet", KEX_uname="kex-only-name",
            KEX={"host": "kex.example", "port": 8048, "tls": True, "ca_file": None},
            WR={"host": "wr.example", "port": 9000, "tls": False, "ca_file": None},
        )
        self.write()
        original = self.path.read_text()
        for command in (["--miner", "WR", "vpn-check"], ["vpn-check", "--miner", "WR"]):
            with self.subTest(command=command), \
                 patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
                 patch("LRAEP_trainer.vpn_check.run_vpn_check", return_value=0) as operation:
                self.assertEqual(cli.main(self.arguments(command)), 0)
            selected = operation.call_args.args[0]
            pool = selected["pool"]
            self.assertEqual((pool["protocol"], pool["host"], pool["port"], pool["tls"]),
                             ("WR", "wr.example", 9000, False))
            self.assertEqual((pool["WR_wallet"], pool["KEX_uname"]),
                             ("wr-only-wallet", "kex-only-name"))
            self.assertTrue(selected["vpn"]["enabled"])
            self.assertEqual(selected["vpn"]["link"], self.settings["vpn"]["link"])
            self.assertNotIn("WR", pool)
            self.assertNotIn("KEX", pool)
            core.assert_not_called()
            self.assertEqual(self.path.read_text(), original)

    def test_selected_wr_endpoint_is_validated_before_vpn_check_network_activity(self):
        self.settings["pool"].update(
            protocol="KEX", KEX={"host": "kex.example", "port": 8048},
            WR={"host": "", "port": 9000},
        )
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
             patch("LRAEP_trainer.vpn_check._probe_pool") as pool, \
             self.assertLogs(cli.LOG, level="ERROR") as logs:
            self.assertEqual(cli.main(self.arguments(["vpn-check", "--miner", "WR"])), 1)
        core.assert_not_called()
        pool.assert_not_called()
        self.assertIn("host", " ".join(logs.output))

    def test_vpn_check_accepts_explicit_input_and_report_flags(self):
        self.write()
        with patch("LRAEP_trainer.vpn_check.run_vpn_check", return_value=0) as operation:
            self.assertEqual(cli.main(self.arguments([
                "vpn-check", "--links", "chosen.txt", "--timeout", "2.5", "--attempts", "2",
                "--output", "report.json",
            ])), 0)
        self.assertEqual(operation.call_args.args[1], "chosen.txt")
        self.assertEqual(operation.call_args.kwargs,
                         {"timeout": 2.5, "attempts": 2, "output": "report.json"})

    def test_vpn_check_without_config_uses_links_txt_and_does_no_gpu_work(self):
        with patch.object(cli, "default_config_path", return_value=self.directory / "absent.json"), \
             patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
             patch("LRAEP_trainer.vpn_check.run_vpn_check", return_value=1) as operation, \
             patch.object(cli, "load_official_helpers") as helpers, \
             patch.object(cli, "require_cuda") as cuda:
            self.assertEqual(cli.main(["vpn-check"]), 1)
        self.assertEqual(operation.call_args.args[1], "links.txt")
        core.assert_not_called()
        helpers.assert_not_called()
        cuda.assert_not_called()

    def test_disabling_vpn_does_not_start_core(self):
        self.settings["vpn"]["enabled"] = False
        self.write()
        with patch("LRAEP_trainer.vpn.ManagedVPN") as core, \
             patch.object(cli, "pool_check", return_value=0):
            self.assertEqual(cli.main(self.arguments(["pool-check"])), 0)
        core.assert_not_called()
        self.assertIsNone(current_proxy())


if __name__ == "__main__":
    unittest.main()
