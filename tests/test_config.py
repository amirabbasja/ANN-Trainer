"""Configuration validation, path portability, and secret-safe errors."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config
from LRAEP_trainer.paths import artifacts_dir


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "trainer.json"

    def load(self, value):
        self.path.write_text(json.dumps(value))
        return load_config(self.path)

    def test_defaults_and_empty_pool_identities_are_valid(self):
        result = self.load({"version": 1})
        self.assertEqual(result["mode"], "pool")
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["official_source"], str(self.directory / "src/LRAEP_helpers"))
        self.assertEqual(result["training"]["output_dir"], str(artifacts_dir().resolve()))
        self.assertEqual(result["training"]["main_effort"], [(60, 80), (120, 70), (180, 10)])
        self.assertEqual(result["training"]["decoy_effort"], [])
        self.assertNotIn("max_gpu_utilization", result["training"])
        self.assertEqual(result["pool"]["host"], "")
        self.assertEqual(result["pool"]["port"], 7048)
        self.assertFalse(result["pool"]["tls"])
        self.assertEqual(result["pool"]["WR_wallet"], "")
        self.assertEqual(result["pool"]["KEX_uname"], "")
        self.assertNotIn("wallet", result["pool"])
        self.assertEqual(result["pool"]["password"], "x")
        self.assertEqual(result["pool"]["protocol"], "KEX")
        self.assertIsNone(result["pool"]["protocol_documentation"])
        self.assertEqual(result["pool"]["compression"], "none")
        self.assertIsNone(result["pool"]["certificate_version"])
        self.assertEqual(result["pool"]["agent"], "LRAEP-trainer/0.1")
        self.assertTrue(result["pool"]["stop_on_unknown_request"])
        self.assertIsNone(result["pool"]["unknown_request_dir"])
        self.assertEqual(result["node"]["rpc_url"], "https://127.0.0.1:44107")

    def test_effort_schedules_preserve_pause_exit_and_decoy_entries(self):
        schedule = [[0.5, 100], [1, 0], [0, -1]]
        result = self.load({"version": 1, "training": {
            "main_effort": schedule, "decoy_effort": [[2, 1], [3, 99]],
        }})
        self.assertEqual(result["training"]["main_effort"], [(0.5, 100), (1, 0), (0, -1)])
        self.assertEqual(result["training"]["decoy_effort"], [(2, 1), (3, 99)])
        self.assertEqual(schedule, [[0.5, 100], [1, 0], [0, -1]])

    def test_effort_schedules_reject_invalid_pairs_without_echoing_values(self):
        malformed = (None, True, "PRIVATE_EFFORT", {}, [1, 80], [[1]], [[1, 80, 2]],
                     [[0, 80]], [[-1, 80]], [[True, 80]], [["1", 80]],
                     [[1, -2]], [[1, 101]], [[1, True]], [[1, 70.5]], [[1, "PRIVATE_EFFORT"]],
                     [[-1, -1]])
        for field in ("main_effort", "decoy_effort"):
            for value in malformed:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, field) as caught:
                    self.load({"version": 1, "training": {field: value}})
                self.assertNotIn("PRIVATE_EFFORT", str(caught.exception))
        with self.assertRaisesRegex(ConfigError, "main_effort"):
            self.load({"version": 1, "training": {"main_effort": []}})

    def test_effort_durations_must_be_finite(self):
        for duration in (float("inf"), float("-inf"), float("nan")):
            for field in ("main_effort", "decoy_effort"):
                with self.subTest(field=field, duration=duration), self.assertRaises(ConfigError):
                    self.load({"version": 1, "training": {field: [[duration, 80]]}})

    def test_legacy_gpu_limit_becomes_a_constant_schedule(self):
        for percent in (1, 70, 100):
            with self.subTest(percent=percent):
                result = self.load({"version": 1, "training": {"max_gpu_utilization": percent}})
                self.assertEqual(result["training"]["main_effort"], [(60, percent)])
                self.assertNotIn("max_gpu_utilization", result["training"])
        for value in (0, -1, 101, True, 80.5, "80", None):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
                self.load({"version": 1, "training": {"max_gpu_utilization": value}})

    def test_explicit_main_effort_takes_precedence_over_legacy_limit(self):
        result = self.load({"version": 1, "training": {
            "main_effort": [[30, 40]], "max_gpu_utilization": 70,
        }})
        self.assertEqual(result["training"]["main_effort"], [(30, 40)])
        with self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
            self.load({"version": 1, "training": {"main_effort": [[30, 40]], "max_gpu_utilization": 101}})

    def test_effort_defaults_are_independent_between_loads_and_export(self):
        result = self.load({"version": 1})
        result["training"]["main_effort"].append((1, 100))
        result["training"]["decoy_effort"].append((1, 0))
        again = self.load({"version": 1})
        self.assertEqual(again["training"]["main_effort"], [(60, 80), (120, 70), (180, 10)])
        self.assertEqual(again["training"]["decoy_effort"], [])
        exported = copy.deepcopy(DEFAULT_CONFIG["training"])
        try:
            DEFAULT_CONFIG["training"]["main_effort"][0][1] = 99
            self.assertEqual(self.load({"version": 1})["training"]["main_effort"][0], (60, 80))
        finally:
            DEFAULT_CONFIG["training"] = exported

    def test_consensus_proof_input_limit_is_checked_before_gpu_work(self):
        with self.assertRaisesRegex(ConfigError, "4 MiB"):
            self.load({"version": 1, "training": {"k": 32384, "tile_h": 128, "tile_w": 2}})
        valid = self.load({"version": 1, "training": {"k": 32256, "tile_h": 128, "tile_w": 2}})
        self.assertEqual(valid["training"]["k"], 32256)

    def test_all_filesystem_paths_are_relative_to_the_config_file(self):
        result = self.load({
            "version": 1, "official_source": "helpers/LRAEP",
            "training": {"output_dir": "results"},
            "pool": {"tls": True, "ca_file": "certs/pool.pem",
                     "protocol_documentation": "docs/protocol.json"},
            "node": {"ca_file": "certs/node.pem"},
        })
        self.assertEqual(result["official_source"], str(self.directory / "helpers/LRAEP"))
        self.assertEqual(result["training"]["output_dir"], str(self.directory / "results"))
        self.assertEqual(result["pool"]["ca_file"], str(self.directory / "certs/pool.pem"))
        self.assertEqual(result["node"]["ca_file"], str(self.directory / "certs/node.pem"))
        self.assertEqual(result["pool"]["protocol_documentation"], str(self.directory / "docs/protocol.json"))

    def test_documentation_url_is_preserved_without_loading_it(self):
        url = "https://pool.example/en/prl#framing"
        result = self.load({"version": 1, "pool": {"protocol_documentation": url}})
        self.assertEqual(result["pool"]["protocol_documentation"], url)

    def test_exported_example_defaults_can_be_loaded_without_mutation(self):
        result = self.load(DEFAULT_CONFIG)
        self.assertEqual(result["version"], 1)
        self.assertEqual(DEFAULT_CONFIG["official_source"], "src/LRAEP_helpers")
        self.assertEqual(DEFAULT_CONFIG["training"]["output_dir"], str(artifacts_dir()))

    def test_explicit_pool_protocol_options_are_preserved(self):
        result = self.load({"version": 1, "pool": {
            "protocol": "KEX", "compression": "gzip", "certificate_version": 3,
            "agent": "my-custom-trainer/2", "KEX_uname": "my-account", "WR_wallet": "my-wr-address",
            "worker": "rig2",
            "password": "d=2097152",
        }})
        self.assertEqual(result["pool"]["protocol"], "KEX")
        self.assertEqual(result["pool"]["compression"], "gzip")
        self.assertEqual(result["pool"]["certificate_version"], 3)
        self.assertEqual(result["pool"]["agent"], "my-custom-trainer/2")
        self.assertEqual(result["pool"]["KEX_uname"], "my-account")
        self.assertEqual(result["pool"]["WR_wallet"], "my-wr-address")
        self.assertEqual(result["pool"]["password"], "d=2097152")
        self.assertEqual(self.load({"version": 1, "pool": {"protocol": "pearlhash"}})["pool"]["protocol"], "WR")
        self.assertEqual(self.load({"version": 1, "pool": {"protocol": "WR"}})["pool"]["protocol"], "WR")

    def test_pool_identity_fields_are_separate_and_case_sensitive(self):
        for protocol in ("KEX", "WR"):
            loaded = self.load({"version": 1, "pool": {
                "protocol": protocol, "KEX_uname": "kex-account", "WR_wallet": "wr-payout-address",
            }})
            self.assertEqual(loaded["pool"]["KEX_uname"], "kex-account")
            self.assertEqual(loaded["pool"]["WR_wallet"], "wr-payout-address")
        for field in ("wallet", "kex_uname", "wr_wallet", "KEX_wallet", "WR_uname"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "unknown field") as caught:
                self.load({"version": 1, "pool": {field: "PRIVATE_IDENTITY"}})
            self.assertNotIn("PRIVATE_IDENTITY", str(caught.exception))

    def test_pool_identity_fields_require_strings_without_controls(self):
        for field in ("KEX_uname", "WR_wallet"):
            for value in (None, True, 17, [], {}, "PRIVATE_IDENTITY\n"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, field) as caught:
                    self.load({"version": 1, "pool": {field: value}})
                self.assertNotIn("PRIVATE_IDENTITY", str(caught.exception))

    def test_pool_profiles_resolve_paths_beside_one_configuration_file(self):
        loaded = self.load({"version": 1, "pool": {
            "KEX": {"host": "kex.pool.example", "port": 8048, "tls": True,
                    "ca_file": "certs/kex.pem", "compression": "gzip",
                    "protocol_documentation": "docs/kex.md"},
            "WR": {"host": "wr.pool.example", "port": 9000, "certificate_version": 3,
                   "protocol_documentation": "https://docs.example/wr#protocol"},
        }})
        self.assertEqual(loaded["pool"]["KEX"]["ca_file"], str(self.directory / "certs/kex.pem"))
        self.assertEqual(loaded["pool"]["KEX"]["protocol_documentation"], str(self.directory / "docs/kex.md"))
        self.assertEqual(loaded["pool"]["WR"]["protocol_documentation"], "https://docs.example/wr#protocol")
        self.assertEqual(loaded["pool"]["WR"]["certificate_version"], 3)
        self.assertFalse(loaded["pool"]["WR"]["tls"])

    def test_inactive_pool_profiles_are_validated_with_safe_field_errors(self):
        for field, value in (("host", "user:PRIVATE_PROFILE_VALUE@pool.example"), ("port", True),
                             ("port", 65536), ("tls", 0), ("ca_file", 7),
                             ("compression", "unknown"), ("certificate_version", True),
                             ("certificate_version", 4),
                             ("protocol_documentation", "https://user:PRIVATE_PROFILE_VALUE@docs.example")):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, f"pool.WR.{field}") as caught:
                self.load({"version": 1, "pool": {"protocol": "KEX", "WR": {field: value}}})
            self.assertNotIn("PRIVATE_PROFILE_VALUE", str(caught.exception))
        with self.assertRaisesRegex(ConfigError, "pool.WR.ca_file requires pool.WR.tls"):
            self.load({"version": 1, "pool": {"WR": {"ca_file": "cert.pem"}}})

    def test_pool_profiles_only_allow_connection_settings(self):
        for field in ("WR_wallet", "KEX_uname", "worker", "password", "protocol", "PRIVATE_PROFILE_FIELD"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "pool.WR contains an unknown field") as caught:
                self.load({"version": 1, "pool": {"WR": {field: "PRIVATE_PROFILE_VALUE"}}})
            self.assertNotIn("PRIVATE_PROFILE_VALUE", str(caught.exception))
            self.assertNotIn("PRIVATE_PROFILE_FIELD", str(caught.exception))
        for value in (None, [], "profile", True):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "pool.KEX must be an object"):
                self.load({"version": 1, "pool": {"KEX": value}})

    def test_unknown_request_settings_have_strict_types_and_relative_paths(self):
        result = self.load({"version": 1, "pool": {
            "stop_on_unknown_request": False, "unknown_request_dir": "records/unknown",
        }})
        self.assertFalse(result["pool"]["stop_on_unknown_request"])
        self.assertEqual(result["pool"]["unknown_request_dir"], str(self.directory / "records/unknown"))
        for value in (0, 1, None, "true", []):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load({"version": 1, "pool": {"stop_on_unknown_request": value}})

    def test_invalid_protocol_compression_and_certificate_versions_are_rejected(self):
        for field, values in {
            "protocol": ["stratum", "kex", "wr", True, None, []],
            "compression": ["zstd", "auto", True, None, []],
            "certificate_version": [True, False, "3", 0, -1, 4, 3.0],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                    self.load({"version": 1, "pool": {field: value}})
        for version in (None, 1, 2, 3):
            with self.subTest(version=version):
                self.assertEqual(self.load({"version": 1, "pool": {"certificate_version": version}})["pool"]["certificate_version"], version)

    def test_agent_validation_does_not_echo_untrusted_content(self):
        secret = "PRIVATE_CONFIGURATION_VALUE"
        for agent in ("", "   ", "agent\n" + secret, "agent\x00", "x" * 129, None, 1):
            with self.subTest(agent=agent), self.assertRaises(ConfigError) as caught:
                self.load({"version": 1, "pool": {"agent": agent}})
            self.assertNotIn(secret, str(caught.exception))

    def test_solo_settings_and_passwords_are_preserved(self):
        result = self.load({"version": 1, "mode": "solo", "device": "cuda:2",
                            "node": {"username": "rpc-user", "password": "pass with spaces",
                                     "address": "payout-address"},
                            "training": {"max_batches": 10, "max_seconds": 1.5}})
        self.assertEqual(result["mode"], "solo")
        self.assertEqual(result["device"], "cuda:2")
        self.assertEqual(result["node"]["password"], "pass with spaces")
        self.assertEqual(result["training"]["max_seconds"], 1.5)

    def test_version_is_explicit_and_only_version_one_is_accepted(self):
        for value in ({}, {"version": True}, {"version": "1"}, {"version": 2}):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(value)

    def test_unknown_keys_in_every_section_are_rejected(self):
        for section in (None, "pool", "node", "training"):
            value = {"version": 1}
            if section is None:
                value["misspelled"] = 1
            else:
                value[section] = {"misspelled": 1}
            with self.subTest(section=section), self.assertRaisesRegex(ConfigError, "unknown field"):
                self.load(value)

    def test_booleans_are_never_accepted_as_numbers(self):
        for section, field in (("training", "m"), ("training", "max_batches"),
                               ("training", "max_seconds"), ("pool", "port"),
                               ("pool", "read_timeout"), ("node", "poll_interval")):
            with self.subTest(field=field), self.assertRaises(ConfigError):
                self.load({"version": 1, section: {field: True}})

    def test_numeric_ranges_and_nonfinite_numbers_are_rejected(self):
        invalid = [
            ("pool", "port", 0), ("pool", "port", 65536),
            ("pool", "connect_timeout", 0), ("pool", "read_timeout", -1),
            ("pool", "reconnect_delay", float("inf")),
            ("node", "rpc_timeout", float("nan")),
            ("training", "max_seconds", -1), ("training", "max_batches", 0.5),
            ("training", "k", 2049), ("training", "m", 64),
            ("training", "tile_h", 3), ("training", "tile_w", 128),
        ]
        for section, field, value in invalid:
            with self.subTest(section=section, field=field, value=value), self.assertRaises(ConfigError):
                self.load({"version": 1, section: {field: value}})

    def test_host_accepts_names_and_ips_but_rejects_urls_and_credentials(self):
        for host in ("", "pool.example", "localhost", "127.0.0.1", "::1"):
            with self.subTest(host=host):
                self.assertEqual(self.load({"version": 1, "pool": {"host": host}})["pool"]["host"], host)
        for host in ("stratum+tcp://pool.example", "user:secret@pool.example",
                     "pool.example:7048", "bad host", " ", "a..b", "-bad.example"):
            with self.subTest(host=host), self.assertRaises(ConfigError):
                self.load({"version": 1, "pool": {"host": host}})

    def test_automatic_pool_detection_requires_explicit_adapter_selection(self):
        with self.assertRaisesRegex(ConfigError, "explicitly select KEX or WR"):
            self.load({"version": 1, "pool": {"protocol": "auto", "host": "pool.example"}})

    def test_tls_and_ca_settings_must_match(self):
        for pool in ({"tls": "false"}, {"tls": 0}, {"ca_file": "ca.pem"}):
            with self.subTest(pool=pool), self.assertRaises(ConfigError):
                self.load({"version": 1, "pool": pool})
        with self.assertRaises(ConfigError):
            self.load({"version": 1, "node": {"rpc_url": "http://localhost:44107", "ca_file": "ca.pem"}})

    def test_device_and_section_types_are_checked(self):
        for changes in ({"mode": "unknown"}, {"device": "cpu"}, {"device": "cuda:-1"},
                        {"pool": []}, {"training": None}, {"node": "rpc"}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                self.load({"version": 1, **changes})

    def test_invalid_json_duplicates_and_unreadable_files_have_clean_errors(self):
        for raw in ('{"version": 1,', '{"version": 1,"version": 1}',
                    '{"version": 1,"pool":{"port":9000,"port":9001}}', '[1]'):
            self.path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                load_config(self.path)
        self.path.unlink()
        with self.assertRaisesRegex(ConfigError, "could not read"):
            load_config(self.path)

    def test_error_messages_do_not_echo_secrets(self):
        secret = "DO_NOT_LEAK_THIS_SECRET"
        for value in (
            {"version": 1, "pool": {"host": f"user:{secret}@pool.example"}},
            {"version": 1, "node": {"rpc_url": f"https://user:{secret}@node.example"}},
            {"version": 1, "pool": {secret: "accidentally a field"}},
            {"version": 1, "pool": {"password": {"bad": secret}}},
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError) as caught:
                self.load(value)
            self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
