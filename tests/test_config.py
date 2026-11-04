"""Configuration validation, path portability, and secret-safe errors."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, load_config
from LRAEP_trainer.paths import artifacts_dir
from LRAEP_trainer.wire_constants import s7_legacy_alias


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "trainer.json"

    def load(self, value):
        self.path.write_text(json.dumps(value))
        return load_config(self.path)

    def test_defaults_and_empty_S3_identities_are_valid(self):
        result = self.load({"version": 1})
        self.assertEqual(result["mode"], "S3")
        self.assertEqual(result["device"], "cuda:0")
        self.assertEqual(result["official_source"], str(self.directory / "src/LRAEP_helpers"))
        self.assertEqual(result["training"]["output_dir"], str(artifacts_dir().resolve()))
        self.assertEqual(result["training"]["job_sequence"], [{"title": 'job-0', "time": 60, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 120, "max_gpu_utilization": 70}, {"title": 'job-0', "time": 180, "max_gpu_utilization": 10}])
        self.assertNotIn("side_job_effort", result["training"])
        self.assertNotIn("max_gpu_utilization", result["training"])
        self.assertEqual(result["S3"]["host"], "")
        self.assertEqual(result["S3"]["port"], 7048)
        self.assertFalse(result["S3"]["tls"])
        self.assertEqual(result["S3"]["WR_S2"], "")
        self.assertEqual(result["S3"]["KEX_uname"], "")
        self.assertNotIn("wallet", result["S3"])
        self.assertEqual(result["S3"]["password"], "x")
        self.assertEqual(result["S3"]["protocol"], "KEX")
        self.assertIsNone(result["S3"]["protocol_documentation"])
        self.assertEqual(result["S3"]["compression"], "none")
        self.assertIsNone(result["S3"]["certificate_version"])
        self.assertEqual(result["S3"]["agent"], "LRAEP-trainer/0.1")
        self.assertTrue(result["S3"]["stop_on_unknown_request"])
        self.assertIsNone(result["S3"]["unknown_request_dir"])
        self.assertEqual(result["node"]["rpc_url"], "https://127.0.0.1:44107")

    def test_job_sequence_preserves_order_pause_exit_and_roles(self):
        sequence = [{"title": 'job-0', "time": 0.5, "max_gpu_utilization": 100}, {"title": 'job-1', "time": 2, "max_gpu_utilization": 1}, {"title": 'job-0', "time": 1, "max_gpu_utilization": 0}, {"title": 'job-1', "time": 0, "max_gpu_utilization": -1}]
        original = copy.deepcopy(sequence)
        result = self.load({"version": 1, "training": {"job_sequence": sequence}})
        self.assertEqual(result["training"]["job_sequence"],
                         [{"title": 'job-0', "time": 0.5, "max_gpu_utilization": 100}, {"title": 'job-1', "time": 2, "max_gpu_utilization": 1}, {"title": 'job-0', "time": 1, "max_gpu_utilization": 0}, {"title": 'job-1', "time": 0, "max_gpu_utilization": -1}])
        self.assertEqual(sequence, original)
        self.assertNotIn("main_effort", result["training"])
        self.assertNotIn("side_job_effort", result["training"])

    def test_job_sequence_rejects_invalid_entries_without_echoing_values(self):
        malformed = (None, True, "PRIVATE_SEQUENCE", {}, [], ['job-0', 1, 80],
                     [[1, 80]], [['job-0']], [['job-0', 1]], [['job-0', 1, 80, 2, 0]],
                     [["MAIN", 1, 80]], [["Side-job", 1, 80]], [["PRIVATE_SEQUENCE", 1, 80]],
                     [[None, 1, 80]], [[True, 1, 80]], [[[], 1, 80]], [[{}, 1, 80]],
                     [{"title": 'job-0', "time": 0, "max_gpu_utilization": 80}], [{"title": 'job-1', "time": -1, "max_gpu_utilization": 80}], [{"title": 'job-0', "time": True, "max_gpu_utilization": 80}],
                     [{"title": 'job-0', "time": "1", "max_gpu_utilization": 80}], [{"title": 'job-0', "time": 1, "max_gpu_utilization": -2}], [{"title": 'job-1', "time": 1, "max_gpu_utilization": 101}],
                     [{"title": 'job-0', "time": 1, "max_gpu_utilization": True}], [{"title": 'job-0', "time": 1, "max_gpu_utilization": 70.5}],
                     [{"title": 'job-0', "time": 1, "max_gpu_utilization": "PRIVATE_SEQUENCE"}], [{"title": 'job-0', "time": -1, "max_gpu_utilization": -1}])
        for value in malformed:
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "job_sequence") as caught:
                self.load({"version": 1, "training": {"job_sequence": value}})
            self.assertNotIn("PRIVATE_SEQUENCE", str(caught.exception))

    def test_job_sequence_durations_must_be_finite_for_both_roles(self):
        for duration in (float("inf"), float("-inf"), float("nan")):
            for role in ('job-0', 'job-1'):
                with self.subTest(role=role, duration=duration), self.assertRaises(ConfigError):
                    self.load({"version": 1, "training": {"job_sequence": [[role, duration, 80]]}})

    def test_separate_effort_keys_are_rejected_without_echoing_values(self):
        for field in ("main_effort", "side_job_effort"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "replaced by training.job_sequence") as caught:
                self.load({"version": 1, "training": {field: "PRIVATE_SEQUENCE"}})
            self.assertNotIn("PRIVATE_SEQUENCE", str(caught.exception))

    def test_legacy_gpu_limit_becomes_a_constant_schedule(self):
        for percent in (1, 70, 100):
            with self.subTest(percent=percent):
                result = self.load({"version": 1, "training": {"max_gpu_utilization": percent}})
                self.assertEqual(result["training"]["job_sequence"], [{"title": 'job-0', "time": 60, "max_gpu_utilization": percent}])
                self.assertNotIn("max_gpu_utilization", result["training"])
        for value in (0, -1, 101, True, 80.5, "80", None):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
                self.load({"version": 1, "training": {"max_gpu_utilization": value}})

    def test_explicit_job_sequence_takes_precedence_over_legacy_limit(self):
        result = self.load({"version": 1, "training": {
            "job_sequence": [{"title": 'job-0', "time": 30, "max_gpu_utilization": 40}], "max_gpu_utilization": 70,
        }})
        self.assertEqual(result["training"]["job_sequence"], [{"title": 'job-0', "time": 30, "max_gpu_utilization": 40}])
        with self.assertRaisesRegex(ConfigError, "training.max_gpu_utilization"):
            self.load({"version": 1, "training": {"job_sequence": [{"title": 'job-0', "time": 30, "max_gpu_utilization": 40}], "max_gpu_utilization": 101}})

    def test_effort_defaults_are_independent_between_loads_and_export(self):
        result = self.load({"version": 1})
        result["training"]["job_sequence"].append({"title": "job-1", "time": 1, "max_gpu_utilization": 100})
        again = self.load({"version": 1})
        self.assertEqual(again["training"]["job_sequence"], [{"title": 'job-0', "time": 60, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 120, "max_gpu_utilization": 70}, {"title": 'job-0', "time": 180, "max_gpu_utilization": 10}])
        self.assertNotIn("side_job_effort", again["training"])
        exported = copy.deepcopy(DEFAULT_CONFIG["training"])
        try:
            DEFAULT_CONFIG["training"]["job_sequence"][0]["max_gpu_utilization"] = 99
            self.assertEqual(self.load({"version": 1})["training"]["job_sequence"][0], {"title": "job-0", "time": 60, "max_gpu_utilization": 80})
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
            "S3": {"tls": True, "ca_file": "certs/S3.pem",
                     "protocol_documentation": "docs/protocol.json"},
            "node": {"ca_file": "certs/node.pem"},
        })
        self.assertEqual(result["official_source"], str(self.directory / "helpers/LRAEP"))
        self.assertEqual(result["training"]["output_dir"], str(self.directory / "results"))
        self.assertEqual(result["S3"]["ca_file"], str(self.directory / "certs/S3.pem"))
        self.assertEqual(result["node"]["ca_file"], str(self.directory / "certs/node.pem"))
        self.assertEqual(result["S3"]["protocol_documentation"], str(self.directory / "docs/protocol.json"))

    def test_documentation_url_is_preserved_without_loading_it(self):
        url = "https://pool.example/en/prl#framing"
        result = self.load({"version": 1, "S3": {"protocol_documentation": url}})
        self.assertEqual(result["S3"]["protocol_documentation"], url)

    def test_exported_example_defaults_can_be_loaded_without_mutation(self):
        result = self.load(DEFAULT_CONFIG)
        self.assertEqual(result["version"], 1)
        self.assertEqual(DEFAULT_CONFIG["official_source"], "src/LRAEP_helpers")
        self.assertEqual(DEFAULT_CONFIG["training"]["output_dir"], str(artifacts_dir()))

    def test_explicit_S3_protocol_options_are_preserved(self):
        result = self.load({"version": 1, "S3": {
            "protocol": "KEX", "compression": "gzip", "certificate_version": 3,
            "agent": "my-custom-trainer/2", "KEX_uname": "my-account", "WR_S2": "my-wr-address",
            "worker": "rig2",
            "password": "d=2097152",
        }})
        self.assertEqual(result["S3"]["protocol"], "KEX")
        self.assertEqual(result["S3"]["compression"], "gzip")
        self.assertEqual(result["S3"]["certificate_version"], 3)
        self.assertEqual(result["S3"]["agent"], "my-custom-trainer/2")
        self.assertEqual(result["S3"]["KEX_uname"], "my-account")
        self.assertEqual(result["S3"]["WR_S2"], "my-wr-address")
        self.assertEqual(result["S3"]["password"], "d=2097152")
        for protocol in ("S7", "WR", s7_legacy_alias()):
            with self.subTest(protocol=protocol):
                self.assertEqual(self.load({"version": 1, "S3": {"protocol": protocol}})["S3"]["protocol"], "WR")

    def test_S3_identity_fields_are_separate_and_case_sensitive(self):
        for protocol in ("KEX", "WR"):
            loaded = self.load({"version": 1, "S3": {
                "protocol": protocol, "KEX_uname": "kex-account", "WR_S2": "wr-payout-address",
            }})
            self.assertEqual(loaded["S3"]["KEX_uname"], "kex-account")
            self.assertEqual(loaded["S3"]["WR_S2"], "wr-payout-address")
        for field in ("wallet", "kex_uname", "wr_S2", "KEX_S2", "WR_uname"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "unknown field") as caught:
                self.load({"version": 1, "S3": {field: "PRIVATE_IDENTITY"}})
            self.assertNotIn("PRIVATE_IDENTITY", str(caught.exception))

    def test_S3_identity_fields_require_strings_without_controls(self):
        for field in ("KEX_uname", "WR_S2"):
            for value in (None, True, 17, [], {}, "PRIVATE_IDENTITY\n"):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, field) as caught:
                    self.load({"version": 1, "S3": {field: value}})
                self.assertNotIn("PRIVATE_IDENTITY", str(caught.exception))

    def test_S3_profiles_resolve_paths_beside_one_configuration_file(self):
        loaded = self.load({"version": 1, "S3": {
            "KEX": {"host": "kex.S3.example", "port": 8048, "tls": True,
                    "ca_file": "certs/kex.pem", "compression": "gzip",
                    "protocol_documentation": "docs/kex.md"},
            "WR": {"host": "wr.S3.example", "port": 9000, "certificate_version": 3,
                   "protocol_documentation": "https://docs.example/wr#protocol"},
        }})
        self.assertEqual(loaded["S3"]["KEX"]["ca_file"], str(self.directory / "certs/kex.pem"))
        self.assertEqual(loaded["S3"]["KEX"]["protocol_documentation"], str(self.directory / "docs/kex.md"))
        self.assertEqual(loaded["S3"]["WR"]["protocol_documentation"], "https://docs.example/wr#protocol")
        self.assertEqual(loaded["S3"]["WR"]["certificate_version"], 3)
        self.assertFalse(loaded["S3"]["WR"]["tls"])

    def test_inactive_S3_profiles_are_validated_with_safe_field_errors(self):
        for field, value in (("host", "user:PRIVATE_PROFILE_VALUE@S3.example"), ("port", True),
                             ("port", 65536), ("tls", 0), ("ca_file", 7),
                             ("compression", "unknown"), ("certificate_version", True),
                             ("certificate_version", 4),
                             ("protocol_documentation", "https://user:PRIVATE_PROFILE_VALUE@docs.example")):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, f"S3.WR.{field}") as caught:
                self.load({"version": 1, "S3": {"protocol": "KEX", "WR": {field: value}}})
            self.assertNotIn("PRIVATE_PROFILE_VALUE", str(caught.exception))
        with self.assertRaisesRegex(ConfigError, "S3.WR.ca_file requires S3.WR.tls"):
            self.load({"version": 1, "S3": {"WR": {"ca_file": "cert.pem"}}})

    def test_S3_profiles_only_allow_connection_settings(self):
        for field in ("WR_S2", "KEX_uname", "worker", "password", "protocol", "PRIVATE_PROFILE_FIELD"):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "S3.WR contains an unknown field") as caught:
                self.load({"version": 1, "S3": {"WR": {field: "PRIVATE_PROFILE_VALUE"}}})
            self.assertNotIn("PRIVATE_PROFILE_VALUE", str(caught.exception))
            self.assertNotIn("PRIVATE_PROFILE_FIELD", str(caught.exception))
        for value in (None, [], "profile", True):
            with self.subTest(value=value), self.assertRaisesRegex(ConfigError, "S3.KEX must be an object"):
                self.load({"version": 1, "S3": {"KEX": value}})

    def test_unknown_request_settings_have_strict_types_and_relative_paths(self):
        result = self.load({"version": 1, "S3": {
            "stop_on_unknown_request": False, "unknown_request_dir": "records/unknown",
        }})
        self.assertFalse(result["S3"]["stop_on_unknown_request"])
        self.assertEqual(result["S3"]["unknown_request_dir"], str(self.directory / "records/unknown"))
        for value in (0, 1, None, "true", []):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load({"version": 1, "S3": {"stop_on_unknown_request": value}})

    def test_invalid_protocol_compression_and_certificate_versions_are_rejected(self):
        for field, values in {
            "protocol": ["S4", "kex", "wr", True, None, []],
            "compression": ["zstd", "auto", True, None, []],
            "certificate_version": [True, False, "3", 0, -1, 4, 3.0],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ConfigError):
                    self.load({"version": 1, "S3": {field: value}})
        for version in (None, 1, 2, 3):
            with self.subTest(version=version):
                self.assertEqual(self.load({"version": 1, "S3": {"certificate_version": version}})["S3"]["certificate_version"], version)

    def test_agent_validation_does_not_echo_untrusted_content(self):
        secret = "PRIVATE_CONFIGURATION_VALUE"
        for agent in ("", "   ", "agent\n" + secret, "agent\x00", "x" * 129, None, 1):
            with self.subTest(agent=agent), self.assertRaises(ConfigError) as caught:
                self.load({"version": 1, "S3": {"agent": agent}})
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
        for section in (None, "S3", "node", "training"):
            value = {"version": 1}
            if section is None:
                value["misspelled"] = 1
            else:
                value[section] = {"misspelled": 1}
            with self.subTest(section=section), self.assertRaisesRegex(ConfigError, "unknown field"):
                self.load(value)

    def test_booleans_are_never_accepted_as_numbers(self):
        for section, field in (("training", "m"), ("training", "max_batches"),
                               ("training", "max_seconds"), ("S3", "port"),
                               ("S3", "read_timeout"), ("node", "poll_interval")):
            with self.subTest(field=field), self.assertRaises(ConfigError):
                self.load({"version": 1, section: {field: True}})

    def test_numeric_ranges_and_nonfinite_numbers_are_rejected(self):
        invalid = [
            ("S3", "port", 0), ("S3", "port", 65536),
            ("S3", "connect_timeout", 0), ("S3", "read_timeout", -1),
            ("S3", "reconnect_delay", float("inf")),
            ("node", "rpc_timeout", float("nan")),
            ("training", "max_seconds", -1), ("training", "max_batches", 0.5),
            ("training", "k", 2049), ("training", "m", 64),
            ("training", "tile_h", 3), ("training", "tile_w", 128),
        ]
        for section, field, value in invalid:
            with self.subTest(section=section, field=field, value=value), self.assertRaises(ConfigError):
                self.load({"version": 1, section: {field: value}})

    def test_host_accepts_names_and_ips_but_rejects_urls_and_credentials(self):
        for host in ("", "S3.example", "localhost", "127.0.0.1", "::1"):
            with self.subTest(host=host):
                self.assertEqual(self.load({"version": 1, "S3": {"host": host}})["S3"]["host"], host)
        for host in ("stratum+tcp://pool.example", "user:secret@S3.example",
                     "S3.example:7048", "bad host", " ", "a..b", "-bad.example"):
            with self.subTest(host=host), self.assertRaises(ConfigError):
                self.load({"version": 1, "S3": {"host": host}})

    def test_automatic_S3_detection_requires_explicit_adapter_selection(self):
        with self.assertRaisesRegex(ConfigError, "explicitly select KEX or WR"):
            self.load({"version": 1, "S3": {"protocol": "auto", "host": "S3.example"}})

    def test_tls_and_ca_settings_must_match(self):
        for S3 in ({"tls": "false"}, {"tls": 0}, {"ca_file": "ca.pem"}):
            with self.subTest(S3=S3), self.assertRaises(ConfigError):
                self.load({"version": 1, "S3": S3})
        with self.assertRaises(ConfigError):
            self.load({"version": 1, "node": {"rpc_url": "http://localhost:44107", "ca_file": "ca.pem"}})

    def test_device_and_section_types_are_checked(self):
        for changes in ({"mode": "unknown"}, {"device": "cpu"}, {"device": "cuda:-1"},
                        {"S3": []}, {"training": None}, {"node": "rpc"}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                self.load({"version": 1, **changes})

    def test_invalid_json_duplicates_and_unreadable_files_have_clean_errors(self):
        for raw in ('{"version": 1,', '{"version": 1,"version": 1}',
                    '{"version": 1,"S3":{"port":9000,"port":9001}}', '[1]'):
            self.path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                load_config(self.path)
        self.path.unlink()
        with self.assertRaisesRegex(ConfigError, "could not read"):
            load_config(self.path)

    def test_error_messages_do_not_echo_secrets(self):
        secret = "DO_NOT_LEAK_THIS_SECRET"
        for value in (
            {"version": 1, "S3": {"host": f"user:{secret}@S3.example"}},
            {"version": 1, "node": {"rpc_url": f"https://user:{secret}@node.example"}},
            {"version": 1, "S3": {secret: "accidentally a field"}},
            {"version": 1, "S3": {"password": {"bad": secret}}},
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError) as caught:
                self.load(value)
            self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
