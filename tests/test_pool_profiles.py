"""A single configuration selects independent pool connection profiles."""

import copy
import unittest

from LRAEP_trainer.config import ConfigError, DEFAULT_CONFIG, resolve_pool_settings


class PoolProfileResolutionTests(unittest.TestCase):
    def setUp(self):
        self.settings = copy.deepcopy(DEFAULT_CONFIG["pool"])
        self.settings.update(
            KEX_uname="kex-account", WR_wallet="wr-payout-address", worker="shared-rig",
            password="private-pool-password", connect_timeout=7, read_timeout=19,
            reconnect_delay=2,
            KEX={"host": "kex.pool.example", "port": 8048, "tls": True,
                 "ca_file": "/tmp/kex-ca.pem", "compression": "gzip",
                 "certificate_version": 3, "protocol_documentation": "/tmp/kex.md"},
            WR={"host": "wr.pool.example", "port": 9000, "tls": False,
                "compression": "none", "certificate_version": 3},
        )

    def test_explicit_selection_copies_only_its_profile_and_keeps_common_options(self):
        before = copy.deepcopy(self.settings)
        kex = resolve_pool_settings(self.settings, "KEX")
        wr = resolve_pool_settings(self.settings, "WR")
        self.assertEqual((kex["protocol"], kex["host"], kex["port"], kex["tls"]),
                         ("KEX", "kex.pool.example", 8048, True))
        self.assertEqual((wr["protocol"], wr["host"], wr["port"], wr["tls"]),
                         ("WR", "wr.pool.example", 9000, False))
        self.assertEqual(kex["ca_file"], "/tmp/kex-ca.pem")
        self.assertIsNone(wr["ca_file"])
        self.assertEqual(kex["compression"], "gzip")
        self.assertEqual(wr["compression"], "none")
        for effective in (kex, wr):
            for field in ("KEX_uname", "WR_wallet", "worker", "password", "connect_timeout",
                          "read_timeout", "reconnect_delay"):
                self.assertEqual(effective[field], self.settings[field])
            self.assertNotIn("KEX", effective)
            self.assertNotIn("WR", effective)
        kex["host"] = "changed.example"
        self.assertEqual(self.settings, before)

    def test_stored_protocol_and_alias_select_wr_without_a_separate_file(self):
        for protocol in ("WR", "pearlhash"):
            self.settings["protocol"] = protocol
            effective = resolve_pool_settings(self.settings)
            self.assertEqual(effective["protocol"], "WR")
            self.assertEqual(effective["host"], "wr.pool.example")
            self.assertEqual(resolve_pool_settings(self.settings, "KEX")["host"], "kex.pool.example")

    def test_missing_or_empty_selected_profile_never_uses_other_or_old_flat_endpoint(self):
        self.settings.update(host="old-flat.pool.example", port=9001, tls=True,
                             ca_file="/tmp/old-ca.pem", compression="gzip",
                             certificate_version=2, protocol_documentation="/tmp/old.md")
        del self.settings["WR"]
        for empty_profile in (False, True):
            if empty_profile:
                self.settings["WR"] = {}
            effective = resolve_pool_settings(self.settings, "WR")
            self.assertEqual(effective["host"], "")
            self.assertEqual(effective["port"], 7048)
            self.assertFalse(effective["tls"])
            self.assertIsNone(effective["ca_file"])
            self.assertEqual(effective["compression"], "none")
            self.assertIsNone(effective["certificate_version"])
            self.assertIsNone(effective["protocol_documentation"])
            self.assertEqual(effective["WR_wallet"], "wr-payout-address")

    def test_empty_profile_mode_does_not_fall_back_to_a_flat_host(self):
        for profile in ("KEX", "WR"):
            effective = resolve_pool_settings({"host": "old-flat.pool.example", profile: {}}, "WR")
            self.assertEqual(effective["host"], "")

    def test_legacy_flat_settings_remain_usable_and_resolution_is_idempotent(self):
        flat = copy.deepcopy(DEFAULT_CONFIG["pool"])
        flat.update(host="legacy.pool.example", protocol="KEX", port=8048, tls=True,
                    KEX_uname="legacy-account", compression="gzip")
        effective = resolve_pool_settings(flat)
        self.assertEqual(effective, flat)
        self.assertIsNot(effective, flat)
        self.assertEqual(resolve_pool_settings(effective), effective)
        self.assertEqual(resolve_pool_settings(effective, "WR")["protocol"], "WR")

    def test_invalid_protocol_and_profile_shapes_have_safe_errors(self):
        for protocol in ("auto", "invalid", "wr", None, True, []):
            with self.subTest(protocol=protocol), self.assertRaises(ConfigError):
                resolve_pool_settings({"protocol": protocol})
        for settings in (None, [], "PRIVATE_VALUE", {"WR": None}, {"KEX": []},
                         {"WR": {"WR_wallet": "PRIVATE_VALUE"}}):
            with self.subTest(settings=settings), self.assertRaises(ConfigError) as caught:
                resolve_pool_settings(settings)
            self.assertNotIn("PRIVATE_VALUE", str(caught.exception))

    def test_defaults_contain_no_profile_maps_or_pool_addresses(self):
        self.assertNotIn("KEX", DEFAULT_CONFIG["pool"])
        self.assertNotIn("WR", DEFAULT_CONFIG["pool"])
        self.assertEqual(resolve_pool_settings({})["host"], "")


if __name__ == "__main__":
    unittest.main()
