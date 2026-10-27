"""Pure wire-format tests; no GPU, network connection, or native helper needed."""

import base64
import gzip
import unittest

from LRAEP_trainer.pool_protocol import KEXDialect, PoolJob, PoolProtocolError


# A real, redacted-wallet KEX TLS capture from 2026-09-26, pinned in
# spark-pearl-miner 65365633e3bff61b0f29d72109622dd22ff5a379.
CAPTURED_JOB = {
    "header": "000000200744ad4ffd147f89b5a2561e8aceb08a57a657f118c8decf385e658bbcbee9df9ee575bcd77d8901894f6f9e6e0b15cd490b5b6d7f211d0eaa6ecd630ccd6a2add0fb86a6a910018",
    "height": 119376,
    "job_id": "4bdc9bb9_2097152",
    "target": "00000000000007ffffffffffffffffffffffffffffffffffffffffffffffffff",
    "cert_version": 3,
}


class KEXProtocolTests(unittest.TestCase):
    def setUp(self):
        self.dialect = KEXDialect("public-wallet", "rig0", agent="our-client/1")

    def test_capture_preserves_header_bytes_and_base_target(self):
        job = self.dialect.parse_job(CAPTURED_JOB)
        self.assertEqual(job.header_bytes, bytes.fromhex(CAPTURED_JOB["header"]))
        self.assertEqual(job.header_bytes[:4], (0x20000000).to_bytes(4, "little"))
        self.assertEqual(job.cert_version, 3)
        self.assertEqual(job.height, 119376)
        self.assertEqual(job.target, (1 << 224) // 2097152 - 1)
        self.assertNotEqual(job.target, (0xFFFF << 208) // 2097152)
        self.assertEqual(job.raw_target, CAPTURED_JOB["target"])

    def test_v1_capture_handshake_does_not_wait_for_subscribe(self):
        requests = self.dialect.auth_requests()
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], {
            "jsonrpc": "2.0", "id": 1, "method": "mining.subscribe", "params": ["our-client/1"],
        })
        self.assertEqual(requests[1], {
            "jsonrpc": "2.0", "id": 2, "method": "mining.authorize", "params": ["public-wallet.rig0", "x"],
        })

    def test_operator_gzip_spec_login_and_ack(self):
        dialect = KEXDialect("public-wallet", "rig0", password="d=2097152", agent="our-client/1", request_gzip=True)
        self.assertEqual(dialect.auth_requests(start_id=7), [{
            "jsonrpc": "2.0", "id": 7, "method": "mining.authorize", "params": {
                "wallet": "public-wallet.rig0", "agent": "our-client/1", "password": "d=2097152", "type": "v2",
            },
        }])
        self.assertTrue(dialect.negotiate_gzip({"id": 7, "result": True, "error": None, "type": "v2"}))
        for response in [
            {"id": 7, "result": True},
            {"id": 7, "result": True, "type": "plain"},
            {"id": 7, "result": True, "type": "v3"},
            {"id": 7, "result": False, "type": "v2"},
            {"id": 7, "result": True, "error": {"code": 20}, "type": "v2"},
        ]:
            with self.subTest(response=response):
                self.assertFalse(dialect.negotiate_gzip(response))
        self.assertFalse(self.dialect.negotiate_gzip({"result": True, "type": "v2"}))

    def test_strict_acceptance(self):
        self.assertTrue(self.dialect.accept({"id": 9, "result": True, "error": None}))
        for result in [False, None, 1, "true", {}, []]:
            self.assertFalse(self.dialect.accept({"id": 9, "result": result}))
        self.assertFalse(self.dialect.accept({"id": 9, "result": True, "error": "rejected"}))
        self.assertFalse(self.dialect.accept({"method": "mining.notify", "result": True}))

    def test_plain_submission_preserves_native_serialization_byte_for_byte(self):
        job = self.dialect.parse_job(CAPTURED_JOB)
        raw = b"opaque official bincode proof\x00"
        b64 = base64.b64encode(raw).decode()
        request = self.dialect.submit_request(job, b64, request_id=13)
        self.assertEqual(request, {
            "jsonrpc": "2.0", "id": 13, "method": "mining.submit",
            "params": {"worker": "public-wallet.rig0", "job_id": job.job_id, "plain_proof": b64},
        })
        self.assertEqual(base64.b64decode(request["params"]["plain_proof"]), raw)

    def test_gzip_wraps_official_bytes_only_after_negotiation(self):
        dialect = KEXDialect("public-wallet", "rig0", request_gzip=True)
        job = dialect.parse_job(CAPTURED_JOB)
        raw = b"opaque official bincode proof\x00" * 100
        b64 = base64.b64encode(raw).decode()
        plain = dialect.submit_request(job, b64, request_id=10, negotiated_gzip=False)
        self.assertEqual(plain["params"]["plain_proof"], b64)
        compressed = dialect.submit_request(job, b64, request_id=11, negotiated_gzip=True)
        wire = base64.b64decode(compressed["params"]["plain_proof"])
        self.assertEqual(wire[:2], b"\x1f\x8b")
        self.assertEqual(gzip.decompress(wire), raw)
        with self.assertRaises(PoolProtocolError):
            self.dialect.submit_request(job, b64, 12, negotiated_gzip=True)

    def test_certificate_version_is_never_guessed(self):
        missing = {k: v for k, v in CAPTURED_JOB.items() if k != "cert_version"}
        with self.assertRaisesRegex(PoolProtocolError, "missing cert_version"):
            self.dialect.parse_job(missing)
        self.assertEqual(self.dialect.parse_job(missing, configured_certificate_version=3).cert_version, 3)
        self.assertEqual(self.dialect.parse_job(CAPTURED_JOB, configured_certificate_version=2).cert_version, 3)
        for version in [None, True, "3", 0, 4, -1]:
            with self.subTest(version=version), self.assertRaises(PoolProtocolError):
                self.dialect.parse_job({**CAPTURED_JOB, "cert_version": version})

    def test_malformed_jobs_fail_before_work_starts(self):
        bad_fields = [
            {"job_id": ""}, {"job_id": "x" * 129},
            {"header": "00" * 75}, {"header": "00 " * 76}, {"header": "gg" * 76},
            {"target": "0"}, {"target": "f" * 65}, {"target": " ff"}, {"target": 3},
            {"height": True}, {"height": -1}, {"height": "12"},
        ]
        for replacement in bad_fields:
            with self.subTest(replacement=replacement), self.assertRaises(PoolProtocolError):
                self.dialect.parse_job({**CAPTURED_JOB, **replacement})
        with self.assertRaises(PoolProtocolError):
            self.dialect.parse_job([])

    def test_unknown_optional_job_fields_are_ignored(self):
        expected = self.dialect.parse_job(CAPTURED_JOB)
        self.assertEqual(self.dialect.parse_job({**CAPTURED_JOB, "new_optional_field": {"a": 1}}), expected)

    def test_noncanonical_base64_is_not_sent(self):
        job = self.dialect.parse_job(CAPTURED_JOB)
        for proof in ["", "not base64!", "YQ==\n", "YR==", "YQ===", "é", None]:
            with self.subTest(proof=proof), self.assertRaises(PoolProtocolError):
                self.dialect.submit_request(job, proof, 3)

    def test_oversized_proof_is_not_sent(self):
        job = self.dialect.parse_job(CAPTURED_JOB)
        proof = base64.b64encode(b"\x00" * (4 * 1024 * 1024 + 1)).decode()
        with self.assertRaisesRegex(PoolProtocolError, "4 MiB"):
            self.dialect.submit_request(job, proof, 3)

    def test_invalid_ids_and_control_characters(self):
        for value in [0, -1, True, "1"]:
            with self.subTest(value=value), self.assertRaises(PoolProtocolError):
                self.dialect.auth_requests(value)
        for wallet, worker in [("wallet\n", "worker"), ("wallet", "wo rker"), ("", "worker")]:
            with self.subTest(wallet=wallet, worker=worker), self.assertRaises(PoolProtocolError):
                KEXDialect(wallet, worker)


if __name__ == "__main__":
    unittest.main()
