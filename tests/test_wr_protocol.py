"""Independent capture vectors and conservative PearlHash schema validation."""

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

from LRAEP_trainer import pearl_codec
from LRAEP_trainer.pearl_requests import build_submit_request
from LRAEP_trainer.pool_protocol import PoolProtocolError
from LRAEP_trainer.wr_protocol import (
    UnknownWRRequest, WRDialect, WRShare,
)


FIXTURES = Path(__file__).parent / "fixtures"


def fixtures(name):
    return json.loads((FIXTURES / ("wr_" + name + ".json")).read_text())


def job_result():
    return next(frame["message"]["result"] for frame in fixtures("pearl_codec_frames")
                if isinstance(frame["message"].get("result"), list)
                and frame["message"]["result"][0] == "notify")


def compact_share(job):
    return WRShare(
        miner_nonce=bytes(8), worker_nonce=job.worker_nonce, m=16, n=16, k=64,
        rank=128, tile_h=8, tile_w=8, a_rows=tuple(range(8)), b_cols=tuple(range(8)),
    )


class WRCodecTests(unittest.TestCase):
    def test_real_capture_plaintext_and_wire_are_exact(self):
        for frame in fixtures("pearl_codec_frames"):
            with self.subTest(line=frame["source_line"], direction=frame["direction"]):
                payload = bytes.fromhex(frame["plaintext_hex"])
                self.assertEqual(pearl_codec.decode_payload(frame["wire"]), payload)
                self.assertEqual(pearl_codec.decode_frame(frame["wire"]), frame["message"])
                self.assertEqual(
                    pearl_codec.encode_payload(payload, bytes.fromhex(frame["wire"][:8])),
                    frame["wire"].encode() + b"\n",
                )

    def test_seed_endianness_matches_captured_key(self):
        expected = bytes.fromhex(
            "8c687a466f88f5b34ebc16bdf575c532aad832fa3fe03e3fda0ffd0bb8c25a66"
            "ce0eb4aa068c0a1c2a66db7c3daaa0c9"
        )
        self.assertEqual(pearl_codec.derive_key(0x74F4785A), expected)
        self.assertEqual(pearl_codec.derive_key(bytes.fromhex("74f4785a")), expected)
        self.assertEqual(pearl_codec.derive_key(0), bytes(48))

    def test_malformed_frames_and_ambiguous_json_fail(self):
        for payload in [b'{"id":1,"id":2}', b'[]', b'{"n":NaN}', b'{"s":"\xff"}',
                        b'{"id":' + b'1' * 5000 + b'}']:
            with self.subTest(payload=payload), self.assertRaises(pearl_codec.CodecError):
                pearl_codec.decode_frame(pearl_codec.encode_payload(payload, 0))
        for frame in [b"", b"00000000", b"00000000f", b"00000000gg", b"0000 00007b7d"]:
            with self.subTest(frame=frame), self.assertRaises(pearl_codec.CodecError):
                pearl_codec.decode_frame(frame)

    def test_three_accepted_share_requests_reproduce_exact_wire(self):
        for fixture in fixtures("accepted_pool_shares"):
            request = fixture["request"]
            fields = {key: value for key, value in request["params"][0].items()
                      if key not in {"nonce", "tail_mode", "sb_p"}}
            built = build_submit_request(request["id"], **fields)
            self.assertEqual(built, request)
            self.assertEqual(pearl_codec.encode_frame(built, bytes.fromhex(fixture["request_wire"][:8])),
                             fixture["request_wire"].encode() + b"\n")
            reply = pearl_codec.decode_frame(fixture["response_wire"])
            self.assertEqual(reply, fixture["response"])
            self.assertTrue(WRDialect.accept(reply))


class WRProtocolTests(unittest.TestCase):
    def setUp(self):
        self.dialect = WRDialect(WR_wallet="test-wallet", worker="test-worker", agent="our-client/1")

    def test_login_keeps_wr_wallet_worker_agent_separate(self):
        self.assertEqual(self.dialect.auth_requests(0), [{
            "id": 0, "method": "login", "params": ["test-wallet", "test-worker", "our-client/1"],
        }])

    def test_wr_wallet_validation_names_only_the_selected_field(self):
        for value in ("", None, "has whitespace"):
            with self.subTest(value=value), self.assertRaisesRegex(PoolProtocolError, "WR_wallet"):
                WRDialect(WR_wallet=value, worker="test-worker")

    def test_gpu_reports_match_recorded_raw_rates_and_device_names(self):
        reports = [frame for frame in fixtures("pearl_codec_frames")
                   if frame["message"].get("method") == "report_info"]
        self.assertEqual(len(reports), 8)
        for frame in reports:
            captured = frame["message"]
            device = captured["params"][0]
            with self.subTest(request_id=captured["id"]):
                report = self.dialect.report_request(device["name"], device["hashrate"], captured["id"])
                self.assertEqual(report, captured)
                self.assertEqual(report, pearl_codec.decode_frame(frame["wire"]))
                self.assertEqual(list(report["params"][0]), ["hashrate", "name"])
                self.assertIsInstance(report["params"][0]["hashrate"], float)
                self.assertEqual(report["params"][0]["name"], "NVIDIA GeForce GTX 1660 SUPER ")
                self.assertNotIn("WR_wallet", report)

    def test_gpu_reports_preserve_raw_h_per_second_and_printable_name(self):
        for value in (0, 0.0, 1, 4_000_000_000_000.0, sys.float_info.max):
            with self.subTest(value=value):
                report = self.dialect.report_request("GPU \u03bb ", value, (1 << 64) - 1)
                device = report["params"][0]
                self.assertEqual(device["name"], "GPU \u03bb ")
                self.assertEqual(device["hashrate"], float(value))
                self.assertTrue(math.isfinite(device["hashrate"]))
        self.assertEqual(self.dialect.report_request("G" * 256, 0, 0)["id"], 0)

    def test_gpu_report_rejects_invalid_rates_names_and_ids_without_echoing_them(self):
        for value in (True, False, None, "PRIVATE_RATE", -1, float("nan"), float("inf"),
                      -float("inf"), 10 ** 400):
            with self.subTest(rate=value), self.assertRaisesRegex(PoolProtocolError, "hashrate") as caught:
                self.dialect.report_request("GPU", value, 1)
            self.assertNotIn("PRIVATE_RATE", str(caught.exception))
        for name in ("", "   ", "PRIVATE_GPU\n", "PRIVATE_GPU\x7f", "PRIVATE_GPU\u202e", "G" * 257, None, 17):
            with self.subTest(name=name), self.assertRaisesRegex(PoolProtocolError, "gpu_name") as caught:
                self.dialect.report_request(name, 1.0, 1)
            self.assertNotIn("PRIVATE_GPU", str(caught.exception))
        for request_id in (-1, 1 << 64, True, 1.0, None):
            with self.subTest(request_id=request_id), self.assertRaisesRegex(PoolProtocolError, "request ID"):
                self.dialect.report_request("GPU", 1.0, request_id)

    def test_three_recorded_jobs_recover_native_verified_headers(self):
        for fixture in fixtures("pearl_jobs"):
            result = list(job_result())
            result[1] = fixture["notify_payload_hex"]
            job = self.dialect.parse_job(result, worker_nonce=bytes.fromhex("19df092c6c027fee"))
            self.assertEqual(job.header_bytes.hex(), fixture["header_hex"])
            self.assertEqual(job.notify_payload.hex(), fixture["notify_payload_hex"])
            self.assertEqual(job.target, 2 << 200)
            self.assertEqual(job.target, int.from_bytes(bytes.fromhex(result[2]), "little"))
            self.assertEqual(job.cert_version, 3)
            self.assertEqual(job.height, 0)
            self.assertEqual(job.worker_nonce.hex(), "19df092c6c027fee")

    def test_unknown_commands_are_distinct_from_corrupt_known_jobs(self):
        for message in [{"method": "run", "params": ["example"]},
                        {"id": -1, "status": 1, "result": ["new_command", "x"]}]:
            with self.assertRaises(UnknownWRRequest):
                self.dialect.classify(message)
        for message in [{"id": -1, "status": 1, "result": ["extranonce", "zz"]},
                        {"id": -1, "status": 1, "result": ["notify", "short"]},
                        {"id": True, "status": 1, "result": True},
                        {"id": 1 << 64, "status": 1, "result": True},
                        {"id": 2, "status": True, "result": True}]:
            with self.assertRaises(PoolProtocolError) as caught:
                self.dialect.classify(message)
            self.assertNotIsInstance(caught.exception, UnknownWRRequest)

    def test_unsupported_job_formats_fail(self):
        for field, value in [(1, "00" * 199), (1, "gg" * 200), (2, "00" * 32),
                             (2, "00" * 31), (3, "unsafe\njob"), (4, "4"),
                             (4, "2"), (4, "03")]:
            result = list(job_result())
            result[field] = value
            with self.subTest(field=field, value=value[:10]), self.assertRaises(PoolProtocolError):
                self.dialect.parse_job(result)
        for certificate in [True, 1, 2, 4]:
            with self.assertRaises(PoolProtocolError):
                self.dialect.parse_job(job_result(), certificate)

    def test_share_nonce_must_match_job_and_gzip_is_never_used(self):
        job = self.dialect.parse_job(job_result(), worker_nonce=bytes.fromhex("19df092c6c027fee"))
        share = compact_share(job)
        request = self.dialect.submit_request(job, share, 7)
        self.assertEqual(request["method"], "submit")
        self.assertEqual(request["params"][0]["worker_nonce"], job.worker_nonce.hex())
        self.assertEqual(request["params"][0]["miner_nonce"], "00" * 8)
        self.assertNotIn("plain_proof", request["params"][0])
        for incompatible in [replace(share, worker_nonce=b"x" * 8), "base64-proof"]:
            with self.assertRaises(PoolProtocolError):
                self.dialect.submit_request(job, incompatible, 7)
        with self.assertRaises(PoolProtocolError):
            self.dialect.submit_request(job, share, 7, True)


if __name__ == "__main__":
    unittest.main()
