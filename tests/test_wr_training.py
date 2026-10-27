"""Real GPU work and compact-share exchange with a local encoded PearlHash peer."""

import copy
import itertools
import math
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from LRAEP_trainer.config import DEFAULT_CONFIG
from LRAEP_trainer import native_cuda, pool_training as worker
from LRAEP_trainer.pearl_codec import decode_frame, encode_frame
from LRAEP_trainer.protocol import load_official_helpers

from test_wr_protocol import job_result


class LocalWRGPUTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not worker.torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(cls.root / "src" / "LRAEP_helpers")
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest(f"official helper unavailable: {error}") from None

    def check_gpu_exchange(self, native):
        if native and not native_cuda.available("cuda:0"):
            self.skipTest("native CUDA unavailable")
        configuration = copy.deepcopy(DEFAULT_CONFIG)
        configuration["official_source"] = str(self.root / "src" / "LRAEP_helpers")
        configuration["device"] = "cuda:0"
        configuration["training"].update(m=128, n=128, k=2048, tile_h=8, tile_w=8,
                                          max_batches=1, max_seconds=30)
        configuration["pool"].update(protocol="WR", host="127.0.0.1", tls=False,
                                     WR_wallet="local-test-account", worker="gpu-test",
                                     compression="none", connect_timeout=3, read_timeout=15)
        notify = list(job_result())
        # Make every correctly computed GPU ticket eligible; the client still
        # executes its real arithmetic and independent candidate verification.
        factor = 8 * 8 * configuration["training"]["k"]
        target = ((1 << 256) - 1) // factor
        notify[2] = target.to_bytes(32, "little").hex()
        notify[3] = "local-gpu-job"
        extranonce = "19df092c6c027fee"
        observed, errors = {"reports": [], "request_ids": []}, []
        expected_gpu_name = worker.torch.cuda.get_device_name(worker.torch.device(configuration["device"]))
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(10)
        configuration["pool"]["port"] = listener.getsockname()[1]

        def serve():
            try:
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as stream:
                    connection.settimeout(30)
                    login = decode_frame(stream.readline())
                    self.assertEqual(login["method"], "login")
                    self.assertEqual(login["params"][:2], ["local-test-account", "gpu-test"])
                    frames = encode_frame({"id": login["id"], "status": 1, "result": True}, 0)
                    frames += encode_frame({"id": -1, "status": 1,
                                            "result": ["extranonce", extranonce]}, 1)
                    frames += encode_frame({"id": -1, "status": 1, "result": notify}, 2)
                    connection.sendall(frames)
                    while raw := stream.readline():
                        request = decode_frame(raw)
                        self.assertNotIn(request["id"], observed["request_ids"])
                        observed["request_ids"].append(request["id"])
                        if request["method"] == "report_info":
                            self.assertEqual(len(request["params"]), 1)
                            device = request["params"][0]
                            self.assertEqual(set(device), {"name", "hashrate"})
                            self.assertEqual(device["name"], expected_gpu_name)
                            self.assertIsInstance(device["hashrate"], (int, float))
                            self.assertTrue(math.isfinite(device["hashrate"]))
                            self.assertGreaterEqual(device["hashrate"], 0)
                            observed["reports"].append(device)
                        else:
                            self.assertEqual(request["method"], "submit")
                            self.assertNotIn("share", observed)
                            self.assertIsInstance(request["params"], list)
                            self.assertEqual(len(request["params"]), 1)
                            share = request["params"][0]
                            self.assertEqual(set(share), {
                                "m", "n", "k", "rank", "tile_h", "tile_w", "sb", "sb_p", "tail_mode",
                                "a_rows", "b_cols", "miner_nonce", "worker_nonce", "nonce",
                            })
                            self.assertEqual((share["m"], share["n"], share["k"]), (129, 128, 2048))
                            self.assertEqual((share["rank"], share["tile_h"], share["tile_w"]), (128, 8, 8))
                            self.assertEqual(share["worker_nonce"], extranonce)
                            self.assertEqual(len(bytes.fromhex(share["miner_nonce"])), 8)
                            self.assertEqual(share["nonce"], share["miner_nonce"])
                            self.assertEqual((share["sb"], share["sb_p"], share["tail_mode"]), (1, [], 1))
                            self.assertEqual(len(share["a_rows"]), 8)
                            self.assertEqual(len(share["b_cols"]), 8)
                            self.assertTrue(all(0 <= index < 128 for index in share["a_rows"] + share["b_cols"]))
                            observed["share"] = share
                        connection.sendall(encode_frame({"id": request["id"], "status": 1, "result": True}, 3))
            except BaseException as error:
                errors.append(error)

        server = threading.Thread(target=serve, daemon=True)
        server.start()
        original_search = worker.search_batch
        original_rate = worker.pool_hashrate
        rate_measurements = []

        def measured_rate(stats, cfg, elapsed):
            value = original_rate(stats, cfg, elapsed)
            rate_measurements.append((stats.batches, cfg.m * cfg.n * cfg.k, elapsed, value))
            return value

        def one_candidate(*args):
            return itertools.islice(original_search(*args), 1)

        with tempfile.TemporaryDirectory() as directory:
            configuration["pool"]["unknown_request_dir"] = directory
            try:
                with patch.object(worker.native_cuda, "available", return_value=native), \
                        patch.object(worker, "search_batch", side_effect=one_candidate), \
                        patch.object(worker, "pool_hashrate", side_effect=measured_rate), \
                        self.assertLogs(worker.LOG, level="INFO") as messages:
                    result = worker.run_pool_training(configuration)
            finally:
                server.join(5)
        if errors:
            raise errors[0]
        self.assertFalse(server.is_alive(), "local PearlHash peer did not stop")
        self.assertEqual(result, 0)
        self.assertIn("share", observed)
        self.assertEqual(len(observed["reports"]), 2)
        self.assertEqual(observed["reports"][0]["hashrate"], 0)
        reported_rate = observed["reports"][1]["hashrate"]
        self.assertGreater(reported_rate, 0)
        matching = [(batches, work, elapsed, value) for batches, work, elapsed, value in rate_measurements
                    if batches == 1 and value == reported_rate]
        self.assertTrue(matching, "reported H/s must use the actual completed-batch measurement")
        for batches, work, elapsed, value in matching:
            self.assertEqual(value, batches * work / max(elapsed, 0.001))
        self.assertIn("batches=1 submitted=1 accepted=1 rejected=0", " ".join(messages.output))

    def test_native_gpu_work_reaches_local_pearlhash_peer(self):
        self.check_gpu_exchange(native=True)

    def test_reference_gpu_work_reaches_local_pearlhash_peer(self):
        self.check_gpu_exchange(native=False)


if __name__ == "__main__":
    unittest.main()
