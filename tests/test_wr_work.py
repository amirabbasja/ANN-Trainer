"""Deterministic PearlHash generation against actual accepted-share evidence."""

from dataclasses import replace
import json
from pathlib import Path
import struct
import unittest

from blake3 import blake3
import numpy as np
import torch

from LRAEP_trainer.protocol import TrainingConfig, load_official_helpers
from LRAEP_trainer.wr_protocol import WRJob, WRShare
from LRAEP_trainer.wr_work import (
    prepare_wr_batch, reconstruct_wr_digest,
    verified_wr_share, wr_nonce_tail,
)


class WRNonceTailTests(unittest.TestCase):
    def test_tail_matches_native_cpu_oracle_byte_order_and_mapping(self):
        # WR 0.51.8 CPU reference, independently emulated with no GPU/network.
        miner, worker = bytes.fromhex("0200000000000000"), bytes.fromhex("f8d10b2c6c027fee")
        self.assertEqual(wr_nonce_tail(miner, worker).hex(),
                         "d5dfcbf13011c4eceaf504391c2a1bed")
        self.assertNotEqual(wr_nonce_tail(miner, worker), wr_nonce_tail(worker, miner))
        with self.assertRaises(ValueError):
            wr_nonce_tail(bytes(7), worker)


class WRWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        try:
            load_official_helpers(root / "src" / "LRAEP_helpers")
            import LRAEP_mining as pm
        except (ImportError, RuntimeError) as exc:
            raise unittest.SkipTest(f"official helpers unavailable: {exc}") from None
        cls.pm = pm
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(4, cls.previous_threads))
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.captures = json.loads((root / "tests/fixtures/wr_accepted_work.json").read_text())

    def setUp(self):
        self.cfg = TrainingConfig(m=128, n=128, k=2048)
        header = self.pm.IncompleteBlockHeader(
            version=1, prev_block=bytes(32), merkle_root=bytes(range(32)),
            timestamp=1793041998, nbits=0x1D00FFFF,
        )
        self.job = WRJob(job_id="job", header_bytes=bytes(header.to_bytes()),
                              height=0, cert_version=3, target=1 << 200,
                              notify_payload=bytes(200), worker_nonce=bytes.fromhex("f8d10b2c6c027fee"))

    @staticmethod
    def share_from_fixture(fixture):
        fields = {name: value for name, value in fixture["share"].items()
                  if name not in {"nonce", "tail_mode", "sb_p"}}
        for name in ("miner_nonce", "worker_nonce"):
            fields[name] = bytes.fromhex(fields[name])
        for name in ("a_rows", "b_cols"):
            fields[name] = tuple(fields[name])
        return WRShare(**fields)

    def test_all_three_pool_accepted_shares_reconstruct_below_exact_target(self):
        # These requests and matching true replies were observed on the real
        # pool. Reconstructing arbitrary incorrect signals would fail by ~2^35.
        self.assertEqual(len(self.captures), 3)
        for fixture in self.captures:
            with self.subTest(request_id=fixture["request_id"]):
                share = self.share_from_fixture(fixture)
                digest = reconstruct_wr_digest(bytes.fromhex(fixture["header_hex"]), share)
                self.assertEqual(digest.hex(), fixture["digest_hex"])
                self.assertLessEqual(int.from_bytes(digest, "little"),
                                     int(fixture["final_target_hex"], 16))
                mining = self.pm.MiningConfiguration(
                    common_dim=share.k, rank=128, mma_type=self.pm.MMAType.Int7xInt7ToInt32,
                    rows_pattern=self.pm.PeriodicPattern.from_list([v-share.a_rows[0] for v in share.a_rows]),
                    cols_pattern=self.pm.PeriodicPattern.from_list([v-share.b_cols[0] for v in share.b_cols]),
                )
                target = self.pm.penalized_target_bound(
                    int.from_bytes(bytes.fromhex(fixture["wire_target"]), "little"), mining,
                )
                self.assertEqual(target, int(fixture["final_target_hex"], 16))

    def test_prepared_signals_reserve_nonce_row_and_bind_full_dimensions(self):
        from miner_base.commitment_hash import CommitmentHasher

        batch = prepare_wr_batch(self.job, self.cfg, np.random.default_rng(15))
        self.assertEqual(tuple(batch.a.shape), (129, 2048))
        self.assertEqual(tuple(batch.a_noised.shape), (128, 2048))
        self.assertTrue(bool((batch.a[:-1] == 64).all()))
        self.assertTrue(bool((batch.bt == 64).all()))
        self.assertEqual(batch.a[-1, -16:].numpy().tobytes(),
                         wr_nonce_tail(batch.miner_nonce, self.job.worker_nonce))
        header = self.pm.IncompleteBlockHeader.from_bytes(self.job.header_bytes)
        key = CommitmentHasher.get_key(bytes(header.to_bytes()), self.cfg.official_config())
        expected = CommitmentHasher.commitment_hash_from_merkle_roots(
            batch.a_tree.root, batch.b_tree.root, key, salted_dims=(129, 128),
        )
        self.assertEqual(batch.commitment.noise_seed_A, expected.noise_seed_A)
        wrong = CommitmentHasher.commitment_hash_from_merkle_roots(
            batch.a_tree.root, batch.b_tree.root, key, salted_dims=(128, 128),
        )
        self.assertNotEqual(batch.commitment.noise_seed_A, wrong.noise_seed_A)

    def test_share_verification_reconstructs_independently_and_checks_digest_target(self):
        from miner_base.noisy_gemm import NoisyGemm

        batch = prepare_wr_batch(self.job, self.cfg, np.random.default_rng(22))
        reference = NoisyGemm(hash_tile_h=8, hash_tile_w=8)
        _, _, transcripts = reference._process_output_tile(
            batch.a_noised, batch.b_noised, 2048, 0, 8, 0, 8,
        )
        digest = blake3(struct.pack("<16I", *(int(v) for v in transcripts[0][0].data)),
                        key=batch.commitment.noise_seed_A).digest()
        share = verified_wr_share(batch, 0, 0, digest, (1 << 256) - 1)
        self.assertEqual((share.m, share.n, share.sb), (129, 128, 1))
        self.assertEqual(share.a_rows, tuple(range(8)))
        # CPU reconstruction derives its own signals instead of trusting a
        # mutable GPU work buffer or serialized caller-supplied digest.
        batch.a_noised.zero_()
        self.assertEqual(verified_wr_share(batch, 0, 0, digest, (1 << 256) - 1), share)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            verified_wr_share(batch, 0, 0, bytes(32), (1 << 256) - 1)
        with self.assertRaisesRegex(ValueError, "target"):
            verified_wr_share(batch, 0, 0, digest, 1)
        with self.assertRaises(ValueError):
            verified_wr_share(batch, 16, 0, digest, (1 << 256) - 1)

    def test_native_cuda_preparation_matches_cpu_with_reserved_row_and_padding(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from LRAEP_trainer import native_cuda
        if not native_cuda.available("cuda:0"):
            self.skipTest("native CUDA unavailable")
        # 129 * 2176 is not chunk-aligned and creates an odd Merkle tree.
        cfg = TrainingConfig(m=128, n=128, k=2176)
        cpu = prepare_wr_batch(self.job, cfg, np.random.default_rng(33))
        gpu = prepare_wr_batch(self.job, cfg, np.random.default_rng(33),
                                    device="cuda:0", native=True)
        self.assertEqual(cpu.miner_nonce, gpu.miner_nonce)
        self.assertEqual(cpu.a_tree.root, gpu.a_tree.root)
        self.assertEqual(cpu.b_tree.root, gpu.b_tree.root)
        self.assertEqual(cpu.commitment.noise_seed_A, gpu.commitment.noise_seed_A)
        self.assertEqual(cpu.commitment.noise_seed_B, gpu.commitment.noise_seed_B)
        self.assertTrue(torch.equal(cpu.a_noised, gpu.a_noised.cpu()))
        self.assertTrue(torch.equal(cpu.b_noised, gpu.b_noised.cpu()))

    def test_missing_extranonce_and_reserved_row_submission_fail(self):
        with self.assertRaisesRegex(ValueError, "extranonce"):
            prepare_wr_batch(replace(self.job, worker_nonce=None), self.cfg,
                                  np.random.default_rng(1))
        fixture = self.captures[0]
        share = self.share_from_fixture(fixture)
        with self.assertRaisesRegex(ValueError, "indices"):
            reconstruct_wr_digest(bytes.fromhex(fixture["header_hex"]),
                                      replace(share, a_rows=tuple(range(share.m-8, share.m))))


if __name__ == "__main__":
    unittest.main()
